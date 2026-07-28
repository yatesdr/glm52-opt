#!/usr/bin/env python3
"""Compare NVFP4 split-K decode and single-pass extend at deep slot ids.

PR #171 changes compact-NVFP4 MTP verification from the split-K decode kernel
to the single-pass extend kernel. Existing numeric coverage uses small,
dense cache ids and does not exercise production DCP4 local physical ids.
This no-model probe writes valid 368-byte NVFP4 records at selected deep slots,
runs both kernels over identical queries/indices, and compares each result to
an explicit dequantized-record attention oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from sparkinfer.attention._shared.mla.kernel import run_unified_decode
from sparkinfer.attention._shared.mla.kv_cache import (
    concat_and_cache_nvfp4_mla_fp8_rope,
)
from sparkinfer.attention._shared.mla.prefill import run_unified_prefill
from sparkinfer.attention._shared.mla.traits import ScaleFormat
from sparkinfer.attention.sparse_mla._scratch import (
    SPARKINFERSparseMLAScratchCaps,
    plan_sparse_mla_scratch,
)


RECORD_BYTES = 368
NOPE_BYTES = 256
GROUP_SCALES_OFFSET = 256
ROPE_SCALE_OFFSET = 288
PAD_OFFSET = 292
ROPE_OFFSET = 304
PAGE_SIZE = 64
HEAD_DIM = 576
VALUE_DIM = 512
E2M1_TO_FLOAT32 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _dequantize_records(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed = records[:, :NOPE_BYTES]
    codes = torch.stack((packed & 0xF, (packed >> 4) & 0xF), dim=-1).reshape(
        records.shape[0], VALUE_DIM
    )
    values = torch.tensor(
        E2M1_TO_FLOAT32,
        dtype=torch.float32,
        device=records.device,
    )
    scales = (
        records[:, GROUP_SCALES_OFFSET:ROPE_SCALE_OFFSET]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    nope = (
        values[codes.long()].reshape(records.shape[0], 32, 16)
        * scales.unsqueeze(-1)
    ).reshape(records.shape[0], VALUE_DIM)
    rope_scale = (
        records[:, ROPE_SCALE_OFFSET:PAD_OFFSET]
        .contiguous()
        .view(torch.float32)
        .reshape(-1)
    )
    rope = (
        records[:, ROPE_OFFSET:RECORD_BYTES]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
        * rope_scale.unsqueeze(-1)
    )
    return nope, rope


def _oracle(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    flat = cache.reshape(-1, RECORD_BYTES)
    rows = []
    for row in range(q.shape[0]):
        records = flat.index_select(0, indices[row].long())
        nope, rope = _dequantize_records(records)
        keys = torch.cat((nope, rope), dim=-1)
        scores = q[row].float() @ keys.T
        probs = torch.softmax(scores * sm_scale, dim=-1)
        rows.append(probs @ nope)
    return torch.stack(rows)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    flat_actual = actual_f.reshape(-1)
    flat_expected = expected_f.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(
        flat_actual,
        flat_expected,
        dim=0,
    )
    delta = (actual_f - expected_f).abs()
    return {
        "cosine": float(cosine.item()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def _make_decode_workspace(
    *,
    device: torch.device,
    rows: int,
    heads: int,
    topk: int,
) -> object:
    caps = SPARKINFERSparseMLAScratchCaps(
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=heads,
        max_q_rows=rows,
        max_batch=rows,
        max_width=topk,
        max_kv_rows=topk,
        head_dim=HEAD_DIM,
        v_head_dim=VALUE_DIM,
        max_chunks_per_row=32,
        page_size=PAGE_SIZE,
        mode="decode",
    )
    plan = plan_sparse_mla_scratch(caps)
    (spec,) = plan.scratch_specs()
    storage = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    return plan, storage


@torch.inference_mode()
def run_case(
    *,
    device: torch.device,
    local_depth: int,
    rows: int,
    heads: int,
    topk: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    num_blocks = math.ceil(local_depth / PAGE_SIZE)
    capacity = num_blocks * PAGE_SIZE
    cache = torch.zeros(
        num_blocks,
        PAGE_SIZE,
        RECORD_BYTES,
        dtype=torch.uint8,
        device=device,
    )

    # Spread the live records across the entire physical address range, with
    # the final record at the deepest valid slot.
    slots = torch.linspace(
        0,
        capacity - 1,
        topk,
        dtype=torch.float64,
        device=device,
    ).round().to(torch.int64)
    if torch.unique(slots).numel() != topk:
        raise RuntimeError("slot construction produced duplicates")
    kv_c = (
        torch.randn(
            topk,
            VALUE_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.25
    )
    k_pe = (
        torch.randn(
            topk,
            HEAD_DIM - VALUE_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.25
    )
    concat_and_cache_nvfp4_mla_fp8_rope(kv_c, k_pe, cache, slots)

    q = (
        torch.randn(
            rows,
            heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.25
    ).contiguous()
    indices = torch.stack(
        [slots.roll(shifts=(row * 37) % topk) for row in range(rows)]
    ).to(torch.int32)
    cache_lengths = torch.full(
        (rows,),
        capacity,
        dtype=torch.int32,
        device=device,
    )
    topk_lengths = torch.full((rows,), topk, dtype=torch.int32, device=device)
    sm_scale = HEAD_DIM**-0.5

    plan, storage = _make_decode_workspace(
        device=device,
        rows=rows,
        heads=heads,
        topk=topk,
    )
    binding = plan.bind(
        scratch=storage,
        q=q,
        selected_indices=indices,
        cache_seqlens_int32=cache_lengths,
        nsa_cache_seqlens_int32=topk_lengths,
    )
    decode = run_unified_decode(
        q_all=q,
        swa_k_cache=cache,
        swa_indices=indices,
        swa_topk_lengths=topk_lengths,
        workspace=binding.scratch,
        sm_scale=sm_scale,
        swa_page_size=PAGE_SIZE,
        forced_num_splits=32,
        scale_format_override=ScaleFormat.NVFP4_E4M3,
        fp8_rope_override=True,
    )
    extend, _ = run_unified_prefill(
        q=q,
        kv_cache=cache,
        topk_indices=indices,
        topk_length=topk_lengths,
        sm_scale=sm_scale,
        page_block_size=PAGE_SIZE,
        scale_format=ScaleFormat.NVFP4_E4M3,
        fp8_rope=True,
    )
    expected = _oracle(q, cache, indices, sm_scale)
    torch.cuda.synchronize(device)

    decode_metrics = _metrics(decode, expected)
    extend_metrics = _metrics(extend, expected)
    cross_metrics = _metrics(extend, decode)
    return {
        "kind": "nvfp4_decode_extend_high_index",
        "local_depth": local_depth,
        "capacity": capacity,
        "rows": rows,
        "heads": heads,
        "topk": topk,
        "seed": seed,
        "decode": decode_metrics,
        "extend": extend_metrics,
        "extend_vs_decode": cross_metrics,
        "decode_sha256": _sha256(decode),
        "extend_sha256": _sha256(extend),
        "status": (
            "PASS"
            if decode_metrics["cosine"] > 0.999
            and extend_metrics["cosine"] > 0.999
            and decode_metrics["max_abs"] < 0.04
            and extend_metrics["max_abs"] < 0.04
            else "FAIL"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--local-depths",
        default="25000,37500,62500,118750",
        help="DCP4-local depths corresponding to 100k/150k/250k/475k global",
    )
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--seeds", default="7,19,41")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _csv_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.output is not None:
        args.output.unlink(missing_ok=True)

    records = []
    for local_depth in _csv_ints(args.local_depths):
        for seed in _csv_ints(args.seeds):
            record = run_case(
                device=device,
                local_depth=local_depth,
                rows=args.rows,
                heads=args.heads,
                topk=args.topk,
                seed=seed,
            )
            print(json.dumps(record, sort_keys=True), flush=True)
            records.append(record)
            if args.output is not None:
                with args.output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
    failed = sum(record["status"] != "PASS" for record in records)
    summary = {
        "kind": "summary",
        "cases": len(records),
        "failed": failed,
        "status": "PASS" if failed == 0 else "FAIL",
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if args.output is not None:
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    if failed:
        raise SystemExit(1)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
