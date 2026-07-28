#!/usr/bin/env python3
"""Compare v20 sparse-MLA token-major and head-major output contracts.

This is a no-model GPU proof for a remaining v19 -> v20 semantic delta.
v19 plans sparse-MLA extend output as token-major ``(M, H, L)``.  v20
requests a head-major pitched view with the same logical shape, then feeds
that view directly to the absorbed value-projection ``torch.bmm``.

The probe holds Q, valid NVFP4+FP8-RoPE cache records, selected token IDs,
and BF16 projection weights fixed.  It compares:

1. sparse-MLA output after normalizing both layouts to contiguous bytes;
2. the BF16 value-projection output produced from each native layout.

Any attention mismatch is a kernel/layout bug.  Attention equality followed
by projection mismatch isolates layout-sensitive cuBLAS reduction numerics.
The production capacities matter because head-major pitch is planned from
``max_num_batched_tokens`` rather than the current query-row count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass

import torch


HEADS = 16
Q_DIM = 576
LATENT = 512
TOPK = 2048
PAGE = 64
RECORD_BYTES = 368
KV_TOKENS = 4096


@dataclass(frozen=True)
class Result:
    cap: int
    rows: int
    attention_changed: int
    attention_numel: int
    attention_max_abs: float
    attention_token_sha256: str
    attention_head_sha256: str
    projection_changed: int
    projection_numel: int
    projection_max_abs: float
    projection_token_sha256: str
    projection_head_sha256: str
    token_stride: tuple[int, ...]
    head_stride: tuple[int, ...]
    elapsed_seconds: float


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _plan_and_run(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    selected: torch.Tensor,
    active: torch.Tensor,
    cap: int,
    head_major: bool,
) -> torch.Tensor:
    from sparkinfer.attention.sparse_mla import Caps, plan, run_extend

    cache_lens = torch.tensor(
        [KV_TOKENS], dtype=torch.int32, device=q.device
    )
    scratch_plan = plan(
        Caps(
            mode="extend",
            device=q.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=HEADS,
            head_dim=Q_DIM,
            v_head_dim=LATENT,
            max_width=TOPK,
            max_q_rows=cap,
            max_batch=1,
            page_size=PAGE,
            head_major_output=head_major,
        )
    )
    (spec,) = scratch_plan.scratch_specs()
    storage = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=storage,
        q=q,
        selected_indices=selected,
        cache_seqlens_int32=cache_lens,
        nsa_cache_seqlens_int32=active,
    )
    output = run_extend(
        binding=binding,
        kv_cache=kv_cache,
        sm_scale=1.0 / math.sqrt(Q_DIM),
        latent_scale=1.0,
        v_head_dim=LATENT,
        scale_format=2,
        fp8_rope=True,
    )
    assert isinstance(output, torch.Tensor)
    torch.cuda.synchronize()
    # Preserve the production stride/pitch while the caller-owned scratch
    # backing this view is still live.  clone(preserve_format) is allowed to
    # compact non-dense views, which would erase the contract under test.
    saved = torch.empty_strided(
        output.shape,
        output.stride(),
        dtype=output.dtype,
        device=output.device,
    )
    saved.copy_(output)
    return saved


def _project_native_layout(
    attention: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    rows = int(attention.shape[0])
    x = attention.view(rows, HEADS, LATENT).transpose(0, 1)
    output = torch.empty(
        (rows, HEADS, LATENT),
        dtype=torch.bfloat16,
        device=attention.device,
    )
    torch.bmm(x, weight, out=output.transpose(0, 1))
    torch.cuda.synchronize()
    return output


def _make_cache(device: torch.device) -> torch.Tensor:
    from sparkinfer.attention._shared.mla.kv_cache import (
        concat_and_cache_nvfp4_mla_fp8_rope,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(0xC0FFEE)
    kv_c = (
        torch.randn(
            (KV_TOKENS, LATENT),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        * 0.05
    ).to(torch.bfloat16)
    k_pe = (
        torch.randn(
            (KV_TOKENS, Q_DIM - LATENT),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        * 0.05
    ).to(torch.bfloat16)
    cache = torch.empty(
        (KV_TOKENS // PAGE, PAGE, RECORD_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(KV_TOKENS, dtype=torch.int64, device=device)
    concat_and_cache_nvfp4_mla_fp8_rope(kv_c, k_pe, cache, slots)
    torch.cuda.synchronize()
    return cache


def _run_case(
    *,
    device: torch.device,
    kv_cache: torch.Tensor,
    weight: torch.Tensor,
    cap: int,
    rows: int,
    seed: int,
) -> Result:
    started = time.monotonic()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = (
        torch.randn(
            (rows, HEADS, Q_DIM),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        * 0.05
    ).to(torch.bfloat16)
    row_ids = torch.arange(rows, dtype=torch.int64, device=device)[:, None]
    col_ids = torch.arange(TOPK, dtype=torch.int64, device=device)[None, :]
    selected = ((row_ids * 104729 + col_ids * 15485863 + seed) % KV_TOKENS).to(
        torch.int32
    )
    active = torch.full((rows,), TOPK, dtype=torch.int32, device=device)

    token = _plan_and_run(
        q=q,
        kv_cache=kv_cache,
        selected=selected,
        active=active,
        cap=cap,
        head_major=False,
    )
    head = _plan_and_run(
        q=q,
        kv_cache=kv_cache,
        selected=selected,
        active=active,
        cap=cap,
        head_major=True,
    )
    token_contiguous = token.contiguous()
    head_contiguous = head.contiguous()
    attn_diff = token_contiguous != head_contiguous

    projected_token = _project_native_layout(token, weight)
    projected_head = _project_native_layout(head, weight)
    projection_diff = projected_token != projected_head

    return Result(
        cap=cap,
        rows=rows,
        attention_changed=int(attn_diff.sum().item()),
        attention_numel=token.numel(),
        attention_max_abs=float(
            (token_contiguous.float() - head_contiguous.float()).abs().max().item()
        ),
        attention_token_sha256=_digest(token_contiguous),
        attention_head_sha256=_digest(head_contiguous),
        projection_changed=int(projection_diff.sum().item()),
        projection_numel=projected_token.numel(),
        projection_max_abs=float(
            (projected_token.float() - projected_head.float()).abs().max().item()
        ),
        projection_token_sha256=_digest(projected_token),
        projection_head_sha256=_digest(projected_head),
        token_stride=tuple(int(v) for v in token.stride()),
        head_stride=tuple(int(v) for v in head.stride()),
        elapsed_seconds=time.monotonic() - started,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="CAP:ROWS; defaults to 2048:1711,2048:2048,3072:2735,3072:3072",
    )
    args = parser.parse_args()
    cases = args.case or ["2048:1711", "2048:2048", "3072:2735", "3072:3072"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    print(
        json.dumps(
            {
                "kind": "meta",
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(device),
                "cases": cases,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    kv_cache = _make_cache(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(0xA11CE)
    weight = (
        torch.randn(
            (HEADS, LATENT, LATENT),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        / math.sqrt(LATENT)
    ).to(torch.bfloat16)

    failed = False
    for index, text in enumerate(cases):
        cap_text, rows_text = text.split(":", 1)
        cap, rows = int(cap_text), int(rows_text)
        if not 0 < rows <= cap:
            raise ValueError(f"invalid CAP:ROWS case {text!r}")
        result = _run_case(
            device=device,
            kv_cache=kv_cache,
            weight=weight,
            cap=cap,
            rows=rows,
            seed=20260726 + index,
        )
        print(json.dumps({"kind": "case", **asdict(result)}, sort_keys=True), flush=True)
        failed |= result.attention_changed != 0

    print(
        json.dumps(
            {
                "kind": "summary",
                "cases": len(cases),
                "attention_layout_equivalent": not failed,
                "verdict": "PASS" if not failed else "FAIL",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
