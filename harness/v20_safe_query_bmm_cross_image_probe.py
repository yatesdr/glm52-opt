#!/usr/bin/env python3
"""Fingerprint the staged MLA query path across two v20 images.

This is intentionally a no-model probe. Run the same file in the pre-992
(``6d32``) and post-992 candidate images with the same arguments, then compare
the emitted JSON records. The only operation under test is the compiled
``torch.ops._C.safe_mla_query_bmm`` followed by the production static FP8
quantizer.

The pre-992 image built the custom op with
``CUBLAS_COMPUTE_32F_PEDANTIC``. The post-992 image uses
``CUBLAS_COMPUTE_32F`` so cuBLAS can select tensor-core kernels. A changed BF16
or FP8 digest proves the staged-query source rollback did not restore the old
numeric path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _emit(record: dict[str, Any], output: Path | None) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        with output.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _load_ops() -> None:
    import vllm._custom_ops  # noqa: F401

    missing = [
        name
        for name in ("safe_mla_query_bmm", "static_scaled_fp8_quant")
        if not hasattr(torch.ops._C, name)
    ]
    if missing:
        raise RuntimeError("missing vLLM custom ops: " + ", ".join(missing))


def _retrieval_ids(
    query: torch.Tensor,
    *,
    seed: int,
    seq_len: int,
    topk: int,
    chunk_tokens: int,
) -> torch.Tensor:
    generator = torch.Generator(device=query.device)
    generator.manual_seed(0x51A0_0000 + seed)
    # One token row, all local query heads. This is not a model-quality test;
    # it amplifies changed FP8 query bytes into an observable selected-id set.
    # Stream exact chunks and fold their local winners. Materializing all 150k
    # BF16 keys and then their FP32 conversion costs about 500 MiB; the chunked
    # merge is mathematically the same global top-k with O(chunk_tokens) scratch.
    query_row = query[0].float()
    best_scores = torch.empty(
        query_row.shape[0],
        0,
        dtype=torch.float32,
        device=query.device,
    )
    best_ids = torch.empty(
        query_row.shape[0],
        0,
        dtype=torch.int64,
        device=query.device,
    )
    for start in range(0, seq_len, chunk_tokens):
        count = min(chunk_tokens, seq_len - start)
        keys = torch.randn(
            count,
            query.shape[-1],
            dtype=torch.bfloat16,
            device=query.device,
            generator=generator,
        )
        scores = query_row @ keys.float().T
        local_k = min(topk, count)
        local_scores, local_ids = torch.topk(
            scores,
            local_k,
            dim=-1,
            sorted=True,
        )
        local_ids = local_ids + start
        if best_scores.shape[1]:
            local_scores = torch.cat((best_scores, local_scores), dim=-1)
            local_ids = torch.cat((best_ids, local_ids), dim=-1)
        keep = min(topk, local_scores.shape[1])
        best_scores, positions = torch.topk(
            local_scores,
            keep,
            dim=-1,
            sorted=True,
        )
        best_ids = torch.gather(local_ids, 1, positions)
    return best_ids


@torch.inference_mode()
def run_case(
    *,
    device: torch.device,
    tokens: int,
    heads: int,
    seed: int,
    q_scale_value: float,
    retrieval_seq_len: int,
    retrieval_topk: int,
    retrieval_chunk_tokens: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    query_storage = (
        torch.randn(
            tokens,
            heads,
            256,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.5
    )
    q_nope_storage, q_pe = query_storage.split((192, 64), dim=-1)
    q_nope = q_nope_storage.transpose(0, 1)
    weight = (
        torch.randn(
            heads,
            192,
            512,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.05
    )
    projected = torch.empty(
        heads,
        tokens,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    assembled = torch.empty(
        tokens,
        heads,
        576,
        dtype=torch.bfloat16,
        device=device,
    )
    quantized = torch.empty_like(assembled, dtype=torch.float8_e4m3fn)
    q_scale = torch.tensor([q_scale_value], dtype=torch.float32, device=device)

    assert not q_nope.is_contiguous()
    torch.ops._C.safe_mla_query_bmm(q_nope, weight, projected)
    torch.cat((projected.transpose(0, 1), q_pe), dim=-1, out=assembled)
    torch.ops._C.static_scaled_fp8_quant(
        quantized.view(tokens, -1),
        assembled.view(tokens, -1),
        q_scale,
    )
    torch.cuda.synchronize(device)

    ids = _retrieval_ids(
        quantized,
        seed=seed,
        seq_len=retrieval_seq_len,
        topk=retrieval_topk,
        chunk_tokens=retrieval_chunk_tokens,
    )
    torch.cuda.synchronize(device)

    # A high-precision diagnostic reference. It is not used as the cross-image
    # oracle because its own library implementation may differ by image.
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference = torch.bmm(q_nope.float(), weight.float()).to(torch.bfloat16)
        torch.cuda.synchronize(device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    abs_error = (projected.float() - reference.float()).abs()

    return {
        "kind": "safe_query_bmm_fingerprint",
        "tokens": tokens,
        "heads": heads,
        "seed": seed,
        "q_scale": q_scale_value,
        "bf16_sha256": _sha256(projected),
        "fp8_sha256": _sha256(quantized),
        "retrieval_ids_sha256": _sha256(ids),
        "reference_bf16_sha256": _sha256(reference),
        "reference_max_abs_error": float(abs_error.max().item()),
        "reference_mean_abs_error": float(abs_error.mean().item()),
        "retrieval_seq_len": retrieval_seq_len,
        "retrieval_topk": retrieval_topk,
        "retrieval_chunk_tokens": retrieval_chunk_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", default="1,4,9,16,32,3072")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seeds", default="7,19,41")
    parser.add_argument("--q-scales", default="0.5,1.0,2.0")
    parser.add_argument("--retrieval-seq-len", type=int, default=150_000)
    parser.add_argument("--retrieval-topk", type=int, default=2048)
    parser.add_argument("--retrieval-chunk-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _csv_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def _csv_floats(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.output is not None:
        args.output.unlink(missing_ok=True)
    if args.retrieval_chunk_tokens <= 0:
        raise SystemExit("--retrieval-chunk-tokens must be positive")
    _load_ops()

    count = 0
    for tokens in _csv_ints(args.tokens):
        for seed in _csv_ints(args.seeds):
            for q_scale in _csv_floats(args.q_scales):
                _emit(
                    run_case(
                        device=device,
                        tokens=tokens,
                        heads=args.heads,
                        seed=seed,
                        q_scale_value=q_scale,
                        retrieval_seq_len=args.retrieval_seq_len,
                        retrieval_topk=args.retrieval_topk,
                        retrieval_chunk_tokens=args.retrieval_chunk_tokens,
                    ),
                    args.output,
                )
                count += 1
    _emit({"kind": "summary", "cases": count, "status": "PASS"}, args.output)


if __name__ == "__main__":
    main()
