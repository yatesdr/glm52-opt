#!/usr/bin/env python3
"""Validate indexer precision modes on real activations with production GPU ops.

This consumes the immutable post-RoPE BF16 activation trace produced by
``indexer_prequant_trace``.  It deliberately does not load the model.  The
production vLLM FP8 quantizer is used for Q and K, and a sample of its K output
is checked byte-for-byte against ``indexer_k_quant_and_cache``.

The proof compares the current raw-vector path, a same-record-size symmetric
INT8 candidate, and two normalized WHT implementations:

* float32 Sylvester WHT, rounded back to BF16 before quantization;
* the shipped BF16 Hadacore kernel, explicitly normalized by sqrt(128).

Scores use the GLM reference order: FP32 dot, per-head ReLU, learned head
weights, then exact top-k.  Only K rows committed before the in-flight tail
chunk are eligible, matching paged prefill runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from v20_indexer_hadamard_activation_proof import _load_trace


HEAD_DIM = 128
TOPK = 2048
WHT_NORMALIZER = 1.0 / math.sqrt(HEAD_DIM)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _set_metrics(candidate: torch.Tensor, oracle: torch.Tensor) -> dict[str, Any]:
    a = set(int(value) for value in candidate.cpu().tolist())
    b = set(int(value) for value in oracle.cpu().tolist())
    intersection = a & b
    union = a | b
    return {
        "intersection": len(intersection),
        "recall": len(intersection) / len(b),
        "jaccard": len(intersection) / len(union),
        "false_negatives": len(b - a),
        "set_exact": a == b,
    }


def _normalized_wht_f32(rows: torch.Tensor) -> torch.Tensor:
    out = rows.float().contiguous()
    width = 1
    while width < HEAD_DIM:
        groups = out.reshape(-1, HEAD_DIM // (2 * width), 2, width)
        left = groups[:, :, 0, :].clone()
        right = groups[:, :, 1, :].clone()
        groups[:, :, 0, :] = left + right
        groups[:, :, 1, :] = left - right
        width *= 2
    out.mul_(WHT_NORMALIZER)
    return out.to(torch.bfloat16)


def _normalized_wht_hadacore(rows: torch.Tensor) -> torch.Tensor:
    from vllm import _custom_ops as ops

    # Hadacore's current out-of-place entrypoint allocates ``out`` but the CUDA
    # launcher writes the input pointer.  Clone and use the production
    # in-place contract so the returned tensor is initialized and the caller's
    # source is not mutated.
    # Hadacore's CUDA kernel already emits the normalized orthogonal transform
    # (norm and dot products are preserved); do not scale it a second time.
    return ops.hadacore_transform(rows.contiguous().clone(), inplace=True)


def _quant_roundtrip(
    rows: torch.Tensor,
    *,
    use_ue8m0: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    quant, scale = per_token_group_quant_fp8(
        rows.contiguous(),
        HEAD_DIM,
        eps=1e-10,
        column_major_scales=False,
        use_ue8m0=use_ue8m0,
    )
    dequant = quant.float() * scale.float()
    return quant, scale, dequant


def _int8_roundtrip(
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-vector INT8 candidate with the existing 4-byte scale budget."""

    source = rows.float()
    scale = source.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-10).div_(127.0)
    quant = torch.round(source / scale).clamp_(-127, 127).to(torch.int8)
    dequant = quant.float() * scale
    return quant, scale, dequant


def _validate_k_cache_bytes(sample: torch.Tensor) -> dict[str, Any]:
    from vllm import _custom_ops as ops

    generic_quant, generic_scale, _ = _quant_roundtrip(sample)
    rows = int(sample.shape[0])
    page_size = 64
    pages = (rows + page_size - 1) // page_size
    cache = torch.empty(
        (pages, page_size, HEAD_DIM + 4),
        dtype=torch.uint8,
        device=sample.device,
    )
    slots = torch.arange(rows, dtype=torch.int64, device=sample.device)
    ops.indexer_k_quant_and_cache(sample, cache, slots, HEAD_DIM, "ue8m0")
    torch.cuda.synchronize(sample.device)

    # The apparent [page, token, 132] tensor is a byte-capacity allocation.
    # Each page is planar: all 64x128 value bytes first, then 64 FP32 scales.
    packed = cache.view(pages, -1)
    cache_quant = (
        packed[:, : page_size * HEAD_DIM]
        .view(pages, page_size, HEAD_DIM)
        .reshape(-1, HEAD_DIM)[:rows]
        .contiguous()
        .view(torch.float8_e4m3fn)
    )
    cache_scale = (
        packed[:, page_size * HEAD_DIM :]
        .view(pages, page_size, 4)
        .reshape(-1, 4)[:rows]
        .contiguous()
        .view(torch.float32)
        .reshape(rows, 1)
    )
    value_equal = torch.equal(cache_quant.view(torch.uint8), generic_quant.view(torch.uint8))
    scale_equal = torch.equal(cache_scale.view(torch.uint8), generic_scale.view(torch.uint8))
    return {
        "rows": rows,
        "value_bytes_equal": value_equal,
        "scale_bytes_equal": scale_equal,
        "generic_value_sha256": _sha256_tensor(generic_quant.view(torch.uint8)),
        "cache_value_sha256": _sha256_tensor(cache_quant.view(torch.uint8)),
        "generic_scale_sha256": _sha256_tensor(generic_scale.view(torch.uint8)),
        "cache_scale_sha256": _sha256_tensor(cache_scale.view(torch.uint8)),
    }


def _score_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    per_head = torch.matmul(q.float(), k.float().T)
    per_head.clamp_min_(0.0)
    return torch.matmul(weights.float(), per_head)


def _topk_metrics(
    *,
    scores: torch.Tensor,
    oracle_scores: torch.Tensor,
    positions: torch.Tensor,
    oracle_indices: torch.Tensor,
    needle_center: int,
    needle_radius: int,
) -> dict[str, Any]:
    rows = torch.topk(scores, TOPK, largest=True, sorted=False).indices
    selected = positions[rows]
    oracle_rows = torch.topk(oracle_scores, TOPK, largest=True, sorted=False).indices
    threshold = oracle_scores[oracle_rows].min()
    missed_mask = ~torch.isin(positions[oracle_rows], selected)
    margins = (oracle_scores[oracle_rows][missed_mask] - threshold).clamp_min(0)
    all_margins = (oracle_scores[oracle_rows] - threshold).clamp_min(0)
    needle_mask = (positions - needle_center).abs() <= needle_radius
    needle_best = scores[needle_mask].max()
    return {
        "vs_oracle_topk": _set_metrics(selected, oracle_indices),
        "score_rmse": float(
            torch.sqrt(torch.mean((scores - oracle_scores).double().square())).item()
        ),
        "score_max_abs_error": float((scores - oracle_scores).abs().max().item()),
        "score_weighted_false_negatives": {
            "count": int(missed_mask.sum().item()),
            "oracle_margin_sum": float(margins.sum().item()),
            "oracle_margin_fraction": float(
                margins.sum().item()
                / max(all_margins.sum().item(), torch.finfo(torch.float32).tiny)
            ),
        },
        "needle_window": {
            "center": needle_center,
            "radius": needle_radius,
            "best_rank": int((scores > needle_best).sum().item() + 1),
            "selected_tokens": [
                int(value)
                for value in positions[needle_mask][
                    torch.isin(positions[needle_mask], selected)
                ].tolist()
            ],
        },
    }


def run(
    *,
    trace_dir: Path,
    device: torch.device,
    chunk_rows: int,
    needle_fraction: float,
    needle_radius: int,
) -> dict[str, Any]:
    trace = _load_trace(trace_dir)
    tail_start = int(trace["tail_start_position"])
    eligible = trace["positions"] < tail_start
    positions_cpu = torch.from_numpy(trace["positions"][eligible]).to(torch.int64)
    keys_cpu = torch.from_numpy(trace["k"][eligible]).to(torch.bfloat16)
    q_cpu = torch.from_numpy(trace["q"]).to(torch.bfloat16)
    weights = torch.from_numpy(trace["weights"]).to(device=device, dtype=torch.float32)

    q_raw = q_cpu.to(device)
    q_wht_f32 = _normalized_wht_f32(q_raw)
    q_wht_hadacore = _normalized_wht_hadacore(q_raw)
    q_modes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "raw_bf16": (q_raw, q_raw.float()),
        "raw_fp8": (q_raw, _quant_roundtrip(q_raw)[2]),
        "raw_q_bf16_k_fp8": (q_raw, q_raw.float()),
        "raw_q_fp8_k_bf16": (q_raw, _quant_roundtrip(q_raw)[2]),
        "raw_fp8_float_scale": (
            q_raw,
            _quant_roundtrip(q_raw, use_ue8m0=False)[2],
        ),
        "raw_int8": (q_raw, _int8_roundtrip(q_raw)[2]),
        "wht_f32_bf16": (q_wht_f32, q_wht_f32.float()),
        "wht_f32_fp8": (q_wht_f32, _quant_roundtrip(q_wht_f32)[2]),
        "wht_hadacore_bf16": (q_wht_hadacore, q_wht_hadacore.float()),
        "wht_hadacore_fp8": (
            q_wht_hadacore,
            _quant_roundtrip(q_wht_hadacore)[2],
        ),
    }
    transform: dict[str, Callable[[torch.Tensor], torch.Tensor] | None] = {
        "raw_bf16": None,
        "raw_fp8": None,
        "raw_q_bf16_k_fp8": None,
        "raw_q_fp8_k_bf16": None,
        "raw_fp8_float_scale": None,
        "raw_int8": None,
        "wht_f32_bf16": _normalized_wht_f32,
        "wht_f32_fp8": _normalized_wht_f32,
        "wht_hadacore_bf16": _normalized_wht_hadacore,
        "wht_hadacore_fp8": _normalized_wht_hadacore,
    }
    k_encoding = {
        "raw_bf16": "bf16",
        "raw_fp8": "fp8_ue8m0",
        "raw_q_bf16_k_fp8": "fp8_ue8m0",
        "raw_q_fp8_k_bf16": "bf16",
        "raw_fp8_float_scale": "fp8_float_scale",
        "raw_int8": "int8",
        "wht_f32_bf16": "bf16",
        "wht_f32_fp8": "fp8_ue8m0",
        "wht_hadacore_bf16": "bf16",
        "wht_hadacore_fp8": "fp8_ue8m0",
    }
    if not (q_modes.keys() == transform.keys() == k_encoding.keys()):
        raise RuntimeError("query, K transform, and K encoding modes are out of sync")
    score_parts: dict[str, list[torch.Tensor]] = {name: [] for name in q_modes}
    cache_validation: dict[str, Any] | None = None

    for start in range(0, int(keys_cpu.shape[0]), chunk_rows):
        end = min(start + chunk_rows, int(keys_cpu.shape[0]))
        k_raw = keys_cpu[start:end].to(device, non_blocking=False)
        if cache_validation is None:
            cache_validation = _validate_k_cache_bytes(k_raw[: min(4096, len(k_raw))])
        transformed_cache: dict[str, torch.Tensor] = {}
        for name, (_, q_scoring) in q_modes.items():
            transform_fn = transform[name]
            transform_key = "raw" if transform_fn is None else transform_fn.__name__
            k_mode = transformed_cache.get(transform_key)
            if k_mode is None:
                k_mode = k_raw if transform_fn is None else transform_fn(k_raw)
                transformed_cache[transform_key] = k_mode
            encoding = k_encoding[name]
            if encoding == "int8":
                k_scoring = _int8_roundtrip(k_mode)[2]
            elif encoding == "fp8_float_scale":
                k_scoring = _quant_roundtrip(
                    k_mode,
                    use_ue8m0=False,
                )[2]
            elif encoding == "fp8_ue8m0":
                k_scoring = _quant_roundtrip(k_mode)[2]
            elif encoding == "bf16":
                k_scoring = k_mode.float()
            else:
                raise RuntimeError(f"unsupported K encoding {encoding!r}")
            score_parts[name].append(
                _score_chunk(q_scoring, k_scoring, weights).cpu()
            )

    if cache_validation is None:
        raise RuntimeError("trace contains no eligible K rows")
    scores = {name: torch.cat(parts) for name, parts in score_parts.items()}
    oracle_scores = scores["raw_bf16"]
    positions = positions_cpu
    oracle_rows = torch.topk(oracle_scores, TOPK, largest=True, sorted=False).indices
    oracle_indices = positions[oracle_rows]
    needle_center = int(
        round(float(trace["runtime_absolute_position"]) * needle_fraction)
    )

    return {
        "schema": "v20-indexer-real-activation-gpu-quant-proof-v2",
        "claim_boundary": (
            "production quantizer proof on one real layer/query; end-to-end "
            "cold needle ladder remains the model acceptance gate"
        ),
        "geometry": {
            "tokens": int(keys_cpu.shape[0]),
            "heads": int(q_cpu.shape[0]),
            "head_dim": int(q_cpu.shape[1]),
            "topk": TOPK,
            "tail_start_position": tail_start,
            "runtime_absolute_position": int(trace["runtime_absolute_position"]),
        },
        "k_cache_quantizer_equivalence": cache_validation,
        "candidate_formats": {
            "raw_fp8": {
                "value_bytes": HEAD_DIM,
                "scale_bytes": 4,
                "record_bytes": HEAD_DIM + 4,
                "production_kernel": True,
                "scale_format": "ue8m0",
            },
            "raw_fp8_float_scale": {
                "value_bytes": HEAD_DIM,
                "scale_bytes": 4,
                "record_bytes": HEAD_DIM + 4,
                "production_kernel": False,
                "scale_format": "float32",
            },
            "raw_int8": {
                "value_bytes": HEAD_DIM,
                "scale_bytes": 4,
                "record_bytes": HEAD_DIM + 4,
                "production_kernel": False,
                "scale_format": "float32",
            },
        },
        "q_transform_agreement": {
            "f32_vs_hadacore_max_abs": float(
                (q_wht_f32.float() - q_wht_hadacore.float()).abs().max().item()
            ),
            "f32_vs_hadacore_mean_abs": float(
                (q_wht_f32.float() - q_wht_hadacore.float()).abs().mean().item()
            ),
        },
        "modes": {
            name: _topk_metrics(
                scores=mode_scores,
                oracle_scores=oracle_scores,
                positions=positions,
                oracle_indices=oracle_indices,
                needle_center=needle_center,
                needle_radius=needle_radius,
            )
            for name, mode_scores in scores.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--needle-fraction", type=float, default=0.4)
    parser.add_argument("--needle-radius", type=int, default=24)
    args = parser.parse_args()

    report = run(
        trace_dir=args.trace_dir,
        device=torch.device(args.device),
        chunk_rows=args.chunk_rows,
        needle_fraction=args.needle_fraction,
        needle_radius=args.needle_radius,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
