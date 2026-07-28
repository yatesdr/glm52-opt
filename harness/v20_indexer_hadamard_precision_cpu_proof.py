#!/usr/bin/env python3
"""CPU proof for precision-conditioning GLM's FP8 sparse indexer.

The GLM reference selector scores full-precision Q/K vectors and then takes an
exact top-k.  The optimized vLLM path first quantizes both 128-wide vectors to
E4M3 with one UE8M0 scale per row.  This proof compares:

  * the full-FP32 selector oracle;
  * the current raw-vector FP8 selector; and
  * FP8 after applying the same normalized 128-point Sylvester Hadamard to Q
    and K.

Applying the same orthonormal transform to Q and K preserves every
full-precision dot product.  It is therefore precision conditioning, not a
selector-policy change.  The fixture is deliberately heavy-tailed because
outliers are the failure mode of one-scale-per-row quantization.  A real-model
activation trace remains the model-specific acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HEADS = 32
HEAD_DIM = 128
TOPK = 2048
FP8_MAX = np.float32(448.0)


def _e4m3_positive_lut() -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    codes: list[int] = []
    for code in range(0x7F):
        exponent = (code >> 3) & 0xF
        mantissa = code & 0x7
        if exponent == 0:
            value = mantissa * (2.0**-9)
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values.append(value)
        codes.append(code)
    return np.asarray(values, dtype=np.float32), np.asarray(codes, dtype=np.uint8)


_E4M3_VALUES, _E4M3_CODES = _e4m3_positive_lut()


def _quantize_e4m3(values: np.ndarray) -> np.ndarray:
    """Saturating E4M3FN round-to-nearest-even, returned as FP32 values."""

    source = np.asarray(values, dtype=np.float32)
    magnitude = np.minimum(np.abs(source), FP8_MAX)
    upper = np.searchsorted(_E4M3_VALUES, magnitude, side="left")
    upper = np.minimum(upper, len(_E4M3_VALUES) - 1)
    lower = np.maximum(upper - 1, 0)
    lower_distance = magnitude - _E4M3_VALUES[lower]
    upper_distance = _E4M3_VALUES[upper] - magnitude
    choose_upper = upper_distance < lower_distance
    ties = upper_distance == lower_distance
    choose_upper |= ties & ((_E4M3_CODES[upper] & np.uint8(1)) == 0)
    selected = np.where(choose_upper, upper, lower)
    decoded = _E4M3_VALUES[selected] * np.where(source < 0, -1.0, 1.0)
    return decoded.astype(np.float32)


def _ue8m0_fp8_roundtrip(
    rows: np.ndarray,
    *,
    amax_floor: float = 1e-4,
    group_size: int = HEAD_DIM,
) -> np.ndarray:
    """Production indexer E4M3 quantization with one UE8M0 scale per row."""

    source = np.asarray(rows, dtype=np.float32)
    if source.shape[-1] % group_size != 0:
        raise ValueError(
            f"last dimension {source.shape[-1]} is not divisible by {group_size}"
        )
    original_shape = source.shape
    groups = source.reshape(
        *source.shape[:-1],
        source.shape[-1] // group_size,
        group_size,
    )
    amax = np.maximum(
        np.max(np.abs(groups), axis=-1, keepdims=True),
        np.float32(amax_floor),
    )
    scale = np.exp2(np.ceil(np.log2(amax / FP8_MAX))).astype(np.float32)
    return (_quantize_e4m3(groups / scale) * scale).reshape(original_shape)


def _normalized_hadamard(rows: np.ndarray) -> np.ndarray:
    """Apply a normalized 128-point Sylvester Hadamard to the last axis."""

    source = np.asarray(rows, dtype=np.float32)
    if source.shape[-1] != HEAD_DIM:
        raise AssertionError(f"expected last dimension {HEAD_DIM}, got {source.shape}")
    transformed = source.copy().reshape(-1, HEAD_DIM)
    width = 1
    while width < HEAD_DIM:
        groups = transformed.reshape(-1, HEAD_DIM // (2 * width), 2, width)
        left = groups[:, :, 0, :].copy()
        right = groups[:, :, 1, :].copy()
        groups[:, :, 0, :] = left + right
        groups[:, :, 1, :] = left - right
        width *= 2
    transformed *= np.float32(1.0 / math.sqrt(HEAD_DIM))
    return transformed.reshape(source.shape)


def _selector_scores(
    q: np.ndarray, k: np.ndarray, head_weights: np.ndarray
) -> np.ndarray:
    per_head = np.matmul(q, k.T, dtype=np.float32)
    np.maximum(per_head, 0.0, out=per_head)
    return np.matmul(head_weights, per_head, dtype=np.float32)


def _make_queries(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    q = rng.standard_normal((HEADS, HEAD_DIM), dtype=np.float32)
    # A small number of large coordinates models the outlier-sensitive rows
    # that one-scale-per-row E4M3 represents poorly.
    outlier_mask = rng.random(q.shape) < 0.0125
    q += outlier_mask * rng.normal(0.0, 12.0, q.shape).astype(np.float32)
    # Learned head weights are not constrained positive in the reference.
    weights = rng.normal(0.0, 0.35, HEADS).astype(np.float32)
    weights += np.float32(1.0 / HEADS)
    return q, weights


def _make_keys(rng: np.random.Generator, rows: int) -> np.ndarray:
    k = rng.standard_normal((rows, HEAD_DIM), dtype=np.float32)
    row_scale = np.exp(
        rng.normal(0.0, 0.25, (rows, 1)).astype(np.float32)
    ).astype(np.float32)
    k *= row_scale
    outlier_mask = rng.random(k.shape) < 0.00625
    k += outlier_mask * rng.normal(0.0, 10.0, k.shape).astype(np.float32)
    return k


def _topk(scores: np.ndarray, width: int) -> np.ndarray:
    return np.argpartition(scores, -width)[-width:]


def _metrics(
    oracle: np.ndarray, approximate: np.ndarray, width: int
) -> dict[str, Any]:
    oracle_indices = _topk(oracle, width)
    approximate_indices = _topk(approximate, width)
    oracle_set = set(int(value) for value in oracle_indices.tolist())
    approximate_set = set(int(value) for value in approximate_indices.tolist())
    intersection = oracle_set & approximate_set
    union = oracle_set | approximate_set
    error = approximate - oracle
    threshold = float(np.min(oracle[oracle_indices]))
    oracle_margin = np.abs(oracle - threshold)
    return {
        "topk_recall": len(intersection) / width,
        "topk_jaccard": len(intersection) / len(union),
        "oracle_topk_false_negatives": width - len(intersection),
        "score_rmse": float(np.sqrt(np.mean(np.square(error, dtype=np.float64)))),
        "score_max_abs_error": float(np.max(np.abs(error))),
        "oracle_kth_score": threshold,
        "candidates_within_one_rmse_of_boundary": int(
            np.count_nonzero(oracle_margin <= np.sqrt(np.mean(np.square(error))))
        ),
    }


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def run(
    *,
    seed: int,
    contexts: tuple[int, ...],
    block_rows: int,
    topk: int,
) -> dict[str, Any]:
    if min(contexts) < topk:
        raise ValueError("every context must be at least topk")
    rng = np.random.default_rng(seed)
    q, head_weights = _make_queries(rng)
    q_hadamard = _normalized_hadamard(q)
    q_raw_fp8 = _ue8m0_fp8_roundtrip(q, amax_floor=1e-10)
    q_hadamard_fp8 = _ue8m0_fp8_roundtrip(
        q_hadamard, amax_floor=1e-10
    )

    max_context = max(contexts)
    oracle_scores = np.empty(max_context, dtype=np.float32)
    raw_fp8_scores = np.empty(max_context, dtype=np.float32)
    hadamard_fp8_scores = np.empty(max_context, dtype=np.float32)
    orthogonal_max_abs = 0.0

    for start in range(0, max_context, block_rows):
        stop = min(start + block_rows, max_context)
        k = _make_keys(rng, stop - start)
        k_hadamard = _normalized_hadamard(k)

        oracle = _selector_scores(q, k, head_weights)
        rotated_oracle = _selector_scores(q_hadamard, k_hadamard, head_weights)
        orthogonal_max_abs = max(
            orthogonal_max_abs,
            float(np.max(np.abs(rotated_oracle - oracle))),
        )
        oracle_scores[start:stop] = oracle

        k_raw_fp8 = _ue8m0_fp8_roundtrip(k)
        raw_fp8_scores[start:stop] = _selector_scores(
            q_raw_fp8, k_raw_fp8, head_weights
        )
        del k_raw_fp8

        k_hadamard_fp8 = _ue8m0_fp8_roundtrip(k_hadamard)
        hadamard_fp8_scores[start:stop] = _selector_scores(
            q_hadamard_fp8, k_hadamard_fp8, head_weights
        )

    results = []
    for context in contexts:
        oracle = oracle_scores[:context]
        raw = raw_fp8_scores[:context]
        rotated = hadamard_fp8_scores[:context]
        raw_metrics = _metrics(oracle, raw, topk)
        rotated_metrics = _metrics(oracle, rotated, topk)
        results.append(
            {
                "context": context,
                "raw_fp8": raw_metrics,
                "hadamard_fp8": rotated_metrics,
                "false_negative_reduction": (
                    raw_metrics["oracle_topk_false_negatives"]
                    - rotated_metrics["oracle_topk_false_negatives"]
                ),
                "rmse_ratio_hadamard_over_raw": (
                    rotated_metrics["score_rmse"] / raw_metrics["score_rmse"]
                ),
            }
        )

    return {
        "schema": "v20-indexer-hadamard-precision-cpu-proof-v1",
        "seed": seed,
        "geometry": {
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "topk": topk,
            "contexts": list(contexts),
            "scale_format": "ue8m0",
            "value_format": "e4m3fn",
        },
        "claim_boundary": (
            "synthetic production-shape numeric proof; real checkpoint "
            "activation trace and end-to-end ladder remain required"
        ),
        "full_precision_orthogonality": {
            "max_abs_score_delta": orthogonal_max_abs,
            "relative_to_max_score": orthogonal_max_abs
            / max(float(np.max(np.abs(oracle_scores))), np.finfo(np.float32).tiny),
        },
        "fingerprints": {
            "query": _sha256_array(q),
            "head_weights": _sha256_array(head_weights),
            "oracle_scores": _sha256_array(oracle_scores),
            "raw_fp8_scores": _sha256_array(raw_fp8_scores),
            "hadamard_fp8_scores": _sha256_array(hadamard_fp8_scores),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[50_000, 150_000, 350_000, 475_000],
    )
    parser.add_argument("--block-rows", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run(
        seed=args.seed,
        contexts=tuple(args.contexts),
        block_rows=args.block_rows,
        topk=args.topk,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
