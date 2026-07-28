#!/usr/bin/env python3
"""CPU-only proof for the v20 long-context safe-query failure mechanism.

This proof deliberately separates three claims:

1. measured fact: the regular safe-query BMM differs from the PEDANTIC
   reference at the production prefill width (M=3072);
2. source fact: nvfp4_ds_mla consumes that projection as BF16 in SparkInfer's
   NVFP4 QK arm (``supports_quant_query_input=False`` is therefore orthogonal);
3. constructive numeric fact: a reduced-precision BF16 reduction can reverse a
   valid NVFP4 sparse-attention result inside the already-selected 2,048-token
   window, while a full-FP32 reduction does not.

It does not claim to execute GLM-5.2 on CPU.  The final fixed-seed model ladder
remains an end-to-end confirmation, but this proof makes the candidate fix a
causal, falsifiable change rather than a boot-and-guess experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

TOKENS_PER_FULL_CHUNK = 3072
HEADS = 8
Q_INPUT_DIM = 192
Q_LATENT_DIM = 512
TOPK = 2048
MODEL_LAYERS = 78
CONTEXT_DEPTHS = (50_000, 100_000, 150_000, 250_000, 350_000, 475_000)
PRODUCTION_SPARKINFER_DECODE_MATH_SHA256 = (
    "0256763b141601bffb080e440756d98504968ad0d9d60a602efa27e967767413"
)

_E2M1_MAGNITUDES = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round_bf16(values: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16, ties-to-even, and return an FP32 container."""

    source = np.asarray(values, dtype=np.float32)
    bits = source.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + bias) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


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


def _quantize_e4m3(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Saturating E4M3FN quantization with nearest-even tie handling."""

    source = np.asarray(values, dtype=np.float32)
    magnitude = np.minimum(np.abs(source), np.float32(448.0))
    upper = np.searchsorted(_E4M3_VALUES, magnitude, side="left")
    upper = np.minimum(upper, len(_E4M3_VALUES) - 1)
    lower = np.maximum(upper - 1, 0)
    lower_distance = magnitude - _E4M3_VALUES[lower]
    upper_distance = _E4M3_VALUES[upper] - magnitude
    choose_upper = upper_distance < lower_distance
    ties = upper_distance == lower_distance
    choose_upper |= ties & ((_E4M3_CODES[upper] & np.uint8(1)) == 0)
    selected = np.where(choose_upper, upper, lower)
    code = _E4M3_CODES[selected].copy()
    code |= np.where(source < 0, np.uint8(0x80), np.uint8(0))
    decoded = _E4M3_VALUES[selected] * np.where(source < 0, -1.0, 1.0)
    return code, decoded.astype(np.float32)


def _quantize_e2m1(values: np.ndarray) -> np.ndarray:
    """SparkInfer's exact E2M1 thresholds from ``_fp4_quantize_values``."""

    source = np.asarray(values, dtype=np.float32)
    magnitude = np.abs(source)
    result = np.empty_like(magnitude)
    result[(magnitude >= 0.0) & (magnitude <= 0.25)] = 0.0
    result[(magnitude > 0.25) & (magnitude < 0.75)] = 0.5
    result[(magnitude >= 0.75) & (magnitude <= 1.25)] = 1.0
    result[(magnitude > 1.25) & (magnitude < 1.75)] = 1.5
    result[(magnitude >= 1.75) & (magnitude <= 2.5)] = 2.0
    result[(magnitude > 2.5) & (magnitude < 3.5)] = 3.0
    result[(magnitude >= 3.5) & (magnitude <= 5.0)] = 4.0
    result[magnitude > 5.0] = 6.0
    return result * np.sign(source)


def _nvfp4_roundtrip(rows: np.ndarray) -> np.ndarray:
    """Quantize/dequantize GLM's 512-wide NVFP4 latent, group size 16."""

    source = _round_bf16(np.asarray(rows, dtype=np.float32))
    original_shape = source.shape
    if original_shape[-1] != Q_LATENT_DIM:
        raise AssertionError(f"expected latent width {Q_LATENT_DIM}, got {original_shape}")
    groups = source.reshape(-1, Q_LATENT_DIM // 16, 16)
    block_max = np.max(np.abs(groups), axis=-1, keepdims=True)
    _, decoded_scale = _quantize_e4m3(block_max / np.float32(6.0))
    inverse = np.divide(
        np.float32(1.0),
        decoded_scale,
        out=np.zeros_like(decoded_scale),
        where=decoded_scale != 0,
    )
    fp4 = _quantize_e2m1(np.clip(groups * inverse, -6.0, 6.0))
    # The production NVFP4 BF16-QK arm dequantizes each pair to BF16 before
    # feeding the BF16 tensor-core MMA.
    return _round_bf16(fp4 * decoded_scale).reshape(original_shape)


def _precise_projection(query: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return _round_bf16(
        np.matmul(query.astype(np.float32), weight.astype(np.float32))
    )


def _reduced_projection(
    query: np.ndarray,
    weight: np.ndarray,
    *,
    reduction_tile: int = 16,
) -> np.ndarray:
    """Model a legal reduced-precision intermediate-reduction strategy."""

    accumulator = np.zeros((weight.shape[1],), dtype=np.float32)
    for start in range(0, query.shape[0], reduction_tile):
        stop = min(start + reduction_tile, query.shape[0])
        partial = np.matmul(
            query[start:stop].astype(np.float32),
            weight[start:stop].astype(np.float32),
        )
        accumulator = _round_bf16(accumulator + partial)
    return accumulator


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def _measured_operator_evidence(evidence_dir: Path) -> dict[str, Any]:
    current_path = evidence_dir / "w54-equiv-current.jsonl"
    precise_path = evidence_dir / "w54-equiv-pedantic.jsonl"
    current = {
        (row["tokens"], row["seed"], row["q_scale"]): row
        for row in _load_jsonl(current_path)
        if row.get("kind") == "safe_query_reduction_equivalence_case"
    }
    precise = {
        (row["tokens"], row["seed"], row["q_scale"]): row
        for row in _load_jsonl(precise_path)
        if row.get("kind") == "safe_query_reduction_equivalence_case"
    }
    common = sorted(set(current) & set(precise))
    assert len(common) == 54, f"expected 54 paired cases, got {len(common)}"

    by_m: dict[int, dict[str, int]] = {}
    lower_bounds: list[int] = []
    elements = HEADS * TOKENS_PER_FULL_CHUNK * Q_LATENT_DIM
    for key in common:
        tokens, _, _ = key
        bucket = by_m.setdefault(tokens, {"cases": 0, "bf16_changed": 0, "fp8_changed": 0})
        bucket["cases"] += 1
        bucket["bf16_changed"] += (
            current[key]["bf16_sha256"] != precise[key]["bf16_sha256"]
        )
        bucket["fp8_changed"] += (
            current[key]["fp8_sha256"] != precise[key]["fp8_sha256"]
        )
        if tokens == TOKENS_PER_FULL_CHUNK:
            max_error = float(current[key]["reference_max_abs_error"])
            mean_error = float(current[key]["reference_mean_abs_error"])
            assert max_error > 0 and mean_error > 0
            # sum(abs(error)) <= changed_elements * max(abs(error))
            lower_bounds.append(math.ceil(mean_error * elements / max_error))

    assert by_m[TOKENS_PER_FULL_CHUNK]["bf16_changed"] == 9
    assert by_m[TOKENS_PER_FULL_CHUNK]["fp8_changed"] == 9
    assert min(lower_bounds) > 0
    return {
        "current_sha256": _sha256(current_path),
        "pedantic_sha256": _sha256(precise_path),
        "by_m": by_m,
        "m3072_changed_elements_lower_bound": {
            "minimum": min(lower_bounds),
            "maximum": max(lower_bounds),
            "per_case": lower_bounds,
            "fixture_elements": elements,
        },
    }


def _source_route_evidence(vllm_repo: Path, sparkinfer_repo: Path) -> dict[str, Any]:
    backend_path = (
        vllm_repo / "vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
    )
    layer_path = (
        vllm_repo / "vllm/model_executor/layers/attention/mla_attention.py"
    )
    backend = backend_path.read_text()
    layer = layer_path.read_text()

    decode_math_path = (
        sparkinfer_repo / "sparkinfer/attention/_shared/mla/decode_math.py"
    )
    decode_math = decode_math_path.read_text()
    decode_math_sha256 = _sha256(decode_math_path)

    assert 'self.kv_cache_dtype == "nvfp4_ds_mla"' in backend
    assert "self._b12x_scale_format = 2" in backend
    assert "self.supports_quant_query_input = False" in backend
    assert "s1_qk_nope_nvfp4_bf16" in decode_math
    assert "BF16 QK-NoPE" in decode_math
    assert decode_math_sha256 == PRODUCTION_SPARKINFER_DECODE_MATH_SHA256, (
        "SparkInfer NVFP4 math source does not match the production be0edcaa "
        f"pin: {decode_math_sha256}"
    )
    assert "requires_precise_query_projection: bool = True" in backend
    assert "def _requires_precise_mla_query_bmm(" in layer
    assert 'getattr(impl, "requires_precise_query_projection", False)' in layer

    return {
        "target_cache": "nvfp4_ds_mla",
        "target_rope": "KV_FP8_ROPE=1",
        "query_boundary": "BF16 query x dequantized NVFP4 latent",
        "outer_prequantized_query": False,
        "precise_route_active": True,
        "files": {
            str(backend_path.relative_to(vllm_repo)): _sha256(backend_path),
            str(layer_path.relative_to(vllm_repo)): _sha256(layer_path),
            str(decode_math_path.relative_to(sparkinfer_repo)): decode_math_sha256,
        },
    }


def _construct_nvfp4_attention_witness() -> dict[str, Any]:
    """Find a valid production-shape input whose attention result flips."""

    for seed in range(1, 2049):
        rng = np.random.default_rng(seed)
        query = _round_bf16(
            rng.standard_normal(Q_INPUT_DIM, dtype=np.float32) * np.float32(0.5)
        )
        weight = _round_bf16(
            rng.standard_normal(
                (Q_INPUT_DIM, Q_LATENT_DIM), dtype=np.float32
            )
            * np.float32(0.05)
        )
        precise = _precise_projection(query, weight)
        reduced = _reduced_projection(query, weight)
        if np.array_equal(precise, reduced):
            continue

        precise_norm = float(np.linalg.norm(precise.astype(np.float64)))
        reduced_norm = float(np.linalg.norm(reduced.astype(np.float64)))
        direction = precise / precise_norm - reduced / reduced_norm
        # Scaling does not change NVFP4's relative group quantization, but
        # keeps its E4M3 scales comfortably away from subnormal underflow.
        boundary_key = _nvfp4_roundtrip(direction * np.float32(64.0))
        precise_boundary = float(np.dot(precise, boundary_key))
        reduced_boundary = float(np.dot(reduced, boundary_key))
        if not (precise_boundary > 0.0 and reduced_boundary < 0.0):
            continue

        # The indexer has already selected this window when query absorption
        # runs.  Put the needle and a distractor in a valid 2,048-entry sparse
        # window, with zero-key/value neutral entries in the other slots.
        precise_scores = np.concatenate(
            (
                np.zeros(TOPK - 2, dtype=np.float32),
                np.asarray([precise_boundary, -precise_boundary], dtype=np.float32),
            )
        )
        reduced_scores = np.concatenate(
            (
                np.zeros(TOPK - 2, dtype=np.float32),
                np.asarray([reduced_boundary, -reduced_boundary], dtype=np.float32),
            )
        )

        def attention_output(scores: np.ndarray) -> float:
            weights = np.exp(scores - np.max(scores))
            weights /= np.sum(weights)
            # One output component: neutral values are 0, needle is +1,
            # distractor is -1.
            return float(weights[-2] - weights[-1])

        precise_output = attention_output(precise_scores)
        reduced_output = attention_output(reduced_scores)
        assert precise_output > 0.0
        assert reduced_output < 0.0

        changed = int(np.count_nonzero(precise.view(np.uint32) != reduced.view(np.uint32)))
        return {
            "seed": seed,
            "projection_shape": [Q_INPUT_DIM, Q_LATENT_DIM],
            "reduction_tile": 16,
            "changed_bf16_values": changed,
            "max_abs_projection_delta": float(
                np.max(np.abs(precise - reduced))
            ),
            "nvfp4_boundary_scores": {
                "precise_needle": precise_boundary,
                "precise_distractor": -precise_boundary,
                "reduced_needle": reduced_boundary,
                "reduced_distractor": -reduced_boundary,
            },
            "selected_sparse_width": TOPK,
            "attention_output_component": {
                "precise": precise_output,
                "reduced": reduced_output,
            },
            "precise_favors_needle": True,
            "reduced_favors_distractor": True,
            "precise_matches_full_fp32_reduction_oracle": True,
        }

    raise AssertionError("failed to construct an NVFP4 attention witness")


def _context_exposure(lower_bound_per_call: int) -> list[dict[str, int]]:
    result = []
    for depth in CONTEXT_DEPTHS:
        full_chunks = depth // TOKENS_PER_FULL_CHUNK
        projection_calls = full_chunks * MODEL_LAYERS
        result.append(
            {
                "context_tokens": depth,
                "full_width_chunks": full_chunks,
                "full_width_layer_calls": projection_calls,
                # This is an exposure calculation using the measured 8-head
                # fixture's conservative per-call lower bound, not a claim
                # about the count on unknown model activations.
                "fixture_changed_values_lower_bound_if_repeated": (
                    projection_calls * lower_bound_per_call
                ),
            }
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=ROOT / "harness/cn4-evidence-archive/20260725/gateA",
    )
    parser.add_argument(
        "--vllm-repo",
        type=Path,
        default=ROOT / "workspace/vllm-v20-safe-query-accurate-clean",
    )
    parser.add_argument(
        "--sparkinfer-repo",
        type=Path,
        default=ROOT / "workspace/sparkinfer-v20-review",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    measured = _measured_operator_evidence(args.evidence_dir.resolve())
    route = _source_route_evidence(
        args.vllm_repo.resolve(), args.sparkinfer_repo.resolve()
    )
    witness = _construct_nvfp4_attention_witness()
    minimum = measured["m3072_changed_elements_lower_bound"]["minimum"]
    result = {
        "status": "PASS",
        "claim": (
            "Reduced-precision safe-query reduction can reverse an exact "
            "NVFP4 sparse-attention result inside the selected window; the "
            "scoped precise route removes that reduction while preserving the "
            "B12X/SparkInfer runtime."
        ),
        "measured_operator_evidence": measured,
        "production_route": route,
        "nvfp4_attention_witness": witness,
        "context_exposure": _context_exposure(minimum),
        "scope_limit": (
            "CPU proof establishes mechanism and fix routing; one fixed-seed "
            "model ladder remains required to confirm end-to-end semantics."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
