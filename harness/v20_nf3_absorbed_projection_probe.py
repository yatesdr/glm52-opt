#!/usr/bin/env python3
"""No-model decomposition of the v19/v20 MLA projection seam.

The v19 route materializes ModelOpt MXFP8 ``kv_b_proj`` weights as BF16 and
uses staged ``torch.bmm`` calls for absorbed query and value projection.  The
v20 fast route keeps the native pack and combines two independent changes:

* a native MXFP8 tiny-M BMM;
* fused query assembly and optional static E4M3 conversion.

This probe separates those changes.  It reports exact BF16 and post-FP8
differences for the production GLM-5.2 shapes, including the BF16-weight fused
route that can retain the v20 assembly optimization while materializing only
the query projection.  It does not load a model or checkpoint and is a
numeric discriminator, not an end-to-end quality verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch


HEADS = 16
NOPE_DIM = 192
LATENT_DIM = 512
VALUE_DIM = 256
PACK_ROWS = NOPE_DIM + VALUE_DIM
ROPE_DIM = 64
QUERY_DIM = LATENT_DIM + ROPE_DIM


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dequant_physical(
    values: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    scale_values = scales.view(torch.float8_e8m0fnu).to(torch.bfloat16)
    return values.to(torch.bfloat16) * scale_values.repeat_interleave(32, dim=-1)


def _comparison(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, int | float]:
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        raise AssertionError(
            "comparison contract mismatch: "
            f"{candidate.shape}/{candidate.dtype} vs "
            f"{reference.shape}/{reference.dtype}"
        )
    candidate_bits = candidate.contiguous().view(torch.uint16)
    reference_bits = reference.contiguous().view(torch.uint16)
    mismatches = int(torch.count_nonzero(candidate_bits != reference_bits).item())
    abs_error = (candidate.float() - reference.float()).abs()
    return {
        "elements": candidate.numel(),
        "bit_mismatches": mismatches,
        "mismatch_fraction": mismatches / candidate.numel(),
        "max_abs_error": float(abs_error.max().item()),
        "mean_abs_error": float(abs_error.mean().item()),
    }


def _fp8_comparison(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, int | float]:
    if candidate.dtype != torch.float8_e4m3fn:
        raise AssertionError(f"candidate must be E4M3, got {candidate.dtype}")
    if reference.dtype != torch.float8_e4m3fn:
        raise AssertionError(f"reference must be E4M3, got {reference.dtype}")
    if candidate.shape != reference.shape:
        raise AssertionError(
            f"FP8 shape mismatch: {candidate.shape} vs {reference.shape}"
        )
    candidate_bits = candidate.contiguous().view(torch.uint8)
    reference_bits = reference.contiguous().view(torch.uint8)
    mismatches = int(torch.count_nonzero(candidate_bits != reference_bits).item())
    abs_error = (candidate.float() - reference.float()).abs()
    return {
        "elements": candidate.numel(),
        "byte_mismatches": mismatches,
        "mismatch_fraction": mismatches / candidate.numel(),
        "max_abs_error": float(abs_error.max().item()),
        "mean_abs_error": float(abs_error.mean().item()),
    }


def _static_fp8(query: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
    """Mirror the static scaled-E4M3 contract used by fused MLA queries."""
    if query.dtype != torch.bfloat16:
        raise AssertionError(f"query must be BF16, got {query.dtype}")
    if q_scale.dtype != torch.float32 or q_scale.numel() != 1:
        raise AssertionError("q_scale must be one float32 value")
    return (
        (query.float() / q_scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )


def _topk_comparison(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    keys: torch.Tensor,
    topk: int,
) -> dict[str, int | float]:
    candidate_scores = candidate[0].float() @ keys.float().T
    reference_scores = reference[0].float() @ keys.float().T
    candidate_ids = torch.topk(
        candidate_scores, topk, dim=-1, sorted=False
    ).indices
    reference_ids = torch.topk(
        reference_scores, topk, dim=-1, sorted=False
    ).indices
    overlap = torch.zeros_like(reference_ids, dtype=torch.bool)
    for head in range(HEADS):
        overlap[head] = torch.isin(reference_ids[head], candidate_ids[head])
    return {
        "topk_overlap_fraction": float(overlap.float().mean().item()),
        "top1_disagreements": int(
            torch.count_nonzero(
                torch.argmax(candidate_scores, dim=-1)
                != torch.argmax(reference_scores, dim=-1)
            ).item()
        ),
    }


def _run_case(
    *,
    device: torch.device,
    m: int,
    seed: int,
    q_scale_value: float,
    rope_parent_width: int,
    score_rows: int,
    topk: int,
) -> dict[str, object]:
    from sparkinfer import gemm
    from sparkinfer.gemm import mla_query_projection

    generator = torch.Generator(device=device).manual_seed(seed)
    values = (
        torch.randn(
            HEADS,
            PACK_ROWS,
            LATENT_DIM,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    scales = torch.randint(
        118,
        132,
        (HEADS, PACK_ROWS, LATENT_DIM // 32),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    )

    uk_values = values[:, :NOPE_DIM, :]
    uk_scales = scales[:, :NOPE_DIM, :]
    uv_values = values[:, NOPE_DIM:, :]
    uv_scales = scales[:, NOPE_DIM:, :]
    uk_bf16 = _dequant_physical(uk_values, uk_scales)
    uv_bf16 = _dequant_physical(uv_values, uv_scales).transpose(1, 2)

    # Match the split-and-transpose query view entering forward_impl.
    query_storage = torch.randn(
        m,
        HEADS,
        NOPE_DIM + ROPE_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    q_nope_storage, _ = query_storage.split((NOPE_DIM, ROPE_DIM), dim=-1)
    q_nope = q_nope_storage.transpose(0, 1)
    if rope_parent_width < ROPE_DIM:
        raise ValueError("rope parent width cannot be smaller than the RoPE suffix")
    q_pe_parent = torch.randn(
        m,
        HEADS,
        rope_parent_width,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    q_pe = q_pe_parent[..., -ROPE_DIM:]
    q_scale = torch.tensor([q_scale_value], device=device, dtype=torch.float32)

    # v19 reference: materialize the exact MXFP8 checkpoint values as BF16,
    # then use the established torch.bmm reduction.
    legacy_projected = torch.bmm(q_nope.contiguous(), uk_bf16)
    legacy_query = torch.cat((legacy_projected.transpose(0, 1), q_pe), dim=-1)

    # v20 stage 1: native MXFP8 BMM without fused assembly.
    native_projected = torch.empty_like(legacy_projected)
    gemm.bmm(
        q_nope,
        (uk_values, uk_scales),
        native_projected,
        a_dtype="bfloat16",
        b_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        b_major="n",
        sf_axis="n",
    )
    native_staged_query = torch.cat(
        (native_projected.transpose(0, 1), q_pe), dim=-1
    )

    # v20 stage 2: fuse assembly around the same native MXFP8 BMM.  SparkInfer
    # promises this is byte-identical to native BMM + concat.
    native_fused_bf16 = torch.empty_like(legacy_query)
    mla_query_projection.run(
        q_nope,
        (uk_values, uk_scales),
        q_pe,
        native_fused_bf16,
    )
    native_fused_fp8 = torch.empty_like(
        legacy_query, dtype=torch.float8_e4m3fn
    )
    mla_query_projection.run(
        q_nope,
        (uk_values, uk_scales),
        q_pe,
        native_fused_fp8,
        q_scale=q_scale,
    )

    # Forward-fix candidate: materialize only W_UK_T, but keep the v20 fused
    # assembly path.  This separates native-weight BMM numerics from fusion.
    bf16_fused_bf16 = torch.empty_like(legacy_query)
    mla_query_projection.run(
        q_nope,
        uk_bf16,
        q_pe,
        bf16_fused_bf16,
    )
    bf16_fused_fp8 = torch.empty_like(native_fused_fp8)
    mla_query_projection.run(
        q_nope,
        uk_bf16,
        q_pe,
        bf16_fused_fp8,
        q_scale=q_scale,
    )

    legacy_fp8 = _static_fp8(legacy_query, q_scale)
    native_staged_fp8 = _static_fp8(native_staged_query, q_scale)
    native_fused_bf16_fp8 = _static_fp8(native_fused_bf16, q_scale)
    bf16_fused_bf16_fp8 = _static_fp8(bf16_fused_bf16, q_scale)

    latent = torch.randn(
        HEADS,
        m,
        LATENT_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    legacy_value = torch.bmm(latent, uv_bf16)
    native_value_backing = torch.empty(
        m,
        HEADS,
        VALUE_DIM + 8,
        device=device,
        dtype=torch.bfloat16,
    )
    native_value = native_value_backing[..., :VALUE_DIM].transpose(0, 1)
    gemm.bmm(
        latent,
        (uv_values, uv_scales),
        native_value,
        a_dtype="bfloat16",
        b_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        b_major="k",
        sf_axis="k",
    )

    keys = torch.randn(
        score_rows,
        QUERY_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )

    torch.cuda.synchronize(device)
    return {
        "m": m,
        "seed": seed,
        "q_scale": q_scale_value,
        "rope_parent_width": rope_parent_width,
        "native_bmm_vs_legacy_bmm": _comparison(
            native_projected, legacy_projected
        ),
        "native_fusion_vs_native_staged_bf16": _comparison(
            native_fused_bf16, native_staged_query
        ),
        "bf16_fusion_vs_legacy_staged_bf16": _comparison(
            bf16_fused_bf16, legacy_query
        ),
        "native_staged_vs_legacy_post_fp8": _fp8_comparison(
            native_staged_fp8, legacy_fp8
        ),
        "native_fusion_vs_native_staged_post_fp8": _fp8_comparison(
            native_fused_fp8, native_staged_fp8
        ),
        "native_fusion_epilogue_consistency": _fp8_comparison(
            native_fused_fp8, native_fused_bf16_fp8
        ),
        "bf16_fusion_vs_legacy_post_fp8": _fp8_comparison(
            bf16_fused_fp8, legacy_fp8
        ),
        "bf16_fusion_epilogue_consistency": _fp8_comparison(
            bf16_fused_fp8, bf16_fused_bf16_fp8
        ),
        "native_value_vs_legacy_value": _comparison(native_value, legacy_value),
        "score_rows": score_rows,
        "topk": topk,
        "native_vs_legacy_topk_bf16": _topk_comparison(
            native_fused_bf16,
            legacy_query,
            keys=keys,
            topk=topk,
        ),
        "native_vs_legacy_topk_post_fp8": _topk_comparison(
            native_fused_fp8,
            legacy_fp8,
            keys=keys,
            topk=topk,
        ),
        "bf16_fused_vs_legacy_topk_post_fp8": _topk_comparison(
            bf16_fused_fp8,
            legacy_fp8,
            keys=keys,
            topk=topk,
        ),
    }


def _csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split(",") if item.strip())


def _csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(item) for item in raw.split(",") if item.strip())


def _changed(record: dict[str, Any], field: str, mismatch: str) -> bool:
    value = record[field]
    if not isinstance(value, dict):
        raise AssertionError(f"{field} is not a comparison record")
    return int(value[mismatch]) > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m-values", default="1,4,9,16,32")
    parser.add_argument("--seeds", default="47,131,991")
    parser.add_argument("--q-scales", default="0.5,1.0,2.0")
    parser.add_argument("--rope-parent-widths", default="256,576")
    parser.add_argument("--score-rows", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=2048)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SparkInfer discriminator")
    device = torch.device(args.device)
    m_values = _csv_ints(args.m_values)
    seeds = _csv_ints(args.seeds)
    q_scales = _csv_floats(args.q_scales)
    rope_parent_widths = _csv_ints(args.rope_parent_widths)
    if not m_values or any(not 1 <= m <= 32 for m in m_values):
        raise ValueError("all M values must be in the qualified range 1..32")
    if not seeds or not q_scales or any(scale <= 0 for scale in q_scales):
        raise ValueError("seeds and positive q_scales are required")
    if not rope_parent_widths or any(
        width < ROPE_DIM for width in rope_parent_widths
    ):
        raise ValueError("all RoPE parent widths must be at least 64")
    if args.topk <= 0 or args.score_rows < args.topk:
        raise ValueError("require score_rows >= topk > 0")

    import sparkinfer

    cases: list[dict[str, object]] = []
    for m in m_values:
        for seed in seeds:
            for q_scale in q_scales:
                for rope_parent_width in rope_parent_widths:
                    case = _run_case(
                        device=device,
                        m=m,
                        seed=seed,
                        q_scale_value=q_scale,
                        rope_parent_width=rope_parent_width,
                        score_rows=args.score_rows,
                        topk=args.topk,
                    )
                    cases.append(case)
                    print(
                        json.dumps({"kind": "case", **case}, sort_keys=True),
                        flush=True,
                    )

    native_bmm_changed = sum(
        _changed(case, "native_bmm_vs_legacy_bmm", "bit_mismatches")
        for case in cases
    )
    native_post_fp8_changed = sum(
        _changed(case, "native_staged_vs_legacy_post_fp8", "byte_mismatches")
        for case in cases
    )
    native_fusion_changed = sum(
        _changed(
            case,
            "native_fusion_vs_native_staged_post_fp8",
            "byte_mismatches",
        )
        for case in cases
    )
    bf16_fusion_post_fp8_changed = sum(
        _changed(case, "bf16_fusion_vs_legacy_post_fp8", "byte_mismatches")
        for case in cases
    )
    value_changed = sum(
        _changed(case, "native_value_vs_legacy_value", "bit_mismatches")
        for case in cases
    )
    summary = {
        "kind": "summary",
        "status": "COMPLETE",
        "cases": len(cases),
        "native_bmm_changed_cases": native_bmm_changed,
        "native_post_fp8_changed_cases": native_post_fp8_changed,
        "native_fusion_changed_cases": native_fusion_changed,
        "bf16_fusion_post_fp8_changed_cases": bf16_fusion_post_fp8_changed,
        "native_value_changed_cases": value_changed,
        "interpretation": {
            "native_fusion_expected_changed_cases": 0,
            "native_bmm_isolated": native_bmm_changed > 0
            and native_fusion_changed == 0,
            "bf16_fusion_preserves_legacy_post_fp8": (
                bf16_fusion_post_fp8_changed == 0
            ),
        },
        "torch": torch.__version__,
        "sparkinfer": getattr(sparkinfer, "__version__", "unknown"),
        "python": platform.python_version(),
        "probe_sha256": _sha256(Path(__file__)),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
