#!/usr/bin/env python3
"""GPU-free proof of the v19/v20 absorbed-MLA projection delta.

This proof establishes four facts without loading a model:

1. v19 materializes ModelOpt weights to BF16 and stages both projections
   through ``torch.bmm``.
2. v20 replaces both generated-token projections with native/fused kernels.
3. The current fast-kernel tests permit numerical drift versus the staged
   route; they do not require byte equivalence.
4. The candidate ``legacy`` mode bypasses both fast paths, rather than only
   disabling native weight absorption.

The final reduction-order witness demonstrates sensitivity, not end-to-end
causality.  Causality still requires the fixed-prompt model discriminator.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V19 = ROOT / "workspace/upstream-vllm-v19-profiler"
V20 = ROOT / "workspace/vllm-v20-nf3-legacy-projection"
SPARK = ROOT / "workspace/sparkinfer-v20-review"
V19_REV = "7ea567a2458a4800a6a0e3e0a6ba41fcbd00d146"
V20_REV = "551719766029e78824a30d97ae6ac63917405b5f"
SPARK_REV = "be0edcaae6f5d284bb29a82325aba7a0ead6960f"
MLA = "vllm/model_executor/layers/attention/mla_attention.py"


def _git_show(tree: Path, revision: str, relative: str) -> str:
    return subprocess.run(
        ("git", "-C", str(tree), "show", f"{revision}:{relative}"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bf16(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _left_reduce(products: np.ndarray) -> np.float32:
    accumulator = np.float32(0)
    for value in products:
        accumulator = np.float32(accumulator + value)
    return accumulator


def _k16_reduce(products: np.ndarray) -> np.float32:
    accumulator = np.float32(0)
    for start in range(0, products.size, 16):
        partial = np.float32(0)
        for value in products[start : start + 16]:
            partial = np.float32(partial + value)
        accumulator = np.float32(accumulator + partial)
    return accumulator


def main() -> int:
    v19 = _git_show(V19, V19_REV, MLA)
    v20 = _git_show(V20, V20_REV, MLA)
    spark_bmm_test = _git_show(SPARK, SPARK_REV, "tests/gemm/test_bmm.py")
    spark_query_test = _git_show(
        SPARK,
        SPARK_REV,
        "tests/gemm/test_mla_query_projection.py",
    )
    patched = (V20 / MLA).read_text()
    patched_envs = (V20 / "vllm/envs.py").read_text()

    # Exact route delta.
    assert "_try_fused_mla_query" not in v19
    assert "torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)" in v19
    assert "torch.bmm(x, self.W_UV, out=out.transpose(0, 1))" in v19
    assert "run_mxfp8_mla_query" in v20
    assert "run_b12x_mxfp8_bmm(" in v20
    assert "_b12x_absorb_uk_rhs" in v20
    assert "_b12x_absorb_uv_rhs" in v20

    # Existing tests prove bounded error and fast-path self-consistency, not
    # byte identity with the v19 staged route.
    assert "candidate_error <= cublas_error * 1.05" in spark_bmm_test
    assert "torch.equal(out, expected)" in spark_query_test
    assert "expected = torch.bmm(fresh, logical_b)" not in spark_query_test
    assert "rtol=2e-2, atol=2e-2" in spark_query_test

    # The candidate mode must disable both native absorption and the later
    # BF16 fused-query path; disabling only VLLM_B12X_ABSORB_BMM is incomplete.
    assert 'Literal["auto", "legacy"]' in patched_envs
    assert "def _use_legacy_b12x_mla_projection()" in patched
    prepare_start = patched.index("def _prepare_b12x_absorb_bmm")
    prepare_end = patched.index("def _dequant_b12x_absorbed_pair", prepare_start)
    prepare = patched[prepare_start:prepare_end]
    assert prepare.index("_use_legacy_b12x_mla_projection()") < prepare.index(
        "_b12x_absorb_bmm_enabled()"
    )
    fused_start = patched.index("def _try_fused_mla_query")
    fused_end = patched.index("def forward_impl", fused_start)
    fused = patched[fused_start:fused_end]
    assert "_use_legacy_b12x_mla_projection()" in fused
    assert fused.index("_use_legacy_b12x_mla_projection()") < fused.index(
        "can_implement_mxfp8_mla_query"
    )

    # Deterministic production-K sensitivity witness. Both operands first
    # cross a BF16 boundary. Two legal FP32 reduction groupings then land on
    # adjacent BF16 results.
    generator = np.random.default_rng(12109)
    lhs = _bf16(generator.normal(0, 3, 192).astype(np.float32))
    rhs = _bf16(generator.normal(0, 0.2, 192).astype(np.float32))
    products = (lhs * rhs).astype(np.float32)
    left_f32 = _left_reduce(products)
    k16_f32 = _k16_reduce(products)
    left_bf16 = _bf16(np.array([left_f32]))[0]
    k16_bf16 = _bf16(np.array([k16_f32]))[0]
    left_bits = int(np.array([left_bf16], dtype=np.float32).view(np.uint32)[0])
    k16_bits = int(np.array([k16_bf16], dtype=np.float32).view(np.uint32)[0])
    assert left_bits != k16_bits
    assert abs((left_bits >> 16) - (k16_bits >> 16)) == 1

    result = {
        "status": "PASS",
        "finding": (
            "v20 changes both generated-token MLA projections; existing tests "
            "do not require v19 byte equivalence"
        ),
        "candidate": (
            "legacy mode restores materialized BF16 weights and bypasses both "
            "native MXFP8 and fused BF16 query routes"
        ),
        "causal_status": "model A/B required",
        "reduction_witness": {
            "k": 192,
            "seed": 12109,
            "left_f32": float(left_f32),
            "k16_f32": float(k16_f32),
            "left_bf16_bits": f"0x{left_bits:08x}",
            "k16_bf16_bits": f"0x{k16_bits:08x}",
            "bf16_ulp_distance": 1,
        },
        "pins": {
            "v19_revision": V19_REV,
            "v19_mla_sha256": _sha256_text(v19),
            "v20_revision": V20_REV,
            "v20_mla_sha256": _sha256_text(v20),
            "sparkinfer_revision": SPARK_REV,
            "candidate_mla_sha256": _sha256_file(V20 / MLA),
            "candidate_envs_sha256": _sha256_file(V20 / "vllm/envs.py"),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
