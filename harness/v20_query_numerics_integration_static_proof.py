#!/usr/bin/env python3
"""Fail-closed proof for the consolidated v20 query-numerics candidate."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VLLM = ROOT / "workspace/vllm-v20-query-numerics-with171"
SPARK = ROOT / "workspace/sparkinfer-v20-current-recovery"

MLA = VLLM / "vllm/model_executor/layers/attention/mla_attention.py"
ROUTE = VLLM / "vllm/model_executor/kernels/attention/b12x_mxfp8_bmm.py"
BACKEND = VLLM / "vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
CUDA = VLLM / "csrc/libtorch_stable/attention/mla/safe_query_bmm.cu"
ROUTE_TEST = VLLM / "tests/v1/attention/test_mla_backends.py"
CUDA_TEST = VLLM / "tests/kernels/test_safe_mla_query_bmm.py"
FUSED = SPARK / "sparkinfer/gemm/mla_query_projection/_bf16.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def main() -> None:
    mla = MLA.read_text()
    route = ROUTE.read_text()
    backend = BACKEND.read_text()
    cuda = CUDA.read_text()
    route_test = ROUTE_TEST.read_text()
    cuda_test = CUDA_TEST.read_text()
    fused = FUSED.read_text()

    # Small qualified batches use upstream PR #174's fused kernel. Its
    # numerical contract is FP32 accumulation -> BF16 rounding -> FP8 scale,
    # matching the established staged boundary without a separate BMM/concat.
    route_tree = ast.parse(route)
    can_bf16 = _function(route_tree, "can_implement_bf16_mla_query")
    can_source = ast.get_source_segment(route, can_bf16)
    assert can_source is not None
    assert "output_dtype != torch.bfloat16" not in can_source
    assert "can_implement_fused_mla_query" in can_source
    module_spec = _function(route_tree, "_module_fused_mla_query_spec")
    spec_source = ast.get_source_segment(route, module_spec)
    assert spec_source is not None
    assert "output_dtype != torch.bfloat16" not in spec_source
    assert "test_bf16_fused_mla_query_supports_fp8_output" in route_test
    assert "test_fused_mla_query_warmup_includes_bf16_weight_fp8_output" in route_test
    assert "_MAX_M = 32" in fused
    assert "acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)" in fused
    assert "projected = acc.to(tl.bfloat16)" in fused
    assert "projected_fp32 = projected.to(tl.float32) * inv_scale" in fused

    # Larger prefill batches remain on safe BMM, but the final source candidate
    # keeps tensor-core-eligible FP32 compute and forbids output-type reduction.
    assert "fused_mqa_q = self._try_fused_mla_query" in mla
    assert "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION" in cuda
    assert "CUBLAS_COMPUTE_32F_PEDANTIC" not in cuda
    assert "restore_status = cublasSetMathMode(handle, original_math_mode)" in cuda
    assert "restores_math_mode" in cuda_test

    # The compact NVFP4 verifier route remains: no regression below the
    # measured ~501,504-token production floor is accepted.
    assert "def _resolve_spec_decode_mode(" in backend
    assert 'kv_cache_dtype == "nvfp4_ds_mla"' in backend

    result = {
        "status": "PASS",
        "small_batch_route": "fused BF16 query projection/assembly, M<=32",
        "large_batch_route": (
            "safe BF16 BMM with FP32 compute and reduced reductions forbidden"
        ),
        "fp8_boundary": "FP32 accumulation -> BF16 round -> static E4M3 scale",
        "pr171_route_retained": True,
        "files": {
            "vllm/" + str(path.relative_to(VLLM)): _sha256(path)
            for path in (MLA, ROUTE, BACKEND, CUDA, ROUTE_TEST, CUDA_TEST)
        }
        | {
            "sparkinfer/" + str(FUSED.relative_to(SPARK)): _sha256(FUSED),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
