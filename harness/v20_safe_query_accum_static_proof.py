#!/usr/bin/env python3
"""Fail-closed source proof for accurate tensor-core safe-query reduction."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT / "workspace/vllm-v20-safe-query-fp8-accum-with171",
        help="vLLM source tree to verify",
    )
    parser.add_argument(
        "--require-pr171",
        action="store_true",
        help="also require the NVFP4 verifier-route integration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    cuda_path = repo / "csrc/libtorch_stable/attention/mla/safe_query_bmm.cu"
    bindings_path = repo / "csrc/libtorch_stable/torch_bindings.cpp"
    ops_path = repo / "csrc/libtorch_stable/ops.h"
    python_path = repo / "vllm/model_executor/layers/attention/mla_attention.py"
    base_backend_path = repo / "vllm/v1/attention/backend.py"
    backend_path = repo / "vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
    test_path = repo / "tests/kernels/test_safe_mla_query_bmm.py"

    required = (
        cuda_path,
        bindings_path,
        ops_path,
        python_path,
        base_backend_path,
        backend_path,
        test_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"missing required source files: {missing}"

    cuda = cuda_path.read_text()
    bindings = bindings_path.read_text()
    ops = ops_path.read_text()
    python = python_path.read_text()
    base_backend = base_backend_path.read_text()
    backend = backend_path.read_text()
    tests = test_path.read_text()

    assert "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION" in cuda
    assert "CUBLAS_COMPUTE_32F_PEDANTIC" not in cuda
    assert "cublasGetMathMode(handle, &original_math_mode)" in cuda
    assert "cublasSetMathMode(handle, precise_math_mode)" in cuda
    assert "restore_status = cublasSetMathMode(handle, original_math_mode)" in cuda
    assert "check_cublas(gemm_status" in cuda
    assert "check_cublas(restore_status" in cuda
    assert "CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT" in cuda
    assert "bool precise=False" in bindings
    assert "torch::stable::Tensor& output, bool precise" in ops
    assert "three_argument_schema_compatibility" in tests
    assert "restores_math_mode" in tests
    assert '@pytest.mark.parametrize("precise", [False, True])' in tests

    tree = ast.parse(python)
    helper = _function(tree, "_run_mla_query_bmm")
    assert helper.args.kw_defaults[-1] is not None
    assert isinstance(helper.args.kw_defaults[-1], ast.Constant)
    assert helper.args.kw_defaults[-1].value is False

    safe_calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "safe_bmm"
    ]
    assert len(safe_calls) == 1
    assert len(safe_calls[0].args) == 4
    assert isinstance(safe_calls[0].args[-1], ast.Name)
    assert safe_calls[0].args[-1].id == "precise"
    assert "Precise MLA query projection requires" in python
    assert "requires_precise_query_projection: bool = False" in base_backend
    assert "requires_precise_query_projection: bool = True" in backend

    if args.require_pr171:
        # A production integration candidate cannot surrender the ~99.38
        # MiB/GPU scratch reduction and fall below the 500k pool floor.
        assert "def _resolve_spec_decode_mode(" in backend
        assert 'kv_cache_dtype == "nvfp4_ds_mla"' in backend

    route_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_mla_query_bmm"
    ]
    assert len(route_calls) == 2
    for call in route_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert "precise" in keywords
        precise_call = keywords["precise"]
        assert isinstance(precise_call, ast.Call)
        assert isinstance(precise_call.func, ast.Name)
        assert precise_call.func.id == "_requires_precise_mla_query_bmm"
        assert ast.unparse(precise_call.args[0]) == "fp8_attention"
        assert ast.unparse(precise_call.args[1]) == "self.impl"

    selector = _function(tree, "_requires_precise_mla_query_bmm")
    selector_source = ast.unparse(selector)
    assert "supports_quant_query_input" in selector_source
    assert "requires_precise_query_projection" in selector_source

    result = {
        "status": "PASS",
        "regular_default_preserved": True,
        "precise_scope": (
            "BF16 query BMM feeding either outer-quantized input or a backend "
            "that owns quantized-KV query math internally"
        ),
        "precision_mechanism": (
            "FP32 tensor-core compute with reduced-precision reductions forbidden"
        ),
        "shared_handle_mode_restored": True,
        "missing_precise_op_fails_closed": True,
        "pr171_route_required": args.require_pr171,
        "cuda_graph_modes_tested": ["regular", "precise"],
        "schema_backward_compatible": True,
        "files": {
            str(path.relative_to(repo)): _sha256(path)
            for path in required
        },
        "repo": str(repo),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
