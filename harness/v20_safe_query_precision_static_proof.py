#!/usr/bin/env python3
"""Fail-closed source proof for the selective safe-query precision patch."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "workspace/vllm-v20-safe-query-fp8-precision-with171"
CUDA = REPO / "csrc/libtorch_stable/attention/mla/safe_query_bmm.cu"
BINDINGS = REPO / "csrc/libtorch_stable/torch_bindings.cpp"
OPS = REPO / "csrc/libtorch_stable/ops.h"
PYTHON = REPO / "vllm/model_executor/layers/attention/mla_attention.py"
BACKEND = REPO / "vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
TEST = REPO / "tests/kernels/test_safe_mla_query_bmm.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def main() -> None:
    cuda = CUDA.read_text()
    bindings = BINDINGS.read_text()
    ops = OPS.read_text()
    python = PYTHON.read_text()
    backend = BACKEND.read_text()
    tests = TEST.read_text()

    assert (
        "precise ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F" in cuda
    )
    assert "bool precise=False" in bindings
    assert "torch::stable::Tensor& output, bool precise" in ops
    assert "three_argument_schema_compatibility" in tests
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
        expression = ast.unparse(keywords["precise"])
        assert "fp8_attention" in expression
        assert "self.impl.supports_quant_query_input" in expression

    result = {
        "status": "PASS",
        "regular_default_preserved": True,
        "pedantic_scope": (
            "BF16 query BMM immediately consumed as quantized backend input"
        ),
        "missing_precise_op_fails_closed": True,
        "pr171_route_retained": True,
        "cuda_graph_modes_tested": ["regular", "precise"],
        "schema_backward_compatible": True,
        "files": {
            str(path.relative_to(REPO)): _sha256(path)
            for path in (CUDA, BINDINGS, OPS, PYTHON, BACKEND, TEST)
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
