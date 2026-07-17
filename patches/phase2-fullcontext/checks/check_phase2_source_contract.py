#!/usr/bin/env python3
"""Static integration checks for the Gate-C phase-2 transport overlay."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "transport" / "overlays"
SPARSE = (
    ROOT
    / "vllm"
    / "v1"
    / "attention"
    / "backends"
    / "mla"
    / "b12x_mla_sparse.py"
)
PCIE = ROOT / "b12x" / "distributed" / "pcie_dma.py"
WORKER = ROOT / "vllm" / "v1" / "worker" / "gpu_worker.py"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    result: list[ast.Call] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        target = child.func
        while isinstance(target, ast.Subscript):
            target = target.value
        if (
            isinstance(target, ast.Name)
            and target.id == name
            or isinstance(target, ast.Attribute)
            and target.attr == name
        ):
            result.append(child)
    return result


def main() -> None:
    sparse_text = SPARSE.read_text()
    pcie_text = PCIE.read_text()
    worker_text = WORKER.read_text()
    sparse_tree = ast.parse(sparse_text, filename=str(SPARSE))
    pcie_tree = ast.parse(pcie_text, filename=str(PCIE))
    worker_tree = ast.parse(worker_text, filename=str(WORKER))

    assert "_CKV_ACTIVE_CAPACITY = 2403" in sparse_text
    assert "packed_blocks > _CKV_ACTIVE_CAPACITY" in sparse_text
    assert "communicator_slab=0" in sparse_text
    assert "GroupCoordinator.all_gather fallback is forbidden" in sparse_text
    assert "Packed-CKV physical-pool route activated" in sparse_text

    forward = _function(sparse_tree, "dcp_packed_ckv_forward")
    direct_gathers = _calls(forward, "_ckv_direct_all_gather_into")
    assert len(direct_gathers) == 4, len(direct_gathers)
    assert len(_calls(forward, "_remap_ckv_pool_topk_kernel")) == 1
    assert len(_calls(forward, "release_ckv_headroom_after_first_attention")) == 1

    direct_helper = _function(sparse_tree, "_ckv_direct_all_gather_into")
    assert len(_calls(direct_helper, "all_gather_into_tensor")) == 1
    assert len(_calls(direct_helper, "all_gather")) == 1

    assert "_CKV_ESCROW_BYTES = 192 * (1 << 20)" in pcie_text
    assert "_CKV_HEADROOM_GATE_BYTES = 150 * (1 << 20)" in pcie_text
    escrow = _function(pcie_tree, "release_after_first_attention")
    assert len(_calls(escrow, "cudaFree")) == 0
    assert len(_calls(escrow, "_free_local")) == 1
    assert len(_calls(escrow, "_probe")) == 1
    layer_entry = _function(pcie_tree, "layer_entry")
    assert len(_calls(layer_entry, "_probe")) == 1

    warmup = _function(worker_tree, "compile_or_warm_up_model")
    worker_lines = worker_text.splitlines()
    empty_cache_line = max(
        call.lineno for call in _calls(warmup, "empty_cache")
    )
    arm_line = _calls(warmup, "arm_ckv_headroom_escrow")[0].lineno
    seed_line = max(call.lineno for call in _calls(warmup, "set_random_seed"))
    assert empty_cache_line < arm_line < seed_line, (
        empty_cache_line,
        arm_line,
        seed_line,
        worker_lines[arm_line - 1],
    )

    print(
        "PASS phase-2 source contract: active_cap=2403 "
        "direct_gathers=4 pool_remap=1 escrow=192MiB gate=150MiB "
        "arm_seam=post-empty-cache"
    )


if __name__ == "__main__":
    main()
