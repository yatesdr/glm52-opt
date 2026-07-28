#!/usr/bin/env python3
"""GPU-free proof for the latest-head v20 production integration branch."""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TREE = ROOT / "workspace/vllm-v20-prod-integration-head"
EXPECTED_BASE = "89b4a98d1ffebb2dda1e1ac5e55238e3a9cfbd58"
EXPECTED_HEAD = "625ac3b75bf26741b4d8de06a46ec803a8a80f23"
EXPECTED_SUBJECTS = (
    "Add bounded filesystem tier capacity",
    "exl3(mtp): gate use_flattening off for SM120+B12X native next_n>2 path",
    "test(mla): cover B12X native MTP flattening policy",
    "perf(mla): release absorbed B12X MXFP8 projection weights",
    "docs(mla): document absorbed-weight helpers",
    "test(mla): adapt absorbed reclaim coverage to fused query wrapper",
    "fix(mla): preserve staged BF16-to-FP8 query path",
    "fix(attention): keep compact NVFP4 MTP on qualified path",
    "fix(mrv2): reuse profiled CUDA graph pool",
)
EXPECTED_FILES = {
    "docs/features/kv_offloading_usage.md",
    "tests/model_executor/layers/test_sparse_attn_indexer_b12x.py",
    "tests/v1/attention/test_mla_backends.py",
    "tests/v1/attention/test_sparse_mla_backends.py",
    "tests/v1/cudagraph/test_breakable_cudagraph.py",
    "tests/v1/kv_offload/tiering/test_fs_tier.py",
    "vllm/model_executor/kernels/attention/b12x_mxfp8_bmm.py",
    "vllm/model_executor/layers/attention/mla_attention.py",
    "vllm/v1/attention/backend.py",
    "vllm/v1/attention/backends/mla/b12x_mla_sparse.py",
    "vllm/v1/attention/backends/mla/indexer.py",
    "vllm/v1/kv_offload/tiering/fs/manager.py",
    "vllm/v1/worker/gpu/model_runner.py",
}


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(tree), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _text(tree: Path, relative: str) -> str:
    return (tree / relative).read_text(encoding="utf-8")


def _function(source: str, name: str) -> ast.FunctionDef:
    parsed = ast.parse(source)
    for node in ast.walk(parsed):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _execute_pure_helper(source: str, name: str):
    node = _function(source, name)
    node.decorator_list = []
    node.returns = None
    for arg in (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ):
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace: dict[str, object] = {}
    exec(compile(module, f"<{name}>", "exec"), namespace)
    return namespace[name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    args = parser.parse_args()
    tree = args.tree.resolve()

    assert _git(tree, "rev-parse", "HEAD").strip() == EXPECTED_HEAD
    assert _git(tree, "rev-parse", "HEAD~9").strip() == EXPECTED_BASE
    assert not _git(tree, "status", "--short")
    subprocess.run(
        ("git", "-C", str(tree), "diff", "--check", f"{EXPECTED_BASE}..HEAD"),
        check=True,
    )

    subjects = tuple(
        line
        for line in _git(
            tree,
            "log",
            "--reverse",
            "--format=%s",
            f"{EXPECTED_BASE}..HEAD",
        ).splitlines()
        if line
    )
    assert subjects == EXPECTED_SUBJECTS
    changed = {
        line.split("\t", 1)[1]
        for line in _git(
            tree,
            "diff",
            "--name-status",
            f"{EXPECTED_BASE}..HEAD",
        ).splitlines()
    }
    assert changed == EXPECTED_FILES
    assert not (tree / "vllm/model_executor/layers/compute_phase_profiler.py").exists()

    for relative in EXPECTED_FILES:
        if relative.endswith(".py"):
            ast.parse(_text(tree, relative), filename=relative)

    manager = _text(tree, "vllm/v1/kv_offload/tiering/fs/manager.py")
    evict = ast.get_source_segment(
        manager,
        _function(manager, "_evict_until_fits_locked"),
    )
    reserve = ast.get_source_segment(manager, _function(manager, "_reserve_store"))
    store = ast.get_source_segment(manager, _function(manager, "_store_block_bounded"))
    assert evict is not None and reserve is not None and store is not None
    assert (
        "self._cache_size_bytes + self._pending_store_bytes + required_bytes"
        in evict
    )
    assert "self._pending_store_bytes += self._block_size" in reserve
    assert store.index("if not self._reserve_store(key)") < store.index("store_block(")

    indexer = _text(tree, "vllm/v1/attention/backends/mla/indexer.py")
    flatten = _execute_pure_helper(indexer, "_should_flatten_mtp_indexer")
    assert flatten(is_sm100_family=True, next_n=4, use_b12x_indexer=False) is False
    assert flatten(is_sm100_family=False, next_n=4, use_b12x_indexer=True) is False
    assert flatten(is_sm100_family=False, next_n=4, use_b12x_indexer=False) is True
    assert flatten(is_sm100_family=False, next_n=2, use_b12x_indexer=False) is False
    assert "use_b12x_sparse_indexer()" in indexer

    query = _text(
        tree,
        "vllm/model_executor/kernels/attention/b12x_mxfp8_bmm.py",
    )
    query_spec = ast.get_source_segment(
        query,
        _function(query, "_module_fused_mla_query_spec"),
    )
    assert query_spec is not None
    assert "if output_dtype != torch.bfloat16:" in query_spec
    assert query_spec.index("if output_dtype != torch.bfloat16:") < (
        query_spec.index('weight_format = "bf16"')
    )

    backend = _text(tree, "vllm/v1/attention/backends/mla/b12x_mla_sparse.py")
    route = _execute_pure_helper(backend, "_resolve_spec_decode_mode")
    assert route("0", kv_cache_dtype="fp8_ds_mla") == (False, False)
    assert route("auto", kv_cache_dtype="fp8_ds_mla") == (True, False)
    assert route("auto", kv_cache_dtype="nvfp4_ds_mla") == (False, False)
    assert route("1", kv_cache_dtype="nvfp4_ds_mla") == (True, True)

    mla = _text(tree, "vllm/model_executor/layers/attention/mla_attention.py")
    assert "def _release_b12x_mxfp8_kv_b_proj(" in mla
    assert "can_release_kv_b_proj_after_loading" in mla
    assert "delattr(layer, name)" in mla
    assert "layer.b12x_mxfp8_packed_weight = None" in mla
    assert "_release_b12x_mxfp8_kv_b_proj(self.kv_b_proj)" in mla

    runner = _text(tree, "vllm/v1/worker/gpu/model_runner.py")
    profile = ast.get_source_segment(
        runner,
        _function(runner, "profile_cudagraph_memory"),
    )
    assert profile is not None
    assert "current_platform.get_global_graph_pool()" in profile
    assert "manager.pool = profiling_pool" in profile
    assert "wrapper.graph_pool = profiling_pool" in profile

    authors = _git(
        tree,
        "log",
        "--reverse",
        "--format=%an <%ae>",
        f"{EXPECTED_BASE}..HEAD",
    )
    assert "Brandon Music <brandonmusic@pop-os.tail8674da.ts.net>" in authors
    assert "Martin Vit <martin@voipmonitor.org>" in authors

    print(
        "PASS v20 production integration: exact latest base, nine independent "
        "commits, bounded NVMe reservation, native B12X MTP3, absorbed-weight "
        "reclaim, staged BF16->FP8 safety, compact verifier qualification, "
        "MRV2 pool reuse, preserved contributor authorship, no profiler"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
