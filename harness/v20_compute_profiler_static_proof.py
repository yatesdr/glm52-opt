#!/usr/bin/env python3
"""GPU-free integration proof for the current-head v20 compute profiler."""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TREE = ROOT / "workspace/vllm-v20-compute-profiler-head"
EXPECTED_BASE = "89b4a98d1ffebb2dda1e1ac5e55238e3a9cfbd58"
INTEGRATED_FILES = (
    "vllm/compilation/passes/fusion/allreduce_rms_fusion.py",
    "vllm/distributed/parallel_state.py",
    "vllm/model_executor/layers/attention/mla_attention.py",
    "vllm/model_executor/layers/compute_phase_profiler.py",
    "vllm/model_executor/layers/mla.py",
    "vllm/model_executor/layers/quantization/nvfp4_nf3_hybrid.py",
    "vllm/model_executor/models/deepseek_v2.py",
    "vllm/v1/attention/backends/mla/b12x_mla_sparse.py",
    "vllm/v1/attention/ops/common.py",
    "vllm/v1/attention/ops/dcp_alltoall.py",
    "vllm/v1/worker/gpu_worker.py",
)


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(tree), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _text(tree: Path, relative: str) -> str:
    return (tree / relative).read_text(encoding="utf-8")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_name(node.value)}.{node.attr}"
    return ""


def _jit_functions(source: str, filename: str) -> dict[str, str]:
    tree = ast.parse(source, filename=filename)
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = {_call_name(item) for item in node.decorator_list}
        if "triton.jit" in decorators:
            result[node.name] = ast.dump(node, include_attributes=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument(
        "--expected-parent",
        default=EXPECTED_BASE,
        help="Exact parent revision whose runtime semantics must be preserved.",
    )
    args = parser.parse_args()
    tree = args.tree.resolve()
    expected_parent = args.expected_parent

    assert _git(tree, "rev-parse", "HEAD^").strip() == expected_parent
    subprocess.run(("git", "-C", str(tree), "diff", "--check"), check=True)
    for relative in INTEGRATED_FILES:
        ast.parse(_text(tree, relative), filename=relative)

    profiler = _text(tree, "vllm/model_executor/layers/compute_phase_profiler.py")
    assert 'init_logger("vllm.compute_phase_profiler")' in profiler
    profiler_tree = ast.parse(
        profiler,
        filename="vllm/model_executor/layers/compute_phase_profiler.py",
    )
    for node in ast.walk(profiler_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assert all(
            "compiler.disable" not in _call_name(decorator)
            for decorator in node.decorator_list
        ), f"graph-breaking profiler hook: {node.name}"
    assert profiler.count('mutates_args=["dependency"]') == 6
    assert 'external = self.mode == "decode"' in profiler
    assert "capture_replay_counter.add_(1)" in profiler
    assert "CUDAGraphMode.NONE" in profiler
    assert "ledger_valid" in profiler and "ordinal_valid" in profiler
    assert "phase_count_valid" in profiler
    for phase in (
        "moe_hybrid_kernel",
        "moe_kept_kernel",
        "moe_nf3_kernel",
        "moe_tier_sum",
        "ckv_prefetch_pack",
        "ckv_prefetch_ag",
    ):
        assert f'"{phase}"' in profiler

    model = _text(tree, "vllm/model_executor/models/deepseek_v2.py")
    moe = model[model.index("class DeepseekV2MoE") :]
    assert moe.index('phase_start(hidden_states, "moe_stack")') < moe.index(
        "self.experts("
    ) < moe.index('phase_stop(final_hidden_states, "moe_stack")')
    decoder = model[model.index("class DeepseekV2DecoderLayer") :]
    assert decoder.index("self.input_layernorm(") < decoder.index(
        "compute_profiler_layer_enter("
    ) < decoder.index("self.self_attn(")
    assert decoder.index("self.post_attention_layernorm(") < decoder.index(
        "compute_profiler_after_attention_norm("
    ) < decoder.index("self.mlp(")

    hybrid = _text(
        tree,
        "vllm/model_executor/layers/quantization/nvfp4_nf3_hybrid.py",
    )
    one_grid = hybrid[hybrid.index("if state.grid188_ready") :]
    assert one_grid.index('phase_start(x, "moe_hybrid_kernel")') < (
        one_grid.index("self._run_grid188(")
    ) < one_grid.index('phase_stop(output, "moe_hybrid_kernel")')
    for phase in ("moe_kept_kernel", "moe_nf3_kernel", "moe_tier_sum"):
        assert f'"{phase}"' in one_grid

    mla = _text(tree, "vllm/model_executor/layers/mla.py")
    tail = mla[mla.index('phase_start(q, "attention_path")') :]
    assert tail.index('phase_stop(attn_out, "attention_path")') < tail.index(
        "compute_profiler_o_proj_enter(attn_out)"
    ) < tail.index("return self.o_proj(attn_out)[0]")

    worker = _text(tree, "vllm/v1/worker/gpu_worker.py")
    warmup = worker[worker.index("def compile_or_warm_up_model(") :]
    assert warmup.index("compute_profiler_prepare()") < warmup.index(
        "self.model_runner.capture_model()"
    )
    assert 'logger.info("B12X_COMPUTE_PROF_V20 state=engine_started")' in worker

    capture = _text(tree, "vllm/v1/worker/gpu/cudagraph_utils.py")
    assert "for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]" in capture
    graph_body = capture[capture.index("torch.cuda.graph(graph, self.pool)") :]
    assert "forward_fn(CUDAGraphMode.NONE)" in graph_body

    # This branch must remain measurement-only. The older scratch tree also
    # contained a decode-query wire experiment; accepting it here would make
    # the resulting ledger an A/B of two changes.
    aggregate = "\n".join(_text(tree, relative) for relative in INTEGRATED_FILES)
    assert "VLLM_B12X_DCP_QUERY_FP8" not in aggregate
    assert "allow_fp8_wire" not in aggregate

    triton_checked = 0
    for relative in (
        "vllm/v1/attention/ops/common.py",
        "vllm/v1/attention/ops/dcp_alltoall.py",
        "vllm/v1/attention/backends/mla/b12x_mla_sparse.py",
    ):
        output = _text(tree, relative)
        base = _git(tree, "show", f"{expected_parent}:{relative}")
        output_jit = _jit_functions(output, relative)
        base_jit = _jit_functions(base, f"HEAD:{relative}")
        assert output_jit == base_jit, f"Triton kernel delta in {relative}"
        triton_checked += len(output_jit)
    assert triton_checked > 0

    print(
        "PASS v20 compute-profiler integration: current head, syntax, "
        "capture selection, complete attention/MoE ledger, no query-wire "
        f"experiment, no Triton delta ({triton_checked} kernels checked)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
