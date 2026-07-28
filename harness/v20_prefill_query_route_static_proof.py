#!/usr/bin/env python3
"""Prove whether the staged BF16->FP8 query guard can affect cold prefill.

This is deliberately a source/compose proof: it does not import vLLM,
SparkInfer, torch, or require a GPU.  It checks the exact routing predicates
that decide whether the tiny-M fused BF16 MLA query kernel can run and then
evaluates the configured single-request prefill chunk geometry.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPOSE = ROOT / "deploy/glm52-v20-prod-ready-20260724.yaml"
DEFAULT_VLLM = ROOT / "workspace/vllm-v20-staged-bf16-fp8-query"
DEFAULT_SPARKINFER = ROOT / "workspace/sparkinfer-v20-current-recovery"
DEFAULT_PROMPT_TOKENS = (7889, 54091)


def _literal_assignment(path: Path, name: str) -> object:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        value = node.value
        if value is None:
            break
        return ast.literal_eval(value)
    raise AssertionError(f"{path}: literal assignment for {name} not found")


def _function_source(path: Path, name: str) -> str:
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(text, node)
            if segment is None:
                break
            return segment
    raise AssertionError(f"{path}: function {name} not found")


def _max_num_batched_tokens(path: Path) -> int:
    match = re.search(r"--max-num-batched-tokens=(\d+)", path.read_text())
    if match is None:
        raise AssertionError(f"{path}: --max-num-batched-tokens is not pinned")
    return int(match.group(1))


def _chunk_sizes(prompt_tokens: int, chunk_size: int) -> tuple[int, ...]:
    full, tail = divmod(prompt_tokens, chunk_size)
    chunks = (chunk_size,) * full
    return chunks + ((tail,) if tail else ())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose", type=Path, default=DEFAULT_COMPOSE)
    parser.add_argument("--vllm", type=Path, default=DEFAULT_VLLM)
    parser.add_argument("--sparkinfer", type=Path, default=DEFAULT_SPARKINFER)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=DEFAULT_PROMPT_TOKENS,
        help="single-request cold-prefill prompt lengths to evaluate",
    )
    args = parser.parse_args()

    bmm_path = (
        args.vllm
        / "vllm/model_executor/kernels/attention/b12x_mxfp8_bmm.py"
    )
    attention_path = (
        args.vllm
        / "vllm/model_executor/layers/attention/mla_attention.py"
    )
    fused_path = (
        args.sparkinfer
        / "sparkinfer/gemm/mla_query_projection/_bf16.py"
    )

    max_batch = _max_num_batched_tokens(args.compose)
    fused_max_m = int(_literal_assignment(fused_path, "_MAX_M"))

    can_implement = _function_source(fused_path, "can_implement")
    assert "1 <= int(max_m) <= _MAX_M" in can_implement

    vllm_gate = _function_source(bmm_path, "can_implement_bf16_mla_query")
    assert "output_dtype != torch.bfloat16" in vllm_gate
    assert "return False" in vllm_gate

    fused_dispatch = _function_source(attention_path, "_try_fused_mla_query")
    assert "max_m=num_tokens" in fused_dispatch

    assert max_batch > fused_max_m
    reports: list[str] = []
    for prompt_tokens in args.prompt_tokens:
        if prompt_tokens <= 0:
            raise AssertionError("prompt token counts must be positive")
        chunks = _chunk_sizes(prompt_tokens, max_batch)
        smallest = min(chunks)
        assert smallest > fused_max_m, (
            f"{prompt_tokens} tokens has a {smallest}-row tail that enters the "
            f"fused M<={fused_max_m} envelope"
        )
        reports.append(
            f"prompt={prompt_tokens} chunks={len(chunks)} "
            f"full={max_batch} tail={chunks[-1]} min={smallest}"
        )

    print(
        "PASS staged-query prefill-route proof: "
        f"scheduler_chunk={max_batch} fused_max_m={fused_max_m}"
    )
    for report in reports:
        print(report)
    print(
        "finding=the BF16-output-only safety guard changes tiny-M "
        "decode/speculative routing, not these single-request cold-prefill batches"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
