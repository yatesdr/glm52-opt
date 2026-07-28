#!/usr/bin/env python3
"""Fail-closed smoke test for the long-context trace custom operator."""

from __future__ import annotations

import os
from pathlib import Path

import torch

# Importing the model module registers torch.ops.vllm.longctx_layer_trace.
import vllm.model_executor.models.deepseek_v2  # noqa: F401


def traced_add(
    positions: torch.Tensor,
    hidden: torch.Tensor,
    residual: torch.Tensor,
    topk: torch.Tensor,
) -> torch.Tensor:
    torch.ops.vllm.longctx_layer_trace(
        positions,
        hidden,
        residual,
        topk,
        0,
        "input",
    )
    return hidden + 1


def main() -> None:
    trace_dir = Path(os.environ["VLLM_LONGCTX_TRACE_DIR"])
    batch_tokens = int(os.environ["VLLM_LONGCTX_TRACE_BATCH_TOKENS"])
    trace_path = trace_dir / "tp0" / "layer00-input.pt"
    if trace_path.exists():
        raise RuntimeError(f"stale trace output exists: {trace_path}")

    positions = torch.arange(batch_tokens, device="cuda", dtype=torch.long)
    hidden = torch.randn(batch_tokens, 64, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(hidden)
    topk = torch.arange(batch_tokens * 8, device="cuda", dtype=torch.int32).view(
        batch_tokens, 8
    )

    compiled = torch.compile(traced_add, fullgraph=True)
    output = compiled(positions, hidden, residual, topk)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, hidden + 1)
    if not trace_path.is_file():
        raise RuntimeError(f"compiled execution did not write {trace_path}")
    record = torch.load(trace_path, map_location="cpu", weights_only=False)
    assert record["schema"] == "v20-longctx-first-divergence-v1"
    assert record["layer"] == 0
    assert record["stage"] == "input"
    assert record["batch_tokens"] == batch_tokens
    assert record["absolute_position"] == batch_tokens - 1
    torch.testing.assert_close(record["hidden"], hidden[-1].cpu())
    torch.testing.assert_close(record["residual"], residual[-1].cpu())
    torch.testing.assert_close(record["topk_indices"], topk[-1].cpu())
    print(f"PASS fullgraph trace custom op: {trace_path}")


if __name__ == "__main__":
    main()
