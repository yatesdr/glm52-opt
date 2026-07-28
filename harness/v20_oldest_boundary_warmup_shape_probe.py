#!/usr/bin/env python3
"""Exercise the production tiled selector at server-profile geometries.

This is intentionally model-free.  It covers the many-row, short-history
shape used while vLLM profiles a 3,072-token batch, which is materially
different from the single-row, long-history frozen selector proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=3072)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--block-q", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument(
        "--length-mode",
        choices=("ramp", "fixed"),
        default="ramp",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rows <= 0 or args.max_length <= 0:
        raise ValueError("rows and max-length must be positive")

    from sparkinfer.attention.nsa_indexer.tiled_topk import (
        clear_tiled_topk_kernel_cache,
        run_tiled_topk,
    )

    torch.manual_seed(20260727)
    device = torch.device("cuda")
    num_q_tiles = (args.rows + args.block_q - 1) // args.block_q
    num_k_tiles = (args.max_length + args.block_k - 1) // args.block_k
    logits = torch.randn(
        (
            num_q_tiles,
            num_k_tiles,
            args.block_q,
            args.block_k,
        ),
        dtype=torch.float32,
        device=device,
    ).contiguous()
    k_start = torch.zeros(args.rows, dtype=torch.int32, device=device)
    if args.length_mode == "ramp":
        k_end = torch.arange(
            1,
            args.rows + 1,
            dtype=torch.int32,
            device=device,
        ).clamp_max_(args.max_length)
    else:
        k_end = torch.full(
            (args.rows,),
            args.max_length,
            dtype=torch.int32,
            device=device,
        )

    clear_tiled_topk_kernel_cache()
    values, indices = run_tiled_topk(
        tile_logits=logits,
        k_start=k_start,
        k_end=k_end,
        topk=args.topk,
        block_q=args.block_q,
        block_k=args.block_k,
        num_k_tiles=num_k_tiles,
    )
    torch.cuda.synchronize()

    lengths = k_end.cpu().to(torch.int64)
    indices_cpu = indices.cpu().to(torch.int64)
    expected_valid = torch.minimum(
        lengths,
        torch.full_like(lengths, args.topk),
    )
    actual_valid = (indices_cpu >= 0).sum(dim=1)
    in_bounds = (indices_cpu < lengths.unsqueeze(1)) | (indices_cpu < 0)
    report = {
        "schema": "v20-oldest-boundary-warmup-shape-v1",
        "input": {
            "rows": args.rows,
            "max_length": args.max_length,
            "length_mode": args.length_mode,
            "topk": args.topk,
            "block_q": args.block_q,
            "block_k": args.block_k,
            "num_q_tiles": num_q_tiles,
            "num_k_tiles": num_k_tiles,
        },
        "valid_count_matches": bool(
            torch.equal(actual_valid, expected_valid)
        ),
        "all_indices_in_bounds": bool(in_bounds.all().item()),
        "finite_selected_values": int(torch.isfinite(values).sum().item()),
    }
    report["pass"] = bool(
        report["valid_count_matches"] and report["all_indices_in_bounds"]
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
