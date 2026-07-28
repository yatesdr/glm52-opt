#!/usr/bin/env python3
"""Isolate tiled-topk repeatability from the learned indexer scorer.

The input row is reconstructed once on CPU from a preserved production trace,
then copied into the exact tiled float32 layout consumed by the selector.  Any
set drift across repetitions therefore belongs to tiled-topk itself, not the
FP8 query/key scorer that normally precedes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from v20_indexer_boundary_policy_cpu_proof import (
    TOPK,
    _dense_scores,
    _select,
    _set_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--block-q", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--supertile-k", type=int, default=131072)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.repetitions < 2:
        raise ValueError("at least two repetitions are required")

    from sparkinfer.attention.nsa_indexer.tiled_topk import (
        clear_tiled_topk_kernel_cache,
        run_tiled_supertile_topk,
    )

    record = torch.load(args.trace, map_location="cpu", weights_only=True)
    scores_cpu = _dense_scores(record).contiguous()
    seq_len = int(scores_cpu.numel())
    num_k_tiles = (seq_len + args.block_k - 1) // args.block_k
    padded = torch.full(
        (num_k_tiles * args.block_k,),
        float("-inf"),
        dtype=torch.float32,
    )
    padded[:seq_len] = scores_cpu
    tile_logits = torch.full(
        (num_k_tiles, args.block_q, args.block_k),
        float("-inf"),
        dtype=torch.float32,
        device="cuda",
    )
    tile_logits[:, 0, :] = padded.view(num_k_tiles, args.block_k).to("cuda")
    tile_logits = tile_logits.contiguous().view(-1)
    k_start = torch.zeros(1, dtype=torch.int32, device="cuda")
    k_end = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

    expected, policy = _select(
        scores_cpu.numpy(),
        topk=TOPK,
        cap=4096,
        policy="oldest",
    )
    clear_tiled_topk_kernel_cache()
    outputs: list[torch.Tensor] = []
    for _ in range(args.repetitions):
        _, indices = run_tiled_supertile_topk(
            tile_logits=tile_logits,
            k_start=k_start,
            k_end=k_end,
            topk=TOPK,
            block_q=args.block_q,
            block_k=args.block_k,
            supertile_k=args.supertile_k,
        )
        torch.cuda.synchronize()
        outputs.append(indices[0].cpu())

    baseline = outputs[0].to(torch.int64).numpy()
    repeats = []
    for repetition, output in enumerate(outputs):
        values = output.to(torch.int64).numpy()
        repeats.append(
            {
                "repetition": repetition,
                "vs_first": _set_metrics(values, baseline),
                "vs_cpu_policy": _set_metrics(values, expected),
            }
        )
    report = {
        "schema": "v20-oldest-boundary-selector-determinism-v1",
        "trace": str(args.trace),
        "input": {
            "seq_len": seq_len,
            "block_q": args.block_q,
            "block_k": args.block_k,
            "supertile_k": args.supertile_k,
            "topk": TOPK,
        },
        "policy": policy,
        "repeats": repeats,
        "pass": all(
            result["vs_first"]["set_exact"]
            and result["vs_cpu_policy"]["set_exact"]
            for result in repeats
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
