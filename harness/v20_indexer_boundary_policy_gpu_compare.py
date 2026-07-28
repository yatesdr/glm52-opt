#!/usr/bin/env python3
"""Compare actual GPU oldest-boundary outputs with the CPU policy oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from v20_indexer_boundary_policy_cpu_proof import (
    TOPK,
    _dense_scores,
    _select,
    _set_metrics,
)


def _run_rank(
    *,
    trace_path: Path,
    replay_path: Path,
) -> dict[str, Any]:
    trace = torch.load(trace_path, map_location="cpu", weights_only=True)
    replay = torch.load(replay_path, map_location="cpu", weights_only=True)
    scores = _dense_scores(trace).numpy()
    expected, metadata = _select(
        scores,
        topk=TOPK,
        cap=4096,
        policy="oldest",
    )
    actual = replay["replay_indices"].to(torch.int64).numpy()
    metrics = _set_metrics(actual, expected)
    return {
        "rank": int(trace["tp_rank"]),
        "metrics": metrics,
        "policy": metadata,
        "pass": bool(metrics["set_exact"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [
        _run_rank(
            trace_path=(
                args.trace_root
                / f"tp{rank}"
                / "layer00-indexer-local.pt"
            ),
            replay_path=args.replay_root / f"tp{rank}-replay.pt",
        )
        for rank in range(args.ranks)
    ]
    report = {
        "schema": "v20-indexer-boundary-policy-gpu-compare-v1",
        "results": results,
        "pass": all(result["pass"] for result in results),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
