#!/usr/bin/env python3
"""Compare exact and bounded sparse-indexer selections on identical tensors.

The learned replay artifacts contain the selected logical indices and their
proxy scores. This report asks whether the bounded compatibility behavior has
a systematic positional effect, rather than treating set divergence alone as
an explanation for model recovery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _quantiles(values: torch.Tensor) -> dict[str, float | None]:
    if values.numel() == 0:
        return {key: None for key in ("min", "p10", "p25", "p50", "p75", "p90", "max")}
    values = values.to(torch.float64)
    probs = torch.tensor(
        [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0], dtype=torch.float64
    )
    result = torch.quantile(values, probs)
    return {
        key: float(value)
        for key, value in zip(
            ("min", "p10", "p25", "p50", "p75", "p90", "max"),
            result.tolist(),
            strict=True,
        )
    }


def _position_metrics(indices: torch.Tensor, seq_len: int) -> dict[str, Any]:
    indices = indices.to(torch.int64)
    normalized = indices.to(torch.float64) / max(seq_len - 1, 1)
    bins = torch.bincount(
        torch.clamp((normalized * 8).to(torch.int64), min=0, max=7),
        minlength=8,
    )
    return {
        "count": int(indices.numel()),
        "index_quantiles": _quantiles(indices),
        "normalized_quantiles": _quantiles(normalized),
        "octile_counts": [int(value) for value in bins.tolist()],
        "first_half": int(torch.count_nonzero(normalized < 0.5).item()),
        "last_quarter": int(torch.count_nonzero(normalized >= 0.75).item()),
    }


def _score_map(indices: torch.Tensor, scores: torch.Tensor) -> dict[int, float]:
    return {
        int(index): float(score)
        for index, score in zip(indices.tolist(), scores.tolist(), strict=True)
    }


def _load(path: Path) -> dict[str, Any]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    required = ("rank", "replay_indices", "replay_scores")
    missing = [key for key in required if key not in record]
    if missing:
        raise RuntimeError(f"{path}: missing fields {missing}")
    return record


def _analyze_rank(exact_path: Path, bounded_path: Path, seq_len: int) -> dict[str, Any]:
    exact = _load(exact_path)
    bounded = _load(bounded_path)
    if int(exact["rank"]) != int(bounded["rank"]):
        raise RuntimeError(f"rank mismatch: {exact_path} vs {bounded_path}")

    exact_indices = exact["replay_indices"].to(torch.int64)
    bounded_indices = bounded["replay_indices"].to(torch.int64)
    exact_set = set(int(value) for value in exact_indices.tolist())
    bounded_set = set(int(value) for value in bounded_indices.tolist())
    common = exact_set & bounded_set
    exact_only = exact_set - bounded_set
    bounded_only = bounded_set - exact_set

    exact_scores = _score_map(exact_indices, exact["replay_scores"])
    bounded_scores = _score_map(bounded_indices, bounded["replay_scores"])
    exact_only_scores = torch.tensor(
        [exact_scores[index] for index in sorted(exact_only)], dtype=torch.float64
    )
    bounded_only_scores = torch.tensor(
        [bounded_scores[index] for index in sorted(bounded_only)], dtype=torch.float64
    )

    exact_only_tensor = torch.tensor(sorted(exact_only), dtype=torch.int64)
    bounded_only_tensor = torch.tensor(sorted(bounded_only), dtype=torch.int64)
    return {
        "rank": int(exact["rank"]),
        "seq_len": seq_len,
        "intersection": len(common),
        "union": len(exact_set | bounded_set),
        "jaccard": len(common) / max(len(exact_set | bounded_set), 1),
        "exact_only": {
            "positions": _position_metrics(exact_only_tensor, seq_len),
            "proxy_score_quantiles": _quantiles(exact_only_scores),
        },
        "bounded_only": {
            "positions": _position_metrics(bounded_only_tensor, seq_len),
            "proxy_score_quantiles": _quantiles(bounded_only_scores),
        },
        "all_exact": _position_metrics(exact_indices, seq_len),
        "all_bounded": _position_metrics(bounded_indices, seq_len),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-dir", type=Path, required=True)
    parser.add_argument("--bounded-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = [
        _analyze_rank(
            args.exact_dir / f"tp{rank}-replay.pt",
            args.bounded_dir / f"tp{rank}-replay.pt",
            args.seq_len,
        )
        for rank in range(args.ranks)
    ]
    report = {
        "schema": "v20-indexer-selection-bias-v1",
        "exact_dir": str(args.exact_dir),
        "bounded_dir": str(args.bounded_dir),
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
