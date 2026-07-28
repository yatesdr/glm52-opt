#!/usr/bin/env python3
"""Fail closed unless two indexer replay directories emit identical sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for rank in range(args.ranks):
        first = torch.load(
            args.first / f"tp{rank}-replay.pt",
            map_location="cpu",
            weights_only=True,
        )
        second = torch.load(
            args.second / f"tp{rank}-replay.pt",
            map_location="cpu",
            weights_only=True,
        )
        first_set = set(int(value) for value in first["replay_indices"].tolist())
        second_set = set(int(value) for value in second["replay_indices"].tolist())
        first_scores = {
            int(index): float(score)
            for index, score in zip(
                first["replay_indices"].tolist(),
                first["replay_scores"].tolist(),
                strict=True,
            )
        }
        second_scores = {
            int(index): float(score)
            for index, score in zip(
                second["replay_indices"].tolist(),
                second["replay_scores"].tolist(),
                strict=True,
            )
        }
        common = first_set & second_set
        first_only = sorted(first_set - second_set)
        second_only = sorted(second_set - first_set)
        results.append(
            {
                "rank": rank,
                "first_count": len(first_set),
                "second_count": len(second_set),
                "intersection": len(common),
                "set_exact": first_set == second_set,
                "first_only": [
                    {"index": index, "score": first_scores[index]}
                    for index in first_only
                ],
                "second_only": [
                    {"index": index, "score": second_scores[index]}
                    for index in second_only
                ],
            }
        )
    report = {
        "schema": "v20-indexer-replay-repeatability-v1",
        "results": results,
        "pass": all(result["set_exact"] for result in results),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
