#!/usr/bin/env python3
"""Prove whether a deep needle survives the TP4/DCP4 candidate merge.

Each DCP rank selects ``topk`` entries from its interleaved local history.
vLLM then merges ``world_size * topk`` candidates to a global ``topk``.
This proof uses frozen real GLM activations to answer two questions:

1. Does the needle survive each rank-local exact selection and reach the
   global DCP candidate table?
2. If it does, does a deterministic minimum for the older half recover it
   using only that already-bounded table?

No historical overflow or truncated radix bucket is modeled here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from v20_indexer_age_aware_policy_proof import (
    _policy_metrics,
    _score_sources,
)
from v20_indexer_real_activation_gpu_quant_proof import TOPK


def _dcp_owner(
    positions: torch.Tensor,
    *,
    world_size: int,
    interleave: int,
) -> torch.Tensor:
    return torch.div(positions, interleave, rounding_mode="floor") % world_size


def _candidate_rows(
    scores: torch.Tensor,
    positions: torch.Tensor,
    *,
    world_size: int,
    interleave: int,
) -> tuple[torch.Tensor, list[int]]:
    owner = _dcp_owner(
        positions,
        world_size=world_size,
        interleave=interleave,
    )
    rank_rows: list[torch.Tensor] = []
    rank_counts: list[int] = []
    for rank in range(world_size):
        local_rows = torch.nonzero(owner == rank, as_tuple=False).flatten()
        width = min(TOPK, int(local_rows.numel()))
        selected_local = torch.topk(
            scores[local_rows],
            width,
            largest=True,
            sorted=False,
        ).indices
        rank_rows.append(local_rows[selected_local])
        rank_counts.append(width)
    candidates = torch.cat(rank_rows)
    if int(torch.unique(candidates).numel()) != int(candidates.numel()):
        raise RuntimeError("DCP candidate rows are not disjoint across ranks")
    return candidates, rank_counts


def _needle_report(
    rows: torch.Tensor,
    positions: torch.Tensor,
    *,
    needle_center: int,
    needle_radius: int,
) -> dict[str, Any]:
    selected_positions = positions[rows]
    selected = selected_positions[
        (selected_positions - needle_center).abs() <= needle_radius
    ]
    return {
        "selected": bool(selected.numel()),
        "selected_tokens": [int(value) for value in selected.tolist()],
    }


def _selection_with_older_half_floor(
    scores: torch.Tensor,
    positions: torch.Tensor,
    *,
    history_end: int,
    floor: int,
) -> torch.Tensor:
    """Reserve the exact ``floor`` best candidates from the older half."""
    selected = torch.zeros(int(scores.numel()), dtype=torch.bool)
    old_rows = torch.nonzero(
        positions < history_end // 2,
        as_tuple=False,
    ).flatten()
    width = min(floor, int(old_rows.numel()))
    if width:
        old_local = torch.topk(
            scores[old_rows],
            width,
            largest=True,
            sorted=False,
        ).indices
        selected[old_rows[old_local]] = True
    remaining = TOPK - int(selected.sum().item())
    fill_scores = scores.clone()
    fill_scores[selected] = -torch.inf
    fill = torch.topk(
        fill_scores,
        remaining,
        largest=True,
        sorted=False,
    ).indices
    selected[fill] = True
    return torch.nonzero(selected, as_tuple=False).flatten()


def run(
    *,
    trace_dir: Path,
    device: torch.device,
    chunk_rows: int,
    world_size: int,
    interleave: int,
    floor: int,
    needle_fraction: float,
    needle_radius: int,
) -> dict[str, Any]:
    scores_by_source, positions, metadata = _score_sources(
        trace_dir=trace_dir,
        device=device,
        chunk_rows=chunk_rows,
    )
    oracle_scores = scores_by_source["bf16_oracle"]
    oracle_rows = torch.topk(
        oracle_scores,
        TOPK,
        largest=True,
        sorted=False,
    ).indices
    needle_center = int(
        round(metadata["runtime_absolute_position"] * needle_fraction)
    )

    reports: dict[str, Any] = {}
    for source, full_scores in scores_by_source.items():
        candidates, rank_counts = _candidate_rows(
            full_scores,
            positions,
            world_size=world_size,
            interleave=interleave,
        )
        candidate_scores = full_scores[candidates]
        candidate_positions = positions[candidates]
        exact_local = torch.topk(
            candidate_scores,
            TOPK,
            largest=True,
            sorted=False,
        ).indices
        floor_local = _selection_with_older_half_floor(
            candidate_scores,
            candidate_positions,
            history_end=metadata["tokens"],
            floor=floor,
        )
        exact_rows = candidates[exact_local]
        floor_rows = candidates[floor_local]
        reports[source] = {
            "candidate_count": int(candidates.numel()),
            "rank_candidate_counts": rank_counts,
            "candidate_needle": _needle_report(
                candidates,
                positions,
                needle_center=needle_center,
                needle_radius=needle_radius,
            ),
            "candidate_position_quartiles": [
                int(
                    (
                        (
                            candidate_positions
                            >= metadata["tokens"] * quarter // 4
                        )
                        & (
                            candidate_positions
                            < metadata["tokens"] * (quarter + 1) // 4
                        )
                    ).sum().item()
                )
                for quarter in range(4)
            ],
            "exact_merge": {
                **_needle_report(
                    exact_rows,
                    positions,
                    needle_center=needle_center,
                    needle_radius=needle_radius,
                ),
                **_policy_metrics(
                    rows=exact_rows,
                    scores=full_scores,
                    oracle_scores=oracle_scores,
                    oracle_rows=oracle_rows,
                    positions=positions,
                    needle_center=needle_center,
                    needle_radius=needle_radius,
                ),
            },
            "older_half_floor_merge": {
                "floor": floor,
                **_needle_report(
                    floor_rows,
                    positions,
                    needle_center=needle_center,
                    needle_radius=needle_radius,
                ),
                **_policy_metrics(
                    rows=floor_rows,
                    scores=full_scores,
                    oracle_scores=oracle_scores,
                    oracle_rows=oracle_rows,
                    positions=positions,
                    needle_center=needle_center,
                    needle_radius=needle_radius,
                ),
            },
        }

    return {
        "schema": "v20-indexer-dcp-candidate-policy-proof-v1",
        "claim_boundary": (
            "one frozen layer/query candidate-survival proof; an end-to-end "
            "causal boot remains required"
        ),
        "geometry": {
            **metadata,
            "topk": TOPK,
            "world_size": world_size,
            "interleave": interleave,
            "needle_center": needle_center,
            "needle_radius": needle_radius,
            "older_half_floor": floor,
        },
        "score_sources": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--interleave", type=int, default=1)
    parser.add_argument("--floor", type=int, default=64)
    parser.add_argument("--needle-fraction", type=float, default=0.4)
    parser.add_argument("--needle-radius", type=int, default=24)
    args = parser.parse_args()
    if args.world_size <= 1:
        raise ValueError("--world-size must exceed one")
    if args.interleave <= 0:
        raise ValueError("--interleave must be positive")
    if args.floor < 0 or args.floor * 2 > TOPK:
        raise ValueError("--floor is outside the top-k budget")

    result = run(
        trace_dir=args.trace_dir,
        device=torch.device(args.device),
        chunk_rows=args.chunk_rows,
        world_size=args.world_size,
        interleave=args.interleave,
        floor=args.floor,
        needle_fraction=args.needle_fraction,
        needle_radius=args.needle_radius,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
