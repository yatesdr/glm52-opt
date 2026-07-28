#!/usr/bin/env python3
"""Calibrate local and global chronological coverage on real DCP activations.

The failed global-history-floor experiment established that the TP4 candidate
union contains only one needle-local token at the first retrieval-divergence
layer.  This proof moves the policy to the only seam that can recover the
missing candidates: each rank's bounded local top-k generation.

For every requested pair of local/global floors:

1. each DCP rank selects exactly ``topk`` rows from its own disjoint history,
   reserving ``local_floor`` exact-score winners per chronological segment;
2. the four local sets form the bounded ``world_size * topk`` candidate union;
3. the global merge selects exactly ``topk`` rows, optionally reserving
   ``global_floor`` exact-score winners per chronological segment.

No threshold bucket, overflow truncation, or out-of-bounds write is modeled.
All choices are exact-score winners within an explicit bounded region.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from v20_indexer_age_aware_policy_proof import _policy_metrics, _score_sources
from v20_indexer_dcp_candidate_policy_proof import _dcp_owner, _needle_report
from v20_indexer_dcp_segment_policy_proof import _selection_with_segment_floor
from v20_indexer_real_activation_gpu_quant_proof import TOPK


def _local_candidates(
    scores: torch.Tensor,
    positions: torch.Tensor,
    *,
    history_end: int,
    world_size: int,
    interleave: int,
    segments: int,
    floor: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    owners = _dcp_owner(
        positions,
        world_size=world_size,
        interleave=interleave,
    )
    candidates: list[torch.Tensor] = []
    rank_reports: list[dict[str, Any]] = []
    for rank in range(world_size):
        local_rows = torch.nonzero(owners == rank, as_tuple=False).flatten()
        local_selection = _selection_with_segment_floor(
            scores[local_rows],
            positions[local_rows],
            history_end=history_end,
            segments=segments,
            floor=floor,
        )
        selected_rows = local_rows[local_selection]
        candidates.append(selected_rows)
        rank_reports.append(
            {
                "rank": rank,
                "selected": int(selected_rows.numel()),
                "position_quartiles": [
                    int(
                        (
                            (positions[selected_rows] >= history_end * q // 4)
                            & (
                                positions[selected_rows]
                                < history_end * (q + 1) // 4
                            )
                        ).sum()
                    )
                    for q in range(4)
                ],
            }
        )
    candidate_rows = torch.cat(candidates)
    if int(torch.unique(candidate_rows).numel()) != int(candidate_rows.numel()):
        raise RuntimeError("rank-local candidate sets are not disjoint")
    return candidate_rows, rank_reports


def run(
    *,
    trace_dir: Path,
    device: torch.device,
    chunk_rows: int,
    world_size: int,
    interleave: int,
    needle_fraction: float,
    needle_radius: int,
    segments: int,
    local_floors: tuple[int, ...],
    global_floors: tuple[int, ...],
) -> dict[str, Any]:
    scores_by_source, positions, metadata = _score_sources(
        trace_dir=trace_dir,
        device=device,
        chunk_rows=chunk_rows,
    )
    history_end = int(metadata["tokens"])
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
        policies: list[dict[str, Any]] = []
        for local_floor in local_floors:
            if local_floor * segments > TOPK:
                continue
            candidate_rows, rank_reports = _local_candidates(
                full_scores,
                positions,
                history_end=history_end,
                world_size=world_size,
                interleave=interleave,
                segments=segments,
                floor=local_floor,
            )
            candidate_scores = full_scores[candidate_rows]
            candidate_positions = positions[candidate_rows]
            candidate_needle = _needle_report(
                candidate_rows,
                positions,
                needle_center=needle_center,
                needle_radius=needle_radius,
            )
            for global_floor in global_floors:
                if global_floor * segments > TOPK:
                    continue
                global_selection = _selection_with_segment_floor(
                    candidate_scores,
                    candidate_positions,
                    history_end=history_end,
                    segments=segments,
                    floor=global_floor,
                )
                final_rows = candidate_rows[global_selection]
                policies.append(
                    {
                        "segments": segments,
                        "local_floor_per_segment": local_floor,
                        "local_reserved_budget_per_rank": segments
                        * local_floor,
                        "global_floor_per_segment": global_floor,
                        "global_reserved_budget": segments * global_floor,
                        "candidate_count": int(candidate_rows.numel()),
                        "candidate_needle": candidate_needle,
                        "rank_reports": rank_reports,
                        **_needle_report(
                            final_rows,
                            positions,
                            needle_center=needle_center,
                            needle_radius=needle_radius,
                        ),
                        **_policy_metrics(
                            rows=final_rows,
                            scores=full_scores,
                            oracle_scores=oracle_scores,
                            oracle_rows=oracle_rows,
                            positions=positions,
                            needle_center=needle_center,
                            needle_radius=needle_radius,
                        ),
                    }
                )
        reports[source] = {"policies": policies}

    return {
        "schema": "v20-indexer-dcp-two-stage-policy-proof-v1",
        "claim_boundary": (
            "one frozen layer/query proof of bounded local+global selection; "
            "end-to-end causal validation remains required"
        ),
        "geometry": {
            **metadata,
            "topk": TOPK,
            "world_size": world_size,
            "interleave": interleave,
            "segments": segments,
            "needle_center": needle_center,
            "needle_radius": needle_radius,
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
    parser.add_argument("--needle-fraction", type=float, default=0.4)
    parser.add_argument("--needle-radius", type=int, default=24)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument(
        "--local-floors",
        type=int,
        nargs="+",
        default=[0, 16, 32, 64, 128, 256, 512],
    )
    parser.add_argument(
        "--global-floors",
        type=int,
        nargs="+",
        default=[0, 32, 64, 128, 256, 512],
    )
    args = parser.parse_args()
    if args.world_size <= 1:
        raise ValueError("--world-size must exceed one")
    if args.interleave <= 0:
        raise ValueError("--interleave must be positive")
    if args.segments <= 0:
        raise ValueError("--segments must be positive")

    report = run(
        trace_dir=args.trace_dir,
        device=torch.device(args.device),
        chunk_rows=args.chunk_rows,
        world_size=args.world_size,
        interleave=args.interleave,
        needle_fraction=args.needle_fraction,
        needle_radius=args.needle_radius,
        segments=args.segments,
        local_floors=tuple(args.local_floors),
        global_floors=tuple(args.global_floors),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
