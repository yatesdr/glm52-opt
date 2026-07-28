#!/usr/bin/env python3
"""Evaluate segmented exact selection inside the real TP4/DCP4 candidate pool.

Each DCP rank first contributes its exact local top-k, so the production merge
has at most ``world_size * topk`` candidates.  This proof keeps that bound and
compares deterministic chronological coverage policies on frozen real GLM
activations:

1. reserve the exact-score best ``floor`` candidates in each equal
   chronological segment;
2. fill all remaining slots with the exact global winners from the same
   candidate pool.

The proof does not model v19's threshold-bucket overflow and never selects a
row outside the bounded TP4 candidate union.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from v20_indexer_age_aware_policy_proof import _policy_metrics, _score_sources
from v20_indexer_dcp_candidate_policy_proof import (
    _candidate_rows,
    _needle_report,
)
from v20_indexer_real_activation_gpu_quant_proof import TOPK


def _selection_with_segment_floor(
    scores: torch.Tensor,
    positions: torch.Tensor,
    *,
    history_end: int,
    segments: int,
    floor: int,
) -> torch.Tensor:
    if segments <= 0:
        raise ValueError("segments must be positive")
    if floor < 0 or floor * segments > TOPK:
        raise ValueError("segment floor is outside the top-k budget")

    selected = torch.zeros(int(scores.numel()), dtype=torch.bool)
    if floor:
        for segment in range(segments):
            start = history_end * segment // segments
            end = history_end * (segment + 1) // segments
            segment_rows = torch.nonzero(
                (positions >= start) & (positions < end),
                as_tuple=False,
            ).flatten()
            width = min(floor, int(segment_rows.numel()))
            if width:
                local = torch.topk(
                    scores[segment_rows],
                    width,
                    largest=True,
                    sorted=False,
                ).indices
                selected[segment_rows[local]] = True

    remaining = TOPK - int(selected.sum())
    if remaining:
        fill_scores = scores.clone()
        fill_scores[selected] = -torch.inf
        fill = torch.topk(
            fill_scores,
            remaining,
            largest=True,
            sorted=False,
        ).indices
        selected[fill] = True

    rows = torch.nonzero(selected, as_tuple=False).flatten()
    if int(rows.numel()) != TOPK:
        raise RuntimeError(
            f"segmented policy emitted {int(rows.numel())}, expected {TOPK}"
        )
    return rows


def run(
    *,
    trace_dir: Path,
    device: torch.device,
    chunk_rows: int,
    world_size: int,
    interleave: int,
    needle_fraction: float,
    needle_radius: int,
    segments_values: tuple[int, ...],
    floor_values: tuple[int, ...],
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
        policies: list[dict[str, Any]] = []
        for segments in segments_values:
            for floor in floor_values:
                if floor * segments > TOPK:
                    continue
                selected_local = _selection_with_segment_floor(
                    candidate_scores,
                    candidate_positions,
                    history_end=metadata["tokens"],
                    segments=segments,
                    floor=floor,
                )
                selected_rows = candidates[selected_local]
                policies.append(
                    {
                        "segments": segments,
                        "floor_per_segment": floor,
                        "reserved_budget": segments * floor,
                        **_needle_report(
                            selected_rows,
                            positions,
                            needle_center=needle_center,
                            needle_radius=needle_radius,
                        ),
                        **_policy_metrics(
                            rows=selected_rows,
                            scores=full_scores,
                            oracle_scores=oracle_scores,
                            oracle_rows=oracle_rows,
                            positions=positions,
                            needle_center=needle_center,
                            needle_radius=needle_radius,
                        ),
                    }
                )
        reports[source] = {
            "candidate_count": int(candidates.numel()),
            "rank_candidate_counts": rank_counts,
            "candidate_needle": _needle_report(
                candidates,
                positions,
                needle_center=needle_center,
                needle_radius=needle_radius,
            ),
            "policies": policies,
        }

    return {
        "schema": "v20-indexer-dcp-segment-policy-proof-v1",
        "claim_boundary": (
            "one frozen layer/query proof inside the production-bounded DCP "
            "candidate union; end-to-end causal validation remains required"
        ),
        "geometry": {
            **metadata,
            "topk": TOPK,
            "world_size": world_size,
            "interleave": interleave,
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
    parser.add_argument(
        "--segments",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
    )
    parser.add_argument(
        "--floors",
        type=int,
        nargs="+",
        default=[0, 32, 64, 128, 256, 512],
    )
    args = parser.parse_args()
    if args.world_size <= 1:
        raise ValueError("--world-size must exceed one")
    if args.interleave <= 0:
        raise ValueError("--interleave must be positive")

    report = run(
        trace_dir=args.trace_dir,
        device=torch.device(args.device),
        chunk_rows=args.chunk_rows,
        world_size=args.world_size,
        interleave=args.interleave,
        needle_fraction=args.needle_fraction,
        needle_radius=args.needle_radius,
        segments_values=tuple(args.segments),
        floor_values=tuple(args.floors),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
