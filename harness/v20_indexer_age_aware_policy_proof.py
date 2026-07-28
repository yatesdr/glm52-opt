#!/usr/bin/env python3
"""Evaluate deterministic age-aware selector policies on real GLM activations.

The v20 exact selector is exact for its quantized proxy, but a frozen 350k
failure showed that the full-BF16 GLM score also ranks the needle outside the
2,048-entry sparse-attention budget.  This proof asks a narrower design
question without booting the model: what is the smallest explicit
chronological coverage floor that includes the needle, and what exact-score
budget does that floor displace?

For each score source (the BF16 GLM oracle and the production E4M3/UE8M0
proxy), the policy:

1. reserves ``floor`` exact-score winners in every chronological segment;
2. fills the remaining budget with the exact global winners not already
   selected.

``floor=0`` is the current exact selector.  A fully balanced policy has
``floor = topk / segments``.  Every policy is deterministic and bounds-safe;
none relies on the historical threshold-bucket overflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from v20_indexer_hadamard_activation_proof import _load_trace
from v20_indexer_real_activation_gpu_quant_proof import (
    HEAD_DIM,
    TOPK,
    _quant_roundtrip,
    _score_chunk,
)


DEFAULT_SEGMENTS = (2, 4, 8, 16)
DEFAULT_FLOORS = (0, 16, 32, 64, 128, 256, 512, 1024)


def _selection_with_segment_floor(
    scores: torch.Tensor,
    *,
    segments: int,
    floor: int,
) -> torch.Tensor:
    rows = int(scores.numel())
    if segments <= 0:
        raise ValueError("segments must be positive")
    if floor < 0 or floor * segments > TOPK:
        raise ValueError("invalid segment floor")

    selected = torch.zeros(rows, dtype=torch.bool)
    if floor:
        for segment in range(segments):
            start = rows * segment // segments
            end = rows * (segment + 1) // segments
            width = min(floor, end - start)
            if width:
                local = torch.topk(
                    scores[start:end],
                    width,
                    largest=True,
                    sorted=False,
                ).indices
                selected[start + local] = True

    remaining = TOPK - int(selected.sum().item())
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

    rows_out = torch.nonzero(selected, as_tuple=False).flatten()
    if int(rows_out.numel()) != TOPK:
        raise RuntimeError(
            f"policy emitted {int(rows_out.numel())} rows, expected {TOPK}"
        )
    return rows_out


def _quartile_counts(rows: torch.Tensor, total: int) -> list[int]:
    return [
        int(
            (
                (rows >= total * quarter // 4)
                & (rows < total * (quarter + 1) // 4)
            ).sum().item()
        )
        for quarter in range(4)
    ]


def _policy_metrics(
    *,
    rows: torch.Tensor,
    scores: torch.Tensor,
    oracle_scores: torch.Tensor,
    oracle_rows: torch.Tensor,
    positions: torch.Tensor,
    needle_center: int,
    needle_radius: int,
) -> dict[str, Any]:
    selected_positions = positions[rows]
    oracle_positions = positions[oracle_rows]
    needle_mask = (positions - needle_center).abs() <= needle_radius
    needle_positions = positions[needle_mask]
    selected_needle = needle_positions[
        torch.isin(needle_positions, selected_positions)
    ]
    intersection = int(
        torch.isin(selected_positions, oracle_positions).sum().item()
    )
    oracle_sum = oracle_scores[oracle_rows].double().sum()
    selected_oracle_sum = oracle_scores[rows].double().sum()
    own_sum = scores[rows].double().sum()
    own_global_rows = torch.topk(
        scores,
        TOPK,
        largest=True,
        sorted=False,
    ).indices
    own_global_sum = scores[own_global_rows].double().sum()
    oracle_loss = oracle_sum - selected_oracle_sum
    own_loss = own_global_sum - own_sum
    if oracle_loss < -1e-6 or own_loss < -1e-6:
        raise RuntimeError(
            "segment-floor policy outscored an exact top-k reference: "
            f"oracle_loss={float(oracle_loss)}, own_loss={float(own_loss)}"
        )
    return {
        "needle_selected": bool(selected_needle.numel()),
        "needle_selected_tokens": [
            int(value) for value in selected_needle.tolist()
        ],
        "bf16_oracle_recall": intersection / TOPK,
        "bf16_oracle_displaced": TOPK - intersection,
        "bf16_oracle_score_sum_loss": float(oracle_loss),
        "bf16_oracle_score_mean_loss": float(oracle_loss / TOPK),
        "own_score_sum_loss_vs_global": float(own_loss),
        "own_score_mean_loss_vs_global": float(own_loss / TOPK),
        "selected_position_quartiles": _quartile_counts(
            rows,
            int(scores.numel()),
        ),
    }


def _score_sources(
    *,
    trace_dir: Path,
    device: torch.device,
    chunk_rows: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    trace = _load_trace(trace_dir)
    tail_start = int(trace["tail_start_position"])
    eligible = trace["positions"] < tail_start
    positions = torch.from_numpy(trace["positions"][eligible]).to(torch.int64)
    keys = torch.from_numpy(trace["k"][eligible]).to(torch.bfloat16)
    query = torch.from_numpy(trace["q"]).to(
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.from_numpy(trace["weights"]).to(
        device=device,
        dtype=torch.float32,
    )
    query_fp8 = _quant_roundtrip(query)[2]

    parts: dict[str, list[torch.Tensor]] = {
        "bf16_oracle": [],
        "fp8_ue8m0": [],
    }
    for start in range(0, int(keys.shape[0]), chunk_rows):
        end = min(start + chunk_rows, int(keys.shape[0]))
        key = keys[start:end].to(device)
        parts["bf16_oracle"].append(
            _score_chunk(query.float(), key.float(), weights).cpu()
        )
        key_fp8 = _quant_roundtrip(key)[2]
        parts["fp8_ue8m0"].append(
            _score_chunk(query_fp8, key_fp8, weights).cpu()
        )

    scores = {name: torch.cat(value) for name, value in parts.items()}
    metadata = {
        "runtime_absolute_position": int(trace["runtime_absolute_position"]),
        "tail_start_position": tail_start,
        "tokens": int(keys.shape[0]),
        "heads": int(query.shape[0]),
        "head_dim": int(query.shape[1]),
    }
    return scores, positions, metadata


def run(
    *,
    trace_dir: Path,
    device: torch.device,
    chunk_rows: int,
    needle_fraction: float,
    needle_radius: int,
    segments_values: tuple[int, ...],
    floor_values: tuple[int, ...],
) -> dict[str, Any]:
    scores, positions, metadata = _score_sources(
        trace_dir=trace_dir,
        device=device,
        chunk_rows=chunk_rows,
    )
    oracle_scores = scores["bf16_oracle"]
    oracle_rows = torch.topk(
        oracle_scores,
        TOPK,
        largest=True,
        sorted=False,
    ).indices
    needle_center = int(
        round(metadata["runtime_absolute_position"] * needle_fraction)
    )
    needle_mask = (positions - needle_center).abs() <= needle_radius
    if not bool(needle_mask.any()):
        raise RuntimeError("needle window is outside eligible history")

    source_reports: dict[str, Any] = {}
    for source_name, source_scores in scores.items():
        policies: list[dict[str, Any]] = []
        for segments in segments_values:
            valid_floors = sorted(
                {
                    floor
                    for floor in floor_values
                    if floor * segments <= TOPK
                }
            )
            for floor in valid_floors:
                rows = _selection_with_segment_floor(
                    source_scores,
                    segments=segments,
                    floor=floor,
                )
                policies.append(
                    {
                        "segments": segments,
                        "floor_per_segment": floor,
                        "reserved_entries": floor * segments,
                        **_policy_metrics(
                            rows=rows,
                            scores=source_scores,
                            oracle_scores=oracle_scores,
                            oracle_rows=oracle_rows,
                            positions=positions,
                            needle_center=needle_center,
                            needle_radius=needle_radius,
                        ),
                    }
                )
        source_reports[source_name] = {
            "needle_best_global_rank": int(
                (
                    source_scores
                    > source_scores[needle_mask].max()
                ).sum().item()
                + 1
            ),
            "policies": policies,
        }

    return {
        "schema": "v20-indexer-age-aware-policy-proof-v2",
        "claim_boundary": (
            "one real layer/query policy proof; a model boot remains required "
            "to establish end-to-end retrieval"
        ),
        "geometry": {
            **metadata,
            "topk": TOPK,
            "needle_center": needle_center,
            "needle_radius": needle_radius,
        },
        "score_sources": source_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--needle-fraction", type=float, default=0.4)
    parser.add_argument("--needle-radius", type=int, default=24)
    parser.add_argument(
        "--segments",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEGMENTS),
    )
    parser.add_argument(
        "--floors",
        type=int,
        nargs="+",
        default=list(DEFAULT_FLOORS),
    )
    args = parser.parse_args()

    report = run(
        trace_dir=args.trace_dir,
        device=torch.device(args.device),
        chunk_rows=args.chunk_rows,
        needle_fraction=args.needle_fraction,
        needle_radius=args.needle_radius,
        segments_values=tuple(args.segments),
        floor_values=tuple(args.floors),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
