#!/usr/bin/env python3
"""GPU proof for the v20 local+global segmented-exact selector.

The probe consumes the frozen layer/query activation trace that exposed the
first irreversible long-context loss.  It recreates the production DCP4
candidate pipeline:

1. exact top-2048 candidates per 16k local chronological slice;
2. SparkInfer local 4x64 segmented fold on each rank;
3. vLLM global 4x256 segmented merge over the bounded 8192-row union.

Both stages are compared with the independent Torch policy oracle.  The test
also captures and replays the local selector to cover serving graph semantics.
No model is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from v20_indexer_age_aware_policy_proof import _score_sources
from v20_indexer_dcp_segment_policy_proof import (
    _selection_with_segment_floor,
)

from sparkinfer.attention.nsa_indexer.paged import _run_segmented_exact_fold
from vllm.model_executor.layers.sparse_attn_indexer import (
    _b12x_dcp_segmented_exact_specs,
    _select_b12x_dcp_candidates,
)

TOPK = 2048
WORLD_SIZE = 4
LOCAL_SLICE = 16384
LOCAL_FLOOR = 64
GLOBAL_FLOOR = 256
SEGMENTS = 4


def _rank_candidates(
    scores: torch.Tensor,
    positions: torch.Tensor,
    *,
    history_end: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    owned = torch.nonzero(
        torch.remainder(positions, WORLD_SIZE) == rank,
        as_tuple=False,
    ).flatten()
    local_positions = torch.div(
        positions[owned],
        WORLD_SIZE,
        rounding_mode="floor",
    ).to(torch.int32)
    local_scores = scores[owned]
    local_history_end = (history_end + WORLD_SIZE - 1 - rank) // WORLD_SIZE
    chunks: list[torch.Tensor] = []
    for start in range(0, local_history_end, LOCAL_SLICE):
        end = min(start + LOCAL_SLICE, local_history_end)
        in_slice = torch.nonzero(
            (local_positions >= start) & (local_positions < end),
            as_tuple=False,
        ).flatten()
        width = min(TOPK, int(in_slice.numel()))
        winners = in_slice[
            torch.topk(
                local_scores[in_slice],
                width,
                largest=True,
                sorted=False,
            ).indices
        ]
        if width < TOPK:
            pad = torch.full(
                (TOPK - width,),
                -1,
                dtype=torch.int64,
                device=scores.device,
            )
            winners = torch.cat((winners, pad))
        chunks.append(winners)

    values: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for winners in chunks:
        valid = winners >= 0
        values.append(
            torch.where(
                valid,
                local_scores[winners.clamp_min(0)],
                torch.full(
                    winners.shape,
                    -float("inf"),
                    dtype=torch.float32,
                    device=scores.device,
                ),
            )
        )
        indices.append(
            torch.where(
                valid,
                local_positions[winners.clamp_min(0)],
                torch.full_like(local_positions[winners.clamp_min(0)], -1),
            )
        )
    return torch.cat(values), torch.cat(indices), local_history_end


def _spec_tensors(
    specs: list[tuple[tuple[int, ...], torch.dtype]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return tuple(torch.empty(shape, dtype=dtype, device=device) for shape, dtype in specs)


def run(
    *,
    trace_dir: Path,
    device: torch.device,
    chunk_rows: int,
    needle_fraction: float,
    needle_radius: int,
    graph_replay: bool,
) -> dict[str, Any]:
    sources, positions, metadata = _score_sources(
        trace_dir=trace_dir,
        device=device,
        chunk_rows=chunk_rows,
    )
    scores = sources["fp8_ue8m0"]
    history_end = int(metadata["tokens"])
    needle_center = int(
        round(int(metadata["runtime_absolute_position"]) * needle_fraction)
    )

    rank_values: list[torch.Tensor] = []
    rank_indices: list[torch.Tensor] = []
    rank_lengths: list[int] = []
    for rank in range(WORLD_SIZE):
        values, indices, local_end = _rank_candidates(
            scores,
            positions,
            history_end=history_end,
            rank=rank,
        )
        rank_values.append(values)
        rank_indices.append(indices)
        rank_lengths.append(local_end)

    candidate_width = max(int(v.numel()) for v in rank_values)
    local_values = torch.full(
        (WORLD_SIZE, candidate_width),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    local_indices = torch.full(
        (WORLD_SIZE, candidate_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for rank, (values, indices) in enumerate(zip(rank_values, rank_indices)):
        local_values[rank, : values.numel()].copy_(values)
        local_indices[rank, : indices.numel()].copy_(indices)

    candidate_lengths = torch.full(
        (WORLD_SIZE,),
        candidate_width,
        dtype=torch.int32,
        device=device,
    )
    history_lengths = torch.tensor(rank_lengths, dtype=torch.int32, device=device)
    local_out_values = torch.empty(
        (WORLD_SIZE, TOPK),
        dtype=torch.float32,
        device=device,
    )
    local_out_indices = torch.empty(
        (WORLD_SIZE, TOPK),
        dtype=torch.int32,
        device=device,
    )

    def local_launch() -> None:
        _run_segmented_exact_fold(
            candidate_values=local_values,
            candidate_indices=local_indices,
            candidate_lengths=candidate_lengths,
            history_lengths=history_lengths,
            topk=TOPK,
            output_values=local_out_values,
            output_indices=local_out_indices,
        )

    local_launch()
    torch.cuda.synchronize(device)
    if graph_replay:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            local_launch()
        graph.replay()
        torch.cuda.synchronize(device)

    local_oracle_sets: list[set[int]] = []
    local_runtime_sets: list[set[int]] = []
    global_candidates: list[torch.Tensor] = []
    global_values: list[torch.Tensor] = []
    for rank in range(WORLD_SIZE):
        valid = local_indices[rank] >= 0
        oracle_rows = _selection_with_segment_floor(
            local_values[rank, valid],
            local_indices[rank, valid].to(torch.int64),
            history_end=rank_lengths[rank],
            segments=SEGMENTS,
            floor=LOCAL_FLOOR,
        )
        oracle_local = local_indices[rank, valid][oracle_rows]
        runtime_local = local_out_indices[rank]
        local_oracle_sets.append(set(int(x) for x in oracle_local.cpu().tolist()))
        local_runtime_sets.append(set(int(x) for x in runtime_local.cpu().tolist()))
        global_candidates.append(runtime_local.to(torch.int64) * WORLD_SIZE + rank)
        global_values.append(local_out_values[rank])

    if local_runtime_sets != local_oracle_sets:
        raise RuntimeError("SparkInfer local segmented selector differs from oracle")

    candidate_global_indices = torch.cat(global_candidates).view(1, -1).to(torch.int32)
    candidate_global_values = torch.cat(global_values).view(1, -1)
    global_lengths = torch.tensor(
        [candidate_global_indices.shape[1]],
        dtype=torch.int32,
        device=device,
    )
    global_out_values = torch.empty((1, TOPK), dtype=torch.float32, device=device)
    global_out_indices = torch.empty((1, TOPK), dtype=torch.int32, device=device)
    policy_scratch = _spec_tensors(
        _b12x_dcp_segmented_exact_specs(
            1,
            int(candidate_global_indices.shape[1]),
        ),
        device=device,
    )
    _select_b12x_dcp_candidates(
        candidate_indices=candidate_global_indices,
        candidate_scores=candidate_global_values,
        candidate_lengths=global_lengths,
        output_values=global_out_values,
        output_indices=global_out_indices,
        topk_tokens=TOPK,
        segmented_exact=True,
        policy_scratch=policy_scratch,
    )
    torch.cuda.synchronize(device)

    oracle_global_rows = _selection_with_segment_floor(
        candidate_global_values.flatten(),
        candidate_global_indices.flatten().to(torch.int64),
        history_end=history_end,
        segments=SEGMENTS,
        floor=GLOBAL_FLOOR,
    )
    oracle_global = candidate_global_indices.flatten()[oracle_global_rows]
    runtime_global_set = set(int(x) for x in global_out_indices.flatten().cpu().tolist())
    oracle_global_set = set(int(x) for x in oracle_global.cpu().tolist())
    if runtime_global_set != oracle_global_set:
        raise RuntimeError("vLLM global segmented selector differs from oracle")

    needle_tokens = sorted(
        x
        for x in runtime_global_set
        if abs(x - needle_center) <= needle_radius
    )
    if not needle_tokens:
        raise RuntimeError("segmented selector did not retain a needle-local token")

    return {
        "schema": "v20-indexer-segmented-exact-runtime-probe-v1",
        "claim_boundary": (
            "GPU production-selector proof on one frozen layer/query; "
            "end-to-end model validation remains required"
        ),
        "trace": str(trace_dir),
        "geometry": {
            **metadata,
            "world_size": WORLD_SIZE,
            "topk": TOPK,
            "local_slice": LOCAL_SLICE,
            "segments": SEGMENTS,
            "local_floor": LOCAL_FLOOR,
            "global_floor": GLOBAL_FLOOR,
            "candidate_width_per_rank": candidate_width,
        },
        "local_oracle_exact": True,
        "global_oracle_exact": True,
        "graph_replay": graph_replay,
        "needle_center": needle_center,
        "needle_radius": needle_radius,
        "needle_tokens": needle_tokens,
        "needle_count": len(needle_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--needle-fraction", type=float, default=0.4)
    parser.add_argument("--needle-radius", type=int, default=24)
    parser.add_argument("--graph-replay", action="store_true")
    args = parser.parse_args()
    report = run(
        trace_dir=args.trace_dir,
        device=torch.device(args.device),
        chunk_rows=args.chunk_rows,
        needle_fraction=args.needle_fraction,
        needle_radius=args.needle_radius,
        graph_replay=args.graph_replay,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
