#!/usr/bin/env python3
"""GPU/graph proof for the v20 DCP older-half floor implementation.

This loads the frozen failing GLM activation, reconstructs the exact TP4/DCP4
candidate table, executes the production Triton + SparkInfer merge sequence,
and compares it to an independent CPU reference.  It also proves that the
minimum-context gate is an exact no-op and that CUDA graph replay is stable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from v20_indexer_age_aware_policy_proof import _score_sources
from v20_indexer_dcp_candidate_policy_proof import (
    _candidate_rows,
    _selection_with_older_half_floor,
)
from v20_indexer_real_activation_gpu_quant_proof import TOPK


def _run_production_merge(
    *,
    candidate_ids: torch.Tensor,
    candidate_scores: torch.Tensor,
    floor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sparkinfer.attention.nsa_indexer.tiled_topk import run_row_topk
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _B12X_DCP_HISTORY_MIN_CONTEXT,
        _b12x_dcp_boost_history_kernel,
        _b12x_dcp_gather_selected_candidates_kernel,
        _b12x_dcp_history_priority_kernel,
    )
    from vllm.triton_utils import triton

    rows, candidate_width = candidate_scores.shape
    merge_scores = torch.empty_like(candidate_scores)
    history_values = torch.empty(
        (rows, floor),
        dtype=torch.float32,
        device=candidate_scores.device,
    )
    history_positions = torch.empty(
        (rows, floor),
        dtype=torch.int64,
        device=candidate_scores.device,
    )
    history_cutoffs = torch.empty(
        (rows,),
        dtype=torch.int32,
        device=candidate_scores.device,
    )
    lengths = torch.full(
        (rows,),
        candidate_width,
        dtype=torch.int32,
        device=candidate_scores.device,
    )
    output_values = torch.empty(
        (rows, TOPK),
        dtype=torch.float32,
        device=candidate_scores.device,
    )
    output_indices = torch.empty(
        (rows, TOPK),
        dtype=torch.int32,
        device=candidate_scores.device,
    )

    def launch() -> None:
        _b12x_dcp_history_priority_kernel[(rows,)](
            candidate_ids,
            candidate_scores,
            merge_scores,
            history_cutoffs,
            CANDIDATE_WIDTH=candidate_width,
            MIN_CONTEXT=_B12X_DCP_HISTORY_MIN_CONTEXT,
            BLOCK_K=triton.next_power_of_2(candidate_width),
            num_warps=8,
        )
        torch.topk(
            merge_scores,
            floor,
            dim=1,
            largest=True,
            sorted=False,
            out=(history_values, history_positions),
        )
        merge_scores.copy_(candidate_scores)
        _b12x_dcp_boost_history_kernel[(rows,)](
            merge_scores,
            candidate_scores,
            history_positions,
            history_cutoffs,
            CANDIDATE_WIDTH=candidate_width,
            HISTORY_FLOOR=floor,
            BLOCK_K=triton.next_power_of_2(floor),
            num_warps=1,
        )
        run_row_topk(
            row_logits=merge_scores,
            lengths=lengths,
            topk=TOPK,
            output_values=output_values,
            output_indices=output_indices,
        )
        _b12x_dcp_gather_selected_candidates_kernel[(rows,)](
            candidate_ids,
            candidate_scores,
            output_indices,
            output_indices,
            output_values,
            CANDIDATE_WIDTH=candidate_width,
            TOPK_TOKENS=TOPK,
            BLOCK_K=triton.next_power_of_2(TOPK),
            num_warps=8,
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    return output_indices.cpu(), output_values.cpu()


def _check_case(
    *,
    name: str,
    candidate_ids: torch.Tensor,
    candidate_scores: torch.Tensor,
    expected_rows: torch.Tensor,
    floor: int,
) -> dict[str, Any]:
    output_ids, output_values = _run_production_merge(
        candidate_ids=candidate_ids.cuda().to(torch.int32).view(1, -1),
        candidate_scores=candidate_scores.cuda().to(torch.float32).view(1, -1),
        floor=floor,
    )
    actual_ids = output_ids[0].to(torch.int64)
    expected_ids = candidate_ids[expected_rows].to(torch.int64)
    set_equal = bool(
        torch.equal(
            torch.sort(actual_ids).values,
            torch.sort(expected_ids).values,
        )
    )
    score_by_id = {
        int(idx): float(score)
        for idx, score in zip(candidate_ids.tolist(), candidate_scores.tolist())
    }
    expected_values = torch.tensor(
        [score_by_id[int(idx)] for idx in actual_ids.tolist()],
        dtype=torch.float32,
    )
    scores_exact = bool(torch.equal(output_values[0], expected_values))
    if not set_equal or not scores_exact:
        raise RuntimeError(
            f"{name} mismatch: set_equal={set_equal} scores_exact={scores_exact}"
        )
    return {
        "name": name,
        "candidate_count": int(candidate_ids.numel()),
        "set_equal": set_equal,
        "scores_bit_exact": scores_exact,
        "selected_min": int(actual_ids.min().item()),
        "selected_max": int(actual_ids.max().item()),
    }


def run(
    *,
    trace_dir: Path,
    device: torch.device,
    chunk_rows: int,
    floor: int,
) -> dict[str, Any]:
    scores, positions, metadata = _score_sources(
        trace_dir=trace_dir,
        device=device,
        chunk_rows=chunk_rows,
    )
    fp8_scores = scores["fp8_ue8m0"]
    candidate_rows, _ = _candidate_rows(
        fp8_scores,
        positions,
        world_size=4,
        interleave=1,
    )
    candidate_ids = positions[candidate_rows]
    candidate_scores = fp8_scores[candidate_rows]
    expected_local = _selection_with_older_half_floor(
        candidate_scores,
        candidate_ids,
        history_end=metadata["tokens"],
        floor=floor,
    )
    deep = _check_case(
        name="frozen_350k_fp8",
        candidate_ids=candidate_ids,
        candidate_scores=candidate_scores,
        expected_rows=expected_local,
        floor=floor,
    )

    generator = torch.Generator().manual_seed(20260726)
    short_ids = torch.arange(8192, dtype=torch.int64)
    short_scores = torch.randn(8192, generator=generator, dtype=torch.float32)
    short_expected = torch.topk(
        short_scores,
        TOPK,
        largest=True,
        sorted=False,
    ).indices
    short = _check_case(
        name="short_context_noop",
        candidate_ids=short_ids,
        candidate_scores=short_scores,
        expected_rows=short_expected,
        floor=floor,
    )
    return {
        "schema": "v20-dcp-history-floor-gpu-proof-v1",
        "claim_boundary": (
            "operator and CUDA-graph equivalence on frozen activation; "
            "end-to-end model acceptance remains required"
        ),
        "floor": floor,
        "cases": [deep, short],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--floor", type=int, default=64)
    args = parser.parse_args()
    result = run(
        trace_dir=args.trace_dir,
        device=torch.device(args.device),
        chunk_rows=args.chunk_rows,
        floor=args.floor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
