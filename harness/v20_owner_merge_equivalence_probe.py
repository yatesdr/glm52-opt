#!/usr/bin/env python3
"""No-model GPU equivalence proof for v20's DCP top-k owner merge.

The v20 TP4/DCP4 auto-policy enables ``VLLM_DCP_TOPK_OWNER_MERGE=1``.
This probe runs the packaged production owner-sharded implementation and the
established replicated DCP oracle on identical rank-local FP32 candidates.
It compares their global top-k ID sets at the full 2048-row x 2048-candidate
prefill geometry, using the real all-to-all, PyNccl all-gather, Triton remap,
and SparkInfer row-top-k kernels.  No model weights are loaded.
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist


def _make_local_candidates(
    *,
    rows: int,
    topk: int,
    local_width: int,
    rank: int,
    world_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    stride = local_width // topk
    if stride < 2:
        raise ValueError("local_width is too small to span the long-context range")
    columns = torch.arange(topk, dtype=torch.int32, device=device)
    row_offsets = (
        torch.arange(rows, dtype=torch.int32, device=device).unsqueeze(1) * 7
    ) % stride
    indices = (
        columns.unsqueeze(0) * stride
        + row_offsets
    )
    # Use a recency-skewed, globally unique FP32 score field.  This resembles
    # the frozen failure's collapse toward recent context and guarantees that
    # the minimum-history policy changes the exact set; a balanced random
    # field would already contain far more than 64 older-half winners and
    # would only prove the policy's disabled/no-op branch.
    candidate_count = world_size * rows * topk
    if candidate_count > (1 << 24):
        raise ValueError("candidate geometry exceeds the exact FP32 score space")
    linear = (
        rank * rows * topk
        + torch.arange(rows * topk, dtype=torch.int64, device=device)
    ).reshape(rows, topk)
    hashed = (linear * 1_140_671_485 + seed) & ((1 << 24) - 1)
    global_ids = indices.to(torch.int64) * world_size + rank
    scores = global_ids.to(torch.float32) + (hashed & 7).to(torch.float32) / 16.0
    return indices.contiguous(), scores.contiguous()


@torch.inference_mode()
def run_case(
    *,
    rows: int,
    topk: int,
    local_width: int,
    interleave: int,
    seed: int,
    history_floor: int,
    rank: int,
    world_size: int,
    device: torch.device,
    indexer_mod,
) -> dict[str, object]:
    base_indices, base_scores = _make_local_candidates(
        rows=rows,
        topk=topk,
        local_width=local_width,
        rank=rank,
        world_size=world_size,
        seed=seed,
        device=device,
    )

    owner_indices = base_indices.clone()
    owner_scores = base_scores.clone()
    owner_used = indexer_mod._merge_b12x_dcp_topk_by_owner(
        topk_indices=owner_indices,
        topk_scores=owner_scores,
        gathered_topk_indices=owner_indices,
        topk_tokens=topk,
        dcp_world_size=world_size,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
        history_floor=history_floor,
    )
    if not owner_used:
        raise RuntimeError("production TP4/DCP4 owner merge unexpectedly fell back")
    torch.cuda.synchronize(device)

    oracle_indices = base_indices.clone()
    oracle_scores = base_scores.clone()
    indexer_mod._merge_b12x_dcp_topk(
        topk_indices=oracle_indices,
        topk_scores=oracle_scores,
        topk_tokens=topk,
        dcp_world_size=world_size,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
        history_floor=history_floor,
    )
    torch.cuda.synchronize(device)

    exact_indices = base_indices.clone()
    exact_scores = base_scores.clone()
    indexer_mod._merge_b12x_dcp_topk(
        topk_indices=exact_indices,
        topk_scores=exact_scores,
        topk_tokens=topk,
        dcp_world_size=world_size,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
        history_floor=0,
    )
    torch.cuda.synchronize(device)

    owner_sorted = torch.sort(owner_indices, dim=1).values
    oracle_sorted = torch.sort(oracle_indices, dim=1).values
    exact_sorted = torch.sort(exact_indices, dim=1).values
    mismatch = owner_sorted.ne(oracle_sorted)
    policy_delta = oracle_sorted.ne(exact_sorted)
    local_mismatch_ids = int(mismatch.count_nonzero().item())
    local_mismatch_rows = int(mismatch.any(dim=1).count_nonzero().item())
    local_policy_changed_ids = int(policy_delta.count_nonzero().item())
    local_policy_changed_rows = int(policy_delta.any(dim=1).count_nonzero().item())
    local_unique_ok = bool(
        torch.all(owner_sorted[:, 1:] != owner_sorted[:, :-1]).item()
    )
    counts = torch.tensor(
        [
            local_mismatch_ids,
            local_mismatch_rows,
            int(not local_unique_ok),
            local_policy_changed_ids,
            local_policy_changed_rows,
        ],
        dtype=torch.int64,
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    (
        mismatch_ids,
        mismatch_rows,
        nonunique_ranks,
        policy_changed_ids,
        policy_changed_rows,
    ) = (
        int(value) for value in counts.tolist()
    )
    local_max = int(base_indices.max().item())
    global_max = (
        (local_max // interleave) * (interleave * world_size)
        + rank * interleave
        + (local_max % interleave)
    )

    first_mismatch: dict[str, int] | None = None
    if local_mismatch_ids:
        location = torch.nonzero(mismatch, as_tuple=False)[0]
        row = int(location[0].item())
        column = int(location[1].item())
        first_mismatch = {
            "rank": rank,
            "row": row,
            "column": column,
            "owner_id": int(owner_sorted[row, column].item()),
            "oracle_id": int(oracle_sorted[row, column].item()),
        }

    return {
        "rows": rows,
        "topk": topk,
        "local_width": local_width,
        "max_global_candidate": global_max,
        "interleave": interleave,
        "seed": seed,
        "history_floor": history_floor,
        "local_mismatch_ids": local_mismatch_ids,
        "local_mismatch_rows": local_mismatch_rows,
        "global_mismatch_ids": mismatch_ids,
        "global_mismatch_rows": mismatch_rows,
        "local_policy_changed_ids": local_policy_changed_ids,
        "local_policy_changed_rows": local_policy_changed_rows,
        "global_policy_changed_ids": policy_changed_ids,
        "global_policy_changed_rows": policy_changed_rows,
        "nonunique_ranks": nonunique_ranks,
        "first_local_mismatch": first_mismatch,
        "passed": (
            mismatch_ids == 0
            and nonunique_ranks == 0
            and (history_floor == 0 or policy_changed_ids > 0)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="4,2048")
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--local-width", type=int, default=120000)
    parser.add_argument("--interleave", type=int, default=1)
    parser.add_argument("--seeds", default="7,19")
    parser.add_argument("--history-floor", type=int, default=0)
    args = parser.parse_args()

    if not os.environ.get("NCCL_P2P_LEVEL"):
        raise SystemExit("set NCCL_P2P_LEVEL explicitly")

    # A CPU group carries PyNccl bootstrap and fail-closed result reductions;
    # a separate NCCL group drives the production all-to-all path.
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise SystemExit(f"expected four ranks, found {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    nccl_group = dist.new_group(backend="nccl")

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers import sparse_attn_indexer as indexer_mod
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    pynccl = PyNcclCommunicator(dist.group.WORLD, device)
    if pynccl.disabled:
        raise RuntimeError("production PyNccl communicator is unavailable")
    group = SimpleNamespace(
        world_size=world_size,
        rank_in_group=rank,
        device_group=nccl_group,
        device_communicator=SimpleNamespace(
            pynccl_comm=pynccl,
            device_group=nccl_group,
        ),
    )

    original_tp_getter = parallel_state.get_tp_group
    original_dcp_getter = parallel_state.get_dcp_group
    original_indexer_getter = parallel_state.get_indexer_dcp_group
    original_owner_getter = indexer_mod._get_owner_merge_dcp_group
    parallel_state.get_tp_group = lambda: group
    parallel_state.get_dcp_group = lambda: group
    parallel_state.get_indexer_dcp_group = lambda expected=None: group
    indexer_mod._get_owner_merge_dcp_group = lambda expected: group
    init_workspace_manager(device)

    failures = 0
    try:
        for rows in (int(value) for value in args.rows.split(",")):
            for seed in (int(value) for value in args.seeds.split(",")):
                result = run_case(
                    rows=rows,
                    topk=args.topk,
                    local_width=args.local_width,
                    interleave=args.interleave,
                    seed=seed,
                    history_floor=args.history_floor,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    indexer_mod=indexer_mod,
                )
                failures += int(not result["passed"])
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "kind": "dcp_owner_merge_equivalence",
                                "nccl_p2p_level": os.environ["NCCL_P2P_LEVEL"],
                                **result,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        reset_workspace_manager()
        parallel_state.get_tp_group = original_tp_getter
        parallel_state.get_dcp_group = original_dcp_getter
        parallel_state.get_indexer_dcp_group = original_indexer_getter
        indexer_mod._get_owner_merge_dcp_group = original_owner_getter
        pynccl.destroy()
        dist.destroy_process_group(nccl_group)
        dist.destroy_process_group()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
