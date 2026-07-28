#!/usr/bin/env python3
"""No-model GPU proof for v20 CKV prefetch record equivalence.

This compares the two production paths at the 368-byte NVFP4+FP8-RoPE
record boundary:

* synchronous: each DCP owner writes its current-token subset to the local
  cache, then the four rank-local caches are gathered;
* prefetched: each future-layer cache is gathered before its current-token
  write, then every consumer reconstructs the missing current records in the
  rank-ordered gathered buffer.

The probe uses the packaged production writer and two PyNccl communicators.
The prefetch gather runs on a side stream and is handed to the main stream by
a CUDA event, matching ``B12xMLASparseImpl._dcp_gather_ckv``.  No model
weights are loaded.

Identical full-latent inputs on every rank are the implementation's stated
contract and must produce byte-identical buffers.  ``--rank-ulp`` is a
diagnostic mode that perturbs rank-local BF16 inputs by representable steps;
it measures the otherwise-implicit dependence on cross-rank input identity
and is not part of the pass/fail gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


PAGE_SIZE = 64
RECORD_BYTES = 368
KV_DIM = 512
ROPE_DIM = 64


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def owner_local(
    positions: torch.Tensor,
    *,
    world_size: int,
    interleave: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    owner = torch.div(positions, interleave, rounding_mode="floor").remainder(
        world_size
    )
    local = (
        torch.div(
            positions,
            world_size * interleave,
            rounding_mode="floor",
        )
        * interleave
        + positions.remainder(interleave)
    )
    return owner, local


def rank_length(
    seq_len: int,
    rank: int,
    *,
    world_size: int,
    interleave: int,
) -> int:
    cycle = world_size * interleave
    full_cycles, remainder = divmod(seq_len, cycle)
    return (
        full_cycles * interleave
        + min(interleave, max(0, remainder - rank * interleave))
    )


@dataclass(frozen=True)
class Case:
    final_seq_len: int
    chunk_len: int
    input_scale: float


def parse_cases(raw: str) -> list[Case]:
    cases = []
    for item in raw.split(","):
        fields = item.split(":")
        if len(fields) != 3:
            raise ValueError(
                "cases must be final_seq_len:chunk_len:input_scale"
            )
        cases.append(Case(int(fields[0]), int(fields[1]), float(fields[2])))
    return cases


def make_inputs(
    *,
    case: Case,
    seed: int,
    rank: int,
    rank_ulp: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    kv_c = (
        torch.randn(case.chunk_len, KV_DIM, generator=generator)
        * case.input_scale
    ).to(dtype=torch.bfloat16)
    k_pe = (
        torch.randn(case.chunk_len, ROPE_DIM, generator=generator)
        * case.input_scale
    ).to(dtype=torch.bfloat16)
    if rank_ulp:
        # nextafter is unavailable for BF16 on some torch builds. Increment
        # the unsigned BF16 payload directly, restricted to finite positive
        # source elements so each step is one representable value.
        for tensor in (kv_c, k_pe):
            bits = tensor.view(torch.int16)
            mask = tensor > 0
            bits[mask] += int(rank * rank_ulp)
    return kv_c.to(device), k_pe.to(device)


def fill_history(
    local: torch.Tensor,
    *,
    history_len: int,
    rank: int,
    sentinel: int,
) -> None:
    local.fill_(sentinel)
    if history_len:
        # A rank- and byte-dependent valid prefix proves rank ordering without
        # allocating a context-sized int32 arange.
        local[:history_len].fill_((17 + rank * 53) & 0xFF)
        local[:history_len, 0].copy_(
            torch.arange(history_len, device=local.device, dtype=torch.int64)
            .remainder(251)
            .to(torch.uint8)
        )


@torch.inference_mode()
def run_case(
    *,
    case: Case,
    seed: int,
    rank: int,
    world_size: int,
    interleave: int,
    rank_ulp: int,
    sync_comm,
    prefetch_comm,
    writer,
    device: torch.device,
) -> dict[str, object]:
    if case.chunk_len <= 0 or case.chunk_len > case.final_seq_len:
        raise ValueError(f"invalid case {case}")
    history_global = case.final_seq_len - case.chunk_len
    final_rank_lengths = [
        rank_length(
            case.final_seq_len,
            source_rank,
            world_size=world_size,
            interleave=interleave,
        )
        for source_rank in range(world_size)
    ]
    history_rank_lengths = [
        rank_length(
            history_global,
            source_rank,
            world_size=world_size,
            interleave=interleave,
        )
        for source_rank in range(world_size)
    ]
    padded = ceil_div(max(final_rank_lengths), PAGE_SIZE) * PAGE_SIZE
    sentinel = 0xCD

    local_prefetch = torch.empty(
        padded, RECORD_BYTES, dtype=torch.uint8, device=device
    )
    fill_history(
        local_prefetch,
        history_len=history_rank_lengths[rank],
        rank=rank,
        sentinel=sentinel,
    )
    sync_local = local_prefetch.clone()
    prefetch_gathered = torch.empty(
        world_size, padded, RECORD_BYTES, dtype=torch.uint8, device=device
    )
    sync_gathered = torch.empty_like(prefetch_gathered)

    kv_c, k_pe = make_inputs(
        case=case,
        seed=seed,
        rank=rank,
        rank_ulp=rank_ulp,
        device=device,
    )
    global_positions = torch.arange(
        history_global,
        case.final_seq_len,
        dtype=torch.int64,
        device=device,
    )
    owners, local_positions = owner_local(
        global_positions,
        world_size=world_size,
        interleave=interleave,
    )

    default_stream = torch.cuda.current_stream(device)
    side_stream = torch.cuda.Stream(device=device)
    side_stream.wait_stream(default_stream)
    with torch.cuda.stream(side_stream):
        prefetch_comm.all_gather(
            prefetch_gathered.view(-1),
            local_prefetch.view(-1),
        )
    ready = torch.cuda.Event(blocking=False)
    ready.record(side_stream)

    owned = owners == rank
    writer(
        kv_c[owned].contiguous(),
        k_pe[owned].contiguous(),
        sync_local.view(-1, PAGE_SIZE, RECORD_BYTES),
        local_positions[owned].contiguous(),
    )

    ready.wait()
    append_slots = owners * padded + local_positions
    writer(
        kv_c,
        k_pe,
        prefetch_gathered.view(-1, PAGE_SIZE, RECORD_BYTES),
        append_slots.contiguous(),
    )
    sync_comm.all_gather(sync_gathered.view(-1), sync_local.view(-1))
    torch.cuda.synchronize(device)

    mismatch = prefetch_gathered.ne(sync_gathered)
    local_mismatch_bytes = int(mismatch.count_nonzero().item())
    mismatch_records = int(mismatch.any(dim=-1).count_nonzero().item())
    local_counts = torch.tensor(
        [local_mismatch_bytes, mismatch_records],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
    global_mismatch_bytes, global_mismatch_records = (
        int(value) for value in local_counts.cpu().tolist()
    )

    return {
        "final_seq_len": case.final_seq_len,
        "chunk_len": case.chunk_len,
        "history_len": history_global,
        "input_scale": case.input_scale,
        "interleave": interleave,
        "rank_ulp": rank_ulp,
        "padded_rank_tokens": padded,
        "gathered_mib_per_rank": round(
            world_size * padded * RECORD_BYTES / (1024 * 1024),
            3,
        ),
        "local_mismatch_bytes": local_mismatch_bytes,
        "local_mismatch_records": mismatch_records,
        "global_mismatch_bytes": global_mismatch_bytes,
        "global_mismatch_records": global_mismatch_records,
        "passed": global_mismatch_bytes == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="65537:2048:0.25,343727:1711:0.25,343727:2048:1.0",
    )
    parser.add_argument("--interleave", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--rank-ulp", type=int, default=0)
    args = parser.parse_args()

    p2p_level = os.environ.get("NCCL_P2P_LEVEL")
    if not p2p_level:
        raise SystemExit("set NCCL_P2P_LEVEL explicitly")

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise SystemExit(f"expected four ranks, found {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from sparkinfer.attention._shared.mla.kv_cache import (
        concat_and_cache_nvfp4_mla_fp8_rope,
    )
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    sync_comm = PyNcclCommunicator(dist.group.WORLD, device)
    prefetch_comm = PyNcclCommunicator(dist.group.WORLD, device)
    if sync_comm.disabled or prefetch_comm.disabled:
        raise RuntimeError("production PyNccl communicators are unavailable")

    # Compile the exact writer before the measured communicator overlap.
    warm_kv = torch.zeros((1, KV_DIM), dtype=torch.bfloat16, device=device)
    warm_pe = torch.zeros((1, ROPE_DIM), dtype=torch.bfloat16, device=device)
    warm_cache = torch.empty(
        (1, PAGE_SIZE, RECORD_BYTES), dtype=torch.uint8, device=device
    )
    warm_slot = torch.zeros((1,), dtype=torch.int64, device=device)
    concat_and_cache_nvfp4_mla_fp8_rope(
        warm_kv, warm_pe, warm_cache, warm_slot
    )
    torch.cuda.synchronize(device)
    dist.barrier()

    failures = 0
    try:
        for index, case in enumerate(parse_cases(args.cases)):
            result = run_case(
                case=case,
                seed=args.seed + index,
                rank=rank,
                world_size=world_size,
                interleave=args.interleave,
                rank_ulp=args.rank_ulp,
                sync_comm=sync_comm,
                prefetch_comm=prefetch_comm,
                writer=concat_and_cache_nvfp4_mla_fp8_rope,
                device=device,
            )
            failures += int(not result["passed"])
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "kind": "ckv_prefetch_record_equivalence",
                            "nccl_p2p_level": p2p_level,
                            **result,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        prefetch_comm.destroy()
        sync_comm.destroy()
        dist.destroy_process_group()

    # Perturbed-rank mode is diagnostic by construction; only the stated
    # identical-input contract is a fail-closed equivalence gate.
    if args.rank_ulp:
        return 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
