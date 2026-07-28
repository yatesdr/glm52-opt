#!/usr/bin/env python3
"""Prove that a default-output DMA all-reduce survives a later call.

Run under ``torchrun --nproc-per-node=4`` with the model stopped.  The first
result stays live while a different second collective runs.  Reusing one
persistent output buffer aliases the two results and deterministically fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch
import torch.distributed as dist


def _digest(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _source(
    *,
    rows: int,
    hidden: int,
    rank: int,
    generation: int,
    device: torch.device,
) -> torch.Tensor:
    values = torch.arange(rows * hidden, dtype=torch.int32, device=device)
    # Small exactly representable BF16 values, with different generations so
    # an overwritten first result cannot accidentally compare equal.
    values = (values % 31) - 15 + rank * 3 + generation * 41
    return values.to(torch.bfloat16).reshape(rows, hidden)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2735)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--max-rows", type=int, default=3072)
    parser.add_argument("--wire-mode", default="i8_ring")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"proof requires four ranks, got {world}")

    from sparkinfer.comm.pcie.pcie_dma import PCIeDmaAllReduce

    max_bytes = args.max_rows * args.hidden * torch.bfloat16.itemsize
    collective = PCIeDmaAllReduce(
        exchange_group=dist.group.WORLD,
        device=device,
        max_bytes=max_bytes,
        fp8=args.wire_mode,
    )
    try:
        first_input = _source(
            rows=args.rows,
            hidden=args.hidden,
            rank=rank,
            generation=0,
            device=device,
        )
        first_output = collective.all_reduce(first_input)
        torch.cuda.synchronize(device)
        first_snapshot = first_output.clone()
        first_pointer = first_output.data_ptr()

        second_input = _source(
            rows=args.rows,
            hidden=args.hidden,
            rank=rank,
            generation=1,
            device=device,
        )
        second_output = collective.all_reduce(second_input)
        torch.cuda.synchronize(device)

        retained_mismatches = int(
            torch.count_nonzero(
                first_output.view(torch.uint16)
                != first_snapshot.view(torch.uint16)
            ).item()
        )
        second_differs = int(
            torch.count_nonzero(
                second_output.view(torch.uint16)
                != first_snapshot.view(torch.uint16)
            ).item()
        )
        pointer_alias = first_pointer == second_output.data_ptr()

        first_hashes: list[str | None] = [None] * world
        second_hashes: list[str | None] = [None] * world
        dist.all_gather_object(first_hashes, _digest(first_snapshot))
        dist.all_gather_object(second_hashes, _digest(second_output))
        rank_consistent = (
            len(set(first_hashes)) == 1 and len(set(second_hashes)) == 1
        )
        local_pass = (
            retained_mismatches == 0
            and second_differs > 0
            and not pointer_alias
            and rank_consistent
        )
        pass_tensor = torch.tensor(
            [int(local_pass)], dtype=torch.int32, device=device
        )
        dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)
        global_pass = bool(pass_tensor.item())

        if rank == 0:
            print(
                json.dumps(
                    {
                        "kind": "pcie_dma_output_lifetime",
                        "rows": args.rows,
                        "hidden": args.hidden,
                        "max_rows": args.max_rows,
                        "wire_mode": args.wire_mode,
                        "reported_wire_mode": collective.wire_mode,
                        "bytes_per_output": (
                            args.rows
                            * args.hidden
                            * torch.bfloat16.itemsize
                        ),
                        "pointer_alias": pointer_alias,
                        "retained_mismatches": retained_mismatches,
                        "second_differs_from_first": second_differs,
                        "rank_consistent": rank_consistent,
                        "first_sha256": first_hashes,
                        "second_sha256": second_hashes,
                        "status": "PASS" if global_pass else "FAIL",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dist.barrier(device_ids=[local_rank])
        return 0 if global_pass else 1
    finally:
        collective.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
