#!/usr/bin/env python3
"""Prove production-size PCIe DMA in-place/out-of-place equivalence.

Run with the model stopped:

    torchrun --standalone --nproc-per-node=4 \
      v20_pcie_dma_inplace_equivalence_proof.py

The generic DMA API must remain out-of-place by default because callers may
retain a result across a later collective.  The B12X fused all-reduce +
RMSNorm caller is narrower: it immediately copies the result back over its
input.  This proof establishes that the ring itself can safely write that
specific call's reduction directly into the input allocation, avoiding a
late 36 MiB allocation at the production 3072 x 6144 BF16 geometry.
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
    # All values are exactly representable in BF16.  Rank and generation both
    # affect the payload, preventing a stale or accidentally aliased output
    # from passing.
    values = torch.arange(rows * hidden, dtype=torch.int32, device=device)
    values = (values % 251) - 125 + rank * 17 + generation * 401
    return values.to(torch.bfloat16).reshape(rows, hidden)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3072)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--wire-mode", default="i8_ring")
    parser.add_argument("--generations", type=int, default=2)
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

    max_bytes = args.rows * args.hidden * torch.bfloat16.itemsize
    collective = PCIeDmaAllReduce(
        exchange_group=dist.group.WORLD,
        device=device,
        max_bytes=max_bytes,
        fp8=args.wire_mode,
    )
    rows: list[dict[str, object]] = []
    try:
        for generation in range(args.generations):
            source = _source(
                rows=args.rows,
                hidden=args.hidden,
                rank=rank,
                generation=generation,
                device=device,
            )

            out_of_place = collective.all_reduce(source.clone())
            torch.cuda.synchronize(device)
            retained_snapshot = out_of_place.clone()

            in_place = source.clone()
            input_pointer = in_place.data_ptr()
            returned = collective.all_reduce(in_place, out=in_place)
            torch.cuda.synchronize(device)

            mismatch_count = int(
                torch.count_nonzero(
                    retained_snapshot.view(torch.uint16)
                    != in_place.view(torch.uint16)
                ).item()
            )
            retained_mismatches = int(
                torch.count_nonzero(
                    retained_snapshot.view(torch.uint16)
                    != out_of_place.view(torch.uint16)
                ).item()
            )
            pointer_preserved = (
                returned.data_ptr() == input_pointer == in_place.data_ptr()
            )

            local_hash = _digest(in_place)
            hashes: list[str | None] = [None] * world
            dist.all_gather_object(hashes, local_hash)
            rank_consistent = len(set(hashes)) == 1
            local_pass = (
                mismatch_count == 0
                and retained_mismatches == 0
                and pointer_preserved
                and rank_consistent
            )
            pass_tensor = torch.tensor(
                [int(local_pass)], dtype=torch.int32, device=device
            )
            dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)
            generation_pass = bool(pass_tensor.item())
            rows.append(
                {
                    "generation": generation,
                    "mismatch_count": mismatch_count,
                    "retained_mismatches": retained_mismatches,
                    "pointer_preserved": pointer_preserved,
                    "rank_consistent": rank_consistent,
                    "sha256": hashes,
                    "status": "PASS" if generation_pass else "FAIL",
                }
            )

        global_pass = all(row["status"] == "PASS" for row in rows)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "kind": "pcie_dma_inplace_equivalence",
                        "rows": args.rows,
                        "hidden": args.hidden,
                        "wire_mode": args.wire_mode,
                        "reported_wire_mode": collective.wire_mode,
                        "bytes_per_output": max_bytes,
                        "generations": rows,
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
