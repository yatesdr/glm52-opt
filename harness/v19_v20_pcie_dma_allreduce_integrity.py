#!/usr/bin/env python3
"""Four-rank PCIe-DMA hidden-state all-reduce integrity proof.

Run under ``torchrun --nproc-per-node=4``.  Integer-valued BF16 inputs make
the exact all-reduce result independent of reduction order, so any mismatch
is transport, routing, synchronization, or persistent-output corruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch
import torch.distributed as dist


def _load_api():
    try:
        import sparkinfer
        from sparkinfer.comm.pcie import DmaAllReduce

        return (
            "sparkinfer",
            str(getattr(sparkinfer, "__version__", "(unknown)")),
            DmaAllReduce,
        )
    except ImportError:
        import b12x
        from b12x.distributed.pcie_dma import PCIeDmaAllReduce

        return (
            "b12x",
            str(getattr(b12x, "__version__", "(unknown)")),
            PCIeDmaAllReduce,
        )


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1711)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--wire-mode", default="")
    args = parser.parse_args()
    if min(args.rows, args.hidden, args.layers, args.steps) <= 0:
        raise ValueError("all dimensions/counts must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"this proof requires four ranks, got {world}")
    package, package_version, DmaAllReduce = _load_api()

    elements = args.rows * args.hidden
    bytes_per_tensor = elements * torch.bfloat16.itemsize
    if elements % (world * 8):
        raise ValueError("element count must be divisible by world_size * 8")
    base = (
        (torch.arange(elements, dtype=torch.int32, device=device) % 31) - 15
    ).to(torch.bfloat16).reshape(args.rows, args.hidden)
    rank_offset = rank * 2
    rank_offset_sum = sum(peer * 2 for peer in range(world))
    collective = DmaAllReduce(
        exchange_group=dist.group.WORLD,
        device=device,
        max_bytes=bytes_per_tensor,
        fp8=args.wire_mode,
    )
    dist.barrier(device_ids=[local_rank])
    local_mismatches = 0
    last_output: torch.Tensor | None = None
    started = time.monotonic()
    for step in range(args.steps):
        for layer in range(args.layers):
            iteration_shift = (step * args.layers + layer) % 7
            source = base + (rank_offset + iteration_shift)
            expected = base * world + (
                rank_offset_sum + world * iteration_shift
            )
            output = collective.all_reduce(source)
            torch.cuda.synchronize(device)
            mismatches = int(
                torch.count_nonzero(
                    output.contiguous().view(torch.uint16)
                    != expected.contiguous().view(torch.uint16)
                ).item()
            )
            local_mismatches += mismatches
            if mismatches:
                raise AssertionError(
                    f"rank={rank} step={step} layer={layer} mismatches={mismatches}"
                )
            last_output = output
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    mismatch_tensor = torch.tensor(
        [local_mismatches], dtype=torch.int64, device=device
    )
    dist.all_reduce(mismatch_tensor, op=dist.ReduceOp.SUM)
    assert last_output is not None
    hashes: list[str | None] = [None] * world
    dist.all_gather_object(hashes, _digest(last_output))
    if len(set(hashes)) != 1:
        raise AssertionError(f"rank outputs differ: {hashes}")
    result = {
        "kind": "pcie_dma_allreduce_integrity",
        "package": package,
        "package_version": package_version,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "world_size": world,
        "rows": args.rows,
        "hidden": args.hidden,
        "bytes_per_tensor": bytes_per_tensor,
        "layers": args.layers,
        "steps": args.steps,
        "operations": args.layers * args.steps,
        "requested_wire_mode": args.wire_mode,
        "reported_wire_mode": collective.wire_mode,
        "global_mismatches": int(mismatch_tensor.item()),
        "rank_output_sha256": hashes,
        "elapsed_seconds": elapsed,
        "status": "PASS",
    }
    if rank == 0:
        print(json.dumps(result, sort_keys=True), flush=True)
    collective.close()
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
