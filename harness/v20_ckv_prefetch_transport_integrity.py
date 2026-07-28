#!/usr/bin/env python3
"""Exercise the production CKV prefetch transport without loading a model.

Run this script twice, in fresh processes, because NCCL reads
``NCCL_P2P_LEVEL`` when communicators are created:

    NCCL_P2P_LEVEL=SYS torchrun --standalone --nproc-per-node=4 \
      harness/v20_ckv_prefetch_transport_integrity.py
    NCCL_P2P_LEVEL=PXB torchrun --standalone --nproc-per-node=4 \
      harness/v20_ckv_prefetch_transport_integrity.py

The probe mirrors the correctness-sensitive parts of
``B12xMLASparseImpl._dcp_gather_ckv``:

* four ranks contribute contiguous 368-byte CKV records;
* a second PyNccl communicator runs the next-layer all-gather on a side
  stream;
* a CUDA event hands the ping-pong buffer back to the default stream;
* the current buffer is consumed while the next buffer is in flight; and
* every gathered byte is compared with the rank/layer/step-specific source.

This is a byte-copy test, not a floating-point tolerance test. Any mismatch is
a transport or ordering defect and exits nonzero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist


def _offset(rank: int, layer: int, step: int) -> int:
    return (rank * 53 + layer * 17 + step * 29) & 0xFF


def _fill(source: torch.Tensor, base: torch.Tensor, offset: int) -> None:
    source.copy_(base)
    source.add_(offset)


def _enqueue_check(
    gathered: torch.Tensor,
    base: torch.Tensor,
    expected: torch.Tensor,
    mismatches: torch.Tensor,
    *,
    world_size: int,
    layer: int,
    step: int,
) -> None:
    for source_rank in range(world_size):
        expected.copy_(base)
        expected.add_(_offset(source_rank, layer, step))
        mismatches.add_(
            torch.count_nonzero(gathered[source_rank].ne(expected))
        )


def _run_case(
    *,
    sync_comm,
    prefetch_comm,
    device: torch.device,
    rank: int,
    world_size: int,
    global_tokens: int,
    record_bytes: int,
    block_size: int,
    layers: int,
    steps: int,
) -> dict[str, object]:
    local_records = math.ceil(
        math.ceil(global_tokens / world_size) / block_size
    ) * block_size
    input_bytes = local_records * record_bytes

    # Avoid an input-sized int32 resident: arange is transient and the retained
    # base is the exact uint8 payload used by every step.
    base = torch.arange(input_bytes, dtype=torch.int32, device=device)
    base.remainder_(251)
    base_u8 = base.to(torch.uint8)
    del base

    sources = [
        torch.empty(input_bytes, dtype=torch.uint8, device=device)
        for _ in range(2)
    ]
    gathered = [
        torch.empty(
            (world_size, input_bytes),
            dtype=torch.uint8,
            device=device,
        )
        for _ in range(2)
    ]
    expected = torch.empty_like(base_u8)
    mismatches = torch.zeros((), dtype=torch.int64, device=device)
    side_stream = torch.cuda.Stream(device=device)
    default_stream = torch.cuda.current_stream(device)

    torch.cuda.synchronize(device)
    started = time.monotonic()
    for step in range(steps):
        current_buf = 0
        _fill(sources[current_buf], base_u8, _offset(rank, 0, step))
        sync_comm.all_gather(
            gathered[current_buf].view(-1),
            sources[current_buf],
        )

        ready: torch.cuda.Event | None = None
        for layer in range(layers):
            if ready is not None:
                ready.wait()

            if layer + 1 < layers:
                next_buf = 1 - current_buf
                _fill(
                    sources[next_buf],
                    base_u8,
                    _offset(rank, layer + 1, step),
                )
                side_stream.wait_stream(default_stream)
                with torch.cuda.stream(side_stream):
                    prefetch_comm.all_gather(
                        gathered[next_buf].view(-1),
                        sources[next_buf],
                    )
                ready = torch.cuda.Event(blocking=False)
                ready.record(side_stream)
            else:
                next_buf = current_buf
                ready = None

            # This read remains on the default stream while the next layer's
            # gather runs on the side stream, reproducing the intended overlap.
            _enqueue_check(
                gathered[current_buf],
                base_u8,
                expected,
                mismatches,
                world_size=world_size,
                layer=layer,
                step=step,
            )
            current_buf = next_buf

        # Match the metadata-builder boundary: no event or buffer ownership is
        # carried from the last layer into layer zero of the next scheduler step.
        default_stream.wait_stream(side_stream)

    torch.cuda.synchronize(device)
    elapsed_s = time.monotonic() - started
    local_mismatches = int(mismatches.item())
    mismatch_tensor = torch.tensor(local_mismatches, dtype=torch.int64)
    dist.all_reduce(mismatch_tensor, op=dist.ReduceOp.SUM)
    global_mismatches = int(mismatch_tensor.item())

    return {
        "global_tokens": global_tokens,
        "local_records": local_records,
        "record_bytes": record_bytes,
        "input_bytes_per_rank": input_bytes,
        "gathered_bytes_per_rank": input_bytes * world_size,
        "layers": layers,
        "steps": steps,
        "local_mismatches": local_mismatches,
        "global_mismatches": global_mismatches,
        "elapsed_s": elapsed_s,
        "passed": global_mismatches == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-tokens", default="100000,350000,475000")
    parser.add_argument("--record-bytes", type=int, default=368)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    p2p_level = os.environ.get("NCCL_P2P_LEVEL")
    if p2p_level not in {"SYS", "PXB"}:
        raise SystemExit("set NCCL_P2P_LEVEL explicitly to SYS or PXB")

    # PyNcclCommunicator uses this CPU group only to exchange NCCL unique IDs.
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise SystemExit(f"expected four ranks, found {world_size}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    sync_comm = PyNcclCommunicator(dist.group.WORLD, device)
    prefetch_comm = PyNcclCommunicator(dist.group.WORLD, device)
    if sync_comm.disabled or prefetch_comm.disabled:
        raise RuntimeError("the exact PyNccl communicators are unavailable")

    failures = 0
    try:
        for global_tokens in (
            int(value) for value in args.global_tokens.split(",") if value
        ):
            result = _run_case(
                sync_comm=sync_comm,
                prefetch_comm=prefetch_comm,
                device=device,
                rank=rank,
                world_size=world_size,
                global_tokens=global_tokens,
                record_bytes=args.record_bytes,
                block_size=args.block_size,
                layers=args.layers,
                steps=args.steps,
            )
            failures += int(not result["passed"])
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "kind": "ckv_prefetch_transport",
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

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
