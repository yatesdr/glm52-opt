#!/usr/bin/env python3
"""Small fail-closed NCCL all-reduce probe for choosing a P2P level.

Run under ``torchrun --standalone --nproc-per-node=4`` with one process per
GPU. The payloads cover latency-sensitive decode collectives and the larger
prefill regime. Rank 0 emits one JSON object after every rank has completed
the same number of collectives.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"expected world_size=4, got {world}")

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    rows: list[dict[str, float | int]] = []

    for payload_bytes, iterations in (
        (1 << 20, 100),
        (8 << 20, 60),
        (64 << 20, 30),
    ):
        tensor = torch.ones(payload_bytes // 4, dtype=torch.float32, device=device)
        for _ in range(10):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()
        for _ in range(iterations):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device=device)
        gathered = [torch.empty_like(elapsed_tensor) for _ in range(world)]
        dist.all_gather(gathered, elapsed_tensor)
        if rank == 0:
            rank_times = [float(item.item()) for item in gathered]
            worst = max(rank_times)
            seconds_per_call = worst / iterations
            algorithm_gbps = payload_bytes / seconds_per_call / 1e9
            bus_gbps = algorithm_gbps * (2 * (world - 1) / world)
            rows.append(
                {
                    "payload_bytes": payload_bytes,
                    "iterations": iterations,
                    "seconds_per_call": seconds_per_call,
                    "algorithm_gbps": algorithm_gbps,
                    "bus_gbps": bus_gbps,
                    "rank_time_mean_s": statistics.mean(rank_times),
                    "rank_time_max_s": worst,
                }
            )
        del tensor

    if rank == 0:
        print(
            json.dumps(
                {
                    "nccl_p2p_level": os.environ.get("NCCL_P2P_LEVEL"),
                    "world_size": world,
                    "torch": torch.__version__,
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
