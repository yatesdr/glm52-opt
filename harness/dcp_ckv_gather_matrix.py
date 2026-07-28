#!/usr/bin/env python3
"""Benchmark the exact full-CKV DCP all-gather shape without a model.

Each rank contributes ``local_tokens * record_bytes`` contiguous bytes and
receives rank-concatenated output, matching
``_dcp_all_gather_current_stream`` in the B12X sparse-MLA backend.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist


def _rank_max_us(started: torch.cuda.Event, ended: torch.cuda.Event) -> float:
    ended.synchronize()
    elapsed = torch.tensor(
        started.elapsed_time(ended) * 1000.0,
        dtype=torch.float64,
        device=f"cuda:{torch.cuda.current_device()}",
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-tokens",
        default="768,4096,8192,13750,30000,120000",
    )
    parser.add_argument("--record-bytes", type=int, default=368)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    if world != 4:
        raise SystemExit(f"expected four ranks, found {world}")

    for local_tokens in (
        int(value) for value in args.local_tokens.split(",") if value
    ):
        input_bytes = local_tokens * args.record_bytes
        source = torch.full(
            (input_bytes,),
            rank,
            dtype=torch.uint8,
            device=device,
        )
        gathered = torch.empty(
            (world * input_bytes,),
            dtype=torch.uint8,
            device=device,
        )

        for _ in range(args.warmup):
            dist.all_gather_into_tensor(gathered, source)
        torch.cuda.synchronize(device)

        samples: list[float] = []
        for _ in range(args.samples):
            dist.barrier(device_ids=[rank])
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            dist.all_gather_into_tensor(gathered, source)
            ended.record()
            samples.append(_rank_max_us(started, ended))

        expected = torch.arange(
            world,
            dtype=torch.uint8,
            device=device,
        ).repeat_interleave(input_bytes)
        correct = bool(torch.equal(gathered, expected))
        median_us = statistics.median(samples)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "local_tokens": local_tokens,
                        "record_bytes": args.record_bytes,
                        "input_bytes_per_rank": input_bytes,
                        "output_bytes_per_rank": input_bytes * world,
                        "median_us": median_us,
                        "algorithmic_gbps": (
                            input_bytes * (world - 1) / median_us / 1.0e3
                        ),
                        "correct": correct,
                        "samples": args.samples,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not correct:
            raise RuntimeError("all-gather output is not rank-concatenated")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
