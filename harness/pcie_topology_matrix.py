#!/usr/bin/env python3
"""Decompose four-GPU PCIe peer traffic without loading a model.

The expected topology is two local GPU pairs connected through separate
PEX switches: (0, 1) and (2, 3).  This probe distinguishes switch-uplink
contention, direction asymmetry, and ring-order effects.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Result:
    label: str
    edges: list[str]
    schedule: str
    bytes_per_copy: int
    iterations: int
    seconds: float
    aggregate_gbps: float
    per_edge_gbps: float


def _emit(result: Result) -> None:
    print(json.dumps(asdict(result), sort_keys=True), flush=True)


def _allocate(size_bytes: int, devices: list[int]):
    src = {
        device: torch.empty(size_bytes, dtype=torch.uint8, device=f"cuda:{device}")
        for device in devices
    }
    dst = {device: torch.empty_like(src[device]) for device in devices}
    return src, dst


def _sync_all(devices: list[int]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def _run(
    *,
    label: str,
    edges: list[tuple[int, int]],
    schedule: str,
    src: dict[int, torch.Tensor],
    dst: dict[int, torch.Tensor],
    streams: dict[int, torch.cuda.Stream],
    size_bytes: int,
    warmup: int,
    iterations: int,
) -> Result:
    devices = list(streams)

    def issue_once() -> None:
        for source, target in edges:
            with torch.cuda.stream(streams[target]):
                dst[target].copy_(src[source], non_blocking=True)
            if schedule == "serial":
                torch.cuda.synchronize(target)

    for _ in range(warmup):
        issue_once()
    _sync_all(devices)

    started = time.perf_counter()
    for _ in range(iterations):
        issue_once()
    _sync_all(devices)
    seconds = time.perf_counter() - started
    aggregate = size_bytes * iterations * len(edges) / seconds / 1.0e9
    return Result(
        label=label,
        edges=[f"{source}->{target}" for source, target in edges],
        schedule=schedule,
        bytes_per_copy=size_bytes,
        iterations=iterations,
        seconds=seconds,
        aggregate_gbps=aggregate,
        per_edge_gbps=aggregate / len(edges),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    args = parser.parse_args()

    devices = list(range(torch.cuda.device_count()))
    if devices != [0, 1, 2, 3]:
        raise SystemExit(f"expected exactly four CUDA devices, found {devices}")
    for source in devices:
        for target in devices:
            if source != target and not torch.cuda.can_device_access_peer(
                source, target
            ):
                raise SystemExit(f"CUDA peer access unavailable: {source}->{target}")

    size_bytes = args.size_mib << 20
    src, dst = _allocate(size_bytes, devices)
    streams = {
        device: torch.cuda.Stream(device=device) for device in devices
    }

    cases: list[tuple[str, list[tuple[int, int]], str]] = [
        ("cross_forward_02_13", [(0, 2), (1, 3)], "concurrent"),
        ("cross_forward_03_12", [(0, 3), (1, 2)], "concurrent"),
        ("cross_reverse_20_31", [(2, 0), (3, 1)], "concurrent"),
        ("cross_reverse_21_30", [(2, 1), (3, 0)], "concurrent"),
        ("cross_bidir_02", [(0, 2), (2, 0)], "concurrent"),
        ("cross_bidir_13", [(1, 3), (3, 1)], "concurrent"),
        ("cross_forward_02_13_serial", [(0, 2), (1, 3)], "serial"),
        ("cross_reverse_20_31_serial", [(2, 0), (3, 1)], "serial"),
        ("cross_bidir_02_serial", [(0, 2), (2, 0)], "serial"),
        (
            "cross_all_bidirectional",
            [(0, 2), (2, 0), (1, 3), (3, 1)],
            "concurrent",
        ),
        (
            "local_all_bidirectional",
            [(0, 1), (1, 0), (2, 3), (3, 2)],
            "concurrent",
        ),
    ]

    for permutation in itertools.permutations((1, 2, 3)):
        order = (0, *permutation)
        edges = [
            (order[index], order[(index + 1) % len(order)])
            for index in range(len(order))
        ]
        cases.append(
            ("ring_" + "".join(str(device) for device in order), edges, "concurrent")
        )

    for label, edges, schedule in cases:
        _emit(
            _run(
                label=label,
                edges=edges,
                schedule=schedule,
                src=src,
                dst=dst,
                streams=streams,
                size_bytes=size_bytes,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
