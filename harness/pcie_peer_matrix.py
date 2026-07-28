#!/usr/bin/env python3
"""Measure CUDA peer-copy bandwidth without a model process.

Run the same image and command on both comparison hosts after all serving
containers have stopped.  The output separates single-edge peer bandwidth
from concurrent within-switch, cross-switch, and ring traffic.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Result:
    label: str
    edges: list[str]
    bytes_per_copy: int
    iterations: int
    seconds: float
    aggregate_gbps: float
    per_edge_gbps: float


def _emit(result: Result) -> None:
    print(json.dumps(asdict(result), sort_keys=True), flush=True)


def _allocate(size_bytes: int, devices: list[int]):
    count = size_bytes // torch.empty((), dtype=torch.uint8).element_size()
    src = {
        device: torch.empty(count, dtype=torch.uint8, device=f"cuda:{device}")
        for device in devices
    }
    dst = {
        device: torch.empty_like(src[device])
        for device in devices
    }
    return src, dst


def _run_edges(
    *,
    label: str,
    edges: list[tuple[int, int]],
    src: dict[int, torch.Tensor],
    dst: dict[int, torch.Tensor],
    streams: dict[int, torch.cuda.Stream],
    size_bytes: int,
    warmup: int,
    iterations: int,
) -> Result:
    for _ in range(warmup):
        for source, target in edges:
            with torch.cuda.stream(streams[target]):
                dst[target].copy_(src[source], non_blocking=True)
    for device in streams:
        torch.cuda.synchronize(device)

    started = time.perf_counter()
    for _ in range(iterations):
        for source, target in edges:
            with torch.cuda.stream(streams[target]):
                dst[target].copy_(src[source], non_blocking=True)
    for device in streams:
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    aggregate = size_bytes * iterations * len(edges) / seconds / 1.0e9
    return Result(
        label=label,
        edges=[f"{source}->{target}" for source, target in edges],
        bytes_per_copy=size_bytes,
        iterations=iterations,
        seconds=seconds,
        aggregate_gbps=aggregate,
        per_edge_gbps=aggregate / len(edges),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    count = torch.cuda.device_count()
    if count != 4:
        raise SystemExit(f"expected exactly 4 CUDA devices, found {count}")
    for source in range(count):
        for target in range(count):
            if source != target and not torch.cuda.can_device_access_peer(
                source, target
            ):
                raise SystemExit(f"CUDA peer access unavailable: {source}->{target}")

    devices = list(range(count))
    size_bytes = args.size_mib << 20
    src, dst = _allocate(size_bytes, devices)
    streams = {
        device: torch.cuda.Stream(device=device) for device in devices
    }

    cases: list[tuple[str, list[tuple[int, int]]]] = []
    for source in devices:
        for target in devices:
            if source != target:
                cases.append((f"single_{source}_{target}", [(source, target)]))
    cases.extend(
        [
            ("within_pairs", [(0, 1), (1, 0), (2, 3), (3, 2)]),
            ("cross_pairs", [(0, 2), (2, 0), (1, 3), (3, 1)]),
            ("ring_0123", [(0, 1), (1, 2), (2, 3), (3, 0)]),
            ("ring_0132", [(0, 1), (1, 3), (3, 2), (2, 0)]),
        ]
    )

    for label, edges in cases:
        _emit(
            _run_edges(
                label=label,
                edges=edges,
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
