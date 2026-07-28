#!/usr/bin/env python3
"""Compare the production INT8 DMA ring with NCCL without loading a model.

Run inside the exact serving image after every model process has stopped:

    torchrun --standalone --nproc-per-node=4 \
      /workspace/harness/pcie_collective_matrix.py

The benchmark uses one operation per timing sample so NCCL's in-place result
can be reset outside the measured interval. Timings are MAX-reduced across
ranks. Correctness checks both error against NCCL and exact equality among
all DMA-ring ranks.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class CollectiveResult:
    rows: int
    hidden: int
    bytes: int
    samples: int
    nccl_median_us: float
    dma_explicit_median_us: float
    dma_default_median_us: float
    dma_explicit_over_nccl: float
    dma_default_over_nccl: float
    dma_default_over_explicit: float
    explicit_rank_max_abs_divergence: float
    explicit_vs_nccl_max_abs_error: float
    explicit_vs_nccl_mean_abs_error: float
    explicit_vs_nccl_max_relative_error: float
    explicit_finite: bool
    default_rank_max_abs_divergence: float
    default_vs_nccl_max_abs_error: float
    default_vs_nccl_mean_abs_error: float
    default_vs_nccl_max_relative_error: float
    default_finite: bool
    graph_rank_max_abs_divergence: float
    graph_vs_nccl_max_abs_error: float
    graph_vs_nccl_mean_abs_error: float
    graph_vs_nccl_max_relative_error: float
    graph_finite: bool


def _rank_max_elapsed_us(
    started: torch.cuda.Event,
    ended: torch.cuda.Event,
    device: torch.device,
) -> float:
    ended.synchronize()
    elapsed = torch.tensor(
        started.elapsed_time(ended) * 1000.0,
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def _time_nccl(
    source: torch.Tensor,
    work: torch.Tensor,
    *,
    warmup: int,
    samples: int,
) -> list[float]:
    device = source.device
    for _ in range(warmup):
        work.copy_(source)
        dist.all_reduce(work)
    torch.cuda.synchronize(device)

    timings: list[float] = []
    for _ in range(samples):
        work.copy_(source)
        dist.barrier(device_ids=[device.index])
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        dist.all_reduce(work)
        ended.record()
        timings.append(_rank_max_elapsed_us(started, ended, device))
    return timings


def _time_dma(
    dma,
    source: torch.Tensor,
    output: torch.Tensor | None,
    *,
    warmup: int,
    samples: int,
) -> list[float]:
    device = source.device
    for _ in range(warmup):
        if output is None:
            dma.all_reduce(source)
        else:
            dma.all_reduce(source, out=output)
    torch.cuda.synchronize(device)

    timings: list[float] = []
    for _ in range(samples):
        dist.barrier(device_ids=[device.index])
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        if output is None:
            dma.all_reduce(source)
        else:
            dma.all_reduce(source, out=output)
        ended.record()
        timings.append(_rank_max_elapsed_us(started, ended, device))
    return timings


def _compare_output(
    source: torch.Tensor,
    output: torch.Tensor,
) -> tuple[float, float, float, float, bool]:
    expected = source.clone()
    dist.all_reduce(expected)
    torch.cuda.synchronize(source.device)

    rank_min = output.clone()
    rank_max = output.clone()
    dist.all_reduce(rank_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
    rank_divergence = float(
        (rank_max.float() - rank_min.float()).abs().max().item()
    )

    error = (output.float() - expected.float()).abs()
    relative = error / expected.float().abs().clamp_min(1.0e-6)
    return (
        rank_divergence,
        float(error.max().item()),
        float(error.mean().item()),
        float(relative.max().item()),
        bool(torch.isfinite(output).all().item()),
    )


def _correctness(
    dma,
    source: torch.Tensor,
    output: torch.Tensor | None,
) -> tuple[float, float, float, float, bool]:
    if output is None:
        actual = dma.all_reduce(source)
    else:
        actual = dma.all_reduce(source, out=output)
    return _compare_output(source, actual)


def _graph_correctness(
    dma,
    source: torch.Tensor,
) -> tuple[float, float, float, float, bool]:
    # Allocate the persistent result before capture. Production vLLM profiles
    # this default-output path before sizing KV, then records the same stable
    # pointer in its CUDA graphs.
    dma.all_reduce(source)
    torch.cuda.synchronize(source.device)
    graph_input = source.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = dma.all_reduce(graph_input)

    # Replay with fresh bytes so a stale capture output cannot pass.
    graph_input.copy_(source * 1.03125)
    graph.replay()
    torch.cuda.synchronize(source.device)
    return _compare_output(graph_input, graph_output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--rows", default="64,256,1024,3072")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() != 4:
        raise SystemExit(
            f"expected four ranks, found {dist.get_world_size()}"
        )

    from sparkinfer.comm.pcie import DmaAllReduce

    rows_values = tuple(
        int(value) for value in args.rows.split(",") if value
    )
    if not rows_values or any(value <= 0 for value in rows_values):
        raise SystemExit("--rows must contain positive integers")
    element_size = torch.empty((), dtype=torch.bfloat16).element_size()
    max_bytes = max(rows_values) * args.hidden * element_size
    dma = DmaAllReduce(
        exchange_group=dist.group.WORLD,
        device=device,
        max_bytes=max_bytes,
        fp8="i8_ring",
    )
    dma.min_bytes = 0

    failures = 0
    try:
        for rows in rows_values:
            generator = torch.Generator(device=device)
            generator.manual_seed(0x5EED + local_rank * 101 + rows)
            source = (
                torch.randn(
                    (rows, args.hidden),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                * 0.05
                + (local_rank + 1) * 0.01
            )
            if not dma.should_allreduce(source):
                raise RuntimeError(
                    f"DMA rejected production-shaped input {tuple(source.shape)}"
                )
            nccl_output = torch.empty_like(source)
            dma_output = torch.empty_like(source)

            explicit_correctness = _correctness(
                dma, source, dma_output
            )
            default_correctness = _correctness(
                dma, source, None
            )
            graph_correctness = _graph_correctness(
                dma, source
            )
            nccl_timings = _time_nccl(
                source,
                nccl_output,
                warmup=args.warmup,
                samples=args.samples,
            )
            dma_explicit_timings = _time_dma(
                dma,
                source,
                dma_output,
                warmup=args.warmup,
                samples=args.samples,
            )
            dma_default_timings = _time_dma(
                dma,
                source,
                None,
                warmup=args.warmup,
                samples=args.samples,
            )
            nccl_median = statistics.median(nccl_timings)
            dma_explicit_median = statistics.median(
                dma_explicit_timings
            )
            dma_default_median = statistics.median(
                dma_default_timings
            )
            result = CollectiveResult(
                rows=rows,
                hidden=args.hidden,
                bytes=source.numel() * source.element_size(),
                samples=args.samples,
                nccl_median_us=nccl_median,
                dma_explicit_median_us=dma_explicit_median,
                dma_default_median_us=dma_default_median,
                dma_explicit_over_nccl=dma_explicit_median / nccl_median,
                dma_default_over_nccl=dma_default_median / nccl_median,
                dma_default_over_explicit=(
                    dma_default_median / dma_explicit_median
                ),
                explicit_rank_max_abs_divergence=explicit_correctness[0],
                explicit_vs_nccl_max_abs_error=explicit_correctness[1],
                explicit_vs_nccl_mean_abs_error=explicit_correctness[2],
                explicit_vs_nccl_max_relative_error=explicit_correctness[3],
                explicit_finite=explicit_correctness[4],
                default_rank_max_abs_divergence=default_correctness[0],
                default_vs_nccl_max_abs_error=default_correctness[1],
                default_vs_nccl_mean_abs_error=default_correctness[2],
                default_vs_nccl_max_relative_error=default_correctness[3],
                default_finite=default_correctness[4],
                graph_rank_max_abs_divergence=graph_correctness[0],
                graph_vs_nccl_max_abs_error=graph_correctness[1],
                graph_vs_nccl_mean_abs_error=graph_correctness[2],
                graph_vs_nccl_max_relative_error=graph_correctness[3],
                graph_finite=graph_correctness[4],
            )
            if (
                explicit_correctness[0] != 0.0
                or not explicit_correctness[4]
                or default_correctness[0] != 0.0
                or not default_correctness[4]
                or graph_correctness[0] != 0.0
                or not graph_correctness[4]
            ):
                failures += 1
            if dist.get_rank() == 0:
                print(json.dumps(asdict(result), sort_keys=True), flush=True)
    finally:
        dist.barrier(device_ids=[device.index])
        dma.close()
        dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
