#!/usr/bin/env python3
"""Profile GLM-5.2 decode-time BF16 linears against B12X MXFP8.

This is a no-model, one-GPU discriminator for the residual TR3-vs-NF3
MTP0/C1 gap.  It uses the exact TP4 per-rank GLM-5.2 projection shapes and
the production SparkInfer MXFP8 linear API.  Weights are created and measured
one shape at a time so the probe does not depend on model-sized free VRAM.

The aggregate is an operator accounting estimate, not an end-to-end claim:
it deliberately excludes attention, collectives, norms, routing and the
routed-expert kernel.  A candidate is useful only if it is finite, clears the
cosine floor, and the estimated savings are large enough to explain the
measured serving gap.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from sparkinfer.gemm import mxfp8_linear
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)


@dataclass(frozen=True)
class Projection:
    name: str
    k: int
    n: int
    count: int
    note: str


# GLM-5.2: H=6144, q_lora=2048, kv_lora=512, 64 heads, TP=4,
# qk_nope=192, qk_rope=64, v=256, shared I=2048, dense I=12288.
PROJECTIONS = (
    Projection(
        "fused_qkv_a",
        6144,
        2624,
        78,
        "q_a[2048] + kv_a[512+64], replicated",
    ),
    Projection("q_b", 2048, 4096, 78, "64 heads / TP4 * (192+64)"),
    Projection("o_proj", 4096, 6144, 78, "64 heads / TP4 * v256"),
    Projection(
        "shared_gate_up",
        6144,
        1024,
        75,
        "two shared-I=2048 projections, column-sharded TP4",
    ),
    Projection(
        "shared_down",
        512,
        6144,
        75,
        "shared-I=2048 row-sharded TP4",
    ),
    Projection(
        "dense_gate_up",
        6144,
        6144,
        3,
        "layers 0..2, two I=12288 projections, column-sharded TP4",
    ),
    Projection(
        "dense_down",
        3072,
        6144,
        3,
        "layers 0..2, I=12288 row-sharded TP4",
    ),
)


def capture(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def event_times_us(graph: torch.cuda.CUDAGraph, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "min_us": min(values),
        "max_us": max(values),
        "stdev_us": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def run_projection(
    spec: Projection,
    *,
    seed: int,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    x = (torch.randn((1, spec.k), device="cuda", dtype=torch.bfloat16) / 4).contiguous()
    weight = (
        torch.randn((spec.n, spec.k), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()

    weight_q, weight_scale = mxfp8_e4m3_quantize(weight)
    packed = mxfp8_linear.pack_weight(weight_q, weight_scale)

    bf16_out = torch.empty((1, spec.n), device="cuda", dtype=torch.bfloat16)
    mxfp8_out = torch.empty_like(bf16_out)

    def bf16_call() -> None:
        torch.mm(x, weight.t(), out=bf16_out)

    def mxfp8_call() -> None:
        mxfp8_out.copy_(
            mxfp8_linear.mm(
                x,
                packed,
                expected_m=1,
                stream=torch.cuda.current_stream().cuda_stream,
            )
        )

    bf16_graph = capture(bf16_call)
    mxfp8_graph = capture(mxfp8_call)
    bf16_stats = summarize(event_times_us(bf16_graph, warmup, iters))
    mxfp8_stats = summarize(event_times_us(mxfp8_graph, warmup, iters))

    bf16_graph.replay()
    mxfp8_graph.replay()
    torch.cuda.synchronize()
    ref = bf16_out.float()
    cand = mxfp8_out.float()
    finite = bool(torch.isfinite(cand).all().item())
    cosine = float(F.cosine_similarity(ref.flatten(), cand.flatten(), dim=0).item())
    diff = cand - ref
    rmse = float(diff.square().mean().sqrt().item())
    max_abs = float(diff.abs().max().item())

    bf16_med = float(bf16_stats["median_us"])
    mxfp8_med = float(mxfp8_stats["median_us"])
    return {
        "projection": asdict(spec),
        "bf16": bf16_stats,
        "mxfp8": mxfp8_stats,
        "delta_us_per_call": bf16_med - mxfp8_med,
        "weighted_bf16_us": bf16_med * spec.count,
        "weighted_mxfp8_us": mxfp8_med * spec.count,
        "weighted_savings_us": (bf16_med - mxfp8_med) * spec.count,
        "quality": {
            "finite": finite,
            "cosine": cosine,
            "rmse": rmse,
            "max_abs": max_abs,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--cosine-floor", type=float, default=0.995)
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.cuda.get_device_properties(0)
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(PROJECTIONS):
        row = run_projection(
            spec,
            seed=args.seed + index,
            warmup=args.warmup,
            iters=args.iters,
        )
        quality = row["quality"]
        assert isinstance(quality, dict)
        if not quality["finite"] or float(quality["cosine"]) < args.cosine_floor:
            raise RuntimeError(
                f"{spec.name} failed quality gate: {json.dumps(quality, sort_keys=True)}"
            )
        rows.append(row)
        torch.cuda.empty_cache()

    totals = {
        "bf16_us": sum(float(row["weighted_bf16_us"]) for row in rows),
        "mxfp8_us": sum(float(row["weighted_mxfp8_us"]) for row in rows),
        "savings_us": sum(float(row["weighted_savings_us"]) for row in rows),
    }
    result = {
        "schema": "tr3-dense-decode-profile-v1",
        "device": {
            "name": device.name,
            "major": device.major,
            "minor": device.minor,
            "total_memory": device.total_memory,
        },
        "m": 1,
        "tp": 4,
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
        "totals": totals,
        "measured_end_to_end_gap_us": 6520.0,
        "fraction_of_gap_explained": totals["savings_us"] / 6520.0,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
