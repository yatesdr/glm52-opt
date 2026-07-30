#!/usr/bin/env python3
"""Profile the production EXL3 Trellis MoE decode operator without a model boot.

The default geometry matches GLM-5.2 TP4:

* 256 routed experts
* hidden size 6144
* TP-local intermediate size 512
* top-k 8
* a capacity-32 planned operator bound to one live decode row

Synthetic native Trellis tensors are intentional.  This probe measures launch
topology and kernel cost; it does not attempt to evaluate model quality.  It
does verify that eager and CUDA-graph replay produce stable finite output.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from sparkinfer.moe import trellis_moe
from sparkinfer.moe._shared.kernels.w4a16 import kernel as w4a16_kernel


MCG_SENTINEL = 0xCBAC1FED


def _event_samples(fn, *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    return [
        start.elapsed_time(end)
        for start, end in zip(starts, ends, strict=True)
    ]


def _stats(samples_ms: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples_ms)
    return {
        "count": len(samples_ms),
        "min_us": ordered[0] * 1000.0,
        "median_us": statistics.median(ordered) * 1000.0,
        "mean_us": statistics.mean(ordered) * 1000.0,
        "p95_us": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1000.0,
        "stdev_us": statistics.stdev(ordered) * 1000.0
        if len(ordered) > 1
        else 0.0,
    }


def _kernel_rows(prof) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for event in prof.key_averages():
        device_total = float(getattr(event, "self_device_time_total", 0.0))
        if device_total <= 0.0:
            device_total = float(getattr(event, "self_cuda_time_total", 0.0))
        if device_total <= 0.0:
            continue
        rows.append(
            {
                "name": event.key,
                "count": int(event.count),
                "self_device_total_us": device_total,
                "self_device_mean_us": device_total / max(1, int(event.count)),
            }
        )
    return sorted(rows, key=lambda row: float(row["self_device_total_us"]), reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--plan-capacity", type=int, default=32)
    parser.add_argument("--live-rows", type=int, default=1)
    parser.add_argument("--block-m", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument(
        "--grid-x",
        type=int,
        help=(
            "Override the fused cooperative grid for a controlled scheduling "
            "sweep. Must not exceed the kernel residency cap."
        ),
    )
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260729)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.live_rows < 1 or args.live_rows > args.plan_capacity:
        raise ValueError("live rows must be in [1, plan capacity]")
    if args.hidden % 16 or args.intermediate % 16:
        raise ValueError("hidden and intermediate sizes must be divisible by 16")

    bits = 3
    tile_config = (64, 128, 64, 128)
    build_started = time.monotonic()
    w13 = torch.randint(
        -32768,
        32767,
        (
            2,
            args.experts,
            args.hidden // 16,
            args.intermediate // 16,
            16 * bits,
        ),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (
            args.experts,
            args.intermediate // 16,
            args.hidden // 16,
            16 * bits,
        ),
        dtype=torch.int16,
        device=device,
    )

    # ±1 is the real checkpoint contract for all rotation/sign tables.
    def signs(shape: tuple[int, ...]) -> torch.Tensor:
        values = torch.randint(0, 2, shape, dtype=torch.int8, device=device)
        return values.mul_(2).sub_(1).to(torch.float16).contiguous()

    gate_suh = signs((args.experts, args.hidden))
    up_suh = signs((args.experts, args.hidden))
    intermediate_rotations = signs((args.experts, 3 * args.intermediate))
    down_svh = signs((args.experts, args.hidden))
    weights = trellis_moe.prepare_weights(
        w13,
        w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        codebook="mcg",
        mcg=MCG_SENTINEL,
        tile_config=tile_config,
    )
    plan = trellis_moe.plan(
        trellis_moe.Caps(
            max_tokens=args.plan_capacity,
            num_topk=args.topk,
            num_experts=args.experts,
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            route_num_experts=args.experts,
            block_size_m=args.block_m,
            trellis_bits=bits,
            tile_config=tile_config,
            input_dtype=torch.float16,
            device=device,
        )
    )
    residency_cap = int(
        torch.cuda.get_device_properties(device).multi_processor_count
        * plan.fused_launch.blocks_per_sm
    )
    if args.grid_x is not None:
        if args.grid_x < 1 or args.grid_x > residency_cap:
            raise ValueError(
                f"grid-x must be in [1, {residency_cap}], got {args.grid_x}"
            )

        # The launch helper is the only policy seam. The compiled kernel remains
        # byte-identical, and every candidate stays under the cooperative
        # residency cap, so this isolates scheduling without rebuilding code.
        def fixed_grid_x(**_kwargs) -> int:
            return int(args.grid_x)

        w4a16_kernel._w4a16_fused_persistent_grid_x = fixed_grid_x
    scratch_spec = plan.scratch_specs()[0]
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    x = torch.randn(
        (args.live_rows, args.hidden), dtype=torch.float16, device=device
    ).mul_(1.0e-3)
    topk_ids = torch.arange(
        args.live_rows * args.topk, dtype=torch.int32, device=device
    ).view(args.live_rows, args.topk)
    topk_weights = torch.full(
        (args.live_rows, args.topk),
        1.0 / args.topk,
        dtype=torch.float32,
        device=device,
    )
    output = torch.empty(
        (args.plan_capacity, args.hidden), dtype=torch.float32, device=device
    )
    binding = trellis_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        weights=weights,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
    )

    eager_output = binding.run()
    torch.cuda.synchronize()
    if not torch.isfinite(eager_output).all():
        raise RuntimeError("eager output contains non-finite values")
    eager_snapshot = eager_output.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = binding.run()
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(captured_output, eager_snapshot):
        max_abs = float((captured_output - eager_snapshot).abs().max().item())
        raise RuntimeError(
            "eager and first graph replay outputs differ; "
            f"max_abs={max_abs:.9g}"
        )

    eager_samples = _event_samples(
        binding.run, warmup=args.warmup, iterations=args.iterations
    )
    graph_samples = _event_samples(
        graph.replay, warmup=args.warmup, iterations=args.iterations
    )

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.profile_iterations):
            binding.run()
    torch.cuda.synchronize()
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))

    result = {
        "schema": "tr3-trellis-decode-profile-v1",
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "geometry": {
            "experts": args.experts,
            "hidden": args.hidden,
            "intermediate_tp_local": args.intermediate,
            "topk": args.topk,
            "plan_capacity": args.plan_capacity,
            "live_rows": args.live_rows,
            "block_m": args.block_m,
            "bits": bits,
        },
        "plan": {
            "scratch_bytes": int(scratch.numel() * scratch.element_size()),
            "max_m_blocks": int(plan.fused_launch.max_m_blocks),
            # Route-packed launches use the full cooperative residency cap.
            # The compile result carries blocks_per_sm; the live device carries
            # the SM count.  Direct-route launches may right-size this value,
            # but the current full-rotation Trellis contract rejects that path.
            "grid_x": int(args.grid_x or residency_cap),
            "residency_cap": residency_cap,
            "full_rotation": bool(plan.fused_launch.full_rotation),
            "direct_topk_routes": bool(plan.fused_launch.direct_topk_routes),
            "fc1_tile": [
                int(plan.fused_launch.fc1_tile_k),
                int(plan.fused_launch.fc1_tile_n),
            ],
            "fc2_tile": [
                int(plan.fused_launch.fc2_tile_k),
                int(plan.fused_launch.fc2_tile_n),
            ],
        },
        "build_and_compile_s": time.monotonic() - build_started,
        "eager": _stats(eager_samples),
        "cuda_graph": _stats(graph_samples),
        "profile_iterations": args.profile_iterations,
        "kernel_rows": _kernel_rows(prof),
        "checks": {
            "finite": True,
            "eager_graph_bit_exact": True,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
