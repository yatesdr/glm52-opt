#!/usr/bin/env python3
"""GPU equivalence gate for the two-tier full-rotation Trellis kernel.

The reference is the production mixed-K association:
  serial tier-0 planned op -> serial tier-1 planned op -> fp32 add.
The candidate packs global routes once, emits both weight tiers from one
cooperative kernel, then uses two FP32 accumulators in one top-k kernel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from sparkinfer.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)
from sparkinfer.moe import trellis_moe

TrellisMoECaps = trellis_moe.Caps
TrellisMoEWeights = trellis_moe.Weights
bind_trellis_moe = trellis_moe.bind
bind_trellis_moe_full_rotation_hybrid = trellis_moe.bind_hybrid
build_trellis_moe_tier_local_map = trellis_moe.build_tier_local_map
plan_trellis_moe = trellis_moe.plan
plan_trellis_moe_full_rotation_hybrid = trellis_moe.plan_hybrid
run_trellis_moe = trellis_moe.run
run_trellis_moe_full_rotation_hybrid = trellis_moe.run_hybrid


def _weights(
    *,
    bits: int,
    experts: int,
    hidden: int,
    intermediate: int,
    tile: tuple[int, int, int, int],
    seed: int,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
) -> TrellisMoEWeights:
    prepared = prepare_trellis256_moe_weights(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=tile[1],
        fc2_tile_n=tile[3],
        device=gate_suh.device,
        seed=seed,
        params_dtype=torch.float16,
        w13_layout="trellis3_t256_proj",
        trellis_bits=bits,
        codebook="mcg",
        gate_suh=gate_suh,
        up_suh=up_suh,
    )
    return TrellisMoEWeights(
        w13=prepared.w13,
        w2=prepared.w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        trellis_bits=bits,
        tile_config=tile,
        device=gate_suh.device,
        _prepared=prepared,
    )


def _scratch(plan, device: torch.device) -> torch.Tensor:
    return torch.empty(plan.scratch_nbytes, dtype=torch.uint8, device=device)


def _tier_map(global_ids: list[int], total: int, device: torch.device) -> torch.Tensor:
    result = torch.full((total,), -1, dtype=torch.int32, device=device)
    ids = torch.tensor(global_ids, dtype=torch.int64, device=device)
    result[ids] = torch.arange(len(global_ids), dtype=torch.int32, device=device)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--exact-geometry", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-repeats", type=int, default=100)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)

    if args.exact_geometry:
        hidden, intermediate = 6144, 512
        # Match the production 192/64 mix while forcing non-contiguous
        # global->tier-local descriptors through every path.
        tier1_ids = list(range(3, 256, 4))
        tier1_set = set(tier1_ids)
        tier0_ids = [expert for expert in range(256) if expert not in tier1_set]
        max_m = 32
    else:
        hidden, intermediate = 256, 256
        tier0_ids = [0, 2, 4]
        tier1_ids = [1, 3]
        max_m = 32
    total_experts = len(tier0_ids) + len(tier1_ids)
    topk = min(8, total_experts)
    tile = (64, 256, 64, 256)

    gen = torch.Generator(device=device)
    gen.manual_seed(20260730)
    global_gate = (
        torch.randn(total_experts, hidden, generator=gen, device=device)
        .mul_(0.04)
        .add_(1.0)
        .to(torch.float16)
        .contiguous()
    )
    global_up = (
        torch.randn(total_experts, hidden, generator=gen, device=device)
        .mul_(0.04)
        .add_(1.0)
        .to(torch.float16)
        .contiguous()
    )
    global_inter = (
        torch.randn(total_experts, 3 * intermediate, generator=gen, device=device)
        .mul_(0.04)
        .add_(1.0)
        .to(torch.float16)
        .contiguous()
    )
    global_down = (
        torch.randn(total_experts, hidden, generator=gen, device=device)
        .mul_(0.04)
        .add_(1.0)
        .to(torch.float16)
        .contiguous()
    )
    idx0 = torch.tensor(tier0_ids, dtype=torch.int64, device=device)
    idx1 = torch.tensor(tier1_ids, dtype=torch.int64, device=device)
    weights0 = _weights(
        bits=3,
        experts=len(tier0_ids),
        hidden=hidden,
        intermediate=intermediate,
        tile=tile,
        seed=17,
        gate_suh=global_gate.index_select(0, idx0).contiguous(),
        up_suh=global_up.index_select(0, idx0).contiguous(),
        intermediate_rotations=global_inter.index_select(0, idx0).contiguous(),
        down_svh=global_down.index_select(0, idx0).contiguous(),
    )
    weights1 = _weights(
        bits=4,
        experts=len(tier1_ids),
        hidden=hidden,
        intermediate=intermediate,
        tile=tile,
        seed=29,
        gate_suh=global_gate.index_select(0, idx1).contiguous(),
        up_suh=global_up.index_select(0, idx1).contiguous(),
        intermediate_rotations=global_inter.index_select(0, idx1).contiguous(),
        down_svh=global_down.index_select(0, idx1).contiguous(),
    )
    route_map0 = _tier_map(tier0_ids, total_experts, device)
    route_map1 = _tier_map(tier1_ids, total_experts, device)

    def _plan(bits: int, experts: int):
        return plan_trellis_moe(
            TrellisMoECaps(
                max_tokens=max_m,
                num_topk=topk,
                num_experts=experts,
                hidden_size=hidden,
                intermediate_size=intermediate,
                route_num_experts=total_experts,
                block_size_m=8,
                trellis_bits=bits,
                tile_config=tile,
                input_dtype=torch.bfloat16,
                device=device,
            )
        )

    plan0 = _plan(3, len(tier0_ids))
    plan1 = _plan(4, len(tier1_ids))
    hybrid_plan = plan_trellis_moe_full_rotation_hybrid(plan0, plan1)
    tier_map = build_trellis_moe_tier_local_map(
        hybrid_plan,
        tier0_ids,
        tier1_ids,
    )
    scratch0 = _scratch(plan0, device)
    scratch1 = _scratch(plan1, device)
    cases = []
    m_values = () if args.benchmark_only else (1, 2, 4, 8, 16, 24, 32)
    route_modes = ("tier0", "tier1", "mixed")
    for m in m_values:
        for route_mode in route_modes:
            x = torch.randn(
                m, hidden, dtype=torch.bfloat16, generator=gen, device=device
            ).contiguous()
            if args.exact_geometry:
                # Synthetic prepared weights are deliberately unconstrained;
                # keep the exact production-width activation finite so this
                # remains an arithmetic-equivalence proof rather than an
                # overflow comparison.
                x.mul_(0.01)
            logits = torch.randn(
                m, topk, dtype=torch.float32, generator=gen, device=device
            )
            topk_weights = torch.softmax(logits, dim=-1).contiguous()
            pool = (
                tier0_ids
                if route_mode == "tier0"
                else tier1_ids
                if route_mode == "tier1"
                else list(range(total_experts))
            )
            routes_cpu = [
                [pool[(row * topk + col) % len(pool)] for col in range(topk)]
                for row in range(m)
            ]
            topk_ids = torch.tensor(
                routes_cpu, dtype=torch.int64, device=device
            ).contiguous()

            ref0 = torch.empty(m, hidden, dtype=torch.float32, device=device)
            bind0 = bind_trellis_moe(
                plan0,
                scratch=scratch0,
                a=x,
                weights=weights0,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                route_expert_map=route_map0,
                output_expert_map=route_map0,
                output=ref0,
            )
            run_trellis_moe(binding=bind0)
            ref0_rotation_gate = (
                bind0.rotation_a_gate[: m * topk * hidden].clone()
            )
            ref0_rotation_up = (
                bind0.rotation_a_up[: m * topk * hidden].clone()
            )
            ref0_activated = (
                bind0.intermediate_cache2[
                    : m * topk * intermediate
                ].clone()
            )
            ref0_routes = (
                bind0.intermediate_cache13[: m * topk * hidden]
                .view(m, topk, hidden)
                .clone()
            )
            bind1 = bind_trellis_moe(
                plan1,
                scratch=scratch1,
                a=x,
                weights=weights1,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                route_expert_map=route_map1,
                output_expert_map=route_map1,
            )
            ref1 = run_trellis_moe(binding=bind1)
            ref1_routes = (
                bind1.intermediate_cache13[: m * topk * hidden]
                .view(m, topk, hidden)
                .clone()
            )
            reference = ref0.add(ref1)
            tier0_mask = (tier_map[topk_ids] >> 8) == 0
            expected_routes = torch.where(
                tier0_mask.unsqueeze(-1), ref0_routes, ref1_routes
            )

            candidate = torch.empty(
                max_m, hidden, dtype=torch.float32, device=device
            )
            hybrid_binding = bind_trellis_moe_full_rotation_hybrid(
                hybrid_plan,
                scratch=scratch0,
                a=x,
                tier0_weights=weights0,
                tier1_weights=weights1,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                tier_local_map=tier_map,
                global_gate_suh=global_gate,
                global_up_suh=global_up,
                global_down_svh=global_down,
                output=candidate,
            )
            run_trellis_moe_full_rotation_hybrid(binding=hybrid_binding)
            torch.cuda.synchronize()
            cand = candidate[:m]
            candidate_routes = (
                hybrid_binding.intermediate_cache13[: m * topk * hidden]
                .view(m, topk, hidden)
            )
            rotation_gate_equal = torch.equal(
                ref0_rotation_gate,
                hybrid_binding.rotation_a_gate[: m * topk * hidden],
            )
            rotation_up_equal = torch.equal(
                ref0_rotation_up,
                hybrid_binding.rotation_a_up[: m * topk * hidden],
            )
            activated_equal = torch.equal(
                ref0_activated,
                hybrid_binding.intermediate_cache2[
                    : m * topk * intermediate
                ],
            )
            routes_equal = torch.equal(expected_routes, candidate_routes)
            routes_max_abs = float(
                (expected_routes - candidate_routes).abs().max().item()
            )
            equal = torch.equal(reference, cand)
            max_abs = float((reference - cand).abs().max().item())
            cases.append(
                {
                    "m": m,
                    "route_mode": route_mode,
                    "rotation_gate_equal_to_tier0": rotation_gate_equal,
                    "rotation_up_equal_to_tier0": rotation_up_equal,
                    "activated_equal_to_tier0": activated_equal,
                    "routes_equal": routes_equal,
                    "routes_max_abs": routes_max_abs,
                    "equal": equal,
                    "max_abs": max_abs,
                }
            )
            print(json.dumps(cases[-1]), flush=True)
            if not torch.isfinite(reference).all() or not torch.isfinite(cand).all():
                raise AssertionError(
                    f"non-finite equivalence fixture at M={m} mode={route_mode}"
                )
            if not equal:
                raise AssertionError(
                    f"fused mismatch at M={m} mode={route_mode}: max_abs={max_abs}"
                )

    benchmarks = []
    if args.benchmark or args.benchmark_only:
        for bench_m in (1, 2, 4, 8, 16, 24, 32):
            bench_x = torch.randn(
                bench_m,
                hidden,
                dtype=torch.bfloat16,
                generator=gen,
                device=device,
            ).contiguous()
            if args.exact_geometry:
                bench_x.mul_(0.01)
            bench_weights = torch.softmax(
                torch.randn(
                    bench_m,
                    topk,
                    dtype=torch.float32,
                    generator=gen,
                    device=device,
                ),
                dim=-1,
            ).contiguous()
            bench_ids = torch.tensor(
                [
                    [
                        (row * topk + col) % total_experts
                        for col in range(topk)
                    ]
                    for row in range(bench_m)
                ],
                dtype=torch.int64,
                device=device,
            ).contiguous()
            bench_scratch0 = _scratch(plan0, device)
            bench_scratch1 = _scratch(plan1, device)
            serial_accum = torch.empty(
                max_m, hidden, dtype=torch.float32, device=device
            )
            serial0 = bind_trellis_moe(
                plan0,
                scratch=bench_scratch0,
                a=bench_x,
                weights=weights0,
                topk_weights=bench_weights,
                topk_ids=bench_ids,
                route_expert_map=route_map0,
                output_expert_map=route_map0,
                output=serial_accum,
            )
            serial1 = bind_trellis_moe(
                plan1,
                scratch=bench_scratch1,
                a=bench_x,
                weights=weights1,
                topk_weights=bench_weights,
                topk_ids=bench_ids,
                route_expert_map=route_map1,
                output_expert_map=route_map1,
            )
            fused_output = torch.empty(
                max_m, hidden, dtype=torch.float32, device=device
            )
            hybrid_binding = bind_trellis_moe_full_rotation_hybrid(
                hybrid_plan,
                scratch=bench_scratch0,
                a=bench_x,
                tier0_weights=weights0,
                tier1_weights=weights1,
                topk_weights=bench_weights,
                topk_ids=bench_ids,
                tier_local_map=tier_map,
                global_gate_suh=global_gate,
                global_up_suh=global_up,
                global_down_svh=global_down,
                output=fused_output,
            )

            def _serial() -> None:
                run_trellis_moe(binding=serial0)
                tier1_output = run_trellis_moe(binding=serial1)
                serial0.output.add_(tier1_output)

            def _fused() -> None:
                run_trellis_moe_full_rotation_hybrid(
                    binding=hybrid_binding
                )

            # Resolve/compile every launch before capture.
            _serial()
            serial_expected = serial0.output.clone()
            _fused()
            if not torch.equal(serial_expected, fused_output[:bench_m]):
                raise AssertionError(
                    f"pre-capture benchmark mismatch at M={bench_m}"
                )
            torch.cuda.synchronize()

            serial_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(serial_graph):
                _serial()
            fused_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(fused_graph):
                _fused()
            serial_graph.replay()
            serial_expected.copy_(serial0.output)
            fused_graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(serial_expected, fused_output[:bench_m]):
                raise AssertionError(
                    f"graph replay mismatch at M={bench_m}"
                )

            def _time_graph(graph: torch.cuda.CUDAGraph) -> float:
                for _ in range(20):
                    graph.replay()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.benchmark_repeats):
                    graph.replay()
                end.record()
                end.synchronize()
                return float(start.elapsed_time(end)) / args.benchmark_repeats

            serial_ms = _time_graph(serial_graph)
            fused_ms = _time_graph(fused_graph)
            result = {
                "m": bench_m,
                "serial_ms": serial_ms,
                "fused_ms": fused_ms,
                "saved_ms": serial_ms - fused_ms,
                "speedup": serial_ms / fused_ms,
                "repeats": args.benchmark_repeats,
                "graph_replay_equal": True,
            }
            benchmarks.append(result)
            print(json.dumps({"benchmark": result}), flush=True)

    print(
        json.dumps(
            {
                "status": "PASS",
                "geometry": {
                    "hidden": hidden,
                    "intermediate": intermediate,
                    "tier0_experts": len(tier0_ids),
                    "tier1_experts": len(tier1_ids),
                    "topk": topk,
                },
                "fused_resources": {
                    "registers_per_thread": (
                        hybrid_plan.fused_launch.registers_per_thread
                    ),
                    "local_memory_bytes": (
                        hybrid_plan.fused_launch.local_memory_bytes
                    ),
                    "shared_memory_bytes": (
                        hybrid_plan.fused_launch.shared_memory_bytes
                    ),
                    "blocks_per_sm": (
                        hybrid_plan.fused_launch.blocks_per_sm
                    ),
                },
                "cases": len(cases),
                "benchmarks": benchmarks,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
