#!/usr/bin/env python3
"""Compare mixed-K serial Trellis decode and prefill plans at MTP3 shapes.

vLLM flattens the target-model verification batch to::

    live_rows = requests * (1 + num_speculative_tokens)

For MTP3, C8/C16/C24/C32 therefore exercise M=32/64/96/128.  The current
production configuration caps ``VLLM_EXL3_TRELLIS_MAX_M`` at 32, so the
larger three shapes are dispatched through the capacity-3072, block-M=64
prefill plan.  This probe measures that boundary directly without a model
boot.

The arithmetic oracle is the current production prefill plan.  Candidate
decode plans must reproduce its BF16 result bit-for-bit before their timing
is reported.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass

import torch

from sparkinfer.moe import trellis_moe
from tr3_mixk_fused_gpu_equivalence import _tier_map, _weights


@dataclass(frozen=True)
class PlanSpec:
    name: str
    capacity: int
    block_m: int


def _event_samples(fn, *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [
        start.elapsed_time(end)
        for start, end in zip(starts, ends, strict=True)
    ]


def _stats(samples_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "p95_ms": sorted(samples_ms)[
            min(len(samples_ms) - 1, int(0.95 * len(samples_ms)))
        ],
        "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    hidden = 6144
    intermediate = 512
    topk = 8
    total_experts = 256
    tile = (64, 256, 64, 256)
    live_rows = (32, 64, 96, 128)
    specs = (
        PlanSpec("prefill_cap3072_bm64", 3072, 64),
        PlanSpec("decode_cap128_bm8", 128, 8),
        PlanSpec("decode_cap128_bm64", 128, 64),
    )

    tier1_ids = list(range(3, total_experts, 4))
    tier1_set = set(tier1_ids)
    tier0_ids = [
        expert for expert in range(total_experts) if expert not in tier1_set
    ]
    gen = torch.Generator(device=device)
    gen.manual_seed(20260730)

    def signs(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            torch.randint(0, 2, shape, generator=gen, device=device)
            .mul_(2)
            .sub_(1)
            .to(torch.float16)
            .contiguous()
        )

    global_gate = signs((total_experts, hidden))
    global_up = signs((total_experts, hidden))
    global_inter = signs((total_experts, 3 * intermediate))
    global_down = signs((total_experts, hidden))
    idx0 = torch.tensor(tier0_ids, dtype=torch.int64, device=device)
    idx1 = torch.tensor(tier1_ids, dtype=torch.int64, device=device)
    weights = (
        _weights(
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
        ),
        _weights(
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
        ),
    )
    route_maps = (
        _tier_map(tier0_ids, total_experts, device),
        _tier_map(tier1_ids, total_experts, device),
    )

    max_rows = max(live_rows)
    x_all = (
        torch.randn(
            max_rows,
            hidden,
            dtype=torch.bfloat16,
            generator=gen,
            device=device,
        )
        .mul_(0.01)
        .contiguous()
    )
    logits = torch.randn(
        max_rows, topk, dtype=torch.float32, generator=gen, device=device
    )
    topk_weights_all = torch.softmax(logits, dim=-1).contiguous()
    topk_ids_all = torch.tensor(
        [
            [
                (row * topk + col * 37) % total_experts
                for col in range(topk)
            ]
            for row in range(max_rows)
        ],
        dtype=torch.int64,
        device=device,
    ).contiguous()

    rows: list[dict[str, object]] = []
    oracle_outputs: dict[int, torch.Tensor] = {}
    for spec_index, spec in enumerate(specs):
        plans = []
        arenas = []
        for bits, experts in ((3, len(tier0_ids)), (4, len(tier1_ids))):
            plan = trellis_moe.plan(
                trellis_moe.Caps(
                    max_tokens=spec.capacity,
                    num_topk=topk,
                    num_experts=experts,
                    hidden_size=hidden,
                    intermediate_size=intermediate,
                    route_num_experts=total_experts,
                    block_size_m=spec.block_m,
                    trellis_bits=bits,
                    tile_config=tile,
                    input_dtype=torch.bfloat16,
                    device=device,
                )
            )
            plans.append(plan)
            arenas.append(
                torch.empty(
                    plan.scratch_specs()[0].shape,
                    dtype=plan.scratch_specs()[0].dtype,
                    device=device,
                )
            )

        for m in live_rows:
            x = x_all[:m]
            topk_weights = topk_weights_all[:m]
            topk_ids = topk_ids_all[:m]
            accum = torch.empty(m, hidden, dtype=torch.float32, device=device)
            other = torch.empty(m, hidden, dtype=torch.float32, device=device)
            final = torch.empty(m, hidden, dtype=torch.bfloat16, device=device)
            bindings = (
                trellis_moe.bind(
                    plans[0],
                    scratch=arenas[0],
                    a=x,
                    weights=weights[0],
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    route_expert_map=route_maps[0],
                    output_expert_map=route_maps[0],
                    output=accum,
                ),
                trellis_moe.bind(
                    plans[1],
                    scratch=arenas[1],
                    a=x,
                    weights=weights[1],
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    route_expert_map=route_maps[1],
                    output_expert_map=route_maps[1],
                    output=other,
                ),
            )

            def run():
                trellis_moe.run(binding=bindings[0])
                trellis_moe.run(binding=bindings[1])
                accum.add_(other)
                final.copy_(accum)
                return final

            eager = run()
            torch.cuda.synchronize()
            eager_snapshot = eager.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(captured, eager_snapshot):
                raise RuntimeError(
                    f"{spec.name} M={m}: eager/graph output mismatch"
                )
            if spec_index == 0:
                oracle_outputs[m] = eager_snapshot
            bit_exact = torch.equal(eager_snapshot, oracle_outputs[m])
            if not bit_exact:
                max_abs = float(
                    (eager_snapshot.float() - oracle_outputs[m].float())
                    .abs()
                    .max()
                    .item()
                )
                raise RuntimeError(
                    f"{spec.name} M={m}: output differs from prefill oracle; "
                    f"max_abs={max_abs:.9g}"
                )
            samples = _event_samples(
                graph.replay, warmup=args.warmup, repeats=args.repeats
            )
            rows.append(
                {
                    "plan": spec.name,
                    "capacity": spec.capacity,
                    "block_m": spec.block_m,
                    "live_rows": m,
                    "scratch_mib": sum(
                        arena.numel() * arena.element_size() for arena in arenas
                    )
                    / (1 << 20),
                    "bit_exact_vs_prefill": bit_exact,
                    "graph": _stats(samples),
                }
            )

    result = {
        "schema": "tr3-mixk-serial-window-v1",
        "device": torch.cuda.get_device_name(device),
        "geometry": {
            "hidden": hidden,
            "intermediate_tp_local": intermediate,
            "experts": total_experts,
            "tier_signature": [[3, len(tier0_ids)], [4, len(tier1_ids)]],
            "topk": topk,
            "mtp": 3,
            "concurrency_to_rows": {"8": 32, "16": 64, "24": 96, "32": 128},
        },
        "rows": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


if __name__ == "__main__":
    main()
