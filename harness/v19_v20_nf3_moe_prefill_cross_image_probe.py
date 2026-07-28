#!/usr/bin/env python3
"""Cross-image NF3 W4A16 MoE fingerprint at production prefill geometry.

This runs the exact packaged W4A16 prefill path from the known-good v19 and
failing v20 images with bit-identical inputs and prepared NF3 weights.  The
problem keeps GLM-5.2's hidden/intermediate dimensions, top-k, block-64 route
packing, tile pin, compile capacity, and frozen 350k tail row count.  It uses
four synthetic experts instead of 192 so the proof isolates kernel numerics
without loading the checkpoint.

The script emits hashes for every source/prepared tensor and writes the raw
BF16 output when ``--output`` is supplied.  Equal prepared hashes plus unequal
output hashes localize drift to W4A16 execution rather than packing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch


HIDDEN = 6144
INTERMEDIATE = 512
TOPK = 8
EXPERTS = 4
CAPACITY_M = 2048
ACTIVE_M = 1711
TILES = (64, 256, 64, 256)
MOE_BLOCK_SIZE = 64


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _round_to_e4m3_scale(scale: torch.Tensor) -> torch.Tensor:
    scale = scale.to(torch.float32).clamp(min=2.0**-7)
    exponent = torch.floor(torch.log2(scale))
    step = torch.pow(2.0, exponent - 3)
    return torch.round(scale / step) * step


def _launch_metadata(launch: object) -> dict[str, object]:
    """Return stable scalar compile-plan fields without serializing the kernel."""
    if not is_dataclass(launch):
        return {}
    result: dict[str, object] = {}
    for field in fields(launch):
        value = getattr(launch, field.name)
        if isinstance(value, (bool, float, int, str)) or value is None:
            result[field.name] = value
    return result


def _write_raw(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )


def _load_api():
    try:
        import sparkinfer
        from sparkinfer.moe._shared.kernels.w4a16.host import (
            make_w4a16_packed_buffers,
            max_packed_route_slots,
        )
        from sparkinfer.moe._shared.kernels.w4a16.kernel import (
            compile_w4a16_fused_moe,
            run_w4a16_moe,
        )
        from sparkinfer.moe._shared.kernels.w4a16.prepare import (
            prepare_nf3_moe_weights,
        )

        return (
            "sparkinfer",
            str(getattr(sparkinfer, "__version__", "(unknown)")),
            make_w4a16_packed_buffers,
            max_packed_route_slots,
            compile_w4a16_fused_moe,
            run_w4a16_moe,
            prepare_nf3_moe_weights,
        )
    except ImportError:
        import b12x
        from b12x.moe.fused.w4a16.host import (
            make_w4a16_packed_buffers,
            max_packed_route_slots,
        )
        from b12x.moe.fused.w4a16.kernel import (
            compile_w4a16_fused_moe,
            run_w4a16_moe,
        )
        from b12x.moe.fused.w4a16.prepare import prepare_nf3_moe_weights

        return (
            "b12x",
            str(getattr(b12x, "__version__", "(unknown)")),
            make_w4a16_packed_buffers,
            max_packed_route_slots,
            compile_w4a16_fused_moe,
            run_w4a16_moe,
            prepare_nf3_moe_weights,
        )


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-m", type=int, default=CAPACITY_M)
    parser.add_argument("--active-m", type=int, default=ACTIVE_M)
    parser.add_argument("--experts", type=int, default=EXPERTS)
    parser.add_argument(
        "--local-routes",
        action="store_true",
        help=(
            "disable the production expert-map filter so FC2 route outputs "
            "and the top-k sum provide a deterministic numeric discriminator"
        ),
    )
    parser.add_argument(
        "--direct-topk-routes",
        action="store_true",
        help=(
            "compile the small-M direct-top-k specialization; requires "
            "--local-routes and capacity-m <= 6"
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument(
        "--mean-output-prefix",
        type=Path,
        help=(
            "write float32 overall/first-half/second-half repeat means using "
            "this path as a filename prefix"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.active_m <= args.capacity_m:
        raise ValueError("active-m must be in [1, capacity-m]")
    if args.experts < TOPK:
        # Repeated routes are valid, but every production token still carries
        # eight routed slots. Four experts keeps preparation bounded.
        pass
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeat_count < 1:
        raise ValueError("repeat-count must be positive")
    if args.mean_output_prefix is not None and args.repeat_count < 2:
        raise ValueError("mean-output-prefix requires repeat-count >= 2")
    if args.direct_topk_routes and not args.local_routes:
        raise ValueError("direct-topk-routes requires local-routes")
    if args.direct_topk_routes and args.capacity_m > 6:
        raise ValueError("direct-topk-routes requires capacity-m <= 6")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    (
        package,
        package_version,
        make_buffers,
        max_packed_route_slots,
        compile_moe,
        run_moe,
        prepare_nf3,
    ) = _load_api()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260726)
    w13_codes = torch.randint(
        0,
        8,
        (args.experts, 2 * INTERMEDIATE, HIDDEN),
        dtype=torch.int32,
        generator=generator,
    ).to(device)
    w2_codes = torch.randint(
        0,
        8,
        (args.experts, HIDDEN, INTERMEDIATE),
        dtype=torch.int32,
        generator=generator,
    ).to(device)
    w13_scale = _round_to_e4m3_scale(
        0.01
        + 0.24
        * torch.rand(
            args.experts,
            2 * INTERMEDIATE,
            HIDDEN // 32,
            generator=generator,
        )
    ).to(device)
    w2_scale = _round_to_e4m3_scale(
        0.01
        + 0.24
        * torch.rand(
            args.experts,
            HIDDEN,
            INTERMEDIATE // 32,
            generator=generator,
        )
    ).to(device)
    x = (
        torch.randn(
            args.active_m,
            HIDDEN,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.1
    ).to(torch.bfloat16).to(device)
    topk_weights = torch.softmax(
        torch.randn(
            args.active_m,
            TOPK,
            dtype=torch.float32,
            generator=generator,
        ),
        dim=-1,
    ).to(device)
    topk_ids = torch.randint(
        0,
        args.experts,
        (args.active_m, TOPK),
        dtype=torch.int32,
        generator=generator,
    ).to(device)
    expert_map = (
        None
        if args.local_routes
        else torch.arange(
            args.experts,
            dtype=torch.int32,
            device=device,
        )
    )

    source_hashes = {
        "w13_codes": _digest(w13_codes),
        "w2_codes": _digest(w2_codes),
        "w13_scale": _digest(w13_scale),
        "w2_scale": _digest(w2_scale),
        "x": _digest(x),
        "topk_weights": _digest(topk_weights),
        "topk_ids": _digest(topk_ids),
    }
    if expert_map is not None:
        source_hashes["expert_map"] = _digest(expert_map)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    prepared = prepare_nf3(
        w13_codes,
        w13_scale,
        w2_codes,
        w2_scale,
        activation="silu",
        fc1_tile_n=TILES[1],
        fc2_tile_n=TILES[3],
        params_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize(device)
    prepare_seconds = time.monotonic() - started
    prepared_hashes = {
        "w13": _digest(prepared.w13),
        "w2": _digest(prepared.w2),
        "w13_scale": _digest(prepared.w13_scale),
        "w2_scale": _digest(prepared.w2_scale),
        "w13_global_scale": _digest(prepared.w13_global_scale),
        "w2_global_scale": _digest(prepared.w2_global_scale),
    }
    del w13_codes, w2_codes, w13_scale, w2_scale

    properties = torch.cuda.get_device_properties(device)
    cap_slots = max_packed_route_slots(
        args.capacity_m * TOPK,
        MOE_BLOCK_SIZE,
        args.experts,
    )
    compile_max_m_blocks = (
        args.capacity_m * TOPK
        if args.direct_topk_routes
        else (cap_slots + MOE_BLOCK_SIZE - 1) // MOE_BLOCK_SIZE
    )
    started = time.monotonic()
    launch = compile_moe(
        size_m=args.capacity_m,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=args.experts,
        top_k=TOPK,
        activation="silu",
        apply_router_weight_on_input=False,
        zero_fc2_output=expert_map is not None,
        moe_block_size=MOE_BLOCK_SIZE,
        max_m_blocks=compile_max_m_blocks,
        element_dtype="bf16",
        fast_math=True,
        sms=int(properties.multi_processor_count),
        max_shared_mem=int(
            getattr(properties, "shared_memory_per_block_optin", 101_376)
        ),
        weight_layout=prepared.weight_layout,
        scale_format=prepared.scale_format,
        force_tile_config=TILES,
        direct_topk_routes=args.direct_topk_routes,
        tc_decode_fused_sum=False,
    )
    torch.cuda.synchronize(device)
    compile_seconds = time.monotonic() - started
    buffers = make_buffers(
        prepared,
        m=args.capacity_m,
        topk=TOPK,
        dtype=torch.bfloat16,
        device=device,
        route_num_experts=args.experts,
    )

    def run_once() -> torch.Tensor:
        return run_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation="silu",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output[: args.active_m],
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_map=expert_map,
            fused_launch=launch,
        )

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize(device)

    started = time.monotonic()
    repeat_hashes: list[str] = []
    first_hash: str | None = None
    changed_from_first = 0
    maximum_delta_from_first = 0.0
    first_output: torch.Tensor | None = None
    first_count = (args.repeat_count + 1) // 2
    first_sum: torch.Tensor | None = None
    second_sum: torch.Tensor | None = None
    output: torch.Tensor | None = None
    for repeat_index in range(args.repeat_count):
        output = run_once()
        torch.cuda.synchronize(device)
        output_hash = _digest(output)
        repeat_hashes.append(output_hash)
        if first_output is None:
            first_output = output.clone()
            first_hash = output_hash
        elif output_hash != first_hash:
            changed_from_first += 1
            maximum_delta_from_first = max(
                maximum_delta_from_first,
                float((output.float() - first_output.float()).abs().max().item()),
            )
        if args.mean_output_prefix is not None:
            target = "first" if repeat_index < first_count else "second"
            if target == "first":
                if first_sum is None:
                    first_sum = torch.zeros_like(output, dtype=torch.float64)
                first_sum.add_(output)
            else:
                if second_sum is None:
                    second_sum = torch.zeros_like(output, dtype=torch.float64)
                second_sum.add_(output)
    torch.cuda.synchronize(device)
    run_seconds = time.monotonic() - started
    assert output is not None

    if tuple(output.shape) != (args.active_m, HIDDEN):
        raise RuntimeError(f"unexpected output shape {tuple(output.shape)}")
    finite = bool(torch.isfinite(output).all().item())
    if args.output is not None:
        _write_raw(args.output, output)
    mean_paths: dict[str, str] = {}
    if args.mean_output_prefix is not None:
        assert first_sum is not None and second_sum is not None
        second_count = args.repeat_count - first_count
        first_mean = (first_sum / first_count).to(torch.float32)
        second_mean = (second_sum / second_count).to(torch.float32)
        overall_mean = (
            (first_sum + second_sum) / args.repeat_count
        ).to(torch.float32)
        for label, tensor in (
            ("overall", overall_mean),
            ("first_half", first_mean),
            ("second_half", second_mean),
        ):
            path = Path(f"{args.mean_output_prefix}.{label}.f32")
            _write_raw(path, tensor)
            mean_paths[label] = str(path)
    result = {
        "kind": "nf3_moe_prefill_cross_image",
        "package": package,
        "package_version": package_version,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "capacity_m": args.capacity_m,
        "active_m": args.active_m,
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "experts": args.experts,
        "topk": TOPK,
        "tiles": TILES,
        "cap_slots": cap_slots,
        "compile_max_m_blocks": compile_max_m_blocks,
        "local_routes": args.local_routes,
        "direct_topk_routes": args.direct_topk_routes,
        "launch": _launch_metadata(launch),
        "source_sha256": source_hashes,
        "prepared_sha256": prepared_hashes,
        "output_sha256": _digest(output),
        "warmup": args.warmup,
        "repeat_count": args.repeat_count,
        "repeat_output_sha256": repeat_hashes,
        "repeat_unique_outputs": len(set(repeat_hashes)),
        "repeat_changed_from_first": changed_from_first,
        "repeat_max_abs_delta_from_first": maximum_delta_from_first,
        "mean_output_paths": mean_paths,
        "output_abs_max": float(output.float().abs().max().item()),
        "finite": finite,
        "prepare_seconds": prepare_seconds,
        "compile_seconds": compile_seconds,
        "run_seconds": run_seconds,
        "status": "PASS" if finite else "FAIL_NONFINITE",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
