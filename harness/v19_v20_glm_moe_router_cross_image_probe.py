#!/usr/bin/env python3
"""Fingerprint GLM-5.2's gate GEMM and grouped top-k across images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _write_raw(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1711)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--repeat-count", type=int, default=4)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    if min(args.m, args.hidden, args.experts, args.topk, args.repeat_count) <= 0:
        raise ValueError("all dimensions and repeat-count must be positive")
    if args.topk > args.experts:
        raise ValueError("topk cannot exceed experts")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import vllm
    import vllm.envs as envs
    import vllm.model_executor.layers.linear as linear_module
    import vllm.model_executor.parameter as parameter_module

    # ReplicatedLinear only consults these helpers to annotate its parameters.
    # A standalone numeric probe has no distributed process group, so provide
    # the exact TP=1 construction context without initializing NCCL.
    linear_module.get_tensor_model_parallel_rank = lambda: 0
    linear_module.get_tensor_model_parallel_world_size = lambda: 1
    parameter_module.get_tensor_model_parallel_rank = lambda: 0
    parameter_module.get_tensor_model_parallel_world_size = lambda: 1

    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        grouped_topk,
    )

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device="cpu").manual_seed(20260726)

    source_i8 = torch.randint(
        -16,
        17,
        (args.m, args.hidden),
        dtype=torch.int8,
        generator=generator,
    )
    weight_i8 = torch.randint(
        -16,
        17,
        (args.experts, args.hidden),
        dtype=torch.int8,
        generator=generator,
    )
    bias_i16 = torch.randint(
        -64,
        65,
        (args.experts,),
        dtype=torch.int16,
        generator=generator,
    )
    source = (
        source_i8.to(device=device, dtype=torch.bfloat16) * 0.03125
    ).contiguous()
    weight = (
        weight_i8.to(device=device, dtype=torch.bfloat16) * 0.03125
    ).contiguous()
    correction_bias = (
        bias_i16.to(device=device, dtype=torch.float32) * (1.0 / 1024.0)
    ).contiguous()
    del source_i8, weight_i8, bias_i16

    old_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        gate = GateLinear(
            args.hidden,
            args.experts,
            bias=False,
            out_dtype=torch.float32,
        ).to(device)
    finally:
        torch.set_default_dtype(old_default)
    gate.weight.copy_(weight)

    def run_once() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_result = gate(source)
        logits = logits_result[0] if isinstance(logits_result, tuple) else logits_result
        topk_weights, topk_ids = grouped_topk(
            hidden_states=source,
            gating_output=logits,
            topk=args.topk,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            routed_scaling_factor=2.5,
            e_score_correction_bias=correction_bias,
        )
        return logits, topk_weights, topk_ids

    repeat_hashes: list[dict[str, str]] = []
    first_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    maximum_deltas = {"logits": 0.0, "topk_weights": 0.0}
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for _ in range(args.repeat_count):
        outputs = run_once()
        torch.cuda.synchronize(device)
        logits, topk_weights, topk_ids = outputs
        repeat_hashes.append(
            {
                "logits": _digest(logits),
                "topk_weights": _digest(topk_weights),
                "topk_ids": _digest(topk_ids),
            }
        )
        if first_outputs is None:
            first_outputs = tuple(value.clone() for value in outputs)
        else:
            maximum_deltas["logits"] = max(
                maximum_deltas["logits"],
                float((logits - first_outputs[0]).abs().max().item()),
            )
            maximum_deltas["topk_weights"] = max(
                maximum_deltas["topk_weights"],
                float((topk_weights - first_outputs[1]).abs().max().item()),
            )
            if not torch.equal(topk_ids, first_outputs[2]):
                raise RuntimeError("top-k ids changed between identical repeats")
    assert outputs is not None
    logits, topk_weights, topk_ids = outputs
    expected_shapes = (
        (args.m, args.experts),
        (args.m, args.topk),
        (args.m, args.topk),
    )
    if tuple(logits.shape) != expected_shapes[0]:
        raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)}")
    if tuple(topk_weights.shape) != expected_shapes[1]:
        raise RuntimeError(
            f"unexpected top-k weight shape {tuple(topk_weights.shape)}"
        )
    if tuple(topk_ids.shape) != expected_shapes[2]:
        raise RuntimeError(f"unexpected top-k id shape {tuple(topk_ids.shape)}")
    if args.output_prefix is not None:
        _write_raw(args.output_prefix.with_suffix(".logits.bin"), logits)
        _write_raw(args.output_prefix.with_suffix(".weights.bin"), topk_weights)
        _write_raw(args.output_prefix.with_suffix(".ids.bin"), topk_ids)

    finite = bool(
        torch.isfinite(logits).all().item()
        and torch.isfinite(topk_weights).all().item()
    )
    result = {
        "kind": "glm_moe_router_cross_image",
        "vllm_version": str(getattr(vllm, "__version__", "(unknown)")),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "m": args.m,
        "hidden": args.hidden,
        "experts": args.experts,
        "topk": args.topk,
        "gate_weight_dtype": str(gate.weight.dtype),
        "gate_out_dtype": str(gate.out_dtype),
        "allow_ll_bf16_gemm": bool(gate.allow_ll_bf16_gemm),
        "allow_dsv3_router_gemm": bool(gate.allow_dsv3_router_gemm),
        "allow_cublas_router_gemm": bool(gate.allow_cublas_router_gemm),
        "use_fused_moe_grouped_topk": bool(
            envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK
        ),
        "source_sha256": _digest(source),
        "weight_sha256": _digest(weight),
        "correction_bias_sha256": _digest(correction_bias),
        "logits_sha256": _digest(logits),
        "topk_weights_sha256": _digest(topk_weights),
        "topk_ids_sha256": _digest(topk_ids),
        "repeat_output_sha256": repeat_hashes,
        "repeat_max_abs_delta_from_first": maximum_deltas,
        "finite": finite,
        "status": "PASS" if finite else "FAIL_NONFINITE",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
