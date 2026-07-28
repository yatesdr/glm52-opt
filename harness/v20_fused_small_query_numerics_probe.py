#!/usr/bin/env python3
"""Compare staged and upstream-fused MLA query assembly for M<=32.

This no-model probe can run in both the current fa71 image and the PEDANTIC
binary-rewind discriminator. It fingerprints:

* safe BMM -> BF16 concat -> static FP8 quantization (the current overlay);
* upstream PR #174's fused BF16 projection/assembly -> FP8 output;
* a float32-reference -> BF16 -> static FP8 oracle.

The fused path only covers M<=32. Long prefill remains the responsibility of
the accurate safe-BMM reduction fix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import torch


def _sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _emit(record: dict[str, Any], output: Path | None) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        with output.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _load() -> Any:
    import vllm._custom_ops  # noqa: F401

    missing = [
        name
        for name in ("safe_mla_query_bmm", "static_scaled_fp8_quant")
        if not hasattr(torch.ops._C, name)
    ]
    if missing:
        raise RuntimeError("missing vLLM custom ops: " + ", ".join(missing))
    fused = importlib.import_module("sparkinfer.gemm.mla_query_projection")
    for name in ("run", "can_implement", "prewarm"):
        if not callable(getattr(fused, name, None)):
            raise RuntimeError(f"missing fused MLA query API: {name}")
    return fused


def _time(operation: Callable[[], None], *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


@torch.inference_mode()
def run_case(
    *,
    fused: Any,
    device: torch.device,
    tokens: int,
    heads: int,
    seed: int,
    q_scale_value: float,
    warmup: int,
    iterations: int,
    graph_sizes: set[int],
) -> dict[str, Any]:
    q_dim = 192
    latent_dim = 512
    rope_dim = 64
    torch.manual_seed(seed)
    query_storage = (
        torch.randn(
            tokens,
            heads,
            q_dim + rope_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.5
    )
    q_nope_storage, q_pe = query_storage.split((q_dim, rope_dim), dim=-1)
    q_nope = q_nope_storage.transpose(0, 1)
    weight = (
        torch.randn(
            heads,
            q_dim,
            latent_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.05
    )
    q_scale = torch.tensor([q_scale_value], dtype=torch.float32, device=device)
    projected = torch.empty(
        heads,
        tokens,
        latent_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    staged_bf16 = torch.empty(
        tokens,
        heads,
        latent_dim + rope_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    staged_fp8 = torch.empty_like(staged_bf16, dtype=torch.float8_e4m3fn)
    fused_fp8 = torch.empty_like(staged_fp8)
    reference_bf16 = torch.empty_like(staged_bf16)
    reference_fp8 = torch.empty_like(staged_fp8)

    if not fused.can_implement(
        num_heads=heads,
        max_m=tokens,
        nope_dim=q_dim,
        latent_dim=latent_dim,
        output_dtype=torch.float8_e4m3fn,
        weight_format="bf16",
        device=device,
    ):
        raise AssertionError(f"fused BF16/FP8 route rejected M={tokens}")

    def staged_operation() -> None:
        torch.ops._C.safe_mla_query_bmm(q_nope, weight, projected)
        torch.cat((projected.transpose(0, 1), q_pe), dim=-1, out=staged_bf16)
        torch.ops._C.static_scaled_fp8_quant(
            staged_fp8.view(tokens, -1),
            staged_bf16.view(tokens, -1),
            q_scale,
        )

    def fused_operation() -> None:
        fused.run(
            q_nope,
            weight,
            q_pe,
            fused_fp8,
            q_scale=q_scale,
        )

    staged_operation()
    fused_operation()
    reference_projected = torch.bmm(q_nope.float(), weight.float()).to(
        torch.bfloat16
    )
    torch.cat((reference_projected.transpose(0, 1), q_pe), dim=-1, out=reference_bf16)
    torch.ops._C.static_scaled_fp8_quant(
        reference_fp8.view(tokens, -1),
        reference_bf16.view(tokens, -1),
        q_scale,
    )
    torch.cuda.synchronize(device)

    staged_ms = _time(
        staged_operation,
        warmup=warmup,
        iterations=iterations,
    )
    fused_ms = _time(
        fused_operation,
        warmup=warmup,
        iterations=iterations,
    )

    graph_fused_sha256 = None
    if tokens in graph_sizes:
        eager_fused = fused_fp8.clone()
        graph_output = torch.empty_like(fused_fp8)

        def graph_operation() -> None:
            fused.run(
                q_nope,
                weight,
                q_pe,
                graph_output,
                q_scale=q_scale,
            )

        graph_operation()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_operation()
        graph.replay()
        graph.replay()
        torch.cuda.synchronize(device)
        if not torch.equal(graph_output, eager_fused):
            raise AssertionError(f"fused graph replay changed bytes at M={tokens}")
        graph_fused_sha256 = _sha256(graph_output)

    return {
        "kind": "fused_small_query_case",
        "tokens": tokens,
        "heads": heads,
        "seed": seed,
        "q_scale": q_scale_value,
        "staged_fp8_sha256": _sha256(staged_fp8),
        "fused_fp8_sha256": _sha256(fused_fp8),
        "reference_fp8_sha256": _sha256(reference_fp8),
        "staged_matches_reference": torch.equal(staged_fp8, reference_fp8),
        "fused_matches_reference": torch.equal(fused_fp8, reference_fp8),
        "fused_matches_staged": torch.equal(fused_fp8, staged_fp8),
        "staged_max_abs_error": float(
            (staged_fp8.float() - reference_fp8.float()).abs().max().item()
        ),
        "fused_max_abs_error": float(
            (fused_fp8.float() - reference_fp8.float()).abs().max().item()
        ),
        "staged_ms": staged_ms,
        "fused_ms": fused_ms,
        "fused_over_staged": fused_ms / staged_ms,
        "graph_fused_sha256": graph_fused_sha256,
    }


def _csv_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def _csv_floats(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", default="1,4,9,16,32")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seeds", default="7,19,41")
    parser.add_argument("--q-scales", default="0.5,1.0,2.0")
    parser.add_argument("--graph-sizes", default="9,32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.output is not None:
        args.output.unlink(missing_ok=True)
    if args.warmup < 1 or args.iterations < 1:
        raise SystemExit("--warmup and --iterations must be positive")
    fused = _load()

    graph_sizes = set(_csv_ints(args.graph_sizes))
    count = 0
    fused_exact = 0
    fused_better_or_equal = 0
    max_ratio = 0.0
    for tokens in _csv_ints(args.tokens):
        for seed in _csv_ints(args.seeds):
            for q_scale in _csv_floats(args.q_scales):
                record = run_case(
                    fused=fused,
                    device=device,
                    tokens=tokens,
                    heads=args.heads,
                    seed=seed,
                    q_scale_value=q_scale,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    graph_sizes=graph_sizes,
                )
                _emit(record, args.output)
                count += 1
                fused_exact += int(record["fused_matches_reference"])
                fused_better_or_equal += int(
                    record["fused_max_abs_error"]
                    <= record["staged_max_abs_error"]
                )
                max_ratio = max(max_ratio, record["fused_over_staged"])

    _emit(
        {
            "kind": "summary",
            "cases": count,
            "fused_exact_reference_cases": fused_exact,
            "fused_better_or_equal_cases": fused_better_or_equal,
            "max_fused_over_staged": max_ratio,
            "candidate_supported": fused_better_or_equal == count,
            "status": "COMPLETE",
        },
        args.output,
    )


if __name__ == "__main__":
    main()
