#!/usr/bin/env python3
"""GPU proof for the accurate tensor-core safe MLA query BMM.

Run this against a source-built image containing the four-argument
``safe_mla_query_bmm`` schema. It proves three properties before a model boot:

* the precise path is no less accurate than the regular path;
* its result can change bytes after the production FP8 quantization boundary;
* setting the reduction-precision guard does not leak cuBLAS handle state and
  remains CUDA-graph compatible.

Timing is reported, not gated: end-to-end acceptance still owns the production
prefill/decode performance thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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


def _load_ops() -> None:
    import vllm._custom_ops  # noqa: F401

    missing = [
        name
        for name in ("safe_mla_query_bmm", "static_scaled_fp8_quant")
        if not hasattr(torch.ops._C, name)
    ]
    if missing:
        raise RuntimeError("missing vLLM custom ops: " + ", ".join(missing))


def _time_mode(
    query: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    precise: bool,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        torch.ops._C.safe_mla_query_bmm(query, weight, output, precise)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        torch.ops._C.safe_mla_query_bmm(query, weight, output, precise)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def _graph_replay(
    query: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    precise: bool,
) -> torch.Tensor:
    stream = torch.cuda.Stream(device=query.device)
    stream.wait_stream(torch.cuda.current_stream(query.device))
    with torch.cuda.stream(stream):
        torch.ops._C.safe_mla_query_bmm(query, weight, output, precise)
    torch.cuda.current_stream(query.device).wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.safe_mla_query_bmm(query, weight, output, precise)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize(query.device)
    return output.clone()


@torch.inference_mode()
def run_case(
    *,
    device: torch.device,
    tokens: int,
    heads: int,
    seed: int,
    warmup: int,
    iterations: int,
    graph_sizes: set[int],
) -> dict[str, Any]:
    q_dim = 192
    latent_dim = 512
    torch.manual_seed(seed)
    query_storage = (
        torch.randn(
            tokens,
            heads,
            q_dim + 64,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.5
    )
    query = query_storage[..., :q_dim].transpose(0, 1)
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
    regular_before = torch.empty(
        heads,
        tokens,
        latent_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    precise = torch.empty_like(regular_before)
    regular_after = torch.empty_like(regular_before)

    torch.ops._C.safe_mla_query_bmm(query, weight, regular_before, False)
    torch.ops._C.safe_mla_query_bmm(query, weight, precise, True)
    torch.ops._C.safe_mla_query_bmm(query, weight, regular_after, False)
    torch.cuda.synchronize(device)
    if not torch.equal(regular_before, regular_after):
        raise AssertionError("precise call leaked cuBLAS math mode into regular call")

    reference = torch.bmm(query.float(), weight.float()).to(torch.bfloat16)
    regular_error = (regular_before.float() - reference.float()).abs()
    precise_error = (precise.float() - reference.float()).abs()
    regular_max = float(regular_error.max().item())
    precise_max = float(precise_error.max().item())
    if precise_max > regular_max:
        raise AssertionError(
            f"precise max error {precise_max} exceeds regular {regular_max}"
        )

    scale = torch.ones(1, dtype=torch.float32, device=device)
    regular_fp8 = torch.empty(
        tokens,
        heads * latent_dim,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    precise_fp8 = torch.empty_like(regular_fp8)
    torch.ops._C.static_scaled_fp8_quant(
        regular_fp8,
        regular_before.transpose(0, 1).contiguous().view(tokens, -1),
        scale,
    )
    torch.ops._C.static_scaled_fp8_quant(
        precise_fp8,
        precise.transpose(0, 1).contiguous().view(tokens, -1),
        scale,
    )
    torch.cuda.synchronize(device)

    regular_ms = _time_mode(
        query,
        weight,
        regular_after,
        precise=False,
        warmup=warmup,
        iterations=iterations,
    )
    precise_ms = _time_mode(
        query,
        weight,
        precise,
        precise=True,
        warmup=warmup,
        iterations=iterations,
    )

    graph_regular_sha256 = None
    graph_precise_sha256 = None
    if tokens in graph_sizes:
        eager_regular = regular_after.clone()
        eager_precise = precise.clone()
        graph_regular = _graph_replay(
            query,
            weight,
            regular_after,
            precise=False,
        )
        graph_precise = _graph_replay(
            query,
            weight,
            precise,
            precise=True,
        )
        if not torch.equal(graph_regular, eager_regular):
            raise AssertionError("regular CUDA-graph replay changed output")
        if not torch.equal(graph_precise, eager_precise):
            raise AssertionError("precise CUDA-graph replay changed output")
        graph_regular_sha256 = _sha256(graph_regular)
        graph_precise_sha256 = _sha256(graph_precise)

    return {
        "kind": "safe_query_accum_case",
        "tokens": tokens,
        "heads": heads,
        "seed": seed,
        "regular_bf16_sha256": _sha256(regular_before),
        "precise_bf16_sha256": _sha256(precise),
        "regular_fp8_sha256": _sha256(regular_fp8),
        "precise_fp8_sha256": _sha256(precise_fp8),
        "reference_bf16_sha256": _sha256(reference),
        "regular_max_abs_error": regular_max,
        "precise_max_abs_error": precise_max,
        "regular_mean_abs_error": float(regular_error.mean().item()),
        "precise_mean_abs_error": float(precise_error.mean().item()),
        "bf16_changed": not torch.equal(regular_before, precise),
        "fp8_changed": not torch.equal(regular_fp8, precise_fp8),
        "regular_ms": regular_ms,
        "precise_ms": precise_ms,
        "precise_over_regular": precise_ms / regular_ms,
        "graph_regular_sha256": graph_regular_sha256,
        "graph_precise_sha256": graph_precise_sha256,
        "handle_mode_restored": True,
    }


def _csv_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", default="1,4,9,16,32,256,1024,3072,8192")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seeds", default="7,19,41")
    parser.add_argument("--graph-sizes", default="9,3072")
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
    _load_ops()

    graph_sizes = set(_csv_ints(args.graph_sizes))
    count = 0
    fp8_changed = 0
    max_slowdown = 0.0
    for tokens in _csv_ints(args.tokens):
        for seed in _csv_ints(args.seeds):
            record = run_case(
                device=device,
                tokens=tokens,
                heads=args.heads,
                seed=seed,
                warmup=args.warmup,
                iterations=args.iterations,
                graph_sizes=graph_sizes,
            )
            _emit(record, args.output)
            count += 1
            fp8_changed += int(record["fp8_changed"])
            max_slowdown = max(max_slowdown, record["precise_over_regular"])

    if fp8_changed == 0:
        raise AssertionError("precise accumulation never changed post-FP8 bytes")
    _emit(
        {
            "kind": "summary",
            "cases": count,
            "fp8_changed_cases": fp8_changed,
            "max_precise_over_regular": max_slowdown,
            "status": "PASS",
        },
        args.output,
    )


if __name__ == "__main__":
    main()
