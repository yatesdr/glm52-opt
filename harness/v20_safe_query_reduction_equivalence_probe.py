#!/usr/bin/env python3
"""Fingerprint one safe-query reduction mode without booting a model.

Run this probe four times across three images:

* current image: ``--mode legacy-current``;
* PEDANTIC rewind: ``--mode legacy-pedantic``;
* accurate-reduction image: ``--mode accurate-regular``;
* accurate-reduction image: ``--mode accurate-precise``.

The companion comparator proves that the new regular call preserves the
current route, and that the new precise call reproduces PEDANTIC at the FP8
boundary consumed by the model. Timings cover both the BMM and the complete
BMM -> BF16 assembly -> FP8 quantization pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch


LEGACY_MODES = {"legacy-current", "legacy-pedantic"}
ACCURATE_MODES = {"accurate-regular", "accurate-precise"}
ALL_MODES = LEGACY_MODES | ACCURATE_MODES


def _sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _emit(record: dict[str, Any], output: Path | None) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        with output.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ops(expected_stable_sha256: str) -> str:
    import vllm
    import vllm._custom_ops  # noqa: F401

    package_dir = Path(vllm.__file__).resolve().parent
    candidates = sorted(package_dir.glob("_C_stable_libtorch*.so"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one stable-libtorch extension, got {candidates}"
        )
    stable_sha256 = _file_sha256(candidates[0])
    if stable_sha256 != expected_stable_sha256:
        raise RuntimeError(
            "stable-libtorch byte pin mismatch: "
            f"expected {expected_stable_sha256}, got {stable_sha256}"
        )
    missing = [
        name
        for name in ("safe_mla_query_bmm", "static_scaled_fp8_quant")
        if not hasattr(torch.ops._C, name)
    ]
    if missing:
        raise RuntimeError("missing vLLM custom ops: " + ", ".join(missing))
    return stable_sha256


def _call_safe_bmm(
    mode: str,
    query: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if mode in LEGACY_MODES:
        torch.ops._C.safe_mla_query_bmm(query, weight, output)
        return
    torch.ops._C.safe_mla_query_bmm(
        query,
        weight,
        output,
        mode == "accurate-precise",
    )


def _time(
    operation: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
) -> float:
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


def _graph_replay(
    operation: Callable[[], None],
    *,
    device: torch.device,
) -> None:
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        operation()
    torch.cuda.current_stream(device).wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize(device)


@torch.inference_mode()
def run_case(
    *,
    mode: str,
    device: torch.device,
    tokens: int,
    heads: int,
    seed: int,
    q_scale_value: float,
    warmup: int,
    iterations: int,
    graph_replay: bool,
) -> dict[str, Any]:
    q_dim = 192
    rope_dim = 64
    latent_dim = 512
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
    query = q_nope_storage.transpose(0, 1)
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
    projected = torch.empty(
        heads,
        tokens,
        latent_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    assembled = torch.empty(
        tokens,
        heads,
        latent_dim + rope_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    quantized = torch.empty_like(assembled, dtype=torch.float8_e4m3fn)
    q_scale = torch.tensor([q_scale_value], dtype=torch.float32, device=device)

    def bmm_operation() -> None:
        _call_safe_bmm(mode, query, weight, projected)

    def pipeline_operation() -> None:
        bmm_operation()
        torch.cat((projected.transpose(0, 1), q_pe), dim=-1, out=assembled)
        torch.ops._C.static_scaled_fp8_quant(
            quantized.view(tokens, -1),
            assembled.view(tokens, -1),
            q_scale,
        )

    assert not query.is_contiguous()
    pipeline_operation()
    torch.cuda.synchronize(device)

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference_projected = torch.bmm(query.float(), weight.float()).to(
            torch.bfloat16
        )
        reference_assembled = torch.empty_like(assembled)
        torch.cat(
            (reference_projected.transpose(0, 1), q_pe),
            dim=-1,
            out=reference_assembled,
        )
        reference_fp8 = torch.empty_like(quantized)
        torch.ops._C.static_scaled_fp8_quant(
            reference_fp8.view(tokens, -1),
            reference_assembled.view(tokens, -1),
            q_scale,
        )
        torch.cuda.synchronize(device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    error = (projected.float() - reference_projected.float()).abs()
    eager_bf16_sha256 = _sha256(projected)
    eager_fp8_sha256 = _sha256(quantized)
    bmm_ms = _time(
        bmm_operation,
        warmup=warmup,
        iterations=iterations,
    )
    pipeline_ms = _time(
        pipeline_operation,
        warmup=warmup,
        iterations=iterations,
    )

    graph_bf16_sha256 = None
    graph_fp8_sha256 = None
    if graph_replay:
        pipeline_operation()
        torch.cuda.synchronize(device)
        eager_bf16_sha256 = _sha256(projected)
        eager_fp8_sha256 = _sha256(quantized)
        _graph_replay(pipeline_operation, device=device)
        graph_bf16_sha256 = _sha256(projected)
        graph_fp8_sha256 = _sha256(quantized)
        if graph_bf16_sha256 != eager_bf16_sha256:
            raise AssertionError(f"{mode}: CUDA graph changed BF16 bytes at M={tokens}")
        if graph_fp8_sha256 != eager_fp8_sha256:
            raise AssertionError(f"{mode}: CUDA graph changed FP8 bytes at M={tokens}")

    return {
        "kind": "safe_query_reduction_equivalence_case",
        "mode": mode,
        "tokens": tokens,
        "heads": heads,
        "seed": seed,
        "q_scale": q_scale_value,
        "bf16_sha256": eager_bf16_sha256,
        "fp8_sha256": eager_fp8_sha256,
        "reference_bf16_sha256": _sha256(reference_projected),
        "reference_fp8_sha256": _sha256(reference_fp8),
        "reference_max_abs_error": float(error.max().item()),
        "reference_mean_abs_error": float(error.mean().item()),
        "bmm_ms": bmm_ms,
        "pipeline_ms": pipeline_ms,
        "graph_bf16_sha256": graph_bf16_sha256,
        "graph_fp8_sha256": graph_fp8_sha256,
    }


def _csv_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def _csv_floats(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(ALL_MODES))
    parser.add_argument("--expected-stable-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", default="1,4,9,16,32,3072")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seeds", default="7,19,41")
    parser.add_argument("--q-scales", default="0.5,1.0,2.0")
    parser.add_argument("--graph-sizes", default="9,3072")
    parser.add_argument(
        "--skip-graph-replay",
        action="store_true",
        help=(
            "run the complete numeric/timing sweep without creating CUDA "
            "graphs; use when a live model leaves insufficient handle memory"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
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
    stable_sha256 = _load_ops(args.expected_stable_sha256)

    tokens_values = _csv_ints(args.tokens)
    seed_values = _csv_ints(args.seeds)
    scale_values = _csv_floats(args.q_scales)
    if not tokens_values or not seed_values or not scale_values:
        raise SystemExit("--tokens, --seeds, and --q-scales must be non-empty")
    graph_sizes = set(_csv_ints(args.graph_sizes))
    count = 0
    graph_cases = 0
    for tokens in tokens_values:
        for seed in seed_values:
            for q_scale in scale_values:
                # Graph compatibility is a property of the BMM/pipeline
                # geometry, not of the random seed or static quant scale.
                # Repeated captures retain cuBLAS handles and can exhaust a
                # nearly-full GPU, so capture one representative per width.
                graph_replay = (
                    not args.skip_graph_replay
                    and tokens in graph_sizes
                    and seed == seed_values[0]
                    and q_scale == scale_values[0]
                )
                _emit(
                    run_case(
                        mode=args.mode,
                        device=device,
                        tokens=tokens,
                        heads=args.heads,
                        seed=seed,
                        q_scale_value=q_scale,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        graph_replay=graph_replay,
                    ),
                    args.output,
                )
                count += 1
                graph_cases += int(graph_replay)
    _emit(
        {
            "kind": "summary",
            "mode": args.mode,
            "cases": count,
            "graph_cases": graph_cases,
            "graph_replay_enabled": not args.skip_graph_replay,
            "stable_libtorch_sha256": stable_sha256,
            "status": "PASS",
        },
        args.output,
    )


if __name__ == "__main__":
    main()
