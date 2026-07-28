#!/usr/bin/env python3
"""Cross-image MXFP8 linear fingerprint at GLM-5.2 prefill geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
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


def _load_api():
    try:
        import sparkinfer
        from sparkinfer.gemm.mxfp8_linear import mm, pack_weight

        return (
            "sparkinfer",
            str(getattr(sparkinfer, "__version__", "(unknown)")),
            pack_weight,
            mm,
        )
    except ImportError:
        import b12x
        from b12x.gemm.mxfp8_linear import (
            mxfp8_linear,
            pack_mxfp8_linear_weight,
        )

        return (
            "b12x",
            str(getattr(b12x, "__version__", "(unknown)")),
            pack_mxfp8_linear_weight,
            mxfp8_linear,
        )


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1711)
    parser.add_argument("--n", type=int, default=6144)
    parser.add_argument("--k", type=int, default=6144)
    parser.add_argument("--expected-m", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat-count", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.expected_m, args.repeat_count) <= 0:
        raise ValueError("all dimensions and repeat-count must be positive")
    if args.k % 32:
        raise ValueError("k must be divisible by 32")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    package, package_version, pack_weight, linear = _load_api()
    generator = torch.Generator(device="cpu").manual_seed(20260726)

    source_i8 = torch.randint(
        -16,
        17,
        (args.m, args.k),
        dtype=torch.int8,
        generator=generator,
    )
    weight_i8 = torch.randint(
        -16,
        17,
        (args.n, args.k),
        dtype=torch.int8,
        generator=generator,
    )
    scale_u8 = torch.randint(
        122,
        131,
        (args.n, args.k // 32),
        dtype=torch.uint8,
        generator=generator,
    )
    source = (
        source_i8.to(device=device, dtype=torch.bfloat16) * 0.03125
    ).contiguous()
    weight = (
        weight_i8.to(device=device, dtype=torch.bfloat16) * 0.03125
    ).to(torch.float8_e4m3fn)
    weight_scale = scale_u8.to(device).contiguous()
    del source_i8, weight_i8, scale_u8
    source_hash = _digest(source)
    weight_hash = _digest(weight)
    weight_scale_hash = _digest(weight_scale)

    started = time.monotonic()
    packed = pack_weight(weight, weight_scale)
    torch.cuda.synchronize(device)
    pack_seconds = time.monotonic() - started
    packed_hashes = {
        "values": _digest(packed.weight.values),
        "scale_rows": _digest(packed.weight.scale_rows),
        "scale_mma": _digest(packed.weight.scale_mma),
    }

    def run_once() -> torch.Tensor:
        return linear(source, packed, expected_m=args.expected_m)

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize(device)
    started = time.monotonic()
    repeat_hashes: list[str] = []
    first: torch.Tensor | None = None
    maximum_delta = 0.0
    output: torch.Tensor | None = None
    for _ in range(args.repeat_count):
        output = run_once()
        torch.cuda.synchronize(device)
        repeat_hashes.append(_digest(output))
        if first is None:
            first = output.clone()
        else:
            maximum_delta = max(
                maximum_delta,
                float((output.float() - first.float()).abs().max().item()),
            )
    run_seconds = time.monotonic() - started
    assert output is not None
    if tuple(output.shape) != (args.m, args.n):
        raise RuntimeError(f"unexpected output shape {tuple(output.shape)}")
    if args.output is not None:
        _write_raw(args.output, output)
    finite = bool(torch.isfinite(output).all().item())
    result = {
        "kind": "mxfp8_linear_cross_image",
        "package": package,
        "package_version": package_version,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "expected_m": args.expected_m,
        "source_sha256": source_hash,
        "weight_sha256": weight_hash,
        "weight_scale_sha256": weight_scale_hash,
        "packed_sha256": packed_hashes,
        "output_sha256": _digest(output),
        "repeat_output_sha256": repeat_hashes,
        "repeat_unique_outputs": len(set(repeat_hashes)),
        "repeat_max_abs_delta_from_first": maximum_delta,
        "output_abs_max": float(output.float().abs().max().item()),
        "finite": finite,
        "pack_seconds": pack_seconds,
        "run_seconds": run_seconds,
        "status": "PASS" if finite else "FAIL_NONFINITE",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
