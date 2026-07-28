#!/usr/bin/env python3
"""Compare cross-image W4A16 repeat means against within-image noise."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _load(path: Path, elements: int) -> torch.Tensor:
    raw = path.read_bytes()
    expected = elements * 4
    if len(raw) != expected:
        raise ValueError(f"{path}: {len(raw)} bytes != expected {expected}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    delta = left.double() - right.double()
    absolute = delta.abs()
    return {
        "different": int(torch.count_nonzero(delta).item()),
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "rms": float(torch.sqrt(torch.mean(delta * delta)).item()),
        "signed_mean": float(delta.mean().item()),
        "p99_abs": float(torch.quantile(absolute, 0.99).item()),
        "p999_abs": float(torch.quantile(absolute, 0.999).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1711)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    elements = args.rows * args.hidden
    paths = {
        f"{image}_{part}": args.directory / f"{image}-large-mean.{part}.f32"
        for image in ("v19", "v20")
        for part in ("overall", "first_half", "second_half")
    }
    tensors = {name: _load(path, elements) for name, path in paths.items()}
    comparisons = {
        "within_v19_halves": _metrics(
            tensors["v19_first_half"], tensors["v19_second_half"]
        ),
        "within_v20_halves": _metrics(
            tensors["v20_first_half"], tensors["v20_second_half"]
        ),
        "cross_overall_means": _metrics(
            tensors["v19_overall"], tensors["v20_overall"]
        ),
        "cross_first_halves": _metrics(
            tensors["v19_first_half"], tensors["v20_first_half"]
        ),
        "cross_second_halves": _metrics(
            tensors["v19_second_half"], tensors["v20_second_half"]
        ),
    }
    within_rms = max(
        float(comparisons["within_v19_halves"]["rms"]),
        float(comparisons["within_v20_halves"]["rms"]),
    )
    cross_rms = float(comparisons["cross_overall_means"]["rms"])
    result = {
        "kind": "v19_v20_nf3_moe_repeat_mean_comparison",
        "rows": args.rows,
        "hidden": args.hidden,
        "elements": elements,
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
        "comparisons": comparisons,
        "cross_overall_rms_to_max_within_half_rms": (
            cross_rms / within_rms if within_rms else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
