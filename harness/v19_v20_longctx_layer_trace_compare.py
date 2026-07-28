#!/usr/bin/env python3
"""Compare the frozen final-query layer trace from exact v19 and v20 images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SCHEMA = "v20-longctx-first-divergence-v1"
STAGES = ("input", "attention", "mlp_input", "mlp_output")


def tensor_sha256(tensor: torch.Tensor | None) -> str | None:
    if tensor is None:
        return None
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_trace(root: Path) -> dict[tuple[int, str, int], dict[str, Any]]:
    rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    for path in sorted(root.glob("tp*/layer??-*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record.get("schema") != SCHEMA:
            raise RuntimeError(f"{path}: unexpected schema {record.get('schema')!r}")
        key = (
            int(record["layer"]),
            str(record["stage"]),
            int(record["tp_rank"]),
        )
        if key in rows:
            raise RuntimeError(f"duplicate trace key {key}: {path}")
        rows[key] = record
    if not rows:
        raise RuntimeError(f"no trace records found under {root}")
    return rows


def validate_trace(
    name: str,
    rows: dict[tuple[int, str, int], dict[str, Any]],
    *,
    layers: int,
    ranks: int,
    expected_position: int,
) -> dict[str, Any]:
    expected = {
        (layer, stage, rank)
        for layer in range(layers)
        for stage in STAGES
        for rank in range(ranks)
    }
    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    if missing or extra:
        raise RuntimeError(
            f"{name}: trace key mismatch missing={missing[:8]} extra={extra[:8]}"
        )

    wrong_positions = sorted(
        (key, int(record["absolute_position"]))
        for key, record in rows.items()
        if int(record["absolute_position"]) != expected_position
    )
    if wrong_positions:
        raise RuntimeError(f"{name}: wrong absolute positions: {wrong_positions[:8]}")

    rank_checks: list[dict[str, Any]] = []
    rank_consistent = True
    for layer in range(layers):
        for stage in STAGES:
            records = [rows[(layer, stage, rank)] for rank in range(ranks)]
            for field in ("hidden", "residual", "topk_indices"):
                hashes = [tensor_sha256(record.get(field)) for record in records]
                present = [value is not None for value in hashes]
                if len(set(present)) != 1:
                    raise RuntimeError(
                        f"{name}: {layer}/{stage}/{field} presence differs by rank"
                    )
                if not any(present):
                    continue
                exact = len(set(hashes)) == 1
                rank_consistent &= exact
                rank_checks.append(
                    {
                        "layer": layer,
                        "stage": stage,
                        "field": field,
                        "exact": exact,
                        "hashes": hashes,
                    }
                )
    return {
        "name": name,
        "records": len(rows),
        "rank_consistent": rank_consistent,
        "rank_checks": rank_checks,
    }


def numeric_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.shape != b.shape or a.dtype != b.dtype:
        raise RuntimeError(
            f"tensor contract differs: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
        )
    exact = torch.equal(a, b)
    af = a.float()
    bf = b.float()
    delta = af - bf
    denom = float(torch.linalg.vector_norm(af).item())
    rel_l2 = float(torch.linalg.vector_norm(delta).item()) / max(denom, 1e-30)
    cosine = float(
        torch.nn.functional.cosine_similarity(af.flatten(), bf.flatten(), dim=0).item()
    )
    return {
        "exact": exact,
        "a_sha256": tensor_sha256(a),
        "b_sha256": tensor_sha256(b),
        "changed": int(torch.count_nonzero(a != b).item()),
        "numel": a.numel(),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rel_l2": rel_l2,
        "cosine": cosine,
    }


def residual_stream(record: dict[str, Any]) -> torch.Tensor:
    hidden = record["hidden"].float()
    residual = record.get("residual")
    return hidden if residual is None else hidden + residual.float()


def topk_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    needle_start: int,
    needle_end: int,
) -> dict[str, Any]:
    if a.shape != b.shape:
        raise RuntimeError(f"top-k shapes differ: {a.shape} vs {b.shape}")
    a_valid = {int(value) for value in a.tolist() if int(value) >= 0}
    b_valid = {int(value) for value in b.tolist() if int(value) >= 0}
    union = a_valid | b_valid
    intersection = a_valid & b_valid
    return {
        "exact": torch.equal(a, b),
        "a_sha256": tensor_sha256(a),
        "b_sha256": tensor_sha256(b),
        "a_valid": len(a_valid),
        "b_valid": len(b_valid),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": 1.0 if not union else len(intersection) / len(union),
        "a_needle_hits": sorted(
            value for value in a_valid if needle_start <= value <= needle_end
        ),
        "b_needle_hits": sorted(
            value for value in b_valid if needle_start <= value <= needle_end
        ),
        "only_a": sorted(a_valid - b_valid),
        "only_b": sorted(b_valid - a_valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v19", type=Path, required=True)
    parser.add_argument("--v20", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=78)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--expected-position", type=int, default=343726)
    parser.add_argument("--needle-start", type=int, default=137470)
    parser.add_argument("--needle-end", type=int, default=137540)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    v19 = load_trace(args.v19)
    v20 = load_trace(args.v20)
    v19_validation = validate_trace(
        "v19",
        v19,
        layers=args.layers,
        ranks=args.ranks,
        expected_position=args.expected_position,
    )
    v20_validation = validate_trace(
        "v20",
        v20,
        layers=args.layers,
        ranks=args.ranks,
        expected_position=args.expected_position,
    )

    comparisons: list[dict[str, Any]] = []
    first_nonexact: dict[str, dict[str, int | str] | None] = {
        "hidden": None,
        "residual_stream": None,
        "topk": None,
    }
    first_needle_regression: dict[str, Any] | None = None
    for layer in range(args.layers):
        for stage in STAGES:
            old = v19[(layer, stage, 0)]
            new = v20[(layer, stage, 0)]
            hidden = numeric_metrics(old["hidden"], new["hidden"])
            stream = numeric_metrics(residual_stream(old), residual_stream(new))
            topk = None
            if old.get("topk_indices") is not None or new.get("topk_indices") is not None:
                if old.get("topk_indices") is None or new.get("topk_indices") is None:
                    raise RuntimeError(
                        f"top-k presence differs at layer={layer} stage={stage}"
                    )
                topk = topk_metrics(
                    old["topk_indices"],
                    new["topk_indices"],
                    needle_start=args.needle_start,
                    needle_end=args.needle_end,
                )
            location = {"layer": layer, "stage": stage}
            if not hidden["exact"] and first_nonexact["hidden"] is None:
                first_nonexact["hidden"] = location
            if not stream["exact"] and first_nonexact["residual_stream"] is None:
                first_nonexact["residual_stream"] = location
            if topk is not None:
                if not topk["exact"] and first_nonexact["topk"] is None:
                    first_nonexact["topk"] = location
                if (
                    first_needle_regression is None
                    and topk["a_needle_hits"]
                    and not topk["b_needle_hits"]
                ):
                    first_needle_regression = {
                        **location,
                        "v19_needle_hits": topk["a_needle_hits"],
                        "v20_needle_hits": topk["b_needle_hits"],
                        "jaccard": topk["jaccard"],
                    }
            comparisons.append(
                {
                    "layer": layer,
                    "stage": stage,
                    "hidden": hidden,
                    "residual_stream": stream,
                    "topk": topk,
                }
            )

    report = {
        "schema": "v19-v20-longctx-layer-trace-comparison-v2",
        "v19": v19_validation,
        "v20": v20_validation,
        "contract": {
            "layers": args.layers,
            "ranks": args.ranks,
            "expected_position": args.expected_position,
            "needle_range": [args.needle_start, args.needle_end],
        },
        "first_nonexact_location": first_nonexact,
        "first_needle_regression": first_needle_regression,
        "comparisons": comparisons,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
