#!/usr/bin/env python3
"""Compare two complete final-query sparse-selection traces layer by layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from v20_indexer_cross_layer_selection_trace_report import (
    DEFAULT_INDEXER_LAYERS,
)


def _load(root: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in sorted((root / "tp0").glob("layer-*-selected.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if record.get("schema") != "v20-indexer-final-selection-v1":
            raise ValueError(f"{path}: unexpected schema")
        layer = int(record["layer"])
        if layer in records:
            raise ValueError(f"{path}: duplicate layer {layer}")
        if int(record["tp_rank"]) != 0:
            raise ValueError(f"{path}: expected TP rank zero")
        records[layer] = record
    return records


def _ids(record: dict[str, Any]) -> list[int]:
    tensor = record["topk_indices"]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("topk_indices is not a tensor")
    values = [int(value) for value in tensor.flatten().tolist() if value >= 0]
    if len(values) != len(set(values)):
        raise ValueError("topk_indices contains duplicate valid IDs")
    return values


def _quartiles(values: set[int], history_end: int) -> list[int]:
    return [
        sum(
            history_end * quarter // 4
            <= value
            < history_end * (quarter + 1) // 4
            for value in values
        )
        for quarter in range(4)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--needle-min", type=int, required=True)
    parser.add_argument("--needle-max", type=int, required=True)
    parser.add_argument("--expected-batch-tokens", type=int, required=True)
    parser.add_argument("--expected-absolute-position", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = _load(args.reference)
    candidate = _load(args.candidate)
    expected = set(DEFAULT_INDEXER_LAYERS)
    for name, records in (
        (args.reference_name, reference),
        (args.candidate_name, candidate),
    ):
        if set(records) != expected:
            raise ValueError(
                f"{name}: layer coverage {sorted(records)} != "
                f"{sorted(expected)}"
            )

    rows: list[dict[str, Any]] = []
    first_set_divergence: int | None = None
    first_reference_hit_candidate_miss: int | None = None
    first_candidate_hit_reference_miss: int | None = None
    reference_hit_candidate_miss_layers: list[int] = []
    candidate_hit_reference_miss_layers: list[int] = []

    for layer in DEFAULT_INDEXER_LAYERS:
        ref_record = reference[layer]
        cand_record = candidate[layer]
        for name, record in (
            (args.reference_name, ref_record),
            (args.candidate_name, cand_record),
        ):
            if int(record["batch_tokens"]) != args.expected_batch_tokens:
                raise ValueError(f"{name} layer {layer}: batch width mismatch")
            if (
                int(record["absolute_position"])
                != args.expected_absolute_position
            ):
                raise ValueError(
                    f"{name} layer {layer}: absolute position mismatch"
                )

        ref = set(_ids(ref_record))
        cand = set(_ids(cand_record))
        intersection = ref & cand
        union = ref | cand
        ref_hits = sorted(
            value
            for value in ref
            if args.needle_min <= value <= args.needle_max
        )
        cand_hits = sorted(
            value
            for value in cand
            if args.needle_min <= value <= args.needle_max
        )
        if ref != cand and first_set_divergence is None:
            first_set_divergence = layer
        if ref_hits and not cand_hits:
            reference_hit_candidate_miss_layers.append(layer)
            if first_reference_hit_candidate_miss is None:
                first_reference_hit_candidate_miss = layer
        if cand_hits and not ref_hits:
            candidate_hit_reference_miss_layers.append(layer)
            if first_candidate_hit_reference_miss is None:
                first_candidate_hit_reference_miss = layer

        rows.append(
            {
                "layer": layer,
                f"{args.reference_name}_needle_hits": ref_hits,
                f"{args.candidate_name}_needle_hits": cand_hits,
                f"{args.reference_name}_quartiles": _quartiles(
                    ref, args.expected_absolute_position + 1
                ),
                f"{args.candidate_name}_quartiles": _quartiles(
                    cand, args.expected_absolute_position + 1
                ),
                f"{args.reference_name}_only_count": len(ref - cand),
                f"{args.candidate_name}_only_count": len(cand - ref),
                "intersection_count": len(intersection),
                "jaccard": len(intersection) / len(union),
            }
        )

    report = {
        "schema": "v20-cross-layer-selection-comparison-v1",
        "status": "PASS",
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "needle_window": [args.needle_min, args.needle_max],
        "batch_tokens": args.expected_batch_tokens,
        "absolute_position": args.expected_absolute_position,
        "first_set_divergence": first_set_divergence,
        "first_reference_hit_candidate_miss": (
            first_reference_hit_candidate_miss
        ),
        "first_candidate_hit_reference_miss": (
            first_candidate_hit_reference_miss
        ),
        "reference_hit_candidate_miss_layers": (
            reference_hit_candidate_miss_layers
        ),
        "candidate_hit_reference_miss_layers": (
            candidate_hit_reference_miss_layers
        ),
        "rows": rows,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
