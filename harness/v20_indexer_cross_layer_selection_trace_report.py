#!/usr/bin/env python3
"""Validate and summarize a v20 cross-layer sparse-index selection trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_INDEXER_LAYERS = (
    0,
    1,
    2,
    6,
    10,
    14,
    18,
    22,
    26,
    30,
    34,
    38,
    42,
    46,
    50,
    54,
    58,
    62,
    66,
    70,
    74,
)


def _csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_record(path: Path) -> dict[str, Any]:
    record = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(record, dict):
        raise TypeError(f"{path}: expected a dictionary, got {type(record)!r}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument(
        "--expected-layers",
        type=_csv_ints,
        default=DEFAULT_INDEXER_LAYERS,
    )
    parser.add_argument("--needle-min", type=int, required=True)
    parser.add_argument("--needle-max", type=int, required=True)
    parser.add_argument("--expected-batch-tokens", type=int)
    parser.add_argument("--expected-absolute-position", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.needle_min > args.needle_max:
        parser.error("--needle-min must not exceed --needle-max")

    rank_dir = args.trace_dir / "tp0"
    files = sorted(rank_dir.glob("layer-*-selected.pt"))
    if not files:
        raise FileNotFoundError(f"no selection records found in {rank_dir}")

    expected_layers = tuple(args.expected_layers)
    observed_layers: list[int] = []
    rows: list[dict[str, Any]] = []
    positions: set[int] = set()
    batch_widths: set[int] = set()

    for path in files:
        record = _load_record(path)
        if record.get("schema") != "v20-indexer-final-selection-v1":
            raise ValueError(f"{path}: unexpected schema {record.get('schema')!r}")
        if int(record.get("tp_rank", -1)) != 0:
            raise ValueError(f"{path}: expected tp_rank=0")

        layer = int(record["layer"])
        expected_name = f"layer-{layer:03d}-selected.pt"
        if path.name != expected_name:
            raise ValueError(f"{path}: filename does not match layer {layer}")
        observed_layers.append(layer)

        selected = record["topk_indices"]
        if not isinstance(selected, torch.Tensor):
            raise TypeError(f"{path}: topk_indices is not a tensor")
        selected_ids = [int(value) for value in selected.flatten().tolist()]
        unique_valid_ids = {value for value in selected_ids if value >= 0}
        needle_ids = sorted(
            value
            for value in unique_valid_ids
            if args.needle_min <= value <= args.needle_max
        )

        absolute_position = int(record["absolute_position"])
        batch_tokens = int(record["batch_tokens"])
        positions.add(absolute_position)
        batch_widths.add(batch_tokens)
        rows.append(
            {
                "layer": layer,
                "absolute_position": absolute_position,
                "batch_tokens": batch_tokens,
                "buffer_rows": int(record["buffer_rows"]),
                "selected_row": int(record["selected_row"]),
                "topk_entries": len(selected_ids),
                "unique_valid_entries": len(unique_valid_ids),
                "needle_window_hits": needle_ids,
                "needle_window_hit_count": len(needle_ids),
                "sha256": _sha256(path),
            }
        )

    if len(observed_layers) != len(set(observed_layers)):
        raise ValueError(f"duplicate layer records: {observed_layers}")
    if tuple(observed_layers) != expected_layers:
        missing = sorted(set(expected_layers) - set(observed_layers))
        unexpected = sorted(set(observed_layers) - set(expected_layers))
        raise ValueError(
            "trace layer coverage mismatch: "
            f"observed={observed_layers}, missing={missing}, unexpected={unexpected}"
        )
    if len(positions) != 1:
        raise ValueError(f"absolute positions differ across layers: {positions}")
    if len(batch_widths) != 1:
        raise ValueError(f"batch widths differ across layers: {batch_widths}")
    if (
        args.expected_batch_tokens is not None
        and batch_widths != {args.expected_batch_tokens}
    ):
        raise ValueError(
            f"batch width {batch_widths} != expected {args.expected_batch_tokens}"
        )
    if (
        args.expected_absolute_position is not None
        and positions != {args.expected_absolute_position}
    ):
        raise ValueError(
            "absolute position "
            f"{positions} != expected {args.expected_absolute_position}"
        )

    layers_with_hits = [
        row["layer"] for row in rows if row["needle_window_hit_count"]
    ]
    report = {
        "schema": "v20-indexer-cross-layer-selection-report-v1",
        "status": "PASS",
        "trace_dir": str(args.trace_dir),
        "needle_window": [args.needle_min, args.needle_max],
        "expected_layers": list(expected_layers),
        "observed_layers": observed_layers,
        "absolute_position": next(iter(positions)),
        "batch_tokens": next(iter(batch_widths)),
        "layers_with_needle_window_hits": layers_with_hits,
        "first_layer_with_needle_window_hit": (
            layers_with_hits[0] if layers_with_hits else None
        ),
        "last_layer_with_needle_window_hit": (
            layers_with_hits[-1] if layers_with_hits else None
        ),
        "rows": rows,
    }

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
