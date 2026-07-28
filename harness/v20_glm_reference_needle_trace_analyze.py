#!/usr/bin/env python3
"""Summarize GLM reference needle-selection JSON records from container logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


MARKER = "GLM_REFERENCE_NEEDLE_TRACE "
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def layer_of(prefix: str) -> int:
    match = LAYER_RE.search(prefix)
    if match is None:
        raise ValueError(f"cache prefix has no layer index: {prefix!r}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--needle-start", type=int, required=True)
    parser.add_argument("--needle-end", type=int, required=True)
    parser.add_argument("--min-local-seq-len", type=int, default=0)
    parser.add_argument("--max-local-seq-len", type=int)
    args = parser.parse_args()

    raw = args.log.read_bytes()
    parsed_records: list[dict[str, object]] = []
    marked_lines = 0
    for line_number, line in enumerate(
        raw.decode(errors="replace").splitlines(), start=1
    ):
        if MARKER not in line:
            continue
        marked_lines += 1
        payload = json.loads(line.split(MARKER, 1)[1])
        if payload.get("schema") != "glm-reference-needle-selection-trace-v1":
            raise ValueError(f"unexpected trace schema on line {line_number}")
        if payload.get("dcp_rank") != 0:
            raise ValueError(f"unexpected traced DCP rank on line {line_number}")
        if (
            payload.get("needle_start") != args.needle_start
            or payload.get("needle_end") != args.needle_end
        ):
            raise ValueError(f"needle range drift on line {line_number}")
        payload["layer"] = layer_of(str(payload["cache_prefix"]))
        payload["line_number"] = line_number
        parsed_records.append(payload)

    if marked_lines == 0 or not parsed_records:
        raise ValueError("no needle-selection trace records found")
    if len(parsed_records) != marked_lines:
        raise ValueError("not every marked trace line produced one record")
    records = [
        record
        for record in parsed_records
        if int(record["local_seq_len"]) >= args.min_local_seq_len
        and (
            args.max_local_seq_len is None
            or int(record["local_seq_len"]) <= args.max_local_seq_len
        )
    ]
    if not records:
        raise ValueError(
            "no trace records survived the local-sequence-length filter"
        )

    layers = sorted({int(record["layer"]) for record in records})
    decode_calls = sorted({int(record["decode_call"]) for record in records})
    rows = sorted({int(record["row"]) for record in records})
    if rows != [0]:
        raise ValueError(f"expected one MTP0 decode row, got rows={rows}")

    expected_pairs = {
        (layer, decode_call) for layer in layers for decode_call in decode_calls
    }
    observed_pairs = {
        (int(record["layer"]), int(record["decode_call"])) for record in records
    }
    missing_pairs = sorted(expected_pairs - observed_pairs)
    duplicate_pairs = len(observed_pairs) != len(records)
    if missing_pairs or duplicate_pairs:
        raise ValueError(
            "trace layer/decode matrix incomplete: "
            f"missing={missing_pairs[:8]} duplicates={duplicate_pairs}"
        )

    by_layer: dict[str, dict[str, object]] = {}
    for layer in layers:
        selected = [
            record for record in records if int(record["layer"]) == layer
        ]
        nearest_distances = [
            int(record["nearest"][0]["distance"])
            for record in selected
            if record.get("nearest")
        ]
        by_layer[str(layer)] = {
            "decode_calls": len(selected),
            "exact_hit_calls": [
                int(record["decode_call"])
                for record in selected
                if bool(record["exact_hit"])
            ],
            "context_hit_calls": [
                int(record["decode_call"])
                for record in selected
                if bool(record["context_hit"])
            ],
            "minimum_nearest_distance": (
                min(nearest_distances) if nearest_distances else None
            ),
        }

    by_decode_call: dict[str, dict[str, object]] = {}
    for decode_call in decode_calls:
        selected = [
            record
            for record in records
            if int(record["decode_call"]) == decode_call
        ]
        nearest_distances = [
            int(record["nearest"][0]["distance"])
            for record in selected
            if record.get("nearest")
        ]
        by_decode_call[str(decode_call)] = {
            "exact_hit_layers": [
                int(record["layer"])
                for record in selected
                if bool(record["exact_hit"])
            ],
            "context_hit_layers": [
                int(record["layer"])
                for record in selected
                if bool(record["context_hit"])
            ],
            "minimum_nearest_distance": (
                min(nearest_distances) if nearest_distances else None
            ),
        }

    report = {
        "schema": "v20-glm-reference-needle-trace-analysis-v1",
        "status": "PASS",
        "input_log": str(args.log),
        "input_log_sha256": hashlib.sha256(raw).hexdigest(),
        "needle_start": args.needle_start,
        "needle_end": args.needle_end,
        "min_local_seq_len": args.min_local_seq_len,
        "max_local_seq_len": args.max_local_seq_len,
        "parsed_record_count": len(parsed_records),
        "record_count": len(records),
        "layers": layers,
        "decode_calls": decode_calls,
        "exact_hit_record_count": sum(
            bool(record["exact_hit"]) for record in records
        ),
        "context_hit_record_count": sum(
            bool(record["context_hit"]) for record in records
        ),
        "by_layer": by_layer,
        "by_decode_call": by_decode_call,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
