#!/usr/bin/env python3
"""Validate and summarize fail-closed v20 compute-profiler log records."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path


MARKER = "B12X_COMPUTE_PROF_V20 rank="
ANSI = re.compile(r"\x1b\[[0-9;]*m")
REQUIRED_SCALARS = {
    "rank",
    "mode",
    "calls",
    "rows",
    "route",
    "mean_layer_ms",
    "ledger_valid",
    "ordinal_valid",
    "phase_count_valid",
    "baseline_valid",
    "unaccounted_pct",
    "negative_buckets",
    "tp_attention_bad_layers",
    "tp_moe_bad_layers",
}


def _pairs(value: str) -> dict[str, float]:
    if not value:
        return {}
    result: dict[str, float] = {}
    for item in value.split(","):
        name, raw = item.rsplit(":", 1)
        result[name] = float(raw)
    return result


def _record(line: str) -> dict[str, str] | None:
    line = ANSI.sub("", line)
    position = line.find(MARKER)
    if position < 0:
        return None
    payload = line[position + len("B12X_COMPUTE_PROF_V20 ") :].strip()
    record: dict[str, str] = {}
    for item in payload.split():
        key, separator, value = item.partition("=")
        if separator:
            record[key] = value
    missing = REQUIRED_SCALARS - record.keys()
    assert not missing, f"profiler record missing {sorted(missing)}"
    return record


def _read(path: Path | None) -> str:
    if path is None:
        return sys.stdin.read()
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", type=Path)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--calls", type=int, default=78)
    parser.add_argument("--rows", type=int, default=3072)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally write the validated summary as JSON.",
    )
    args = parser.parse_args()

    records: dict[int, dict[str, str]] = {}
    bodies_seen: set[tuple[tuple[str, str], ...]] = set()
    for line in _read(args.log).splitlines():
        record = _record(line)
        if record is None:
            continue
        body = tuple(sorted(record.items()))
        if body in bodies_seen:
            continue
        bodies_seen.add(body)
        rank = int(record["rank"])
        assert rank not in records, f"multiple distinct records for rank {rank}"
        records[rank] = record

    expected_ranks = set(range(args.world_size))
    assert set(records) == expected_ranks, (
        f"expected ranks {sorted(expected_ranks)}, got {sorted(records)}"
    )

    phase_values: dict[str, list[float]] = {}
    rank_summaries: list[dict[str, object]] = []
    for rank in sorted(records):
        record = records[rank]
        assert record["mode"] == args.mode
        assert int(record["calls"]) == args.calls
        assert int(record["rows"]) == args.rows
        expected_route = (
            "split_tiers" if args.mode == "prefill" else "hybrid_one_grid"
        )
        assert record["route"] == expected_route
        for key in (
            "ledger_valid",
            "ordinal_valid",
            "phase_count_valid",
            "baseline_valid",
        ):
            assert int(record[key]) == 1, f"rank {rank}: {key}=0"
        assert int(record["negative_buckets"]) == 0
        assert int(record["tp_attention_bad_layers"]) == 0
        assert int(record["tp_moe_bad_layers"]) == 0
        assert abs(float(record["unaccounted_pct"])) <= 2.0

        exclusive = _pairs(record["exclusive_ms"])
        for phase, value in exclusive.items():
            phase_values.setdefault(phase, []).append(value)
        rank_summaries.append(
            {
                "rank": rank,
                "mean_layer_ms": float(record["mean_layer_ms"]),
                "unaccounted_pct": float(record["unaccounted_pct"]),
                "top_exclusive_ms": _pairs(record["top_exclusive_ms"]),
                "exclusive_ms": exclusive,
                "aux_overlap_ms": _pairs(record["aux_overlap_ms"]),
                "tp_attention_bytes": int(record["tp_attention_bytes"]),
                "tp_moe_bytes": int(record["tp_moe_bytes"]),
                "dcp_lse_bytes": int(record["dcp_lse_bytes"]),
                "dcp_rs_bytes": int(record["dcp_rs_bytes"]),
                "dcp_a2a_bytes": int(record["dcp_a2a_bytes"]),
            }
        )

    phase_summary = []
    for phase, values in phase_values.items():
        assert len(values) == args.world_size
        mean = statistics.fmean(values)
        phase_summary.append(
            {
                "phase": phase,
                "mean_ms_per_layer": mean,
                "min_ms_per_layer": min(values),
                "max_ms_per_layer": max(values),
                "rank_spread_pct": (
                    100.0 * (max(values) - min(values)) / mean if mean else 0.0
                ),
            }
        )
    phase_summary.sort(
        key=lambda item: float(item["mean_ms_per_layer"]),
        reverse=True,
    )

    summary = {
        "verdict": "PASS",
        "mode": args.mode,
        "calls": args.calls,
        "rows": args.rows,
        "world_size": args.world_size,
        "mean_layer_ms": statistics.fmean(
            float(record["mean_layer_ms"]) for record in records.values()
        ),
        "rank_records": rank_summaries,
        "phases": phase_summary,
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
