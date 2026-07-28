#!/usr/bin/env python3
"""Compare current, PEDANTIC, and accurate-reduction fingerprints."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


KEY_FIELDS = ("tokens", "heads", "seed", "q_scale")
REFERENCE_FIELDS = ("reference_bf16_sha256", "reference_fp8_sha256")
OUTPUT_FIELDS = ("bf16_sha256", "fp8_sha256")


def _load(
    path: Path,
    *,
    expected_mode: str,
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], str]:
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    summary = None
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        record = json.loads(line)
        kind = record.get("kind")
        if kind == "summary":
            summary = record
            continue
        if kind != "safe_query_reduction_equivalence_case":
            raise AssertionError(f"{path}:{line_number}: unexpected kind {kind!r}")
        if record.get("mode") != expected_mode:
            raise AssertionError(
                f"{path}:{line_number}: expected mode {expected_mode!r}, "
                f"got {record.get('mode')!r}"
            )
        key = tuple(record[field] for field in KEY_FIELDS)
        if key in records:
            raise AssertionError(f"{path}:{line_number}: duplicate key {key}")
        for prefix in ("bf16", "fp8"):
            graph = record[f"graph_{prefix}_sha256"]
            if graph is not None and graph != record[f"{prefix}_sha256"]:
                raise AssertionError(f"{path}:{line_number}: graph mismatch")
        records[key] = record
    assert summary is not None and summary.get("status") == "PASS", path
    assert summary.get("mode") == expected_mode, (path, summary)
    assert summary.get("cases") == len(records), (path, summary, len(records))
    stable_sha256 = summary.get("stable_libtorch_sha256", "")
    assert len(stable_sha256) == 64, (path, summary)
    return records, stable_sha256


def _same_count(
    left: dict[tuple[Any, ...], dict[str, Any]],
    right: dict[tuple[Any, ...], dict[str, Any]],
    field: str,
    *,
    tokens: int | None = None,
) -> int:
    return sum(
        left[key][field] == right[key][field]
        for key in left
        if tokens is None or key[0] == tokens
    )


def _mean_timing(
    records: dict[tuple[Any, ...], dict[str, Any]],
    field: str,
    *,
    tokens: int,
) -> float:
    return statistics.mean(
        float(record[field])
        for key, record in records.items()
        if key[0] == tokens
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--pedantic", required=True, type=Path)
    parser.add_argument("--accurate-regular", required=True, type=Path)
    parser.add_argument("--accurate-precise", required=True, type=Path)
    parser.add_argument(
        "--max-regular-overhead",
        type=float,
        default=1.25,
        help="maximum accurate-precise/current M=3072 pipeline ratio",
    )
    parser.add_argument(
        "--max-pedantic-ratio",
        type=float,
        default=0.90,
        help="maximum accurate-precise/PEDANTIC M=3072 pipeline ratio",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current, current_sha256 = _load(
        args.current,
        expected_mode="legacy-current",
    )
    pedantic, pedantic_sha256 = _load(
        args.pedantic,
        expected_mode="legacy-pedantic",
    )
    accurate_regular, accurate_regular_sha256 = _load(
        args.accurate_regular,
        expected_mode="accurate-regular",
    )
    accurate_precise, accurate_precise_sha256 = _load(
        args.accurate_precise,
        expected_mode="accurate-precise",
    )
    assert accurate_regular_sha256 == accurate_precise_sha256, (
        accurate_regular_sha256,
        accurate_precise_sha256,
    )
    sets = [
        set(records)
        for records in (current, pedantic, accurate_regular, accurate_precise)
    ]
    assert all(keys == sets[0] for keys in sets[1:]), "case-key drift"
    cases = len(current)
    m3072_cases = sum(key[0] == 3072 for key in current)
    assert m3072_cases > 0

    reference_stable = all(
        _same_count(current, records, field) == cases
        for records in (pedantic, accurate_regular, accurate_precise)
        for field in REFERENCE_FIELDS
    )
    regular_parity = {
        field: _same_count(current, accurate_regular, field)
        for field in OUTPUT_FIELDS
    }
    precise_pedantic_parity = {
        field: _same_count(pedantic, accurate_precise, field)
        for field in OUTPUT_FIELDS
    }
    current_pedantic_m3072_changes = {
        field: m3072_cases
        - _same_count(current, pedantic, field, tokens=3072)
        for field in OUTPUT_FIELDS
    }
    precise_no_worse_than_current_cases = sum(
        accurate_precise[key]["reference_max_abs_error"]
        <= current[key]["reference_max_abs_error"]
        for key in current
    )
    precise_no_worse_than_pedantic_cases = sum(
        accurate_precise[key]["reference_max_abs_error"]
        <= pedantic[key]["reference_max_abs_error"]
        for key in current
    )

    timing = {}
    for field in ("bmm_ms", "pipeline_ms"):
        values = {
            "current": _mean_timing(current, field, tokens=3072),
            "pedantic": _mean_timing(pedantic, field, tokens=3072),
            "accurate_regular": _mean_timing(
                accurate_regular,
                field,
                tokens=3072,
            ),
            "accurate_precise": _mean_timing(
                accurate_precise,
                field,
                tokens=3072,
            ),
        }
        values["accurate_precise_over_current"] = (
            values["accurate_precise"] / values["current"]
        )
        values["accurate_precise_over_pedantic"] = (
            values["accurate_precise"] / values["pedantic"]
        )
        timing[field] = values

    fp8_equivalent = precise_pedantic_parity["fp8_sha256"] == cases
    bf16_bit_identical = precise_pedantic_parity["bf16_sha256"] == cases
    numeric_pass = (
        reference_stable
        and regular_parity["bf16_sha256"] == cases
        and regular_parity["fp8_sha256"] == cases
        and fp8_equivalent
        and current_pedantic_m3072_changes["bf16_sha256"] == m3072_cases
        and current_pedantic_m3072_changes["fp8_sha256"] == m3072_cases
    )
    performance_pass = (
        timing["pipeline_ms"]["accurate_precise_over_current"]
        <= args.max_regular_overhead
        and timing["pipeline_ms"]["accurate_precise_over_pedantic"]
        <= args.max_pedantic_ratio
    )
    status = "PASS" if numeric_pass and performance_pass else "FAIL"
    if numeric_pass:
        equivalence = (
            "BF16_BIT_IDENTICAL"
            if bf16_bit_identical
            else "MODEL_BOUNDARY_FP8_EQUIVALENT"
        )
    else:
        equivalence = "NOT_EQUIVALENT"

    result = {
        "kind": "safe_query_reduction_equivalence_comparison",
        "cases": cases,
        "m3072_cases": m3072_cases,
        "reference_stable": reference_stable,
        "accurate_regular_current_parity": regular_parity,
        "accurate_precise_pedantic_parity": precise_pedantic_parity,
        "current_pedantic_m3072_changes": current_pedantic_m3072_changes,
        # PyTorch's FP32 bmm is an environmental-stability reference, not a
        # bitwise reduction-order oracle for cuBLAS. PEDANTIC itself can
        # differ from it at qualified widths. Report error dominance, but gate
        # on exact PEDANTIC equivalence at the post-FP8 model boundary.
        "accurate_precise_reference_no_worse_than_current_cases": (
            precise_no_worse_than_current_cases
        ),
        "accurate_precise_reference_no_worse_than_pedantic_cases": (
            precise_no_worse_than_pedantic_cases
        ),
        "equivalence": equivalence,
        "numeric_pass": numeric_pass,
        "performance_thresholds": {
            "max_accurate_precise_over_current": args.max_regular_overhead,
            "max_accurate_precise_over_pedantic": args.max_pedantic_ratio,
        },
        "performance_pass": performance_pass,
        "m3072_timing": timing,
        "stable_libtorch_sha256": {
            "current": current_sha256,
            "pedantic": pedantic_sha256,
            "accurate": accurate_regular_sha256,
        },
        "status": status,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
