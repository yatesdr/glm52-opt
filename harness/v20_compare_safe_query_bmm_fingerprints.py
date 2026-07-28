#!/usr/bin/env python3
"""Compare two v20 safe-query-BMM fingerprint JSONL files.

This comparator is dependency-free so it can run on the CN4 host after the
GPU probe has been executed once in each container image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEY_FIELDS = ("tokens", "heads", "seed", "q_scale")
DIGEST_FIELDS = (
    "bf16_sha256",
    "fp8_sha256",
    "retrieval_ids_sha256",
    "reference_bf16_sha256",
)


def _load(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    summary = None
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        record = json.loads(line)
        kind = record.get("kind")
        if kind == "summary":
            summary = record
            continue
        if kind != "safe_query_bmm_fingerprint":
            raise AssertionError(f"{path}:{line_number}: unexpected kind {kind!r}")
        key = tuple(record[field] for field in KEY_FIELDS)
        if key in records:
            raise AssertionError(f"{path}:{line_number}: duplicate key {key}")
        records[key] = record
    assert summary is not None and summary.get("status") == "PASS", path
    assert summary.get("cases") == len(records), (path, summary, len(records))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path, help="pre-992/PEDANTIC JSONL")
    parser.add_argument("new", type=Path, help="post-992/regular-FP32 JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old = _load(args.old)
    new = _load(args.new)
    assert old.keys() == new.keys(), (
        sorted(old.keys() - new.keys()),
        sorted(new.keys() - old.keys()),
    )

    counts = {field: 0 for field in DIGEST_FIELDS}
    changed_cases = []
    for key in sorted(old):
        changed = [
            field for field in DIGEST_FIELDS if old[key][field] != new[key][field]
        ]
        for field in changed:
            counts[field] += 1
        if changed:
            changed_cases.append({"key": key, "changed": changed})

    reference_stable = counts["reference_bf16_sha256"] == 0
    operator_changed = counts["bf16_sha256"] > 0
    result = {
        "kind": "safe_query_bmm_cross_image_comparison",
        "cases": len(old),
        "changed_cases": len(changed_cases),
        "digest_change_counts": counts,
        "reference_stable": reference_stable,
        "operator_changed": operator_changed,
        "post_quant_changed": counts["fp8_sha256"] > 0,
        "retrieval_ids_changed": counts["retrieval_ids_sha256"] > 0,
        "verdict": (
            "CANDIDATE_SUPPORTED"
            if reference_stable and operator_changed
            else "CANDIDATE_EXONERATED"
            if reference_stable
            else "INVALID_ENVIRONMENTAL_DRIFT"
        ),
        "status": "PASS" if reference_stable else "FAIL",
        "examples": changed_cases[:12],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
