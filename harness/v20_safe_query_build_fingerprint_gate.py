#!/usr/bin/env python3
"""Fail closed on an unexplained post-FP8 safe-query numeric change."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


CASE_KIND = "safe_query_bmm_fingerprint"
SUMMARY_KIND = "summary"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            kind = record.get("kind")
            if kind == CASE_KIND:
                rows.append(record)
            elif kind == SUMMARY_KIND:
                summaries.append(record)
            else:
                raise ValueError(f"{path}:{line_number}: unexpected kind={kind!r}")
    if len(summaries) != 1:
        raise ValueError(f"{path}: expected exactly one summary, got {len(summaries)}")
    return rows, summaries[0]


def _key(row: dict[str, Any]) -> tuple[int, int, int, float]:
    return (
        int(row["tokens"]),
        int(row["heads"]),
        int(row["seed"]),
        float(row["q_scale"]),
    )


def _index(
    rows: list[dict[str, Any]], *, label: str
) -> dict[tuple[int, int, int, float], dict[str, Any]]:
    indexed: dict[tuple[int, int, int, float], dict[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if key in indexed:
            raise ValueError(f"{label}: duplicate case {key}")
        for field in ("bf16_sha256", "fp8_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{label}: invalid {field} for {key}: {value!r}")
        indexed[key] = row
    return indexed


def _expected_keys(meta: dict[str, Any]) -> set[tuple[int, int, int, float]]:
    geometry = meta["geometry"]
    return {
        (int(tokens), int(geometry["heads"]), int(seed), float(scale))
        for tokens in geometry["tokens"]
        for seed in geometry["seeds"]
        for scale in geometry["q_scales"]
    }


def _fingerprint_set_sha256(
    rows: dict[tuple[int, int, int, float], dict[str, Any]]
) -> str:
    normalized = [
        {
            "tokens": key[0],
            "heads": key[1],
            "seed": key[2],
            "q_scale": key[3],
            "fp8_sha256": rows[key]["fp8_sha256"],
        }
        for key in sorted(rows)
    ]
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _case_delta(
    key: tuple[int, int, int, float],
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tokens": key[0],
        "heads": key[1],
        "seed": key[2],
        "q_scale": key[3],
        "expected_fp8_sha256": expected["fp8_sha256"],
        "observed_fp8_sha256": observed["fp8_sha256"],
    }


def _validate_waiver(
    *,
    waiver_path: Path,
    gate_id: str,
    reference_sha256: str,
    source_commit: str,
    observed_set_sha256: str,
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
    required_equal = {
        "schema_version": 1,
        "gate_id": gate_id,
        "status": "approved",
        "reference_sha256": reference_sha256,
        "intended_change_commit": source_commit,
        "observed_fingerprint_set_sha256": observed_set_sha256,
    }
    for field, expected in required_equal.items():
        if waiver.get(field) != expected:
            raise ValueError(
                f"waiver {field} mismatch: expected={expected!r} "
                f"observed={waiver.get(field)!r}"
            )
    author = waiver.get("author")
    reviewer = waiver.get("reviewed_by")
    if not isinstance(author, str) or not author.strip():
        raise ValueError("waiver author is required")
    if not isinstance(reviewer, str) or not reviewer.strip() or reviewer == author:
        raise ValueError("waiver requires an independent reviewed_by identity")
    reviewed_at = waiver.get("reviewed_at")
    if not isinstance(reviewed_at, str) or not reviewed_at.endswith("Z"):
        raise ValueError("waiver reviewed_at must be an ISO-8601 UTC timestamp")
    datetime.fromisoformat(reviewed_at.removesuffix("Z") + "+00:00")
    reason = waiver.get("reason")
    if not isinstance(reason, str) or len(reason.strip()) < 20:
        raise ValueError("waiver reason must explain the intended numeric change")
    evidence = waiver.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(value, str) and value.strip() for value in evidence)
    ):
        raise ValueError("waiver evidence must contain at least one record")
    if waiver.get("changed_cases") != mismatches:
        raise ValueError(
            "waiver changed_cases must exactly enumerate every observed mismatch"
        )
    return waiver


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-meta", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--waiver", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not COMMIT_RE.fullmatch(args.source_commit):
        raise SystemExit("--source-commit must be an exact 40-character git SHA")
    meta = json.loads(args.reference_meta.read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1:
        raise SystemExit("unsupported reference metadata schema")
    reference_sha256 = _sha256(args.reference)
    if reference_sha256 != meta.get("reference_sha256"):
        raise SystemExit(
            "reference byte pin mismatch: "
            f"expected={meta.get('reference_sha256')} observed={reference_sha256}"
        )

    reference_rows, reference_summary = _load_jsonl(args.reference)
    observed_rows, observed_summary = _load_jsonl(args.observed)
    reference = _index(reference_rows, label="reference")
    observed = _index(observed_rows, label="observed")
    expected_keys = _expected_keys(meta)
    if set(reference) != expected_keys:
        raise SystemExit(
            f"reference case grid mismatch: expected={len(expected_keys)} "
            f"observed={len(reference)}"
        )
    if set(observed) != expected_keys:
        missing = sorted(expected_keys - set(observed))
        extra = sorted(set(observed) - expected_keys)
        raise SystemExit(f"observed case grid mismatch: missing={missing} extra={extra}")
    expected_count = int(meta["geometry"]["cases"])
    if (
        reference_summary.get("status") != "PASS"
        or int(reference_summary.get("cases", -1)) != expected_count
    ):
        raise SystemExit("reference summary is not PASS/54")
    if (
        observed_summary.get("status") != "PASS"
        or int(observed_summary.get("cases", -1)) != expected_count
    ):
        raise SystemExit("observed summary is not PASS/54")
    platform = meta["platform"]
    if observed_summary.get("platform_id") != platform["id"]:
        raise SystemExit("observed platform_id does not match reference")
    if observed_summary.get("compute_capability") != platform["compute_capability"]:
        raise SystemExit("observed compute capability does not match reference")
    if observed_summary.get("torch_cuda") != platform["torch_cuda"]:
        raise SystemExit("observed torch CUDA version does not match reference")
    if observed_summary.get("call_mode") != "precise":
        raise SystemExit("build gate requires the production precise call mode")

    fp8_mismatch_keys = [
        key
        for key in sorted(expected_keys)
        if reference[key]["fp8_sha256"] != observed[key]["fp8_sha256"]
    ]
    bf16_mismatch_count = sum(
        reference[key]["bf16_sha256"] != observed[key]["bf16_sha256"]
        for key in expected_keys
    )
    mismatches = [
        _case_delta(key, reference[key], observed[key]) for key in fp8_mismatch_keys
    ]
    observed_set_sha256 = _fingerprint_set_sha256(observed)
    report: dict[str, Any] = {
        "kind": "safe_query_build_fingerprint_gate",
        "gate_id": meta["gate_id"],
        "source_commit": args.source_commit,
        "reference_sha256": reference_sha256,
        "observed_fingerprint_set_sha256": observed_set_sha256,
        "cases": len(observed),
        "bf16_mismatch_count_diagnostic": bf16_mismatch_count,
        "fp8_mismatch_count": len(mismatches),
        "changed_cases": mismatches,
        "waiver": None,
    }
    if not mismatches:
        report["status"] = "PASS"
        _write_report(args.output, report)
        return
    if args.waiver is None:
        report["status"] = "FAIL_UNEXPLAINED_NUMERIC_CHANGE"
        _write_report(args.output, report)
        raise SystemExit(1)
    try:
        waiver = _validate_waiver(
            waiver_path=args.waiver,
            gate_id=meta["gate_id"],
            reference_sha256=reference_sha256,
            source_commit=args.source_commit,
            observed_set_sha256=observed_set_sha256,
            mismatches=mismatches,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        report["status"] = "FAIL_INVALID_WAIVER"
        report["waiver_error"] = str(error)
        _write_report(args.output, report)
        raise SystemExit(1) from error
    report["status"] = "PASS_REVIEWED_WAIVER"
    report["waiver"] = {
        "path": str(args.waiver),
        "intended_change_commit": waiver["intended_change_commit"],
        "reviewed_by": waiver["reviewed_by"],
        "reviewed_at": waiver["reviewed_at"],
    }
    _write_report(args.output, report)


if __name__ == "__main__":
    main()
