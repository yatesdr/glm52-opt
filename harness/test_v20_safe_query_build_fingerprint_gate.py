#!/usr/bin/env python3
"""CPU-only fail-closed tests for the safe-query build fingerprint gate."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HARNESS = Path(__file__).resolve().parent
GATE = HARNESS / "v20_safe_query_build_fingerprint_gate.py"
REFERENCE = (
    HARNESS
    / "references"
    / "safe_mla_query_bmm_sm120_cu132_pedantic_v1.jsonl"
)
META = (
    HARNESS
    / "references"
    / "safe_mla_query_bmm_sm120_cu132_pedantic_v1.meta.json"
)
SOURCE_COMMIT = "0eb51f992c5d49f494a407b8ae1f785175a977f1"


def _reference_records() -> list[dict[str, object]]:
    records = [
        json.loads(line)
        for line in REFERENCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = [row for row in records if row["kind"] == "safe_query_bmm_fingerprint"]
    cases.append(
        {
            "kind": "summary",
            "status": "PASS",
            "cases": 54,
            "call_mode": "precise",
            "platform_id": "sm120-cu132",
            "compute_capability": [12, 0],
            "torch_cuda": "13.2",
            "gpu_name": "test",
            "stable_libtorch_sha256": "0" * 64,
        }
    )
    return cases


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


class FingerprintGateTest(unittest.TestCase):
    def _run(
        self,
        directory: Path,
        records: list[dict[str, object]],
        *,
        waiver: Path | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        observed = directory / "observed.jsonl"
        report = directory / "report.json"
        _write_jsonl(observed, records)
        command = [
            sys.executable,
            str(GATE),
            "--observed",
            str(observed),
            "--reference",
            str(REFERENCE),
            "--reference-meta",
            str(META),
            "--source-commit",
            SOURCE_COMMIT,
            "--output",
            str(report),
        ]
        if waiver is not None:
            command.extend(("--waiver", str(waiver)))
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        parsed = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
        return result, parsed

    def test_exact_reference_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result, report = self._run(Path(raw), _reference_records())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["fp8_mismatch_count"], 0)

    def test_unexplained_change_fails(self) -> None:
        records = _reference_records()
        records[0]["fp8_sha256"] = "f" * 64
        with tempfile.TemporaryDirectory() as raw:
            result, report = self._run(Path(raw), records)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["status"], "FAIL_UNEXPLAINED_NUMERIC_CHANGE")
        self.assertEqual(report["fp8_mismatch_count"], 1)

    def test_exact_reviewed_waiver_passes(self) -> None:
        records = _reference_records()
        records[0]["fp8_sha256"] = "f" * 64
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            first, failure = self._run(directory, records)
            self.assertEqual(first.returncode, 1)
            waiver = directory / "waiver.json"
            waiver.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "gate_id": failure["gate_id"],
                        "status": "approved",
                        "reference_sha256": failure["reference_sha256"],
                        "intended_change_commit": SOURCE_COMMIT,
                        "author": "author",
                        "reviewed_by": "reviewer",
                        "reviewed_at": "2026-07-25T19:00:00Z",
                        "reason": "Intentional numeric change with quality evidence.",
                        "evidence": ["artifact://quality-and-performance"],
                        "observed_fingerprint_set_sha256": failure[
                            "observed_fingerprint_set_sha256"
                        ],
                        "changed_cases": failure["changed_cases"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            second, report = self._run(directory, records, waiver=waiver)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(report["status"], "PASS_REVIEWED_WAIVER")

    def test_missing_case_fails_closed(self) -> None:
        records = _reference_records()
        del records[0]
        with tempfile.TemporaryDirectory() as raw:
            result, report = self._run(Path(raw), records)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report, {})
        self.assertIn("observed case grid mismatch", result.stderr)

    def test_stale_waiver_commit_fails(self) -> None:
        records = _reference_records()
        records[0]["fp8_sha256"] = "f" * 64
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            first, failure = self._run(directory, records)
            self.assertEqual(first.returncode, 1)
            waiver = directory / "waiver.json"
            waiver.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "gate_id": failure["gate_id"],
                        "status": "approved",
                        "reference_sha256": failure["reference_sha256"],
                        "intended_change_commit": "deadbee",
                        "author": "author",
                        "reviewed_by": "reviewer",
                        "reviewed_at": "2026-07-25T19:00:00Z",
                        "reason": "Intentional numeric change with quality evidence.",
                        "evidence": ["artifact://quality-and-performance"],
                        "observed_fingerprint_set_sha256": failure[
                            "observed_fingerprint_set_sha256"
                        ],
                        "changed_cases": failure["changed_cases"],
                    }
                ),
                encoding="utf-8",
            )
            second, report = self._run(directory, records, waiver=waiver)
        self.assertEqual(second.returncode, 1)
        self.assertEqual(report["status"], "FAIL_INVALID_WAIVER")
        self.assertIn("intended_change_commit mismatch", report["waiver_error"])


if __name__ == "__main__":
    unittest.main()
