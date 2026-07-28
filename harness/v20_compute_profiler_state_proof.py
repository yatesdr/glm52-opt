#!/usr/bin/env python3
"""Run the proven profiler state-machine battery against the v20 port."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILER = (
    ROOT
    / "workspace/vllm-v20-compute-profiler-head"
    / "vllm/model_executor/layers/compute_phase_profiler.py"
)
V19_STATE_PROOF = (
    ROOT
    / "workspace/sol-v19-compute-profiler"
    / "checks/test_profiler_state_machine.py"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=Path, default=DEFAULT_PROFILER)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location(
        "v20_profiler_state_proof",
        V19_STATE_PROOF,
    )
    assert spec is not None and spec.loader is not None
    proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof)
    proof.PROFILER = args.profiler.resolve()

    proof.test_prefill()
    proof.test_missing_prefill_kernel_marker_invalidates_ledger()
    proof.test_forbidden_route_invalidates_ledger()
    proof.test_decode_replay_gate()
    proof.test_decode_rejects_piecewise_then_selects_full_capture()
    proof.test_decode_rows_required()
    proof.test_no_selected_capture_diagnostic()
    proof.test_negative_replay_counter_fails_closed()
    proof.test_fail_closed_parent_order()
    print(
        "PASS v20 profiler state machine: split-tier prefill, one-grid decode, "
        "FULL-capture replay evidence, and fail-closed ledger gates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
