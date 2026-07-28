#!/usr/bin/env python3
"""Validate and summarize the matched GLM-5.2 EXL3-TR3 KLD runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics


EXPECTED_FIRST16 = [
    284,
    8396,
    425,
    10960,
    465,
    284,
    14721,
    8396,
    425,
    10960,
    465,
    374,
    458,
    6364,
    4531,
    1154,
]
EXPECTED_POSITIONS = 2047
NF3_DYNAMIC_VALUES = [
    0.13999698092705107,
    0.14038452243547964,
    0.13672544879717802,
]
INVARIANT_KEYS = (
    "image",
    "image_id",
    "model_dir",
    "model_revision",
    "reference_logits_sha256",
    "reference_manifest_sha256",
    "runner_sha256",
    "context_length",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "kv_cache_dtype",
    "quantization",
    "exl3_trellis_min_m",
    "kv_fp8_rope",
    "dynamic_per_token",
    "tensor_parallel_size",
    "decode_context_parallel_size",
    "enforce_eager",
    "selector_policy",
    "token_first16",
    "total_positions",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def single_marker(text: str, marker: str) -> dict:
    matches = re.findall(rf"(?m)^{re.escape(marker)}\s+(\{{.*\}})\s*$", text)
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one {marker!r} marker, found {len(matches)}"
        )
    return json.loads(matches[0])


def validate(run_dir: pathlib.Path) -> dict:
    log_path = run_dir / "prefill_dcp1.log"
    config_path = run_dir / "config.json"
    compile_path = run_dir / "writer-compile-proof.json"
    for path in (log_path, config_path, compile_path):
        if not path.is_file():
            raise SystemExit(f"missing required artifact: {path}")

    text = log_path.read_text(errors="replace")
    if "Traceback (most recent call last)" in text:
        raise SystemExit(f"traceback present in {log_path}")
    if re.search(r"(?i)(out of memory|oom-kill|engine core failed)", text):
        raise SystemExit(f"fatal engine signature present in {log_path}")
    if "fallback_prefill_kld_cleanup_done" not in text:
        raise SystemExit(f"clean runner shutdown marker missing in {log_path}")

    tokenized = single_marker(text, "tokenized")
    done = single_marker(text, "fallback_prefill_kld_done")
    if tokenized.get("first16") != EXPECTED_FIRST16:
        raise SystemExit(f"reference-token mismatch in {run_dir}")
    if int(done.get("total_positions", -1)) != EXPECTED_POSITIONS:
        raise SystemExit(f"position-count mismatch in {run_dir}")
    mean_kld = float(done.get("mean_kld", math.nan))
    if not math.isfinite(mean_kld) or mean_kld < 0:
        raise SystemExit(f"invalid KLD in {run_dir}: {mean_kld!r}")

    config = json.loads(config_path.read_text())
    if config.get("quantization") != "exl3":
        raise SystemExit(f"wrong quantization in {config_path}")
    if config.get("dynamic_per_token") is not True:
        raise SystemExit(f"dynamic mode not pinned in {config_path}")
    if config.get("static_scales_file") is not None:
        raise SystemExit(f"static scales unexpectedly configured in {config_path}")

    compile_proof = json.loads(compile_path.read_text())
    if compile_proof.get("expected_per_token_scale") is not True:
        raise SystemExit(f"wrong compile expectation in {compile_path}")
    if int(compile_proof.get("matching_writer_specs", 0)) < 1:
        raise SystemExit(f"no dynamic writer spec found in {compile_path}")
    if compile_proof.get("all_writer_specs_match_expected") is not True:
        raise SystemExit(f"mixed writer semantics found in {compile_path}")

    summary = {
        **config,
        "token_first16": tokenized["first16"],
        "total_positions": EXPECTED_POSITIONS,
        "mean_kld": mean_kld,
        "elapsed_sec": float(done["elapsed_sec"]),
        "log_sha256": sha256(log_path),
        "writer_compile_proof_sha256": sha256(compile_path),
        "valid": True,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return summary


def aggregate(root: pathlib.Path, expected_runs: int) -> dict:
    paths = sorted(
        (root / "results" / "dynamic_per_token").glob("run*/summary.json"),
        key=lambda path: int(path.parent.name.removeprefix("run")),
    )
    if len(paths) != expected_runs:
        raise SystemExit(
            f"expected {expected_runs} valid TR3 runs, found {len(paths)}"
        )
    rows = [json.loads(path.read_text()) for path in paths]
    if [int(row.get("run", -1)) for row in rows] != list(
        range(1, expected_runs + 1)
    ):
        raise SystemExit("TR3 run numbers are incomplete or out of order")
    reference = rows[0]
    for row in rows[1:]:
        mismatches = {
            key: (reference.get(key), row.get(key))
            for key in INVARIANT_KEYS
            if row.get(key) != reference.get(key)
        }
        if mismatches:
            raise SystemExit(f"cross-run invariant mismatch: {mismatches}")
    log_hashes = [row.get("log_sha256") for row in rows]
    if len(set(log_hashes)) != len(log_hashes):
        raise SystemExit("duplicate KLD run logs; cold runs are not independent")

    tr3_values = [float(row["mean_kld"]) for row in rows]
    tr3_mean = statistics.mean(tr3_values)
    tr3_sd = statistics.stdev(tr3_values) if len(tr3_values) > 1 else 0.0
    nf3_mean = statistics.mean(NF3_DYNAMIC_VALUES)
    nf3_sd = statistics.stdev(NF3_DYNAMIC_VALUES)
    result = {
        "reference_direction": "KL(BF16_reference || candidate)",
        "selector_sensitive": False,
        "positions_per_run": EXPECTED_POSITIONS,
        "tr3_dynamic": {
            "runs": expected_runs,
            "values": tr3_values,
            "mean_kld": tr3_mean,
            "sd_kld": tr3_sd,
            "min_kld": min(tr3_values),
            "max_kld": max(tr3_values),
        },
        "nf3_dynamic_archived": {
            "runs": 3,
            "values": NF3_DYNAMIC_VALUES,
            "mean_kld": nf3_mean,
            "sd_kld": nf3_sd,
            "source": (
                "harness/cn4-evidence-archive/20260728/"
                "nvfp4-dynamic-token-scale-kld-n3-v1/aggregate_summary.json"
            ),
        },
        "tr3_minus_nf3": {
            "mean_kld": tr3_mean - nf3_mean,
            "percent_of_nf3_mean": (tr3_mean - nf3_mean) / nf3_mean * 100.0,
        },
        "invariants": {key: reference.get(key) for key in INVARIANT_KEYS},
        "runs_detail": [
            {
                "run": int(row["run"]),
                "mean_kld": float(row["mean_kld"]),
                "elapsed_sec": float(row["elapsed_sec"]),
                "log_sha256": row["log_sha256"],
                "writer_compile_proof_sha256": row[
                    "writer_compile_proof_sha256"
                ],
            }
            for row in rows
        ],
    }
    (root / "aggregate_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# GLM-5.2 EXL3-TR3 dynamic-NVFP4 KLD comparison",
        "",
        "| Candidate | Runs | Mean KLD | Sample SD | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| TR3 dynamic | {expected_runs} | {tr3_mean:.8f} | "
            f"{tr3_sd:.8f} | {min(tr3_values):.8f} | "
            f"{max(tr3_values):.8f} |"
        ),
        (
            f"| NF3 dynamic (archived) | 3 | {nf3_mean:.8f} | "
            f"{nf3_sd:.8f} | {min(NF3_DYNAMIC_VALUES):.8f} | "
            f"{max(NF3_DYNAMIC_VALUES):.8f} |"
        ),
        "",
        (
            f"TR3 − NF3 mean: {tr3_mean - nf3_mean:+.8f} "
            f"({(tr3_mean - nf3_mean) / nf3_mean * 100.0:+.2f}%)."
        ),
        "",
        (
            "> This is a 2,048-token shallow no-regression gate. The selector "
            "budget is also 2,048, so deep retrieval is qualified separately."
        ),
        "",
    ]
    text = "\n".join(lines)
    (root / "summary.md").write_text(text)
    print(text)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("run_dir", type=pathlib.Path)
    aggregate_parser = subparsers.add_parser("summarize")
    aggregate_parser.add_argument("root", type=pathlib.Path)
    aggregate_parser.add_argument("--expected-runs", type=int, default=3)
    args = parser.parse_args()
    if args.command == "validate":
        validate(args.run_dir)
    else:
        aggregate(args.root, args.expected_runs)


if __name__ == "__main__":
    main()
