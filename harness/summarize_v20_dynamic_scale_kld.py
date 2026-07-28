#!/usr/bin/env python3
"""Validate and summarize matched GLM-5.2 NVFP4 scale-mode KLD runs."""

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
POLICIES = ("static_calibrated", "dynamic_per_token")
AGGREGATE_INVARIANT_KEYS = (
    "image",
    "image_id",
    "model_dir",
    "reference_logits",
    "reference_logits_sha256",
    "reference_manifest_sha256",
    "runner_sha256",
    "context_length",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "kv_cache_dtype",
    "quantization",
    "online_quantization",
    "kv_fp8_rope",
    "tensor_parallel_size",
    "decode_context_parallel_size",
    "enforce_eager",
    "selector_policy",
    "token_first16",
    "total_positions",
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_json_marker(text: str, marker: str) -> dict:
    matches = re.findall(rf"(?m)^{re.escape(marker)}\s+(\{{.*\}})\s*$", text)
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one {marker!r} marker, found {len(matches)}"
        )
    return json.loads(matches[0])


def _validate_scale_contract(config: dict, run_dir: pathlib.Path) -> None:
    policy = config.get("policy")
    dynamic = config.get("dynamic_per_token")
    scales_file = config.get("static_scales_file")
    scales_sha = config.get("static_scales_sha256")
    compile_proof_path = run_dir / "writer-compile-proof.json"
    if not compile_proof_path.is_file():
        raise SystemExit(f"missing writer compile proof: {compile_proof_path}")
    compile_proof = json.loads(compile_proof_path.read_text())

    if policy == "static_calibrated":
        if dynamic is not False or not scales_file or len(str(scales_sha)) != 64:
            raise SystemExit(f"invalid static scale contract in {run_dir}")
        expected_flag = False
    elif policy == "dynamic_per_token":
        if dynamic is not True or scales_file is not None or scales_sha is not None:
            raise SystemExit(f"invalid dynamic scale contract in {run_dir}")
        expected_flag = True
    else:
        raise SystemExit(f"invalid policy in {run_dir}: {policy!r}")

    if compile_proof.get("expected_per_token_scale") is not expected_flag:
        raise SystemExit(f"wrong expected compile flag in {compile_proof_path}")
    if int(compile_proof.get("matching_writer_specs", 0)) < 1:
        raise SystemExit(f"no matching writer compile spec in {compile_proof_path}")
    if not compile_proof.get("all_writer_specs_match_expected"):
        raise SystemExit(f"mixed/wrong writer semantics in {compile_proof_path}")


def validate_run(run_dir: pathlib.Path) -> dict:
    log_path = run_dir / "prefill_dcp1.log"
    config_path = run_dir / "config.json"
    if not log_path.is_file() or not config_path.is_file():
        raise SystemExit(f"missing log/config in {run_dir}")

    text = log_path.read_text(errors="replace")
    if "Traceback (most recent call last)" in text:
        raise SystemExit(f"traceback present in {log_path}")
    if re.search(r"(?i)(out of memory|oom-kill|killed|engine core failed)", text):
        raise SystemExit(f"fatal memory/engine signature present in {log_path}")

    tokenized = _single_json_marker(text, "tokenized")
    done = _single_json_marker(text, "fallback_prefill_kld_done")
    first16 = tokenized.get("first16")
    if first16 != EXPECTED_FIRST16:
        raise SystemExit(f"reference-token mismatch in {run_dir}: {first16!r}")

    positions = int(done.get("total_positions", -1))
    if positions != EXPECTED_POSITIONS:
        raise SystemExit(
            f"expected {EXPECTED_POSITIONS} positions in {run_dir}, got {positions}"
        )
    mean_kld = float(done.get("mean_kld", math.nan))
    if not math.isfinite(mean_kld) or mean_kld < 0:
        raise SystemExit(f"invalid mean_kld in {run_dir}: {mean_kld!r}")

    config = json.loads(config_path.read_text())
    if config.get("policy") not in POLICIES:
        raise SystemExit(f"invalid policy in {config_path}")
    _validate_scale_contract(config, run_dir)

    summary = {
        **config,
        "token_first16": first16,
        "total_positions": positions,
        "mean_kld": mean_kld,
        "elapsed_sec": float(done["elapsed_sec"]),
        "log_sha256": _sha256(log_path),
        "writer_compile_proof_sha256": _sha256(
            run_dir / "writer-compile-proof.json"
        ),
        "valid": True,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return summary


def summarize(root: pathlib.Path, expected_runs: int) -> dict:
    rows: dict[str, list[dict]] = {}
    for policy in POLICIES:
        summaries = []
        paths = list((root / "results" / policy).glob("run*/summary.json"))
        for path in sorted(
            paths, key=lambda item: int(item.parent.name.removeprefix("run"))
        ):
            data = json.loads(path.read_text())
            if data.get("valid") is not True or data.get("policy") != policy:
                raise SystemExit(f"invalid summary: {path}")
            summaries.append(data)
        if len(summaries) != expected_runs:
            raise SystemExit(
                f"{policy}: expected {expected_runs} valid runs, "
                f"found {len(summaries)}"
            )
        run_numbers = [int(row.get("run", -1)) for row in summaries]
        expected_numbers = list(range(1, expected_runs + 1))
        if run_numbers != expected_numbers:
            raise SystemExit(
                f"{policy}: expected run numbers {expected_numbers}, "
                f"found {run_numbers}"
            )
        rows[policy] = summaries

    all_runs = [row for policy in POLICIES for row in rows[policy]]
    reference = all_runs[0]
    for row in all_runs[1:]:
        mismatches = {
            key: (reference.get(key), row.get(key))
            for key in AGGREGATE_INVARIANT_KEYS
            if row.get(key) != reference.get(key)
        }
        if mismatches:
            raise SystemExit(
                f"aggregate invariant mismatch in "
                f"{row.get('policy')}/run{row.get('run')}: {mismatches}"
            )
    log_hashes = [str(row.get("log_sha256", "")) for row in all_runs]
    if len(set(log_hashes)) != len(log_hashes) or any(
        len(value) != 64 for value in log_hashes
    ):
        raise SystemExit("missing, malformed, or duplicate per-run log SHA-256")

    aggregate_rows = []
    for policy in POLICIES:
        values = [float(row["mean_kld"]) for row in rows[policy]]
        aggregate_rows.append(
            {
                "policy": policy,
                "runs": len(values),
                "mean_kld": statistics.mean(values),
                "sd_kld": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min_kld": min(values),
                "max_kld": max(values),
                "values": values,
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
                    for row in rows[policy]
                ],
            }
        )

    static = aggregate_rows[0]["values"]
    dynamic = aggregate_rows[1]["values"]
    paired_delta = [
        candidate - control for control, candidate in zip(static, dynamic)
    ]
    aggregate = {
        "root": str(root),
        "expected_runs": expected_runs,
        "reference_direction": "KL(BF16_reference || candidate)",
        "context_length": 2048,
        "selector_budget": 2048,
        "selector_sensitive": False,
        "invariants": {
            key: reference.get(key) for key in AGGREGATE_INVARIANT_KEYS
        },
        "rows": aggregate_rows,
        "paired_delta_dynamic_minus_static": {
            "values": paired_delta,
            "mean": statistics.mean(paired_delta),
            "sd": statistics.stdev(paired_delta) if len(paired_delta) > 1 else 0.0,
            "percent_of_static_mean": (
                statistics.mean(paired_delta)
                / float(aggregate_rows[0]["mean_kld"])
                * 100.0
            ),
        },
    }
    (root / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# GLM-5.2 v20 dynamic NVFP4 scale shallow BF16 KLD",
        "",
        "| Scale mode | Runs | KLD mean ± sample SD | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| `{row['policy']}` | {row['runs']} | "
            f"{row['mean_kld']:.8f} ± {row['sd_kld']:.8f} | "
            f"{row['min_kld']:.8f} | {row['max_kld']:.8f} |"
        )
    delta = aggregate["paired_delta_dynamic_minus_static"]
    lines.extend(
        [
            "",
            "Paired `dynamic_per_token - static_calibrated` KLD deltas: "
            + ", ".join(f"{value:+.8f}" for value in paired_delta),
            "",
            f"Mean paired delta: {delta['mean']:+.8f} "
            f"({delta['percent_of_static_mean']:+.2f}% of the static mean).",
            "",
            "> This is a 2,048-token no-regression gate. The selector budget is "
            "also 2,048, so this cell is not selector-sensitive. The frozen "
            "350k gate and randomized 475k ladder provide deep-context evidence.",
            "",
        ]
    )
    summary_text = "\n".join(lines)
    (root / "summary.md").write_text(summary_text)
    print(summary_text)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("run_dir", type=pathlib.Path)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("root", type=pathlib.Path)
    summary_parser.add_argument("--expected-runs", type=int, default=3)

    args = parser.parse_args()
    if args.command == "validate":
        validate_run(args.run_dir)
    else:
        summarize(args.root, args.expected_runs)


if __name__ == "__main__":
    main()
