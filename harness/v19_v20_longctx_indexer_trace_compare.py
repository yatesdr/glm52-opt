#!/usr/bin/env python3
"""Compare learned layer-0 sparse-indexer boundaries from v19 and v20."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SCHEMA = "v19-v20-longctx-indexer-boundary-v1"
STAGES = ("local", "merged")
LOCAL_FIELDS = (
    "q_fp8",
    "weights",
    "k",
    "seq_len",
    "page_ids",
    "cache_pages",
    "topk_indices",
    "topk_scores",
)
MERGED_FIELDS = ("topk_indices", "topk_scores")


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_trace(root: Path, *, ranks: int) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(root.glob("tp*/layer00-indexer-*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record.get("schema") != SCHEMA:
            raise RuntimeError(f"{path}: unexpected schema {record.get('schema')!r}")
        if int(record.get("layer", -1)) != 0:
            raise RuntimeError(f"{path}: expected layer 0")
        key = (str(record["stage"]), int(record["tp_rank"]))
        if key in rows:
            raise RuntimeError(f"{path}: duplicate trace key {key}")
        rows[key] = record

    expected = {(stage, rank) for stage in STAGES for rank in range(ranks)}
    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    if missing or extra:
        raise RuntimeError(
            f"{root}: trace key mismatch missing={missing} extra={extra}"
        )
    for key, record in rows.items():
        expected_fields = LOCAL_FIELDS if key[0] == "local" else MERGED_FIELDS
        absent = [field for field in expected_fields if field not in record]
        if absent:
            raise RuntimeError(f"{root}: {key} missing fields {absent}")
    return rows


def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.shape != b.shape or a.dtype != b.dtype:
        return {
            "contract_exact": False,
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
            "a_dtype": str(a.dtype),
            "b_dtype": str(b.dtype),
            "exact": False,
            "a_sha256": tensor_sha256(a),
            "b_sha256": tensor_sha256(b),
        }

    exact = torch.equal(a, b)
    metrics: dict[str, Any] = {
        "contract_exact": True,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "exact": exact,
        "a_sha256": tensor_sha256(a),
        "b_sha256": tensor_sha256(b),
        "changed": int(torch.count_nonzero(a != b).item()),
        "numel": a.numel(),
    }
    if a.numel() and (a.is_floating_point() or a.dtype == torch.uint8):
        af = a.float()
        bf = b.float()
        delta = af - bf
        metrics.update(
            {
                "max_abs": float(delta.abs().max().item()),
                "mean_abs": float(delta.abs().mean().item()),
                "rel_l2": float(torch.linalg.vector_norm(delta).item())
                / max(float(torch.linalg.vector_norm(af).item()), 1e-30),
            }
        )
    return metrics


def topk_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    metrics = tensor_metrics(a, b)
    if a.shape != b.shape:
        return metrics
    a_valid = {int(value) for value in a.tolist() if int(value) >= 0}
    b_valid = {int(value) for value in b.tolist() if int(value) >= 0}
    union = a_valid | b_valid
    intersection = a_valid & b_valid
    metrics.update(
        {
            "a_valid": len(a_valid),
            "b_valid": len(b_valid),
            "intersection": len(intersection),
            "union": len(union),
            "jaccard": 1.0 if not union else len(intersection) / len(union),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v19", type=Path, required=True)
    parser.add_argument("--v20", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    old = load_trace(args.v19, ranks=args.ranks)
    new = load_trace(args.v20, ranks=args.ranks)

    comparisons: list[dict[str, Any]] = []
    first_nonexact: dict[str, Any] | None = None
    ordered_boundaries = (
        ("local", "q_fp8"),
        ("local", "weights"),
        ("local", "k"),
        ("local", "seq_len"),
        ("local", "cache_pages"),
        ("local", "topk_indices"),
        ("local", "topk_scores"),
        ("merged", "topk_indices"),
    )
    all_metrics: dict[tuple[str, str, int], dict[str, Any]] = {}

    for stage in STAGES:
        fields = LOCAL_FIELDS if stage == "local" else MERGED_FIELDS
        for rank in range(args.ranks):
            old_record = old[(stage, rank)]
            new_record = new[(stage, rank)]
            metadata = {
                key: {"v19": old_record.get(key), "v20": new_record.get(key)}
                for key in (
                    "batch_tokens",
                    "dcp_rank",
                    "dcp_world_size",
                    "cp_kv_cache_interleave_size",
                    "dcp_global_topk",
                    "query_split_active",
                    "active_page_width",
                    "owner_merge_used",
                )
                if key in old_record or key in new_record
            }
            field_metrics: dict[str, Any] = {}
            for field in fields:
                a = old_record[field]
                b = new_record[field]
                metrics = (
                    topk_metrics(a, b)
                    if field == "topk_indices"
                    else tensor_metrics(a, b)
                )
                field_metrics[field] = metrics
                all_metrics[(stage, field, rank)] = metrics
            comparisons.append(
                {
                    "stage": stage,
                    "rank": rank,
                    "metadata": metadata,
                    "fields": field_metrics,
                }
            )

    for stage, field in ordered_boundaries:
        differing_ranks = [
            rank
            for rank in range(args.ranks)
            if not all_metrics[(stage, field, rank)]["exact"]
        ]
        if differing_ranks:
            first_nonexact = {
                "stage": stage,
                "field": field,
                "ranks": differing_ranks,
            }
            break

    if first_nonexact is None:
        localization = "no difference through merged top-k"
    elif first_nonexact["field"] in {"q_fp8", "weights", "k"}:
        localization = "learned projection/RoPE/quantization input"
    elif first_nonexact["field"] in {"seq_len", "cache_pages"}:
        localization = "index-cache construction or prefill metadata"
    elif first_nonexact["stage"] == "local":
        localization = "local paged-indexer scoring/selection"
    else:
        localization = "DCP global top-k merge"

    report = {
        "schema": "v19-v20-longctx-indexer-trace-comparison-v1",
        "contract": {
            "layer": 0,
            "ranks": args.ranks,
            "stages": list(STAGES),
        },
        "first_nonexact_boundary": first_nonexact,
        "localization": localization,
        "comparisons": comparisons,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
