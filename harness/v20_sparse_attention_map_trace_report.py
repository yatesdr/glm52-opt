#!/usr/bin/env python3
"""Validate a real sparse-index logical-to-attention mapping trace.

The trace is captured after DCP filtering/gather mapping, valid-count capping,
and tail masking. This report independently reconstructs the expected physical
or gathered slots from the logical top-k IDs and the captured request metadata.
Any mismatch is a real selector-to-attention handoff error, not a model-output
quality heuristic.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


SCHEMA = "v20-sparse-attention-map-v1"


def _as_int_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a tensor")
    if value.ndim != 1:
        raise ValueError(f"{field} must be rank 1, got {tuple(value.shape)}")
    if value.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{field} must be int32/int64, got {value.dtype}")
    return [int(item) for item in value.tolist()]


def _logical_to_local(token: int, dcp_size: int, interleave: int) -> tuple[int, int]:
    owner = (token // interleave) % dcp_size
    local = (
        token // (dcp_size * interleave)
    ) * interleave + token % interleave
    return owner, local


def _expected_gathered_slots(record: dict[str, Any], logical: list[int]) -> list[int]:
    dcp_size = int(record["dcp_size"])
    interleave = int(record["dcp_interleave"])
    padded = int(record["padded_rank_tokens"])
    starts = _as_int_list(record["rank_req_starts"], "rank_req_starts")
    lengths = _as_int_list(record["rank_req_lens"], "rank_req_lens")
    if len(starts) != dcp_size or len(lengths) != dcp_size:
        raise ValueError("gather request metadata does not cover every DCP rank")
    if padded <= 0:
        raise ValueError("gather route requires positive padded_rank_tokens")

    expected: list[int] = []
    for token in logical:
        if token < 0:
            continue
        owner, local = _logical_to_local(token, dcp_size, interleave)
        if 0 <= local < lengths[owner]:
            expected.append(owner * padded + starts[owner] + local)
    return expected


def _expected_rank_local_slots(
    record: dict[str, Any], logical: list[int]
) -> list[int]:
    dcp_size = int(record["dcp_size"])
    dcp_rank = int(record["dcp_rank"])
    interleave = int(record["dcp_interleave"])
    block_size = int(record["block_size"])
    block_table = _as_int_list(record["block_table"], "block_table")
    if block_size <= 0:
        raise ValueError("rank-local route requires positive block_size")

    expected: list[int] = []
    for token in logical:
        if token < 0:
            continue
        owner, local = _logical_to_local(token, dcp_size, interleave)
        if owner != dcp_rank:
            continue
        block = local // block_size
        offset = local % block_size
        if block < 0 or block >= len(block_table):
            continue
        physical_block = block_table[block]
        if physical_block >= 0:
            expected.append(physical_block * block_size + offset)
    return expected


def _validate_record(
    path: Path,
    *,
    expected_layer: int,
    expected_batch_tokens: int,
    expected_topk: int,
    needle_min: int,
    needle_max: int,
) -> tuple[dict[str, Any], list[int]]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unexpected trace schema")
    if int(record["layer"]) != expected_layer:
        raise ValueError(f"{path}: layer does not match {expected_layer}")
    if int(record["batch_tokens"]) != expected_batch_tokens:
        raise ValueError(
            f"{path}: batch_tokens does not match {expected_batch_tokens}"
        )

    logical = _as_int_list(record["topk_indices"], "topk_indices")
    selected = _as_int_list(record["selected_indices"], "selected_indices")
    if len(logical) != expected_topk or len(selected) != expected_topk:
        raise ValueError(f"{path}: top-k width does not match {expected_topk}")
    nsa_len = int(record["nsa_cache_seqlen"])
    if not 0 <= nsa_len <= expected_topk:
        raise ValueError(f"{path}: invalid nsa_cache_seqlen {nsa_len}")
    if any(slot < 0 for slot in selected[:nsa_len]):
        raise ValueError(f"{path}: invalid slot inside active attention prefix")
    if any(slot != -1 for slot in selected[nsa_len:]):
        raise ValueError(f"{path}: non-masked slot after active attention prefix")

    route = record["route"]
    if route == "ckv_gather":
        expected = _expected_gathered_slots(record, logical)
    elif route == "rank_local":
        expected = _expected_rank_local_slots(record, logical)
    else:
        raise ValueError(f"{path}: unsupported route {route!r}")

    # At the frozen long-context row, the causal cap is much larger than top-k.
    # Requiring equality prevents a silently truncated mapping from passing.
    if nsa_len != len(expected):
        raise ValueError(
            f"{path}: active length {nsa_len} != reconstructed {len(expected)}"
        )
    actual = selected[:nsa_len]
    missing_slots = list((Counter(expected) - Counter(actual)).elements())
    extra_slots = list((Counter(actual) - Counter(expected)).elements())
    if missing_slots or extra_slots:
        raise ValueError(
            f"{path}: mapping mismatch missing={missing_slots[:16]} "
            f"extra={extra_slots[:16]}"
        )

    needle_hits = sorted(
        token for token in logical if needle_min <= token <= needle_max
    )
    summary = {
        "path": str(path),
        "route": route,
        "dcp_rank": int(record["dcp_rank"]),
        "logical_count": sum(token >= 0 for token in logical),
        "attention_count": nsa_len,
        "cache_seq_len": int(record["cache_seq_len"]),
        "needle_hits": needle_hits,
        "mapping_exact": True,
    }
    return summary, logical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--expected-layer", type=int, default=34)
    parser.add_argument("--expected-ranks", type=int, default=4)
    parser.add_argument("--expected-batch-tokens", type=int, default=2735)
    parser.add_argument("--expected-topk", type=int, default=2048)
    parser.add_argument("--needle-min", type=int, default=137472)
    parser.add_argument("--needle-max", type=int, default=137520)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(args.trace_dir.glob("dcp*/layer-*-mapping.pt"))
    if len(paths) != args.expected_ranks:
        raise SystemExit(
            f"FAIL: expected {args.expected_ranks} rank traces, found {len(paths)}"
        )

    summaries: list[dict[str, Any]] = []
    logical_rows: list[list[int]] = []
    for path in paths:
        summary, logical = _validate_record(
            path,
            expected_layer=args.expected_layer,
            expected_batch_tokens=args.expected_batch_tokens,
            expected_topk=args.expected_topk,
            needle_min=args.needle_min,
            needle_max=args.needle_max,
        )
        summaries.append(summary)
        logical_rows.append(logical)

    ranks = sorted(summary["dcp_rank"] for summary in summaries)
    if ranks != list(range(args.expected_ranks)):
        raise SystemExit(f"FAIL: DCP ranks are incomplete or duplicated: {ranks}")
    routes = {summary["route"] for summary in summaries}
    if len(routes) != 1:
        raise SystemExit(f"FAIL: route differs across ranks: {sorted(routes)}")
    if any(row != logical_rows[0] for row in logical_rows[1:]):
        raise SystemExit("FAIL: merged logical top-k differs across DCP ranks")

    result = {
        "schema": "v20-sparse-attention-map-report-v1",
        "verdict": "PASS",
        "layer": args.expected_layer,
        "route": summaries[0]["route"],
        "logical_topk_rank_invariant": True,
        "any_needle_hit": any(summary["needle_hits"] for summary in summaries),
        "ranks": summaries,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
