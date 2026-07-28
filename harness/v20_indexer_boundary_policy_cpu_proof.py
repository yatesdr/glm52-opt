#!/usr/bin/env python3
"""CPU proof for bounds-safe sparse-indexer boundary policies.

The historical selector first found an 8-bit FP16 score bucket containing the
Kth element, buffered at most 4096 members of that bucket, and refined only the
buffered members.  Buffer insertion was performed by shared atomics, so the
result depended on an implementation capacity and scan order.

This proof makes candidate policies explicit and compares them with captured
production selections:

* exact: full float32 top-k (control);
* oldest: refine the oldest N logical positions in the threshold bucket;
* stratified: refine N positions evenly across the threshold bucket;
* oldest_stratified: half oldest and half evenly distributed.

All policies select every score in a strictly higher coarse bucket.  The
candidate cap applies only to the threshold bucket and every output remains
exactly K elements.  No GPU or model process is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


TRACE_SCHEMA = "v19-v20-longctx-indexer-boundary-v1"
PAGE_SIZE = 64
HEAD_DIM = 128
SCALE_BYTES = 4
TOPK = 2048
CAPS = (2048, 4096, 8192)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _set_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aset = {int(value) for value in a.tolist() if int(value) >= 0}
    bset = {int(value) for value in b.tolist() if int(value) >= 0}
    intersection = aset & bset
    union = aset | bset
    return {
        "a_count": len(aset),
        "b_count": len(bset),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": 1.0 if not union else len(intersection) / len(union),
        "set_exact": aset == bset,
    }


def _position_metrics(indices: np.ndarray, seq_len: int) -> dict[str, Any]:
    values = np.asarray(indices, dtype=np.int64)
    normalized = values.astype(np.float64) / max(seq_len - 1, 1)
    return {
        "count": int(values.size),
        "min": None if not values.size else int(values.min()),
        "median": None if not values.size else float(np.median(values)),
        "max": None if not values.size else int(values.max()),
        "first_half": int(np.count_nonzero(normalized < 0.5)),
        "last_quarter": int(np.count_nonzero(normalized >= 0.75)),
    }


def _dense_scores(record: dict[str, Any]) -> torch.Tensor:
    cache = record["cache_pages"].contiguous()
    if cache.ndim != 3 or tuple(cache.shape[1:]) != (
        PAGE_SIZE,
        HEAD_DIM + SCALE_BYTES,
    ):
        raise RuntimeError(f"unexpected cache shape {tuple(cache.shape)}")
    flat = cache.view(cache.shape[0], -1)
    data_bytes = PAGE_SIZE * HEAD_DIM
    k_quant = (
        flat[:, :data_bytes]
        .contiguous()
        .view(cache.shape[0], PAGE_SIZE, HEAD_DIM)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    k_scale = (
        flat[:, data_bytes:]
        .contiguous()
        .view(torch.float32)
        .view(cache.shape[0], PAGE_SIZE)
    )
    seq_len = int(record["seq_len"].item())
    k = k_quant.reshape(-1, HEAD_DIM)[:seq_len]
    scales = k_scale.reshape(-1)[:seq_len]
    q = record["q_fp8"].to(torch.float32)
    weights = record["weights"].to(torch.float32)
    per_head = torch.matmul(q, k.transpose(0, 1))
    return (torch.relu(per_head) * weights[:, None]).sum(dim=0) * scales


def _coarse_bins(scores: np.ndarray) -> np.ndarray:
    fp16_bits = scores.astype(np.float16).view(np.uint16)
    sign = fp16_bits & np.uint16(0x8000)
    keys = np.where(
        sign != 0,
        np.uint16(0xFFFF) ^ fp16_bits,
        fp16_bits | np.uint16(0x8000),
    ).astype(np.uint16)
    return (keys >> np.uint16(8)).astype(np.int64)


def _threshold(scores: np.ndarray, topk: int) -> tuple[np.ndarray, int, int]:
    bins = _coarse_bins(scores)
    hist = np.bincount(bins, minlength=256)
    count_greater = 0
    for threshold_bin in range(255, -1, -1):
        bucket_count = int(hist[threshold_bin])
        if count_greater < topk <= count_greater + bucket_count:
            return bins, threshold_bin, topk - count_greater
        count_greater += bucket_count
    raise RuntimeError("failed to locate the coarse threshold bucket")


def _evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    if values.size <= count:
        return values
    # Midpoint sampling avoids privileging either endpoint while remaining
    # deterministic for a given logical-position ordering.
    offsets = ((np.arange(count, dtype=np.int64) * 2 + 1) * values.size) // (
        count * 2
    )
    return values[offsets]


def _candidate_pool(
    boundary: np.ndarray,
    *,
    cap: int,
    policy: str,
) -> np.ndarray:
    if boundary.size <= cap:
        return boundary
    if policy == "oldest":
        return boundary[:cap]
    if policy == "stratified":
        return _evenly_spaced(boundary, cap)
    if policy == "oldest_stratified":
        oldest_count = cap // 2
        oldest = boundary[:oldest_count]
        remaining = boundary[oldest_count:]
        distributed = _evenly_spaced(remaining, cap - oldest_count)
        return np.unique(np.concatenate((oldest, distributed)))
    raise ValueError(f"unknown policy {policy!r}")


def _select(
    scores: np.ndarray,
    *,
    topk: int,
    cap: int,
    policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    bins, threshold_bin, boundary_need = _threshold(scores, topk)
    higher = np.flatnonzero(bins > threshold_bin)
    boundary = np.flatnonzero(bins == threshold_bin)
    pool = _candidate_pool(boundary, cap=cap, policy=policy)
    if pool.size < boundary_need:
        raise RuntimeError(
            f"{policy}/cap={cap}: pool {pool.size} cannot fill {boundary_need}"
        )
    winners = pool[
        np.argpartition(scores[pool], -boundary_need)[-boundary_need:]
    ]
    selected = np.concatenate((higher, winners))
    if selected.size != topk:
        raise RuntimeError(
            f"{policy}/cap={cap}: selected {selected.size}, expected {topk}"
        )
    return selected, {
        "threshold_bin": threshold_bin,
        "strictly_higher_count": int(higher.size),
        "threshold_bucket_count": int(boundary.size),
        "threshold_bucket_needed": boundary_need,
        "candidate_pool_count": int(pool.size),
        "candidate_pool_positions": _position_metrics(pool, scores.size),
    }


def _run_trace(path: Path) -> dict[str, Any]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("schema") != TRACE_SCHEMA:
        raise RuntimeError(f"{path}: wrong schema {record.get('schema')!r}")
    scores_t = _dense_scores(record)
    scores = scores_t.numpy()
    captured = record["topk_indices"].to(torch.int64).numpy()
    exact = np.argpartition(scores, -TOPK)[-TOPK:]
    result: dict[str, Any] = {
        "trace": str(path),
        "rank": int(record["tp_rank"]),
        "seq_len": int(record["seq_len"].item()),
        "fingerprints": {
            "scores": _sha256_tensor(scores_t),
            "captured_indices": _sha256_tensor(record["topk_indices"]),
        },
        "captured_vs_exact": _set_metrics(captured, exact),
        "captured_positions": _position_metrics(captured, scores.size),
        "policies": {},
    }
    for policy in ("oldest", "stratified", "oldest_stratified"):
        for cap in CAPS:
            selected, metadata = _select(
                scores,
                topk=TOPK,
                cap=cap,
                policy=policy,
            )
            result["policies"][f"{policy}_{cap}"] = {
                **metadata,
                "vs_captured": _set_metrics(selected, captured),
                "vs_exact": _set_metrics(selected, exact),
                "positions": _position_metrics(selected, scores.size),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-root",
        type=Path,
        required=True,
        help="directory containing tpN/layer00-indexer-local.pt",
    )
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    traces = [
        args.trace_root / f"tp{rank}" / "layer00-indexer-local.pt"
        for rank in range(args.ranks)
    ]
    report = {
        "schema": "v20-indexer-boundary-policy-cpu-proof-v1",
        "claim_boundary": (
            "operator policy reconstruction on preserved layer-0 rows; "
            "end-to-end long-context retrieval requires a causal model boot"
        ),
        "results": [_run_trace(path) for path in traces],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
