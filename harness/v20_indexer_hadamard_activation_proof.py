#!/usr/bin/env python3
"""Compare indexer precision modes on a captured real GLM activation row.

Input is produced by the opt-in ``indexer_prequant_trace`` custom op.  The
trace contains one layer's real post-RoPE BF16 K rows, the final query row,
learned head weights, and the runtime-selected indices.  This script runs on
CPU and compares the checkpoint's full-precision selector with:

  * the current raw-vector E4M3/UE8M0 indexer;
  * the same quantizer after an orthonormal 128-point Hadamard rotation.

No model is loaded and no GPU is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v20_indexer_hadamard_precision_cpu_proof import (
    HEAD_DIM,
    _normalized_hadamard,
    _selector_scores,
    _ue8m0_fp8_roundtrip,
)
from v20_indexer_boundary_policy_cpu_proof import _select as _boundary_select


ACTIVATION_SCHEMA = "v20-indexer-prequant-activation-v1"
SELECTION_SCHEMAS = {
    "v20-indexer-prequant-runtime-selection-v1",
    "v20-indexer-prequant-runtime-selection-v2",
}
EQUAL_SEGMENT_COUNTS = (2, 4, 8, 16, 32, 64, 128)
FIXED_TILE_WIDTHS = (4096, 8192, 16384, 32768)
BLOCK_EXPANSION_WIDTHS = (4, 8, 16, 32, 64, 128, 256)


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _load_trace(trace_dir: Path) -> dict[str, Any]:
    chunk_paths = sorted(trace_dir.glob("chunk-*.pt"))
    if not chunk_paths:
        raise RuntimeError(f"{trace_dir}: no chunk traces")

    keys: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    q_final: np.ndarray | None = None
    weights_final: np.ndarray | None = None
    tail_start_position: int | None = None
    expected_chunk = 0
    tail_count = 0
    for path in chunk_paths:
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record.get("schema") != ACTIVATION_SCHEMA:
            raise RuntimeError(f"{path}: wrong schema {record.get('schema')!r}")
        if int(record["chunk"]) != expected_chunk:
            raise RuntimeError(
                f"{path}: expected chunk {expected_chunk}, got {record['chunk']}"
            )
        expected_chunk += 1
        k = record["k_bf16"]
        pos = record["positions"]
        if tuple(k.shape) != (int(record["batch_tokens"]), HEAD_DIM):
            raise RuntimeError(f"{path}: invalid K shape {tuple(k.shape)}")
        if pos.numel() != k.shape[0]:
            raise RuntimeError(f"{path}: position/K mismatch")
        keys.append(k.float().numpy())
        positions.append(pos.to(torch.int64).numpy())
        if "q_final_bf16" in record:
            tail_count += 1
            q_final = record["q_final_bf16"].float().numpy()
            weights_final = record["weights_final_bf16"].float().numpy()
            tail_start_position = int(pos[0].item())

    if (
        tail_count != 1
        or q_final is None
        or weights_final is None
        or tail_start_position is None
    ):
        raise RuntimeError(f"expected exactly one tail record, got {tail_count}")
    k_all = np.concatenate(keys)
    positions_all = np.concatenate(positions)
    if len(np.unique(positions_all)) != positions_all.size:
        raise RuntimeError("trace positions contain duplicates")
    if positions_all.size > 1 and not np.all(np.diff(positions_all) == 1):
        raise RuntimeError("trace positions are not strictly contiguous")

    selection_path = trace_dir / "runtime-selection.pt"
    selection = torch.load(selection_path, map_location="cpu", weights_only=True)
    if selection.get("schema") not in SELECTION_SCHEMAS:
        raise RuntimeError(
            f"{selection_path}: wrong schema {selection.get('schema')!r}"
        )
    if selection.get("schema") == "v20-indexer-prequant-runtime-selection-v2":
        active_rows = int(selection["batch_tokens"])
        selected_row = int(selection["selected_row"])
        buffer_rows = int(selection["buffer_rows"])
        if selected_row != active_rows - 1 or buffer_rows < active_rows:
            raise RuntimeError(
                f"{selection_path}: invalid active-row selection metadata"
            )
    return {
        "chunk_paths": chunk_paths,
        "q": q_final,
        "k": k_all,
        "weights": weights_final,
        "positions": positions_all,
        "runtime_indices": selection["topk_indices"].to(torch.int64).numpy(),
        "runtime_absolute_position": int(selection["absolute_position"]),
        "tail_start_position": tail_start_position,
    }


def _topk_positions(
    scores: np.ndarray, positions: np.ndarray, width: int
) -> np.ndarray:
    rows = np.argpartition(scores, -width)[-width:]
    return positions[rows]


def _set_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a_set = set(int(value) for value in a.tolist() if int(value) >= 0)
    b_set = set(int(value) for value in b.tolist() if int(value) >= 0)
    intersection = a_set & b_set
    union = a_set | b_set
    return {
        "a_count": len(a_set),
        "b_count": len(b_set),
        "intersection": len(intersection),
        "jaccard": 1.0 if not union else len(intersection) / len(union),
        "set_exact": a_set == b_set,
    }


def _local_rank_metrics(
    *,
    scores: np.ndarray,
    positions: np.ndarray,
    needle_rows: np.ndarray,
) -> dict[str, Any]:
    """Measure whether exact per-history-partition quotas can cover a needle.

    The total-budget fields answer the useful feasibility question directly:
    if every partition receives the quota required by the worst needle token,
    does the resulting deterministic coverage reservation fit in top-k=2048?
    """

    def partition_metrics(
        bounds_by_row: dict[int, tuple[int, int]],
        partition_count: int,
    ) -> dict[str, Any]:
        token_ranks: dict[str, int] = {}
        token_bounds: dict[str, dict[str, int]] = {}
        for row in needle_rows.tolist():
            start, end = bounds_by_row[row]
            rank = int(np.count_nonzero(scores[start:end] > scores[row]) + 1)
            position = int(positions[row])
            token_ranks[str(position)] = rank
            token_bounds[str(position)] = {
                "start_position": int(positions[start]),
                "end_position": int(positions[end - 1]),
                "rows": end - start,
            }
        required_quota = max(token_ranks.values())
        return {
            "partition_count": partition_count,
            "token_ranks": token_ranks,
            "token_bounds": token_bounds,
            "required_uniform_quota": required_quota,
            "required_uniform_total_budget": partition_count * required_quota,
            "fits_topk_2048": partition_count * required_quota <= 2048,
        }

    row_count = int(positions.size)
    equal_segments: dict[str, Any] = {}
    for segment_count in EQUAL_SEGMENT_COUNTS:
        bounds: dict[int, tuple[int, int]] = {}
        for row in needle_rows.tolist():
            segment = min(segment_count - 1, row * segment_count // row_count)
            start = segment * row_count // segment_count
            end = (segment + 1) * row_count // segment_count
            bounds[row] = (start, end)
        equal_segments[str(segment_count)] = partition_metrics(
            bounds, segment_count
        )

    fixed_tiles: dict[str, Any] = {}
    origin = int(positions[0])
    history_end = int(positions[-1]) + 1
    for tile_width in FIXED_TILE_WIDTHS:
        tile_count = (history_end - origin + tile_width - 1) // tile_width
        bounds = {}
        for row in needle_rows.tolist():
            position = int(positions[row])
            tile = (int(position) - origin) // tile_width
            start_position = origin + tile * tile_width
            end_position = min(history_end, start_position + tile_width)
            start = int(np.searchsorted(positions, start_position, side="left"))
            end = int(np.searchsorted(positions, end_position, side="left"))
            bounds[row] = (start, end)
        fixed_tiles[str(tile_width)] = partition_metrics(bounds, tile_count)

    return {
        "equal_segments": equal_segments,
        "fixed_tiles": fixed_tiles,
    }


def _block_expansion_metrics(
    *,
    scores: np.ndarray,
    positions: np.ndarray,
    needle_rows: np.ndarray,
    topk: int,
) -> dict[str, Any]:
    """Evaluate max-score block selection with exact contiguous expansion."""

    origin = int(positions[0])
    history_end = int(positions[-1]) + 1
    results: dict[str, Any] = {}
    for block_width in BLOCK_EXPANSION_WIDTHS:
        block_count = (history_end - origin + block_width - 1) // block_width
        block_ids = (positions - origin) // block_width
        starts = np.flatnonzero(
            np.r_[True, block_ids[1:] != block_ids[:-1]]
        )
        block_scores = np.maximum.reduceat(scores, starts)
        if int(block_scores.size) != block_count:
            raise RuntimeError(
                f"expected {block_count} blocks at width {block_width}, "
                f"got {int(block_scores.size)}"
            )
        needle_blocks = np.unique(block_ids[needle_rows])
        needle_block_ranks = {
            str(int(block)): int(
                np.count_nonzero(block_scores > block_scores[block]) + 1
            )
            for block in needle_blocks.tolist()
        }
        selected_block_budget = topk // block_width
        required_rank = max(needle_block_ranks.values())
        results[str(block_width)] = {
            "block_count": block_count,
            "selected_block_budget": selected_block_budget,
            "expanded_token_budget": selected_block_budget * block_width,
            "needle_block_ranks": needle_block_ranks,
            "required_block_rank": required_rank,
            "needle_covered": required_rank <= selected_block_budget,
            "needle_blocks": [
                {
                    "block": int(block),
                    "start_position": origin + int(block) * block_width,
                    "end_position": min(
                        history_end - 1,
                        origin + (int(block) + 1) * block_width - 1,
                    ),
                }
                for block in needle_blocks.tolist()
            ],
        }
    return results


def _mode_metrics(
    *,
    oracle_scores: np.ndarray,
    mode_scores: np.ndarray,
    oracle_indices: np.ndarray,
    mode_indices: np.ndarray,
    positions: np.ndarray,
    needle_fraction: float,
    needle_radius: int,
    needle_reference_position: int,
    needle_center_override: int | None,
) -> dict[str, Any]:
    error = mode_scores - oracle_scores
    width = int(oracle_indices.size)
    oracle_rows = np.argpartition(oracle_scores, -width)[-width:]
    oracle_threshold = float(np.min(oracle_scores[oracle_rows]))
    mode_set = set(int(value) for value in mode_indices.tolist())
    oracle_topk_positions = positions[oracle_rows]
    missed_rows = oracle_rows[
        np.asarray(
            [int(value) not in mode_set for value in oracle_topk_positions],
            dtype=np.bool_,
        )
    ]
    missed_margin = np.maximum(
        oracle_scores[missed_rows] - oracle_threshold,
        0.0,
    )
    all_margin = np.maximum(
        oracle_scores[oracle_rows] - oracle_threshold,
        0.0,
    )
    needle_center = (
        int(needle_center_override)
        if needle_center_override is not None
        else int(round(float(needle_reference_position) * needle_fraction))
    )
    needle_mask = np.abs(positions - needle_center) <= needle_radius
    if not np.any(needle_mask):
        raise RuntimeError("needle window is outside the captured positions")
    best_needle_score = float(np.max(mode_scores[needle_mask]))
    best_needle_rank = int(np.count_nonzero(mode_scores > best_needle_score) + 1)
    needle_positions = positions[needle_mask]
    needle_rows = np.flatnonzero(needle_mask)
    boundary_policies: dict[str, Any] = {}
    for policy in ("oldest", "stratified", "oldest_stratified"):
        for cap in (2048, 4096, 8192):
            selected_rows, metadata = _boundary_select(
                mode_scores,
                topk=width,
                cap=cap,
                policy=policy,
            )
            selected_positions = positions[selected_rows]
            selected_set = {
                int(value) for value in selected_positions.tolist()
            }
            boundary_policies[f"{policy}_{cap}"] = {
                **metadata,
                "vs_exact_mode_topk": _set_metrics(
                    selected_positions, mode_indices
                ),
                "needle_selected_tokens": [
                    int(value)
                    for value in needle_positions
                    if int(value) in selected_set
                ],
            }
    return {
        "vs_oracle_topk": _set_metrics(mode_indices, oracle_indices),
        "score_weighted_false_negatives": {
            "count": int(missed_rows.size),
            "oracle_margin_sum": float(np.sum(missed_margin)),
            "oracle_margin_fraction": float(np.sum(missed_margin))
            / max(float(np.sum(all_margin)), np.finfo(np.float32).tiny),
            "largest_omitted_oracle_margin": (
                0.0 if missed_margin.size == 0 else float(np.max(missed_margin))
            ),
        },
        "score_rmse": float(np.sqrt(np.mean(np.square(error, dtype=np.float64)))),
        "score_max_abs_error": float(np.max(np.abs(error))),
        "needle_window": {
            "center": needle_center,
            "radius": needle_radius,
            "best_rank": best_needle_rank,
            "token_ranks": {
                str(int(position)): int(
                    np.count_nonzero(mode_scores > mode_scores[row]) + 1
                )
                for row, position in zip(needle_rows, needle_positions, strict=True)
            },
            "selected_tokens": [
                int(value) for value in needle_positions if int(value) in mode_set
            ],
            "local_partition_ranks": _local_rank_metrics(
                scores=mode_scores,
                positions=positions,
                needle_rows=needle_rows,
            ),
            "max_score_block_expansion": _block_expansion_metrics(
                scores=mode_scores,
                positions=positions,
                needle_rows=needle_rows,
                topk=width,
            ),
            "coarse_boundary_policies": boundary_policies,
        },
    }


def run(
    *,
    trace_dir: Path,
    needle_fraction: float,
    needle_radius: int,
    needle_center: int | None = None,
) -> dict[str, Any]:
    trace = _load_trace(trace_dir)
    q = trace["q"]
    k = trace["k"]
    weights = trace["weights"]
    positions = trace["positions"]
    runtime_indices = trace["runtime_indices"]
    topk = int(np.count_nonzero(runtime_indices >= 0))
    if topk <= 0 or topk > k.shape[0]:
        raise RuntimeError(f"invalid runtime top-k width {topk}")

    # The paged prefill selector only scores K rows committed before the
    # current scheduler chunk.  Dense attention covers the in-flight chunk.
    # Including those tail rows in the CPU oracle makes the current chunk
    # dominate every top-k mode and compares a different problem than runtime.
    tail_start_position = int(trace["tail_start_position"])
    eligible = positions < tail_start_position
    if int(np.count_nonzero(eligible)) < topk:
        raise RuntimeError(
            "eligible prefill history is narrower than runtime top-k: "
            f"{int(np.count_nonzero(eligible))} < {topk}"
        )
    runtime_valid = runtime_indices[runtime_indices >= 0]
    if np.any(runtime_valid >= tail_start_position):
        raise RuntimeError(
            "runtime selection contains an in-flight chunk position; "
            f"tail starts at {tail_start_position}, "
            f"max runtime index is {int(np.max(runtime_valid))}"
        )
    k = k[eligible]
    positions = positions[eligible]

    oracle_scores = _selector_scores(q, k, weights)
    q_h = _normalized_hadamard(q)
    k_h = _normalized_hadamard(k)
    rotated_oracle_scores = _selector_scores(q_h, k_h, weights)
    orthogonal_delta = np.abs(rotated_oracle_scores - oracle_scores)

    q_raw_fp8 = _ue8m0_fp8_roundtrip(q, amax_floor=1e-10)
    k_raw_fp8 = _ue8m0_fp8_roundtrip(k)
    raw_scores = _selector_scores(q_raw_fp8, k_raw_fp8, weights)

    q_h_fp8 = _ue8m0_fp8_roundtrip(q_h, amax_floor=1e-10)
    k_h_fp8 = _ue8m0_fp8_roundtrip(k_h)
    hadamard_scores = _selector_scores(q_h_fp8, k_h_fp8, weights)

    q_split64_fp8 = _ue8m0_fp8_roundtrip(
        q, amax_floor=1e-10, group_size=64
    )
    k_split64_fp8 = _ue8m0_fp8_roundtrip(k, group_size=64)
    split64_scores = _selector_scores(q_split64_fp8, k_split64_fp8, weights)

    # Four power-of-two UE8M0 bytes fit in the existing four-byte scale slot,
    # so this precision cell has no cache-capacity cost.  It requires reader
    # support because the current slot is interpreted as one FP32 scale.
    q_split32_fp8 = _ue8m0_fp8_roundtrip(
        q, amax_floor=1e-10, group_size=32
    )
    k_split32_fp8 = _ue8m0_fp8_roundtrip(k, group_size=32)
    split32_scores = _selector_scores(q_split32_fp8, k_split32_fp8, weights)

    oracle_indices = _topk_positions(oracle_scores, positions, topk)
    raw_indices = _topk_positions(raw_scores, positions, topk)
    hadamard_indices = _topk_positions(hadamard_scores, positions, topk)
    split64_indices = _topk_positions(split64_scores, positions, topk)
    split32_indices = _topk_positions(split32_scores, positions, topk)

    return {
        "schema": "v20-indexer-hadamard-real-activation-proof-v2",
        "trace_dir": str(trace_dir),
        "claim_boundary": (
            "operator-level proof on one real layer/query; end-to-end cold "
            "needle ladder remains the model acceptance gate"
        ),
        "geometry": {
            "tokens": int(k.shape[0]),
            "heads": int(q.shape[0]),
            "head_dim": int(q.shape[1]),
            "topk": topk,
            "runtime_absolute_position": trace["runtime_absolute_position"],
            "eligible_history_end": int(positions[-1]),
            "tail_start_position": tail_start_position,
        },
        "fingerprints": {
            "q_bf16": _sha256(q),
            "k_bf16": _sha256(k),
            "weights_bf16": _sha256(weights),
            "oracle_scores": _sha256(oracle_scores),
            "raw_fp8_scores": _sha256(raw_scores),
            "hadamard_fp8_scores": _sha256(hadamard_scores),
            "split64_fp8_scores": _sha256(split64_scores),
            "split32_fp8_scores": _sha256(split32_scores),
        },
        "full_precision_orthogonality": {
            "max_abs_score_delta": float(np.max(orthogonal_delta)),
            "mean_abs_score_delta": float(np.mean(orthogonal_delta)),
            "relative_max": float(np.max(orthogonal_delta))
            / max(float(np.max(np.abs(oracle_scores))), np.finfo(np.float32).tiny),
        },
        "runtime_vs_cpu_raw_fp8": _set_metrics(runtime_indices, raw_indices),
        "full_precision_oracle": _mode_metrics(
            oracle_scores=oracle_scores,
            mode_scores=oracle_scores,
            oracle_indices=oracle_indices,
            mode_indices=oracle_indices,
            positions=positions,
            needle_fraction=needle_fraction,
            needle_radius=needle_radius,
            needle_reference_position=trace["runtime_absolute_position"],
            needle_center_override=needle_center,
        ),
        "raw_fp8": _mode_metrics(
            oracle_scores=oracle_scores,
            mode_scores=raw_scores,
            oracle_indices=oracle_indices,
            mode_indices=raw_indices,
            positions=positions,
            needle_fraction=needle_fraction,
            needle_radius=needle_radius,
            needle_reference_position=trace["runtime_absolute_position"],
            needle_center_override=needle_center,
        ),
        "hadamard_fp8": _mode_metrics(
            oracle_scores=oracle_scores,
            mode_scores=hadamard_scores,
            oracle_indices=oracle_indices,
            mode_indices=hadamard_indices,
            positions=positions,
            needle_fraction=needle_fraction,
            needle_radius=needle_radius,
            needle_reference_position=trace["runtime_absolute_position"],
            needle_center_override=needle_center,
        ),
        "split64_fp8": _mode_metrics(
            oracle_scores=oracle_scores,
            mode_scores=split64_scores,
            oracle_indices=oracle_indices,
            mode_indices=split64_indices,
            positions=positions,
            needle_fraction=needle_fraction,
            needle_radius=needle_radius,
            needle_reference_position=trace["runtime_absolute_position"],
            needle_center_override=needle_center,
        ),
        "split32_ue8m0_fp8": _mode_metrics(
            oracle_scores=oracle_scores,
            mode_scores=split32_scores,
            oracle_indices=oracle_indices,
            mode_indices=split32_indices,
            positions=positions,
            needle_fraction=needle_fraction,
            needle_radius=needle_radius,
            needle_reference_position=trace["runtime_absolute_position"],
            needle_center_override=needle_center,
        ),
        "hadamard_improvement": {
            "topk_intersection_gain": (
                _set_metrics(hadamard_indices, oracle_indices)["intersection"]
                - _set_metrics(raw_indices, oracle_indices)["intersection"]
            ),
            "rmse_ratio_hadamard_over_raw": float(
                np.sqrt(
                    np.mean(
                        np.square(hadamard_scores - oracle_scores, dtype=np.float64)
                    )
                )
                / np.sqrt(
                    np.mean(np.square(raw_scores - oracle_scores, dtype=np.float64))
                )
            ),
        },
        "split64_improvement": {
            "topk_intersection_gain": (
                _set_metrics(split64_indices, oracle_indices)["intersection"]
                - _set_metrics(raw_indices, oracle_indices)["intersection"]
            ),
            "rmse_ratio_split64_over_raw": float(
                np.sqrt(
                    np.mean(
                        np.square(split64_scores - oracle_scores, dtype=np.float64)
                    )
                )
                / np.sqrt(
                    np.mean(np.square(raw_scores - oracle_scores, dtype=np.float64))
                )
            ),
            "cache_record_bytes": 136,
            "current_cache_record_bytes": 132,
        },
        "split32_ue8m0_improvement": {
            "topk_intersection_gain": (
                _set_metrics(split32_indices, oracle_indices)["intersection"]
                - _set_metrics(raw_indices, oracle_indices)["intersection"]
            ),
            "rmse_ratio_split32_over_raw": float(
                np.sqrt(
                    np.mean(
                        np.square(split32_scores - oracle_scores, dtype=np.float64)
                    )
                )
                / np.sqrt(
                    np.mean(np.square(raw_scores - oracle_scores, dtype=np.float64))
                )
            ),
            "cache_record_bytes": 132,
            "current_cache_record_bytes": 132,
            "scale_contract": (
                "four UE8M0 bytes replace the current single FP32 scale"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--needle-fraction", type=float, default=0.4)
    parser.add_argument("--needle-radius", type=int, default=24)
    parser.add_argument(
        "--needle-center",
        type=int,
        help="Override the fraction-derived needle center with an exact token position.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run(
        trace_dir=args.trace_dir,
        needle_fraction=args.needle_fraction,
        needle_radius=args.needle_radius,
        needle_center=args.needle_center,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
