#!/usr/bin/env python3
"""Audit deep-needle selection coverage across all captured model layers.

The long-context trace stores the final active query row's selected logical
token IDs at each attention boundary.  This audit compares a known-good
reference trace with a failing v20 trace without replaying the model.  It does
not infer scores that were not captured; it answers the narrower questions
needed to design the next selector experiment:

* where does the reference first carry needle-local tokens that v20 omits?
* for how many later layers does that omission persist?
* how much chronological coverage does each selected set allocate?
* are selections rank-replicated or rank-local?

The output is deterministic JSON suitable for evidence hashing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


_LAYER_RE = re.compile(r"layer(\d+)-attention\.pt$")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _load_root(root: Path, ranks: int) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for rank in range(ranks):
        rank_dir = root / f"tp{rank}"
        for path in sorted(rank_dir.glob("layer*-attention.pt")):
            match = _LAYER_RE.search(path.name)
            if match is None:
                continue
            record = torch.load(path, map_location="cpu", weights_only=True)
            if "topk_indices" not in record:
                continue
            layer = int(match.group(1))
            if int(record.get("layer", -1)) != layer:
                raise RuntimeError(f"{path}: layer metadata disagrees with path")
            if int(record.get("tp_rank", -1)) != rank:
                raise RuntimeError(f"{path}: rank metadata disagrees with path")
            key = (layer, rank)
            if key in records:
                raise RuntimeError(f"{path}: duplicate trace record {key}")
            records[key] = record
    if not records:
        raise RuntimeError(f"{root}: no attention records with topk_indices")
    return records


def _valid_indices(record: dict[str, Any]) -> torch.Tensor:
    indices = record["topk_indices"].to(torch.int64).flatten()
    return indices[indices >= 0]


def _position_report(indices: torch.Tensor, history_end: int) -> dict[str, Any]:
    if history_end <= 0:
        raise ValueError("history_end must be positive")
    quartiles = [
        int(
            (
                (indices >= history_end * quarter // 4)
                & (indices < history_end * (quarter + 1) // 4)
            ).sum()
        )
        for quarter in range(4)
    ]
    return {
        "count": int(indices.numel()),
        "quartiles": quartiles,
        "older_half": quartiles[0] + quartiles[1],
        "newest_quarter": quartiles[3],
        "minimum": int(indices.min()) if indices.numel() else None,
        "maximum": int(indices.max()) if indices.numel() else None,
        "sha256": _tensor_sha256(indices),
    }


def _needle_tokens(
    indices: torch.Tensor,
    *,
    needle_position: int,
    needle_radius: int,
) -> list[int]:
    values = indices[(indices - needle_position).abs() <= needle_radius]
    return sorted({int(value) for value in values.tolist()})


def _union(rows: list[torch.Tensor]) -> torch.Tensor:
    if not rows:
        return torch.empty(0, dtype=torch.int64)
    return torch.unique(torch.cat(rows), sorted=True)


def _jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    intersection = int(torch.isin(left, right).sum())
    union = int(torch.unique(torch.cat((left, right))).numel())
    return 1.0 if union == 0 else intersection / union


def run(
    *,
    reference_root: Path,
    candidate_root: Path,
    ranks: int,
    needle_position: int,
    needle_radius: int,
) -> dict[str, Any]:
    reference = _load_root(reference_root, ranks)
    candidate = _load_root(candidate_root, ranks)
    reference_layers = {layer for layer, _ in reference}
    candidate_layers = {layer for layer, _ in candidate}
    common_layers = sorted(reference_layers & candidate_layers)
    if not common_layers:
        raise RuntimeError("reference and candidate have no common sparse layers")

    expected = {(layer, rank) for layer in common_layers for rank in range(ranks)}
    missing_reference = sorted(expected - set(reference))
    missing_candidate = sorted(expected - set(candidate))
    if missing_reference or missing_candidate:
        raise RuntimeError(
            "incomplete rank coverage: "
            f"reference={missing_reference}, candidate={missing_candidate}"
        )

    layers: list[dict[str, Any]] = []
    first_reference_present_candidate_absent: int | None = None
    persistent_loss_layers: list[int] = []
    for layer in common_layers:
        ref_rows: list[torch.Tensor] = []
        cand_rows: list[torch.Tensor] = []
        rank_reports: list[dict[str, Any]] = []
        absolute_positions: set[int] = set()
        for rank in range(ranks):
            ref_record = reference[(layer, rank)]
            cand_record = candidate[(layer, rank)]
            ref_abs = int(ref_record["absolute_position"])
            cand_abs = int(cand_record["absolute_position"])
            if ref_abs != cand_abs:
                raise RuntimeError(
                    f"layer {layer} rank {rank}: absolute position mismatch "
                    f"{ref_abs} != {cand_abs}"
                )
            absolute_positions.add(ref_abs)
            ref_indices = _valid_indices(ref_record)
            cand_indices = _valid_indices(cand_record)
            ref_rows.append(ref_indices)
            cand_rows.append(cand_indices)
            rank_reports.append(
                {
                    "rank": rank,
                    "reference": _position_report(ref_indices, ref_abs + 1),
                    "candidate": _position_report(cand_indices, ref_abs + 1),
                    "reference_needle_tokens": _needle_tokens(
                        ref_indices,
                        needle_position=needle_position,
                        needle_radius=needle_radius,
                    ),
                    "candidate_needle_tokens": _needle_tokens(
                        cand_indices,
                        needle_position=needle_position,
                        needle_radius=needle_radius,
                    ),
                    "jaccard": _jaccard(ref_indices, cand_indices),
                }
            )
        if len(absolute_positions) != 1:
            raise RuntimeError(
                f"layer {layer}: ranks disagree on absolute position "
                f"{sorted(absolute_positions)}"
            )
        absolute_position = next(iter(absolute_positions))
        ref_union = _union(ref_rows)
        cand_union = _union(cand_rows)
        ref_needle = _needle_tokens(
            ref_union,
            needle_position=needle_position,
            needle_radius=needle_radius,
        )
        cand_needle = _needle_tokens(
            cand_union,
            needle_position=needle_position,
            needle_radius=needle_radius,
        )
        lost = bool(ref_needle) and not bool(cand_needle)
        if lost:
            persistent_loss_layers.append(layer)
            if first_reference_present_candidate_absent is None:
                first_reference_present_candidate_absent = layer
        layers.append(
            {
                "layer": layer,
                "absolute_position": absolute_position,
                "reference_rank_replicated": all(
                    torch.equal(ref_rows[0], row) for row in ref_rows[1:]
                ),
                "candidate_rank_replicated": all(
                    torch.equal(cand_rows[0], row) for row in cand_rows[1:]
                ),
                "reference_union": _position_report(
                    ref_union, absolute_position + 1
                ),
                "candidate_union": _position_report(
                    cand_union, absolute_position + 1
                ),
                "reference_needle_tokens": ref_needle,
                "candidate_needle_tokens": cand_needle,
                "reference_present_candidate_absent": lost,
                "rank_reports": rank_reports,
            }
        )

    return {
        "schema": "v20-indexer-cross-layer-selection-audit-v1",
        "claim_boundary": (
            "selection-coverage audit of captured final-query rows; it does not "
            "reconstruct unselected scores or prove a replacement policy"
        ),
        "reference_root": str(reference_root),
        "candidate_root": str(candidate_root),
        "ranks": ranks,
        "needle_position": needle_position,
        "needle_radius": needle_radius,
        "common_sparse_layers": common_layers,
        "first_reference_present_candidate_absent": (
            first_reference_present_candidate_absent
        ),
        "persistent_loss_layers": persistent_loss_layers,
        "persistent_loss_count": len(persistent_loss_layers),
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--needle-position", type=int, required=True)
    parser.add_argument("--needle-radius", type=int, default=24)
    args = parser.parse_args()

    report = run(
        reference_root=args.reference_root,
        candidate_root=args.candidate_root,
        ranks=args.ranks,
        needle_position=args.needle_position,
        needle_radius=args.needle_radius,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "schema",
                    "common_sparse_layers",
                    "first_reference_present_candidate_absent",
                    "persistent_loss_layers",
                    "persistent_loss_count",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
