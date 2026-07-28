#!/usr/bin/env python3
"""Dependency-free audit of the current v20 DCP PR #177/#178 candidates.

This is deliberately a source-and-math proof rather than a substitute for the
GPU gates.  It answers two questions before spending a model boot:

* how PR #177 changes the persistent CKV allocation for the production shape;
* whether PR #178's row-owner merge is mathematically the same rank-major FP32
  top-k operation as the replicated oracle, and what it changes in transport,
  merge work, and temporary workspace.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path


PR177_COMMIT = "affff57c0dd482c356e96bc6c774fbd3a3e1e69d"
PR178_COMMIT = "b6fe79ded5878269c2e488dd51e2ce074e43cd26"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(tree: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class CKVShape:
    max_tokens: int = 480_000
    dcp: int = 4
    max_seqs: int = 16
    interleave: int = 1
    block: int = 64
    record_bytes: int = 368
    speculative: bool = True
    num_ubatches: int = 1

    @property
    def local_capacity(self) -> int:
        local_tokens = _ceil_div(self.max_tokens, self.dcp)
        local_tokens += self.max_seqs * self.interleave
        return _ceil_div(local_tokens, self.block) * self.block

    @property
    def execution_lanes(self) -> int:
        return max(1, self.num_ubatches) * (2 if self.speculative else 1)

    def old_workspace_bytes(self) -> int:
        # The pre-#177 singleton is a two-buffer ping-pong allocation. Each
        # half has one local staging region plus DCP gathered output.
        return (
            2
            * (self.dcp + 1)
            * self.local_capacity
            * self.record_bytes
        )

    def pr177_per_lane_bytes(self, depth: int) -> int:
        # One shared local staging region plus depth+1 gathered ring slots.
        ring_slots = max(0, depth) + 1
        return (
            (1 + ring_slots * self.dcp)
            * self.local_capacity
            * self.record_bytes
        )

    def pr177_total_bytes(self, depth: int) -> int:
        return self.execution_lanes * self.pr177_per_lane_bytes(depth)


def _topk_rank_major(
    candidates: list[list[tuple[float, int, int, int]]],
    k: int,
) -> list[list[int]]:
    """Stable rank-major top-k oracle.

    Each tuple is ``(score, global_id, source_rank, source_position)``.  CUDA's
    exact tie policy is implementation-specific; using rank/position as the
    final key gives both simulated routes the same explicit rank-major policy,
    including tie-heavy cases.
    """

    merged: list[list[int]] = []
    for row in candidates:
        ordered = sorted(row, key=lambda x: (-x[0], x[2], x[3]))
        merged.append([item[1] for item in ordered[:k]])
    return merged


def _make_candidates(
    rng: random.Random,
    *,
    rows: int,
    dcp: int,
    topk: int,
    tie_heavy: bool,
) -> list[list[list[tuple[float, int, int, int]]]]:
    per_rank: list[list[list[tuple[float, int, int, int]]]] = []
    for rank in range(dcp):
        rank_rows: list[list[tuple[float, int, int, int]]] = []
        for row in range(rows):
            values = []
            for position in range(topk):
                score = (
                    float(rng.randrange(8))
                    if tie_heavy
                    else rng.uniform(-20.0, 20.0)
                )
                global_id = rank * 10_000_000 + row * topk + position
                values.append((score, global_id, rank, position))
            rank_rows.append(values)
        per_rank.append(rank_rows)
    return per_rank


def _prove_owner_merge_equivalence() -> int:
    cases = 0
    for seed in range(20):
        for dcp in (2, 4):
            for rows_per_owner in (1, 2, 7):
                rows = dcp * rows_per_owner
                for topk in (2, 7, 32):
                    for tie_heavy in (False, True):
                        rng = random.Random(
                            (seed << 16)
                            ^ (dcp << 12)
                            ^ (rows << 6)
                            ^ topk
                            ^ int(tie_heavy)
                        )
                        per_rank = _make_candidates(
                            rng,
                            rows=rows,
                            dcp=dcp,
                            topk=topk,
                            tie_heavy=tie_heavy,
                        )

                        # Existing oracle: every rank merges every row after a
                        # rank-major all-gather.
                        all_rows = [
                            [
                                candidate
                                for rank in range(dcp)
                                for candidate in per_rank[rank][row]
                            ]
                            for row in range(rows)
                        ]
                        oracle = _topk_rank_major(all_rows, topk)

                        # PR #178: owner j receives rows
                        # [j*rows_per_owner:(j+1)*rows_per_owner] from every
                        # source rank in the same rank-major order, merges only
                        # those, then TP all-gather concatenates owners.
                        owner_outputs: list[list[int]] = []
                        for owner in range(dcp):
                            start = owner * rows_per_owner
                            stop = start + rows_per_owner
                            owner_rows = [
                                [
                                    candidate
                                    for rank in range(dcp)
                                    for candidate in per_rank[rank][row]
                                ]
                                for row in range(start, stop)
                            ]
                            owner_outputs.extend(
                                _topk_rank_major(owner_rows, topk)
                            )

                        assert owner_outputs == oracle
                        cases += 1
    return cases


@dataclass(frozen=True)
class OwnerMergeEconomics:
    rows: int = 3072
    topk: int = 2048
    dcp: int = 4
    item_bytes: int = 4

    @property
    def owner_rows(self) -> int:
        assert self.rows % self.dcp == 0
        return self.rows // self.dcp

    def oracle_workspace_bytes(self) -> int:
        candidate_pair = self.rows * 2 * self.topk * self.item_bytes
        gathered_pair = self.dcp * candidate_pair
        merged_ids = self.rows * self.dcp * self.topk * self.item_bytes
        merged_scores = merged_ids
        lengths = self.rows * self.item_bytes
        return (
            candidate_pair
            + gathered_pair
            + merged_ids
            + merged_scores
            + lengths
        )

    def owner_workspace_bytes(self) -> int:
        candidate_pair = self.rows * 2 * self.topk * self.item_bytes
        received_pair = candidate_pair
        merged_ids = self.owner_rows * self.dcp * self.topk * self.item_bytes
        merged_scores = merged_ids
        lengths = self.owner_rows * self.item_bytes
        owner_outputs = (
            3 * self.owner_rows * self.topk * self.item_bytes
        )
        return (
            candidate_pair
            + received_pair
            + merged_ids
            + merged_scores
            + lengths
            + owner_outputs
        )

    def oracle_received_bytes_per_rank(self) -> int:
        local_pair = self.rows * 2 * self.topk * self.item_bytes
        return (self.dcp - 1) * local_pair

    def owner_received_bytes_per_rank(self) -> int:
        # Equal all-to-all retains 1/dcp locally, plus TP all-gather of the
        # final int32 ids retains one owner shard locally.
        local_pair = self.rows * 2 * self.topk * self.item_bytes
        a2a_remote = local_pair * (self.dcp - 1) // self.dcp
        owner_ids = self.owner_rows * self.topk * self.item_bytes
        final_ag_remote = (self.dcp - 1) * owner_ids
        return a2a_remote + final_ag_remote


def _mib(value: int) -> str:
    return f"{value / (1024 * 1024):.2f} MiB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pr177-tree",
        type=Path,
        default=Path("workspace/vllm-pr177-audit"),
    )
    parser.add_argument(
        "--pr178-tree",
        type=Path,
        default=Path("workspace/vllm-pr178-audit"),
    )
    args = parser.parse_args()

    assert _git_head(args.pr177_tree) == PR177_COMMIT
    assert _git_head(args.pr178_tree) == PR178_COMMIT

    pr177_source = (
        args.pr177_tree
        / "vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
    )
    pr178_source = (
        args.pr178_tree
        / "vllm/model_executor/layers/sparse_attn_indexer.py"
    )
    pr177_names = _function_names(pr177_source)
    pr178_names = _function_names(pr178_source)
    assert {
        "_ckv_prefetch_workspace_nbytes",
        "_ckv_prefetch_execution_lanes",
        "_ckv_prefetch_depth_within_budget",
        "_ckv_workspace_identity",
    } <= pr177_names
    assert {
        "_dcp_all_to_all_first_dim_into",
        "_merge_b12x_dcp_topk_by_owner",
        "_merge_b12x_prefill_dcp_topk",
    } <= pr178_names

    shape = CKVShape()
    old_bytes = shape.old_workspace_bytes()
    depth0 = shape.pr177_total_bytes(0)
    depth1 = shape.pr177_total_bytes(1)
    assert old_bytes == depth0
    assert depth1 > old_bytes

    equivalence_cases = _prove_owner_merge_equivalence()
    economics = OwnerMergeEconomics()
    oracle_ws = economics.oracle_workspace_bytes()
    owner_ws = economics.owner_workspace_bytes()
    oracle_rx = economics.oracle_received_bytes_per_rank()
    owner_rx = economics.owner_received_bytes_per_rank()
    assert owner_ws < oracle_ws
    assert owner_rx < oracle_rx

    print("PASS: v20 upstream DCP PR #177/#178 source and math audit")
    print(f"  PR177 source sha256: {_sha256(pr177_source)}")
    print(f"  PR178 source sha256: {_sha256(pr178_source)}")
    print(
        "  production CKV shape: "
        f"local_capacity={shape.local_capacity} records, "
        f"lanes={shape.execution_lanes}"
    )
    print(f"  current singleton ping-pong: {_mib(old_bytes)}")
    print(f"  PR177 depth=0 total:          {_mib(depth0)}")
    print(f"  PR177 depth=1 total:          {_mib(depth1)}")
    print(
        "  PR177 default persistent delta: "
        f"+{_mib(depth1 - old_bytes)}"
    )
    print(
        "  PR178 exact owner/oracle cases: "
        f"{equivalence_cases} (including tied scores)"
    )
    print(
        "  PR178 3072x2048 workspace: "
        f"{_mib(oracle_ws)} -> {_mib(owner_ws)} "
        f"({100.0 * (oracle_ws - owner_ws) / oracle_ws:.1f}% less)"
    )
    print(
        "  PR178 received bytes/rank: "
        f"{_mib(oracle_rx)} -> {_mib(owner_rx)} "
        f"({100.0 * (oracle_rx - owner_rx) / oracle_rx:.1f}% less)"
    )
    print(
        "  PR178 merge rows/rank: "
        f"{economics.rows} -> {economics.owner_rows} "
        f"({economics.dcp}x less top-k work)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
