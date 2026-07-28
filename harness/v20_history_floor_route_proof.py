#!/usr/bin/env python3
"""Fail-closed routing proof for the DCP indexer history-floor policy.

The frozen long-context request exercises the query-split prefill owner merge,
not the decode merge.  This proof imports the packaged image implementation,
replaces only the collective implementations with recording stubs, and proves
that the configured policy reaches both the owner route and its replicated
fallback.  It deliberately requires no model weights or GPU allocation.
"""

from __future__ import annotations

import json
import os

import torch

from vllm.model_executor.layers import sparse_attn_indexer as indexer_mod


def run_case(*, owner_enabled: bool, owner_result: bool) -> dict[str, object]:
    owner_calls: list[dict[str, object]] = []
    oracle_calls: list[dict[str, object]] = []

    def fake_owner(**kwargs):
        owner_calls.append(kwargs)
        return owner_result

    def fake_oracle(**kwargs):
        oracle_calls.append(kwargs)

    original_enabled = indexer_mod.envs.VLLM_DCP_TOPK_OWNER_MERGE
    original_owner = indexer_mod._merge_b12x_dcp_topk_by_owner
    original_oracle = indexer_mod._merge_b12x_dcp_topk
    try:
        indexer_mod.envs.VLLM_DCP_TOPK_OWNER_MERGE = owner_enabled
        indexer_mod._merge_b12x_dcp_topk_by_owner = fake_owner
        indexer_mod._merge_b12x_dcp_topk = fake_oracle
        indices = torch.empty((4, 2048), dtype=torch.int32)
        scores = torch.empty((4, 2048), dtype=torch.float32)
        used = indexer_mod._merge_b12x_prefill_dcp_topk(
            topk_indices=indices,
            topk_scores=scores,
            gathered_topk_indices=indices,
            topk_tokens=2048,
            dcp_world_size=4,
            dcp_rank=0,
            cp_kv_cache_interleave_size=1,
        )
    finally:
        indexer_mod.envs.VLLM_DCP_TOPK_OWNER_MERGE = original_enabled
        indexer_mod._merge_b12x_dcp_topk_by_owner = original_owner
        indexer_mod._merge_b12x_dcp_topk = original_oracle

    expected_owner_calls = int(owner_enabled)
    expected_oracle_calls = int(not (owner_enabled and owner_result))
    if used is not bool(owner_enabled and owner_result):
        raise RuntimeError("prefill wrapper returned the wrong owner-route result")
    if len(owner_calls) != expected_owner_calls:
        raise RuntimeError(
            f"owner calls {len(owner_calls)} != expected {expected_owner_calls}"
        )
    if len(oracle_calls) != expected_oracle_calls:
        raise RuntimeError(
            f"oracle calls {len(oracle_calls)} != expected {expected_oracle_calls}"
        )
    routed = owner_calls + oracle_calls
    if any(call.get("history_floor") != 64 for call in routed):
        raise RuntimeError("history floor did not reach every selected prefill route")

    return {
        "owner_enabled": owner_enabled,
        "owner_result": owner_result,
        "owner_calls": len(owner_calls),
        "oracle_calls": len(oracle_calls),
        "history_floor": 64,
        "passed": True,
    }


def main() -> int:
    if os.environ.get("VLLM_DCP_INDEXER_HISTORY_FLOOR") != "64":
        raise SystemExit("set VLLM_DCP_INDEXER_HISTORY_FLOOR=64")
    rows = [
        run_case(owner_enabled=False, owner_result=True),
        run_case(owner_enabled=True, owner_result=False),
        run_case(owner_enabled=True, owner_result=True),
    ]
    print(
        json.dumps(
            {
                "schema": "v20-history-floor-route-proof-v1",
                "cases": rows,
                "passed": all(row["passed"] for row in rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
