#!/usr/bin/env python3
"""CPU contract proof for synchronous CKV workspace borrowing."""

from __future__ import annotations

import json

import torch

from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xMLASparseImpl,
    _ckv_prefetch_uses_persistent_workspace,
)


def main() -> int:
    assert not _ckv_prefetch_uses_persistent_workspace(0)
    assert _ckv_prefetch_uses_persistent_workspace(1)

    impl = object.__new__(B12xMLASparseImpl)
    impl._pad_heads = False
    impl._ckv_borrow_sync_workspace = True
    impl._max_batched = 4
    impl._kernel_num_heads = 2
    impl.q_head_dim = 8
    impl._scratch_nbytes = 64
    impl._ckv_workspace_nbytes = 128
    impl._ckv_gather_enabled = True
    impl.device = torch.device("cpu")

    q_workspace = torch.empty((4, 2, 8), dtype=torch.bfloat16)
    ckv_workspace = torch.empty(128, dtype=torch.uint8)
    scratch_workspace = torch.empty(64, dtype=torch.uint8)
    impl._borrow_workspaces = lambda: [
        q_workspace,
        ckv_workspace,
        scratch_workspace,
    ]

    q, dense, scratch, ckv = impl._borrow_workspace_parts()
    assert q is q_workspace
    assert dense is None
    assert scratch is scratch_workspace
    assert ckv is ckv_workspace
    assert len({q.data_ptr(), scratch.data_ptr(), ckv.data_ptr()}) == 3

    print(
        json.dumps(
            {
                "depth0_persistent": False,
                "depth1_persistent": True,
                "borrowed_sync_ckv_bytes": ckv.numel(),
                "simultaneous_views_disjoint": True,
                "verdict": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
