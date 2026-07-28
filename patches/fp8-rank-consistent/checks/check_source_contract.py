#!/usr/bin/env python3
"""Static gate for the FP8 rank-consistency overlay."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
BASE = REPO / "workspace/upstream-b12x-v19-kernel/b12x/distributed/pcie_dma.py"
OVERLAY = ROOT / "overlays/b12x/distributed/pcie_dma.py"
EXPECTED_BASE_MD5 = "96e07e55c3843766999b88e184ce06dd"
EXPECTED_OVERLAY_MD5 = "830d136566ec3383fd100eb997aa11b9"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    if BASE.exists():
        assert md5(BASE) == EXPECTED_BASE_MD5, (
            f"input byte drift: expected {EXPECTED_BASE_MD5}, got {md5(BASE)}"
        )
        assert BASE.read_text().count("ext.dma_dequant_store(") == 2
    overlay = OVERLAY.read_text()
    assert md5(OVERLAY) == EXPECTED_OVERLAY_MD5, (
        f"overlay byte drift: expected {EXPECTED_OVERLAY_MD5}, got {md5(OVERLAY)}"
    )
    ast.parse(overlay)

    # One remote materialization exists in the ring and a2a paths in the
    # pinned base.  The patch adds exactly one owner materialization to each.
    assert overlay.count("ext.dma_dequant_store(") == 4
    assert "owner_stage = fp8_stage_piece(send_chunk, p)" in overlay
    assert "piece_ptr(send_chunk, p),\n                        owner_stage," in overlay
    assert "out_base + own,\n                stage_chunk(rank, c)," in overlay

    # The payload-ready events must be recorded before the local stores so the
    # existing CE copies can overlap them.
    assert overlay.index("self._ag_ready.record(main)") < overlay.index(
        "owner_stage = fp8_stage_piece(send_chunk, p)"
    )
    a2a_event = overlay.index("self._a2a_ownq[c].record(main)")
    a2a_owner_store = overlay.index(
        "# Peers materialize this reduced shard", a2a_event
    )
    assert a2a_event < a2a_owner_store

    # No CUDA source change is part of this bundle.
    assert not (ROOT / "overlays/b12x/distributed/pcie_dma.cu").exists()
    print(
        f"PASS base_md5={EXPECTED_BASE_MD5} "
        f"overlay_md5={EXPECTED_OVERLAY_MD5}"
    )


if __name__ == "__main__":
    main()
