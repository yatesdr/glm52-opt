#!/usr/bin/env python3
"""Reproduce SparkInfer #85's cross-width cubin row-stride defect.

This is deliberately a no-model, one-GPU proof. It compiles each top-k kernel
at a narrow table width, reuses the same in-memory cubin at a wider width, and
checks all 17 rows. The pre-fix implementation keeps the first two-dimensional
CuTe row stride in the cubin; the fixed implementation receives the stride as
a runtime argument.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SPARKINFER_COMPILE_DISK_CACHE", "0")
os.environ.setdefault("SPARKINFER_COMPILE_MEMORY_CACHE", "1")

import torch

from sparkinfer._lib.compiler import clear_compile_cache, compile_cache_info
from sparkinfer.attention.nsa_indexer._impl import clear_indexer_caches
from sparkinfer.attention.nsa_indexer.tiled_topk import (
    run_row_topk,
    run_tiled_topk,
)


def _row_topk_gather_stride(device: torch.device) -> dict[str, object]:
    clear_compile_cache()
    clear_indexer_caches()
    rows, topk = 17, 512

    def select(width: int) -> tuple[bool, list[int]]:
        row_logits = (
            torch.arange(width, dtype=torch.float32, device=device)
            .expand(rows, -1)
            .contiguous()
        )
        lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
        gather_table = torch.arange(
            rows * width,
            dtype=torch.int32,
            device=device,
        ).reshape(rows, width)
        _, indices = run_row_topk(
            row_logits=row_logits,
            lengths=lengths,
            topk=topk,
            output_gather_table=gather_table,
        )
        torch.cuda.synchronize(device)
        logical = torch.topk(
            row_logits,
            k=topk,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        expected = torch.gather(gather_table, 1, logical)
        row_equal = torch.all(
            torch.sort(indices, dim=1).values
            == torch.sort(expected, dim=1).values,
            dim=1,
        )
        bad_rows = (~row_equal).nonzero(as_tuple=False).reshape(-1).tolist()
        return not bad_rows, bad_rows

    narrow_ok, narrow_bad_rows = select(512)
    misses_after_narrow = int(compile_cache_info()["compile_misses"])
    wide_ok, wide_bad_rows = select(1024)
    misses_after_wide = int(compile_cache_info()["compile_misses"])
    return {
        "narrow_ok": narrow_ok,
        "narrow_bad_rows": narrow_bad_rows,
        "wide_ok": wide_ok,
        "wide_bad_rows": wide_bad_rows,
        "compile_misses_after_narrow": misses_after_narrow,
        "compile_misses_after_wide": misses_after_wide,
        "cubin_reused": misses_after_wide == misses_after_narrow,
    }


def _paged_table_stride(device: torch.device) -> dict[str, object]:
    clear_compile_cache()
    clear_indexer_caches()
    rows, block_q, block_k, topk = 17, 32, 512, 512
    lengths = torch.full((rows,), 32, dtype=torch.int32, device=device)
    page_ids = torch.arange(1000, 1000 + rows, dtype=torch.int32, device=device)
    tile_logits = torch.zeros(
        block_q * block_k,
        dtype=torch.float32,
        device=device,
    )

    def expected(selected_page_ids: torch.Tensor) -> torch.Tensor:
        result = torch.full(
            (rows, topk),
            -1,
            dtype=torch.int32,
            device=device,
        )
        result[:, :32] = selected_page_ids[:, None] * 64 + torch.arange(
            32,
            dtype=torch.int32,
            device=device,
        )[None, :]
        return torch.sort(result, dim=1).values

    def select(width: int, *, shared: bool = False) -> tuple[bool, list[int]]:
        selected_page_ids = page_ids
        if shared:
            selected_page_ids = torch.full_like(page_ids, 2000)
            page_table = torch.full(
                (1, width),
                -1,
                dtype=torch.int32,
                device=device,
            )
            page_table[:, 0] = selected_page_ids[0]
            page_table = page_table.expand(rows, -1)
        else:
            page_table = torch.full(
                (rows, width),
                -1,
                dtype=torch.int32,
                device=device,
            )
            page_table[:, 0] = selected_page_ids
        _, indices = run_tiled_topk(
            tile_logits=tile_logits,
            k_start=None,
            lengths=lengths,
            topk=topk,
            block_q=block_q,
            block_k=block_k,
            num_k_tiles=1,
            zero_row_start=True,
            output_page_table=page_table,
            output_page_size=64,
        )
        torch.cuda.synchronize(device)
        row_equal = torch.all(
            torch.sort(indices, dim=1).values == expected(selected_page_ids),
            dim=1,
        )
        bad_rows = (~row_equal).nonzero(as_tuple=False).reshape(-1).tolist()
        return not bad_rows, bad_rows

    narrow_ok, narrow_bad_rows = select(1)
    misses_after_narrow = int(compile_cache_info()["compile_misses"])
    wide_ok, wide_bad_rows = select(8192)
    shared_ok, shared_bad_rows = select(8192, shared=True)
    misses_after_reuse = int(compile_cache_info()["compile_misses"])
    return {
        "narrow_ok": narrow_ok,
        "narrow_bad_rows": narrow_bad_rows,
        "wide_ok": wide_ok,
        "wide_bad_rows": wide_bad_rows,
        "shared_ok": shared_ok,
        "shared_bad_rows": shared_bad_rows,
        "compile_misses_after_narrow": misses_after_narrow,
        "compile_misses_after_reuse": misses_after_reuse,
        "cubin_reused": misses_after_reuse == misses_after_narrow,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--expect",
        choices=("bug-present", "bug-absent"),
        required=True,
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    report: dict[str, object] = {
        "schema": "sparkinfer-runtime-stride-cross-width-v1",
        "device": str(device),
        "expect": args.expect,
    }
    try:
        report["row_topk_gather"] = _row_topk_gather_stride(device)
        report["paged_tiled_topk"] = _paged_table_stride(device)
        row = report["row_topk_gather"]
        paged = report["paged_tiled_topk"]
        assert isinstance(row, dict)
        assert isinstance(paged, dict)
        fixed = bool(
            row["narrow_ok"]
            and row["wide_ok"]
            and row["cubin_reused"]
            and paged["narrow_ok"]
            and paged["wide_ok"]
            and paged["shared_ok"]
            and paged["cubin_reused"]
        )
        report["observed"] = "bug-absent" if fixed else "bug-present"
        report["status"] = "PASS" if report["observed"] == args.expect else "FAIL"
    except Exception as exc:
        report["observed"] = "exception"
        report["exception"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "FAIL"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
