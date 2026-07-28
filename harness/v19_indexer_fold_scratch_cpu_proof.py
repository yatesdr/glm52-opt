#!/usr/bin/env python3
"""CPU-only proof for the paged-indexer two-level fold scratch reservation.

The change moves three buffers in b12x/attention/indexer/paged.py::index_topk_fp8
from bare torch.empty (~384 MiB/call, unprofiled) into the pre-reserved indexer
scratch. The scratch is one flat byte block carved at fixed offsets, so the
failure mode that matters is a layout mistake that overlaps an EXISTING buffer —
which would silently corrupt the indexer rather than crash.

This compares the installed (original) layout against the patched one, field by
field, and proves:

  1. every pre-existing offset is byte-identical
  2. the new regions sit entirely beyond the original block (no overlap)
  3. nbytes grows by exactly the fold reservation and nothing else
  4. reserved capacity >= what paged.py asks for at runtime, across widths
  5. carved views have the right shape/dtype and are contiguous

Run inside the container. Touches no CUDA.
    python3 v19_indexer_fold_scratch_cpu_proof.py --patched /tmp/patched_scratch.py
Exit 0 = all passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import torch

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# production geometry on cn3: GLM-5.2 TP4/DCP4, 480k context, page size 64
CAPS = dict(
    num_q_heads=64 // 4,
    topk=2048,
    max_q_rows=3072,
    max_page_table_width=(480_000 + 63) // 64,
    mode="prefill",
    shared_page_table=True,
)


def build_caps(mod, **over):
    """Build the PAGED caps the layout function actually consumes."""
    kw = dict(CAPS); kw.update(over)
    return mod.B12XIndexerPagedScratchCaps(
        device=torch.device("cpu"),
        num_q_heads=kw["num_q_heads"],
        max_q_rows=kw["max_q_rows"],
        max_page_table_width=kw["max_page_table_width"],
        topk=kw["topk"],
        mode=kw["mode"],
        shared_page_table=kw["shared_page_table"],
    )


def paged_runtime_total_slices(width_tokens: int, page_size: int,
                               supertile_tokens: int, num_chunks: int) -> int:
    """Verbatim reimplementation of paged.py::index_topk_fp8's slicing loop."""
    SLICE, MAXSL = 16384, 32
    if width_tokens < 2 * SLICE:
        return 0
    slice_tokens = SLICE
    min_slice = -(-width_tokens // MAXSL)
    if min_slice > slice_tokens:
        slice_tokens = -(-min_slice // page_size) * page_size
    page_table_width = width_tokens // page_size
    supertile_pages = max(supertile_tokens // page_size, 1)
    base = 0
    for c in range(num_chunks):
        c_pages = min((c + 1) * supertile_pages, page_table_width) - c * supertile_pages
        if c_pages <= 0:
            break
        base += max(1, -(-(c_pages * page_size) // slice_tokens))
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patched", required=True, help="path to the patched scratch.py")
    args = ap.parse_args()

    import b12x.attention.indexer.scratch as orig
    patched = load_module(args.patched, "patched_indexer_scratch")

    print("\n=== 1. existing offsets must not move ===")
    lo = orig._indexer_paged_scratch_layout(build_caps(orig))
    lp = patched._indexer_paged_scratch_layout(build_caps(patched))

    offset_fields = [f for f in lo.__dataclass_fields__ if f.endswith("_offset_bytes")]
    moved = [f for f in offset_fields if getattr(lo, f) != getattr(lp, f, None)]
    check(f"all {len(offset_fields)} pre-existing offsets unchanged", not moved,
          "" if not moved else f"MOVED: {moved}")

    other = [f for f in lo.__dataclass_fields__
             if not f.endswith("_offset_bytes") and f != "nbytes"]
    diff = [f for f in other if getattr(lo, f) != getattr(lp, f, None)]
    check("all non-offset layout fields unchanged", not diff,
          "" if not diff else f"CHANGED: {diff}")

    print("\n=== 2. new regions sit beyond the original block ===")
    S = int(lp.two_level_fold_slices)
    check("two-level path engages at 480k", S > 0, f"S_max={S}")
    new_offsets = {
        "values": lp.two_level_fold_values_offset_bytes,
        "indices": lp.two_level_fold_indices_offset_bytes,
        "lengths": lp.two_level_fold_lengths_offset_bytes,
    }
    for nm, off in new_offsets.items():
        check(f"fold {nm} starts at/after the original end", off >= lo.nbytes,
              f"offset={off:,} orig_nbytes={lo.nbytes:,}")

    print("\n=== 3. nbytes grows by exactly the fold reservation ===")
    rows = CAPS["max_q_rows"] * S
    expect = (rows * CAPS["topk"] * 4) + (rows * CAPS["topk"] * 4) + (CAPS["max_q_rows"] * 4)
    grew = lp.nbytes - lo.nbytes
    check("growth matches the reservation (within alignment)",
          expect <= grew <= expect + 4096,
          f"grew {grew:,} B ({grew/2**20:.1f} MiB), expected ~{expect:,} B "
          f"({expect/2**20:.1f} MiB)")

    print("\n=== 4. reserved capacity >= runtime demand, across widths ===")
    ps = 64
    for width_tokens in (32_768, 65_536, 131_072, 262_144, 480_000, 480_000 - 64):
        caps_w = build_caps(patched, max_page_table_width=(width_tokens + ps - 1) // ps)
        lw = patched._indexer_paged_scratch_layout(caps_w)
        want = paged_runtime_total_slices(
            width_tokens, ps, int(lw.supertile_tokens), int(lw.max_chunks))
        have = int(lw.two_level_fold_slices)
        check(f"width {width_tokens:>7,}: reserved {have} >= runtime {want}",
              have >= want, f"reserved={have} runtime={want}")

    print("\n=== 5. short context reserves nothing ===")
    caps_s = build_caps(patched, max_page_table_width=(16_384 + ps - 1) // ps)
    ls = patched._indexer_paged_scratch_layout(caps_s)
    check("no fold reservation below the two-level threshold",
          int(ls.two_level_fold_slices) == 0, f"slices={int(ls.two_level_fold_slices)}")

    print("\n=== 6. carved views: shape, dtype, contiguity, no overlap ===")
    storage = torch.zeros(lp.nbytes, dtype=torch.uint8)
    sc = patched._materialize_indexer_paged_scratch(build_caps(patched), storage, lp)
    v, i, ln = sc.two_level_fold_values, sc.two_level_fold_indices, sc.two_level_fold_lengths
    check("values view float32 + contiguous", v.dtype == torch.float32 and v.is_contiguous())
    check("indices view int32 + contiguous", i.dtype == torch.int32 and i.is_contiguous())
    check("lengths view int32 + contiguous", ln.dtype == torch.int32 and ln.is_contiguous())
    check("values shape == (max_q_rows*S, topk)", tuple(v.shape) == (rows, CAPS["topk"]),
          str(tuple(v.shape)))

    def span(t):
        return (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
    spans = {"values": span(v), "indices": span(i), "lengths": span(ln)}
    ok = True
    names = list(spans)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            (s1, e1), (s2, e2) = spans[names[a]], spans[names[b]]
            if s1 < e2 and s2 < e1:
                ok = False
                print(f"        OVERLAP: {names[a]} vs {names[b]}")
    check("the three fold views do not overlap each other", ok)

    # accessor bounds
    try:
        sc.get_two_level_fold_buffers(row_count=CAPS["max_q_rows"], total_slices=S + 1)
        check("accessor rejects over-capacity slices", False, "no raise")
    except ValueError:
        check("accessor rejects over-capacity slices", True)
    gv, gi, gl = sc.get_two_level_fold_buffers(row_count=8, total_slices=min(4, S))
    check("accessor slices to the requested size",
          tuple(gv.shape) == (8 * min(4, S), CAPS["topk"]) and tuple(gl.shape) == (8,),
          f"{tuple(gv.shape)} {tuple(gl.shape)}")

    print(f"\n=== result: {'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED'} ===")
    for f in FAILED:
        print(f"    FAILED: {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
