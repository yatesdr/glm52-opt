#!/usr/bin/env python3
"""Proof for the budgeted two-level fold reservation (b12x paged indexer).

The design claim has two halves. Both are tested here, on CPU, no GPU:

  A. CORRECTNESS — the number of slices is a free parameter. Every element of the
     true global top-k also ranks within its own slice's top-k, so the union of
     per-slice top-k lists contains the global top-k for ANY partition. This is
     what licenses coarsening the slices to fit a memory budget.

  B. BUDGET — api.plan_two_level_fold bounds q_rows * total_slices to
     TWO_LEVEL_FOLD_BUDGET_ROWS, so a single fixed reservation
     (BUDGET_ROWS * topk) covers every shape, and it never plans MORE slices than
     the original code would have.

Plus the mechanical layout checks: no pre-existing scratch offset moves, the new
regions do not overlap, views are correct, and the accessor enforces the budget.

    python3 v19_two_level_fold_budget_proof.py --patched-dir /tmp/patched
Exit 0 = all passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys

import torch

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def load(path: str, name: str):
    """Load a patched module under its REAL package name.

    api.py uses relative imports, so it only resolves when the module name keeps
    the b12x.attention.indexer prefix. Registering it in sys.modules also makes
    the patched scratch.py's absolute `from ...api import ...` bind to it.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PAGE = 64
TOPK = 2048           # index_topk for GLM-5.2
MAXQ = 3072           # max_num_batched_tokens


def original_plan(width_tokens, page_size, supertile_pages, page_table_width, num_chunks):
    """The shipped planner, for comparison (no budget bound)."""
    SLICE, MAXSL = 16384, 32
    slice_tokens = SLICE
    min_slice = -(-width_tokens // MAXSL)
    if min_slice > slice_tokens:
        slice_tokens = -(-min_slice // page_size) * page_size
    base = 0
    for c in range(num_chunks):
        c_pages = min((c + 1) * supertile_pages, page_table_width) - c * supertile_pages
        if c_pages <= 0:
            break
        base += max(1, -(-(c_pages * page_size) // slice_tokens))
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patched-dir", required=True)
    args = ap.parse_args()

    # bind the ORIGINAL scratch first, against the shipped api
    import b12x.attention.indexer.scratch as orig
    _ = orig._indexer_paged_scratch_layout          # touch it so it is fully bound

    # then swap in the patched api so the patched scratch picks it up
    api = load(f"{args.patched_dir}/api.py", "b12x.attention.indexer.api")
    patched = load(f"{args.patched_dir}/scratch.py",
                   "b12x.attention.indexer.scratch_patched")

    TARGET = api.TWO_LEVEL_FOLD_TARGET_ROWS

    # ---------------------------------------------------------------- A -----
    print("\n=== A. correctness: slice count is a free parameter ===")
    torch.manual_seed(0)
    random.seed(0)
    ok_all, worst = True, None
    for trial in range(200):
        n_cand = random.randint(200, 4000)
        k = random.randint(1, min(64, n_cand))
        logits = torch.randn(n_cand, dtype=torch.float32)
        gold = set(torch.topk(logits, k).indices.tolist())

        # arbitrary partition into S slices, including degenerate ones
        S = random.choice([1, 2, 3, 5, 8, 13, 32, 64, n_cand])
        S = min(S, n_cand)
        cuts = sorted(random.sample(range(1, n_cand), S - 1)) if S > 1 else []
        bounds = [0] + cuts + [n_cand]

        union: set[int] = set()
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = logits[a:b]
            kk = min(k, seg.numel())
            union.update((torch.topk(seg, kk).indices + a).tolist())

        if not gold.issubset(union):
            ok_all = False
            worst = (trial, n_cand, k, S)
            break
    check("global top-k ⊆ union of per-slice top-k (200 random partitions)",
          ok_all, "" if ok_all else f"counterexample at {worst}")

    # the same claim at the shapes we actually plan
    ok_shapes = True
    for S in (1, 2, 4, 8, 15, 16, 30, 32):
        n_cand, k = 480_000 // 64, 64
        logits = torch.randn(n_cand)
        gold = set(torch.topk(logits, k).indices.tolist())
        step = -(-n_cand // S)
        union = set()
        for a in range(0, n_cand, step):
            seg = logits[a:a + step]
            union.update((torch.topk(seg, min(k, seg.numel())).indices + a).tolist())
        if not gold.issubset(union):
            ok_shapes = False
            break
    check("holds at the planned slice counts (1..32) on a 7500-page table", ok_shapes)

    # ---------------------------------------------------------------- B -----
    print("\n=== B. budget: q_rows * total_slices is bounded ===")
    shapes = []
    for width in (32_768, 65_536, 131_072, 262_144, 400_000, 480_000):
        for q in (1, 8, 64, 256, 768, 1024, 2048, 3072, 4096):
            shapes.append((width, q))

    worst_prod, over_reserved, coarser_than_orig, declined = 0, [], [], 0
    for width, q in shapes:
        ptw = -(-width // PAGE)
        supertile_pages = max(32768 // PAGE, 1)
        chunks = max(1, -(-ptw // supertile_pages))
        if not api.two_level_fold_engages(width_tokens=width):
            continue
        _, total = api.plan_two_level_fold(
            width_tokens=width, page_size=PAGE, q_rows=q,
            supertile_pages=supertile_pages, page_table_width=ptw, num_chunks=chunks)
        if total == 0:
            declined += 1
            continue                       # falls back to the reserved legacy path
        prod = q * total
        worst_prod = max(worst_prod, prod)
        reserved = api.two_level_fold_reserved_rows(
            max_q_rows=q, page_table_width=ptw, page_size=PAGE,
            supertile_pages=supertile_pages)
        if prod > reserved:
            over_reserved.append((width, q, total, prod, reserved))
        o = original_plan(width, PAGE, supertile_pages, ptw, chunks)
        if total > o:
            coarser_than_orig.append((width, q, total, o))

    check(f"every ACCEPTED plan fits the flat budget across {len(shapes)} shapes",
          not over_reserved, f"worst accepted product={worst_prod:,} budget={TARGET:,}"
          if not over_reserved else f"OVERFLOW: {over_reserved[:3]}")
    print(f"        declined -> legacy carry path in {declined}/{len(shapes)} shapes")
    check("planner never produces MORE slices than the shipped code",
          not coarser_than_orig,
          "" if not coarser_than_orig else f"regressions: {coarser_than_orig[:3]}")

    # unchanged where it already fit
    same = 0
    for width in (32_768, 65_536, 131_072, 262_144, 480_000):
        ptw = -(-width // PAGE)
        sp = max(32768 // PAGE, 1)
        ch = max(1, -(-ptw // sp))
        q = 768
        _, total = api.plan_two_level_fold(
            width_tokens=width, page_size=PAGE, q_rows=q,
            supertile_pages=sp, page_table_width=ptw, num_chunks=ch)
        if total == original_plan(width, PAGE, sp, ptw, ch):
            same += 1
    check("plan identical to the shipped code at the observed q_rows (768)",
          same == 5, f"{same}/5 widths unchanged")

    # the reservation formula must dominate the planner at the extremes too
    extreme_ok = True
    for q in (1, 3072, 4096, 8192):
        for width in (65_536, 262_144, 480_000):
            ptw = -(-width // PAGE); sp = max(32768 // PAGE, 1)
            ch = max(1, -(-ptw // sp))
            _, t = api.plan_two_level_fold(
                width_tokens=width, page_size=PAGE, q_rows=q,
                supertile_pages=sp, page_table_width=ptw, num_chunks=ch)
            if t and q * t > api.two_level_fold_reserved_rows(
                    max_q_rows=q, page_table_width=ptw, page_size=PAGE,
                    supertile_pages=sp):
                extreme_ok = False
    check("reservation dominates the planner at extreme q_rows", extreme_ok)

    # ---------------------------------------------------------------- C -----
    print("\n=== C. reservation is fixed and cheap ===")

    def caps(mod, **over):
        kw = dict(num_q_heads=16, max_q_rows=MAXQ, topk=TOPK,
                  max_page_table_width=-(-480_000 // PAGE),
                  mode="prefill", shared_page_table=True)
        kw.update(over)
        return mod.B12XIndexerPagedScratchCaps(device=torch.device("cpu"), **kw)

    lo = orig._indexer_paged_scratch_layout(caps(orig))
    lp = patched._indexer_paged_scratch_layout(caps(patched))
    grew = lp.nbytes - lo.nbytes
    rows = api.two_level_fold_reserved_rows(
        max_q_rows=MAXQ, page_table_width=-(-480_000 // PAGE), page_size=PAGE,
        supertile_pages=max(32768 // PAGE, 1))
    expect = rows * TOPK * 4 * 2 + MAXQ * 4
    check("reservation == flat budget rows * topk * (fp32+int32)",
          expect <= grew <= expect + 4096,
          f"{grew/2**20:.0f} MiB for {rows:,} rows "
          f"(expected ~{expect/2**20:.0f} MiB)")
    print(f"        naive (max_q_rows x S_max) would have been "
          f"{MAXQ*30*TOPK*8/2**20:.0f} MiB")

    print("\n=== D. layout safety ===")
    offs = [f for f in lo.__dataclass_fields__ if f.endswith("_offset_bytes")]
    moved = [f for f in offs if getattr(lo, f) != getattr(lp, f, None)]
    check(f"all {len(offs)} pre-existing offsets unchanged", not moved, str(moved[:3]))
    for nm, off in (("values", lp.two_level_fold_values_offset_bytes),
                    ("indices", lp.two_level_fold_indices_offset_bytes),
                    ("lengths", lp.two_level_fold_lengths_offset_bytes)):
        check(f"fold {nm} appended beyond the original block", off >= lo.nbytes)

    ls = patched._indexer_paged_scratch_layout(
        caps(patched, max_page_table_width=-(-16_384 // PAGE)))
    check("short context reserves nothing", int(ls.two_level_fold_rows) == 0)

    print("\n=== E. views and accessor ===")
    storage = torch.zeros(lp.nbytes, dtype=torch.uint8)
    sc = patched._materialize_indexer_paged_scratch(caps(patched), storage, lp)
    v, i, ln = sc.two_level_fold_values, sc.two_level_fold_indices, sc.two_level_fold_lengths
    check("values fp32 contiguous", v.dtype == torch.float32 and v.is_contiguous())
    check("indices int32 contiguous", i.dtype == torch.int32 and i.is_contiguous())
    check("lengths int32 contiguous", ln.dtype == torch.int32 and ln.is_contiguous())
    check("values shape == (reserved_rows, topk)", tuple(v.shape) == (rows, TOPK), str(tuple(v.shape)))

    def span(t): return (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
    sp = {"values": span(v), "indices": span(i), "lengths": span(ln)}
    names, overlap = list(sp), False
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            (s1, e1), (s2, e2) = sp[names[a]], sp[names[b]]
            if s1 < e2 and s2 < e1:
                overlap = True
    check("fold views do not overlap each other", not overlap)

    try:
        sc.get_two_level_fold_buffers(row_count=rows, total_slices=2)
        check("accessor rejects an over-budget request", False, "no raise")
    except ValueError:
        check("accessor rejects an over-budget request", True)
    gv, gi, gl = sc.get_two_level_fold_buffers(row_count=768, total_slices=30)
    check("accessor returns exactly the requested rows",
          tuple(gv.shape) == (768 * 30, TOPK) and tuple(gl.shape) == (768,),
          f"{tuple(gv.shape)} {tuple(gl.shape)}")
    check("returned views stay contiguous", gv.is_contiguous() and gi.is_contiguous())

    print(f"\n=== result: {'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED'} ===")
    for f in FAILED:
        print(f"    FAILED: {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
