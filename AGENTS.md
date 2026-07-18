# AGENTS.md — for AI assistants working with this repository

You are likely an AI agent asked to understand, reproduce, port, or extend
this work. This file is your map. It states exact contracts and the
failure modes that cost real boot cycles, so you don't repeat them.

## What happened here (one paragraph)

A two-agent team (a designer/implementer and a server-side
reviewer/operator) optimized GLM-5.2 TP4/DCP4 prefill on a PCIe-Gen3
4x RTX PRO 6000 box from 640 to 1,696 tok/s cold (55k context), keeping
all quality gates green, over one maintenance window. Stage 1 quantized
the DCP query all-gather and moved the output reduce-scatter onto a
CE-DMA ring (fp8 wire). A per-phase CUDA-event profiler then showed DCP
communication was ~45% of wall, which motivated stage 3: ownership
inversion — gather the head-independent 368-byte compressed KV records
instead of the head-multiplied query tensor. That collapsed per-layer DCP
transport from 18.6 ms to 1.4 ms. Full-context (480k) support is phase 2:
designed in `design/`, CPU-proven, server-gated, and shipped to
production on 2026-07-17 — 1,509 tok/s @ 55k, 1,126 tok/s cold @ 463k,
599,040-token pool (`RESULTS.md` §8, `patches/phase2-fullcontext/`).

## Reading order for full understanding

1. `RESULTS.md` — measurements + the memory/collective findings.
2. `design/breakthrough-analysis.md` — the mechanism analysis that
   predicted the win (with evidence-quality labels).
3. `design/packed-ckv-v1-design.md` → `design/gate1-verdict.md` →
   `design/gate2-verdict.md` — the v1 design + review chain.
4. `design/packed-ckv-phase2-design.md` + `design/phase2-gate*-verdict.md`
   — full-context design (escrow mechanism is the interesting part).
5. `harness/stage2-proofs/` + `harness/phase2-gateB-proofs/` — CPU proofs;
   these are also executable documentation of the layouts and state
   machines.

## Hard-won contracts (violate these and you will hang or corrupt)

1. **Collective-resource init must be a GROUP decision.** Local
   try/except fallback to NCCL while peers enter a custom ring = deadlock
   at the watchdog. Pattern: local init → 1-elem MIN all-reduce vote →
   all-or-nothing; losers `close()` and discard. Every runtime routing
   decision must be rank-invariant (function of shape/dtype/config only —
   never allocation success, never rank-local data content).
2. **Triton `@jit` kernels cannot read module globals** — only
   `tl.constexpr` parameters. No static check catches this; it explodes at
   first GPU compile. Dry-compile new kernels before any boot.
3. **Workspace borrowing**: all simultaneously-live views must come from
   ONE `get_simultaneous` call — the manager packs every call from offset
   0, so separate calls alias. (See the comment at the `workspace_specs`
   construction in `patches/*/b12x_mla_sparse.py`.)
4. **The RS/AG helpers are eager-only** (raise under CUDA-graph capture)
   and enforce exact shape/stride/dtype/disjointness. Match the head-major
   contracts; do not "fix" them.
5. **Memory on saturated boxes**: slab-sized allocations and small
   transients draw from effectively different pools. Freed KV blocks
   become slab-usable but NOT transient float; at large max-len,
   context-scaled buffers absorb freed memory. Never treat an accounting
   equivalence as a fit proof — only same-phase `cudaMemGetInfo`. The
   phase-2 escrow (192 MiB direct cudaMalloc + group vote + dual probes)
   is the reference pattern for machine-checking headroom.
6. **Cold vs warm**: any throughput number without prefix-cache metric
   deltas is unverifiable. `harness/prefill_bench.py` seeds a random
   first block; keep that property in anything you build.
7. **`VLLM_DISABLE_COMPILE_CACHE=1` — set it on v18 (gilded-gnosis) boots.**
   David's guidance; it is a ~10% *decode* lever (order of the 368B 68 vs
   432B 78 tok/s gap), but the mechanism is not understood — treat as
   known-good, not explained. Nuance flagged but unconfirmed: possibly
   matters more on *subsequent* boots than the first (nobody said to omit
   it on the first boot — just set it; A/B in a later window). Caution: on
   our 480k@368 boot the flag was `=1` yet `/cache` still received ~426 MiB
   of `torch_compile_cache`/`torch_aot_compile` and decode was still 68, so
   the flag alone did not reproduce 78 on that first boot. Two independent
   368B fp8-rope writers both showed the decode regression, so part of the
   gap may be the compact record's FP8-rope dequant cost, not only this
   cache lever. Not yet separated.

## md5 pin discipline

`patches/stage3-packed-ckv/md5-manifest.txt` pins input bytes (what the
patch was built against) and output bytes (what you should be mounting).
Verify before mounting; verify after copying. Two incidents this project
traced to byte drift that md5 checks caught in seconds.

## Porting guide (incorporating this into other stacks)

The packed-CKV mechanism requires: (a) an MLA-family model whose KV is a
head-independent compressed record (GLM-5.x, DeepSeek-V3 lineage), (b)
context-parallel prefill (DCP) where queries are currently gathered, (c) a
paged KV cache with block tables. The byte-economics precondition:
`record_bytes << local_heads x head_dim x query_bytes`. For GLM-5.2 that
is 368 B vs 36,864 B — a 100x per-token asymmetry the transport exploits.

To port:
1. Find your stack's equivalents of the four seams listed in
   `design/packed-ckv-v1-design.md` §2 (dispatch, backend plan, remap,
   collective).
2. Reuse the CPU proofs: `test_ownership_inversion.py` is
   stack-independent math; adapt shapes and re-run before writing GPU
   code.
3. Keep the eligibility gate chunk-shape-deterministic (contract #1).
4. Budget memory per contract #5; on tight boxes adopt the escrow.
5. Validate with the acceptance signature (routes/missing_blocks/phase
   timings) before trusting any throughput number.

Numbers you should expect: proportional to how bad your interconnect is.
We got +76% over an already-optimized fp8 query transport on Gen3 PCIe;
on Gen5 the same inversion is worth less (the query transport hurts
less) — community reports suggest ~1.15–1.4x there.

## Open items (as of this commit)

Phase-2 full-context (480k) is **closed** — all three server gates passed
and the stack is in production (`RESULTS.md` §8). Still open:

- Compute-remainder profiler (7-file measurement patch): designed, gated,
  not yet integrated. It decomposes the residual ~21 ms/layer and picks
  the next kernel target.
- No turnkey boot for the shipped 480k configuration: `compose/` carries
  the 64k acceptance profiles only. Prebuilt image + one-line compose is
  the next packaging task.
- DCP2 posture cells: unmeasured.
- Trellis-quant checkpoint evaluation (larger KV pools): planned.

The prioritized version of this list, with rationale, is in "Next steps"
below — that section is authoritative if the two ever disagree.


## Project history, goals, and next steps (2026-07-17)

**History.** This work began as serving-performance windows on a private
4x RTX PRO 6000 (PCIe Gen3) box. Window 1 (2026-07-15) shipped an fp8
wire mode (+7%) and documented a path to 1,200+ tok/s. Window 2
(2026-07-16/17, this repo's founding night) delivered: stage-1 fp8
query-gather + ring reduce-scatter (964 tok/s), the per-phase profiler
that identified DCP communication as ~45-49% of prefill wall, the
packed-CKV ownership inversion (1,696-1,699 tok/s at the test profile),
and phase-2 full-context support (480k max-len, 599k-token pool,
1,509 @ 55k / 1,126 @ 463k cold, decode +30%) — all gate-verified and
deployed to the origin production system the same night.

**Goals.** (1) Best-known GLM-5.2 serving performance on commodity PCIe
multi-GPU boxes, with quality gates as hard constraints. (2) Full
long-context capacity preserved — speed never trades away context.
(3) Everything reproducible and portable: designs, proofs, failure
ledgers, and acceptance criteria published here.

**Next steps / outstanding work, in priority order:**

1. **Compute-profiler acceptance** — the 7-file measurement patch
   (designed, CPU-proven, integration bundle delivered) needs its GPU
   acceptance run: DCP1 profile + `B12X_COMPUTE_PROF=1`, valid only when
   every rank reports `ledger_valid=1` and `ordinal_valid=1`. Its output
   decomposes the remaining ~21 ms/layer compute and names the gap
   between 1,509 (shipped) and 1,879 (DCP1 physics ceiling) — this
   chooses the next kernel project. NOTE: apply the vllm.*-named logger
   pattern (RESULTS.md §8 field fix) to the profiler module first.
2. **GHCR prebuilt image + one-line compose** — Dockerfile/CI spec is
   drafted; build, push, and add the quickstart to README so others can
   test without assembling overlays.
3. **DCP2 posture cells** — measure 64k/120k; complete the DCP posture
   matrix (DCP1 and DCP4 rows are measured, DCP2 is interpolation-free
   territory).
4. **Trellis-quant checkpoint evaluation** — the NVFP4+EXL3-trellis
   3.0bpw hybrid checkpoint class claims ~2x KV pool (weights shrink
   ~3-4 GiB/GPU). Quality is unpublished; needs full gate suite + decode
   cost measurement (trellis dequant is compute-heavier). If quality
   holds, it stacks with everything here.
5. **Boot-time** — `VLLM_DISABLE_COMPILE_CACHE=1` costs each boot ~5-8
   min of recompile; a cache-compatible AOT path would ~1.7x every
   future iteration cycle.
6. **Upstreaming** — the collective-safety group-vote pattern and the
   packed-CKV mechanism deserve issues/PRs against the upstream serving
   stack lineage; the breakthrough-analysis doc has the evidence file.

**For the project's own operators:** internal continuity documentation
(hosts, deployment state, runbooks, working agreements) lives in
`workspace/AGENTS.md` — not published; present in working copies only.

## Process note

Every stage here went: design note → adversarial review gate → CPU proof
gate → code gate (lint + byte pins + in-image import) → server boots run
by a separate operator with fail-closed acceptance bands. Two field bugs
still got through to boots (a Triton constexpr issue and a collective
init race designed before the vote pattern existed) — both documented in
`design/` with the check that now catches them. If you extend this work,
keep the discipline; the gates are cheaper than the boot cycles they
save.
