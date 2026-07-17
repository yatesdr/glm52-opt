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
transport from 18.6 ms to 1.4 ms. Full-context (480k) support is phase 2,
designed and CPU-proven in `design/`, server validation pending.

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

- Phase-2 full-context (480k) server gates: NCCL-swap parity, the
  escrow/probe memory boot, 480k acceptance. Designs and CPU proofs are
  complete; results will land in RESULTS.md when run.
- Compute-remainder profiler (7-file measurement patch): designed, gated,
  not yet integrated. It decomposes the residual ~21 ms/layer and picks
  the next kernel target.
- DCP2 posture cells: unmeasured.
- Trellis-quant checkpoint evaluation (larger KV pools): planned.

## Process note

Every stage here went: design note → adversarial review gate → CPU proof
gate → code gate (lint + byte pins + in-image import) → server boots run
by a separate operator with fail-closed acceptance bands. Two field bugs
still got through to boots (a Triton constexpr issue and a collective
init race designed before the vote pattern existed) — both documented in
`design/` with the check that now catches them. If you extend this work,
keep the discipline; the gates are cheaper than the boot cycles they
save.
