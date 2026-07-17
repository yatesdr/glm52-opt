# Phase-2 Gate A verdict: APPROVED — Gate B is a GO

**Reviewer:** Fable, 2026-07-17
**Subject:** `sol-expanded-charter-phase2-design.md` (commit 7162c51)

## Decisions requested — all APPROVED

1. **No-slab NCCL gather into existing workspaces.** The correct answer to
   the transient-float finding: zero new permanent residents. Conditions
   verified: `all_gather_into_tensor` with caller-owned views is indeed the
   established pattern in the pinned stack, and the prohibition on
   `GroupCoordinator.all_gather` (allocating base path) is right. Startup
   must reject if neither PyNCCL nor process-group path is available —
   as specified.
2. **Active request-major route + physical-pool fallback.** The B_active =
   2,403 disjointness margin and the P+R bound analysis are correct. The
   fallback's owner-authoritative table remap resolves prefix aliasing
   without cross-rank assumptions. Rider: the pool route's per-call bytes
   (210 MiB gathered per eligible chunk) would erase the CKV advantage if
   it ever became the common path — the profiler route counters make this
   observable; ALSO log a once-per-process WARNING the first time the pool
   route activates, so a misconfigured workload is visible in ops logs,
   not only in profiler summaries.
3. **192 MiB direct-CUDA escrow + dual ≥150 MiB group-min probes.** This is
   better than what the charter required: it converts the headroom
   requirement from an accounting claim into a machine-checked reservation
   with fail-closed semantics, outside the PyTorch allocator where the
   fit-boot absorption pathology lives. The honest 76–262 MiB interval and
   the 240–265 prediction with a 150 hard bar are exactly the right
   epistemics. Approved as specified, including the no-retry rule.
4. **BLOCKS=2340 target, no further block-cut ladders.** Agreed — ladder
   results are baked into §2 correctly.
5. **DCP posture bands.** Approved as the P3 deliverable. DCP1 as a
   shipping candidate for ≤64k profiles is reasonable pending its own
   quality/decode acceptance record (not yet run — prefill only tonight).
   DCP2 cells stay measurement-required before entering the matrix.
6. **Revised 1,700–1,800 ceiling + the 1,879 gap as a named profiler
   target.** Approved.
7. **Compute-profiler design.** The nested exclusive accounting with the
   5%-reproduction validity bar and the one-AR-per-region classification
   check are the right discipline. Approved as a separate, measurement-only
   patch reviewed independently of the transport work.

## Byte confirmations (requested for Gate B/C)

| File | Deployed base md5 | Status |
|---|---|---|
| `vllm/v1/worker/gpu_worker.py` (v14eq overlay, mounted) | `0829a65484d4dd14c385366291e7a25c` | MATCHES your mirror — confirmed |
| `vllm/model_executor/models/deepseek_v2.py` (base image, not overlaid) | `e8f115f3349c12d4ff9f7253cd7c9bec` | PINNED — patch against this |
| `vllm/model_executor/layers/mla.py` (base image) | `afd7453cfe9e8478f6f09e6e47697b75` | MATCHES your stated md5 — confirmed |

## Surface approvals

- **Phase-2 transport surface (4 files):** APPROVED — Stage-3 trio +
  `gpu_worker.py` at the confirmed md5.
- **Compute-profiler surface (7 files incl. new
  `compute_phase_profiler.py`, `mla.py`, `deepseek_v2.py`):** APPROVED at
  the pinned md5s above. Keep it measurement-only; any behavioral change
  hiding in a profiler patch is an automatic gate failure.

## Riders

- **A. Escrow-held window observability:** log driver-free at escrow arm
  (not just at the probes) so a failure while held is diagnosable from one
  line.
- **B. Pool-route first-activation WARNING** (see decision 2).
- **C. DCP1 shipping candidacy requires its own acceptance record** —
  quality gates + decode C1 at the DCP1 profile; prefill physics alone
  doesn't ship a config.

Gate B (CPU/source harnesses, 5 items as listed) is a GO.
