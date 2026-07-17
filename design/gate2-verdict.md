# Stage-2 correctness gate verdict: APPROVED — Stage 3 is a GO

**Reviewer:** Fable, 2026-07-17
**Subject:** commit a9ace6c, `sol-packed-ckv-stage2/` (3 harnesses + captured output)

## Review performed

- `test_ownership_inversion.py` read in full. The proof is honest and
  complete: emulated fp32 arithmetic (struct round-trip per op), baseline =
  per-owner partial attention + max-shifted LSE merge, proposed = local-head
  attention over the gathered set, compared per (query_owner, local_head)
  with identical queries asserted. Coverage asserts are self-enforcing:
  uneven owner counts, uneven striped tails, noncontiguous physical blocks,
  ≥4 invalid IDs. Max errors (3e-08 output / 2.4e-07 LSE) are at fp32
  epsilon scale — the mechanism is mathematically equivalence-preserving,
  as the design claimed (reduction-order differences only).
- `test_remap_reads.py` spot-checked: implements the EXACT design-note slot
  formula ((owner*B + packed_block)*64 + local_offset, 368-byte records)
  with owner-authoritative validity — this pins the byte layout Stage 3
  must implement.
- `test_ring_schedule.py` spot-checked: 3 receive slots, forward-from-
  previous-slot schedule, fixed-capacity slot reuse with stale-tail
  exclusion (12 excluded in the smaller-after-larger case) — rider A
  satisfied as specified.

## Carried into Stage 3 (unchanged from gate 1)

- Rider B: startup asserts validating the temporary gathered-cache kernel
  geometry (page stride, alignment, index bounds) against the 16-head plan.
- Rider C: the 48 MiB head-major staging copy timed under its own tag.
- Stream-ordering assertion (rider A's integration half): attention
  completion vs projection-staging overwrite.

## Stage-3 integration notes from tonight's boots (new since gate 1)

- The full-context frontier moved: v1.4's allocator returns freed KV blocks
  to the device (proven fit boot 3: ring vote passed all 4 ranks at
  BLOCKS=2340). Your §9.2 phase-2 accounting is therefore MORE credible
  than the v1.3 ledger suggested — but the boot remains ground truth.
- Deploy base reminder: your patches land against the pinned md5 set in the
  brief; the deployed common.py is the collective-safe rev (255bde14).
  Deliver a unified diff + md5 manifest; I byte-verify before any boot.
