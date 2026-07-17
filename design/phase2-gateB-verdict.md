# Phase-2 Gate B verdict: APPROVED — Gate C is a GO (one condition)

**Reviewer:** Fable, 2026-07-17
**Subject:** commit 7f6b651, `sol-packed-ckv-phase2-gateB/` (5 harnesses)

All five proofs reviewed against captured output; escrow harness read in
structural detail (fake CUDA runtime with per-process state machine, exact
malloc/free accounting, leak assertion via live-pointer set, fatal-path
coverage including vote failure, low probe, and bad ordering). Highlights:

- Gate-A riders A and B are VERIFIED IN HARNESS: the arm-time log is
  asserted present, and the pool route's once-per-process warning is
  asserted per rank (1,1,1,1).
- The profiler validity gate works in BOTH directions — reproduction
  within 5% passes AND deliberate perturbation is detected; the
  exactly-one-TP-per-region classifier detects a faulty extra call.
- Route determinism at the 2403/2404 boundary across 1,008 cases with
  identical byte lengths on all ranks is exactly the collective-safety
  proof the deadlock class demands.

## Condition on Gate C

Your Gate-C integration bundle MUST be built on the FIELD-FIXED Stage-3
`b12x_mla_sparse.py` (md5 `20a2cf60ce2e99d8c90249d458f330f8`), not your
committed `5092bf94` — see `sol-packed-ckv-fix1.md`: the remap Triton
kernel referenced module globals, which Triton JIT cannot see; fixed with
three constexpr params at both launch sites. Fold that diff into your
tree first (or re-derive it identically) and adopt the fix note's
recommendation: add a Triton dry-compile (or constexpr-only grep) for
every @triton.jit kernel to your Gate-C static checks — your phase-2
patch adds more kernels and this class is invisible to every other check
we run.

## Standing

Stage-3 CKV acceptance (the 64k mechanism verdict) is in flight on cn3
with the fixed kernel. Gate-C review happens against its outcome: a
CONFIRM makes your phase-2 server gates 1–3 the critical path to a
full-context ship.
