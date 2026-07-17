# Stage-1 design gate verdict: APPROVED with 3 riders

**Reviewer:** Fable, 2026-07-17
**Subject:** `sol-packed-ckv-design-note.md`

## Requested decisions

1. **Specialized 3-slot copy-only `PCIeDmaAllGather` — APPROVED.**
   The 6-slot allreduce object carries reduce scratch and fp8 stage that a
   byte-preserving gather cannot use; halving the IPC slab (134.8 vs 193.5
   MiB at the acceptance profile) is exactly the right trade. Rider A
   applies: this is new collective code, so the Stage-2 ring-schedule test
   is load-bearing — keep the repeated-schedule case (slot-reuse handshake)
   and add one case where a later call's payload is smaller than the prior
   call's (capacity reuse across differing lengths), since that is the
   production pattern tail chunks will produce.

2. **Hard-disarm of fp8-query staging + RS ring in `ckv` mode — APPROVED.**
   Your reading of the tension is correct and your resolution is the one I
   intended: no-coexistence has precedence; large mixed batches ride plain
   BF16/NCCL query math in a `ckv` process. The lost wire optimizations on
   ineligible chunks are an accepted v1 cost — they only matter for mixed
   batches, which the acceptance workload doesn't produce.

3. **Request-major duplicate packing + startup rejection of 480k/8-seq —
   APPROVED, with the explicit consequence recorded:** v1 is a
   MECHANISM PROOF at the 64k profile, not a ship candidate. Any ship claim
   requires the phase-2 deduplicated layout (your §9.2 bound: 213.8 MiB
   fits the query region; ~92-block pool cost). Nobody should read a green
   v1 acceptance as "ready for prod." Your §9.2 honesty about the
   accounting-vs-allocation distinction is noted and appreciated — the boot
   remains ground truth.

## Riders (carry into Stage 2/3)

- **A. Stream-ordering assertion.** §4.1/§5 rely on "overwritten by
  projection staging on the same ordered CUDA stream" — the gathered-CKV
  region dies only if the attention kernel and the projection staging copy
  are strictly stream-ordered. If any b12x extend path enqueues work on a
  side stream, that assumption breaks silently. The integration patch must
  either (a) assert the extend launch and projection copy share the current
  stream, or (b) record an event after attention and wait it before the
  staging copy. Cheap either way; silent corruption if wrong.

- **B. Gathered-view/kernel contract asserts.** The top stage-3 risk is the
  16-head extend plan consuming the TEMPORARY gathered cache
  (`[4B, 64, 368]` view + remapped `page_table_1`) — the kernel's
  page-geometry expectations (page stride 64*368, alignment, index bounds)
  must be validated by explicit startup asserts against the plan, not
  assumed. CPU tests cannot cover the CUDA kernel's addressing; the asserts
  are the only guard between a green CPU gate and a garbage-output boot.

- **C. Head-major staging copy is an accepted v1 cost.** The extra 48 MiB
  copy (attention output → head-major projection input) is fine for the
  mechanism proof; log its time under `ckv_pack` or a dedicated tag so the
  acceptance profile shows what a direct head-major kernel output would buy
  in phase 2. Do not optimize it in v1.

## Checks performed

- Remap arithmetic verified against interleave-1 striping (owner = g mod 4;
  one virtual block = exactly one 64-record local page per rank; slot
  formula indexes the [4, B, 64, 368] layout correctly).
- Memory bill spot-checked (179.688 MiB gathered region, 134.797 MiB slab,
  arithmetic-constancy identity 64×512 = 16×2048).
- Wire model at late-55k (4.83 MiB local, 3× forward = 14.5 MiB/rank)
  consistent with the ≤7 ms/layer acceptance line at measured fabric rates.
- Rank-invariance argument (§10) accepted: routing from identical metadata,
  validity affects payload only, init via group vote with fatal-error
  (no-fallback) semantics — matches the collective-safety contract proven
  live tonight (fit boot 2 executed the same vote pattern cleanly under a
  real rank-0 OOM).

## Proceed

Stage 2 is a GO under the approved decisions + riders. Deliver the three
CPU harnesses with captured output per the brief; Stage-3 integration only
after that gate.
