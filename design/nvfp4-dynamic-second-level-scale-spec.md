# Spec: dynamic per-token second-level scaling for the NVFP4 MLA KV record

Status: **PARKED** — do not begin until (a) the static-calibration fix has
full promotion validation (redirect-001 P0–P4) and (b) the Phase A audit
questions below are answered. This document defines the problem, the target
design, the migration path, and the validation contract so the project can
start cleanly when unparked.

Author: Fable (at Derek's direction), 2026-07-27
Owners when unparked: Sol (implementation), Fable (review)
Prior art: vLLM PR #145 (static calibrated outer scales — the interim fix),
SparkInfer #85 (record/kernel-contract change precedent), NVIDIA NVFP4
two-level scaling definition.

---

## 1. Problem statement

### 1.1 The defect class

The `nvfp4_ds_mla` KV record stores the 512-dim post-RMSNorm `kv_c` latent
as NVFP4: 4-bit values in 16-element blocks, each block carrying an E4M3
scale factor. E4M3 block scales only have usable precision inside their
normal range. The writer applies a **static per-layer outer (second-level)
scale, hardcoded to 1.0**, that positions where a layer's block scales land
in E4M3 space.

GLM-5.2's `kv_c` magnitudes span ~240× across layers (max_abs 0.02 → 5.2).
At outer scale 1.0, shallow layers' block scales fall into E4M3 subnormals
and quantize with severely reduced precision. The error is amplified through
the network and, at deep context (343k tokens), degrades hidden states
enough that the sparse-indexer score field no longer ranks needle tokens
into the top-2,048 until layers 62–74 — too late to steer the answer. This
was proven causal for the v20 deep-retrieval regression on 2026-07-27
(frozen r1 recovered exactly with correctly positioned scales; stock scorer;
see `fable-review-log.md` engagements 7–12).

### 1.2 Why the static-calibration fix is interim, not final

PR #145 fixes positioning with a per-layer calibrated outer scale:
`s_l = max_abs(kv_c at layer l) / (6 · 448)`, values shipped as a JSON
artifact. It works (our frozen gate proves it), but the artifact is
architecturally homeless:

- the values are **checkpoint-specific** (activation statistics), so they
  don't belong in the engine repo;
- they are **cache-format-specific** (an NVFP4/E4M3 positioning transform),
  so quant producers won't ship them with checkpoints;
- they are **capture-conditions-specific** (calibration context length,
  prompt mix, checkpoint build) and go silently stale on any weight update;
- every new model, quant, or long-context regime needs recapture, tooling,
  provenance, and a place to live.

### 1.3 The correct fix

NVFP4, as specified, is a **two-level** scaling format: per-block E4M3
micro-scales plus a higher-precision second-level scale at an
implementation-chosen granularity. The current implementation chose
"per layer, static, 1.0". The correct granularity for a KV cache — where
tokens arrive one at a time and each write already touches every element —
is **per token-record, computed from the data at write time**:

```
amax_t  = max(|kv_c_normed[t, :]|)            # 512-element reduction
s_t     = amax_t / (6 · 448)                   # (NVFP4 max) · (E4M3 max)
blocks  = quantize_nvfp4(kv_c_normed[t] , s_t) # block scales relative to s_t
record  = { blocks, block_scales, s_t }
```

Every record then carries its own range. Properties:

- **No calibration artifact, for any model, ever.** No sidecar, toolkit,
  first-load pass, staleness, or ownership question.
- **Subnormal/clip positioning failures are impossible by construction**:
  the largest block scale in every record lands at the top of E4M3 range.
- Deterministic, write-time exact, and per-token adaptive — robust to
  activation-range drift across layers, contexts, checkpoints, and future
  models by definition.

## 2. Design

### 2.1 Scale storage — two candidate layouts (Phase A decides)

**Option S (side table, preferred a priori):** keep the 368-byte record
byte-identical; store `s_t` in a parallel per-layer array indexed by the
same physical slot id as the record (`float32[num_slots]`, or `bf16` if
error analysis permits — see §6 Q3).

- +4 B/token/layer ⇒ ~+1.1% KV memory at fp32; no record-layout or
  alignment churn; existing record readers/writers change only by one
  scalar load/store; paging, DMA, and copy paths must carry the side table
  alongside its record pages.

**Option R (inline in record):** append `s_t` to the record.

- 368 → 372 B breaks 16-byte alignment; padding to 384 costs +4.3% KV
  memory. Only preferable if Phase A finds unused header/pad space in the
  current layout, or if side-table transfers prove awkward in the DMA/wire
  paths.

### 2.2 Write path

- Per-token `amax` reduction over 512 values fused into the existing
  quantize-and-pack writer kernel (the data is already in registers/SMEM).
- `amax == 0` ⇒ `s_t = 1.0`, all-zero blocks (defined, not special-cased
  downstream).
- The static per-layer outer scale multiplies out of the pipeline entirely
  (equivalently: fixed to 1.0 and superseded by `s_t`).

### 2.3 Read path

Every consumer of the CKV field multiplies its dequantized output by the
record's `s_t` (one scalar broadcast per token-record). Phase A must
inventory ALL consumers, at minimum:

- main sparse MLA attention (prefill + decode kernels, all MTP/next_n
  variants, graph and eager);
- any CKV gather / prefetch / transfer path that dequantizes (vs moving
  raw bytes — raw-byte movers must move the side table instead);
- DCP wire paths (`i8_ring` etc.): scales travel with their records,
  bit-exactly (they are FP32 metadata, not quantized payload);
- prefix-cache reuse, block copy-on-write, preemption swap in/out;
- offline tools/replays that parse records (the trace/replay harnesses).

### 2.4 The RoPE field is explicitly out of scope

The compact FP8 RoPE field keeps its existing scaling. Evidence to date
(#145 KLD deltas; the fp8-rope vs bf16-rope cells) says it is a small term.
If the randomized ladder ever shows margin thinness attributable to it,
that becomes a separate spec — the same two-level pattern would apply.

### 2.5 Configuration and coexistence

- `VLLM_NVFP4_MLA_DYNAMIC_SCALE=1` selects the new record semantics.
  Server-static, participates in cache/compile identity (see §2.6).
- The #145 static-scales path remains available during transition and as an
  A/B control; dynamic mode ignores the scales file with a loud log line.
- Record/format version tag in the cache spec so a mixed configuration
  fails closed at boot, never silently misreads.

### 2.6 Kernel and cache identity

Per the #85 lesson: bump the kernel policy identity for every kernel whose
argument contract or record interpretation changes, so no stale cubin can
run against the new layout. The dynamic-scale flag joins the compile-cache
key. Fresh-cache boots for all validation cells.

## 3. Performance and capacity budget (gates, pre-committed)

| Dimension | Budget |
|---|---|
| KV capacity | ≥ 500,000 tokens at max-len 480,000 (promotion floor) after the +1.1% (S) or +4.3% (R) overhead |
| Decode throughput | within 1% of static-scales baseline (same posture, serialized gate) |
| Prefill throughput | within 2% of static-scales baseline (amax reduction is fused; the budget covers the extra scalar traffic) |
| Wire overhead | side-table bytes accounted in DCP transfer benchmarks; within noise of baseline |

If any budget fails, the spec returns to design (e.g., bf16 scale storage,
coarser-than-token granularity — see §6 Q4) rather than shipping a
regression.

## 4. Validation contract (reuses the already-built instruments)

1. **Unit/bit level:** writer/reader round-trip vs a reference two-level
   quantizer on frozen activations — production error must equal the
   ideal quantizer's error (the "floor test"); randomized shapes; amax=0;
   single-value blocks; adversarial outlier blocks.
2. **Positioning invariant:** saturation counters must read zero subnormal
   block scales and zero clips across the frozen 343k row — by
   construction, so any nonzero count is a bug, not a tuning question.
3. **Equivalence vs interim fix:** layer-entry margin trace on frozen r1
   must match or beat the static-calibrated baseline (entry layer and
   presence counts); per-layer rank replay (layer-34 harness) within noise.
4. **End to end:** frozen four-row suite; randomized 50k–475k ladder with
   margin scoring; KLD vs (a) static-calibrated NVFP4 and (b) fp8_ds_mla
   reference; graph+eager consistency; fresh/warm/restart repeatability.
5. **Transport:** DCP wire cells proving scale side-table integrity
   (bit-exact scales across ranks) under i8_ring and raw BF16.

## 5. Migration and upstream path

- Phase A (audit, ~days): record layout ground truth (exact 368B field map,
  pad space); full consumer inventory (§2.3); side-table feasibility in
  paging/DMA/prefix-cache code; perf model for the amax fusion. Output: an
  addendum to this spec choosing Option S or R and freezing the kernel list.
- Phase B (prototype): SparkInfer writer/reader kernels + CPU/GPU unit
  proofs (validation items 1–2), eager-path integration behind the flag.
- Phase C (integration): full consumer coverage, graph mode, DCP, gates
  (items 3–5), perf/capacity budgets.
- Phase D (upstream): one SparkInfer PR (kernels/record) + one vLLM PR
  (cache spec, flag, wiring), presented as the format-correct successor to
  #145: "#145 fixed the positioning with calibration; this removes the need
  for calibration." #145's static path stays as the fallback for one
  release cycle, then the derived-default (weights-gamma) becomes the
  static fallback and the JSON knob deprecates.
- Interim state until unparked: v20 ships the validated static calibration
  (redirect-001); the P1 capture harness and saturation counters become
  permanent test fixtures for this spec's validation contract.

## 6. Open questions (answer in Phase A — these gate the design)

1. Exact current 368B layout: is there existing pad/header space that makes
   Option R free? (Would moot the side table.)
2. Which read kernels dequantize CKV inline vs pass raw records — how many
   kernel signatures actually change?
3. Scale dtype: FP32 vs BF16 side table — measure the round-trip error
   delta on frozen activations; BF16 halves the overhead if the error is
   at the floor.
4. Granularity fallback: if per-token fusion costs more than budgeted in
   any kernel, is per-page (per 64-token block) amax an acceptable
   compromise? (Analysis: worst-case within-page range spread on real
   activations — measurable offline from the P1 captures.)
5. Prefix-cache key: do cached blocks' side-table entries need to join the
   block hash, or are they derivable-invariant (same tokens ⇒ same scales,
   so no)? Verify the "no" formally.
6. Does the MXFP8 online path share the writer, and should this spec cover
   it in the same change or as a follow-up? (Same two-level pattern.)
7. Upstream-of-upstream: does mainline vLLM's NVFP4 KV work (if/when it
   lands) define a second-level scale slot we should be layout-compatible
   with?

## 7. Success definition

The next NVFP4 model — or the next quant of this one — boots on this stack
with zero calibration artifacts, zero scale-related configuration, passes
the saturation invariant by construction, and matches or beats the
calibrated-static quality gates. The words "scales file" stop appearing in
deploy documentation.
