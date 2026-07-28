# Adversarial response 004: review of `fable-adversarial-review-003.md`

Date: 2026-07-27
Reviewer: Fable
Review target: shared-precision causal claim and the bisection plan
Format: answers to §9, then focus guardrails

## 0. Headline finding of this review: the probable fix may already be in the image

vLLM PR #145 ("glm52: calibrated NVFP4 MLA KV outer scales"), which is
**included in the RC image** for testing and **default-off** behind
`VLLM_NVFP4_MLA_SCALES_FILE`, documents a known defect in the exact component
now under suspicion:

> GLM-5.2's post-RMSNorm 512-dim kv_c latent spans ~240× across layers
> (max_abs 0.02→5.2). At the writer's default outer scale of 1.0, shallow
> layers quantize with E4M3-subnormal block scales; **the error is amplified
> downstream.**

That is a scale-provenance defect in the NVFP4 write path — precisely the
class §7 of the review proposes to hunt with new tooling — already found,
fixed, measured, and shipped dormant in the running image. Its own KLD table
reorders the bisection priors:

| KV config | KLD mean |
|---|---|
| fp8_ds_mla (the recovered posture's cache) | 0.1263 |
| nvfp4 **+ scales**, bf16 rope | 0.1345 |
| nvfp4 **+ scales**, fp8 rope | 0.1356 |
| nvfp4 no scales, bf16 rope | 0.158 |
| nvfp4 no scales, fp8 rope (≈ the failing posture) | 0.168 |

Two implications:

1. **Uncalibrated NVFP4 CKV is the dominant measured quality term; compact
   FP8 RoPE is a small one** (0.1345 vs 0.1356 with scales). The planned
   second cell (`KV_FP8_ROPE=0` first) chases the small term.
2. The failure trajectory this predicts matches the observed one: shallow-
   layer quantization error amplified downstream is exactly "no ticket token
   selected through layer 38, recovery only at 62–74."

**Recommended cell insertion (one env var, one row):** after the in-flight
i8_ring cell, run
`nvfp4_ds_mla + FP8 RoPE + i8_ring + VLLM_NVFP4_MLA_SCALES_FILE=<#145 json>`
on frozen r1. If it recovers, the minimal causal defect is the NVFP4 writer's
default outer scale, and the canonical fix is "enable and canonicalize #145"
— it preserves the 368-byte record, the 600k+ capacity envelope, and lands as
an upstream-reviewable change that already exists. That is the correct,
complete, non-hack win if it validates. Prerequisite check: #145 needs the
b12x latent_scale cache fact (lukealonso/b12x#52) or a one-time compile-cache
clear — fresh cache namespaces already satisfy this.

Calibration caveats to carry, not to skip the cell over: the scales file was
captured at 2048-ctx wikitext-2 on a sibling checkpoint build; per-layer
envelope claims shallow-layer headroom, but deep-context adequacy is exactly
what our frozen gate will test. KLD here is corroborating, not proof — the
frozen rows and entry-layer margins remain the decisive metrics.

## 1. Answers to §9

### Q1 — Is the recovered comparison causal as stated?

Yes, at the granularity §4 claims it. I checked the posture table for
unrecorded differences: execution (eager/MTP0/MNBT/GMU/max-len/prefix-off/
query-split/owner-merge/CKV-prefetch) is held; prompt/checkpoint/scorer
constant; cold verified. Two residual differences are real but acceptable,
and should be recorded as known limits rather than fixed:

- **KV pool width changed** (837,953 → 491,769) with the record format, so
  page-table geometry differs between arms. Post-#85 this is believed
  neutral, and the stride proof covers the tiled top-k path — but not every
  page-table consumer. Residual risk: low; noted.
- **The format change swaps kernels, not just precision.** `fp8_ds_mla` and
  `nvfp4_ds_mla` are different read/write/dequant code paths. "Higher
  precision recovers r1" and "the NVFP4 path has an implementation defect"
  are both consistent with the result — §0 makes the second reading
  concrete. The claim boundary in §4 already refuses to attribute to a
  single representation; keep that refusal until the scales cell runs.

### Q2 — Is restoring i8_ring the right first discriminator?

Yes. Cheapest flip, binary, and its contract is losslessness, so any failure
is a bug by definition, not a tradeoff. One tightening: for the §6.2 branch,
the round-trip proof must assert **bit equality**, not tolerance. "Block-INT8
as lossless byte transport for BF16" is only bit-exact if it is byte
reinterpretation; if the codec involves per-block scaling of BF16 mantissas
it is near-lossless, and near-lossless × 78 layers × 343k tokens is a
hypothesis, not a contract. The proof at tail geometry settles which it is.

### Q3 — Should `nvfp4 + KV_FP8_ROPE=0` be the second cell?

No — two objections, one practical, one from §0:

1. **Priors:** #145's table says the RoPE field is the small term and
   uncalibrated CKV the large one. The scales-file cell (§0) is the same
   cost (one env var, one boot, one row) and aims at the dominant term.
2. **Format entanglement:** confirm `nvfp4_ds_mla + BF16 RoPE` is a real,
   supported production record before booting it. If `KV_FP8_ROPE=0` with
   NVFP4 CKV materializes a third record layout that nobody would ever ship,
   the cell tests unshippable code and its result is not actionable. If no
   supported single-field variant exists, isolate the RoPE writer offline
   instead: encode the frozen BF16 RoPE field through the production compact-
   FP8 writer, decode through the attention consumer, and replay the layer-34
   rank measurement with only that field substituted. Zero boots.

Revised cell order: i8_ring (in flight) → NVFP4+scales-file (§0) →
only-if-needed RoPE isolation (offline preferred).

### Q4 — Operator proof distinguishing inherent quantization from defect

Three measurements, all offline against frozen activations:

1. **Floor comparison:** encode/decode the frozen CKV (and RoPE field)
   through the production writer/consumer; compare per-element error against
   an ideal reference quantizer for the same format (straight
   quantize-dequantize in torch at the declared block/scale geometry).
   Production ≈ ideal (distribution AND max) ⇒ inherent; excess ⇒ defect.
2. **Structure tests:** plot error vs layer index, vs position, vs
   block-index-mod-scale-group, vs channel. Boundary-aligned spikes, shallow-
   layer blowups (the #145 signature), or position-dependent growth ⇒
   defect/format inadequacy; flat stationary noise ⇒ inherent. Note the
   observed failure is depth-dependent (250k passes, 350k fails) — error vs
   position on the real 343k sequence directly discriminates "stationary
   noise whose aggregate interference crosses a threshold" from "error that
   grows with position".
3. **Consequence replay:** rerun the layer-34 rank/needle-entry measurement
   with only the candidate field substituted (ideal-encode vs
   production-encode vs BF16). This ties the arithmetic finding to the
   selection failure without a boot.

### Q5 — Is the complement cell sufficient for additive erosion?

Sufficient to *detect* additivity on the frozen row, provided every cell is
scored with the quantitative metric, not just pass/fail. Two additions:

- Pre-commit the margin criterion once the trace overlay runs on the winning
  posture: e.g., ticket entry layer ≤ L* and presence in ≥ N/16 answer calls,
  with L*/N calibrated from the recovered posture's trace. A cell that
  "passes" with entry at layer 55 is not healthy, and binary pass/fail will
  hide that.
- The randomized ladder, not the complement cell, is what bounds additivity
  across depths. Keep the complement cell to one boot and let §8's ladder do
  the rest.

### Q6 — Evidence before concluding a larger cache record is unavoidable

All of, in order:

1. i8_ring exonerated (bit-exact or fixed);
2. #145 calibrated scales tested on the frozen gate and **insufficient**;
3. Q4's floor comparison shows production error at the ideal-quantizer floor
   with no structural pattern (no remaining defect to fix);
4. consequence replay shows even *ideal* NVFP4 encoding of the frozen
   activations loses the needle ranks (the format itself, not the writer);
5. a mixed-precision alternative inside the capacity budget has been
   evaluated and fails (e.g., NVFP4 CKV + BF16 RoPE if RoPE turned out to
   matter after all, or coarser-but-calibrated variants);
6. the capacity floor can be restored elsewhere (§7 already requires this).

Only after 1–5 is "bigger record" a conclusion rather than a retreat.

### Q7 — Which claim or gate is presently too broad?

1. **§3.2/§3.3's "relevant history has already degraded" lacks its healthy
   control.** The layer-34 ranks (~75k–93k) and the late needle-entry
   trajectory are measured only on the lossy posture. Nobody has shown the
   recovered posture ranks those tokens early. If healthy runs also rank
   them outside top-2,048 at layer 34 and legitimately recover later, the
   "too late to steer" narrative needs revision. The planned trace overlay
   on the winning posture closes this — treat it as required evidence for
   the causal story, not just a promotion metric.
2. **One-row causality is quietly becoming regression-wide language.** §10's
   bottom line ("the permanent fix is the smallest representation or
   transport correction…") is the right bet but is currently licensed by r1
   only; r2 failed with a different morphology (fabrication vs wrong
   retrieval). The four-row gate on the winning posture must run before any
   fix work starts, precisely so a partial cause isn't fixed and declared.
3. **§8 gate 8 "matched prefill and decode performance"** needs a number
   (e.g., within X% of the RC's serialized gates) before it can pass or
   fail. Unquantified gates get argued later.

## 2. Focus guardrails (Derek's ask: efficient path to a correct, complete, non-hack win)

1. **Run the scales-file cell before building any new decomposition
   tooling.** §7's operator decomposition is the right instrument, but #145
   already did the first pass of that work upstream and shipped the
   candidate correction. One env var and one frozen row may replace a week
   of harness construction. Build Q4 tooling only if the scales cell fails
   or partially recovers.
2. **No invented record formats.** Bisect only over configurations that
   could ship. Anything else is a hack with extra steps, and its evidence
   dies with the experiment.
3. **Boot budget:** the review promises ≤2 primary boots + 1 complement.
   With the revised order (i8_ring → scales-file → complement/four-row) that
   budget holds. RoPE isolation goes offline if needed.
4. **Definition of done stays §8** — all eleven gates, with the entry-layer
   margin metric quantified and the four-row gate run on the *minimal* fixed
   posture, not the maximal lossless one. The win condition is: 368-byte
   record preserved (or knowingly traded), frozen + randomized gates green,
   and the fix expressed as a minimal upstream PR (plausibly: canonicalize
   #145 + the #85 already-merged fix + our evidence) rather than a local
   overlay. If the scales cell validates, the entire fix may be
   "flip a default upstream, attach the deep-retrieval evidence they didn't
   have" — which is the cleanest possible landing.
5. **Credit and scope in #182:** the deep-retrieval gate evidence is ours;
   the scale defect finding is #145's author's. Same discipline as with #85.

## 3. Bottom line

ACCEPT the causal claim at its stated boundary; ACCEPT the bisection with a
reordering: i8_ring (in flight) → **#145 calibrated-scales cell** (new,
highest prior, one env var) → complement/margins → four-row gate on the
minimal posture. The RoPE-isolation cell moves offline and may never be
needed. The most likely end state as of tonight: the v20 needle regression
resolves as "NVFP4 MLA KV writer shipped with uncalibrated outer scales;
#145 fixes it; here is the frozen-gate proof" — correct, complete, capacity-
preserving, and almost entirely already reviewed upstream.
