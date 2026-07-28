# Fable review log — v20 needle-regression consulting

Purpose: fast context recovery for Fable across intermittent sessions. Append
one dated entry per engagement; newest entry LAST. Each entry: what happened,
what was decided, what's pending. Durable pointers only — details live in the
referenced docs.

Standing role (Derek, 2026-07-27): Sol does the core work; Fable provides
adversarial review and consulting at each critical finding, keeping the effort
on the fastest sound path to a shippable fixed v20 patch. Derek relays
between Fable and Sol (do not open comms/ channels unless asked; that bus
exists but Derek is the intermediary for this effort).

Key constraint from Derek: FP8 with bounded selection retrieves needles >350k,
so signal physics is not the limit — the fix must restore at least that
functionality under exact v20 top-k. `oldest_boundary` / `bounded_compat` are
compatibility controls, not endpoints.

---

## 2026-07-27 — engagement 1: adversarial review of Sol's reference-oracle plan

Input: `fable-adversarial-review-001.md` (Sol, SHA verified 4fef02d5…).
Output: `fable-adversarial-response-001.md`.

- Reviewed Sol's official BF16/FP32 indexer reference oracle
  (`workspace/vllm-v20-official-fullprecision-reference/`, server-static env
  `VLLM_GLM_INDEXER_REFERENCE_MODE=official_bf16_v1`) and the causal compose
  `compose/glm52-v20-official-bf16-reference-causal-20260727.yaml`.
- FATAL found: `_run_decode` (glm_official_indexer.py:358) raises on 2-D
  decode seq_lens, but the metadata builder unconditionally unsqueezes to
  (B,1) (indexer.py:1158) — every decode step would crash. Fix: squeeze;
  note stored decode seq_lens are already DCP-localized so post-squeeze use is
  correct.
- Guards recommended: `cu_seqlen_ks == 0` assert; score-buffer contiguity
  check; BF16 cache is ~1.94× production indexer cache — memory-fit check.
- Cheapest missing proofs: live micro-smoke (catches the fatal), 8k DCP
  needle check, safetensors dtype check for indexer `weights_proj`, disable
  prefix caching in the causal compose.
- Verified correct (do not re-litigate): DCP merge contract
  (`_merge_b12x_dcp_topk` consumes rank-local logical indices — exactly what
  the reference produces), prefill local-coordinate lengths, decode insertion
  under DCP, cross-layer skip topology parity.
- Tangents parked: graph-safety/compile-key work, HF contract archaeology,
  randomized harness, per-layer instrumentation.
- Verdict: no-go as-was; go after fix + smoke. Also advised entering the
  post-pass component ladder at the K-cache swap (component 5) — Sol's own
  retention table (143-row FP8 delta vs ~20-row BF16 deltas) says the mass is
  there; saves 2–4 boots.

## 2026-07-27 — engagement 2: upstream RC review → experiment re-order

Input: Discord announcement of RC image
`voipmonitor/vllm:gilded-gnosis-v20-vllm0c79e41-sic3828fd-fi801d57a-cu132-20260727`
(digest `131481b0…`), rtx6kpro issue #33 / PR #39, SparkInfer #79/#81/#85.
Output: `fable-adversarial-response-002.md`.

- HEADLINE: SparkInfer #85 fixes a silent tiled-top-k page-table row-stride
  bug (cubin compiled at one page-table width reused at another reads wrong
  rows; row 0 correct; order/restart-dependent). Introduced `d4f82a6`
  2026-07-22. Ancestry verified via GitHub API: bug IS in `e603f74` (the
  SparkInfer inside Sol's base image digest `10261c7d…`); fix merged
  2026-07-27 (`f06881a`), present in RC's `c3828fd`. vLLM side (`0c79e41`)
  identical between Sol's base and the RC.
- Consequence: all accelerated-arm evidence (incl. the captured 1,905/2,048
  selection and the original 350k failures) was collected on a bugged build.
  The bug's width/order/restart signature fits the 250k-control-first →
  350k-fail pattern disturbingly well (block-table width scales with context;
  persistent /root/.cache volume can carry poisoned cubins). Sol's replay
  evidence survives (fresh single-width compiles can't trigger it).
- Agreed plan (Derek confirmed Sol is proceeding): (0) stock-RC four-row
  frozen gate FIRST — fresh cache volume, prefix caching off, one boot; pass
  ⇒ root cause is #85, jump to randomized validation + rebase; (1) only if
  350k still fails, rebase reference image onto RC base (+ decode fix,
  re-run no-model gates/smoke), re-capture accelerated selection, then the
  reference causal gate.
- Secondary RC content: #79/vLLM#178 owner-merge + DCP8 query_split
  combination (~10% perf, post-fix); #81 calibration probe (adopt for prod
  candidates later; causal composes stay fully pinned); upstream retains
  exact top-k and excludes bounded_compat — goals aligned.
- Methodology lesson recorded: replay/operator gates with fresh compiles
  structurally cannot see compile-cache cubin-reuse bugs; every diagnostic
  mode's gate needs a short live smoke.

- Follow-up (same day): a second Discord announcement in the NF3 channel is
  the SAME image (identical tag/digest, same issue #33 / PR #39) — unified
  refresh announced per-audience, not a second build. NF3 helper path
  (MODEL_FAMILY=glm52-hybrid) is in the same image; the NF3 57.3 tok/s figure
  is a retained control, not rerun on this image.

## 2026-07-27 — engagement 3: stock-RC gate result — #85 not sufficient

Result (Sol): on the #85-fixed stock RC, 250k control PASSED, first 350k row
STILL FAILED (cold, cached=0, clean termination, returned "27" instead of
738216). Run valid. r2/r3 finishing for the record.

Interpretation agreed:
- #85 was a real but independent bug; FP8/accelerated-indexer hypothesis is
  back as lead, strengthened (its strongest challenger just fell).
- The identical wrong answer "27" across two different SparkInfer builds
  points to deterministic arithmetic divergence, not order/cache corruption.
- Correction from Sol (accepted): the helper did NOT auto-enable new DCP
  paths — live env explicitly held query_split=0 / owner_merge=0, calibration
  logged `skipped: explicit-compressed-dma`. So the stock-RC run was a clean
  single-variable #85 A/B (same transport/policies, only the stride fix
  changed, failure persisted). Drop the "two transport stacks" framing.
- Determinism refined: r1 reproduced "27" across builds (within-row
  deterministic); r2 failed with a DIFFERENT fabricated ticket — expected
  (different frozen prompt), does not contradict per-row determinism. Open
  micro-check (free if archived): does r2's fixed-RC output match r2's
  old-build output? Match ⇒ build-stable deterministic scorer divergence;
  mismatch ⇒ #85 altered trajectory without restoring retrieval. Either is
  coherent; ladder wants to know if the accelerated arm is build-stable.
  Note r1 wrong-retrieval vs r2 fabrication are different failure
  morphologies — consistent with a scorer-field perturbation pushing
  different needles across the selection boundary differently.

Next decisive test: reference oracle causal gate on the CLEAN RC base
(digest `131481b0…`; vLLM `0c79e41` identical so the two reference files
apply as-is). Pre-boot checklist from response-001 stands: decode squeeze
fix, cu_seqlen_ks assert, re-run no-model gates, live micro-smoke, 8k needle
check, prefix caching off, fresh cache volume. Re-capture the accelerated
selection (old 1,905/2,048 figure is from the bugged build) when convenient.

## 2026-07-27 — engagement 4: reference oracle live on clean RC; causal rows running

Sol status (all response-001 checklist items executed):
- Stock-RC gate completed: 250k exact, 350k r1/r2/r3 ALL miss (identical
  transport: i8_ring, query_split/owner_merge off, fresh cache). #85 real
  but not the regression cause. Sol also independently reproduced the #85
  stride bug on the old image (rows 1–16 misread on cubin reuse) and proved
  it clean on the RC.
- Corrected reference oracle rebuilt on RC base: accepts (B,1) decode
  lengths, rejects speculative shapes, enforces zero-based prefill key
  windows, separate BF16x128 cache namespace. All no-model proofs pass;
  prior fingerprints reproduce byte-for-byte.
- Reference boot healthy on fixed RC: NVFP4 MLA KV + FP8 RoPE + i8_ring
  retained (held constant; indexer is the only semantic variable), PR#84
  absent, KV pool 837,953 tokens @ 360k, zero restarts.
- Live gates: 499-token smoke PASS; cold 7,847-token DCP needle exact PASS;
  frozen cold 250k control exact PASS in 362 s.
- DECISIVE 350k r1/r2/r3 running at time of writing.

Pre-agreed interpretation:
- All three recover ⇒ accelerated FP8 indexer trajectory causal (not
  selector, top-k kernel, stride, NVFP4 KV, FP8 RoPE, i8_ring). Enter ladder
  at component 5 (BF16 keys vs FP8+scale indexer cache); same instrumented
  boot re-captures the accelerated selection delta (replaces contaminated
  1,905/2,048). Stop condition: smallest component that flips the gate ⇒
  immediately write canonical-format SparkInfer/vLLM PR w/ four-row evidence.
- Partial recovery (1–2 rows) ⇒ do NOT force into either tree branch;
  compare per-row reference-vs-accelerated selection trajectories first.
- All three still miss ⇒ FP8 scoring no longer treated as causal; go to
  review-001 §7 refuted branch (verify loaded weights at every boundary
  before training-contract archaeology).

PENDING: 350k r1/r2/r3 reference-oracle results.

## 2026-07-27 — engagement 5: adversarial review 002 → response 003

Input: `fable-adversarial-review-002.md` (Sol; oracle live, 250k passed,
350k rows in flight). Output: `fable-adversarial-response-003.md`.

Verdict: ACCEPT WITH CONDITIONS. Key findings:
- SEVERITY 1: posture confound — stock causal run was MTP3/graphs/480k,
  reference is MTP0/eager/360k. Verified in code that next_n selects
  different indexer decode metadata/kernel paths, and failures form in the
  first 16–25 decode tokens. Condition A: demonstrate a stock-RC 350k miss
  in reference posture (MTP0/eager/360k, r1 only, one cheap boot — or cite
  an archived equivalent) before publishing attribution.
- SEVERITY 2: 1,905/2,048 accelerated-selection figure predates #85;
  re-capture on clean RC before ladder ranking or publication.
- SEVERITY 3: frozen-set narrowness (same token count/needle depth) —
  claim wording "causal for the frozen set" only.
- Big time-saver accepted into the plan: OFFLINE replay decomposition of the
  FP8 divergence (official-everything + FP8-K-only / FP8-Q-only /
  FP8-accum-only cells on the frozen activation) — zero boots; live ladder
  becomes one confirmation boot of the winning cell.
- Zero-pass branch gets a new first stop: checkpoint config
  rope_scaling/original_max_position_embeddings — 250k-pass/350k-fail
  straddles 262,144; a declared context-extension contract missing from BOTH
  indexer arms uniquely predicts zero-pass. Minutes to check.
- Ladder amendments: pre-commit "material divergence" definition; fallback
  rerank design stays one sentence until decomposition justifies it; each
  promotion boot repeats micro-smoke + 8k needle.
- Publication guardrails for #182: no attribution before Condition A; #85
  independent-repro proof publishable now; keep found-upstream vs
  proved-independently distinction crisp.

Sol pre-boot update (accepted): zero-local-length DCP cell added to the
synthetic gate; chunk16-vs-64 bit exactness proven; checkpoint positional
contract checked — 1,048,576 positions, plain theta-8M RoPE, NO rope_scaling
boundary at 262,144. Zero-pass branch therefore now leads with the
deeper-layer replay (config archaeology pre-closed). Also: bounded selection
already retrieves at 350k (Derek), so a zero-pass cannot be read as a model
capability limit. Corrected CN4 reference boot starting.

## 2026-07-27 — engagement 6: pre-gate status; OOM fixed; plan fully aligned

- First 350k attempt on the reference was an INVALID resource failure (64-row
  scorer chunk OOMed by 492 MiB — the transient-scratch risk from
  response-001 §2.3). Fix: q_chunk_rows 64→16, validated bit-exact across
  three seeds at production geometry (~387 MiB scratch saved). Correctly
  classified and preserved as resource-not-quality.
- Reference decode runs ~1.7 tok/s (deliberately unoptimized) — frozen rows
  need 16–25 output tokens, so this is fine; slow 8k gate = coherence
  side-check generation, not a wedge.
- Sol's published Next list now IS the response-003 tree verbatim: 8k sanity
  → frozen sequence (control first) → on full recovery, immediate Condition A
  stock control in identical MTP0/eager/360k posture → offline FP8-Q/K/accum
  decomposition, only winning cell gets a live boot → on zero-pass, deeper-
  layer replay (emphasis kept: boundary-verify the reference's own inputs at
  layer ≥1 before "shared path" conclusions).

- Corrected-boot cold 8k needle: PASS exact (738216, cached=0, finalized,
  stop). Decisive frozen sequence STARTED (250k control first, then 3×350k).
- Frozen 250k control: PASS exact and cold (738216, cached=0, stop). Gate
  valid; advanced to 350k r1 — first genuinely causal discriminator in
  flight.

## 2026-07-27 — engagement 7: CRITICAL — reference oracle FAILS r1 with same "27"

Result: 250k control exact/cold; 350k r1 cold, stop, answered "27" — the SAME
deterministic wrong answer as the accelerated path. Valid semantic failure
(crossed old OOM point cleanly). r2/r3 finishing for zero/partial-pass
classification.

Refuted: "accelerated FP8 scorer arithmetic alone causes the regression."
Also mooted: Condition A posture confound (stock failed MTP3/graphs,
reference failed MTP0/eager — posture rescues nothing).

Interpretation: identical wrong answer across three arms (bugged stock, fixed
stock, full-precision oracle) ⇒ fault lives in what both arms SHARE:
(a) inputs — hidden states from prefill running NVFP4 MLA KV + FP8 RoPE +
i8_ring (all held constant; v19 precedent: fp8 transport broke deep
retrieval, prod runs fp8-DMA=0/bf16 for that reason); or (b) post-selection
consumption — index space, DCP merge, sparse-attention gather. Also noted:
2^18=262,144 sits between pass (245,497) and fail (343,727) lengths;
positional contract cleared but index-space/metadata 18-bit boundaries are
not.

Ranked next experiments (relayed to Sol):
1. Needle-inclusion instrumentation in the Python reference at r1's
   answering decode steps (near-free; splits post-selection vs upstream).
2. Failure-onset mapping on STOCK (valid probe now; ~250k/258k/265k/275k/
   300k rows, needle 40%): sharp cliff at 262,144 ⇒ boundary bug; gradual ⇒
   precision accumulation.
3. Config parity audit of PR#84 bounded-selection passes (did they run with
   i8_ring/FP8-RoPE/NVFP4? If yes those are exonerated as sufficient causes).
4. Precision levers as indicated: i8_ring→lossless BF16 wire (v19 analog),
   KV_FP8_ROPE=0, NVFP4 — one boot + one row each, guided by 1–3.
5. Sol's deeper-layer reference verification — still required before
   declaring the official contract insufficient, but now AFTER 1–4.
Also: record per-row answer morphology on r2/r3 (identical-vs-different
across arms is itself signal).

## 2026-07-27 — engagement 8: CRITICAL — official scorer never selects the ticket early

Sol's layer trace (336/336 records, 21 layers × 16 decode calls, r1): no
ticket token through layer 38; one value token 1/16 calls at layer 42; all
three tokens together once at layer 62; broad only at layer 74 (15/16) —
too late; answer still "27". Matches the failing exact-selector trajectory.
Downstream page-table/gather loss ruled out as primary: candidates never
reach attention early. Selection itself omits the needle ON THE BIT-EXACT
OFFICIAL SCORER over real activations.

Remaining classes: (a) scorer INPUTS degraded — shared upstream precision
stack (NVFP4 MLA KV + FP8 RoPE + i8_ring accumulating representation noise
by 343k; v19 precedent: same checkpoint retrieved deep under bf16 posture,
fp8 transport broke it); (b) residual reference input-provenance gap at
layers ≥1 (less likely, needs its one deeper-layer verification);
(c) genuine trained-scorer limitation (weakest; contradicts v19 history).

Recommended (relayed): ALL-LOSSLESS POSTURE boot — single decisive
experiment: i8_ring→lossless BF16 wire + KV_FP8_ROPE=0 + highest-precision
main KV, stock exact selector, r1 only. Recovers ⇒ (a); bisect 3 levers
(≤2 boots) ⇒ minimal culprit ⇒ fix is that component's quantization or a
measured config change; if the culprit must stay lossy for capacity, the
broad-candidate + higher-precision-rerank design earns design time.
Still fails ⇒ (a) dead; deeper-layer verification then training-contract.
Parallel near-free: onset mapping on stock (cliff@262,144 vs gradual);
PR#84 config-parity audit. r2/r3 no longer decisive; finish for record.

## 2026-07-27 — engagement 9: BREAKTHROUGH — all-lossless posture recovers r1

Result (Sol): frozen 350k r1 recovered EXACTLY under the higher-precision
shared-input posture (738216, cold, 4 output tokens, stop). Upstream
representation precision is causal for this row; official scorer exonerated.

Guidance relayed:
1. Bisect from recovered posture, one lever→lossy per boot, order:
   i8_ring (experimental compressed DMA, never auto-selected upstream, v19
   fp8-DMA=0 analog) → KV_FP8_ROPE (368B record) → NVFP4 MLA KV. Beware
   additive margin erosion — check the complement before declaring a unique
   culprit.
2. Make boots quantitative: use the layer-trace tooling to record the
   "ticket entry layer" into top-2,048 per posture (lossy baseline: 62–74).
   Continuous margin metric ⇒ ranks lever contributions, feeds the eventual
   operator-level promotion gate.
3. Fix-shape triage: i8_ring ⇒ config fix (drop experimental wire in quality
   posture, matches upstream defaults); KV_FP8_ROPE/NVFP4 ⇒ operator-level
   bug-vs-inherent decomposition (round-trip tests on frozen activations)
   before accepting capacity tradeoffs.
4. Scope: one row so far. Full four-row gate in winning posture →
   randomized 50k–475k sweep → §14 promotion evidence. #182 claim: indexer/
   scorer chain exonerated by oracle (publishable negative result) + #85
   proof.

Derek's check (accepted, important): bounded selection works at depth UNDER
THE LOSSY POSTURE — same "noisy" activations retrieve and answer perfectly
once the needle is in the candidate set, and the stack passes CC32/KLD
quality gates. That shape fits a surgical implementation defect in one
component's tensor path (368B FP8-RoPE record, i8_ring codec block
boundaries) at least as well as inherent quantization noise. Corrected
claim: "the lossy shared-input stack is causal for r1; MECHANISM
UNDETERMINED." Bisection isolates the component; then an operator-level
round-trip test of that component against its format's theoretical
quantization floor decides defect (error above floor / structured) vs
inherent (error at floor). Structured/boundary-aligned error or onset
cliff ⇒ bug ⇒ possibly keep compression AND retrieval.

## 2026-07-27 — engagement 10: review 003 → response 004; #145 lead found

Input: `fable-adversarial-review-003.md` (posture-causality + bisection
plan). Output: `fable-adversarial-response-004.md`.

HEADLINE: vLLM PR #145 ("calibrated NVFP4 MLA KV outer scales") — INCLUDED
in the RC image, DEFAULT-OFF behind VLLM_NVFP4_MLA_SCALES_FILE — documents a
known defect in the NVFP4 writer: default outer scale 1.0 puts shallow
layers in E4M3-subnormal block scales, "error amplified downstream" (kv_c
spans 240× across layers). Its KLD table: nvfp4-no-scales-fp8rope (≈ failing
posture) 0.168 vs nvfp4+scales 0.1345–0.1356 vs fp8_ds_mla 0.1263 — so
uncalibrated CKV is the dominant term and FP8-vs-BF16 RoPE is small.
Predicted trajectory (shallow-layer error amplified downstream) matches the
observed late needle entry (layers 62–74).

Verdict: ACCEPT causal claim at stated boundary; ACCEPT bisection with
REORDER: i8_ring cell (in flight) → #145 scales-file cell (one env var, one
row; needs b12x latent_scale fact or fresh compile cache — fresh cache
already standard) → complement/margins → FOUR-ROW gate on minimal posture.
KV_FP8_ROPE=0 second cell rejected: chases the small term, and
nvfp4+BF16-rope may not be a supported record (no invented formats — bisect
only shippable configs; RoPE isolation goes offline if needed).

Other key answers: i8_ring round-trip proof must assert BIT equality (block-
INT8 "lossless" is only bit-exact if byte reinterpretation); Q4 floor-test =
production-vs-ideal-quantizer error + structure tests (error vs layer/
position/block-boundary; position-growth vs stationary discriminates
accumulation from defect) + layer-34 consequence replay; Q7 too-broad items:
(1) "history already degraded" lacks healthy control — need recovered-
posture trace showing early ranks; (2) one-row causality must not become
regression-wide language — four-row gate before fix work; (3) §8 perf gate
needs a number.

Likely end state: "NVFP4 writer shipped uncalibrated outer scales; #145
fixes it; frozen-gate proof attached" — 368B record preserved, capacity
kept, fix = canonicalize an existing upstream PR. Correct, complete,
non-hack.

Derek prompt → records check on int8 wire (engagement 10 addendum):
- design/int8-dma-ag-design.md + fp8-rank-consistency-gate-verdict.md (v19,
  2026-07-19): INT8 ag proven BIT-IDENTICAL ACROSS RANKS; vs BF16 it is NOT
  bit-lossless — symmetric block-INT8 codec, 128-val blocks, FP32 scale,
  132B/block, error ≤ amax/254. Sol's "lossless byte transport" wording must
  be corrected to "rank-consistent, near-lossless (amax/254) codec".
- int8-dma-ag-gate-verdict.md: on v19, INT8 wire recovered deep retrieval
  6/6 at 300k AND 350k where E4M3 wire failed — codec noise demonstrably
  doesn't break 350k retrieval on this checkpoint.
- MEASUREMENT-LIBRARY.md:382 (2026-07-25 static bisect): v20 regression
  onset occurred between two builds BOTH running i8_ring — wire mode cannot
  explain the onset (held constant across it).
⇒ i8_ring cell outcome strongly predicted: exonerated. If it fails, treat as
harness/config anomaly first. Prior mass now almost entirely on #145
NVFP4-scales cell. Historical cross-check for Sol: did the NVFP4 MLA KV
format/writer arrive at the post-6d32 onset boundary? Would close #145 from
the historical side.

RESULT: i8_ring cell PASSED exactly (cold 343,727-token r1 → 738216, 4
tokens, stop) — as predicted by the three prior records. Transport
exonerated; causal set = {NVFP4 CKV, compact FP8 RoPE}, CKV dominant per
#145 priors.

DEREK CONSTRAINT (binding, 2026-07-27): 656-byte fp8_ds_mla record is NOT an
acceptable endpoint — small (368B-class) records required for speed; they
must be made accurate. The fp8 recovery is diagnostic ceiling evidence only.
Mechanistic support that this is achievable: PR#84 bounded selection
retrieved at 350k UNDER the lossy 368B posture ⇒ the needle's KV survives
the small record (readout works); the deficit is indexer ranking MARGIN,
not information loss ⇒ modest accuracy fixes to the same format can flip it.
Escalation order (all keep 368B until step 4): #145 scales → deep-context
recalibration of scales file → floor-test-driven writer/reader fixes →
(only if forced) small-budget reallocation (~400B NVFP4+BF16-rope) →
(last) deterministic broad-candidate + exact rerank. oldest_boundary stays
rejected: age heuristic, unbounded failure modes, no mechanism.
Updated probabilities: ~65-70% "368B + writer accuracy fix", ~15% small
reallocation, ~10% rerank architecture, <10% stuck.

## 2026-07-27 — engagement 11: scales cell PASSES — but as an INTERACTION claim

Result (Sol): official exact scoring + #145 calibrated compact-NVFP4 scales
→ frozen r1 EXACT (738216, cold, 667 s) on the small-record posture. Sol
frames it as an interaction: official scorer alone failed; "calibrated
scales alone failed on the earlier stock-scorer image"; together they pass.

CRITICAL provenance question raised (blocking the interaction claim): which
image did the "scales alone + stock scorer" fail cell run on? If pre-#85
old base ⇒ contaminated leg (same trap as the 1,905/2,048 figure) ⇒
interaction unproven. Required cheap check: scales + stock accelerated
scorer + CLEAN RC, frozen r1 (one boot, one row).
- If it PASSES: no interaction; scorer exonerated stands; endpoint = ship
  #145 default + evidence. Dream case.
- If it FAILS: interaction real ⇒ accelerated FP8 indexer scorer NOT fully
  exonerated (adds its own ranking noise over calibrated inputs; additive
  margin erosion across two stages). Endpoint = #145 AND an indexer-side
  accuracy fix (offline FP8-K/Q/accum decomposition returns, now against
  calibrated activations). Reference scorer (1.7 tok/s) is unshippable.
Priority relayed: pin evidence → archives provenance check (free) →
clean-RC scales+stock cell if needed → healthy-posture trace (now doubles
as margin audit: entry layer 8 = fix, entry layer 55 = luck) → four-row
suite (r2 fabrication morphology is the wildcard).

Sol confirmed: interaction wording PROVISIONAL pending provenance; will pin
the archived scales-only miss's exact SparkInfer/vLLM revisions first; if
pre-#85, the clean-RC scales+stock-scorer cell runs BEFORE any trace or
four-row expansion.

PROVENANCE RESOLVED: the scales-only miss ran on SparkInfer be0edcaa —
PRE-#85 ⇒ that leg contaminated; interaction NOT proven; Sol corrected the
comms record. Decisive clean-RC complement staged and booting: stock
accelerated scorer + #145 scales, matched MTP0/eager/360k/i8_ring posture,
fresh cache. PASS ⇒ no interaction, fix = #145 default flip + evidence.
FAIL ⇒ interaction real, indexer-side FP8 accuracy work needed (offline
decomposition vs calibrated activations).

## 2026-07-27 — engagement 12: ROOT CAUSE CONFIRMED — #145 scales sufficient

Result: clean-RC stock accelerated scorer + #145 calibrated NVFP4 scales →
frozen r1 EXACT (738216, stop, 4 tokens, cached=0, 327 s). NO interaction;
no reference scorer or indexer patch needed.

ENDPOINT: keep v20 exact top-k + stock accelerated indexer; enable/
canonicalize #145 calibrated NVFP4 MLA KV writer scales. 368B record,
capacity, and speed preserved (327 s vs 667 s reference arm). Meets Derek's
binding constraint exactly.

Remaining confirmation gates (order agreed): healthy layer-entry trace with
pre-committed margin criterion → frozen four-row suite (r2 morphology the
loose thread) → randomized 50k–475k ladder + §8 promotion list (incl.
500k@480k capacity on NVFP4 posture, quantified perf gate). Calibration
file was 2048-ctx wikitext-2 capture — ladder is its real test; remedy if
any depth wobbles = long-context recapture, same format.

Upstream landing: #145 currently "included for testing, not requested for
canonical merge" (issue #33) — our evidence justifies canonicalization +
GLM-5.2 default-on. Credit: MadeBy561 (#145 finding+fix), voipmonitor (#85
stride fix), ours = causal isolation + validation. Postmortem note: both
red herrings (stride bug, scorer arithmetic) were real defects cleared en
route.

Derek Q: are #145's scales right for us (nvfp4 vs e4m3 concern)? Answer
relayed: no format mismatch — E4M3 is the NVFP4 block-scale format, the
outer scale exists to keep block scales out of E4M3 subnormals; scales are
a correctness calibration (plateau: no subnormals, no clipping), not a
hyperparameter to tune. NOT guaranteed: capture provenance (MadeBy561's
checkpoint build, 2048-ctx wikitext) vs our checkpoint + long-ctx stats.
New promotion-gate items — "adopt the mechanism, own the calibration":
(1) provenance audit: measure per-layer max_abs(kv_c) on OUR checkpoint
incl. a 343k prompt, compare vs JSON's retained max_abs envelope;
(2) saturation counters on frozen r1 (zero subnormal, ~zero clip);
(3) if marginal ⇒ recapture scales file on our checkpoint at long ctx
(same format/knob, our provenance).

Derek directive: calibrate OUR OWN scales + sensitivity work. Agreed shape
(guardrailed): (1) recapture per-layer max_abs(kv_c) on OUR checkpoint with
long-context prompts incl. the frozen 343k row; (2) OFFLINE sensitivity
sweep over outer-scale headroom measuring subnormal/clip rates + round-trip
error per layer (zero boots) — confirms plateau width, catches long-ctx
drift/layer-dependence; optional: candidate files through the layer-34 rank
replay for indexer-margin sensitivity; (3) ONE live confirmation with
saturation counters. AVOID: live-boot sweeps on retrieval/KLD decimals —
plateau not peak; overfits frozen rows; ruins upstream reviewability.
Deliverable: our own scales JSON w/ provenance (checkpoint hash, prompt
mix, plateau + saturation evidence) shipped alongside the #145 mechanism —
independent recapture validating their method.

Redirect plan written for Sol: `fable-redirect-001-scales-calibration.md`.
P0 = let the in-flight same-process breadth run (250k + r2/r3 + r1) finish
— sanctioned, breadth-not-promotion, STOP-branch if any row misses.
P1 = capture our per-layer max_abs(kv_c) (long-ctx prompt mix, piggyback
P3 boot). P2 = offline envelope audit + headroom sweep, pre-committed
ADOPT-vs-REGENERATE decision rule, zero boots, no live sweeps. P3 = one
instrumented boot: saturation counters + healthy layer-entry trace +
margin-criterion pre-commitment. P4 = promotion suite (fresh control-first
four-row, randomized ladder w/ margins deciding if any residual term ever
gets discussed, KLD, quantified perf, 500k@480k). P5 = upstream package
(canonicalize #145 + our scales artifact + #182 report + credits; correct
i8_ring wording per design/int8-dma-ag-design.md).

## 2026-07-27 — engagement 13: Tier-1 spec written and PARKED

Derek directed the long-term fix be the general-purpose one, not shipped
magic weights. Spec written: `design/nvfp4-dynamic-second-level-scale-spec.md`
— dynamic per-token second-level scale computed at write time
(s_t = amax/(6·448)), making every record self-describing; no calibration
artifact for any model ever; subnormal/clip positioning failures impossible
by construction. Two storage options (side table +1.1% KV preferred vs
inline 384B +4.3%), Phase A audit decides. Pre-committed perf/capacity
budgets; validation contract reuses all built instruments (floor test,
saturation counters, margin trace, four-row, ladder, KLD). Upstream story:
"#145 fixed positioning with calibration; this removes the need for
calibration" — SparkInfer + vLLM PRs, #145 static path as one-cycle
fallback. PARKED until redirect-001 P0–P4 complete + Phase A questions
answered. Interim ship remains the validated static calibration.

## 2026-07-27 — engagement 14: Tier-1 UNPARKED by Derek — Phase A executed, draft implemented

Derek directive: answer the spec's open questions from full source, branch,
start building; priority = keep 368B records with the scale baked in.

DECISIVE Phase A finding: the 368B record has 12 bytes of zero pad at
[292,304) — the per-token fp32 scale fits at [292,296) with ZERO record
growth, and the record's RoPE lane already implements the identical
per-token-scale pattern. Readers already move those bytes (prefill v4.f32
load at 288 discards .y; decode stages the whole [288,368) tail) ⇒ zero new
tail traffic. All transports raw-byte; prefix hash token-only (verified).
All 7 spec questions answered in
`design/nvfp4-dynamic-second-level-scale-phaseA-addendum.md`.

Implementation drafted on branches at exact RC pins:
- workspace/b12x-nvfp4-dynamic-scale (nvfp4-dynamic-token-scale @ c3828fd):
  writer per_token_scale mode (warp-bfly amax reduce, s_t=amax/2688 fp32 @
  292, groups relative to s_t, spec version 2→3), decode+MG dequant leaves
  reading kv_sc, io/io_mg scalar scale gathers (8-aligned pair at 288, DSV4
  footer idiom), new test file test_mla_kv_cache_per_token_scale.py (ABI,
  positioning invariant incl. #145-defect repro, accuracy dominance, zero
  edge). kernel.py/api.py threading delegated to subagent (in flight),
  flagged as review item #1.
- workspace/vllm-nvfp4-dynamic-scale (@ 0c79e41): committed 91dff5a9 —
  VLLM_NVFP4_MLA_DYNAMIC_SCALE gate, fail-closed checks, writer-call and
  kernel-kwargs wiring, scales-file mutual exclusion.
All hand-edited files pass py_compile. NOTHING COMPILED/RUN — no CUDA on
this Mac; CN4 runbook (7 ordered gates) in the addendum §4.

Threading agent completed and diff-reviewed (traits field + central
fail-closed checks; decode/MG kv_sc smem BI×4 allocations following the
DSV4 footer idiom; api/kernel kwargs default-off; compile specs bumped
decode 18→19, MG prefill 4→5, writer 2→3; latent_scale identity forced on
in-mode). All 11 changed b12x files pass py_compile. COMMITS:
- b12x nvfp4-dynamic-token-scale: 0d9aead9 (416 insertions, 11 files)
- vllm nvfp4-dynamic-token-scale: 91dff5a9

## 2026-07-27 — engagement 15: handoff to Sol + unintended-effects audit

Derek redirected: hand CN4 work (compile/tests/image) to Sol now. Handoff
written and delivered: `fable-tier1-dynamic-scale-handoff.md` (pins, record
contract, authorship-by-layer with agent-threaded layer flagged review-first,
7-gate condensed test order, watch-items). ssh to CN4 not used (first key
rejected; Derek redirected before retry — CN4 access to be done by Sol).

Unintended-effects audit (20-point sweep, both branches) — CLEAN. Verified:
off-mode PTX-identity reasoning per leaf (pure renames under const_expr);
kv_sc buffer parity gather↔consumer (shared `kv_sc_addr + buf*kv_sc_buf`
expr, prologue=buf0); smem additions fail loudly if over budget (compile
gate, not silent); writer warp-0 shuffle is full-warp (no divergence
hazard); disjoint stores (s_t@292 lane0, pad@296+ tids<8, scales lanes
0-31); zero-token/zero-group guards; s_t bit-exactness argument (order-
independent fmax + same f32 constant rounding in kernel and torch test);
4B alignment of dst+292; 432B/non-fp8-rope arms unreachable in-mode via
traits; DSV4 overlap-footer path unreachable; prewarm dummy-cache safe;
spec-extend-as-decode + CKV-gather-requant + graph-capture paths carry the
mode consistently; raw-byte transports carry s_t; non-MLA code untouched.
Two soft notes (in handoff): (a) env silently ignored on non-B12X backends
(matches KV_FP8_ROPE precedent); (b) decode P·V extra 4B smem load per pair
in-mode — hoist mitigation documented if the ≤1% budget is threatened.
CPU-torch algorithm proof: venv prepared; superseded by Derek's redirect to
Sol's GPU gates (the unit tests carry the same assertions on real hardware).

## 2026-07-27/28 — engagement 16: Sol executing Tier-1 gates; KLD phase live

Sol progressed through the gate chain to the KLD quality gate (implies
earlier gates passed — compile, suites, image; awaiting his consolidated
report for the record). Static control run 1 ACCEPTED: mean KLD
0.1457421454, all 2,047 positions, pinned input IDs, clean shutdown,
compile-cache proof of per_token_scale=false. Earlier "failure" root-caused
to host-side unreadable root-owned cache files; proof now reads through the
pinned image. Resuming static 2–3, then dynamic 1–3.

Guidance relayed: (1) no cross-comparison to #145's published KLD table
(different harness/protocol) — dynamic-vs-static within Sol's protocol is
the only valid comparison; (2) PRE-COMMIT acceptance before dynamic data
exists: dynamic mean ≤ static mean + 2σ(static 3-run spread) ⇒ pass;
worse ⇒ stop and investigate as an implementation defect (identical format
capacity ⇒ no legitimate tradeoff explanation).

MILESTONE (Sol status table): Tier-1 dynamic mode gates —
writer/reader GPU correctness PASSED; production GRAPH-MODE boot PASSED;
frozen 250k/350k causal gate 4/4 EXACT; randomized 50k–475k ladder 6/6;
prefill +0.7%..+1.9% (limit 2%) PASSED; KV capacity 550,144 unchanged
PASSED. The calibration-free mode fixes the needle regression end-to-end
on small records with exact top-k — the specified endpoint.

Remaining: KLD n=3 (framework relayed: rank test + paired per-position
block bootstrap + determinism trap check; gate is non-inferiority);
matched decode baseline (same image/posture static mode; P·V hoist is the
documented fix if >1%); restart/warm repeatability (+ two added asserts:
warm cubin reuse, pre-bump cubins refused); packaging (PR pair with
off-mode byte-identity statement, evidence bundle, credits, graph-safety
note).

## 2026-07-28 — engagement 17: draft PRs LIVE; GHCR push blocked on routing

Per Sol's packaging handoff (fable-dynamic-scale-packaging-handoff-20260728.md,
patches#248 acked) and Derek's direction:
- Committed Sol's uncommitted production-reader e2e parameterization
  (b12x 2b72c29d) — do-not-lose item preserved.
- Branches pushed: yatesdr/b12x + yatesdr/vllm-opt nvfp4-dynamic-token-scale.
- DRAFT PRs OPEN, cross-linked, data-only bodies with full test tables:
  - SparkInfer: https://github.com/local-inference-lab/sparkinfer/pull/86
  - vLLM:       https://github.com/local-inference-lab/vllm/pull/189
  Placeholders held for: KLD n=3, matched decode, repeatability, GHCR digest
  (append-before-undraft per handoff §7).
- KLD comparison table delivered to Derek (bounded 0.160441±0.002979 /
  uncalibrated-exact 0.158237±0.004684 from design/v20-pr84-kld-n3-report;
  static #145 partial n=2 0.148441±0.003816; dynamic queued) with
  cross-image comparability + selector-insensitivity caveats.
- BLOCKER: Mac→CN4 direct ssh sessions hang (auth-probe fast-fails, real
  sessions stall — matches Sol's dev#203 "direct route dropped; CN3 jump
  works"). Did NOT guess CN3 creds (prod box). GHCR push staged as
  harness/push-dynamic-scale-review-image-ghcr.sh — read-and-push only
  (tag existing image db82fdcb… → ghcr.io/yatesdr/glm52-serve:
  gilded-gnosis-v20-nvfp4-dynamic-scale-review-20260728), zero contact with
  Sol's running KLD. Needs: working route (Sol/Derek runs script, or route
  fix) + one-time docker login ghcr.io w/ write:packages PAT.

PENDING:
- GHCR push via script (route blocker) → digest into handoff + both PRs.
- Sol: KLD n=3 completion, matched decode, restart/repeatability →
  evidence addenda → append to PRs → undraft decision.
- Healthy-posture layer trace w/ entry-layer margins.
- Four-row frozen suite on the winning minimal posture.
- Then randomized ladder + §8 promotion gates.
- Onset-boundary vs NVFP4-arrival historical cross-check.
- Recovered-posture needle trace (healthy control for the rank claim).
- Four-row gate on minimal posture; then randomized ladder + §8 gates.
- r2/r3 reference-arm results for the record.

## 2026-07-27 — engagement 8: zero-pass complete; trace discriminator built

- Full reference result: 250k control EXACT; r1/r2/r3 all ABSENT and all
  finalized `27`. Summary SHA-256:
  `14757653116ea69d717396742333d7c9f376807e40cd21bc778111713d324229`.
- Morphology correction: fixed-stock r2 returned fabricated
  `MAINT-2024-0917`, while reference r2 returned `27`. The official scorer
  changes some trajectories but is not sufficient for retrieval.
- Parity audit complete: the working `oldest_boundary` positive control used
  the same NVFP4 MLA KV, FP8 RoPE, lossless `i8_ring`, TP4/DCP4/MTP3, and
  MNBT 3072 stack. It passed the frozen gate and randomized 50k--475k ladder.
  Those components are not sufficient causes; do not schedule blind
  precision-lever boots.
- `i8_ring` correction: it is the lossless INT8 wire mode, not the historical
  E4M3/FP8 transport.
- Frozen ticket-number span derived exactly from the pinned token IDs:
  `[137499,137502)` on all three 350k prompts.
- Needle-inclusion trace implemented after global DCP merge. It records exact
  span hits, +/-32-token context hits, and nearest selected indices/scores for
  every active reference-indexer layer and decode call.
- Trace source commit `103473cdbb6bb0abcc0cd034822206d0dd4caeba`;
  image `sha256:739ff8d3eaaf55e6e5ce0d22b2ad9ce210c42a2837af8c76b4adc8bea847e23d`;
  CPU gate PASS. CN4 trace boot in progress.

PENDING:
- frozen r1 needle-inclusion result;
- selected -> post-selection index/gather/attention trace;
- absent -> layer>=1 reference-boundary replay and shared input/metadata
  trace;
- onset map only after inclusion chooses the causal half.

## 2026-07-27 — engagement 9: needle absent until final sparse layers

- Frozen r1 completed cold: 343,727 prompt tokens, cached=0, stop, 16 output
  tokens, deterministic wrong answer `27`.
- Trace matrix is complete and fail-closed: 336/336 records, 21 active layers
  x 16 decode calls, no duplicates.
- Exact ticket-value range `[137499,137502)` is absent from every selection
  through layer 38.
- Sparse exact hits begin at layer 42. All three tokens first occur together
  only once at layer 62. Layer 74 sees at least one value token in 15/16 calls
  and all three in 7/16 calls.
- This matches the failing exact-selector morphology and rejects a
  post-selection page-table/gather drop as the primary cause. The candidates
  are not available to attention early enough.
- Analyzer SHA-256:
  `2278cb0fd6e0ad87c9c16c6f77da1187c78df23984043c4debd8cd8a04b33751`.
- Installed Transformers source was re-read during the run. The reference
  mode matches its indexer projections, K normalization, interleaved RoPE,
  FP32 Q.K, ReLU, learned head weighting, causality, and exact top-k contract.

NEXT:
- Capture a deeper reference layer boundary and independently replay from
  `hidden_states`/`q_resid` through exact top-k.
- If that replay agrees, treat the remaining problem as shared model
  trajectory versus training-time selector contract; do not invent another
  selector heuristic.
- Use onset mapping only after the deeper boundary closes whether the
  transition is sharp or gradual.

## 2026-07-28 — engagement 18: PR polish, GHCR publish, KLD verdict, arm D, topo tool

- PRs rebased onto correct bases after Derek's diff-hygiene catch (sparkinfer#86:
  13 files on master; vllm#189: 2 files on dev/gilded-gnosis; touched files
  bit-identical across bases, noted in bodies). Bodies rewritten to formal
  technical-report register per Derek.
- GHCR publish complete: ghcr.io/yatesdr/glm52-serve@sha256:db82fdcb… (digest =
  tested image ID; ssh fix: CN4 password-only auth — stray keyfile caused
  publickey stalls). Digest written into both PR bodies.
- KLD n=3 FINAL: static 0.146228±0.004688 vs dynamic 0.139036±0.002010 —
  complete rank separation (p=0.05), −4.92%; in PRs with exact wording.
  Remaining before undraft: matched warmed decode (first n=3 sample missed 1%
  budget at −5.27% BUT confounded by −11.8% MTP acceptance; MTP0 isolation +
  n≥10 protocol agreed; draft-layer-78 hypothesis; interim number NOT in PRs),
  restart/repeatability.
- destroyed's scales file: bit-identical to #145 (0/78); his 0.132 = his
  quality-first MXFP8 membership config (protects q_a/kv_a/gates/eh_proj/
  lm_head), confirmed by him. Arm E cancelled; arm D (his config + dynamic)
  first sample 0.1331905 (−4.2% vs B) — n=3 + arm C running.
- kcramp: DCP_TOPK_OWNER_MERGE=0 fixes a prefill regression on base v20 —
  unverified claim, real results; our posture already pins 0; queued for
  independent repro + mechanism (topology-dependent vs upstream's TP8/DCP8
  +10%; owner-merge is static-eligibility, unprobed — same gap class as
  P2P level).
- Best-known compose cleaned (device_ids enumerated, subnet + dead shm_size
  removed, entrypoint verified NECESSARY — image default runs run-kimi26-vllm)
  → compose/glm52-v20-nvfp4-dynamic-scaled-kv-20260728.yaml (+ Derek's
  Downloads copy). Public reproduction Dockerfile:
  docker/Dockerfile.nvfp4-dynamic-scale-review-public.
- NCCL_P2P_LEVEL auto-derivation tool written + validated:
  docker/derive_nccl_p2p_level.py — ANSI-safe topo parser, PXB-capped
  derivation (cap discovered from live data: naive worst-class would output
  SYS = permissive over broken host-bridge paths; validated PXB reproduced
  on BOTH cn4 (NODE worst) and cn3 (PHB worst, read-only test)), 7/7
  fixtures, --permissive/--devices. To be baked with destroyed's MXFP8
  defaults in next image iteration, gated on arm C/D results.

PENDING: MTP0 decode isolation + n≥10; restart/repeatability; arm C + D
n=3 (+acceptance); owner-merge repro; next image bake; undraft decision.

## 2026-07-28 — engagement 19: measured comm calibration (auto contract)

Derek requirement evolution: static topo derivation → measured selection
("best mode across tons of systems needs measurement, not classification")
→ uniform `auto` opt-in contract for BOTH topology and wire protocol.

Built + validated:
- docker/derive_nccl_p2p_level.py: static stage — ANSI-safe parser,
  PXB-capped policy (cap semantics discovered from live data: naive
  worst-class would output SYS = P2P over wedge paths), --permissive,
  7/7 fixtures, live-correct on CN4 (NODE-worst→PXB) and CN3 (PHB-worst→
  PXB, read-only), both matching hand-validated prod values.
- docker/nccl_p2p_probe.py: measured stage — per-candidate killable
  subprocess trials, NCCL collectives at DCP sizes, bit-verification
  every iteration, 60s/trial + 180s total budgets, conservative tiebreak,
  fingerprint cache, three-deep fallback. Explicit-respect guard enforced
  IN the tools (verified on CN4: explicit/unset/auto all correct). Live
  trial run PENDING idle CN4 (not run under Sol's experiments).
- design/measured-comm-calibration-spec-20260728.md: consolidated spec —
  auto contract, wire-protocol tokens (auto=lossless race; i8_auto/mx_auto
  = intra-codec transport race; codec choice never automatic), owner-merge
  measurement (kcramp), hardware validation gates, fixture-corpus ask to
  Discord, bake plan (with destroyed's MXFP8 defaults, gated on arms C/D).

PENDING (calibration track): live probe on idle CN4; #81 wire extension;
owner-merge A/B; community topo fixtures; next image bake.
PENDING (promotion track, unchanged): MTP0 decode isolation + n≥10;
restart/repeatability; arms C/D n=3; undraft decision.

## 2026-07-28 — engagement 20: arm D complete; CN3 live probe

- Arm D n=3 FINAL: {0.1331905, 0.1354213, 0.1311731} mean 0.1332616 ±
  0.0021250. COMPLETE RANK SEPARATION vs arm B (D max 0.13542 < B min
  0.13673): destroyed's quality-first MXFP8 + dynamic scaling beats
  dynamic-alone by −4.16% (p=0.05 exact), −8.87% vs static #145. Levers
  stack as predicted. Awaiting: arm C (factorial decomposition), MTP
  acceptance from D runs (draft-fidelity hypothesis).
- CN3 live probe (Derek-authorized): pre-flight found ~850 MiB free/GPU
  (prod pool resident regardless of user traffic) ⇒ full 64MB probe would
  OOM. Added --sizes-mb; running slim 4MB probe in throwaway container
  (serving container untouched), health bookends before/after. Caveat
  recorded: 4MB ranks latency-dominated behavior; full-size measurement
  still owed on idle CN4.

## 2026-07-28 — engagement 21: preview image baked + pushed (dynscale+autocal+mxq)

- CN3 live probe (Derek-authorized): explicit-guard proved itself in the
  wild first (v19 image BAKES NCCL_P2P_LEVEL=SYS — probe refused to
  override; noteworthy prod-image fact). With auto: both trials FAILED —
  root-caused to plain CUDA OOM (~850MiB free beside resident prod pool;
  a torch context alone doesn't fit). Machinery validated: fail-closed to
  static fallback, prod healthy throughout, memory untouched. Deployment
  context differs: launcher hook runs PRE-model-load on empty GPUs.
- BAKE (Derek directive): new preview image FROM the gate-tested digest
  (python content provably unchanged) + calibration pair at /usr/local/bin
  + serve-with-autocal.sh entrypoint (auto hook: probe→static→defaults,
  explicit wins) + destroyed's quality-first MXFP8 membership defaults
  (ONLINE_QUANT=custom + QUANTIZATION_CONFIG_JSON; arm-D evidence
  0.1332616±0.0021250 n=3 fully rank-separated vs dynamic-alone; arm C +
  retrieval smoke pending ⇒ labeled REVIEW PREVIEW). In-build gates:
  sha-checks, py_compile, topo self-test, bash -n. In-container verify:
  envs present, explicit guard + static derivation correct on CN4.
  PUSHED: ghcr.io/yatesdr/glm52-serve@sha256:9bc5fcc3b175766ed342c682
  064522f33bf5cad556d17def1bdf9c02903d775b (tag ...-dynscale-autocal-mxq-
  preview-20260728).
- Calibration tools published publicly: yatesdr/b12x branch
  calibration-tools @ ffdabd2d (3 files, sha-pinned). Public Dockerfile
  updated with v2 section (ADD from that branch + sha gate + envs +
  entrypoint). Honest scope note everywhere: wire-protocol auto tokens
  documented only; land with SparkInfer #81 extension.

PENDING: arm C + MTP acceptance from D; MTP0 decode + n≥10; restart/
repeatability; full-size probe run on idle CN4; #81 wire extension;
owner-merge A/B; undraft decision; preview→review promotion after Sol's
gates on the new defaults.

## 2026-07-28 — engagement 22: release-config rebuild, probe hardening, STAND-DOWN

Derek escalated scope: shareable config must be the FULL prod flag set with
our auto modes replacing hand-tuned comm knobs (his prod-ring/PROMOTION
yamls = pre-auto era with hard-coded SYS + i8_ring). Delivered:
- serve-with-autocal.sh v5: wraps direct `vllm serve "$@"`, posture from
  real args, wire winner exported as SPARKINFER/VLLM/B12X_PCIE_DMA_FP8,
  dynamic-KV default-on in posture, shm cleanup, explicit-wins everywhere.
- compose/glm52-v20-release-autocal-20260728.yaml: PROMOTION-grade full
  flag set + auto deltas + destroyed quality-first quant (capacity
  alternate commented); digest-pinned to release image 4ea6bcf6…; in
  ~/Downloads.
- Probe hardening from live CN3 debugging: replaced torchrun with direct
  rank spawn (torchrun swallowed tracebacks); fixed empty
  CUDA_VISIBLE_DEVICES export that hid all GPUs; NCCL probe then achieved
  first fully-verified measured selection (PXB 4.77 > PHB 4.71 GB/s).
  Wire race reached the real DMA path; remaining known one-liner: BF16
  hash via numpy (fix text ready, NOT applied — interrupted).
- CN3: prod stopped (authorized); GPUs idle; assorted probe scratch in /tmp.

DEREK STAND-DOWN ORDER: execution handed to Sol; Fable reviews when done.
Full pickup state written to fable-autocal-standdown-handoff-20260728.md
(fix text, artifact inventory, CN3 state, remaining 6-step sequence,
watch-outs). No further machine contact after the order.

PENDING (review duty): Sol's wire fix + rebake + CN3 smoke results; then
the promotion-track items unchanged (decode confirmation, restart/repeat,
arm C, PR undraft).
