# Redirect 001: own the NVFP4 scale calibration → promotion

Date: 2026-07-27
From: Fable (at Derek's direction)
To: Sol
Status of record: root cause confirmed — #145 calibrated NVFP4 MLA outer
scales are sufficient on the clean-RC stock scorer for frozen r1. Endpoint:
keep exact top-k + stock accelerated indexer + 368B record; canonicalize the
scales mechanism. This document is the execution plan from here to promotion.

## P0 — Finish what's running (not a redirect; it's sanctioned)

The in-flight breadth run (250k control + r2/r3 + today's r1 on the
qualified boot) is the right use of the current process. Let it finish.

- Record per-row verdicts AND answer morphology (wrong-retrieval vs
  fabrication), especially r2.
- Your own caveat stands and is accepted: same-process rows are breadth
  evidence, not promotion evidence; a fresh control-first run happens in P4.
- **Branch gate:** if any 350k row misses here, STOP this plan and report —
  that's the partial-cause branch and we reassess before any calibration
  work. If all four pass, proceed.

## P1 — Capture OUR calibration statistics (one instrumented run)

Goal: per-layer `max_abs(kv_c)` measured on OUR checkpoint at OUR contexts.

- Instrument the writer path (or a capture hook) to record per-layer
  running `max_abs(kv_c_normed)` during prefill+decode.
- Prompt mix: frozen 343k r1, the 250k control, one ~50k row, one short row.
  Long context is the point — #145's capture was 2048-ctx wikitext.
- Output artifact: `kv-scales/glm52-<our-checkpoint-hash>-max-abs-v1.json` —
  per-layer max_abs, prompt-mix manifest, checkpoint SHA, image digest.
- If the capture hook can ride along on an existing planned boot (e.g., the
  P3 boot below), combine — this phase needs data, not its own boot.

## P2 — Offline audit + sensitivity sweep (ZERO boots)

All computation against P1 stats and the frozen activation captures.

1. **Envelope audit:** compare our per-layer max_abs against the audit
   values retained inside #145's JSON. Flag any layer where ours exceeds
   their envelope (clip risk) or sits far below (subnormal margin loss).
2. **Headroom sweep:** for factors ~0.5×–2× around the formula
   `s_l = max_abs/(6·448)`, compute per layer: block-scale subnormal rate,
   clip rate, round-trip error vs BF16. Deliverable: plateau width per
   layer.
3. **Optional (cheap, high-value):** push 2–3 candidate scale files through
   the existing layer-34 rank replay to show indexer-margin insensitivity
   inside the plateau — direct evidence the knob is positioned, not tuned.

**Decision rule (pre-committed):**
- Our max_abs within #145's envelope with healthy margin AND plateaus wide
  on all layers ⇒ **ADOPT** their file as-is, validated. Done with P2.
- Any layer drifts or has a narrow plateau ⇒ **REGENERATE** the file with
  the same formula from our P1 stats (per-layer overrides only where
  measured; everything else keeps the formula). No new math either way.

**Explicit non-goals:** no live boots sweeping scales against retrieval or
KLD outcomes. The mechanism is a plateau; hunting a peak inside it overfits
three frozen prompts and destroys upstream reviewability.

## P3 — One instrumented confirmation boot (combine three needs)

One boot on the chosen scales file, carrying:

1. **Saturation counters:** per-layer subnormal/clip counts on frozen r1.
   Pass = zero subnormal, ~zero clip. This is the calibration's proof.
2. **Healthy layer-entry trace:** the needle-entry trace on the WINNING
   posture — the missing healthy control for the causal story, and the
   source of the margin criterion.
3. **Margin criterion pre-commitment:** from that trace, fix the numbers
   (ticket entry layer ≤ L*, presence ≥ N/16 answer calls) that become the
   quantitative regression gate in P4 and in the upstream PR.

## P4 — Promotion suite (unchanged from review-003 §8 / response-004)

1. Fresh-boot control-first frozen four-row suite (the promotion-grade run).
2. Randomized cold 50k–475k ladder, scored on final content AND entry-layer
   margins — margins are what decide whether the scales alone are the whole
   story or whether a residual term (e.g., FP8 RoPE field) ever earns a
   conversation. No margin thinness ⇒ no further mechanisms, period.
3. KLD suite; quantified perf gate (state the % bound vs RC serialized
   gates before running); 500k KV @ 480k capacity check on the NVFP4
   posture; graph+eager consistency; fresh/warm cache, restart, and
   compile-cache repeatability.

## P5 — Upstream package

- Support canonicalizing #145 (currently "included for testing, not
  requested for canonical merge" in rtx6kpro #33) with: our frozen-gate
  evidence, the healthy trace, ladder results, and our provenance-complete
  scales artifact (P2 outcome — theirs-validated or ours-regenerated).
- #182 report: causal narrative (stride bug real but independent; scorer
  exonerated by the oracle; writer scales causal), with the corrected
  i8_ring characterization (rank-consistent amax/254 codec, not "lossless
  byte transport") per `design/int8-dma-ag-design.md`.
- Credits: MadeBy561 (#145 finding + fix), voipmonitor (#85), ours = causal
  isolation + long-context validation + calibration provenance.

## Budget and guardrails

- Boots: P0 uses the current process; P1 piggybacks P3 where possible;
  P3 is one boot; P4 is the promotion suite it was always going to be.
- No invented record formats, no selector policies, no live scale sweeps.
- Anything that looks like a new mechanism requires margin-thinness
  evidence from P4.2 first.
