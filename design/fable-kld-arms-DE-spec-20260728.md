# Spec: KLD arm D — quality-first weight quant × dynamic KV scaling

**2026-07-28 update (destroyed, Discord): arm E is CANCELLED.** He confirmed
no new scale values exist — the circulated file was a metadata-only rename
of the #145 artifact (matching our 0/78 value diff), and his 0.132/0.126
results are attributable to #145 scales plus his online-MXFP8 membership
fix (protecting q_a/kv_a projections, MLP gates, eh_proj, lm_head; shared
experts BF16 in the quality profile). Arm D is now the sole required run;
arm C remains optional. Prerequisite 1 below is void; prerequisite 2
stands.

Date: 2026-07-28
Author: Fable (per Derek)
Executor: Sol — **after the current regiment completes** (matched warmed
decode, restart/cache-reuse qualification, GHCR digest recording). Nothing
here touches the dynamic-scale PR track; this is deployment-configuration
characterization layered on top of it.

## Objective

Complete the weight-quant × KV-scaling matrix on our rig, our harness, our
reference. Arms A and B are done; this spec adds D and E (C optional).

| Arm | Weight quantization | KV scaling | Status |
|---|---|---|---|
| A | current online config | static #145 scales | done: 0.146228 ± 0.004688 |
| B | current online config | dynamic per-token | done: 0.139036 ± 0.002010 |
| D | destroyed's quality-first MXFP8 | dynamic per-token | **this spec** |
| E | destroyed's quality-first MXFP8 | destroyed's NEW static scales | **this spec** |
| C | destroyed's quality-first MXFP8 | static #145 scales | optional, only if idle time |

Questions answered: D−B = weight-config effect with KV scaling held at
dynamic. D−E = dynamic vs his new static calibration under identical
weights (the head-to-head Derek wants). If C runs: D−C tests whether
dynamic's measured advantage (B−A) survives under the protected weight
config.

Context: destroyed reports 0.132 on his rig with his new scales + his
quality-first config. That number bundles three variables (scales, weight
config, rig/harness) and ranks nothing until decomposed here.

## Prerequisites (blockers — do not start without)

1. **Destroyed's NEW scales file.** The file Derek received
   (`fcabad6e…`) is value-identical to #145 (0/78 layers differ; verified) —
   it is NOT the 0.132 artifact. Derek/D-Rock must obtain the actual new
   file from destroyed. On receipt: record SHA-256; validate schema
   (`nvfp4_ds_mla_outer_scale_v1`, 78 layers, denominator 2688); assert the
   values actually differ from #145's (`efd7e23a…`) — if 0/78 differ again,
   stop and report; there is nothing to test.
2. **His exact quality-first config**, verbatim from Discord (the KLD
   winner, NOT the shared-experts capacity variant):

   ```
   ONLINE_QUANT: custom
   QUANTIZATION_CONFIG_JSON: >-
     {"linear":{"weight":"mxfp8"},"ignore":["re:.*\\.fused_qkv_a_proj$","re:.*\\.q_a_proj$","re:.*kv_a_proj_with_mqa","re:.*\\.mlp\\.gate$","model.layers.78.eh_proj","lm_head"]}
   ```

   Confirm the dynamic-capable image (`db82fdcb…`) honors
   `ONLINE_QUANT=custom` + this JSON at boot (log-verified layer list); if
   the env contract differs in this image lineage, stop and report before
   improvising.

## Protocol

Identical to the completed A/B matrix in every respect except the two
variables under test. Same image (`db82fdcb…` — no rebuild; both KV modes
and online-quant are runtime-selected), same pinned 2,048-token window
(2,047 scored positions), same BF16 reference logits (reference is
unquantized — it does not change when the candidate's weight config
changes; assert the pinned reference hash anyway), same runner
(`ac8e57f6…` contract), TP4/DCP1, eager, max_len 4096, fresh cache
namespace per run, n=3 per arm.

Arm D env delta vs completed arm B: add the quality-first ONLINE_QUANT
pair. Keep `VLLM_NVFP4_MLA_DYNAMIC_SCALE=1`, no scales file.

Arm E env delta vs D: `VLLM_NVFP4_MLA_DYNAMIC_SCALE=0`,
`VLLM_NVFP4_MLA_SCALES_FILE=<destroyed's new file>`.

Per-run validity (fail-closed, as before): full 2,047-position output;
writer compile metadata (`per_token_scale=true` for D, `=false` + scales
file consumption proof for E); pinned input IDs; clean shutdown.

Also record per arm (cheap, one boot each, no extra runs):
- KV pool token count at the standard 480k production posture — the
  protected config keeps more weights unquantized and WILL cost capacity;
  quantify it (this is why destroyed's shared-experts variant exists; that
  variant is out of scope here, note it as arm F if Derek wants the
  capacity point later).
- One frozen 350k r1 retrieval row under arm D's posture (production
  config, cold) — a smoke assertion that the weight config does not
  regress deep retrieval while improving KLD. Not a full gate.
- MTP acceptance rate during any arm-D production-posture decode sampling:
  his config protects the draft layer's eh_proj, so arm D doubles as a test
  of whether improved draft fidelity recovers the acceptance delta observed
  in the dynamic-vs-static matched decode (n=3, unconfirmed). Record it;
  do not tune anything based on it here.

## Analysis (pre-committed)

- Report per-run values, mean, sample SD for D and E; retain per-position
  arrays for paired analysis.
- D vs E: rank test at n=3 (complete separation ⇒ p = 0.05 exact; any
  interleaving ⇒ report as indistinguishable at run level) plus paired
  per-position block bootstrap (blocks ~64–128 positions) since inputs are
  pinned-identical — this is the statistically decisive comparison.
- D vs B: same treatment — quantifies the weight-config gain by itself.
- No pass/fail bar: this is characterization. The deliverable is the
  completed matrix table with the same statistical wording standards used
  in the PRs (no cross-rig numbers, no claims beyond the tests run).

## Evidence

Archive under `harness/cn4-evidence-archive/<date>/kld-arms-DE-v1/` with
the established pattern (raw logs + SHA-256 per run, summary JSON, compile
proofs, env dumps). Post results to comms with the table; Fable folds the
completed matrix into the packaging record and, if D or E wins on quality,
into a deployment-posture recommendation — separate from, and additive to,
the dynamic-scale PRs.

## Explicit non-goals

- No changes to the dynamic-scale implementation, PRs, or image.
- No adoption of the shared-experts capacity variant in this pass.
- No cross-rig comparison against destroyed's 0.132 — his number is
  motivation, not a baseline.
