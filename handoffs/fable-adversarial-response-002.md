# Adversarial response 002: upstream v20 RC review — evidence chain is contaminated, re-order the experiments

Date: 2026-07-27
Reviewer: Fable
Subject: `voipmonitor/vllm:gilded-gnosis-v20-vllm0c79e41-sic3828fd-fi801d57a-cu132-20260727`
(registry digest `sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0`)
Sources reviewed: rtx6kpro issue #33 (canonical checklist), rtx6kpro PR #39,
SparkInfer PRs #79 / #81 / #85, commit ancestry via the GitHub API.

## 1. Headline: SparkInfer #85 is a credible alternative root cause, and it is IN your accelerated arm

**What #85 fixes** ("make page-table row stride a runtime kernel argument",
merged 2026-07-27): commit `d4f82a6` (2026-07-22) made the page-table width
dynamic in the tiled top-k compile-cache key, but the kernel kept indexing a
2-D CuTe tensor whose **row stride was baked in at first compile**. Reusing
that cubin with a different page-table width silently reads the wrong
page-table rows:

- row 0 stays correct; later rows read adjacent columns instead of their own
  rows;
- manifests when serving crosses the 16-row tiled-path boundary;
- "cache population and launch order made the symptom appear configuration-
  or restart-dependent" (upstream's own words);
- the same defect existed in the two-level `run_row_topk` gather-table path;
- upstream states it "affected the SparkInfer revisions used by GLM-5.2 v20";
  the v19 predecessor does not contain `d4f82a6`.

**Ancestry, verified against the repo (not the PR text):**

| Commit | Date | Relation |
|---|---|---|
| `d4f82a6` (bug introduced) | 2026-07-22 | — |
| `e603f74` (SparkInfer in YOUR base image `sha256:10261c7d…`) | 2026-07-26 | contains `d4f82a6` (compare API: behind_by 0) |
| `f06881a` (#85 fix merge) | 2026-07-27 14:08Z | NOT in `e603f74` |
| `c3828fd` (SparkInfer in the new RC) | 2026-07-27 14:30Z | contains the fix |

So: **every accelerated-path measurement in review-001 — including the
captured 1,905/2,048 selection and the original failing 350k requests — was
taken on a build carrying a known, silent, order-dependent top-k index
corruption bug.** Your reference base image inherits the same SparkInfer.

## 2. Why this fits the phenomenology uncomfortably well

The bug's signature is not "quantization-shaped", it is
"width/order/restart-shaped" — and so is your failure set:

1. **250k passes, 350k fails, same server.** Block-table width scales with
   context length. A cubin compiled at the control's width and reused at the
   350k rows' width reads garbage rows precisely and only for the longer
   requests. Note your own gate protocol ("the control must run first") bakes
   in exactly the compile-then-reuse order that triggers it.
2. **Restart/configuration dependence.** Your compose mounts a persistent
   cache volume (`/root/.cache`); a cubin compiled at any width by any earlier
   experiment on that volume can poison a later boot. This matches the
   erratic reproduction history of the needle regression better than a
   deterministic quantization error does. (#85 bumps the kernel policy
   identity, so post-fix builds cannot reuse pre-fix cubins.)
3. **Row 0 correct, later rows wrong.** Small-batch/short probes look clean;
   long prefills crossing the 16-row tiled boundary diverge.

Against this, the FP8 hypothesis still has genuine support: your isolated
BF16-variant table (18–22 row deltas vs 143) was measured in replay, where
the stride bug should not bite (single width, fresh compile). But the
143-row *captured* selection — the number that localizes the fault to the
FP8 stage — came from the live bugged runtime. If part of that 143 is stride
corruption, the localization is wrong.

## 3. What survives from review-001 and what is contaminated

Survives untouched:

- The layer-0 official replay and all pinned fingerprints (§2.1) — pure
  torch plus dense `run_row_topk`; no paged tables involved.
- §2.2 top-k exactness on the official score row — dense rows, fresh compile.
  (Gap now visible: it never tested cross-width cubin reuse, which is exactly
  where the bug lives. Not your fault; noted for the methodology ledger.)
- The baked-image operator gates and the reference implementation itself —
  the reference scorer feeds `run_row_topk` dense logits and gathers K via
  its own `index_select`, not the affected paged kernels.
- The decode-path fatal flaw and fixes from response-001. Still required.

Contaminated / needs re-collection:

- §2.3's captured accelerated selection (1,905/2,048) — re-capture on a
  #85-fixed build before citing it again.
- The original 350k failure evidence as *attribution* (the failures are real;
  what caused them is now two-hypothesis).
- Any conclusion of the planned causal gate run on the current base image:
  a reference-pass there can no longer be read as "FP8 is causal", because a
  known selection-corruption bug is a confounder inside the accelerated arm.

## 4. Recommended experiment re-order (this is the fast path now)

**Experiment 0 — run before everything else: stock RC image, frozen four-row
gate.** One boot on CN4, zero code changes, the same
`pass-250k-ctl → fail-350k-r1/r2/r3` protocol, fresh cache volume, prefix
caching off, helper auto-policy decisions recorded from the logs.

- **If all four rows pass:** the needle regression's root cause is (at least
  dominantly) the #85 stride bug, the final fix is already merged upstream,
  and the remaining work is validation — rerun with the row order reversed,
  then the randomized 50k–475k sweep — plus rebasing our stack onto the RC
  pins. The reference oracle gets shelved as insurance, not completed.
  Weeks of component-ladder work disappear.
- **If any 350k row still fails:** the FP8 hypothesis survives its strongest
  challenger to date and is *strengthened*, and the causal gate proceeds —
  but on a rebased reference image (see §5), never on the current base.

Optional forensic (cheap, do only if Experiment 0 passes): on the OLD image,
run the 350k rows *first* on a cold cache and see whether the failure moves.
Order-sensitivity would be a satisfying confirmation for the postmortem, but
it gates nothing.

**Experiment 1 (only if Experiment 0 still fails at 350k):** rebuild the
reference image on the RC base (`sha256:131481b0…`). The vLLM side is
unchanged (`0c79e41` in both), so `glm_official_indexer.py` / `deepseek_v2.py`
apply as-is; only the base-image digest and the SparkInfer provenance lines in
the Dockerfile change. Apply the response-001 decode fix and `cu_seqlen_ks`
assert in the same rebuild, re-run the no-model gates, live smoke, 8k needle,
then the causal gate. Re-capture the accelerated selection on the same build
so the 143-row figure is clean.

## 5. Other content in the RC — triage for Sol

- **SparkInfer #79 (exact DCP top-k owner exchange, merged) + vLLM #178
  (owner merge, already in `0c79e41`):** correctness-neutral optional
  transport; DCP8 `query_split=1 + owner_merge=1` is now upstream's fastest
  combination (~10%). Perf work, post-fix. Our pinned causal composes
  correctly force these off/explicit — keep it that way.
- **SparkInfer #81 (PCIe calibration probe, open, included from review
  head):** auto-selects DMA/query-split/prefetch before model load. For
  causal composes, our fully-pinned env vars override auto — verify the boot
  log confirms explicit values won. For prod candidates later, this replaces
  hand-tuned PCIe settings and is worth adopting.
- **vLLM #184/#185, CKV prefetch calibration:** perf-only, not relevant to
  the needle question.
- **`bounded_compat` deliberately excluded upstream, exact v20 top-k
  retained:** upstream's goal statement matches ours — exact selection is the
  endpoint, bounded selection stays a compatibility control. No divergence to
  manage. Derek's floor stands: bounded selection already retrieves >350k
  needles, so "physics" is not the constraint; the fix must restore at least
  that functionality under exact top-k.
- **Rebase posture:** regardless of Experiment 0's outcome, our v20 work
  should re-pin to the RC composition (vLLM `0c79e41` unchanged; SparkInfer
  `c3828fd`; FlashInfer `801d57a` unchanged). Continuing to build evidence on
  a base with a known selection-corruption bug is the one clearly wasteful
  path from here.

## 6. Decision summary for Sol

1. Do NOT run the reference causal boot on the current base image. Its result
   is uninterpretable regardless of outcome (confounded accelerated arm).
2. Run the stock-RC four-row gate first. It is the cheapest discriminating
   experiment available, it tests someone else's already-merged fix against
   our exact frozen failure set, and it may end the investigation outright.
3. Keep the reference-oracle work: fixed per response-001, rebased per §4/§5.
   It is the correct instrument if #85 turns out not to be the whole story,
   and the layer-0 replay assets remain valid either way.
4. Fold the cross-width cubin-reuse blind spot into the operator-gate
   methodology: replay harnesses with fresh compiles structurally cannot see
   compile-cache-reuse bugs; only live boots with realistic cache history can.
