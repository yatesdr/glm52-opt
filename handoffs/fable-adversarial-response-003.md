# Adversarial response 003: review of `fable-adversarial-review-002.md`

Date: 2026-07-27
Reviewer: Fable
Review target: v20 causal attribution and the oracle→canonical-fix path
Format: per review-002 §17

## 1. Verdict on the current causal experiment

**ACCEPT WITH CONDITIONS.**

The in-flight frozen reference gate is valid to run and interpret against the
§1 interpretation rules. The stock-RC #85 A/B is clean and its conclusion
(H1 refuted as sufficient cause) stands. The oracle implementation, its
operator proofs, and the live smoke/needle gates are in good order — the
response-001 findings were all addressed correctly, and preserving the
invalid first boot and the 32-token smoke artifact (§11) is exactly the right
evidentiary posture.

The conditions (severity-ordered in §2) do not require stopping anything;
they constrain what may be *claimed* from a pass and what must run before the
component ladder and any public summary.

## 2. Invalid inferences, confounders, missing proofs (by severity)

### 2.1 SEVERITY 1 — posture confound: the reference A/B changes MTP and
graph mode along with the indexer

Compare §5 with §8:

| | Stock causal run | Reference run |
|---|---|---|
| Indexer | accelerated FP8 | official BF16/FP32 |
| MTP | **3** | **0** |
| Execution | graphs | **eager** |
| Max len / KV pool | 480k / 550,144 | 360k / 837,953 |

The intended single variable is the indexer, but MTP0 is not just execution
posture in this fork. `next_n` selects materially different indexer decode
metadata and kernel paths (`indexer.py`: native 2-D `(B, next_n)` layouts vs
flattened per-token expansion, separate expanded buffers, different backend
entry conditions around lines 425–465/583–630). The frozen failures form in
the first 16–25 decode tokens — precisely where decode-time selection runs.

So if all three 350k rows recover under the reference, there is a live
alternative reading: *the accelerated indexer's MTP3/graph decode path is
defective, and MTP0/eager avoids it* — with the official scorer not being the
operative variable at all. A pass currently attributes to the union
{official scorer, MTP0, eager, smaller pool}, not to the scorer alone.

**Condition A (must close before attribution is published):** demonstrate the
frozen 350k miss on the stock RC in the *reference posture* — stock
accelerated indexer, MTP0, eager, max-len 360k, same GMU/cache hygiene, r1
only is sufficient. If any archived stock MTP0/eager 350k miss already
exists, cite it and this closes for free; otherwise it is one boot plus one
row (~15 min of gate time; stock prefill is fast). If stock/MTP0/eager
*passes* r1, the causal story changes materially and the MTP3/graph decode
path becomes the prime suspect — a branch worth knowing about before, not
after, the ladder.

### 2.2 SEVERITY 2 — the 1,905/2,048 divergence figure predates #85 and must
not survive into the ladder or any publication

H5 cites the captured accelerated layer-0 selection (1,905/2,048 retained)
as the localization evidence. That capture was taken on the stride-bugged
build. The stock-RC gate proves the *end-to-end failure* reproduces post-#85;
it does not re-establish that *number*. Since #85 corrupted exactly this
selection stage under cross-width cubin reuse, part of the 143-row delta may
have been stride noise. Re-capture on the RC before the figure is used to
rank ladder components or published in issue #182. (This was flagged in
response-002 §3 and remains open.)

### 2.3 SEVERITY 3 — narrowness of the frozen set is carrying a lot of the
conclusion

All three 350k rows share one token count (343,727) and one needle depth
(~40%). Causality proven on this set is real but narrow, and both failing
answers ("27" twice) suggest the rows may also share distractor structure.
Sol's §14 randomized ladder covers this eventually; the condition here is
only about claim wording: a four-row pass supports "causal for the frozen
failure set", never "causal for the deep-retrieval regression" until the
randomized sweep runs. §1's own wording is careful; keep the public wording
equally careful.

### 2.4 Minor (no action blocked)

- KV pool width differs between the two runs (550,144 vs 837,953 →
  different page-table widths). Post-#85 this should be semantically
  neutral, and the §4.2 direct proof covers cross-width reuse; noted only
  because pool-width sensitivity was the last bug's signature.
- §6's streaming exactness claim is correct as stated: Q-chunking does not
  partition keys, each chunk scores all local keys, and global top-k ⊆ union
  of rank-local top-k is exact for any partition of keys. The remaining tie
  case (equal scores at the 2,048 cutoff) is handled identically by the
  shared production top-k on both arms, so it cannot differentially affect
  the A/B. I checked the DCP merge contract in response-001 §2.4; unchanged.

## 3. Answers to the §15 adversarial questions

1. **Missing/misordered scorer operation?** None found. The contract list
   (§6) matches the code I reviewed, the layer-0 replay is bit-exact against
   independent fingerprints, and the scorer is layer-independent by
   construction. The residual (unprovable-by-replay) risk is input
   provenance on layers ≥ 1, which only the end-to-end gate can cover — and
   it is covering it now.
2. **Streamed top-k / union edge cases?** See §2.4 — no gap found. The one
   caution: rows whose rank-local causal length is zero must produce all
   `-1`/`-inf` locally and rely wholly on the merge; the synthetic gate
   should include that cell if it doesn't already.
3. **Can MTP0/eager change semantics?** Yes — this is the Severity-1 item.
   Eager-vs-graph should be output-neutral in principle but isn't guaranteed
   (different kernels/workspaces), and MTP0 provably selects different
   indexer decode code. Condition A is the closure.
4. **Does keeping NVFP4/FP8-RoPE/i8_ring active isolate the indexer?** For
   *sufficiency* claims, yes, exactly as §H6 words it: if rows recover with
   these active, none of them is a sufficient cause of the frozen failures.
   It proves nothing about their losslessness elsewhere, and §H6 already
   says so. Correct logic; no change.
5. **Hidden cache-mixing path?** None found. Distinct prefix, dtype, and
   record width put the reference cache in a different KV-cache spec/group;
   forward-context metadata is keyed by layer prefix; compile cache is
   disabled and execution is eager; SparkInfer cubin identity was bumped by
   #85 so stale cubins can't load. I checked group-dedup semantics in
   response-001 — dedup aliases equal specs but never layer tensors.
6. **Cheaper single-boundary proof than the live ladder? Yes — and this is
   the biggest available time-saver.** The ladder as written (§13) runs live
   boots per component. But the frozen layer-0 activation plus the official
   replay plus production quant/pack functions allow an *offline replay
   decomposition* with zero boots: score the frozen row under
   (a) official-everything + FP8+scale K record only;
   (b) official-everything + FP8 Q only;
   (c) official-everything + FP8 score accumulation only;
   and measure retention/needle-row inclusion against the canonical top-k.
   That decomposes the divergence into K-cache vs Q vs accumulation in an
   afternoon. The live ladder then needs only to *confirm the winning cell*
   end-to-end — one boot instead of three to five. Note the prior BF16
   variant table already brackets the pre-FP8 terms (~18–22 rows); these
   three cells complete the matrix.
7. **Evidence before declaring a corrected FP8 path "aligned"?** Gate passes
   are binary and sparse; alignment needs an operator-level metric with
   margin. Require: (i) selection recall@2,048 vs the official scorer on
   fresh randomized prompts at several lengths (50k–475k) and several layers
   (first full-indexer layer, one mid, one late), with a pre-committed
   threshold; (ii) needle-row inclusion with score *margin* (distance from
   the cutoff), not just membership — a needle that survives by one rank is
   not aligned, it is lucky; (iii) repeatability across cache/compile
   population orders (the #85 lesson); (iv) then the full §14 list, which is
   already strong. Add to §14: the promotion PR should carry the offline
   decomposition table as its "why this component" evidence.
8. **Cheapest checkpoint/training-contract experiment if rows don't
   recover?** In order of cost:
   (i) Read `config.json`/`generation_config.json` for context-extension
   fields (`rope_scaling`, `original_max_position_embeddings`, any
   indexer-specific rope/scale fields). The failure straddles 262,144 —
   suspiciously close to a native-window boundary. If the checkpoint
   declares a scaling regime above its native window that main attention
   applies and the indexer (both accelerated AND official-replay) does not,
   both arms would miss identically at 350k while passing 250k. Minutes of
   work, and it is the only hypothesis on the table that predicts a
   zero-pass result cleanly.
   (ii) Re-run the layer-0-style replay from a *new* trace at a deeper layer
   (e.g., first full-indexer layer ≥ 1 and one late layer) to check input
   provenance/weights at every boundary before blaming training contracts.
   (iii) Only then HF-side multi-layer scoring comparison on the frozen
   activations.

## 4. Decision tree

**All three 350k rows recover:**
1. Run Condition A (posture-matched stock control, r1 only) if no archived
   equivalent exists. Miss ⇒ attribution to the accelerated indexer
   trajectory is clean; announce-able with the frozen-set scope caveat.
   Pass ⇒ pivot: the MTP3/graph decode indexer path is the prime suspect;
   do not start the FP8 ladder.
2. Offline replay decomposition (§3.6) → identifies K-cache vs Q vs
   accumulation with zero boots. Re-capture the RC accelerated selection in
   the same session (closes §2.2).
3. One live confirmation boot for the winning component; smallest canonical
   correction; §14 promotion evidence; upstream-format PR.

**Partial recovery (1–2 rows):**
Do not force either branch. First compare per-row reference-vs-accelerated
selection trajectories and needle-row score margins; check whether the
recovering rows differ in distractor structure. Partial recovery most likely
means the indexer is causal but marginal (needle sits near the cutoff even
under exact scoring) — in that case the corrected component must be
validated with margin metrics (§3.7), and the randomized sweep moves earlier
in the schedule, before the fix is declared.

**Zero rows recover:**
The compressed-indexer-sufficient-cause hypothesis is refuted on the spot.
Run §3.8 in order: config-declared rope/context-extension contract first
(minutes, and uniquely predicts this outcome), then boundary-verified
deeper-layer replay, then HF multi-layer comparison. Do not reach for
selector policies; that ordering is already agreed.

## 5. Critique of the proposed component ladder (§13)

The order is right (K cache first — consistent with where the row mass must
be), the stop condition is right, and the "no selector policy" constraint is
right. Three amendments:

1. **Move the decomposition offline** (§3.6). Live boots confirm, they
   should not explore. This collapses ladder steps 2–5 into one replay
   session plus one confirmation boot.
2. **Pre-commit the "material divergence" definition** before running any
   cell: needle-row exclusion OR recall below a stated threshold OR
   cutoff-margin collapse. Otherwise the ladder invites exactly the
   spin-out risk this engagement exists to prevent — every cell shows
   *some* divergence, and without a pre-committed criterion each one is
   arguable.
3. **§13's fallback design is sound but premature to elaborate** —
   "broad-candidate selection + higher-precision rerank" should stay one
   sentence until the decomposition says FP8 cannot meet the bar after
   canonical scale/rounding fixes. No design work on it before then.

One addition to §14: each promotion candidate boot should repeat the §9
micro-smoke + 8k needle set — they are nearly free and they carry the
live-metadata coverage the operator gates structurally lack.

## 6. Changes required before public summary / issue #182 update

1. Do not publish causal attribution until Condition A closes (or an
   archived stock MTP0/eager 350k miss is cited). If publishing before
   then, the claim must be "official-scorer trajectory + MTP0/eager posture
   recovers the rows", which is honest but weaker.
2. Replace or annotate the 1,905/2,048 figure: "measured on the pre-#85
   build; re-measurement on the fixed RC pending" until §2.2 closes.
3. Scope wording: "causal for the frozen 350k failure set" — the randomized
   sweep, not the frozen gate, licenses any broader claim.
4. The #85 independent reproduction (§4) is publishable now and should be —
   the standalone GPU proof with both-image results is genuinely useful to
   upstream regardless of our regression's outcome.
5. Credit where due in #182: #85 was found upstream; our contribution is
   the independent proof plus the demonstration that it is not the
   deep-retrieval cause. Keeping that distinction crisp protects the
   report's credibility.

## 7. Summary for Sol

The experiment in flight is the right one and nothing here asks you to stop
it. Three things change shape after it lands: close the MTP0/eager posture
confound with one cheap stock control before attributing; decompose the FP8
divergence offline before booting ladder cells; and re-capture the
accelerated selection on the clean RC before its number appears anywhere
permanent. The zero-pass branch has a new first stop — the checkpoint's
declared context-extension contract — because the 250k/350k straddle of
262,144 is too suggestive to leave unchecked, and it is a minutes-cheap read
that uniquely predicts that outcome.
