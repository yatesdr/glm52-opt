# Measurement library — index of measured results and where they live

**Librarian:** Fable · **Started:** 2026-07-25 · **Scope:** every measured result for the GLM-5.2
serving work (v13 → v20), wherever it currently lives.

Purpose: one place that answers "has this already been measured, and where is the evidence?"
It exists because on 2026-07-25 I rebuilt a needle harness that already existed on CN4
(`needle_hunt.py`) and re-derived a deep-context finding Sol had already recorded the night
before. That cost hours. Check here first.

---

## Reading rule — two traps in our historical evidence

**1. `quality_gate.py` scores the needle by substring.** It prints `PASS` if `738216` appears
anywhere in the answer, so it passes corrupted output. Real example, this candidate at 150k:

```
[PASS] needle@150000: '38216, 738216,, 738216, 738216, 738216, 738216, 738216, 38216, 7382, 738216, 738'
```

That is degenerate repetition with truncated corruptions from a document containing the needle
**once** — scored PASS. Treat any deep-context `GATE: PASS` from `quality_gate.py` as
"substring present", not "correct". Verdicts are re-auditable only where the raw answer was
preserved. The rows summarized below link such records; the v19-bf16 outputs quoted in
`needle-hunt-failure.md` are genuinely exact `'738216'`.

**2. `finish_reason=length` is not a miss.** A truncated response with empty `content` means the
reasoning consumed the token budget. Several rows across the library look like failures and are
inconclusive. Always read `finish_reason` and `completion_tokens` before believing an empty
answer. (`6d32-deep2` at 350k is exactly this case.)

Harnesses that avoid both: `needle_hunt.py` (retrieval vs finalization split, Sol),
`harness/v20_gate2_needle_ladder.py` and `harness/v20_needle_duplication_onset_probe.py` (Fable).

---

## Where measurements live

| Location | Contents | Risk |
|---|---|---|
| repo root `*.md` (~30 files) | result/report/proof/spec docs per boot and per investigation | tracked |
| `design/*-verdict.md`, `design/needle-v13-v19-differential.md` | gate verdicts, root-cause differentials | tracked |
| `workspace/v18-results-ledger.md` | v18 durable measured-results ledger | tracked |
| `v20-pr-ledger-20260724.md` | PR-by-PR status ledger | tracked |
| `harness/sol-proof-results/*.jsonl` | no-model proofs: pcie peer/collective matrices, decode-retrieval microprobes | tracked |
| `harness/{int8-dma,fp8-rank-consistency,stage2,phase2-gateB}-proofs/` | per-design proof bundles | tracked |
| `harness/gate2-results/<runtag>/` | Gate 2 cells: full request/response, usage, checks | tracked (new) |
| `harness/cn4-evidence-archive/<date>/` | **evidence recovered from CN4** — see below | tracked (new) |
| CN4 `~/*.log`, `~/needle-out*/`, `~/sol-proof-results/` | 33+ logs and run dirs, some existing **nowhere else** | **AT RISK — dev box, gets rebuilt** |

### CN4 recovery status (2026-07-26) — NF3 `5517197` deep-context session

Archived into **`harness/cn4-evidence-archive/20260726/`** (34 files, 296 K; see its `README.md`
for the per-directory index and reading traps). Analysis in
`design/v20-nf3-350k-findings-handoff-20260726.md`.

- `nothink-consistency-20260726T0125Z/` — the 12-cell ladder, **all `cached_tokens=0`**: 350k
  retrieved 0/6, 450k retrieved 5/6
- `v20-chat-token-identity-250k.tar.gz` — Sol's `proofs#133` probe (18 MB raw → 132 K gz): 5
  response bodies, both prompt-token-id dumps, metadata, summary
- `thinkmode-250k-20260726T0105Z/` — finalization cells: `{}` vs `enable_thinking:false` vs
  `thinking:false` vs untemplated raw
- `deepraw-{350000,450000}-20260726T002335Z/` — raw-vs-chat deep cells. **Not cold**
  (`cached_tokens` 97,792 / 137,216)
- `repro-350k-20260726T0320Z/` — **incomplete, 1 of 9 cells**, stopped deliberately
- `verify_prompt.py` — proves the passing/failing 350k prompts are structurally identical
- `chain-*.log`, `deepraw-*.log` — driver logs, including the 105-byte
  `chain-20260726T0110Z.log` that never ran (self-matching-pgrep deadlock)

Nothing from this session exists only on CN4 any more.

### CN4 recovery status (2026-07-25)

Archived into `harness/cn4-evidence-archive/20260725/`:

- `needle_hunt.py` — Sol's needle harness (retrieval vs finalization split, the good one)
- `needle-ladder-sol.log` — runtag `sol-2600`, 2026-07-24 23:13
- `v20-6d32-deep.log`, `v20-6d32-deep2.log` — runtag `6d32-deep2`, image `6d32a0c3…`
- `needle-out-6d32-deep2/response-*.json` — 50k/250k/350k/475k responses
- `gate2-20260725T161903Z/`, `needle-onset-20260725T162255Z/`, `deep-corruption-20260725T163320Z/`
- `qg-matched-20260725T162818Z.log` — historical-harness control run

Still only on CN4 (lower value, pull before any rebuild): remaining `~/*.log` boot/clock/burn
logs, `needle-ladder{,2..6}.log`, `needle-out{,-240v300w}/`, `~/sol-proof-results/*.jsonl`
(duplicated in repo — verify hashes before trusting the copy).

---

## Deep-context needle retrieval — consolidated across images

> **⚠ Read this table as per-image anecdotes, not as a depth curve.** On 2026-07-26 a matched,
> cold, 3-rep ladder measured **0/6 retrieval at 350k and 5/6 at 450k on the same image and
> process** (see "DEPTH IS NOT THE VARIABLE" below). Retrieval does **not** degrade monotonically
> with depth, so a single PASS or FAIL cell at a given depth predicts very little. Any row here
> with only one or two observations should be treated as a coin flip, and "find the depth where it
> breaks" is not a valid bisection strategy.

Needle `738216` at 40% depth. **Cold** = `cached_tokens=0`. Source column gives the record.

| image / runtag | 50k | 100k | 150k | 250k | 350k | 475k | source |
|---|---|---|---|---|---|---|---|
| v19 `gilded-gnosis-v19`, wire bf16 (`B12X_PCIE_DMA_FP8=0`) | PASS exact | — | — | — | **PASS exact** | — | `needle-hunt-failure.md` |
| v19, wire fp8 `ag` | — | — | — | — | FAIL `''` | FAIL `''` | `needle-hunt-failure.md` |
| v19, wire fp8 `ring` | PASS exact | — | — | — | FAIL `''` ×3 | FAIL `''` | `needle-hunt-failure.md` |
| v20 `sol-2600` (2026-07-24 23:13) | PASS exact | — | FAIL empty | FAIL empty | — | — | `needle-ladder-sol.log` |
| v20 `6d32a0c3…` (2026-07-25 03:35) | PASS exact | — | — | **PASS exact** | inconclusive (`length`) | FAIL empty (reasoning had it) | `v20-6d32-deep2.log` |
| v20 `fa71a0c1…` (2026-07-25 13:48, prod-ready candidate) | marginal: 2/3 cold, 3/3 warm | **0/6** | **0/6** | — | — | — | `corruption-matrix-20260725T163734Z` |
| v20 NF3 `5517197` (2026-07-26, chat API) | — | PASS exact | — | FAIL empty | FAIL absent | — | see NF3 section below |
| v20 NF3 `5517197` (2026-07-26, **raw** `/v1/completions`) | — | — | — | **PASS exact** | **PASS exact** | **PASS exact @450k** | see NF3 section below |

**Reading:** the failure expression is finalized `content` empty (or corrupted) while
`reasoning` often holds the needle. Its observed onset is stochastic and moved across images and
boots: `6d32a0c3` passed 250k cold, `fa71a0c1` failed 100% at 100k in its controlled matrix, and
the earlier `sol-2600` boot failed at 150k. That non-monotonic onset does **not** rule out a source
change; a numerically marginal source regression can move the visible onset with prompt and
execution details. v19-bf16 passing exactly at 350k is the standing quoted reference.

### The transition is a degradation band, not a hard depth threshold (`fa71a0c1`, 2026-07-25)

DCP4 splits context four ways, so per-rank depth = ctx/4. Cold, natural head, temperature 0:

| actual ctx | per-rank | outcome | accept | out_tok |
|---|---|---|---|---|
| 49,118 | 12,280 | 7/9 EXACT | 0.25–0.72 | 93–164 |
| 58,901 | 14,725 | EXACT, then `'738216 0'` | 0.43 / 0.13 | 88 / 8 |
| 60,881 | 15,220 | empty (needle in reasoning) | 0.19 | 57 |
| 60,883 | 15,221 | empty (needle in reasoning) | 0.33 | 83 |
| 62,863 | 15,716 | empty | 0.39 | 14 |
| 62,861 | 15,715 | truncated, `'compressor'` ×3 | 0.92 | 2000 |
| **63,852** | **15,963** | **EXACT `'738216'`** | 0.34 | 100 |
| **63,851** | **15,963** | **empty** | **0.00** | **2** |
| 64,786 | 16,197 | empty | 0.72 | 856 |
| 64,786 | 16,197 | empty (needle in reasoning) | 0.52 | 1516 |
| 68,747+ | 17,187+ | 0/N EXACT, total | 0.00 | 2 |

**The two bolded rows are one token of context apart, temperature 0, and have opposite outcomes.**
Their prompts are not byte-identical, so they do not prove nondeterminism for one fixed request.
They do prove that depth alone is insufficient and rule out a hard index/capacity boundary as the
complete trigger. The later safe-BMM probes supply the independent evidence for numerical
marginality. The 16,384/rank boundary — global 65,536 — is an amplifier at most, and the fused
source is byte-identical across the regression anyway (Sol, `dev#37`). Two distinct failure
expressions coexist in the band: `accept=0.0`/`out_tok=2` immediate stop, and high-acceptance
degenerate repetition.

Earlier claims that this was a sharp cliff at 60k→70k (Fable) or a point boundary at 65,536
(both) were artifacts of coarse sampling. Corrected here rather than in the channel history.

Corrupted-but-non-empty variants of the same mechanism, all from `fa71a0c1`:
`'73838216'` (50k), `'738 738216. 738216. 216.'` (100k),
`'38216, 738216,, … 7382, 738'` (150k), and `'40 40 40 40…'` in reasoning (250k, `sol-2600`).

### NF3 `5517197`: retrieval passes to 450k on the raw API; only the templated stream fails (2026-07-26)

Image `glm52-serve:v20-5517197-pxb-20260725`, KV pool 559,616, GMU 0.97, `KV_FP8_ROPE=1`,
`kv-cache-dtype=nvfp4_ds_mla`, `SPARKINFER_PCIE_DMA_FP8=0` (bf16 wire), `NCCL_P2P_LEVEL=PXB`,
TP4/DCP4/MTP3. Fixed seed 20260725. **One live process for all cells** — no boot, config or weight
change between them. Harness `harness/v20_finalization_discriminator.py`.

| depth | ctx | raw `/v1/completions` | chat `/v1/chat/completions` |
|---|---|---|---|
| 250k | 245,491 | **EXACT** 5 tok `stop` | `REASONING_ONLY` 18 tok `stop`, `content=None` |
| 350k | 343,721 | **EXACT** 5 tok `stop`, `' \n\n738216'` | `ABSENT` 2000 tok `length`, reasoning 10,653 chars |
| 450k | 441,951 | **EXACT** 4 tok `stop`, `'738216'` | `REASONING_ONLY` 19 tok `stop`, `content=None` |

**This is the first v20 image to retrieve correctly at 350k and 450k.** Retrieval, `nvfp4_ds_mla`,
the 368-byte FP8-RoPE record, DCP4 sharding, `a2a` transport and MTP3 are all sound through
441,951 tokens here. The chat-path failure is not a retrieval failure.

> **Do not generalize the raw column to "raw always works."** A later 250k raw cell on the *same
> image, process, prompt and seed* FAILED: `finish=length`, 2000 tokens, needle absent, degenerate
> repetition (`' No explanation. No yapping. No preface. … No echoing. No mirroring.'` looping to
> the ceiling) — `thinkmode-250k-20260726T0105Z/raw-raw.json`, `cached=67584`. Untemplated raw is a
> *fragile* probe: with no chat template the model is not being addressed as an assistant and can
> fall into completion-style loops. This is the same high-acceptance degenerate-repetition
> expression logged for `fa71a0c1` above. Correct statement: **the model can retrieve at
> 250k–450k, but not dependably.** Fable's `dev#139` overstated this; corrected in `proofs#143`.

**Not cold.** `cached_tokens` was 97,792 (350k) and 137,216 (450k). Needle offset ~137k and ~176k
respectively, past the cached prefix, so the needle region was freshly computed — but these are
**not** cold runs and must not be quoted as such. The 250k cell predates this note; check its
`summary.json` before quoting it as cold.

**Mechanism of the blank response (symptom, not regression).** `vllm/parser/glm47_moe.py:125` sets
`initial_state = ParserState.REASONING if thinking else ParserState.CONTENT`, and the only
`REASONING → CONTENT` transition is the literal `</think>`. `/model/chat_template.jinja:118` ends
the assistant turn with a bare unclosed `<think>` unless `enable_thinking` is defined and false.
So generation always begins inside a think block, and any completion that ends without emitting
`</think>` is filed 100% as `reasoning` with `content=None` regardless of length or correctness.
The 450k cell is the cleanest instance: 19 tokens, `finish=stop`, no truncation, and the whole
output is a complete correct answer — *"The maintenance ticket number for the Facility 27
compressor overhaul is **738216**."* — with no closing transition.

### Token-level proof: `</think>` emission is NONDETERMINISTIC (2026-07-26, `proofs#143`)

Sol's `harness/v20_chat_token_identity_probe.py` (`proofs#133`) at 250k, ctx=245,491, live NF3
process, output `cn4:/home/derek/v20-chat-token-identity-250k`. It tokenizes the chat request and
feeds those exact IDs to `/v1/completions`, so the two endpoints are compared on identical input.

| cell | verdict | generated token ids | sha |
|---|---|---|---|
| `raw-plain` (untemplated) | EXACT | `[4710, 22, 100919, 122250, 154827]` | `f8f4f892cfc1` |
| `raw-chat-ids` (thinking IDs → raw) | EXACT | `[22, 100919, 122250, 154827]` | `3deb38ee3d70` |
| `chat-thinking` | EXACT | `[22, 100919, 122250, **154842**, 22, 100919, 122250, 154827]` | `68eb02a4e342` |
| `raw-no-think-ids` → raw, **249.6s cold** | EXACT | `[22, 100919, 122250, 154827]` | `3deb38ee3d70` |
| `chat-no-thinking`, 2.3s warm | EXACT | `[22, 100919, 122250, 154827]` | `3deb38ee3d70` |

**Token decode:** `22,100919,122250` = `738216`; **`154841` = `<think>`**, **`154842` = `</think>`**,
`154827` = eos. Confirmed by the probe's own suffix dump: the thinking suffix ends
`[…154828, 154841]` and the no-think suffix ends `[…154828, 154841, 154842]` — i.e. exactly
`chat_template.jinja:118`.

**The decisive pair.** `raw-chat-ids` and `chat-thinking` consume the **same rendered input token
IDs**. `raw-chat-ids` generated 4 tokens with **no** `154842`. `chat-thinking` generated 8 —
answer, `</think>`, answer again. **Identical input, temperature 0, different generated streams.**
So the v20 stream difference is *nondeterministic*, not a fixed property of the templated path.
The model retrieved `738216` correctly in all five cells; only the closing-tag emission varies.

This reframes the open question. It is not "why does v20 never emit the close" but **"why is
emission of `154842` a coin flip at depth in v20 when v19 emits it reliably."** The parser is the
*amplifier*: `initial_state=REASONING` with no end-of-generation fallback converts an occasionally
omitted token into a hard `content=None`.

**Determinism control.** `raw-no-think-ids` (249.6s, cold prefill) and `chat-no-thinking` (2.3s,
warm) ran on *different endpoints* and produced **byte-identical** ids `3deb38ee3d70`. The no-think
path is stable across cold/warm and across endpoints, which argues the nondeterminism above is not
merely prefix-cache perturbation of numerics.

**Matched thinking-mode cells** (`harness/v20_thinking_mode_finalization_fix.py`, same process,
250k, `cn4:/home/derek/thinkmode-250k-20260726T0105Z`):

| kwargs | verdict | out_tok | cached |
|---|---|---|---|
| `{}` (server default `reasoning_effort=high`) | `REASONING_ONLY`, `content=None`, reasoning=`738216` | 4 | **0 (cold)** |
| `{enable_thinking: false}` | **EXACT**, `content='738216'` | 4 | **0 (cold)** |
| `{thinking: false}` | `IN_CONTENT`, 80 chars | 18 | 244,992 (warm) |

`thinking:false` is read by the **parser** and ignored by the **template** (which checks only
`enable_thinking`, lines 3 and 118), yet content populated anyway. **So parser `initial_state`
alone is sufficient to recover content; the closed-pair render is not required.** That points the
minimal forward fix at the parser rather than the template, and it preserves thinking for users who
want it — unlike `enable_thinking:false`, which disables reasoning wholesale. *No fix has been
applied; this records the discriminator only, per Derek's direction.*

**This is NOT the v19→v20 regression cause.** Sol byte-compared CN3 v19 and CN4 v20:
`chat_template.jinja`, `glm47_moe.py`, `abstract_parser.py`, `tokenizer_config`,
`generation_config` and the chat serving path are **SHA-identical** (`dev#137`). v19 passes with
the same fragile parser because its generated stream emits the closing transition. The open
question is why the v20 stream stops emitting it. Fable's `dev#135` overclaimed this as root
cause; corrected in `dev#138`.

**Depth signal for that open question:** chat was EXACT at 100k on this same image, so v20 does
emit the closing transition at 100k and stops somewhere between 100k and 250k.

**Do not "fix" this by promoting `reasoning` to `content`** — it leaks CoT and would mask a
numerical regression (Sol, `dev#137`).

Two template defects found while reading the above, both live in `/model/chat_template.jinja`:

- **line 2** — `effective_reasoning_effort = 'high' if reasoning_effort == 'high' else 'max'`.
  Only the literal `'high'` is recognized; `low`, `medium` and unset all resolve to **`max`**.
  Cells labelled "chat-low" in the 2026-07-26 logs therefore ran at *maximum* effort, which is
  why 350k burned 10,653 reasoning characters.
- **`glm47_moe.py:184-192`** — `thinking_enabled` is derived from only `thinking` /
  `enable_thinking`. It cannot see `reasoning_effort`, which is the key the server actually sets
  via `--default-chat-template-kwargs`. So the parser is always in thinking mode no matter what
  `reasoning_effort` any client sends.

Filename trap in `v20_finalization_discriminator.py`: outputs are `raw-<variant>.json`, where
`raw-` means *raw response body*, not the raw API. The untemplated `/v1/completions` result is
`raw-raw.json` (`object=text_completion`, `choices[0].text`, no `message`).

### DEPTH IS NOT THE VARIABLE — 350k fails 0/6, 450k passes 5/6 (2026-07-26, `dev#145`)

`harness/v20_nothink_consistency_ladder.py`, live NF3 `5517197`, matched arms on the same document
per rep, temperature 0, **`cached_tokens=0` on every cell**. Seeds `760000+depth+rep`, so each rep
is a *different* document and the prefix cache cannot carry the answer between reps.
Out: `cn4:/home/derek/nothink-consistency-20260726T0125Z`.

| depth | arm | verdicts | retrieved |
|---|---|---|---|
| 350k (ctx 343,721) | `nothink` | ABSENT, ABSENT, ABSENT | **0/3** |
| 350k (ctx 343,721) | `thinking` | ABSENT, ABSENT, ABSENT | **0/3** |
| 450k (ctx ~441,951) | `nothink` | IN_CONTENT ×3 | **3/3** |
| 450k (ctx ~441,951) | `thinking` | EXACT, ABSENT, REASONING_ONLY | **2/3** |

**441,951 tokens works. 343,721 tokens fails completely.** Same image, same process, within two
hours. Every ladder in this document — including the consolidated table above — was built on the
assumption that retrieval degrades monotonically with depth. **That assumption is wrong.** Any
bisection strategy of the form "find the depth where it breaks" will chase noise.

*Scoring caveat:* the script's tally counts only the `EXACT` label, so it printed
"450k nothink 0/3 EXACT". Those cells were `IN_CONTENT` — the needle **was** retrieved, inside a
sentence that also contains "Facility 27", so digit-only equality didn't fire. Retrieval is 3/3.
Reporting bug, not a failure. **Score `reasoning`+`content` together, and record the arm.**

**Two independent perturbation-sensitivity demonstrations** — the most useful result of the night:

1. **Head sensitivity (350k).** Verified on CN4 (`cn4:/home/derek/verify_prompt.py`) that the
   passing and failing 350k prompts are structurally identical: `ctx=343,721`, needle present
   exactly once, at token **137,496**, 40.0% depth, for seed `20260725` (passes) and seeds
   `1110001`/`1110002` (fail). The **only** difference is ~90 characters of random header text at
   position 0 — a packet id and two small integers.
2. **Tail sensitivity (450k), one token.** At 450k rep2 the `nothink` arm retrieved the needle and
   the `thinking` arm answered *"I'm sorry, but there is no maintenance t…"* — **on the same
   document**. The only difference between those requests is the closed-vs-open think tag at the
   prompt tail, i.e. the presence of token `154842`.

A single suffix token cannot change what is visible in a 442k context by any legitimate mechanism.
Both results point at a computation sitting close enough to a **numerical boundary** that an
arbitrary small perturbation of the input distribution tips it — consistent with the degradation-band
finding above, and it explains why the historical onset wandered non-monotonically across images
and boots.

**The parser destroys genuine successes at depth.** 450k rep3 `thinking` retrieved the needle and
returned `content=''`. So the model found the answer 2/3 and a user would have seen it 1/3. Any
pass-rate collected through the chat API with thinking on is biased downward.

**`enable_thinking:false` is NOT a fix, and is arguably worse for production.** 350k `nothink` 0/3
all *fabricated* a plausible ticket number (`MNT-2024-087` and similar) with `finish=stop` and clean
finalization. `thinking` at 350k at least reported that it could not see the document. Trading a
visible failure for a confident fabrication is the wrong trade for CN3.

### CAUSAL GATE 1 — legacy MLA projection: **REFUTED 0/3** (2026-07-26, `dev#158`)

First hypothesis-driven experiment of the investigation, replacing characterization. Sol's pinned
Python-only discriminator (`design/v20-nf3-legacy-projection-causal-gate.md`, spec sha `38788491…`).

**Image** `glm52-serve:v20-5517197-nf3-legacy-projection-20260726`, id `e280ca23…`, revision
`d367318c9a74ddc4d79de0ab6db81e9aab9b81dc`. Built from the exact `5517197` base with a patch to
**only** `envs.py` + `mla_attention.py` — every input and output hash asserted at build time, no
CUDA/C++/SparkInfer/FlashInfer/model byte rebuilt. **Compose diff vs the stock NF3 boot was exactly
one line: the image identity.** MNBT stayed 2048, BF16 wire, PXB, MTP3, `KV_FP8_ROPE=1` unchanged.

**Mode verified at runtime, not assumed:** log contains `mla_attention.py:1677` *"Using materialized
BF16 MLA projection weights and staged BMMs (VLLM_B12X_MLA_PROJECTION_MODE=legacy)"* and **zero**
occurrences of the MXFP8-pack line stock emits at `:1744`.

Frozen inputs — prompt sha256 re-verified at send time, aborting on drift, so a pass could not come
from a substituted input (`harness/v20_freeze_causal_gate_prompts.py` → manifest archived):

| cell | verdict | gate | out_tok | prompt_tok | cached | content |
|---|---|---|---|---|---|---|
| `pass-250k-ctl` | EXACT | **PASS** | 4 | 245,497 | **0** | `738216` |
| `fail-350k-r1` | ABSENT | FAIL | 18 | 343,727 | **0** | `…is **27-27**.` |
| `fail-350k-r2` | ABSENT | FAIL | 32 | 343,727 | **0** | `…is **27**. \n\nThis is` |
| `fail-350k-r3` | ABSENT | FAIL | 72 | 343,727 | **0** | `…is **27**. \n\n(Note: ` |

**The v20-only MLA projection seam — native MXFP8 absorbed query/value BMM plus fused query
assembly — is NOT sufficient to cause cold 350k failure.** Remove it from the long-context causal
path. Control passed, so this negative result stands rather than reflecting a broken boot.

**Independently disqualified from production anyway:** pool **491,520** vs stock 559,616 (−12.2%),
below the 500k floor. Broad legacy materialization could not have shipped even had it recovered.

**Partial effect worth keeping — the failure morphology changed consistently.** Stock *fabricated*
`MNT-2024-087`, a ticket-shaped string absent from the document. Legacy instead returned `27` — a
token that genuinely **is** in the prompt ("Facility 27") — in all three cells. *Caution:* "Facility
27" also appears in the question at the very end of the prompt, so `27` is reachable from **near**
context; this is still not deep retrieval, still a failure, and n=3 on one boot. Recorded as an
observation, not a claim. But the direction was consistent, which argues the projection path
*influences* long-context behaviour without being sufficient to fix it — a reason to run the
`proofs#134` query-BMM / value-BMM / fusion decomposition rather than discard the seam.

**Thermal confound disclosure.** Every cell in this table, and all earlier 2026-07-26 evidence
(the 350k 0/6 ladder, `proofs#133`), ran with the exhaust-end GPU at 88–89 °C — 1 °C under the
Blackwell slowdown point. `clocks_event_reasons.active` was `0x0` on all four cards throughout, so
nothing throttled and no verdict changes. Fixed mid-session; see `harness/gpu_fan_sync.py`.

---

## Safe-query numeric delta is real but not causal (2026-07-25)

Changing `safe_mla_query_bmm` from `CUBLAS_COMPUTE_32F_PEDANTIC` to
tensor-core-eligible `CUBLAS_COMPUTE_32F` changes the operator's BF16 output,
and some changes survive the FP8 boundary. That finding is reproducible, but
an exact binary rewind proved that it does **not** control the observed
long-context retrieval failure.

| evidence | result | conclusion |
|---|---|---|
| static bisect | old and new both use `i8_ring`; top-k control 160/160 clear | wire mode and widened top-k do not explain the onset |
| two independent BMM probes | 45/54 BF16 fingerprints and 16/54 post-FP8 fingerprints changed | the compute-mode delta is real |
| full-width `M=3072` cell | 9/9 post-FP8 fingerprints changed; PEDANTIC costs 2.12× there | the delta reaches production prefill geometry, but global PEDANTIC is not shippable |
| exact binary rewind | discriminator `.so` matches the previously passing `6d32` bytes and 0/54 reference fingerprints differ | the proof image genuinely restored old BMM numerics |
| **cold 100k causal boot** | `cached_tokens=0`, `finish_reason=stop`, 2 completion tokens, acceptance 0, empty content: **MISS** | **safe-query compute mode is not the end-to-end cause** |
| CN3 v19 instrument control | same 98,226-token prompt, cold, `content=738216`: **EXACT** | the harness is sound and the v20 miss is real |

The synthetic CPU witness remains useful as a numeric counterexample: reduced
intermediate precision can flip an NVFP4 attention result for a valid selected
window. It is not evidence that this mechanism causes the model regression.
The accurate-reduction branch and build package are therefore withdrawn from
retrieval qualification and retained only as a separate numeric-correctness
draft.

PR `#171` is also **not causal** for the post-6d32 onset. The no-`#171`
fa71 discriminator boot completed cleanly with a 500,992-token pool, but its
first cold 100k row still returned only `reasoning="The"`, empty finalized
content, `finish_reason=stop`, two completion tokens, acceptance 0, and
`verdict=MISS`. Removing the route change therefore did not repair retrieval.
The model-free high-index arithmetic remains a valid route test, not a
model-quality proof.

The stock vLLM `5517197` NF3 image also lacks `#171`, so booting it under the
same routing environment would not independently test that hypothesis. The
earlier recommendation to prioritize a same-image SYS/PXB control is
superseded by the official-scorer and shared-input precision evidence below:
the failure was reproduced with the trained scorer contract itself, then
recovered by changing upstream main-attention representations while keeping
the scorer fixed.

## Official-scorer shared-precision bisection (2026-07-27)

The production exact selector was replaced end to end with the official GLM
BF16/FP32 scorer while retaining exact top-k. That scorer passed the frozen
250k control but failed all three frozen 350k rows. A per-layer trace showed
the ticket value tokens absent from the selected top-2,048 through layer 38
and only intermittently present at layers 62--74. This localizes the failure
before sparse-attention consumption: the relevant history is not selected
early enough under the failing shared-input posture.

The same immutable 343,727-token r1 row then recovered exactly under the
highest-precision main-attention posture that fits CN4:
`fp8_ds_mla`/BF16-RoPE/raw-BF16 DCP wire. Restoring the block-INT8
`i8_ring` wire while holding the official scorer, cache format, prompt, and
execution posture fixed also returned exact `738216`:

| posture | result | finish | out | cached | elapsed |
|---|---|---|---:|---:|---:|
| `fp8_ds_mla` + BF16 RoPE + raw BF16 wire | `EXACT` | `stop` | 4 | 0 | 708.8 s |
| `fp8_ds_mla` + BF16 RoPE + `i8_ring` | `EXACT` | `stop` | 4 | 0 | 689.0 s |

`i8_ring` is a rank-consistent block-INT8 codec with an `amax/254`
pre-BF16 absolute-error bound, **not** a lossless byte transport. The second
row establishes sufficiency for this frozen prompt under the held cache
posture; it does not establish arbitrary-activation bit equality.

The joint official-scorer + calibrated-scale cell retained the compact
368-byte NVFP4+FP8-RoPE cache and `i8_ring` and recovered the same frozen r1
exactly:

| scorer | calibrated outer scales | runtime | r1 result | cached | elapsed |
|---|---:|---|---|---:|---:|
| official BF16/FP32 | off | clean runtime-stride RC | `ABSENT` | 0 | recorded in official-reference gate |
| stock accelerated | on | pre-runtime-stride `5517197` / `be0edcaa` | `ABSENT` | 0 | 415.2 s |
| official BF16/FP32 | on | clean runtime-stride RC | `EXACT` (`738216`) | 0 | 667 s |
| stock accelerated | on | clean runtime-stride RC | `EXACT` (`738216`) | 0 | 327 s |

The fourth row is the clean complement. It proves that the calibrated scales
are sufficient with the stock accelerated exact scorer on the current RC.
There is no demonstrated scorer interaction, and the official BF16/FP32
scorer remains a diagnostic oracle rather than a production fix. The earlier
scales-only miss used SparkInfer `be0edcaa`, before the clean RC's `c3828fd`
runtime-stride fix, and must not be used to contradict this result.

Historical evidence also prevents calling the scales the unique v19-to-v20
differential: v19 used uncalibrated NVFP4+FP8-RoPE and recovered under BF16
wire. The measured v20 endpoint is nevertheless now minimal: canonicalize PR
#145's per-layer scale calibration while preserving stock exact top-k, the
accelerated indexer, the 368-byte cache record, FP8 RoPE, and `i8_ring`.
Healthy layer-entry trace and the full frozen/randomized gates remain required
before promotion.

Detailed records:

- `design/v20-shared-precision-trajectory-causal-20260727.md`
- `design/v20-i8-ring-causal-confirmation-20260727.md`
- `design/v20-calibrated-scales-clean-rc-complement-20260727.md`
- `harness/cn4-evidence-archive/20260727/official-reference-needle-trace-103473cd/`
- `harness/cn4-evidence-archive/20260727/official-reference-fp8kv-bf16rope-lossless-r1-v1/`
- `harness/cn4-evidence-archive/20260727/official-reference-fp8kv-bf16rope-i8ring-r1-v1/`
- `harness/cn4-evidence-archive/20260727/official-reference-nvfp4kv-fp8rope-i8ring-scales-r1-v1/`
- `harness/cn4-evidence-archive/20260727/runtime-stride-stock-nvfp4-scales-r1-v1/`

## Throughput / capacity reference points

| metric | value | image | source |
|---|---|---|---|
| matched CN4 v19 prefill control, 8k | 415 effective tok/s | v19 | `v20-prod-ready-20260724-fable-handoff.md` |
| matched CN4 v19 prefill control, 55k | 392 effective tok/s | v19 | same |
| Gate 3 floors (90% of control) | 8k ≥ 374, 55k ≥ 353 tok/s | — | same |
| CN4 v20 cold prefill, 8k | 1364 tok/s | `fa71a0c1` | `fabric#5` / `cn4-pcie-fabric-investigation.md` |
| CN4 v20 cold prefill, 55k | 1115 / 1221 tok/s | `fa71a0c1` | same |
| **CN4 v20 cold prefill curve** (server tok/s) | 8k **1411** · 16k **1301** · 32k **1305** · 55k **1222** · 100k **1151** · 150k **1092** | `fa71a0c1` | `context-sweep-20260725T181*` |
| **CN4 v20 decode by context** (aggregate tok/s) | ctx0 C16 **167.9** · ctx16k **134.3** · ctx32k **125.5** · ctx64k **129.1**; single-stream C1 falls hardest (16.6 @16k → 8.1 @64k) | `fa71a0c1` | same |
| CN4 v20 decode aggregate ctx0 C1/C4/C8/C16 | 53.64 / 110.68 / 133.04 / 171.35 tok/s | `fa71a0c1` | `dev#6` (Sol) |
| CN4 v20 decode aggregate ctx0 C4/C8/C16 | 99.93 / 136.07 / 167.92 tok/s | `fa71a0c1` | `v20-baseline-20260725T173314Z` (independent repro) |
| **CN4 v20 decode aggregate ctx16k C1/C4/C8/C16** | **16.59 / 86.06 / 114.15 / 134.25 tok/s** | `fa71a0c1` | same — **first ever measurement of this cell** |
| CN4 v20 MTP acceptance, decode cells | 0.50–0.63 across ctx0 and ctx16k, zero errors | `fa71a0c1` | same |
| CN4 v20 prefill cold 8k, independent repro | 1411 server tok/s (vs Sol's 1364) | `fa71a0c1` | same |
| GPU KV pool | 501,504 (`fa71a0c1`) · 487k (`6d32a0c3`) · 644,864 (v19 CN3) | — | `dev#6`, memory |
| PXB vs SYS packed-CKV gather | 10.1–11.3× improvement | `fa71a0c1` | `fabric#5` |
| SM utilization at 55k, all ranks | 98.1–98.3% | `fa71a0c1` | `fabric#5` |

**Gaps worth filling when CN4 is free** (do not re-measure what is above):

1. ~~v20 decode at ctx16k~~ — **DONE 2026-07-25**, see table above. ctx16k costs ~20% at C16 and
   far more at C1. Decode itself is healthy (all finalized, no errors), so the defect is specific
   to long-context retrieval/finalization, not to generation throughput.
2. v19 vs v20 matched prefill on the *same* box and harness — the 415/392 controls predate PXB.
   **Needs a v19 boot**, so it is blocked behind the fix work, not behind idle time.
3. ~~The fa71 no-`#171` cold 100k discriminator~~ — **DONE; MISS.** Removing
   `#171` did not repair the symptom. Run the same-image `SYS` transport
   control before any stock `5517197` qualification; require effective
   process environment, identical prompt hashes/seeds, and cold-cache proof.
4. ~~MTP acceptance by depth~~ — partially done: acceptance is 0.0 on many deep failures but
   *normal* (0.41) on others at 150k, so collapse is a symptom and not the mechanism. A deliberate
   sweep is still worth doing on a healthy image, where it becomes a regression tripwire.
5. Do not price or promote PEDANTIC as a retrieval fix. Its full-width
   operator cost is already measured at 2.12× and its causal boot missed.

---

## Superseded PEDANTIC window (completed 2026-07-25)

`harness/run_authorized_model_down_window.sh` (`71626eb2…`) remains an
example of a fail-closed diagnostic window, but its PEDANTIC promotion path is
superseded. The free-GPU numeric gate completed, the exact binary rewind
booted, and the first cold 100k causal cell missed. No deep PEDANTIC ladder or
throughput pricing is warranted.

Two fail-open defects in Fable's first draft were caught by Sol's review and are worth remembering:
`set -uo pipefail` without `-e` (failing gate steps would have been stepped over and the boot would
have proceeded as if the gate passed), and a verdict grep that could not detect missing rows (a probe
dying after 2 of 6 cells would have read as all-EXACT). Gate scripts must assert on parsed counts,
not on the absence of known-bad strings.

### Artifact provenance (do not confuse these)

| artifact | built from | note |
|---|---|---|
| `…:gilded-gnosis-v20-pedantic-discriminator-20260725`, manifest `f2e63dc7…` | fa71a0c1 + 6d32's `_C_stable_libtorch.abi3.so` (`fcf056af…`) | `#171` retained exact (`885401fc…`); verified numerically identical to 6d32 (0/54 changed) |
| accurate-reduction source/package | branches `0eb51f992` / `29f5f0e92` and their pinned build packages | withdrawn from retrieval qualification after the PEDANTIC causal MISS; numeric-correctness work only |

## Conventions for new measurements

1. Record `finish_reason`, `completion_tokens`, `prompt_tokens`, `cached_tokens` on every row.
   An answer without those four is not evidence.
2. Prove cold with `cached_tokens=0`. Break the prefix cache with a **short natural-language**
   unique header — never random gibberish: 220 nonsense words made this model finalize empty on
   ~70% of requests at ≥15k (2026-07-25 instrument error, see the probe docstring).
3. Give reasoning models room: budget ≥2000 tokens, and treat `finish_reason=length` as
   inconclusive rather than failure.
4. Score exact finalized content, not substring presence.
5. Archive request+response JSON, not just verdicts, so future re-audits are possible.
6. Pull CN4 artifacts into `harness/cn4-evidence-archive/<date>/` the same day they are produced.
7. Add the run to the tables above, with its runtag and image.
