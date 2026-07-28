# v20 NF3 `5517197` — session findings handoff to Sol

**Author:** Fable · **Date:** 2026-07-26 · **Host:** CN4 (192.168.13.34) · **Prod untouched:** CN3 never taken down.

Everything below was measured on **one live container**, `glm52-v20-candidate`, started
`2026-07-25T23:23:59Z`, `restarts=0`, `OOMKilled=false`. No config change, no reboot, no patch
applied at any point in this session. Where a claim was later retracted it is marked **RETRACTED**
with the correction, rather than deleted.

---

## 1. Exact image under test (verified from labels, not from notes)

`glm52-serve:v20-5517197-pxb-20260725`, image id `sha256:2566f905f132…`

Base `voipmonitor/vllm@sha256:e7a8a8549c10b5d1…` with:

| component | commit |
|---|---|
| vLLM | `551719766029e78824a30d97ae6ac63917405b5f` (`build/gilded-gnosis-v20-dcp-final2-20260725`) |
| SparkInfer / b12x | `be0edcaae6f5d284bb29a82325aba7a0ead6960f` |
| FlashInfer | `801d57a08958c13d375ddbb6be3be4808f48a708` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| NCCL | 2.30.4 `canonical/cu132-nccl2304-amd-noxml` |
| CUDA / torch | 13.2.1 / 2.12.0+cu132 |

`local-inference.vllm.patch_url`, `patch_sha256`, `patch_file` are all **empty** — no vLLM source
patch in the base.

**Exactly one layer added on top**, and it is a shell script, not code:

```
COPY --chmod=0755 launchers/serve-glm52-v16.sh /usr/local/bin/serve-glm52-v16.sh
RUN  test sha256(serve-glm52-v16.sh) == fee02f8cd61a4c7edfc9d2b31b62f35ea18424ecde2968064eb212bd441fd883
  && grep -Fqx 'export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"' /usr/local/bin/serve-glm52-v16.sh
LABEL local-inference.launcher.patch=9590e93
```

Stock launcher hardcoded `NCCL_P2P_LEVEL=SYS` and silently discarded the compose value; this makes
it overridable. Fail-closed on the input hash. **No patch of Sol's is in this image** — the
`dev#129` legacy-projection candidate remains unbuilt and unbooted.

---

## 2. COMPLETED measurements

Needle `738216` at 40% depth. **Cold = `cached_tokens=0`, stated per cell.** Prompt structure
verified identical across seeds (§4).

### 2.1 Consistency ladder — the primary result

`harness/v20_nothink_consistency_ladder.py` · out `cn4:/home/derek/nothink-consistency-20260726T0125Z`
Matched arms on the same document per rep, temperature 0, **`cached_tokens=0` on all 12 cells**.
Seeds `760000+depth+rep`, so every rep is a different document.

| depth | ctx | arm | verdicts | retrieved |
|---|---|---|---|---|
| 350k | 343,721 | `nothink` | ABSENT, ABSENT, ABSENT | **0/3** |
| 350k | 343,721 | `thinking` | ABSENT, ABSENT, ABSENT | **0/3** |
| 450k | ~441,951 | `nothink` | IN_CONTENT ×3 | **3/3** |
| 450k | ~441,951 | `thinking` | EXACT, ABSENT, REASONING_ONLY | **2/3** |

Plus one 350k `nothink` cell on seed `20260725` (cold): ABSENT. **350k `nothink` is 0/4; 350k
overall is 0/7 cold.**

*Scoring caveat:* my summary line prints only the `EXACT` label, so it reported
"450k nothink 0/3 EXACT". Those cells were `IN_CONTENT` — retrieval succeeded, inside a sentence
that also contains "Facility 27" so digit-only equality did not fire. **Score `reasoning`+`content`
together and record the arm.**

### 2.2 Failure morphology at 350k

`thinking`, cold, `prompt_tokens=343,733`, `finish=stop`, 121 out, `content=None`, reasoning 628 chars:

> "The user is asking for the maintenance ticket number for "Facility 27" based on the provided
> document. However, **the document content is not provided in the prompt**."

Same 628 chars contain token-level corruption: `"for Facility 27,, would need to know"`,
`"Since the document is not provided,, review the maintenance ticket number cannot be determined"`,
`"If you have the maintenance ticket number,, review the document details"` — doubled commas,
dropped subjects, spurious `review` tokens spliced mid-clause. Same class as the `fa71a0c1`
corrupted-digit signatures.

`nothink`, same document, cold, `finish=stop`, 138 out, content populated:

> "The maintenance ticket number for the Facility 27 compressor overhaul is **MNT-2024-087**."

plus three confident explanatory sentences. `MNT-2024-087` appears nowhere in the document. Clean
finalization of a fabrication.

**`enable_thinking:false` is not a fix and is worse for production**: it converts a visible failure
into an authoritative invisible one.

### 2.3 Token-identity control — `proofs#133` executed

Sol's `harness/v20_chat_token_identity_probe.py` at 250k, ctx=245,491 ·
out `cn4:/home/derek/v20-chat-token-identity-250k` (5 response bodies, both prompt-token-id files,
metadata, summary — all preserved).

| cell | verdict | generated ids | sha |
|---|---|---|---|
| `raw-plain` | EXACT | `[4710, 22, 100919, 122250, 154827]` | `f8f4f892cfc1` |
| `raw-chat-ids` | EXACT | `[22, 100919, 122250, 154827]` | `3deb38ee3d70` |
| `chat-thinking` | EXACT | `[22, 100919, 122250, **154842**, 22, 100919, 122250, 154827]` | `68eb02a4e342` |
| `raw-no-think-ids` (**249.6 s, cold prefill**) | EXACT | `[22, 100919, 122250, 154827]` | `3deb38ee3d70` |
| `chat-no-thinking` (2.3 s, warm) | EXACT | `[22, 100919, 122250, 154827]` | `3deb38ee3d70` |

Decode: `22,100919,122250` = `738216`; **`154841` = `<think>`, `154842` = `</think>`**;
`154827` = eos. Confirmed by the probe's suffix dump — thinking suffix ends `[…154828, 154841]`,
no-think suffix ends `[…154828, 154841, 154842]`, i.e. exactly `chat_template.jinja:118`.

**`raw-chat-ids` and `chat-thinking` consume the same rendered input IDs and produced different
streams** — 4 tokens without `154842` vs 8 tokens with it, temperature 0. The model retrieved
`738216` in all five cells; only closing-tag emission varied.

**Determinism control:** `raw-no-think-ids` (cold) and `chat-no-thinking` (warm), different
endpoints, produced **byte-identical** ids `3deb38ee3d70`. The no-think path is stable across
cold/warm and across endpoints.

### 2.4 Finalization mechanism — source-located

`vllm/parser/glm47_moe.py:125`
```python
initial_state = ParserState.REASONING if thinking else ParserState.CONTENT
```
Only `REASONING → CONTENT` transition is terminal `THINK_END` (`</think>`). Generation ending
without that literal leaves `content=None` permanently, regardless of correctness or length.

`/model/chat_template.jinja:118` ends the assistant turn with a bare unclosed `<think>` unless
`enable_thinking` is defined and false — so generation always begins inside a think block.

`glm47_moe.py:184-192` derives `thinking_enabled` from only `thinking` / `enable_thinking`. It
cannot see `reasoning_effort`, which is what the server sets via `--default-chat-template-kwargs`.

Matched cells at 250k (`cn4:/home/derek/thinkmode-250k-20260726T0105Z`):

| kwargs | verdict | out | cached |
|---|---|---|---|
| `{}` | `REASONING_ONLY`, `content=None`, reasoning=`738216` | 4 | **0 cold** |
| `{enable_thinking:false}` | **EXACT** `content='738216'` | 4 | **0 cold** |
| `{thinking:false}` | `IN_CONTENT` 80 chars | 18 | 244,992 warm |
| untemplated raw | **ABSENT**, `finish=length`, degenerate repetition | 2000 | 67,584 warm |

`thinking:false` is read by the **parser** and ignored by the **template** (which checks only
`enable_thinking`), yet content populated — so **parser `initial_state` alone is sufficient** to
recover content; the closed-pair render is not required. Minimal forward fix is therefore
parser-side, which preserves thinking. **No fix applied or proposed.**

**Second template defect:** `chat_template.jinja:2`
```jinja
{%- set effective_reasoning_effort = 'high' if reasoning_effort is defined and reasoning_effort == 'high' else 'max' -%}
```
Only literal `'high'` is recognized; `low`, `medium`, unset all render **`max`**. Every cell
labelled "low" in tonight's logs ran at maximum effort. Only two effort states are reachable.

### 2.5 Prompt-structure verification (harness cleared)

`cn4:/home/derek/verify_prompt.py`. Passing and failing 350k prompts:

| seed | ctx | needle count | token offset | depth |
|---|---|---|---|---|
| `20260725` | 343,721 | 1 | 137,496 | 40.0% |
| `1110001` | 343,721 | 1 | 137,496 | 40.0% |
| `1110002` | 343,721 | 1 | 137,496 | 40.0% |

Identical structure; only ~90 characters of random header text at position 0 differ. **No harness
fault.**

### 2.6 Config diff, v19 (passes 350k) vs running v20 — both sides read live

**Identical**, therefore excluded: TP4/DCP4, `dcp-comm-backend=a2a`, interleave 1,
`kv-cache-dtype=nvfp4_ds_mla`, `attention-backend=B12X_MLA_SPARSE`, `moe-backend=b12x`,
`quantization=nvfp4_nf3_hybrid` **and the full quantization-config ignore list**,
`fuse_allreduce_rms`, GMU 0.97, `max-model-len=480000`, `max-num-seqs=16`, graph 64,
chunked-prefill, prefix-caching, async-scheduling, `reasoning-parser=glm45`,
`reasoning_effort:high`, `KV_FP8_ROPE=1`, `VLLM_PCIE_DMA_FP8=0`, `B12X_PCIE_DMA_FP8=0`,
`VLLM_DCP_GLOBAL_TOPK=1`, `VLLM_DCP_SHARD_DRAFT=1`, `VLLM_USE_B12X_SPARSE_INDEXER=1`,
`VLLM_DCP_PROJECT_BEFORE_MERGE=1` + `MIN_PREFILL_TOKENS=1024`, `VLLM_B12X_MLA_CKV_GATHER=1`,
`VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=1`, and **`--hf-overrides` byte-identical**
(`use_index_cache:true`, same 80-char `index_topk_pattern`) and **`--speculative-config`
byte-identical**.

That kills a sparse-indexer *configuration* hypothesis: selection config is the same in the working
and broken version.

**Differences:**

| # | setting | v19 (works) | v20 (fails) |
|---|---|---|---|
| 1 | `VLLM_DCP_QUERY_SPLIT` | **0** | **1** |
| 2 | `--max-num-batched-tokens` | **3072** | **2048** |
| 3 | `VLLM_DCP_A2A_MAX_TOKENS` | 64 | 16 |
| 4 | `VLLM_B12X_ABSORB_BMM` | absent | **1** |
| 5 | `VLLM_DCP_TOPK_OWNER_MERGE` | absent | 1 |
| 6 | `VLLM_B12X_MLA_CKV_PREFETCH_DEPTH` / `_WORKSPACE_MIB` | absent | 1 / 1024 |
| 7 | `VLLM_DCP_INDEXER_SHARDS` | absent | 0 |
| 8 | `B12X_MLA_SM120_UNIFIED` | **1** | absent |
| 9 | `B12X_DENSE_SPLITK_TURBO` | **1** | absent |
| 10 | `B12X_MOE_FORCE_A16` | **1** | absent |
| 11 | `PYTORCH_CUDA_ALLOC_CONF` | expandable_segments:**False** | :**True** |
| 12 | `--load-format` | safetensors | **instanttensor** |
| 13 | `--kv-transfer-config` | OffloadingConnector 64 GB DRAM | none |
| 14 | `NCCL_P2P_LEVEL` | SYS | PXB |
| 15 | vLLM / SparkInfer | 7ea567a / 4cfa530 | 5517197 / be0edcaa |

**Rows 1+2 are coupled and v19 documents the coupling.** v19 compose line 74:
`VLLM_DCP_PROJECT_BEFORE_MERGE=1   # dcp4 prefill workspace (needs batched=3072)`. v20 runs
`PROJECT_BEFORE_MERGE=1` at `batched=2048` — below v19's documented requirement — **and** flips
`QUERY_SPLIT` 0→1. Both are in the DCP long-prefill path where a 343k prompt is processed. Neither
came from my compose: the launcher's `glm52-dcp-prefill-policy.sh` auto-policy selected them.

**Row 4 is your `dev#129` target** — `VLLM_B12X_ABSORB_BMM=1` is v20-only and is one of the two
stacked v20-only generated-token paths you named. Independent convergence.

**Caveat, un-eliminated:** v19 and v20 are different vLLM/SparkInfer commits, so some rows are
*consequences* of the version change rather than independent choices. Rows 8–10 may be flags
`5517197` no longer reads. **I have not verified which of these v20 actually honors**, and none
should be treated as a lever until that is checked.

---

## 3. RETRACTIONS — claims I made and withdrew

1. **"ROOT CAUSE: parser/template"** (`dev#135`) — **RETRACTED** in `dev#138`. Your byte comparison
   showed parser, template, protocol, serving, renderers, tokenizer and generation config
   SHA-identical across v19/v20. v19 passes with the same fragile parser. `dev#135` is the
   *symptom mechanism*, not the regression.
2. **"raw `/v1/completions` retrieves cleanly at 350k and 450k"** (`dev#139`) — **RETRACTED** in
   `proofs#143`. Single seed. A later 250k raw cell failed with degenerate repetition. Untemplated
   raw is a fragile probe; the individual measurements stand, the generalization does not.
3. **"Hold the legacy projection candidate"** — **WITHDRAWN** in `dev#144`/`dev#145`. The evidence
   now favours it as the right lead.
4. **"`enable_thinking:false` is a configuration fix"** — **RETRACTED**. It fixed 250k; at 350k it
   is 0/3 and fabricates.
5. **"Depth is not the variable"** (`dev#145`) — **NARROWED** per your `dev#146`. 350k and 450k used
   different documents, so the table invalidates depth-only bisection but does not prove depth
   irrelevant.
6. **"Perturbation sensitivity proves numerical instability"** — **NARROWED** per `dev#146`. The
   ~90-char header and the open/closed think suffix are semantically meaningful to an
   autoregressive model; different outputs prove *prompt sensitivity*, not numerical instability.
7. **`KV_FP8_ROPE=1` as a v19/v20 difference** — killed before sending. v19 prod sets it too
   (`deploy/glm52-prod-v19/docker-compose.yml:46`).
8. **Sparse-indexer config as a difference** — killed before sending, see §2.6.

---

## 4. NOT COMPLETED — open items, and one I owe you

### 4.1 Exact-prompt reproducibility — **STOPPED BY ME, unfinished**

`harness/v20_prompt_reproducibility_probe.py` (sha256
`3476fb7a1cd1779d06c5378e95cefdc38f0954518ca548e4330d64f9ed914f11`) ran **1 of 9 cells**
(`rep1 seed=20260725 ABSENT cached=0`) and I killed it. Derek challenged whether re-characterizing
a known failure on a stock image was a good use of the box; I agreed and stopped it. Partial output
`cn4:/home/derek/repro-350k-20260726T0320Z`.

**This is the input to your `dev#147` step 1, and I removed it. Two design flaws in what I
built, so restarting it as written would not have satisfied you either:**

- It ran three seeds at **350k on the `nothink` arm only** — a combination now 0/4. **There was no
  PASS in it to freeze.** Your step 1 needs a stable FAIL *and* a stable PASS.
- Its docstring labelled seed `20260725` "known passing at 350k", but that pass was on the **raw**
  endpoint, not chat-nothink. Not a matched reference. Docstring is wrong and I have not yet fixed it.

**Corrected design I propose instead** (not run, awaiting Derek):
freeze the pair across depths on one arm — **stable FAIL = a 350k `nothink` prompt** (0/4),
**stable PASS = a 450k `nothink` prompt** (3/3) — and alternate those two exact prompts A,B,A,B,A,B
×3 each. Alternation forces eviction because two ~350–450k prompts exceed the 559,616-token pool;
`cached_tokens` recorded per cell to prove coldness rather than assume it. 6 cells, ~1 h. Output
would be exactly your requested freeze: serialized chat input IDs, prompt SHA256, sampling fields,
expected token stream.

### 4.2 `proofs#134` stage-decomposition probe — **ACKED, NOT RUN**

`harness/v20_nf3_absorbed_projection_probe.py`, sha256 `ecd99b8890c8d8…`. GPUs are free now. Not
started because Derek asked me to pause for review. Ready on request.

### 4.3 Config levers from §2.6 — **NONE TESTED**

No lever from the diff has been A/B'd. Specifically untested: `VLLM_DCP_QUERY_SPLIT=0`,
`--max-num-batched-tokens=3072`, `VLLM_B12X_ABSORB_BMM=0`, `B12X_MOE_FORCE_A16=1`,
`--load-format=safetensors`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. Each needs a
reboot. **Also not verified: which of rows 8–10 `5517197` still reads.** That check is free and
should precede any boot.

### 4.4 Other incomplete items

- **`deploy/glm52-v20-5517197-nf3-flat.yaml`** — single-file no-launcher compose per Derek's
  standing preference. Written, **never booted**. Unvalidated.
- **250k `chat-default` cold cache state** — the 250k cell in
  `thinkmode-250k-20260726T0105Z` is cold (`cached=0`), but the *earlier* 250k finalization run
  predates that recording; check its `summary.json` before quoting it as cold.
- **Deep cells in `deepraw-{350000,450000}`** — `cached_tokens` was 97,792 and 137,216. **Not cold.**
  Needle offsets ~137k / ~176k are past the cached prefix, but do not log these as cold runs.
- **v19 350k PASS is quoted from the library, not re-measured this session.** I did not re-verify it
  on CN3 — prod was never taken down, per Derek's constraint. Treat the v19 reference as historical.
- **No 475k cell** run this session.
- **Stale hardcoded message** in `~/ladder.sh` on CN4 still says "PEDANTIC does not fix retrieval".
- **Nondeterminism source not isolated** — cold/warm agreement on the no-think path argues against
  prefix-cache perturbation, but the mechanism behind variable `154842` emission is unknown.

---

## 5. What I believe, with confidence levels

- **High:** 350k fails cold on stock v20, 0/7 across four documents and both arms. Not a parser
  artifact — `nothink` returns populated, fabricated content.
- **High:** the `glm47_moe` parser converts a missing `</think>` into `content=None` with no
  end-of-generation fallback, and `154842` emission is variable for identical input IDs. This
  destroys genuine successes at depth (450k rep3) and biases any pass-rate collected through chat
  with thinking on.
- **High:** retrieval reached 441,951 tokens correctly (450k `nothink` 3/3), so the KV/attention/
  transport/MTP stack is not categorically broken at length.
- **Medium:** the trigger is sequence content rather than length alone. Supported by identical-
  structure 350k prompts diverging, and by 450k passing where 350k failed. Weakened by your
  `dev#146` point that these are semantically meaningful prompt changes.
- **Low / unsupported:** any specific mechanism. I have no evidence isolating query BMM, value BMM,
  fusion, DCP query-split, or allocator behaviour. §2.6 rows 1+2 and 4 are the only leads I would
  defend, and only as leads.

Parser/template work remains a separate PR and cannot satisfy retrieval — agreed.
