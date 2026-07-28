# Long-context retrieval failure on GLM-5.2 / vLLM v20 — facts, trials, and detection lessons

Prepared for external replication. Companion script: `needle_hunt.py`.

Everything in §1–§6 is measured. §7–§8 separate **measurement** from **inference** explicitly —
where we say "we think," we have not proven it.

---

## 1. System under test

| | |
|---|---|
| Hardware | 4× NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB, PCIe, **no NVLink** |
| P2P | native, all-to-all `OK` (`nvidia-smi topo -p2p r`); no `ForceP2P` override configured |
| Model | GLM-5.2 MXFP8-NVFP4-NF3-Hybrid (753B MoE), `nvfp4_nf3_hybrid` |
| Parallelism | TP=4, DCP=4, `--dcp-comm-backend=a2a`, `--dcp-kv-cache-interleave-size=1` |
| Speculative decode | MTP, `num_speculative_tokens=3` |
| KV cache | `nvfp4_ds_mla` + `KV_FP8_ROPE=1` → **368 B/token** compact record (~134k tokens/GiB) |
| Context | `max_model_len=480000` |
| Scheduler | `max_num_seqs=16`, `max_num_batched_tokens=3072`, chunked prefill, prefix caching on |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes `[1,2,4,8,16,24,32,40,48,56,64]`, cap 64 |
| Wire | block-INT8 PCIe DMA (`VLLM_PCIE_DMA_FP8=i8_ring`), DMA threshold 6,291,456 B |
| Offload | `TieringOffloadingSpec`: 64 GB DRAM + bounded 8 GiB NVMe fs tier |
| Baseline | prior release ("v19"), same geometry, **2d16h in production, RestartCount 0** |

Two engine generations are compared throughout: **v19** (working baseline) and **v20**
(vLLM base commit `3e731bc0`).

---

## 2. What we set out to do, and what happened

Deploy v20 with four local patches. Twelve boots over ~9 hours. Two distinct problems surfaced:

- **Problem A — boot crash.** v20 could not complete CUDA graph capture. Solved.
- **Problem B — long-context retrieval loss.** Real, still open at time of writing.

A third issue nearly derailed both: **our own test harnesses produced false results for hours.**
That is the most transferable lesson here and is covered in §6.

---

## 3. Problem A — CUDA illegal memory access during graph capture

### 3.1 Symptom

Every boot died ~16 minutes in, all 4 workers, with an identical stack ending at:

```
vllm/v1/worker/gpu/input_batch.py:151  make_dummy
    torch.from_numpy(num_scheduled_tokens).to(device=device)
→ torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

That line is a **synchronizing H2D copy**. It is where the async fault surfaced, not where it
was launched. Chasing it directly wasted several boots.

### 3.2 Trials that did NOT fix it

| Lever | Values tried | Result |
|---|---|---|
| GPU memory utilization | 0.970 / 0.978 / 0.980 | same fault (or clean OOM-fit failure at 0.970) |
| Available KV memory | 3.25 / 4.13 / 4.14 / 4.19 / 4.41 GiB | same fault |
| CUDA graph capture cap | 64 → 32 | same fault |
| `max_num_seqs` | 16 → 8 | same fault |
| CUDA-graph memory-pool lifetime fix | applied | same fault |
| DCP A2A route (`VLLM_DCP_A2A_MAX_TOKENS` 64→16, forcing AG/RS) | applied | same fault |
| Indexer block-table alignment (1875→1876 columns) | applied | same fault |

**Key discriminator we should have used sooner:** the error was always
`cudaErrorIllegalAddress`, **never** `cudaErrorMemoryAllocation`. Memory exhaustion raises OOM.
An out-of-bounds access does not become an OOM under a tighter budget. Roughly four boots were
spent on memory-shaped hypotheses that this single fact ruled out.

### 3.3 What did work — two-stage localization

**Stage 1 — descriptor instrumentation.** A patched `cudagraph_utils.py` synchronized at each
safe capture boundary (`warmup_inputs`, `warmup_forward`, `fresh_capture_inputs`,
`piecewise_capture`, `b12x_prewarm`, `full_capture`, `manager_exit`) and emitted a structured
record per descriptor. One boot, 1,162 records, exactly one failing tuple:

```
label=capturing_decode_cuda_graphs  stage=warmup_forward  cg_mode=FULL
num_tokens=9  num_reqs=9  uniform_token_count=1
```

This immediately killed the large-batch hypotheses: the failure was at **9 tokens**, the 8th of
16 descending descriptors — the *small* end. Every previous fix attempt had targeted the large end.

It also produced the critical A/B: **the identical descriptor passed all five stages during the
profiling pass and failed during the production pass.**

**Stage 2 — synchronous launch.** One boot with `CUDA_LAUNCH_BLOCKING=1` (and nothing else
changed) moved the traceback from the H2D copy to the true launch site:

```
speculator.py:125 capture → cudagraph_utils.py:531 forward_fn(CUDAGraphMode.NONE)
 → speculator.py:567 _generate_draft → speculator.py:438 _run_model
 → deepseek_mtp.py:676 forward                        ← MTP DRAFT MODEL
 → mla_attention.py:2017 unified_mla_attention_with_output
 → mla_attention.py:1214 forward_impl
 → torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)
```

Zero MoE frames in the log — which falsified a plausible competing hypothesis about MoE
shared-expert stream overlap that had a patch already written.

### 3.4 Root cause and fix

`mqa_q_nope` is a **split-and-transpose view** and therefore non-contiguous. A prior change had
removed the contiguity protections around the query-absorption BMM (they were assumed unnecessary
after an unrelated head-major DCP change; that reasoning held for the V-up path but not for query
absorption). cuBLAS read-ahead on a non-contiguous operand crossed an unmapped tile boundary.

Fix: materialize `mqa_q_nope` immediately before the BMM and enforce contiguous absorbed weights.
~29 lines. First boot after the fix: **624/624 capture boundaries passed, M=9 clean in both
passes, API healthy, RestartCount 0.**

**Why profiling passed and production failed** (inference, not proven): the two passes allocate in
different memory neighbourhoods, so the same strided read only crosses an unmapped page in one of
them. This is the classic signature of a latent OOB that appears configuration-dependent.

---

## 4. Problem B — long-context retrieval loss (open)

### 4.1 Measured result, cold, unique prefix per depth, corrected harness

Needle at 40% depth; `cached=0` on every row (verified cold, not prefix-cache hits):

| Context | Needle found | Field | Non-empty `content` | completion tok |
|---|---|---|---|---|
| 50,000 | ✅ | `content` | yes | 90 |
| 150,000 | ✅ | `content` | yes | 80 |
| 250,000 | ✅ | **`reasoning`** | **no** | 18 |
| 300,000 | ✅ | **`reasoning`** | **no** | 18 |
| **350,000** | ❌ | — | no | 88 |
| **475,000** | ❌ | — | no | 171 |

Baseline for comparison: v19 on identical hardware/geometry passed 50k/200k/300k/350k/475k.

### 4.2 Two independent defects, not one

1. **Retrieval failure ≥350k.** Genuine. The model emits coherent prose that does not contain the
   value. `finish_reason=stop`, not truncated.
2. **Finalization failure ≥250k.** `content` is `null` while the answer sits in `reasoning`.
   Retrieval is *fine* here — but every ordinary API client sees an empty response. This is a
   production defect in its own right and is easy to misread as a retrieval failure.

### 4.3 Positional probe at constant 350k

Same context size, needle moved through the window, three cold runs:

| Needle position | Retrieval | Finalization | Observed behaviour |
|---|---|---|---|
| 20% | MISS | PASS | clean refusal: *"not available in the provided document"* |
| 40% | MISS | FAIL | **confabulation** — answered **"27"**, lifted from "Facility **27**" in the question |
| 80% | MISS | FAIL | **degenerate loop** — `finish_reason=length`, all 3,000 tokens spent echoing filler |

**Position-independent failure with three different failure modes.** This is the shape of a broken
kernel/route, not a top-k selection cliff (which would be position-sensitive).

### 4.4 Current leading hypothesis (NOT yet proven on hardware)

v20 commit `3e731bc0` changed the MTP verifier default from
`VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE="0"` (extend kernel) to `"auto"`, routing genuine multi-token
verifier batches to a flattened **split-K decode** kernel. That commit's qualification tests only
`fp8_ds_mla`. Production here uses `nvfp4_ds_mla` — a different record layout (368/432 B E2M1+E4M3),
different scale format, and a different (BF16-QK) math arm. There is no compact-NVFP4 verifier
qualification.

Supporting evidence:
- Ruled out by AST-level comparison (namespace-normalized): compact-NVFP4 KV writer/reader
  primitives, paged-indexer metadata prep, and the extend kernel are **identical** between v19 and
  v20. Byte corruption is not the mechanism.
- DCP4 position mapping proven bijective over `[0, 475000)`.
- Long-context top-k fold proven correct at production geometry (k=2048, 118,750 rows, 32,768-row
  supertiles).
- A separate experiment forcing **one FP32 accumulation split** for verifier batches moved the
  failure boundary outward (~300k → ~350k) but did not restore correct behaviour — and cost 32,256
  KV tokens. BF16 split partials are a real numerical defect but not the whole story.

**Status: unproven.** The prediction that discriminates it cheaply is that reverting compact-NVFP4
to the extend route should *increase* the KV pool (it removes a verifier-only scratch
over-reservation of ~99 MiB/GPU), visible ~16 minutes into a boot, long before any needle runs.

---

## 5. Boot ledger (condensed)

| # | Distinguishing change | Avail KV | KV pool | Outcome |
|---|---|---|---|---|
| 1 | baseline, GMU 0.970 | 3.25 GiB | ~436k est | clean fit failure (`ValueError`) |
| 2 | GMU 0.980 | 4.14 GiB | 544,000 | illegal access |
| 3 | MNS 8 / cap 32 | 4.41 GiB | 592,640 | illegal access |
| 4 | + graph-pool reuse fix | 4.14 GiB | 555,520 | illegal access |
| 5 | + A2A cap 16 → AG/RS | 4.19 GiB | 562,432 | illegal access |
| 6 | + block-table alignment 1876 | 4.13 GiB | 554,496 | illegal access |
| 7 | + descriptor diagnostics | 4.13 GiB | 554,496 | **fault localized (M=9)** |
| 8 | + W4A16 coop grid + CKV reset | 4.14 GiB | 555,520 | illegal access |
| 9 | `CUDA_LAUNCH_BLOCKING=1`, GMU 0.970 | 3.19 GiB | — | fit failure; never reached capture |
| 10 | `CUDA_LAUNCH_BLOCKING=1`, GMU 0.980 | 4.14 GiB | 544,000 | **failing op named** |
| 11 | **+ query-BMM contiguity fix** | 4.15 GiB | 557,824 | ✅ **serving** |
| 12 | + forced one-split verifier | 3.91 GiB | 525,568 | serving; boundary moved, not fixed |

Note boot 9: `CUDA_LAUNCH_BLOCKING` was combined with a lower GMU, and the run died at KV
allocation before reaching the fault — a wasted boot. **Change one variable.**

---

## 6. Harness defects — the expensive lesson

Two bugs in our own tooling produced hours of false conclusions and drove at least one patch that
was justified by phantom evidence.

### 6.1 Wrong response field → false MISSES

The harness read `message["content"]` and `message["reasoning_content"]`. This engine emits
**`message["reasoning"]`**. When the model answered with `content: null` and the value in
`reasoning`, the harness saw two empty strings and reported MISS.

Impact: **250k and 300k were reported as retrieval failures when the model answered correctly.**
This produced a false "regression starts at 300k" bracket that stood for hours and shaped the
diagnosis.

Detection: the tell was `completion_tokens=17–18` with empty content. Tokens were generated and
went *somewhere*. Any nonzero completion count with no visible text means you are reading the
wrong field.

**Fix:** search `content`, `reasoning`, `reasoning_content`, `thinking`, then the whole serialized
message as a backstop.

### 6.2 Prefix caching → false PASSES

Repeating identical prompts hit the server's prefix cache. A warm run at 475k **passed**; the same
depth cold **missed**. Warm runs completed in 3–6 s versus minutes, with `cached_tokens` ≈ full
prompt length.

**Fix:** unique prefix per depth per run, and print `cached_tokens` so a warm run can never be
mistaken for cold.

### 6.3 Two other traps worth naming

- **Reasoning budget starvation.** A liveness check with `max_tokens=64` returned `content: null`
  because the reasoning trace consumed the entire budget. Identical signature to a real failure.
  Use a generous budget and treat `finish_reason=length` as "inconclusive," never as "miss."
- **Buffered output.** Running a multi-depth ladder as one process with stdout redirected means
  Python block-buffers; nothing appears until the run ends. A 30-minute ladder gives no chance to
  stop early. Use `python3 -u` and one invocation per depth.

### 6.4 A monitoring analogue of the same class of bug

A capacity monitor for the bounded NVMe cache reported the configured limit exceeded by up to
0.181%. It was wrong: it scanned a directory tree recursively **while eviction and replacement were
in flight**, counting a file and its replacement in the same pass. Evidence it was an artifact:

- every reported value was an exact filesystem-block multiple (1,049 / 1,050 / 1,051 blocks)
- the monitor **undercounted far more often than it overcounted** (48 samples below the stable
  count vs 5 above) — a real over-commit produces only overshoots

Replacing it with an **inotify ordered-event monitor** (tracking completed + in-flight temp bytes)
gave the definitive answer: **41,334 events, 0 violations, high-water 823,296 B under the cap.**

Lesson: a non-atomic observer of a mutating tree cannot prove a bound. Use kernel event ordering.

---

## 7. Detection methodology — what to reach for, in order

1. **Read the error class before theorizing.** `cudaErrorIllegalAddress` ≠ `cudaErrorMemoryAllocation`.
   Illegal address is never a memory-budget problem.
2. **Distrust the reported line for async faults.** A traceback ending at a synchronizing copy
   (`.to(device)`, `.item()`, `synchronize()`) tells you where it surfaced, not where it launched.
3. **Instrument boundaries before you guess.** Structured per-descriptor records with a synchronize
   at each safe boundary localized in one boot what six configuration boots could not.
4. **Then use `CUDA_LAUNCH_BLOCKING=1`, alone.** It names the kernel. Do not combine it with any
   other change; it also serializes launches and can mask concurrency faults.
5. **Exploit the pass/fail A/B.** "Same descriptor passes in profiling, fails in production" was
   worth more than any single log line.
6. **Validate the harness against a known-good case before trusting a negative.** A shallow-depth
   control (50k) run in the same session distinguishes "model failed" from "harness broken."
7. **Prefer a prediction that fails fast.** The current hypothesis predicts a KV-pool *increase*
   visible 16 minutes into a boot — far cheaper than a 30-minute needle ladder.

---

## 8. Failure-mode signature catalogue

| Signature | Likely meaning |
|---|---|
| `finish_reason=stop`, `completion_tokens` > 0, `content` empty | answer is in another field — check `reasoning` |
| `finish_reason=length`, budget exhausted | inconclusive; raise `max_tokens` and re-run |
| `cached_tokens` ≈ prompt length, seconds-fast | prefix-cache hit; not a cold measurement |
| Coherent prose stating the fact is absent | genuine retrieval miss |
| Answer is a plausible wrong number lifted from the question | confabulation — retrieval failed, model substituted |
| Output degenerates into repeated source text until `length` | severe long-context breakdown |
| Illegal access at a synchronizing copy | async fault launched earlier; instrument or use launch-blocking |
| Fault invariant across memory/graph/route levers | not a resource problem; look for an unqualified code path |

---

## 9. How to replicate

`needle_hunt.py` is stdlib-only — no pip installs, no dependencies. Copy it to any machine that can
reach the server's OpenAI-compatible endpoint.

### 9.1 Needle placement — read this first

Placement is **deterministic, not random, and not at the tail.** The needle is inserted at a fixed
fraction of the document, default **40%**:

```
--position 0.20  ->  needle at 20.2% through the document
--position 0.40  ->  40.2%   (default)
--position 0.80  ->  80.0%
```

There is no RNG in the script. Two runs with the same `--runtag` and `--position` build byte-identical
documents.

**A single fixed position is not sufficient to characterize a failure.** At a constant 350k context we
observed three *different* failure modes purely by moving the needle:

| Position | Behaviour |
|---|---|
| 20% | clean refusal — "not available in the provided document" |
| 40% | confabulation — answered "27", lifted from "Facility 27" in the question |
| 80% | degenerate loop — `finish_reason=length`, 3,000 tokens echoing filler |

Use `--sweep` to test several positions at every depth.

### 9.2 Quick start

```bash
# single depth, sanity check the plumbing (seconds)
python3 needle_hunt.py --base http://HOST:PORT --model NAME --depths 3000

# the standard ladder we ran
python3 needle_hunt.py --base http://HOST:PORT --model NAME \
    --depths 50000,150000,250000,300000,350000,475000 \
    --save-json ./needle-out
```

### 9.3 Position sweep (recommended when you find a failing depth)

```bash
python3 needle_hunt.py --base http://HOST:PORT --model NAME \
    --depths 350000 --sweep 0.2,0.4,0.8 --save-json ./pos-out
```

Every (depth, position) cell gets its own unique prefix, so all cells are cold.

### 9.4 Bisecting a failure boundary

```bash
python3 needle_hunt.py --base http://HOST:PORT --model NAME \
    --depths 250000,275000,300000,325000,350000 --stop-on-miss
```

`--stop-on-miss` exits at the first retrieval failure instead of burning 30 minutes finishing the
ladder. Output is unbuffered and prints per cell.

### 9.5 Reading the output

```
depth=300000 pos=40% ctx=299992 cached=0 completion=18 finish=stop secs=214
   retrieval=PASS (where=reasoning)   finalization=FAIL(content empty)
   content=''
   reasoning_tail='...the maintenance ticket number ... is 738216.'
```

| Field | Meaning |
|---|---|
| `ctx` | actual prompt tokens the server counted |
| `cached` | prompt tokens served from prefix cache. **Must be 0** for a valid cold measurement |
| `completion` | tokens generated. Nonzero with empty `content` means the answer is in another field |
| `finish` | `stop` = clean; `length` = budget exhausted, result is INCONCLUSIVE |
| `retrieval` | needle value present anywhere in the response message |
| `where` | which field held it: `content`, `reasoning`, `other_field`, or `-` |
| `finalization` | whether a usable non-empty `content` was returned |

**`retrieval=PASS, finalization=FAIL` is a real and distinct state**, not a rounding of either verdict.
The model found the answer but returned nothing an ordinary API client can consume. We spent hours
misreading this as a retrieval failure. Do not collapse it.

Exit codes: `0` all retrieved, `1` at least one miss, `2` request error.

### 9.6 Before you trust a MISS

Check all four:

1. **`cached=0`** — otherwise you measured the prefix cache, not the model. We had warm and cold
   disagree at 475k.
2. **`finish` is `stop`, not `length`** — if the budget ran out the result is inconclusive. Raise
   `--max-tokens`. At `--max-tokens 64` a reasoning trace alone can consume the whole budget and
   return `content: null`, which is indistinguishable from a real failure.
3. **A shallow control passed in the same session** — run `--depths 50000` alongside. If that also
   misses, your harness or endpoint is broken, not the model.
4. **The response text actually denies the fact** — a genuine miss produces coherent prose stating
   the value is absent. Empty output with nonzero `completion` is a field-name problem, not a miss.

### 9.7 Options reference

| Flag | Default | Notes |
|---|---|---|
| `--base` | `http://localhost:8000` | server base URL |
| `--model` | `GLM-5.2` | must match the served model name |
| `--depths` | `50000,...,475000` | approximate context sizes in tokens |
| `--position` | `0.40` | needle depth fraction; deterministic |
| `--sweep` | — | positions tested at every depth, e.g. `0.2,0.4,0.8` |
| `--max-tokens` | `3000` | keep generous; see §9.6 |
| `--effort` | `low` | maps to `chat_template_kwargs.reasoning_effort` |
| `--no-effort` | off | omit `chat_template_kwargs` if the template rejects it |
| `--runtag` | epoch seconds | mixed into every prompt to defeat prefix caching |
| `--stop-on-miss` | off | exit at first retrieval miss |
| `--save-json` | — | writes `request-<depth>-pos<NN>.json` / `response-…` per cell |
| `--timeout` | `1800` | per-request seconds; raise for very deep cold prefills |

### 9.8 Adapting the needle

`NEEDLE_VALUE`, `NEEDLE`, `FILLER` and `QUESTION` are module-level constants at the top of the file.
The filler is deliberately repetitive and semantically flat so the needle cannot be found by novelty
alone. If you change the needle, keep the value a distinctive digit string — the matcher strips commas
and does a substring search across the serialized message.

## 10. Open items

- Compact-NVFP4 MTP verifier route hypothesis — **unproven**, needs one GPU boot
- `content: null` above ~250k — mechanism unexplained; not a clean depth threshold (at 350k it
  varied with needle position)
- Whether the pre-fix build genuinely missed at 300k — its data came from the broken harness and
  was never re-measured cold
- NVMe restart persistence / tier promotion — never exercised
- Long-context GPU tests in the upstream tree reach ~32k; production DCP-local widths here are
  ~87.5k at 350k and ~118.75k at 475k. **That gap is where this defect lived.**
