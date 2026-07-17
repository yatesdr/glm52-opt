# GLM-5.2 prefill breakthrough: findings and proposal

**Date:** 2026-07-17  
**Target:** 4x RTX PRO 6000 Blackwell on cn3, TP4/DCP4, PCIe Gen3 + dual PLX  
**Scope:** read-only audit of the supplied spec, local source trees, public repositories, and cn3. No server or source changes were made.

## Decision summary

The best capacity-preserving cold-prefill path is a **packed-CKV gather / ownership inversion for DCP prefill**:

```text
current:  local Q heads -> absorb to 576 -> all-gather 64 heads
          -> attend against local CKV shard -> project -> LSE merge + reduce-scatter

proposed: local Q heads stay local -> all-gather one head-independent packed CKV stream
          -> attend local heads against gathered CKV -> local projection; no output RS
```

This is the mechanism most consistent with the July 16 community report that koush's custom “CKV-gather” path reached 3,200 tok/s at DCP8 on PCIe 5.0 x8 versus roughly 2,400 for the published path on x16. The implementation is not public, so its exact code and benchmark conditions cannot be independently verified. The algorithm, however, fits both the model geometry and cn3's measured bottleneck.

For cn3's 3,072-token chunks, the current FP8 query collective moves a logical 576-byte query for each of 64 heads. The proposed collective moves one existing 368-byte `nvfp4_ds_mla` record per context token. At a 55k late chunk this is about **5.6x fewer source bytes**, and across the whole 55k prompt it is about **10.5x fewer source bytes** because earlier chunks have shorter accumulated contexts. It also eliminates the 4.1 ms/layer output reduce-scatter.

Using cn3's measured layer ledger, the expected cold gain is **1.4–1.65x**, approximately **1,335–1,575 tok/s from the current 955**, with an absolute measured-ledger ceiling near **1,740 tok/s**. An especially good implementation may approach that ceiling at short/medium context. It cannot credibly produce 3,800–8,000 cold tok/s on cn3 without also changing the remaining compute/TP work.

The best immediate short-context configuration experiment is **DCP1**. It should deliver roughly **1.6–1.82x** on cn3, but cuts KV capacity to about one quarter. Public Gen5 TP4 measurements show the same trade: 4,163 tok/s DCP1 versus 2,341 DCP4, or 1.78x, with 185k versus 768k KV tokens. DCP2 is the useful middle point if its reduced capacity covers the workload.

**LMCache is a separate, production-level reuse optimization.** It can easily make effective prompt throughput appear 4x or more when contexts repeat, but it does not accelerate cold model computation. It should be evaluated after a paired cold/warm benchmark and after confirming support for this custom packed KV format and DCP layout.

The remembered 8k screenshot is not presently evidence of a four-GPU cold-prefill breakthrough. The only public 8k-class GLM-5.2 results I could verify are v17 **TP8/DCP1** FP8-DMA cells on eight GPUs (about 7.4–8.0k). David's published four-GPU v1.3/v1.4 exact-32k result remains 2,128–2,131 tok/s. Without the screenshot or raw benchmark record, its hardware, DCP, cache-hit state, and denominator cannot be determined.

## Evidence labels

- **Measured:** cn3 timings, checksums, source sizes, and profiler ledger supplied in the spec or rechecked read-only on cn3.
- **Repository fact:** directly present in a public or locally available source revision.
- **Community report:** Discord-derived statement preserved by the spec or the rtx6kpro daily summary; not independently reproducible from source alone.
- **Inference:** derived from measured timings, tensor geometry, or public results; presented with a falsifiable test.

## 1. v1.4 overlay audit

### 1.1 The five uncopied files

I compared the checkout at a local checkout of `davidsyoung/vllm-glm52` byte-for-byte with the extracted v1.3 image at an extraction of the v1.3 image. Four files are identical. The fifth differs only in comments.

| Uncopied file | v1.4 overlay / v1.3 bytes | Result | Prefill relevance |
|---|---:|---|---|
| `b12x/attention/indexer/paged.py` | 39,156 / 39,156 | SHA-256 identical: `5d342221...5ba5ff0e` | None; already in v1.3 |
| `vllm/distributed/device_communicators/cuda_communicator.py` | 27,846 / 27,846 | SHA-256 identical: `7b5ce536...2f31f264` | None |
| `vllm/distributed/parallel_state.py` | 87,851 / 87,851 | SHA-256 identical: `6eda80ac...59e245d` | None |
| `vllm/v1/attention/ops/common.py` | 18,908 / 18,908 | SHA-256 identical: `4a4d2107...1e13d746` | None |
| `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` | 87,512 / 87,597 | Four comment-only hunks | None; executable source is unchanged |

The `b12x_mla_sparse.py` edits remove four internal `kingdom(nvfp4_ds_mla)` labels and reword the surrounding comments. They do not alter a symbol, value, branch, call, or kernel.

The spec recorded a 90,612-byte v1.3 extraction for that module. The preserved extraction inspected on cn3 is 87,597 bytes with SHA-256 `761718b1...6b675eb`; the overlay is 87,512 bytes with SHA-256 `567a06e2...a85161a`. The live diff is definitive for the files now on cn3, and the older size should be treated as stale or from a different extraction.

**Conclusion:** the uncopied overlays contain no staged prefill implementation.

### 1.2 The 61 added lines in `mla_attention.py`

The complete v1.4-versus-v1.3 diff is one new method, `_v_up_proj_bmm_chunked`, added around line 1,649. It:

1. validates rank-3 shapes, BF16 dtype, common device, and contiguous `W_UV`;
2. sets a 144 MiB combined temporary budget for `[H,T,512]` input and `[H,T,256]` output scratch;
3. computes a maximum token count from that budget; and
4. loops over token slices, calling the existing `_v_up_proj_bmm` unchanged.

The method name occurs only at its definition in the entire v1.4 overlay tree. The actual projection path still calls `_v_up_proj_bmm`, not `_v_up_proj_bmm_chunked`. The query-gather and reduce-scatter seams are unchanged.

**Conclusion:** these 61 lines are an unwired memory-bounding helper. They execute zero times and cannot explain any throughput change. Public v1.4's two 32k measurements of 2,128 tok/s, versus v1.3's 2,131, confirm no measurable prefill change. See David's [v1.4 commit](https://github.com/davidsyoung/vllm-glm52/commit/cfe43b0348f405739728f98faf3115a82b16d6b7) and [benchmark record](https://github.com/davidsyoung/vllm-glm52/blob/v1.4/BENCHMARKS.md).

## 2. What the circulating patch set actually contains

The plausible “all the patches” set reconstructs as follows.

| Patch family | Status and evidence | What it changes | Expected prefill effect |
|---|---|---|---:|
| David v1.3 `fast647` workspace + guarded A2A | Public, shipped | Borrow B12X workspaces, small-row A2A, project before merge | Public isolated v1.3 A/B: +4.2% |
| David v1.4 Grid188/runtime work | Public, source only as of audit | Decode MoE and synchronization paths; the only MLA addition is dead | 0% measured at 32k |
| v17 PR #94 | Public | Generalizes workspace path/topologies; projects 512 -> 256 before RS | TP4/DCP4 +9.2% from its pre-PR baseline |
| FP8 PCIe DMA `ag`/`ring` | Public in v17; custom version already on cn3 | Quantizes large DCP query/output transport | Public TP8/DCP4 ring +11.8%; cn3 already rose 640 -> 955 through stronger FP8 gather + RS work |
| Atomics-free two-stage TP allreduce | Public v17 lineage | Replaces large NCCL collectives | Kernel claim 1.2–1.3x for the collective, not the model; cn3 full boot regressed to 609 |
| Quantized KV DCP context gather fix | Public branch/commit | Fixes dtype mismatch in vLLM's chunked-context cache gather | Correctness fix, not the reported new algorithm |
| Koush packed-CKV gather | Community report only | Inverts DCP ownership: move compressed CKV, retain local Q heads | Reported 3,200 vs ~2,400 at DCP8; highest-value lead |
| LMCache/DCP integration patches | Mentioned in the same Discord discussion | Restore reusable KV rather than recomputing prompt | Potentially very large warm-only gain; no public GLM patch identified |
| DFlash for GLM-5.2 | Aspirational Discord statement | Speculative generation, not cold prefill | No releasable path found |

The current public v17 implementation and measurements are documented in the [GLM-5.2 v17 page](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.2_v17.md), and its workspace generalization is [vLLM PR #94](https://github.com/local-inference-lab/vllm/pull/94). The July 16 community digest is the only public artifact I found naming the [custom CKV result](https://github.com/local-inference-lab/rtx6kpro/blob/master/daily-summaries/2026-07/2026-07-16.md).

No public David branch, tag, image, commit, or GitHub activity contains the new prefill WIP as of 2026-07-17. His public repository has only `main`, `release/v1.4`, and the older guarded-DCP branch. Therefore the precise David breakthrough cannot be determined from available sources.

## 3. The packed-CKV mechanism

### 3.1 Why it changes the byte equation

GLM-5.2 MLA has one head-independent compressed KV record per token, but 64 query heads. The current DCP algorithm keeps CKV sharded and sends absorbed queries to every CKV owner:

```text
Q collective per layer/chunk = query_rows * 64 * 576 * query_bytes
```

cn3 has already reduced `query_bytes` to one byte plus block scales. For a full 3,072-row chunk this is approximately 108 MiB of source query values per layer before collective ring factors and metadata.

The alternative leaves each rank's 16 query heads local and gathers the existing packed, head-independent CKV records:

```text
CKV collective per layer/chunk = accumulated_context_tokens * 368 bytes
```

At 55k context this is 19.3 MiB of source CKV, 5.6x below the 108 MiB query tensor. Across an entire prompt, with 3,072-token chunks:

| Prompt | Current Q source bytes/layer | Cumulative packed CKV source bytes/layer | Byte ratio |
|---:|---:|---:|---:|
| 8,192 | 288.0 MiB | 6.1 MiB | 47.1x |
| 55,000 | 1,933.6 MiB | 184.3 MiB | 10.5x |
| 65,536 | 2,304.0 MiB | 272.0 MiB | 8.47x |
| 480,000 | 16,875 MiB | 13,371 MiB | 1.26x |

The per-chunk crossover is around 308k accumulated tokens. Above that, a 3,072-row FP8 query is smaller than the entire packed context. The implementation should therefore choose the smaller transport dynamically: packed CKV below the measured crossover and current FP8 query AG/RS above it. The integrated packed-CKV traffic remains smaller through the 480k supported maximum, but the hybrid avoids slow late chunks.

### 3.2 Compute and memory consequences

The arithmetic does not multiply if work ownership is inverted correctly:

- Current: each rank evaluates 64 query heads against roughly one quarter of the selected CKV positions.
- Proposed: each rank evaluates its 16 local heads against all selected CKV positions.

Both are approximately 16 head-context equivalents per rank. The DSA indexer already returns global top-k logical token IDs and filters them into each rank's local cache slots. The new path must instead remap those logical IDs into a temporary gathered-CKV layout.

The existing BF16 query workspace is large enough in principle to be reused. A 480k packed context is about 168.5 MiB, while the current full 3,072 x 64 x 576 BF16 gathered-query tensor is 216 MiB. Exact workspace aliasing and B12X plan contracts still require validation.

No lossy conversion is needed: the collective can copy the existing 368-byte packed record. That makes the numerical operation equivalent apart from possible reduction-order differences. It is a substantially safer quality path than adding FP4 query quantization.

### 3.3 Expected gain

cn3's measured layer time is approximately:

```text
14.0 ms query AG + 4.1 ms output RS + 0.48 ms projection + 22.6 ms other
= 41.2 ms/layer/chunk
```

At a 55k late chunk, ideal CKV wire time scaled from the measured AG is about 2.5 ms. Allowing 1–3 ms for packing, reorganization, smaller-collective inefficiency, and index remapping gives roughly 26–29 ms/layer. That predicts **1.4–1.6x** at late 55k and somewhat more on early chunks. The no-DCP ledger ceiling remains about 1.82x.

On a Gen5 switch the stock collective is less punitive, so the expected relative gain is smaller: roughly **1.15–1.4x**. The only reported real measurement is koush's **~1.33x at DCP8**, on a different topology and an unpublished prototype; it is supporting evidence, not a portable benchmark.

### 3.4 Falsifiable predictions

A correct implementation must show all of the following:

1. The current ~14 ms query AG and ~4.1 ms output RS disappear from eligible chunks.
2. A new packed-CKV gather grows with accumulated context length, not `query_rows * 64`.
3. At 55k, its late-chunk collective should be materially below 7 ms/layer; a good implementation should be around 3–5 ms on cn3.
4. Per-rank sparse-attention arithmetic stays approximately constant rather than increasing 4x.
5. Cold 55k throughput rises at least 30%; otherwise packing/remapping or the attention plan has erased the byte advantage.
6. Needle, deep-needle, arithmetic, JSON, coherence, and a BF16-logit/KLD gate remain within the current accepted range.
7. For contexts above roughly 300k, a dynamic selector begins choosing query transport for late chunks.

### 3.5 Port surface and cost

Likely source seams in the v1.3/v17 lineages are:

- `vllm/model_executor/layers/attention/mla_attention.py`: replace the large-row `dcp_b12x_all_gather_heads` / ordinary `all_gather` decision with a CKV option and bypass LSE output merge/RS for it.
- `vllm/v1/attention/backends/mla/b12x_mla_sparse.py`: expose a plan that accepts local heads plus a transient gathered packed-cache view; remap global top-k indices to that view.
- `b12x/attention/indexer/paged.py` or its caller: preserve global logical top-k while selecting the alternate physical mapping. The indexer scoring algorithm itself should not change.
- B12X PCIe DMA: add a byte-preserving ring all-gather for packed records; no quantize/dequantize stage is required.

Estimated cost from scratch is **3–7 engineering days plus validation**. If koush or David supplies a compatible patch, a guarded port may be **1–2 days**. The main risks are packed page layout, DCP interleave, top-k physical index mapping, and scratch lifetime—not the collective itself.

## 4. DSA sparse prefill is already enabled

There is no dormant “turn on sparse attention during prefill” switch in the uncopied files.

Repository evidence:

- `_should_skip_index_topk` in `vllm/model_executor/models/deepseek_v2.py` says an `S` layer reuses the previous layer's top-k indices. `F` computes a fresh indexer. Thus `FFFSSS...` is an **indexer-computation/reuse schedule**, not dense versus sparse attention.
- `b12x_mla_sparse.py` takes the shared `topk_indices_buffer`, converts global DCP winners to physical cache slots, binds them as `selected_indices`, and calls the sparse MLA extend/prefill kernel.
- The B12X extend path is explicitly described as single-pass prefill and consumes `selected_indices` plus per-token valid counts.
- `paged.py` has the large-row tiled indexer path. It accelerates selecting the 2,048 positions; it is not the main attention over all context.

**Prediction:** on `F` layers a profiler sees indexer kernels followed by sparse MLA prefill; on `S` layers the indexer kernels disappear but the same sparse MLA prefill remains and consumes the reused 2,048-wide index buffer. If `selected_indices` is absent or attention reads the full context on an `S` layer, this conclusion is wrong.

The remaining 55% “other” time can still contain indexer, sparse attention, MoE, and TP collective optimization opportunities, but converting dense attention to sparse is not one of them. Forcing dense attention would more likely slow long context and change model semantics.

## 5. Other byte-reduction theories

### 5.1 Gather pre-absorption queries

Before MLA absorption a head has 192 no-PE + 64 RoPE values, versus 512 + 64 afterward.

- BF16 pre-absorption gather: 512 bytes/head versus current FP8's roughly 576 bytes/head. This is only an ~11% byte reduction.
- FP8 pre-absorption gather: roughly 256 bytes/head, a 2.25x reduction versus current FP8 query values.

Every rank would then need all heads' `W_UK` and would perform the absorption BMM for 64 heads rather than its local 16. The gathered weights are only about 12 MiB/GPU, but the extra BMM may erase part of the wire saving.

**cn3 expectation:** 1.1–1.22x; **Gen5:** 1.03–1.1x.  
**Cost:** medium, 2–4 days with a standalone BMM/collective microbenchmark first.  
**Prediction:** gather falls from 14 ms toward 6–8 ms, while absorption BMM increases; a net layer saving under 3 ms kills the full port.

### 5.2 FP4 current-query transport

Quantizing the existing 576-wide query from FP8 to a block-scaled four-bit format can at best halve the 14 ms gather before metadata and kernel costs. If it saved the full 7 ms, the layer ledger improves only about 1.2x. It adds a new quality risk on the attention query and requires a fast SM120 dequant/input path.

**cn3 expectation:** 1.1–1.2x; **Gen5:** 1.02–1.08x.  
**Cost:** high relative to gain.  
**Prediction:** if a wire-only synthetic FP4 ring does not beat the present FP8 path by at least 5 ms at 108 MiB logical input, stop.

### 5.3 Gather only top-k winners

The indexer reduces KV positions, not query rows or query heads. Nearly every query needs winners from several DCP ranks. Sending a query only to ranks with winners therefore approaches the current all-rank behavior.

Gathering CKV independently for every query/top-k pair is much worse: `3,072 * 2,048 * 368` is over 2 GiB per layer before duplicates, versus 108 MiB for the current FP8 query tensor. Deduplicating winners across all rows tends toward gathering the full context once—which is exactly the packed-CKV proposal.

**Conclusion:** “post-indexer queries” and per-query winner transfer are not separate high-value paths. Full-context packed CKV with reuse across query rows is the useful form.

### 5.4 Workspace, projection, and overlap

The v1.3 workspace path and project-before-merge are already in the current stack. v17 PR #94 shows the remaining generalization is worth roughly 4–10%, not 4x. Complete overlap cannot exceed cn3's measured 1,450 tok/s gather-hidden ceiling, and eliminating all DCP communication cannot exceed about 1,740 without reducing the 22.6 ms remainder.

## 6. LMCache and benchmark contamination

LMCache stores reusable KV outside the engine's HBM cache and restores it for future requests, so matched prompt segments skip model computation. That is valuable for RAG, repeated documents, agents, and multi-turn sessions. It must be reported as **cache-effective prompt throughput**, not cold-prefill compute.

Official LMCache documentation explicitly constructs warm/replay workloads and reports lower TTFT because repeated KV is restored rather than recomputed. See the [benchmarking guide](https://docs.lmcache.ai/getting_started/benchmarking.html), [benchmark workload definitions](https://docs.lmcache.ai/cli/bench.html), and [integration description](https://docs.lmcache.ai/developer_guide/integration.html).

David's public v1.4 has no LMCache connector or overlay. The identical 15.4-second 32k repeats are inconsistent with a substantial full-prefix hit, so the published 2,128 result behaves like cold computation. However, the public record does not include the raw prompt-generation harness or cache counters. The label “exact-32k repeats,” with prefix caching enabled, is not by itself a cold attestation.

cn3's `prefill_bench.py` is a good cold harness: it puts `os.urandom(8).hex()` in the first prompt block. Because prefix-cache hashes are chained, that makes the entire block sequence unique on every run.

### Required paired protocol

For every future result, scrape metric deltas immediately before and after the request:

- `vllm:prefix_cache_queries`
- `vllm:prefix_cache_hits`
- `vllm:external_prefix_cache_queries`
- `vllm:external_prefix_cache_hits`
- `vllm:prompt_tokens_cached`
- `vllm:request_prefill_time_seconds_sum`

These are defined in current vLLM's [V1 metrics logger](https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py).

Run two explicitly named cells:

1. **Cold compute:** unique random first full cache block on every request; external cache disabled or empty. Require cached-token delta = 0. Report `actually computed tokens / prefill time`.
2. **Warm reuse:** byte-identical prompt replay with local APC, then with LMCache if available. Report `prompt tokens / TTFT`, cache-hit tokens, restore bytes/time, and the uncached tail separately.

If an 8k result appears only in cell 2 and cached tokens rise by most of the prompt, it is a reuse breakthrough. If it survives cell 1 with zero hits and normal layer/chunk counts, it is a cold-compute breakthrough.

Before porting LMCache, test ordinary vLLM automatic prefix caching. Exact repeats that fit in the current HBM pool do not need LMCache. LMCache becomes valuable when the working set exceeds HBM, survives restarts/instances, or reuses nonresident prefixes. The custom 368-byte record and DCP sharding need explicit connector support; no public GLM-5.2 patch proving that support was found.

## 7. Ranked candidates

Ranks prioritize cn3 impact while preserving correctness; the capacity caveat prevents DCP1 from being the general winner.

| Rank | Candidate | cn3 expected effect | Gen5 expected effect | Capacity / semantics | Port cost |
|---:|---|---|---|---|---|
| 1 | Packed-CKV gather with dynamic crossover | 1.4–1.65x; ~1.34–1.58k from 955 | 1.15–1.4x; community DCP8 report ~1.33x | Preserves sharded KV capacity; exact packed bytes | Medium-high |
| 2 | DCP1 short-context profile; DCP2 middle profile | DCP1 1.6–1.82x; DCP2 ~1.3–1.6x | Public TP4: DCP1 1.78x, DCP2 1.29x vs DCP4 | ~1/4 or ~1/2 KV capacity | Low/configuration |
| 3 | APC/LMCache for genuinely repeated contexts | Cold 1.0x; warm can exceed 4x | Cold 1.0x; warm restore benefits from faster I/O | Same math; requires hits and connector compatibility | Low for APC, medium-high for LMCache |
| 4 | FP8 pre-absorption query gather | 1.1–1.22x | 1.03–1.1x | Extra absorption BMM and replicated `W_UK` | Medium |
| 5 | FP4 query transport | 1.1–1.2x | 1.02–1.08x | New query quantization quality risk | High |
| 6 | v17 workspace/TP collective ports | 0–10% individually; cn3 v17 full stack regressed | Workspace 4–10%; TP kernel-specific | No model change | Medium; isolate patches |
| 7 | “Enable sparse prefill” | 0%; it is already active | 0% | Disabling it is a semantic/perf regression | None |

For a service that never needs more than about 25k context, DCP1 moves to rank 1 because it is already available and has the strongest evidence. For the stated 480k-capable four-GPU service, packed CKV is the best path.

## 8. Three one-boot confirm/kill tests

### Boot 1 — packed CKV prototype

**Prerequisite:** obtain koush/David's patch or implement a guarded runtime overlay with `DCP_PREFILL_TRANSPORT={query,ckv,auto}`. Do not alter the base image.

Boot the existing BLOCKS=400, MNBT=3072, TP4/DCP4 profile in `ckv` or `auto`. Run the current cold 8k and 55k harness, then the full quality gate. Record per layer: pack, CKV AG, sparse attention, projection, and any residual DCP merge.

**Confirm:** query AG and output RS are absent; late-55k packed gather <=7 ms/layer; 55k >=1,240 tok/s (+30%) with all gates passing. A strong result is 1,400–1,600.  
**Kill/redo:** CKV traffic matches query-sized traffic, attention work rises ~4x, RS remains, or throughput is below +15%.

### Boot 2 — DCP1 compute ceiling

Boot TP4/DCP1 with a deliberately capacity-safe max length/pool and the same MoE/KV/MTP settings. Run the cold exact 8k prompt and the existing profiler. No code changes are required.

**Prediction:** both DCP phases become zero and throughput lands around 1,500–1,740 tok/s.  
**Confirm:** >=1,450 with the rest-of-layer time close to 22.6 ms. This both validates the ledger and gives the real short-context profile.  
**Kill the 4x-cold theory:** a normal DCP1 layer still takes around 22 ms; then 3,800–8,000 cannot come from a DCP transport trick on cn3.  
**Follow-up:** if the workload needs roughly 25–50k, make the next boot DCP2 and keep DCP ranks within each PLX pair if group construction permits.

### Boot 3 — cold versus cache-effective prefill

Use an ordinary current-stack boot with prefix caching enabled. Capture the six metric deltas listed above.

1. Run three 32k prompts with a unique random first full block each time.
2. Run one 32k prompt three times byte-identically.
3. If a compatible LMCache connector exists, overflow/evict local APC and repeat from LMCache; otherwise stop after APC.

**Prediction:** unique prompts remain near the cold baseline with zero cached-token delta. Exact replays produce a large cached-token delta and can show apparent 8k+ throughput.  
**Confirm reuse explanation:** only the replay cell is fast and model layer/chunk invocations fall in proportion to cached tokens.  
**Confirm cold breakthrough:** the unique cell is also fast, cache deltas are zero, and all expected layer/chunk invocations occur.

## 9. Recommended execution order

1. **Ask for the artifact, not the headline.** Request from David/koush: commit or overlay, exact image digest, GPU count and PCIe topology, TP/DCP, MNBT, prompt length, cold/warm state, `/metrics` cache deltas, and raw TTFT/token counts. The custom CKV report is actionable; the unqualified 8k screenshot is not.
2. **Run DCP1 first.** It is a zero-development, one-boot validation of cn3's remaining compute floor and an immediately usable short-context profile.
3. **Port packed CKV behind a guard.** Reuse the existing 368-byte record, global top-k result, query workspace, and PCIe ring. Keep current FP8 query AG/RS as fallback and select by measured bytes/context.
4. **Do not spend another window on v1.4 overlay archaeology.** The exact audit is closed: no executable uncopied delta and no wired prefill addition.
5. **Treat LMCache as workload optimization.** Measure actual prefix reuse. Port it only if production hit rate and eviction behavior justify custom packed-KV/DCP connector work.
6. **Profile the 22.6 ms remainder after CKV.** Once DCP transport is reduced, split it into sparse attention, indexer, MoE, and the two TP allreduces. That next profile—not an assumed dense-attention switch—should choose the second kernel project.

For generation throughput, v1.4's Grid188, speculator bound, and synchronization changes are a separate decode-focused port candidate. They should be evaluated independently from prefill so a decode win cannot be misreported as evidence for the cold-prefill theory. No public DFlash-for-GLM artifact is ready to plan around.

## Bottom line

The overlay mystery is resolved negatively; the useful discovery is architectural. The current path moves a head-multiplied query tensor because CKV is sharded. GLM's compressed, head-independent 368-byte CKV record makes the opposite ownership choice much cheaper on cn3. That path preserves long-context capacity, directly attacks the measured 44% DCP wall time, has a community proof point, and has a clean falsification test.

The realistic target for the next cn3 win is **about 1.4–1.6k cold prefill**, not 3.8–8k. DCP1 may reach **about 1.5–1.74k** for short contexts. Numbers above that should be assumed to involve more GPUs, a different DCP profile, cache reuse, or additional compute changes until raw evidence proves otherwise.
