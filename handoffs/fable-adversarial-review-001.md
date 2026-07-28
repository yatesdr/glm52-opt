# Adversarial review 001: v20 official-indexer causal reference

Date: 2026-07-27  
Review target: current long-context root-cause localization and the next
end-to-end causal gate  
Requested reviewer: Fable  
Status: operator proof and baked-image no-model gates pass; corrected causal
boot pending

## 1. Decision requested

Please try to falsify the following experiment before treating its result as
causal:

> Replace the accelerated FP8 sparse-indexer scorer, from layer zero, with a
> server-static implementation of the official GLM BF16/FP32 indexer. Preserve
> v20's production exact top-k entry point and exact DCP4 local/global
> selection. If the frozen cold 250k control remains exact and all three
> previously failing cold 350k prompts become exact, the accelerated indexer
> path is causal. If the reference path also fails, this hypothesis is
> refuted.

The requested review is not “does the code look plausible?” It is:

1. Is the reference scorer actually the checkpoint's intended scorer?
2. Can any cache, DCP, metadata, graph, or weight-loading mismatch make the
   end-to-end result invalid?
3. Does the proposed gate distinguish a scorer defect from a selector policy
   effect?
4. What observation would falsify the current interpretation even if the four
   frozen rows pass?

PR #84's `oldest_boundary` mode is only a positive compatibility control. It
is not included in this branch and is not an acceptable endpoint. The WHT
hypothesis is treated as refuted by the existing real-activation evidence.

## 2. Claims already proved

### 2.1 Complete official layer-0 scorer replay

The frozen input is the real layer-0 activation from a cold, stock-v20,
343,727-token request that returned `27` instead of `738216`.

The replay starts before K normalization and RoPE and reconstructs:

1. separate BF16 Q and K projections;
2. `LayerNorm(K, eps=1e-6)`;
3. GLM interleaved RoPE over the leading 64 dimensions;
4. full 128-D Q and K vectors;
5. literal official tensor ranks for FP32 `Q @ K.T`;
6. `128**-0.5` scaling and ReLU;
7. the separately projected FP32 32-head weights and `32**-0.5` scaling;
8. FP32 learned-head reduction;
9. causality;
10. top-2,048 selection.

Pinned frozen fingerprints:

| Item | SHA-256 |
|---|---|
| trace manifest | `8f39bfc70173038086cc83d5d84d64b7536ff7595e528c7a851f7bf3f7666186` |
| official FP32 score row | `d5d70cf7324a22ce52bf13ad985affe7658474d1ff542b70b60afd678439f8fb` |
| canonical top-k indices | `0d8185a8161a1fb05d81e95f9f2019e64b421bcbcf9be7090756b29d9ec0d0e1` |
| canonical top-k values | `3617e3aadbea1faf0d8d01e83783697469fd4ba5980b6e4f51b002c9ca8af393` |

### 2.2 Production top-k is exact on the official scores

On the identical official FP32 score row:

- `torch.topk`;
- SparkInfer production `run_row_topk`; and
- SparkInfer production `run_tiled_topk` at BQ32/BK256

return the same selected set and bit-identical values. Every production value
matches the source score bitwise. The cutoff is unique: there is no
selection-boundary tie.

This clears the top-k implementation **after** score construction. It does not
clear the accelerated scorer that produced its input.

### 2.3 The accelerated score field differs materially

Against the same official top-2,048 set:

| Path | Official rows retained |
|---|---:|
| runtime BF16 preprocessing + runtime BF16 head weights | 2,029/2,048 |
| official preprocessing + runtime BF16 head weights | 2,030/2,048 |
| runtime preprocessing + official FP32 head weights | 2,026/2,048 |
| captured accelerated FP8 selection | 1,905/2,048 |

The accelerated selection omits 143 official layer-0 candidates. The
pre-FP8 differences account for only 18–22 set changes in the isolated row.
This strongly localizes the larger perturbation to the FP8 cache/scoring
portion, but it does **not** yet prove that those 143 rows cause the final
retrieval failure.

### 2.4 The baked reference implementation passes its operator gates

The derived image was built from:

```text
base image:
voipmonitor/vllm@sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0

base vLLM:
0c79e41db41f250ccdfc4be92d171960a5787f73

reference commit:
728e4902

derived image ID:
sha256:d270c8ddeaabcd8d159a4726233479b8b2f3933a828f374184a45a0f32d8726f
```

PR #84 is absent. The image contains only two changed Python source files.

| Artifact | SHA-256 |
|---|---|
| `glm_official_indexer.py` | `2af054f090e983e85272dda81e1eb1d0ef5120073ffaad2f72511f493e23bca8` |
| `deepseek_v2.py` | `5b14912dad2b006c7d1fb07eba6c706394e300a2d0e60529dba3557871649014` |
| Dockerfile | `a4a777a1dbfb0930d178ac036078476b837a6a272587ac48fc7760b1f60ab904` |
| enforce-eager wrapper | `526fc3dbe51ee969a1468c6cc2718c899d3a7d6e78e963ac9ccb3fb7eee1eba3` |
| synthetic proof | `3dfd3577323eed9060c8efd6cbad4d42477bb5c05f8991ddb2c5ffc311f19b28` |
| frozen activation proof | `c01fba86f59cee1f46aa0cce232a8e2012e7f106cb93d6f336d3c6b315baa913` |
| corrected raw scorer proof | `f7046191c7a850312e7f164c9cfcb0b4894e858d918da0d0cf34b492b8ed1cdf` |

The baked image, without source mounts, passes:

- bit-exact official interleaved RoPE;
- exact padded-slot BF16 cache insertion;
- exact paged BF16 K gathering;
- streamed official scorer plus production top-k;
- top-k values matched to their source scores;
- TF32 state restoration;
- the complete 343,727-row frozen activation/fingerprint gate.

Evidence:

```text
harness/cn4-evidence-archive/20260727/official-reference-mode-v1/no-model-baked-eager/
```

## 3. Claims not yet proved

Do not infer any of the following from the operator proofs:

- that full-precision indexing restores the 350k needle;
- that FP8 quantization is the only causal error;
- that the first bitwise divergence is the causal divergence;
- that the HF reference exactly matches this checkpoint's training-time
  implementation;
- that layer-0 candidate recall predicts the complete 78-layer trajectory;
- that DCP4 local selection/global merge is wired correctly at runtime;
- that the diagnostic mode is graph-safe;
- that the BF16/FP32 reference path is suitable for production performance;
- that the eventual fix should be a selector policy.

The reference path is a diagnostic oracle. A pass localizes the fault domain;
it is not itself the proposed production implementation.

## 4. Reference-mode implementation contract

The mode is server-static:

```text
VLLM_GLM_INDEXER_REFERENCE_MODE=official_bf16_v1
```

Unknown values fail during import. When enabled it requires:

- `glm_moe_dsa`;
- indexer geometry 32 heads × 128 dimensions with 64 rotary dimensions;
- B12X sparse indexing;
- DCP query splitting disabled;
- DCP cache interleave one;
- MTP0 and a single request for this causal test.

It recomputes the official Q, K, and learned head weights instead of consuming
the fused accelerated outputs. Its cache is BF16x128 under:

```text
indexer.k_cache_official_bf16_v1
```

Production uses a uint8x132 FP8+scale cache under `indexer.k_cache`. Prefix,
dtype, and record width all differ, so the two meanings cannot share a runtime
cache layer.

Scoring is streamed in bounded query chunks. Each DCP rank gathers its local
BF16 K shard and selects its local top 2,048 using production
`run_row_topk`. The existing deterministic DCP merge maps local candidates to
global logical indices, all-gathers four candidate sets, and runs exact top-k
over their union. This is mathematically sufficient only if the runtime local
lengths and logical mappings supplied to it are correct.

Primary review locations:

```text
workspace/vllm-v20-official-fullprecision-reference/
  vllm/model_executor/layers/glm_official_indexer.py
  vllm/model_executor/models/deepseek_v2.py
```

## 5. Adversarial risks to inspect

### A. Weight and input semantics

1. The reference Q projection uses `F.linear(qr, wq_b.weight)`. Confirm `qr`
   is exactly the official indexer's post-QA-normalization input on every
   indexed layer, not merely layer 0.
2. K and head-weight rows are sliced from the fused
   `wk_weights_proj.weight`. Confirm loading preserves the official row order
   and that no online quant overlay changes these indexer weights.
3. Confirm the official checkpoint keeps `weights_proj` in FP32 and that
   converting the loaded fused BF16 slice to FP32 is the intended comparison,
   rather than already having discarded source precision during loading.
4. Confirm the installed Transformers implementation is the correct contract
   for the checkpoint revision, rather than only the current upstream model
   definition.

### B. RoPE and numerical semantics

1. The helper intentionally uses the official rank-3 frequency matmul and
   literal batched scorer ranks because algebraically equivalent shortcuts
   were not bit-identical. Check every cast and output permutation.
2. TF32 is disabled only around official FP32 projections/scoring and then
   restored. Check for concurrent-stream/process-global hazards.
3. The trace proves layer-0 equality. Confirm `rope_theta`, positions, and
   interleaving remain correct for every indexed layer and decode step.

### C. Cache semantics

1. Confirm `DeepseekV32IndexerCache` registers the BF16x128 layer under the
   distinct prefix throughout allocation, forward metadata, and block-table
   construction.
2. Confirm `slot_mapping` and `block_table` describe the same DCP-local
   physical namespace used by `_insert_keys` and `_gather_keys`.
3. Confirm padding slots, prefix-cache reuse, and chunk boundaries cannot
   expose uninitialized BF16 records.
4. Confirm every layer has an independent reference cache and no cache group
   deduplication aliases equal specs despite different layer names.

### D. DCP correctness

1. Confirm `local_total_seq_lens` is the correct K extent for the rank-local
   block table.
2. Confirm `cu_seqlen_ke - cu_seqlen_ks` is expressed in local coordinates
   when passed to local top-k.
3. Confirm `_merge_b12x_dcp_topk` receives logical local indices in the
   format it expects and does not expect physical slots.
4. Confirm owner-merge, indexer-shard, and CKV-gather auto-policy settings
   cannot silently change this contract after `VLLM_DCP_QUERY_SPLIT=0`.
5. Look for rank-local conditional behavior that could violate the collective
   group-decision contract.

### E. Graph and compile-cache correctness

The first startup was invalidated before requests because its compose used
`GRAPH=6`. During dummy graph capture, the diagnostic operator intentionally
returns without side effects when live request metadata is absent. A captured
decode graph could therefore bypass reference selection and create a false
failure.

That container was stopped before any causal request. Its logs and invalidation
reason are archived. The corrected compose sets `GRAPH=0`:

```text
compose/glm52-v20-official-bf16-reference-causal-20260727.yaml
SHA-256 610e49d5497cc20955880a19acc2560862194d04766fc26e6e3a26b7013e22b1
```

Persistent vLLM compile-cache reuse is also disabled. Before this mode could
be upstreamed even as a diagnostic, the mode must either become part of the
compile key or implement graph-safe behavior.

### F. Gate validity

1. The 250k control and all three 350k rows must be byte-frozen and cold
   (`cached_tokens=0`).
2. Pass requires finalized `content.strip() == "738216"` and
   `finish_reason=stop`; retrieval in reasoning alone is not a pass.
3. The control must run first. A failed control invalidates the experiment.
4. A four-row pass proves causality only for this frozen set. It must later
   survive randomized 50k–475k prompts before a production claim.

## 6. Corrected causal gate

One corrected CN4 boot will use:

```text
TP=4
DCP=4
MTP=0
MAX_MODEL_LEN=360000
MAX_NUM_SEQS=1
MAX_BATCHED_TOKENS=3072
GRAPH=0
KV_FP8_ROPE=1
GPU_MEMORY_UTILIZATION=0.974
F8_DMA=0
VLLM_DCP_QUERY_SPLIT=0
VLLM_USE_B12X_SPARSE_INDEXER=1
VLLM_GLM_INDEXER_REFERENCE_MODE=official_bf16_v1
VLLM_DISABLE_COMPILE_CACHE=1
```

The gate order is:

1. `pass-250k-ctl`;
2. `fail-350k-r1`;
3. `fail-350k-r2`;
4. `fail-350k-r3`.

Each 350k row has 343,727 rendered prompt tokens, a distinct frozen prompt
hash, temperature zero, 2,000 maximum output tokens, thinking disabled, and a
needle near 40% depth.

## 7. Interpretation tree

### If 250k and all three 350k rows pass

The compressed indexer path is causal for the frozen failure set. Do **not**
ship the slow reference mode. Instrument the reference and accelerated paths
at the first full-indexer layer and add one component at a time:

1. fused versus separate Q projection;
2. fused versus separate K/head-weight projection;
3. runtime versus official RoPE ordering/reduction;
4. BF16 versus FP8 Q;
5. BF16 versus FP8+scale K cache;
6. production FP8 score accumulation/scaling;
7. DCP local/global mapping.

The smallest component that changes the end-to-end result is the production
fix candidate. It must reproduce the reference trajectory without introducing
an operator-managed selector mode.

### If the 250k control fails

The experiment is invalid. Investigate reference implementation, DCP/cache
wiring, memory pressure, or execution-mode defects. Do not interpret it as
evidence about FP8 quality.

### If 250k passes but any 350k row fails

The current “compressed scorer is sufficient cause” hypothesis is refuted.
Before proposing another policy:

1. verify the reference implementation against loaded weights at every
   boundary;
2. determine the checkpoint's training-time selector/scorer contract;
3. inspect IndexCache reuse semantics across layers;
4. compare the official and accelerated selection trajectory from the first
   full-indexer layer through the final indexed layer;
5. test whether the frozen prompt is sensitive to scoring changes that are
   individually accurate but trajectory-changing.

## 8. Reviewer response requested

Please return:

1. **Fatal flaws** — anything that makes the corrected four-row result
   uninterpretable.
2. **Likely defects** — exact file/function/contract and why.
3. **Missing proofs** — the cheapest proof that should precede the corrected
   boot, if any.
4. **Alternative causal models** — ranked, with a discriminating experiment.
5. **Go/no-go** — whether the corrected `GRAPH=0` boot is a valid next test.

Do not recommend `oldest_boundary`, historical out-of-bounds behavior, or WHT
as the endpoint without new evidence that overturns the current constraints.
