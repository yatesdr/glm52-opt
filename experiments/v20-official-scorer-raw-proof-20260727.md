# v20 complete official-indexer-scorer proof

Date: 2026-07-27  
Status: complete — production top-k cleared; accelerated score field diverges

## Question

Does the v20 production top-k kernel select the wrong rows when given the
scores defined by the official GLM indexer, or do the scores already diverge
before top-k?

This is the first proof in the long-context investigation that begins before
the indexer's Q/K normalization and RoPE. It is deliberately independent of
PR #84 and its `oldest_boundary` compatibility policy.

## Immutable inputs

- base image:
  `voipmonitor/vllm@sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0`
- base vLLM commit:
  `0c79e41db41f250ccdfc4be92d171960a5787f73`
- base SparkInfer commit:
  `e603f74bb67d0fce547336f1fb73c3c23e8f1887`
- installed official Transformers GLM source SHA-256:
  `adb8317a21716b01273046e46c807f14f0dbaf035af59b60d52bd6bc3007cf72`
- mounted model config SHA-256:
  `254974797e9f455716a30ab5505ba68272181b20b58a3693e54f94fb8056f3ef`
- model geometry:
  32 indexer heads, 128-D indexer Q/K, 64 rotary dimensions,
  `rope_theta=8000000`, exact top-2048
- frozen request:
  343,727 prompt tokens, final prefill chunk 2,735, query position 343,726

## Official scorer contract

The pinned Transformers implementation performs:

1. separate BF16 Q and K projections;
2. `LayerNorm(K, eps=1e-6)`;
3. GLM interleaved RoPE on the first 64 dimensions;
4. `Q.float() @ K.float().T * 128**-0.5`;
5. FP32 ReLU;
6. a learned per-head projection kept in FP32, scaled by `32**-0.5`;
7. FP32 head reduction, causal masking, and exact top-k.

The checkpoint stores the learned per-head projection in BF16, but the official
loader explicitly keeps `indexer.weights_proj` in FP32. Current vLLM instead
fuses `wk` and `weights_proj` into one BF16 projection. This is a concrete
candidate numerical boundary; it is not yet claimed as the end-to-end root
cause.

The official Transformers RoPE helper and vLLM's GPT-J-style helper place the
rotated values in different but fixed orders. The same permutation is applied
to Q and K, so it preserves Q.K mathematically. An FP32 GEMM may still differ
in its final bits because the permutation changes reduction order. The proof
keeps the official ordering for the scorer and maps it into vLLM order only for
elementwise RoPE comparisons.

## Already closed: scorer suffix and top-k

The earlier frozen post-K-normalization/post-RoPE proof reconstructed the
official FP32 scorer suffix and fed one identical score row to:

- `torch.topk`;
- production `run_row_topk`;
- production `run_tiled_topk` with BQ32/BK256.

Both production kernels matched PyTorch bit-exactly in indices and values.
The cutoff was unique: 2,047 strict winners plus one cutoff row, with no tie
ambiguity.

Evidence:

- harness:
  `harness/v20_glm_reference_scorer_production_topk_proof.py`
- harness SHA-256:
  `86d7236f41ad9e2f25d0deb31560e341899f80329e4260dc776211e03c77f8ba`
- result:
  `harness/cn4-evidence-archive/20260727/official-scorer-topk/post-rope-production-topk-v1.json`
- result SHA-256:
  `f40236b83746093538e5ba4aca18a40d583b3e811bf28767f177aa88268a31ed`

That proof clears the production top-k implementation after the score field.
It does not clear the preprocessing that creates the score field.

## Raw trace and replay

The new trace captures, from a single armed layer and only on TP rank 0:

- Q residual, Q projection weight, separate Q projection, and runtime Q;
- hidden state;
- fused WK/head-weight projection;
- the same K projection recomputed as a standalone GEMM;
- runtime K immediately after LayerNorm;
- runtime Q/K immediately after RoPE;
- K LayerNorm weight and bias;
- actual production FP8 selector output.

The offline replay then computes the complete official scorer and four
controlled score fields:

1. runtime BF16 Q/K preprocessing + runtime BF16 head weights;
2. official Q/K preprocessing + runtime BF16 head weights;
3. runtime Q/K preprocessing + official FP32 head weights;
4. full official Q/K preprocessing + official FP32 head weights.

For each field it records score hashes, score error against the full reference,
top-k Jaccard/recall, and comparison to the captured production FP8 selection.
The full official score row is also sent through both production top-k kernels.

## Build and evidence ledger

| Artifact | Identity |
|---|---|
| diagnostic branch | `diag/v20-official-scorer-trace-20260727` |
| diagnostic commit | `ae8d15619e94e6b6409bd3d238dadbae5f44bf0e` |
| overlaid vLLM source SHA-256 | `3fcaf4b99445e37d1a376494551ca78813ad85c020930daf7ee125c840e261ac` |
| raw replay harness v1, baked into diagnostic image | `bcc425f2ed5d1f3da249ce9d4b66dd908f1c7a5d044ba24e4813a13f942f645c2` |
| raw replay harness v2, used for final result | `4735b3ea8a13e28fd3046dd8f8cf4eb1e8ebbd968ba17c2ddc94478c5b005d46` |
| current Dockerfile SHA-256 | `6ddebf97dbbc4cbc8e101d9882d00176bf91cb194086266dff04a0cb178510e8` |
| Compose SHA-256 | `f3f8efa4068b024c9dc38b8c4efc2398da2562a77b23c78713abfc5ba475bf96` |
| built image ID | `sha256:400d2d404af05938de9424a2a848b2cb4b8cd15716c81e3226c6f276e47a53a8` |
| CN4 evidence directory | `/home/derek/proof-results/20260727/indexer-official-reference-layer0-v1` |
| comms record | `proofs#218` |

Build gates passed:

- exact base source hash in both `/opt/vllm` and installed package;
- exact official Transformers source hash;
- exact overlay and proof hashes;
- `py_compile`;
- post-copy source hashes;
- no-model GPU import and registration of both diagnostic custom-op schemas.

The image was built with harness v1. Harness v2 adds only an explicit
selection-cutoff/tie characterization and its own SHA-256 to the report. It
was hash-verified and bind-mounted into the same image for the final replay.
The current Dockerfile pins v2 for any future rebuild.

## Frozen request result

The trace was armed only after `/health` returned HTTP 200. Before arming:

- the trace directory contained no activation files;
- the container had zero restarts;
- startup profiling and CUDA graph capture had completed.

The one requested frozen row reproduced the stock failure:

| Field | Result |
|---|---|
| frozen label | `fail-350k-r1` |
| prompt tokens | 343,727 |
| cached tokens | 0 |
| finish reason | `stop` |
| completion | `The maintenance ticket number ... is **27**.` |
| needle verdict | `ABSENT` |
| container restarts | 0 |
| activation chunks | 112 |
| activation rows | 343,727, positions 0--343,726 |
| activation manifest SHA-256 | `8f39bfc70173038086cc83d5d84d64b7536ff7595e528c7a851f7bf3f7666186` |

Compact request records:

- `frozen-request-rows.json`:
  `f173812ea5217b16131edfc42aa1a1819b242ec368f0579b4fba2e63fa361d62`
- `frozen-request-summary.json`:
  `e98272ed832027df69755ad33794c344195a7c5f039ffc0f8912b5a6714f72f0`

## Complete official-scorer result

Final result:

`harness/cn4-evidence-archive/20260727/official-scorer-topk/raw-preprocessing-layer0-v1/raw-official-scorer-production-topk-v2.json`

SHA-256:

`8aac830fa22976f74692f4cdd72304a0245d7a382b29c62ae92c9dc7da99c4d1`

### Production top-k

Both production entrypoints matched `torch.topk` on the identical full
official FP32 score row:

| Entry point | Index set | Values by index | Values from source row |
|---|---|---|---|
| `run_row_topk` | exact | bit-exact | bit-exact |
| `run_tiled_topk` BQ32/BK256 | exact | bit-exact | bit-exact |

The boundary is unambiguous: 2,047 rows score strictly above the cutoff,
exactly one row equals the cutoff, and exactly that row is selected. The
cutoff is `5.734244346618652`; there is no selection-boundary tie.

### Score and selected-set deltas

All comparisons use the same 343,727 real rows and full official top-2048 as
the reference:

| Scorer/input path | RMSE vs official | Pearson | Official rows retained |
|---|---:|---:|---:|
| runtime BF16 Q/K preprocessing + runtime BF16 head weights | 0.00974804 | 0.99998970 | 2,029/2,048 (99.072%) |
| official Q/K + runtime BF16 head weights | 0.00653082 | 0.99999544 | 2,030/2,048 (99.121%) |
| runtime Q/K + official FP32 head weights | 0.00620493 | 0.99999074 | 2,026/2,048 (98.926%) |
| captured production FP8 selection | n/a | n/a | 1,905/2,048 (93.018%) |

The captured accelerated path therefore omits 143 official candidates at
layer zero. The BF16 preprocessing/head-weight changes account for only
19--22 selected rows in isolation; the much larger gap appears after the
current FP8 cache/scoring path.

### Tensor boundary localization

| Boundary | Different elements | Total elements | Max absolute delta |
|---|---:|---:|---:|
| separate Q projection vs fused runtime Q | 4 | 4,096 | 0.03125 |
| separate K projection vs fused runtime K | 4 | 43,997,056 | 0.00003052 |
| official K LayerNorm vs runtime | 0 | 43,997,056 | 0 |
| official Q RoPE vs runtime, after order mapping | 655 | 4,096 | 0.125 |
| official K RoPE vs runtime, after order mapping | 6,686,557 | 43,997,056 | 0.03125 |
| official FP32 vs fused BF16 learned head weights | 32 | 32 | 0.00127748 |

The first bitwise divergence is the fused Q projection, but it is only four
BF16 values. The first broad divergence is the RoPE result. The full
end-to-end reference-indexer experiment is still required before assigning
causality to either boundary, because a first bit difference is not
automatically the difference that changes model retrieval.

## Interpretation rule

The first bitwise divergence is a localization result, not automatically the
quality root cause. A candidate becomes causal only after a server-static,
cache-isolated reference mode changes the frozen 350k retrieval result while
the 250k control remains exact. No selector compatibility policy will be
promoted from this proof.
