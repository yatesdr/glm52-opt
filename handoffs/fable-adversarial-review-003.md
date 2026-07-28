# Fable adversarial review 003 — v20 deep-context retrieval pivot

Date: 2026-07-27  
Author: Sol  
Host under test: CN4 only  
Production: CN3 was not stopped or modified  
Status: upstream representation precision is causal for frozen 350k r1; the
minimal responsible component is being bisected

## 1. Review request

Please adversarially review four claims and the narrowing plan:

1. The trained indexer scorer, the exact top-k implementation, and downstream
   sparse gather/page-table consumption are not the primary cause of the
   frozen 350k failure.
2. The hidden-state trajectory produced by the shared main-attention precision
   posture is causal for frozen r1.
3. The successful cell proves only the union of its changed inputs; it does
   not yet prove that NVFP4 MLA KV, FP8 RoPE, or the DCP transport
   implementation is individually necessary or sufficient.
4. The shortest path to a canonical fix is a one-variable bisection from the
   recovered posture, followed by an operator-level bug-versus-quantization
   decomposition of the smallest failing component. `oldest_boundary` remains
   a positive compatibility control, not the final selector.

Please look specifically for:

- any execution, prompt, checkpoint, scorer, or cache-state mismatch that
  invalidates the causal comparison;
- any alternative shared component that changed unintentionally;
- any reason the recovery could be a false positive;
- any cheaper discriminator than the proposed bisection;
- whether the final production requirements are incomplete.

## 2. Executive result

The same immutable 343,727-token request that failed under:

- stock exact v20 scoring,
- the SparkInfer page-table stride fix,
- and an official GLM BF16/FP32 scorer,

recovered exactly after increasing the precision of the **shared upstream
main-attention trajectory** while holding the official scorer constant.

| Field | Result |
|---|---|
| frozen row | `fail-350k-r1` |
| prompt SHA-256 | `f0d1c16d816b777f27a3882d9e6b5ef056852684ea155fb11dd845f9e1654ab5` |
| prompt tokens | 343,727 |
| result | `EXACT` / gate `PASS` |
| content | `738216` |
| completion | 4 tokens |
| finish | `stop` |
| cached tokens | 0 |
| elapsed | 708.8 s |
| container | healthy, zero restarts |
| KV pool | 491,769 tokens |

This is the first direct causal recovery using the exact selector and the
official scorer. It moves the investigation out of selector policy and into
the representations that produce the scorer's hidden-state inputs.

It does **not** yet identify a production fix. The recovered posture changed
two numerical representations plus one implementation-control path, and its
491,769-token pool is below the eventual 500k-at-480k promotion floor.

## 3. Evidence chain

### 3.1 Stock runtime-stride RC still fails

Pinned RC:

```text
voipmonitor/vllm@sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0
vLLM 0c79e41db41f250ccdfc4be92d171960a5787f73
SparkInfer c3828fd7f807ce237a9ac36ef033659e6f6b6dd3
```

SparkInfer #85 corrects a real cross-width page-table compile-cache stride
defect. A standalone GPU proof fails on the old image and passes on this RC.
The frozen end-to-end gate nevertheless remains:

| Row | Result |
|---|---|
| 250k control | `EXACT`, `738216` |
| 350k r1 | `ABSENT`, `27` |
| 350k r2 | `ABSENT`, fabricated ticket |
| 350k r3 | `ABSENT`, `27` |

Query split and owner merge were pinned off, so this was a narrow #85 A/B.
Conclusion: #85 is necessary correctness work but not sufficient to restore
deep retrieval.

### 3.2 Official scorer and exact top-k do not restore the row

The diagnostic mode implements the GLM reference scoring contract:

1. indexer norm;
2. GLM interleaved RoPE;
3. `q.float() @ k.float().T`;
4. FP32 ReLU and head weights;
5. production exact top-k.

Operator proofs established:

- full official-score SparkInfer top-k set and values match Torch;
- chunk widths 16 and 64 are bit-exact;
- zero-local-length DCP metadata is covered;
- layer-zero production/reference fingerprints match;
- a real layer-34 activation replay places the three exact value tokens at
  BF16 ranks 92,886 / 75,542 / 82,581, far outside top-2,048;
- local FP8 scorer quantization changes those ranks but cannot account for a
  movement from top-2,048 to approximately 80k.

End to end, the official scorer passes the 250k control and fails all three
350k rows. Frozen r1 again answers `27`.

Conclusion: a cleaner scorer cannot repair inputs whose relevant history is
already badly ranked.

### 3.3 Exact needle-inclusion trace localizes the failure before attention

The official scorer was instrumented to record whether the three ticket-value
tokens were selected at every sparse layer for each answer-decode call.

Frozen r1 produced 336/336 expected records:

| Sparse-layer range | Exact ticket selection |
|---|---|
| 0–38 | no value token selected in any answer call |
| 42 | one value token in 1/16 calls |
| 50 | sparse partial inclusion |
| 62 | all three together in 1/16 calls |
| 74 | any token in 15/16 calls; all three in 7/16 |

The necessary candidates do not reach sparse attention until layers 62–74,
too late to steer the answer. This rules out downstream page-table/gather loss
as the primary cause of this row. The selector is faithfully selecting from a
score field whose relevant history has already degraded.

Trace evidence:

```text
harness/cn4-evidence-archive/20260727/
  official-reference-needle-trace-103473cd/frozen-r1/

trace-analysis.json
  2278cb0fd6e0ad87c9c16c6f77da1187c78df23984043c4debd8cd8a04b33751
```

### 3.4 Highest-precision fitting posture recovers exactly

A literal BF16 main MLA cache does not fit the 343,727-token request on CN4.
Measured/derived record sizes are:

```text
nvfp4_ds_mla   368 B/token
fp8_ds_mla     656 B/token, BF16 RoPE field
BF16 MLA      ~1152 B/token
```

The highest-precision posture expected to fit therefore used:

| Component | Failed reference posture | Recovered posture |
|---|---|---|
| checkpoint | NF3 hybrid | unchanged |
| scorer/selector | official BF16/FP32 + exact top-k | unchanged |
| main MLA KV | `nvfp4_ds_mla` | `fp8_ds_mla` |
| RoPE cache | compact FP8 | BF16 |
| DCP wire | `i8_ring` | raw BF16 |
| TP/DCP | 4/4 | unchanged |
| MTP | 0 | unchanged |
| graph posture | eager | unchanged |
| prefix cache | disabled | unchanged |
| MNBT | 3,072 | unchanged |
| query split / owner merge | 0 / 0 | unchanged |
| CKV prefetch | 0 | unchanged |
| max length | 360,000 | unchanged |
| GMU | 0.950 | unchanged |

Pinned image and compose:

```text
image tag:
  glm52-serve:v20-official-reference-runtime-stride-chunk16-75715e51
image ID:
  sha256:899e64cc6098407d1e41bca8db53f70ea60f31009b812872e4690540798ded1a
base:
  sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0
compose:
  compose/glm52-v20-official-reference-fp8kv-bf16rope-lossless-r1-20260727.yaml
compose SHA-256:
  73aa5c6af6bd2a6d450f5911d19c7cb62478f453b4e7861198ee482404e2f4bd
```

Pinned result:

```text
harness/cn4-evidence-archive/20260727/
  official-reference-fp8kv-bf16rope-lossless-r1-v1/

run.log
  688cd35059df720aebef73d22e8f1f65a10224ef140d4ed0d1812bd9a384ba53
results/summary.json
  db505691d4a6025eea8789e10d19e372133386fc990990d5845da4e866052cb9
results/rows.json
  828ee3ae1684757a9dac6cd585883c12da4d53cc25c606c561919aced9b0bc14
results/resp-fail-350k-r1.json
  0481d7f9cb209afa7ec5d70df820f777edd539ca9469306ed58ba56a4716ef3d
container.log
  d893248abe03b43ffac1c1720e663de29bff61f17ee6fe5af4d4caa050ccbaae
container-inspect.json
  376bb1abc3a8d576c794c1b790df72b96675470965e7709e7ea6fe94cbc57beb
```

Detailed record:

```text
design/v20-shared-precision-trajectory-causal-20260727.md
```

Issue update:

```text
https://github.com/local-inference-lab/vllm/issues/182#issuecomment-5096732053
```

## 4. Exact claim boundary

Supported:

> For frozen 350k r1, changing the shared main-attention representation
> posture from NVFP4 MLA KV / compact FP8 RoPE / i8_ring to FP8-DS-MLA with
> BF16 RoPE / raw BF16 wire restores exact retrieval while the official
> scorer, exact top-k, execution posture, and request remain fixed.

Not yet supported:

- `i8_ring` is sufficient by itself to cause the frozen failure;
- FP8 RoPE alone is the cause;
- NVFP4 MLA KV alone is the cause;
- the same posture restores r2/r3 or a randomized ladder;
- the recovery is production-ready;
- 491,769 KV tokens satisfy the 500k-at-480k contract;
- `oldest_boundary` is the correct selector;
- ordinary model capability ends near 262k or 350k.

`i8_ring` is rank-consistent block-INT8 transport, but it is numerically
lossy. Each 128-value block uses one FP32 scale and signed-INT8 rounding; its
pre-BF16 absolute codec error is bounded by `amax / 254`. It is higher
precision than the prior E4M3 wire mode, not bit-exact BF16. It remains the
first A/B because restoring it is the cheapest genuine precision
discriminator and a direct test of whether accumulated wire rounding is
responsible.

## 5. Active narrowing cell

The completed reference container was fully archived and removed. A
single-variable replacement is booting on CN4:

```text
launched compose bytes:
  harness/cn4-evidence-archive/20260727/
    official-reference-fp8kv-bf16rope-i8ring-r1-v1/compose.launched.yaml
launched SHA-256:
  6716295d99c15bc04ee414d247eac0674adfa224bf0e439c575cf1867c9ffe20
working-copy compose (comment corrected; effective config unchanged):
  compose/glm52-v20-official-reference-fp8kv-bf16rope-i8ring-r1-20260727.yaml
working-copy SHA-256:
  d9e209c7c7fd760f24f0fc2cc501ec90c6cbbe2db893ab08287471544d3d2baa
image:
  sha256:899e64cc6098407d1e41bca8db53f70ea60f31009b812872e4690540798ded1a
```

Functional delta from the recovered compose:

```text
F8_DMA=0 -> F8_DMA=i8_ring
```

The distinct container/cache/project names and subnet are isolation metadata.
Model/scorer/cache format/RoPE/execution settings are unchanged. The boot
identity is already correct, health is starting, restart count is zero, and
the launcher reports:

```text
GLM-5.2 PCIe calibration: skipped:explicit-compressed-dma
DMA-min=6MB
```

No inference result exists yet. The binary r1 result will be consumed only
after fail-closed boot identity, effective wire variables, 656-byte
`fp8_ds_mla`, BF16 RoPE, KV-fit, health, and cold-request checks pass.

The current image does not contain the needle-entry trace overlay. The binary
frozen result will be recorded first; a trace-capable overlay on the winning
posture is required before ranking additive contributions or declaring the
promotion metric complete.

## 6. Decision tree

### 6.1 FP8-DS-MLA + BF16 RoPE + i8_ring remains EXACT

Then raw BF16 wire is unnecessary, and `i8_ring` is exonerated for this row.
The recovered causal set narrows to:

```text
NVFP4 MLA CKV and/or compact FP8 RoPE
```

Next boot:

```text
nvfp4_ds_mla + KV_FP8_ROPE=0 + i8_ring
```

If that passes, compact FP8 RoPE is the primary minimal lever. If it fails,
the FP8-DS-MLA versus NVFP4 main CKV representation becomes the lead.

### 6.2 Restoring i8_ring makes r1 fail

This would establish `i8_ring` as a causal contributor for the frozen row, but
would not by itself distinguish expected block-INT8 codec loss from a
routing/collective implementation defect. Before changing a production
default:

1. run the exact block-INT8 transport round-trip proof at the observed tail
   geometry;
2. compare rank-by-rank output fingerprints with raw BF16;
3. prove group-init/routing invariance and exact restoration;
4. distinguish transport bytes from changes caused by enabling the custom
   collective path.

The fix would be in transport correctness/routing, not selector policy.

### 6.3 More than one lever contributes

A one-row pass/fail boundary can hide additive margin erosion. After finding
the first minimal pass, perform the complement cell:

```text
candidate culprit changed alone
versus
candidate culprit changed with the other lossy components restored
```

Use ticket-entry layer and exact score/rank margins, not only final text, to
quantify each component.

## 7. Fix-shape requirements

If compact FP8 RoPE or NVFP4 MLA KV is responsible, do not immediately ship
“turn it off.” First establish whether the loss is:

- expected quantization error at the declared record format; or
- a scale, rounding, layout, stale-scale, or encode/decode implementation
  defect.

Required operator decomposition against frozen real activations:

1. capture the pre-encode BF16 CKV/RoPE record;
2. encode through the production writer;
3. decode through the exact attention consumer;
4. compare per-field error, scale provenance, and rank-wise accumulation;
5. replay the official scorer/needle-entry trace with only that component
   substituted;
6. verify graph/eager and cold/restart consistency.

The preferred production fix preserves the 368-byte record if a correctable
implementation defect exists. A larger cache format is acceptable only if
quality requires it and the final 500k-at-480k capacity floor can be restored
elsewhere.

## 8. Promotion requirements after causal isolation

No candidate is production-ready until all are green:

1. frozen 250k control plus all 3×350k failures;
2. randomized cold 50k–475k ladder with content, reasoning,
   `reasoning_content`, and serialized-message scoring;
3. non-empty final `content` at every accepted depth;
4. quantitative ticket-entry-layer/score-margin regression gate;
5. graph and eager consistency;
6. fresh-cache, warm-cache, restart, and compile-cache repeatability;
7. valid KLD/quality suite;
8. matched prefill and decode performance;
9. at least 500,000 KV tokens at max model length 480,000;
10. no fatal signatures, collective hangs, page-table corruption, or NVMe
    capacity breach;
11. reproducible pinned image and minimal upstream PR.

The completed shallow n=3 KLD comparison for `oldest_boundary` remains a
shallow no-regression control only. Its selector budget covered the whole
2,048-token window, so it did not exercise the long-context candidate
selection problem and cannot validate this fix.

## 9. Questions for Fable

Please answer these explicitly:

1. Is the recovered comparison causal as stated, or is there an unrecorded
   difference that could explain exact output?
2. Is restoring `i8_ring` the best first discriminator given its measured
   block-INT8 error contract and the historical E4M3-to-INT8 recovery?
3. Should `nvfp4_ds_mla + KV_FP8_ROPE=0` be the second cell, or is there a
   cleaner way to isolate the RoPE writer from the 368-byte CKV format?
4. What direct operator proof best distinguishes inherent quantization cost
   from a scale/layout defect?
5. Is the proposed complement cell sufficient to detect additive margin
   erosion?
6. What evidence would be required before concluding that a larger cache
   record is unavoidable?
7. Which claim or planned gate is presently too broad?

## 10. Current bottom line

The investigation has moved from correlation to one-row causality:

```text
official scorer on original shared inputs -> fails
official scorer on higher-precision shared inputs -> exact recovery
```

The permanent fix is not another selector mode. It is the smallest
representation or transport correction that restores the trained score
trajectory while preserving the required context capacity and performance.
The active bisection is designed to identify that correction with at most two
additional primary boots, plus one complement cell if the errors are
additive.
