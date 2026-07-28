# v20 official full-precision indexer reference mode

Date: 2026-07-27

Status: no-model, live metadata, 8k, and frozen 250k control gates pass. The
first 350k attempt is invalid due to diagnostic-scorer transient OOM. A
bit-exact resource-fit correction is proved; the corrected image is now
running the full live rerun.

This work is based directly on clean v20 commit
`0c79e41db41f250ccdfc4be92d171960a5787f73`, SparkInfer
`c3828fd7f807ce237a9ac36ef033659e6f6b6dd3`, and does not contain PR #84.
The mode is a diagnostic oracle, not a proposed operator choice or production
selector.

## Question

The frozen 350k failure showed that production top-k is exact on the scores it
receives, while the accelerated FP8 indexer retains only 1,905 of the official
top-2,048 layer-0 candidates.  This mode tests whether the complete official
GLM indexer trajectory restores end-to-end retrieval.

The causal claim is deliberately bounded:

- if the cold 250k control remains exact and all three cold 350k failures
  become exact, the accelerated indexer path is causal;
- if the reference mode also fails, no selector heuristic is justified and
  the training-time contract must be investigated instead.

## Literal official scorer contract

The implementation follows `GlmMoeDsaIndexer.forward` from the pinned
Transformers source (`SHA-256
adb8317a21716b01273046e46c807f14f0dbaf035af59b60d52bd6bc3007cf72`):

1. separate BF16 `wq_b` and `wk` projections;
2. BF16 LayerNorm on the 128-D K vector;
3. GLM interleaved RoPE on the leading 64 dimensions, after which the complete
   128-D vectors are reassembled;
4. literal tensor ranks `Q=[B,S,H,D]`, `K=[B,T,D]`;
5. FP32 `Q @ K.T`, multiplied by `128**-0.5`;
6. FP32 ReLU;
7. a separate FP32 learned 32-head projection, multiplied by `32**-0.5`;
8. FP32 learned-head reduction;
9. causal per-row lengths;
10. SparkInfer production `run_row_topk` with `topk=2048`;
11. exact DCP rank-local selection followed by the existing production DCP
    global candidate merge.

The reference scorer disables TF32 only around its FP32 projection and scorer
matmuls, restoring the previous process setting immediately afterward.

## Cache and server-static contract

The mode is enabled only by:

```text
VLLM_GLM_INDEXER_REFERENCE_MODE=official_bf16_v1
```

Unknown values fail during module import.  When enabled:

- model type must be `glm_moe_dsa`;
- geometry must be 32 heads, 128 dimensions, and 64 RoPE dimensions;
- B12X sparse indexing must be enabled;
- DCP query splitting must be disabled;
- DCP cache interleave must be one;
- the run is MTP0/eager-only for the causal experiment.

The reference cache is a BF16 128-D record with prefix:

```text
indexer.k_cache_official_bf16_v1
```

Production uses a uint8 132-byte FP8+scale record under `indexer.k_cache`.
Prefix, dtype, and record width all differ, so the cache manager and static
forward context cannot mix the two meanings.

## Streaming and DCP

The full score matrix is never retained.  Within each vLLM prefill batch:

1. each rank gathers its own BF16 paged K shard;
2. query rows are scored in bounded chunks;
3. only each rank's local top-2,048 values and indices are retained;
4. after all rows in the batch are locally selected, the existing production
   DCP merge converts local indices to global logical indices, all-gathers the
   four candidate sets, and runs production exact top-k over their union.

This is mathematically exact: a member of the global top-2,048 must be in its
own partition's local top-2,048.  The reference mode does not use physical
slot output under DCP.

## No-model gates

### Synthetic operator gate

`harness/v20_glm_official_reference_indexer_gate.py` passed on CN4 GPU0:

- official interleaved RoPE exact;
- padded-slot cache insertion exact;
- paged K gathering exact;
- streamed scorer plus production top-k exact on three rows;
- output values match their source scores;
- TF32 state restored.

The first version of this gate incorrectly compared the official rank-3 RoPE
frequency matmul with a scalar multiplication shortcut.  The shortcut is
algebraically equal but not bit-identical at large positions and was removed.

The corrected runtime image also proves the live metadata contracts that were
missing from the first build:

- non-speculative decode lengths shaped `(B, 1)` normalize to `(B,)`;
- speculative/multi-column decode lengths are rejected;
- prefill key windows must be zero-based.
- a DCP rank with zero local keys contributes only `-inf` scores and `-1`
  indices.

### Frozen real-activation gate

`harness/v20_glm_official_reference_real_activation_gate.py` passed against
the 343,727-row layer-0 trace:

```text
trace manifest   8f39bfc70173038086cc83d5d84d64b7536ff7595e528c7a851f7bf3f7666186
official scores d5d70cf7324a22ce52bf13ad985affe7658474d1ff542b70b60afd678439f8fb
top-k indices   0d8185a8161a1fb05d81e95f9f2019e64b421bcbcf9be7090756b29d9ec0d0e1
top-k values    3617e3aadbea1faf0d8d01e83783697469fd4ba5980b6e4f51b002c9ca8af393
```

The Q and K RoPE outputs are bit-exact to the independent frozen oracle.  The
production top-k index set and values are exact, and every returned value
matches its source score bitwise.

An intermediate gate failure compared the raw, unsorted production output
order with a fingerprint canonicalized by ascending index.  Canonicalizing
both sides proved that there was no score, membership, or value discrepancy.
That failure is not evidence of a model implementation defect.

### Fail-closed mode parsing

An in-image import with
`VLLM_GLM_INDEXER_REFERENCE_MODE=invalid` failed with the expected `ValueError`.
In-image `py_compile` passes for both changed source files.

## Current source pins

```text
glm_official_indexer.py
2af054f090e983e85272dda81e1eb1d0ef5120073ffaad2f72511f493e23bca8

deepseek_v2.py
5b14912dad2b006c7d1fb07eba6c706394e300a2d0e60529dba3557871649014

synthetic gate
3dfd3577323eed9060c8efd6cbad4d42477bb5c05f8991ddb2c5ffc311f19b28

real-activation gate
c01fba86f59cee1f46aa0cce232a8e2012e7f106cb93d6f336d3c6b315baa913

corrected raw proof
f7046191c7a850312e7f164c9cfcb0b4894e858d918da0d0cf34b492b8ed1cdf

corrected reference source commit
a38fc99af111c2074eb222a19b8f1f24362fab48

glm_official_indexer.py
166a85072398053236a78a31d8d34e29d8183aeb3809307cbdbde62718819950

corrected synthetic gate
915fae6d73d3433e947c9ea5a97d60e7e2a14b864958d4f18d4c3f797adb8f88

runtime-stride reference image
sha256:3b4a665ea5662b389391b39699d6f83398368fdc7506bf1491446de924e7a6c8

validated compose
318ef6cfa61c5f19c7666829366071bf30339cadab4cec2d2fa7bc4b5f1b897e

resource-fit reference source commit
75715e519b73b712873e488f7c42c8d61d14dbbc

resource-fit glm_official_indexer.py
2f73eac49aa199f884bfebaa49c0449f033af665f26a6f48c386fb093ecdbe35

resource-fit synthetic gate, including zero-local-length
8a31f840d6d82311e4227336eb6295cd09f835d590e7515263a3cecf67dc6e95

resource-fit image
sha256:899e64cc6098407d1e41bca8db53f70ea60f31009b812872e4690540798ded1a

resource-fit compose
b6de5bff44ad9777f08b95d6e7a1bbce647d49524e8d77d928d7f851b4e4ba31
```

## Runtime-stride RC no-model results

Before the causal boot, both reference proofs were repeated against the
corrected image on exact base image
`sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0`.
The synthetic gate and real-activation gate both passed. Their archived result
hashes are:

```text
synthetic:
  8f84209a23d5b5657a3f20daa1c0b5c2fa3a060bdf3f9a07f7037a1ac84ba18a
real activation:
  4035d061084d69a5bda91e9b76f18f5dcc2fb691264e98ecffdd36b504e4957a
```

A separate cross-width proof reproduced the SparkInfer row-stride defect on
the old image and proved it absent on the new RC:

```text
old image, bug present:
  2024fb14a49870537d23cc22dc06359268cc57ef2987cd527fc615aa2195dfdb
new RC, bug absent:
  ec62d4dd499ab16a6028e63a1895bc5510a27ad3d91e8fdf5ffb57aae864e035
```

The end-to-end stock RC still failed all three frozen 350k rows, so this real
stride bug is independently fixed but is not sufficient to explain or restore
deep retrieval.

## Resource-fit proof

The first 350k reference request reached 248,832 computed prompt tokens and
then rank 3 failed allocating the diagnostic FP32 `per_head` tensor:

```text
requested:       492 MiB
physically free: 500.69 MiB
```

This is an invalid resource failure, not a model-quality result. The scorer
originally processed 64 query rows at a time. The corrected default is 16.
At production DCP-local geometry (64 rows, 85,932 K rows, 32 heads, 128
dimensions, top-k 2,048), three independent seeds proved:

- identical selected membership;
- bit-identical selected values;
- zero changed rows;
- maximum score delta `0.0`.

Peak allocated bytes fell from approximately 852 MiB to 465 MiB. The exact
result is archived at:

```text
harness/cn4-evidence-archive/20260727/
  official-reference-runtime-stride-chunk16-75715e51/
  no-model/chunk16-vs64.json

SHA-256:
  85f34c6d5c933cb757d7a761a21c41f41f6d450f48c2e4249cb853272253f231
```

The corrected image also repeated the synthetic and real-activation gates:

```text
synthetic, including zero-local-length:
  e0b40927aae3c41eaa42a7561f447b028a2de376201c9769020af8f714bc72ef
real 343,727-row activation:
  4035d061084d69a5bda91e9b76f18f5dcc2fb691264e98ecffdd36b504e4957a
```

## Current live gate

Boot CN4 with TP4/DCP4, MTP0, eager mode, DCP query splitting and owner merge
disabled, `i8_ring` held constant, and enough non-KV headroom for streamed
FP32 scoring.

### Corrected boot result

The corrected image reached healthy state with zero restarts:

```text
reference indexer backend: B12X_NON_COMPRESSED_INDEXER
reference cache record:   BF16 x 128 dimensions
main MLA KV:              nvfp4_ds_mla
FP8 RoPE:                 enabled, 368-byte record
available KV memory:      6.69 GiB
KV pool:                  837,953 tokens
max model length:         360,000
max concurrency:          2.33x
CUDAGraphs:               disabled
prefix caching:           disabled
```

The first 499-token live smoke used a 32-token completion cap. It exercised
prefill and decode without a runtime failure, but exhausted all 32 tokens in
the reasoning field and was therefore inconclusive. The preserved rerun used
a finalization-safe cap and passed:

```text
prompt tokens:      511 (cached 0)
completion tokens:  147
finish reason:      stop
final content:      SYSTEM READY
```

Then run, cold and with the frozen request hashes:

1. 250k control once — must be exact;
2. 350k failure seed 1 — must be exact;
3. 350k failure seed 2 — must be exact;
4. 350k failure seed 3 — must be exact.

Any failure stops the experiment.  A pass isolates the accelerated indexer as
part of the causal set but does not yet attribute recovery solely to scorer
math. The stock gate used MTP3/graphs/480k, while this diagnostic requires
MTP0/eager/360k. If all reference rows recover, run frozen r1 once on the
stock RC in the same MTP0/eager/360k posture:

- stock-posture miss: scorer semantics remain the discriminating variable;
- stock-posture pass: MTP3/graph decode selection becomes the lead.

Only after that control may the accelerated scorer be decomposed into
FP8-K-only, FP8-Q-only, and FP8-accumulation-only offline substitutions.

## Checkpoint positional contract

The checkpoint declares:

```text
max_position_embeddings: 1,048,576
rope_parameters:
  rope_type:  default
  rope_theta: 8,000,000
rope_scaling: null
original_max_position_embeddings: null
```

Therefore the observed 250k-pass/350k-fail transition does not straddle a
declared 262,144-position extension boundary in this checkpoint.

## Completed end-to-end result

The corrected reference mode passed the cold 8k sanity gate and frozen 250k
control, then recovered zero of the three frozen 350k failures:

```text
250k control: EXACT  content=738216  cached=0  finish=stop
350k r1:      ABSENT content=27      cached=0  finish=stop
350k r2:      ABSENT content=27      cached=0  finish=stop
350k r3:      ABSENT content=27      cached=0  finish=stop
```

The result is semantic. The server remained healthy with zero restarts and no
OOM after the 16-row correction. Primary evidence:

```text
summary sha256:
  14757653116ea69d717396742333d7c9f376807e40cd21bc778111713d324229
path:
  harness/cn4-evidence-archive/20260727/
  official-reference-runtime-stride-chunk16-75715e51/
  frozen-causal/results/summary.json
```

Conclusion: accelerated FP8 indexer arithmetic is not the sufficient cause of
the deep-retrieval failure. It remains influential because reference r2 and
fixed-stock r2 finalized different wrong answers.

The working `oldest_boundary` positive control used the same NVFP4 MLA KV,
FP8 RoPE, and rank-consistent block-INT8 `i8_ring` settings and passed through
475k. Those
components are therefore not sufficient causes either.

## Needle-inclusion discriminator

The next diagnostic logs merged global selections during decode for the
frozen 350k r1 prompt. The exact ticket-number tokens occupy logical indices:

```text
[137499, 137502)
```

For every active reference-indexer layer and decode call, the trace records
exact-range membership, +/-32-token context membership, and the nearest eight
selected logical indices. It runs after the exact DCP global merge.

```text
source commit:
  103473cdbb6bb0abcc0cd034822206d0dd4caeba
source sha256:
  3900bf2cd28914f532fd0b7a029bc118a86a494b4c7d8a09cdc0672c50f2b8f8
image:
  sha256:739ff8d3eaaf55e6e5ce0d22b2ad9ce210c42a2837af8c76b4adc8bea847e23d
CPU trace gate:
  02e807e7b9f802cca50959df8be72d260e9f678412284400a520c04ab5e14dd1
```

Interpretation:

- if an exact ticket token is selected and the model still returns `27`,
  investigate post-selection index mapping, selected-KV gather, and sparse
  attention;
- if the exact and nearby ranges are absent, verify a layer >=1 reference
  activation and trace shared hidden-state/metadata inputs before proposing
  any selector policy.

## Completed needle-inclusion result

The trace hook first passed a two-token live smoke: one actual decode emitted
exactly 21 records, one per active sparse-indexer layer. The frozen 350k-r1
run then produced a complete 21-layer x 16-decode-call matrix (336 records)
and reproduced the cold semantic failure:

```text
343727 prompt tokens, cached=0, finish=stop, output=27
```

No exact ticket-value token was selected by any layer through layer 38.
Sparse hits began only at layer 42; all three value tokens first appeared
together in one decode call at layer 62. Layer 74 selected at least one value
token in 15/16 calls and all three in 7/16 calls. That is too late to develop
the answer and is consistent with the prior failing exact-selector trace.

This result rejects the post-selection-drop branch as the primary cause.
The attention backend cannot attend the ticket value at the early/middle
layers because the official exact selector does not supply it. The next
boundary proof is therefore upstream: capture the reference mode at a deeper
layer and independently reconstruct its projections, normalization, RoPE,
score reduction, and top-k from the layer input. No new selection policy is
justified until that boundary is closed.

```text
trace analysis:
  2278cb0fd6e0ad87c9c16c6f77da1187c78df23984043c4debd8cd8a04b33751
complete container log:
  8971b3dd9340aa00ba16ac70d75590d2ecba2b459032f071718b39a205eb88fd
frozen row summary:
  edd93bd557aacaf1d62519babcf364eca8ec60a9ce07da41b0d082b2ae5ccdd2
```
