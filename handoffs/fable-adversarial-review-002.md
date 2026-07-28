# Fable adversarial review 002 — v20 deep-context retrieval

Date: 2026-07-27  
Author: Sol  
Host under test: CN4 only  
Production posture: CN3 was not stopped or modified  
Status: 250k control passes; first 350k attempt is INVALID due to diagnostic
scorer transient OOM; the exact resource-fit correction is proved and the
corrected reference image is being rebuilt

## 1. Review request

Please adversarially review the current causal attribution and the proposed
path from the diagnostic oracle to one canonical production fix.

The immediate question is:

> Does replacing only the accelerated FP8 indexer trajectory with the literal
> official GLM scorer restore the three frozen 350k retrieval failures while
> preserving the frozen 250k control?

The reference image keeps the model checkpoint, NVFP4 MLA KV cache, FP8 RoPE,
`i8_ring`, DCP4, prompt bytes, sampling parameters, and production SparkInfer
top-k. It replaces the indexer projection/cache/scoring trajectory with the
literal full-precision GLM implementation. PR #84 is absent.

The current live result is:

| Frozen cell | Stock runtime-stride RC | Official reference |
|---|---|---|
| 250k control | EXACT `738216` | EXACT `738216` |
| 350k r1 | ABSENT, answered `27` | no answer: diagnostic scorer OOM |
| 350k r2 | ABSENT, fabricated `MAINT-2024-0917` | not run: server exited |
| 350k r3 | ABSENT, answered `27` | not run: server exited |

If all three 350k rows recover, the accelerated indexer is causal. That does
not make the slow BF16/FP32 reference implementation the production fix. It
selects the next experiment: locate the first accelerated component that
changes the official candidate set, then correct that component without
adding a selector policy.

If any 350k row remains a miss, the accelerated indexer is not sufficient to
explain the end-to-end failure. In that branch, selector heuristics and FP8
conditioning are not justified as the permanent fix; checkpoint/training
contract evidence becomes the next target.

## 2. Current conclusions, with evidence strength

| Finding | Status | Evidence |
|---|---|---|
| SparkInfer runtime page-table stride bug exists in the old base | proved | Direct old-image GPU proof reads wrong rows 1–16 after cross-width cubin reuse |
| SparkInfer #85 fixes that stride bug | proved | Same direct proof is exact on the new RC |
| #85 restores the frozen long-context failures | refuted | New stock RC passes 250k control and fails all 3×350k |
| Production top-k is inexact on official scores | refuted | Frozen 343,727-row official score vector produces the same set and values in SparkInfer and Torch |
| Official GLM scorer replay is implemented literally | strongly proved at operator level | Independent real-activation fingerprints, exact RoPE, exact top-k set/values |
| Official reference mode is live-wired correctly | proved through 8k | Metadata smoke and cold 8k DCP needle pass with normal finalization |
| Accelerated FP8 indexer is the end-to-end cause | not yet proved | The first frozen 350k reference attempt was invalidated by transient OOM; corrected rerun pending |
| PR #84 `oldest_boundary` is the permanent implementation | rejected | It is retained only as a positive diagnostic control |
| WHT is the missing permanent correction | refuted on current evidence | Prior real-activation and end-to-end tests did not restore the frozen failures |

## 3. Immutable runtime bases

### 3.1 Old accelerated base

```text
image:
  voipmonitor/vllm@sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0
vLLM:
  0c79e41db41f250ccdfc4be92d171960a5787f73
SparkInfer:
  e603f74bb67d0fce547336f1fb73c3c23e8f1887
```

This image contains the cross-width page-table stride defect fixed by
SparkInfer #85.

### 3.2 Runtime-stride release candidate

```text
image:
  voipmonitor/vllm@sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0
vLLM:
  0c79e41db41f250ccdfc4be92d171960a5787f73
SparkInfer:
  c3828fd7f807ce237a9ac36ef033659e6f6b6dd3
installed tiled_topk.py:
  284bd167a971cc6c992c8b2b3ce120000185ef6ffe93be845036e098bfc834f2
```

The vLLM side is unchanged from the old base. The relevant intended semantic
change is SparkInfer #85. Query-split and owner-merge were pinned off during
the stock causal A/B, so the run did not introduce #79/#178 transport
behavior.

### 3.3 Corrected official-reference image

```text
tag:
  glm52-serve:v20-official-reference-runtime-stride-a38fc99a
image ID:
  sha256:3b4a665ea5662b389391b39699d6f83398368fdc7506bf1491446de924e7a6c8
base:
  sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0
reference source commit:
  a38fc99af111c2074eb222a19b8f1f24362fab48
PR #84:
  absent
Dockerfile:
  docker/Dockerfile.v20-official-fullprecision-reference-runtime-stride-20260727
  f343dd989d40939f2fc00a7294311de91b910201c6098b87bfc5fcee685cf807
compose used:
  compose/glm52-v20-official-bf16-reference-causal-20260727.yaml
  318ef6cfa61c5f19c7666829366071bf30339cadab4cec2d2fa7bc4b5f1b897e
```

Overlaid production source is limited to:

```text
glm_official_indexer.py
  166a85072398053236a78a31d8d34e29d8183aeb3809307cbdbde62718819950
deepseek_v2.py
  5b14912dad2b006c7d1fb07eba6c706394e300a2d0e60529dba3557871649014
```

The image also adds a pinned PATH wrapper that forces eager execution and
disables prefix caching:

```text
docker/vllm-official-reference-eager
  eecb84231e30482cd26128dfff932fb255bcfb360cdc90f68ddc9bf4998c9ced
```

## 4. SparkInfer #85 audit and direct proof

### 4.1 Defect

The old tiled top-k compile-cache key allowed a cubin compiled for one
page-table width to be reused at another width. The runtime width changed,
while the two-dimensional CuTe row stride remained baked into the compiled
kernel. Row zero remained correct; later rows could read adjacent columns.
The failure was silent and sensitive to compile/cache population order.

### 4.2 Standalone proof

Harness:

```text
harness/v20_sparkinfer_runtime_stride_gpu_proof.py
  30bd847de08b41166644acfc7a0816f34f81b38c94e6097ca3160ba325f74108
```

The proof compiles a narrow page table, reuses the cubin at a wide runtime
width, and verifies every row.

Old image:

```text
expected: bug-present
observed: bug-present
cubin reused: true
narrow bad rows: []
wide bad rows:   [1, 2, ..., 16]
shared bad rows: [1, 2, ..., 16]
status: PASS
result SHA-256:
  2024fb14a49870537d23cc22dc06359268cc57ef2987cd527fc615aa2195dfdb
```

New RC:

```text
expected: bug-absent
observed: bug-absent
cubin reused: true
narrow bad rows: []
wide bad rows:   []
shared bad rows: []
status: PASS
result SHA-256:
  ec62d4dd499ab16a6028e63a1895bc5510a27ad3d91e8fdf5ffb57aae864e035
```

Conclusion: #85 fixes a real correctness bug, and the direct proof validates
the fix independently of the model.

## 5. Stock RC end-to-end causal result

Configuration:

```text
TP4 / DCP4 / MTP3
max model length 480000
max batched tokens 3072
NVFP4 MLA KV
FP8 RoPE, 368-byte record
i8_ring
exact selector
DCP query split 0
DCP owner merge 0
CKV prefetch depth 0
prefix caching disabled
fresh cache namespace
KV pool 550,144
```

Frozen input pins:

```text
manifest:
  a2ad521f83750b696add479cd91f1b82bb49582761a34a91f85bdf562e15f79f
250k control:
  fde493ea5b921594d239e2a743229d61c9977557057aa49bd1389700d5a56b54
350k r1:
  f0d1c16d816b777f27a3882d9e6b5ef056852684ea155fb11dd845f9e1654ab5
350k r2:
  d5b6755331b634bbabc24486f74925832179eac7842b7a8a7ee225b52b1cdec6
350k r3:
  a50329d3866ba97ead8ae10291cfb8903b8e542f79ea4e985d365f9db7447b46
```

Results:

| Cell | Prompt tokens | Cached | Finish | Output tokens | Content | Verdict |
|---|---:|---:|---|---:|---|---|
| 250k control | 245,497 | 0 | stop | 4 | `738216` | EXACT |
| 350k r1 | 343,727 | 0 | stop | 16 | `... is 27.` | ABSENT |
| 350k r2 | 343,727 | 0 | stop | 25 | `... is MAINT-2024-0917` | ABSENT |
| 350k r3 | 343,727 | 0 | stop | 17 | `... is 27.` | ABSENT |

The run was healthy with zero restarts. Because the 250k control ran first and
passed, all prompts were cold, and all responses stopped normally, the misses
are quality failures rather than a crash, cache hit, or finalization artifact.

This is a clean single-variable #85 A/B. It proves that the stride defect is
not sufficient to explain the frozen deep-context regression.

Evidence:

```text
harness/cn4-evidence-archive/20260727/
  runtime-stride-stock-causal/control-first/

gate.log:
  b88aca66a54ee161411ccf7385e22819c6abfe6eda672eae4ae42fa38aa673e9
boot-before-gate.log:
  e50d25ab34f3f4e2c6da0910bc56f5ab14e5a02f9b92da30f93a82a64c4fab91
boot-after-gate.log:
  c5d592d00e960bc9b9b6281f6ee1a6b8093f4eba67842320a7126fac04672e70
summary.json:
  32336eebfc0216a2dd479a37706813af616ee0bac1237475feaa08243c016c55
rows.json:
  336a0b70c9430ffe4f2646af3e98558eedaa56db31372d52e43247b6ddf8fe84
```

The repeated wrong answer `27` is within-run evidence of a stable failure
morphology. There is no archived old-build r2 response suitable for a
byte-for-byte cross-build determinism claim, so none is made.

## 6. Official reference scorer contract

The diagnostic implementation follows the pinned Transformers
`GlmMoeDsaIndexer.forward` contract:

1. separate BF16 Q and K projections;
2. BF16 LayerNorm on the 128-D K vector;
3. GLM interleaved RoPE on the leading 64 dimensions;
4. reassemble complete 128-D Q/K vectors;
5. literal tensor ranks `Q=[B,S,H,D]`, `K=[B,T,D]`;
6. FP32 `Q @ K.T`;
7. multiply by `128**-0.5`;
8. FP32 ReLU;
9. separate FP32 learned 32-head projection;
10. multiply by `32**-0.5`;
11. FP32 learned-head reduction;
12. causal per-row lengths;
13. production SparkInfer exact top-k at 2,048;
14. rank-local exact selection followed by the existing deterministic DCP
    global candidate merge.

The full score matrix is not retained. Q rows are streamed in bounded chunks;
only each partition's local top-2,048 survives. This is globally exact because
any member of the global top-2,048 must be in its partition's local
top-2,048.

TF32 is disabled only around the FP32 projection/scorer operations and the
prior process setting is restored immediately.

### 6.1 Server-static cache semantics

The mode is enabled only by:

```text
VLLM_GLM_INDEXER_REFERENCE_MODE=official_bf16_v1
```

It owns:

```text
prefix: indexer.k_cache_official_bf16_v1
dtype:  BF16
width:  128 dimensions / 256 bytes per token
```

Production owns:

```text
prefix: indexer.k_cache
dtype:  uint8
width:  132-byte FP8+scale record
```

Prefix, dtype, and width all differ. The two cache meanings cannot be reused
or silently mixed.

### 6.2 Live metadata corrections made after review 001

Fable correctly found that the first implementation would reject real MTP0
decode metadata because vLLM supplies non-speculative decode lengths as
`(B,1)`, not rank 1. The corrected source:

- normalizes `(B,1)` to `(B,)`;
- rejects multi-column/speculative decode lengths;
- rejects nonzero prefill key-window bases;
- retains all prior geometry and cache checks.

These cases are now part of the synthetic gate.

## 7. No-model proof results

### 7.1 Synthetic operator gate

Harness:

```text
harness/v20_glm_official_reference_indexer_gate.py
  915fae6d73d3433e947c9ea5a97d60e7e2a14b864958d4f18d4c3f797adb8f88
```

Passed assertions:

- official interleaved Q/K RoPE exact;
- BF16 cache insertion exact;
- paged gather exact;
- streamed scorer and production top-k exact;
- returned values match source scores;
- TF32 state restored;
- `(B,1)` live decode lengths accepted/normalized;
- unsupported speculative decode shape rejected;
- nonzero key-window base rejected.

Result:

```text
status: PASS
result SHA-256:
  8f84209a23d5b5657a3f20daa1c0b5c2fa3a060bdf3f9a07f7037a1ac84ba18a
```

### 7.2 Frozen 343,727-row real-activation gate

Harnesses:

```text
harness/v20_glm_official_reference_real_activation_gate.py
  c01fba86f59cee1f46aa0cce232a8e2012e7f106cb93d6f336d3c6b315baa913
harness/v20_glm_official_scorer_raw_production_topk_proof.py
  f7046191c7a850312e7f164c9cfcb0b4894e858d918da0d0cf34b492b8ed1cdf
```

Canonical outputs:

```text
trace manifest:
  8f39bfc70173038086cc83d5d84d64b7536ff7595e528c7a851f7bf3f7666186
official score vector:
  d5d70cf7324a22ce52bf13ad985affe7658474d1ff542b70b60afd678439f8fb
top-k indices:
  0d8185a8161a1fb05d81e95f9f2019e64b421bcbcf9be7090756b29d9ec0d0e1
top-k values:
  3617e3aadbea1faf0d8d01e83783697469fd4ba5980b6e4f51b002c9ca8af393
```

Passed assertions:

- literal Transformers tensor ranks;
- Q RoPE exact;
- K RoPE exact;
- official score fingerprint matches the independent prior proof;
- production top-k set and values equal Torch exact top-k;
- every selected value equals its source score bitwise.

Result:

```text
status: PASS
result SHA-256:
  4035d061084d69a5bda91e9b76f18f5dcc2fb691264e98ecffdd36b504e4957a
```

### 7.3 Local no-model evidence

```text
harness/cn4-evidence-archive/20260727/
  official-reference-runtime-stride-a38fc99a/no-model/
```

## 8. Corrected reference boot

Configuration:

```text
TP4 / DCP4 / MTP0
max model length 360000
max sequences 1
max batched tokens 3072
NVFP4 MLA KV
FP8 RoPE, 368-byte record
i8_ring
official_bf16_v1 reference indexer
DCP query split 0
DCP owner merge 0
CKV prefetch depth 0
prefix caching disabled
eager execution
compile-cache reuse disabled
GMU 0.950
fresh cache namespace
```

Observed healthy state:

```text
reference backend:
  B12X_NON_COMPRESSED_INDEXER
main KV format:
  nvfp4_ds_mla
FP8 RoPE:
  enabled, kv_gmem_stride=368
model load:
  82.54 GiB/GPU
profiled peak activation:
  0.88 GiB
CUDAGraph memory:
  0
available KV memory:
  6.69 GiB
KV pool:
  837,953 tokens
maximum model length:
  360,000
maximum concurrency:
  2.33x
restart count:
  0
```

The lower max length and eager/MTP0 posture are diagnostic requirements, not
production recommendations. All frozen prompts fit below 360k.

## 9. Live gates

### 9.1 First micro-smoke — preserved inconclusive result

The first 499-token smoke set `max_tokens=32`. It exercised prefill and decode
without error but used all 32 tokens in `reasoning`:

```text
cached: 0
finish: length
content: empty
completion tokens: 32
```

This is a budget artifact, not a reference-indexer failure. It is preserved
rather than deleted.

### 9.2 Finalization-safe micro-smoke

The same 499-token prompt was rerun with sufficient completion budget:

```text
prompt tokens: 511
cached tokens: 0
completion tokens: 147
finish: stop
content: SYSTEM READY
status: PASS
```

Harness:

```text
harness/v20_reference_live_micro_smoke.py
  cbc0f5dcd2c1c12883c5623d0439968b92a94f6d96f488c8f68052fa60614ee7
```

### 9.3 Cold 8k DCP needle

```text
target: 8,000
actual prompt: 7,847
needle position: approximately 40%
cached tokens: 0
completion tokens: 92
finish: stop
content: 738216
elapsed for needle response: 60.69 s
arithmetic side check: pass
coherence side check: pass
status: PASS
```

The long wall time of the complete gate came from the coherence generation,
not a hang. Prometheus showed continuous generation-token progress and one
running request; CN4 remained healthy throughout.

## 10. Frozen reference gate — first attempt

Harness:

```text
harness/v20_run_causal_gate.py
  f0ebfbe0bf25410ffb91662592f3ec7c12659ddd65f34074b7361f29432cd0d3
```

Current result:

```text
pass-250k-ctl:
  verdict: EXACT
  content: 738216
  prompt tokens: 245,497
  cached tokens: 0
  completion tokens: 4
  finish: stop
  elapsed: 362 s
  gate: PASS
```

The first 350k row produced no model answer. At 248,832 computed prompt
tokens, rank 3 failed in `_select_local`:

```text
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 492.00 MiB.
GPU 3 free: 500.69 MiB.
failure operation:
  per_head = torch.matmul(
      q[start:end].float().unsqueeze(0),
      keys_fp32.T.unsqueeze(0).unsqueeze(1),
  )
```

The engine subsequently timed out waiting for the failed worker and exited.
r2/r3 therefore received connection resets and were not executed.

The harness's generated summary says `REFUTED`, but that label is not
semantically valid for this outcome: there is no 350k model verdict. The
correct run verdict is:

```text
INVALID — 250k control passed; 350k r1 produced no model answer because the
diagnostic scorer OOMed allocating 492 MiB with 500.69 MiB physically free.
```

The OOM is a diagnostic-oracle resource-fit defect, not evidence for or
against the accelerated-indexer causal hypothesis. It confirms the transient
headroom risk identified in `fable-adversarial-response-001.md`.

Archived evidence:

```text
VERDICT.txt:
  a1d1319f5ba4c069b83bb13f2c160ad59d32941fe1e211fb3c4377e2edcea350
container.log:
  30a5fbaec6cee7b27acd0709634922ceca6fae045b6ff112996f58202dad3289
container-inspect.json:
  13687d2da29ad5886f5ce0b1f03767b459cc32f301c085c391ca60fc5dd72d2f
250k response:
  4469cb63d47d79699034aaea3813ca0eaabb5f7786ef143cd820409a95949412
```

### 10.1 Resource-fit correction proved

The live reference operator used `q_chunk_rows=64`. The failed
`per_head[S,H,T]` tensor scales linearly with that value. Reducing the
diagnostic query chunk bounds the transient without changing prompts, model
weights, cache contents, scorer formula, selector, or DCP merge.

The candidate `q_chunk_rows=16` passed an explicit numeric equivalence gate
against 64-row execution at production DCP-local indexer geometry:

```text
rows:       64
key rows:   85,932
heads:      32
head dim:   128
top-k:      2,048
seeds:      2026072701, 2026072702, 2026072703
```

All three cases had:

- identical top-k membership;
- bit-identical selected score values;
- zero changed rows;
- maximum absolute score delta `0.0`.

Peak allocated bytes fell from 851,916,288–852,047,360 at chunk 64 to
464,754,176–464,885,248 at chunk 16. After first-use compilation, measured
candidate time was 4.8–4.9 ms versus 4.5–4.8 ms for the 64-row reference.
The proof therefore reduces the diagnostic transient by about 387 MiB
without changing selected indices or scores.

The exact evidence is:

```text
harness/v20_glm_reference_chunk_equivalence.py
  4813229f7e72b5cfbf92e53520da438ee604b6dbbb9584b6d1c606aaeba92d0f

harness/cn4-evidence-archive/20260727/
  official-reference-runtime-stride-a38fc99a/resource-fit/chunk16-vs64.json
  c71e3b411025c0e5d71fee7498ad84020d035441db6d185cb95f62034c412a19
```

The implementation changes only the default row chunk:

```text
reference commit:
  75715e519b73b712873e488f7c42c8d61d14dbbc
glm_official_indexer.py:
  2f73eac49aa199f884bfebaa49c0449f033af665f26a6f48c386fb093ecdbe35
```

The final derived image is:

```text
tag:
  glm52-serve:v20-official-reference-runtime-stride-chunk16-75715e51
image ID:
  sha256:899e64cc6098407d1e41bca8db53f70ea60f31009b812872e4690540798ded1a
base digest:
  sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0
compose:
  b6de5bff44ad9777f08b95d6e7a1bbce647d49524e8d77d928d7f851b4e4ba31
```

All no-model gates were rerun inside that exact image:

```text
synthetic + zero-local-length:
  e0b40927aae3c41eaa42a7561f447b028a2de376201c9769020af8f714bc72ef
real 343,727-row activation:
  4035d061084d69a5bda91e9b76f18f5dcc2fb691264e98ecffdd36b504e4957a
chunk16-versus-64:
  85f34c6d5c933cb757d7a761a21c41f41f6d450f48c2e4249cb853272253f231
```

The corrected boot must rerun the cheap live smoke and 8k needle before the
frozen 250k/350k gate. The valid 250k result remains archived but is not
silently combined across image identities.

Remote evidence root:

```text
/home/derek/proof-results/20260727/
  official-reference-runtime-stride-a38fc99a/
```

Local evidence already copied:

```text
harness/cn4-evidence-archive/20260727/
  official-reference-runtime-stride-a38fc99a/
```

## 11. Invalid and inconclusive evidence retained

### 11.1 First reference boot

The earlier old-base reference boot ran no requests. Startup failed because
the extra BF16 reference cache was not fully represented in the final generic
KV allocation budget, causing the last 100 MiB KV allocation to OOM.

It also contained the `(B,1)` decode-length bug that would have failed the
first live decode. This boot is invalid for quality attribution.

```text
/home/derek/proof-results/20260727/
  official-reference-mode-v1/causal-boot-eager-invalid/

VERDICT.txt:
  150733a1198731ebd35d2b7b22728aff6530be0e6ad6003a99f8dc54c39de4ad
container.log:
  97be728d9a6c1424e4a2f54581032cd0cf22b167c4146c760b71c4eee4d3b789
```

### 11.2 Micro-smoke 32-token cap

As recorded in §9.1, this is an explicitly preserved budget-exhaustion
artifact. It must not be counted as a model failure.

## 12. Hypothesis ledger

### H1 — SparkInfer cross-width page-table stride corruption

```text
Component bug: confirmed
Sufficient end-to-end cause: refuted
```

The new RC fixes the direct defect while reproducing all three frozen 350k
misses.

### H2 — top-k selection kernel is mathematically inexact

```text
Refuted for the frozen official score row.
```

Production SparkInfer top-k and Torch exact top-k return identical membership
and values on the full 343,727-key row.

### H3 — selector policy alone is the permanent defect

```text
Unsupported.
```

`oldest_boundary` is a useful positive control because it changes which
candidate ages survive a distorted score field. It does not establish the
checkpoint's intended scoring contract and is not the target end state.

### H4 — official WHT conditioning is the missing GLM step

```text
Refuted as a sufficient correction by prior current-path experiments.
```

An orthogonal transform can reduce FP8 outlier error but does not guarantee
the trained score ordering after quantization. Current evidence does not
justify shipping it as the canonical correction.

### H5 — accelerated FP8 indexer arithmetic/cache trajectory

```text
Refuted as a sufficient cause by the corrected end-to-end reference gate.
```

Before the current boot, the accelerated layer-0 selection retained only
1,905 of the official top 2,048 candidates on a frozen failing row. The
official scorer/top-k oracle is exact on the same row. The full BF16/FP32
reference path nevertheless recovered zero of three frozen 350k failures
while preserving the 250k control. Scorer arithmetic changes some output
trajectories, but is not sufficient to restore retrieval.

### H6 — NVFP4 MLA KV, FP8 RoPE, or `i8_ring`

```text
Not sufficient causes under the tested stack.
```

The working `oldest_boundary` positive control used the same NVFP4 MLA KV,
`KV_FP8_ROPE=1`, and lossless `i8_ring` transport and passed the frozen gate
plus the randomized 50k--475k ladder. These components can still affect
numerical margin, but cannot be sufficient causes of the exact-selector
failure.

### H7 — prompt cache, request order, finalization, or harness scoring

```text
Ruled out for the frozen gate.
```

All prompts differ at character zero, prefix caching is disabled, cached
tokens are checked per request, the 250k control runs first, and success
requires exact finalized content with `finish_reason=stop`.

## 13. Proposed permanent-fix path if the reference rows recover

Do not upstream the diagnostic BF16/FP32 oracle as the production operator.
Use it as the ground truth for a component ladder on the same frozen
activation.

Recommended order:

1. Keep official BF16 Q and all post-projection scorer math fixed.
2. Replace only the official BF16 K cache with the production FP8+scale K
   record and measure:
   - recall/Jaccard@2,048;
   - score-weighted false negatives;
   - frozen needle inclusion;
   - repeatability across cache/compile order.
3. If K-cache quantization creates the first material divergence, test
   canonical scale derivation/rounding and cache packing/unpacking fixes.
4. If K remains aligned, substitute the accelerated Q projection/quantization
   boundary next.
5. Then substitute fused scorer arithmetic, logical/physical index mapping,
   and DCP merge one at a time.
6. Stop at the first substitution that reproduces the missing candidates.
7. Implement the smallest correction at that boundary.

The permanent production target must have:

- one canonical scoring behavior;
- no historical out-of-bounds behavior;
- no age/boundary selector policy;
- no cache-mode ambiguity;
- deterministic DCP4 exactness;
- an operator-level proof against the official scorer;
- an end-to-end frozen and randomized quality gate.

If FP8 quantization cannot meet the quality gate after canonical scale and
rounding corrections, the next principled design is deterministic
broad-candidate selection followed by higher-precision reranking. It is not a
historical overflow emulation.

## 14. Required promotion evidence after a minimal correction

1. Frozen 250k control and all 3×350k failures exact.
2. Randomized cold ladder from 50k through 475k.
3. Exact finalized content; score `content`, `reasoning`,
   `reasoning_content`, and serialized message, but require final content.
4. `cached_tokens=0` for cold cells.
5. n=3 KLD against the accepted baseline, with retained raw samples.
6. Prefill/decode benchmark matrix with cache metrics.
7. At least 500k KV tokens at 480k max length in the production posture.
8. CUDA-graph and eager checks where applicable.
9. Clean restart and cache-semantics checks.
10. CPU and GPU proofs checked into the PR.
11. Minimal, reviewable upstream PR against the latest v20 base.

## 15. Specific adversarial questions

1. Is any material scorer operation missing or ordered incorrectly relative
   to the official GLM reference?
2. Does the streamed local-top-k plus DCP union proof overlook any index-space
   or causal-length edge case?
3. Can MTP0/eager/max-360k change model semantics, rather than only execution
   posture, in a way that invalidates the causal inference?
4. Does keeping NVFP4 MLA KV, FP8 RoPE, and `i8_ring` active sufficiently
   isolate the indexer if all frozen 350k rows recover?
5. Is there any hidden compile/cache key that can mix
   `indexer.k_cache_official_bf16_v1` with the production cache despite the
   distinct prefix/dtype/width?
6. Is the proposed component-ladder order optimal, or is there a cheaper
   single-boundary proof that separates FP8 K-cache quantization from Q-side
   and scorer arithmetic?
7. What evidence would be required before declaring a corrected FP8 path
   aligned with the checkpoint, rather than merely closer on this gate?
8. If the reference rows do not recover, what is the cheapest falsifiable
   checkpoint/training-contract experiment?

## 16. Supporting documents

Primary:

- `design/v20-runtime-stride-rc-longctx-causal-20260727.md`
- `design/v20-official-fullprecision-reference-mode-20260727.md`
- `design/v20-official-scorer-raw-proof-20260727.md`
- `design/v20-official-scorer-topk-proof-20260727.md`
- `fable-adversarial-review-001.md`
- `fable-adversarial-response-001.md`
- `fable-adversarial-response-002.md`

Historical/component context:

- `design/v20-indexer-fp8-precision-conditioning-plan.md`
- `design/v20-indexer-oldest-boundary-permanent-fix.md`
- `design/v20-indexer-oldest-boundary-permanent-fix.md`
- `design/v20-longctx-first-divergence-20260726.md`
- `design/v20-nf3-long-context-causal-status.md`
- `design/github-issue-182-cross-layer-runtime-trace-20260727.md`
- `design/github-issue-182-segmented-exact-refutation-20260727.md`
- `design/github-issue-182-wht-refutation-20260726.md`
- `design/v20-pr84-kld-n3-report-20260727.md`

Evidence archives:

- `harness/cn4-evidence-archive/20260727/runtime-stride-stock-causal/`
- `harness/cn4-evidence-archive/20260727/official-reference-runtime-stride-a38fc99a/`

## 17. Reviewer response format requested

Please return:

1. `ACCEPT`, `ACCEPT WITH CONDITIONS`, or `REJECT` for the current causal
   experiment.
2. Any invalid inference, hidden confounder, or missing proof, ordered by
   severity.
3. A decision tree for the all-3-pass, partial-pass, and zero-pass outcomes.
4. The cheapest next discriminating proof.
5. A critique of the proposed permanent component ladder.
6. Any change required before these results are summarized publicly or used
   to update issue #182.

## 18. Adversarial response 003 and adopted conditions

Fable returned `ACCEPT WITH CONDITIONS` in
`fable-adversarial-response-003.md`. The following conditions are adopted.

### 18.1 Posture-matched stock control before causal attribution

The current stock and reference live gates are not a single-variable A/B:

| Variable | Stock RC gate | Reference gate |
|---|---|---|
| MTP | 3 | 0 |
| CUDA graphs | enabled | eager |
| maximum model length | 480k | 360k |

This matters because `next_n` changes decode metadata and kernel routes, and
the frozen failures form during the first decode tokens. If the corrected
reference gate restores r1/r2/r3, the immediate claim is only:

> The union of official scorer semantics plus MTP0/eager reference posture
> restores the frozen rows.

Before attributing the recovery to scorer math, run the stock runtime-stride
RC in the reference posture (`MTP=0`, eager, 360k) against frozen r1. An
archived stock-MTP0 350k miss is acceptable only if prompt bytes, sampling,
cache state, and relevant runtime flags are identical and pinned.

- If posture-matched stock r1 still fails, scorer semantics remain the
  discriminating variable.
- If it passes, MTP3/graph decode selection becomes the lead and the FP8
  component decomposition must not begin.

### 18.2 Clean-RC activation recapture

The prior `1,905/2,048` accelerated-versus-official overlap measurement was
captured on the SparkInfer runtime containing the pre-#85 stride defect. It
remains useful as historical evidence but cannot rank current components or
support a public causal claim. Re-capture the frozen failing activation on
the clean runtime-stride RC before publishing a current overlap figure.

### 18.3 Offline component decomposition

If both the corrected reference gate and posture-matched stock control
support scorer causality, exploration should be offline first:

1. official scorer with production FP8+scale K substitution only;
2. official scorer with production FP8 Q substitution only;
3. official scorer with production FP8 accumulation substitution only.

For each cell, preserve recall/Jaccard@2,048, score-weighted false negatives,
needle rank/cutoff margin, and repeated cache/compile order. Only the winning
minimal cell advances to one live confirmation boot. This replaces a
boot-heavy component ladder without changing the final proof requirement.

### 18.4 Scope and zero-recovery branch

Any recovery is causal only for the frozen prompt set until the randomized
50k–475k ladder passes. If the corrected reference gate produces zero
semantic recoveries, first inspect the checkpoint's `rope_scaling` and
`original_max_position_embeddings` contract because the 250k/350k boundary
straddles 262,144. Then move to deeper-layer official replay; do not add a
selector heuristic.

### 18.5 Additional edge proof

The streamed DCP selection proof should include a zero-local-length synthetic
cell. The mode already uses a distinct BF16 cache prefix/dtype/record width
and compile-cache reuse is disabled, so no production/reference cache mixing
has been identified.

## 19. Corrected official-reference result and next discriminator

The resource-corrected image completed the frozen gate:

| Cell | Result | Cached | Finish | Final content |
|---|---|---:|---|---|
| 250k control | EXACT | 0 | stop | `738216` |
| 350k r1 | ABSENT | 0 | stop | `27` |
| 350k r2 | ABSENT | 0 | stop | `27` |
| 350k r3 | ABSENT | 0 | stop | `27` |

Primary result:

```text
14757653116ea69d717396742333d7c9f376807e40cd21bc778111713d324229
  harness/cn4-evidence-archive/20260727/
  official-reference-runtime-stride-chunk16-75715e51/
  frozen-causal/results/summary.json
```

The official scorer changes some trajectories: fixed stock r2 fabricated
`MAINT-2024-0917`, whereas the reference returned `27`. It therefore remains
numerically relevant, but it is not the sufficient cause.

Condition A is not run on this branch because there was no reference recovery
to attribute. The next image performs a post-DCP-merge decode trace. For each
active indexer layer and decode call it records:

- whether the exact ticket-number token range `[137499,137502)` is selected;
- whether any selected token lies in a +/-32-token context window;
- the nearest selected logical indices and their scores.

This splits the next investigation:

- exact number selected but retrieval fails: trace sparse-attention
  index consumption, selected-KV gather, and attention precision;
- exact/context range absent: verify the official reference at layer >=1 and
  trace the shared hidden-state/metadata inputs or training-time selection
  contract.

Trace pins:

```text
source commit:
  103473cdbb6bb0abcc0cd034822206d0dd4caeba
trace source:
  3900bf2cd28914f532fd0b7a029bc118a86a494b4c7d8a09cdc0672c50f2b8f8
trace image:
  sha256:739ff8d3eaaf55e6e5ce0d22b2ad9ce210c42a2837af8c76b4adc8bea847e23d
CPU gate:
  02e807e7b9f802cca50959df8be72d260e9f678412284400a520c04ab5e14dd1
```

## 20. Needle-inclusion result: official exact selection is late

The frozen 350k-r1 trace completed cold and reproduced the semantic failure:

```text
prompt_tokens: 343727
cached_tokens: 0
finish_reason: stop
output_tokens: 16
content: The maintenance ticket number for the Facility 27 compressor is **27**.
```

The fail-closed analyzer accepted a complete matrix of 336 records:
21 active sparse-indexer layers x 16 decode calls. The exact ticket-value
range is `[137499,137502)`.

| Layer range | Exact ticket-value behavior |
|---|---|
| 0--38 | no value token selected in any decode call |
| 42 | token 137501 in 1/16 calls |
| 50 | token 137501 in 1/16 calls |
| 54 | tokens 137499/137500 across 1/16 calls |
| 62 | an exact hit in 2/16 calls; all three together in 1/16 |
| 70 | token 137500 in 2/16 calls |
| 74 | an exact hit in 15/16 calls; all three together in 7/16 |

The earliest broad selection is therefore late in the network. This matches
the failing exact-selector trajectory: the value is not provided to sparse
attention early enough to develop the answer. The primary failure is not a
logical-index-to-page-table conversion or selected-KV gather dropping an
already-selected early candidate.

This also sharpens the zero-recovery branch. The full-precision reference
mode reconstructs the installed Transformers scorer contract directly and
uses it from layer zero, but the shared quantized model trajectory still does
not make the value competitive until the final layers. The next proof must
boundary-capture a deeper reference layer and compare the reference mode's
`hidden_states`, `q_resid`, projected/normalized/RoPE Q/K, and learned head
weights with the installed Transformers equations. If that boundary is
correct, the remaining discrepancy is not another selector implementation
bug; it is the relationship between the checkpoint/runtime precision
trajectory and the training-time selection contract.

Primary evidence:

```text
2278cb0fd6e0ad87c9c16c6f77da1187c78df23984043c4debd8cd8a04b33751
  harness/cn4-evidence-archive/20260727/
  official-reference-needle-trace-103473cd/
  frozen-r1/trace-analysis.json

8971b3dd9340aa00ba16ac70d75590d2ecba2b459032f071718b39a205eb88fd
  .../frozen-r1/container-complete.log

edd93bd557aacaf1d62519babcf364eca8ec60a9ce07da41b0d082b2ae5ccdd2
  .../frozen-r1/results/summary.json
```
