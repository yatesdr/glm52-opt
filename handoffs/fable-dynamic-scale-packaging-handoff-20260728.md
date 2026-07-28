# Fable packaging handoff: calibration-free NVFP4 MLA dynamic token scale

Date: 2026-07-28  
Owner: Sol → Fable  
Status: **review-image and draft-PR packaging may begin**. The implementation
has passed its causal, randomized long-context, and KLD n=3 gates. Matched
decode and restart/repeatability are still qualification gates; do not label
the image production-ready until the completion addendum lands.

## 1. Packaging decision

Package the dynamic per-token scale implementation as the primary v20 review
candidate. This is not a selector workaround and does not use
`oldest_boundary`, `bounded_compat`, or the diagnostic BF16 reference scorer.
The tested runtime uses the stock v20 exact selector.

Suggested review tag (not yet pushed):

```text
ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-nvfp4-dynamic-scale-review-20260728
```

The review image must remain digest-pinned to the runtime-stride RC and must
carry the exact source hashes from the existing fail-closed Dockerfile.

## 2. What changed

The 368-byte `nvfp4_ds_mla` record previously quantized the 512-value NoPE
latent using a server-wide or per-layer outer scale. A fixed outer scale of
1.0 can place shallow-layer E4M3 group scales in the subnormal range, losing
information before sparse attention reads the record.

The new mode derives an FP32 outer scale independently for every token:

```text
s_t = token_amax / (6 * 448)
```

It stores `s_t` in four bytes that were already reserved padding:

```text
[   0,256) E2M1 NoPE values
[ 256,288) 32 E4M3 group scales
[ 288,292) FP32 RoPE scale
[ 292,296) FP32 per-token NoPE outer scale s_t   NEW
[ 296,304) zero padding
[ 304,368) E4M3 RoPE values
```

The record remains exactly 368 bytes. Page layout, DCP transport size,
offload format, prefix hashing, and KV capacity economics do not change.
Decode and MG-prefill readers consume the in-record scale. The mode is
server-static and joins the SparkInfer compile identity, with writer/decode/
MG-prefill compile-spec version bumps preventing stale cubin reuse.

Enable it with:

```text
VLLM_NVFP4_MLA_DYNAMIC_SCALE=1
KV_FP8_ROPE=1
--kv-cache-dtype nvfp4_ds_mla
```

Dynamic mode is mutually exclusive with
`VLLM_NVFP4_MLA_SCALES_FILE`. It fails closed with a non-368-byte record,
without FP8 RoPE, or against a SparkInfer build lacking the required writer
and reader arguments. A fresh cache namespace is mandatory because old
records have zero at `[292,296)`.

## 3. Exact source and image pins

| Component | Pin |
|---|---|
| Base image | `voipmonitor/vllm@sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0` |
| Base vLLM | `0c79e41db41f250ccdfc4be92d171960a5787f73` |
| Base SparkInfer | `c3828fd7f807ce237a9ac36ef033659e6f6b6dd3` |
| Dynamic vLLM commit | `91dff5a9e8609cf899be80994494ddd54a55e70c` |
| Dynamic SparkInfer commit | `0d9aead951cca445a77315b4151d26d49b1758b5` |
| Built CN4 image ID | `sha256:db82fdcb5756d4a547853ba1330538bdd8a3dc0c6443c29bc49ba77b69b51cd1` |
| Dockerfile SHA256 | `9ccd5aef22cd68d88490fd26c0cf1b574d156b32622548763d00c166a50eb55c` |
| Compose SHA256 | `f6899caa3ab7c2b4c5d37654f79da7dfd7ec82d383f83242c9931bd1362f28ee` |

Packaging inputs:

- `docker/Dockerfile.v20-nvfp4-dynamic-token-scale-20260727`
- `compose/glm52-v20-nvfp4-dynamic-token-scale-prodposture-20260727.yaml`
- `design/nvfp4-dynamic-second-level-scale-spec.md`
- `design/nvfp4-dynamic-second-level-scale-phaseA-addendum.md`

The Dockerfile pins all 13 overlaid production source files and the new
writer test by SHA256, runs `py_compile`, and asserts the dynamic API and
record offset at build time.

## 4. Tested production posture

```text
MODEL_FAMILY=glm52-hybrid
TP=4
DCP=4
MTP=3
MAX_MODEL_LEN=480000
MAX_BATCHED_TOKENS=3072
GRAPH=64
GPU_MEMORY_UTILIZATION=0.9848
KV_FP8_ROPE=1
VLLM_NVFP4_MLA_DYNAMIC_SCALE=1
F8_DMA=i8_ring
NCCL_P2P_LEVEL=PXB
SPARKINFER_NSA_TOPK_SELECTION_POLICY=exact
VLLM_DCP_QUERY_SPLIT=0
VLLM_DCP_TOPK_OWNER_MERGE=0
VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=0
prefix caching disabled
fresh cache namespace
```

The container reached healthy state with zero restarts and a 550,144-token
KV pool. The matched static-calibrated boot reported 549,888 tokens. The
256-token difference is allocator/block rounding, not record growth; both
modes use the same 368-byte ABI.

Live SM120 JIT metadata proves:

```text
kernel=attention.mla.nvfp4_fp8_rope_kv_cache
version=3
per_token_scale=true
record_bytes=368
```

Writer compile evidence SHA256:
`776a8a804c3935991e1cdab1d7a3719d4ddcf6e957d483bc2f166b9822cb2e28`.

## 5. Completed gates

### 5.1 Source, compile, and reader/writer correctness

| Gate | Result |
|---|---|
| Independent review of agent-threaded API/traits/smem/compile-key layer | PASS |
| Fail-closed derived-image build and source hashes | PASS |
| Python compile/import | PASS |
| Dynamic writer ABI/positioning/zero-token tests | 4/4 PASS |
| Production decode reader fed by dynamic writer records | PASS |
| Production MG-prefill reader fed by dynamic writer records | PASS |
| Static and dynamic reader coverage in broader suite | PASS |
| Full selected SparkInfer suite | 52 passed, 4 warnings |
| Dynamic + static scales file negative boot | refused as required |
| Dynamic + `KV_FP8_ROPE=0` negative boot | refused as required |

The reader E2E additions currently exist as an uncommitted modification to:

```text
workspace/b12x-nvfp4-dynamic-scale/tests/attention/test_attention_mla_kv_cache.py
```

Do not lose this diff while preparing the SparkInfer PR. It parameterizes
both the production multisplit decode and multitile MG-prefill tests across
static and dynamic records.

### 5.2 Frozen causal retrieval gate

Production posture, cold requests, prefix-cache delta zero:

| Row | Actual prompt tokens | Stock result | Dynamic result |
|---|---:|---|---|
| 250k control | 245,497 | EXACT | EXACT |
| 350k r1 | 343,727 | ABSENT | EXACT |
| 350k r2 | 343,727 | ABSENT | EXACT |
| 350k r3 | 343,727 | ABSENT | EXACT |

All recovered rows returned final content `738216`, `finish_reason=stop`,
four output tokens, and `cached_tokens=0`.

Verdict: **3/3 previously failing prompts recovered while the control
remained exact.**

Evidence:

```text
harness/cn4-evidence-archive/20260728/
  nvfp4-dynamic-token-scale-causal-gate-v1/
```

Summary SHA256:
`cb914693595a71252ea39d28e1a6e602f99c3b6e377044a9c7b32686d03ae4db`.

### 5.3 Independent randomized depth ladder

Every request was cold, finalized, exact, and passed the arithmetic,
coherence, and degeneration checks.

| Target | Actual context | Dynamic | Static-calibrated time | Dynamic time | Delta |
|---:|---:|---|---:|---:|---:|
| 50k | 49,098 | PASS | 36.88 s | 37.57 s | +1.87% |
| 150k | 147,273 | PASS | 118.46 s | 119.80 s | +1.13% |
| 250k | 245,503 | PASS | 213.11 s | 215.08 s | +0.92% |
| 300k | 294,619 | PASS | 265.15 s | 267.10 s | +0.74% |
| 350k | 343,735 | PASS | 319.31 s | 322.44 s | +0.98% |
| 475k | 466,493 | PASS | 468.62 s | 472.62 s | +0.85% |

All six rows returned `738216`, `finish_reason=stop`, and
`cached_tokens=0`. All timing deltas remain inside the precommitted 2%
prefill budget.

Evidence:

```text
harness/cn4-evidence-archive/20260728/
  nvfp4-dynamic-token-scale-randomized-ladder-v1/
```

Summary SHA256:
`f10cb1caaf4beadfd68cc996de4a05c70b26d0ec9de13fbecbcf455f94d0fb30`.

### 5.4 Preliminary decode

One production-posture dynamic boot, C=1, 512 output tokens, three
consecutive samples:

| Run | tok/s | MTP acceptance |
|---:|---:|---:|
| 1 | 57.26 | 0.4389 |
| 2 | 67.25 | 0.5099 |
| 3 | 71.68 | 0.5679 |

Mean is 65.40 tok/s with sample SD 7.39. The rising rate and acceptance
show a warmup effect, so this is smoke evidence only. Do not use it for the
1% decode budget until the same-image static-calibrated baseline and explicit
warmup protocol complete.

Evidence:

```text
harness/cn4-evidence-archive/20260728/
  nvfp4-dynamic-token-scale-decode-v1/
```

## 6. KLD n=3 — PASS

The matched matrix uses the same dynamic-capable image for both arms, TP4/
DCP1/eager, max length 4096, one pinned 2,048-token window (2,047 scored
positions), stride 512, and the same BF16 reference logits:

```text
KL(BF16 reference || candidate)
```

Baseline arm:

```text
VLLM_NVFP4_MLA_SCALES_FILE=
  /opt/vllm/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json
VLLM_NVFP4_MLA_DYNAMIC_SCALE=0
```

Candidate arm:

```text
VLLM_NVFP4_MLA_DYNAMIC_SCALE=1
VLLM_NVFP4_MLA_SCALES_FILE absent
```

Validated results:

| Arm | Run | Mean KLD | Compile proof |
|---|---:|---:|---|
| static calibrated | 1 | 0.1457421454 | `per_token_scale=false` |
| static calibrated | 2 | 0.1511394947 | `per_token_scale=false` |
| static calibrated | 3 | 0.1418014718 | `per_token_scale=false` |
| dynamic per-token | 1 | 0.1399969809 | `per_token_scale=true` |
| dynamic per-token | 2 | 0.1403845224 | `per_token_scale=true` |
| dynamic per-token | 3 | 0.1367254488 | `per_token_scale=true` |

Static-calibrated n=3 summary:

```text
mean      0.146227703949
sample SD 0.004687909264
```

Dynamic per-token n=3 summary:

```text
mean      0.139035650720
sample SD 0.002010055173
```

Paired `dynamic - static` deltas:

```text
-0.00574516
-0.01075497
-0.00507602
```

Mean paired delta: `-0.00719205`, or **4.92% lower KLD** than the static
calibrated mean.

For historical context, the prior `oldest_boundary` image measured
`0.16044075 ± 0.00297924`. Dynamic is 13.34% lower than that mean, and
static calibration is 8.86% lower. The historical comparison is not a
single-image A/B; it uses the same BF16 reference, tokens, runner, TP4/DCP1
eager posture, and KLD direction on an older image.

This 2,048-token KLD cell is not selector-sensitive because the selector
budget is also 2,048. It is scale/record-path no-regression evidence. The
frozen 350k gate and randomized ladder supply selector-active deep-context
evidence.

Runner SHA256:
`ac8e57f67194a3e64a779eed54494810e81cca6459283e50baf242d19c714ce0`.
Aggregate summary SHA256:
`996c2a58940bae4c16424a2176e5f9605fe4b2fd94f6f288e79a0b31a06ab579`.
The matrix failed closed unless every run emitted the full 2,047-position
result and matching writer compile metadata. Complete evidence is archived
at:

```text
harness/cn4-evidence-archive/20260728/
  nvfp4-dynamic-token-scale-kld-n3-v1/
```

## 7. Remaining promotion gates

These do not block preparing a review image or opening draft PRs. They do
block a production-ready label:

1. Run a same-image, explicit-warmup static-calibrated decode baseline and
   compare to dynamic; precommitted decode budget is no worse than 1%.
2. Restart the dynamic image on its declared fresh/server-static cache
   posture; re-prove writer compile identity, health, zero restarts, KV pool,
   and a frozen 350k exact result.
3. Record repeatability/entry-margin evidence required by the promotion
   plan. Do not substitute a binary pass for a margin measurement where the
   trace instrument is available.
4. Push the exact local image to GHCR, record the registry digest, then
   update this handoff and both PR descriptions with that immutable digest.

## 8. Draft PR topology

Prepare two functional PRs and one packaging/reproduction update.

### SparkInfer PR

Suggested title:

```text
mla: add self-describing per-token outer scales to 368-byte NVFP4 records
```

Scope:

- derive and store FP32 `s_t` at `[292,296)`;
- quantize NoPE groups relative to `s_t`;
- gather and apply `s_t` in decode and MG-prefill readers;
- thread the server-static trait through API, smem, kernels, and compile
  identity;
- bump writer/decode/MG-prefill compile-spec versions;
- include writer ABI tests and the currently uncommitted production-reader
  parameterization.

### vLLM PR

Suggested title:

```text
mla: wire dynamic per-token outer scaling for B12X NVFP4 KV cache
```

Scope:

- add `VLLM_NVFP4_MLA_DYNAMIC_SCALE=1`;
- enforce the 368-byte FP8-RoPE record contract;
- make static calibration and dynamic scaling mutually exclusive;
- pass `per_token_scale` to both normal writes and DCP re-quantization;
- pass `latent_scale_per_token` to decode/extend kernels;
- fail closed against incompatible SparkInfer versions.

### Packaging/reproduction update

Include:

- immutable image digest;
- exact enablement flags;
- fresh-cache warning;
- causal 4-row table;
- randomized 6-row table;
- final KLD n=3 table from §6;
- matched decode and restart results;
- explicit statement that the selector remains stock `exact`.

## 9. Claim boundary for review

Supported now:

> A self-describing per-token FP32 outer scale in the existing 368-byte
> NVFP4 MLA record restores the frozen 350k failures 3/3 and passes an
> independent cold ladder through 466,493 tokens on the stock v20 exact
> selector, while preserving the record width and remaining within the
> precommitted 2% prefill budget.

Not yet supported:

- production-ready;
- decode overhead within 1%;
- restart/reuse qualification;
- results on hardware or model families other than the tested GLM-5.2 NF3
  TP4/DCP4 Blackwell posture.

## 10. Immediate Fable actions

1. Prepare the review tag and GHCR release text from this document.
2. Stage the SparkInfer and vLLM draft PR branches without changing the
   tested commits.
3. Add and review the uncommitted production-reader test diff before the
   SparkInfer draft is opened.
4. Leave placeholders for matched decode, restart, and registry digest; Sol
   will send an evidence addendum as each remaining gate closes.
