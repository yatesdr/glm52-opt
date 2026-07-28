# v20 shared-precision trajectory causal test — 2026-07-27

## Question

The official GLM full-precision indexer scorer, running end to end from layer
zero on the frozen 350k requests, reproduces the deep-retrieval failure. An
exact needle-inclusion trace further shows that the ticket-number tokens do
not reach sparse attention until late layers:

- layers 0–38: none of the three value tokens selected on any answer decode;
- layer 42: one value token selected in 1/16 answer-decode calls;
- layer 62: all three selected together in only 1/16 calls;
- layer 74: at least one selected in 15/16 calls, all three in 7/16.

The production exact top-k kernel, page-table stride fix, and the official
scorer arithmetic therefore are not sufficient causes. The next causal
question is whether the shared main-attention precision posture degrades the
hidden-state trajectory enough to make the trained indexer omit the relevant
history.

## Important constraints

`i8_ring` is rank-consistent block-INT8 transport, but it is **not**
numerically lossless. Each 128-value block uses one FP32 scale and signed INT8
rounding; before the final BF16 store, its absolute codec error is bounded by
`amax / 254`. It is distinct from the lower-precision E4M3 wire mode, but it
can still suppress small values in outlier-heavy blocks. The working
`oldest_boundary` positive control also passed under the same NF3 checkpoint,
NVFP4 MLA KV, FP8 RoPE, and `i8_ring` settings. Those settings are therefore
not sufficient causes in isolation, although their combined trajectory may
reduce the exact selector's score margins.

A literal BF16 main MLA KV cache is not a feasible 350k test on CN4:

- `nvfp4_ds_mla` compact record: 368 B/token in the measured production
  posture;
- `fp8_ds_mla` record: 656 B/token, with BF16 RoPE;
- BF16 MLA record: approximately 1,152 B/token.

At the measured 837,953-token NVFP4 pool, proportional record accounting
predicts only about 268k BF16 tokens before other workspace effects. That is
below the frozen 343,727-token request. The test below is therefore the
highest-precision main-KV posture expected to fit, not an all-BF16 proof.

## Single boot

Pinned base:

```text
voipmonitor/vllm@sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0
```

Compose:

```text
compose/glm52-v20-official-reference-fp8kv-bf16rope-lossless-r1-20260727.yaml
sha256 73aa5c6af6bd2a6d450f5911d19c7cb62478f453b4e7861198ee482404e2f4bd
```

Derived official-reference image:

```text
glm52-serve:v20-official-reference-runtime-stride-chunk16-75715e51
image ID sha256:899e64cc6098407d1e41bca8db53f70ea60f31009b812872e4690540798ded1a
source commit 75715e51
```

The derived image is based on the pinned stock digest above. It holds the
official BF16/FP32 scorer constant with the completed reference gate and
interposes `--enforce-eager --no-enable-prefix-caching`.

Changed trajectory components:

```text
main MLA CKV: nvfp4_ds_mla -> fp8_ds_mla
RoPE cache:   FP8 compact writer -> BF16 field in fp8_ds_mla record
DCP wire:     i8_ring -> raw BF16 (implementation-control change)
```

Held constant:

```text
checkpoint: NF3 hybrid
selector: official BF16/FP32 scorer + production exact top-k
TP/DCP: 4/4
MTP: 0
graphs: eager
MNBT: 3072
query_split / owner_merge: 0 / 0
CKV prefetch depth: 0
max model length: 360,000
GMU: 0.950
```

The raw-wire change removes both the block-INT8 numerical approximation and
the custom collective implementation from the causal cell.

## Fail-closed boot acceptance

Before the request:

1. exact image digest is present;
2. zero container restarts and no fatal signatures;
3. boot log reports `kv_cache_dtype=fp8_ds_mla`;
4. boot log reports the 656 B/token FP8-DS-MLA format;
5. `KV_FP8_ROPE=1` is absent from the effective compact NVFP4 writer;
6. all three PCIe DMA mode variables resolve to `0`;
7. available KV pool exceeds 360,000 tokens.

## Decisive request

Run only immutable `fail-350k-r1`:

```text
prompt sha256: f0d1c16d816b777f27a3882d9e6b5ef056852684ea155fb11dd845f9e1654ab5
rendered tokens: 343,727
cached_tokens required: 0
expected content: 738216
expected finish: stop
```

Interpretation:

- **EXACT recovery:** the changed main-attention precision trajectory is
  causal for this frozen row. Run two minimal follow-ups:
  1. `nvfp4_ds_mla`, `KV_FP8_ROPE=0`, raw BF16 wire;
  2. `fp8_ds_mla`, BF16 RoPE, `i8_ring`.
  These separate CKV/RoPE from wire integrity.
- **Still ABSENT:** the combined NVFP4-CKV/FP8-RoPE posture is not sufficient
  for the frozen failure. Do not claim that fully BF16 KV was tested. Close
  deeper-layer input provenance and the model's training-time selector
  contract before designing another selector policy.

Neither outcome makes `oldest_boundary` the canonical fix. It remains a
positive compatibility control.

## Result

The frozen request recovered exactly:

| Field | Result |
|---|---|
| verdict | `EXACT` / gate `PASS` |
| content | `738216` |
| prompt tokens | 343,727 |
| completion tokens | 4 |
| finish reason | `stop` |
| cached tokens | 0 |
| elapsed | 708.8 s |
| container health | healthy, zero restarts |
| measured KV pool | 491,769 tokens |

This is causal evidence for the **combined changed trajectory posture** on
the frozen r1 row. The official scorer, MTP0/eager execution posture, prompt,
and decoding parameters were held constant with the failed official-reference
run. The changed inputs were the main MLA cache format/RoPE representation
and the DCP wire implementation control.

The result does not yet identify which changed input is necessary. Because
`i8_ring` is a genuine numerical precision lever, the highest-priority
discriminator is:

1. keep `fp8_ds_mla` with BF16 RoPE and restore `i8_ring`;
2. if that remains exact, test `nvfp4_ds_mla` with `KV_FP8_ROPE=0`;
3. only after locating the minimal precision lever, design the production
   correction and restore the required 500k-at-480k capacity floor.

The current 491,769-token pool is acceptable for this 343,727-token causal
cell but is **not** a production-capacity pass.

Pinned evidence:

```text
harness/cn4-evidence-archive/20260727/official-reference-fp8kv-bf16rope-lossless-r1-v1/

run.log
  sha256 688cd35059df720aebef73d22e8f1f65a10224ef140d4ed0d1812bd9a384ba53
results/summary.json
  sha256 db505691d4a6025eea8789e10d19e372133386fc990990d5845da4e866052cb9
results/rows.json
  sha256 828ee3ae1684757a9dac6cd585883c12da4d53cc25c606c561919aced9b0bc14
results/resp-fail-350k-r1.json
  sha256 0481d7f9cb209afa7ec5d70df820f777edd539ca9469306ed58ba56a4716ef3d
```

## Rejected startup attempt

An initial compose tried to run the stock image with `GRAPH=0` directly:

```text
compose/glm52-v20-runtime-stride-fp8kv-bf16rope-lossless-r1-20260727.yaml
sha256 cd70453d4f1034ea6fe6e9164bbfd98fba68c1084a96f2d62d91e088b8600d64
```

It exited before weight loading with:

```text
ValidationError: Maximum cudagraph size should be greater than or equal to 1
when using cuda graph.
```

No model request ran and no behavioral result was consumed. The corrected
compose uses the already-qualified eager reference image so the scorer and
execution posture remain constant with the prior failed reference gate.
