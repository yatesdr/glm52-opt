# v20 `5517197` NF3 safe-query candidate

Date: 2026-07-25

> **WITHDRAWN FROM RETRIEVAL QUALIFICATION.** The subsequent PEDANTIC
> discriminator reproduced the identical cold 100k miss, proving that the
> safe-query compute mode does not control the end-to-end symptom. Preserve
> this package only as a separate numeric-correctness experiment. Do not build
> or boot it as the v20 production candidate.

## Objective

Qualify one narrow long-context repair on the newest published v20 NF3
release. Preserve the release image's DCP topology, SparkInfer backend, NF3
weight path, and launch policy. Do not add historical production overlays.

The quality gate comes before memory and throughput. A retrieval or
finalization failure stops the run.

## Immutable base

- Image:
  `voipmonitor/vllm:gilded-gnosis-v20-vllm5517197-sibe0edca-fi801d57a-cu132-20260725`
- Manifest:
  `sha256:e7a8a8549c10b5d16899e0fb45ff7eeca09dd7c1d1a83eee13fb03930d8eb80a`
- vLLM:
  `551719766029e78824a30d97ae6ac63917405b5f`
- SparkInfer:
  `be0edcaae6f5d284bb29a82325aba7a0ead6960f`
- FlashInfer:
  `801d57a08958c13d375ddbb6be3be4808f48a708`

Relative to `83a1f7f7d`, the vLLM base adds four DCP
topology/indexer-composition commits. It does not change
`safe_query_bmm.cu`, the MLA caller, or the B12X MLA backend.

## Narrow repair

Branch:
`fix/v20-5517197-safe-query-precise`

Head:
`29f5f0e927e38f7e7bd89ef177454d80b8327d07`

Commits:

1. `daec87d50` — add a source-compatible precise selector to
   `safe_mla_query_bmm`.
2. `14c2ec924` — use FP32 tensor-core compute while disallowing
   reduced-precision intermediate reduction only for precise calls; restore
   the shared cuBLAS handle's original math mode afterward.
3. `29f5f0e92` — declare that B12X's internally quantized KV path requires
   precise query projection and route it correctly even though
   `supports_quant_query_input` remains false.

Production delta is six source files plus the rebuilt stable-libtorch
extension. The patch does not replace B12X or SparkInfer.

## CPU/off-GPU proof

Command:

```bash
python3 harness/v20_long_context_retrieval_cpu_proof.py \
  --vllm-repo workspace/vllm-v20-5517197-safe-query \
  --sparkinfer-repo workspace/sparkinfer-v20-review
```

Result: `PASS`.

- M=3072 current versus PEDANTIC: BF16 differs 9/9 and post-FP8 differs
  9/9.
- Conservative measured changed-value lower bound: 63–133 BF16 values per
  full-width call.
- The production-shaped CPU witness uses a 192x512 query projection,
  group-16 NVFP4 round-trip, and a 2,048-entry selected sparse window.
  Reduced accumulation favors the distractor and reverses an attention output
  component; precise accumulation matches the full-FP32-reduction oracle and
  favors the needle.
- Full-width exposure grows from 1,248 layer calls at 50k to 12,012 at 475k.
- The proof verifies the exact production SparkInfer NVFP4 math source:
  `0256763b141601bffb080e440756d98504968ad0d9d60a602efa27e967767413`.

Scope limit: this proves the operator mechanism and repaired source route. It
does not replace the model-level cold ladder.

## Build package

Under `workspace/blackwell-llm-docker-v20-dcp-release/`:

- `Dockerfile.v20-5517197-safe-query-precise`
- `build-gilded-gnosis-v20-5517197-safe-query-precise-cu132.sh`
- `patches/v20-5517197-safe-query-precise.patch`

The Dockerfile uses the immutable release digest, checks every changed input
and output byte, rebuilds only `_C_stable_libtorch`, copies the three changed
Python production files into both source and installed trees, and preserves
the remaining image byte-for-byte.

The wrapper refuses push until all of the following pass:

1. Base, vLLM, SparkInfer, and FlashInfer label pins.
2. Stable operator schema contains `bool precise=False`.
3. B12X reports `supports_quant_query_input=False` and
   `requires_precise_query_projection=True`.
4. The selector returns precise only for the quantized-attention route.
5. The complete 54-case precise post-FP8 fingerprint exactly matches the
   checked-in PEDANTIC reference, including M=3072.

## Runtime configuration

Use the release entrypoint:

```yaml
entrypoint: ["/usr/local/bin/serve-gilded-gnosis.sh"]
environment:
  MODEL_FAMILY: glm52-hybrid
  SERVED_MODEL_NAME: GLM-5.2
  TP: "4"
  DCP: "4"
  MTP: "3"
  DCP_PREFILL_WORKSPACE: auto
  MAX_MODEL_LEN: "480000"
  KV_FP8_ROPE: "1"
```

Keep CN4's proven host/fabric settings, cache mounts, port, and restart policy.
Do not copy the old low-level backend, CKV, or query-split variables; the
`5517197` release preset owns those choices.

`KV_FP8_ROPE=1` is intentionally explicit. The NF3 launcher selects
`nvfp4_ds_mla`, but neither it nor the unified entrypoint sets
`KV_FP8_ROPE`; vLLM defaults it to `0`. Without this line, the run exercises
the 432-byte BF16-RoPE record rather than the requested 368-byte FP8-RoPE
record and cannot qualify this repair.

## One-boot fail-closed qualification

### Gate 0 — no model

1. Build the derived image with `BUILD_JOBS=16`.
2. Run the wrapper's schema, route, and 54-case fingerprint gates.
3. Run accurate-regular and accurate-precise through
   `v20_safe_query_reduction_equivalence_probe.py`, using the archived
   current and PEDANTIC rows in
   `v20_compare_safe_query_reduction_equivalence.py`.
4. Require:
   - accurate-regular equals current;
   - accurate-precise equals PEDANTIC at the post-FP8 boundary;
   - graph replay matches eager at M=9 and M=3072;
   - M=3072 precise pipeline time is at most 1.25x current and at most 0.90x
     PEDANTIC.

Do not boot if any Gate 0 assertion fails.

### Gate 1 — boot

Use the exact pushed digest, `restart: "no"`, and no source mounts.

Require:

- stable process identity and health;
- all CUDA-graph capture sizes complete;
- zero illegal access, OOM, Xid, assertion, engine-dead, or worker-death
  signatures;
- `B12X GLM MLA KV format: KV_FP8_ROPE=1 kv_gmem_stride=368
  kv_cache_dtype=nvfp4_ds_mla`;
- KV pool at least 500,000 tokens at 480k max length.

### Gate 2 — quality first

Run natural-head cold requests and record `prompt_sha256`, `cached_tokens`,
`finish_reason`, completion tokens, content, reasoning, serialized message,
and MTP acceptance deltas.

1. Discriminator: 100k x3 and 150k x3.
2. If and only if all six are exact, continue 50k, 250k, 350k, and 475k x3.

Every accepted row must have:

- `cached_tokens=0`;
- exact needle retrieval;
- `finish_reason=stop`;
- non-empty finalized `content`;
- no digit or adjacent-word duplication.

Any `MISS`, `DUP`, `REASON_OK`, empty final content, truncation, missing row,
or error fails Gate 2 and stops qualification.

### Gate 3 — performance and capacity

Only after Gate 2 passes:

- record the actual MTP3 KV pool; do not reuse the published MTP0 value;
- run cold 8k and 55k prefill;
- run decode C1/C4/C8/C16 and the existing long-context decode cell;
- compare MTP3 only with MTP3;
- require no more than 10% decode regression and use the established cold
  prefill floors.

NVMe qualification follows on the same process after the quality and
performance gates.
