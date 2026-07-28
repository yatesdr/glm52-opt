# v20 post-6d32 needle-regression: no-model discriminator

Date: 2026-07-25  
Owner: Sol (code/proof design)  
Operator: Fable (CN4 runtime)  
Scope: GPU microprobes only; do not boot or load GLM-5.2

## Objective

Resolve the two remaining source-level deltas before choosing another model
image:

1. PR #171 changes `nvfp4_ds_mla` MTP verification from split-K decode to
   single-pass extend.
2. vLLM `992b874cf` changes `safe_mla_query_bmm` from
   `CUBLAS_COMPUTE_32F_PEDANTIC` to tensor-core-eligible
   `CUBLAS_COMPUTE_32F`.

Also test the exact execution boundary exposed by the field matrix:

3. the production fused indexer switches from serial last-CTA merge to
   cooperative grid-barrier merge at 16,385 DCP-local tokens, or 65,540
   nominal global tokens (the exact DCP4 threshold is 65,536).

The current staged BF16-to-FP8 guard does not undo item 2: it still calls the
compiled `safe_mla_query_bmm` before static FP8 quantization.

## Pins

```text
harness/v20_post_6d32_static_bisect.py
sha256 ec7e3271f45a19ce2c9b71ee20521d2d732ae5e473143a82a5db952bc1fd2cfc

harness/v20_safe_query_bmm_cross_image_probe.py
sha256 84917c898049273a7ed51f6e580a5007328d17f9d67a3ce8f38f343fa3bcb1f4

harness/v20_compare_safe_query_bmm_fingerprints.py
sha256 1e5dcad50abc7f44c120216c3394059ffd7d17ab562f1ea1c1a4be3d1c3b1cdb

harness/v20_nvfp4_decode_extend_high_index_probe.py
sha256 9a4a66a22bc063a84b1593ff6c7b4fc634bd513007988bddb32d42efc70097d0

harness/v20_fused_indexer_16384_crossover_probe.py
sha256 edf2c93be7df939a5d9fb48b0850d2184a2e4de6dfe1c60b0bd39229e0b17a2e
```

Recheck the pins before execution. The scripts are fail-closed and write no
model or cache state.

## Gate 0 — fused-indexer runtime crossover

Run the crossover probe on the NEW image before Gate B. It uses production
GLM geometry (`rows=4`, `heads=32`, `topk=2048`) and compares forced serial,
forced cooperative, and auto dispatch at local lengths
`16383,16384,16385`. It repeats logical/physical output modes and CUDA-graph
replay while retaining a low GPU footprint.

If NEW fails, run the identical pinned probe on OLD. Preserve both JSONL
files. An OLD-pass/NEW-fail result proves an interaction regression even
though `fused_indexer.py` itself is unchanged between the images.

Important source qualification: `_SMEM_CANDS` changed in `tiled_topk.py`, but
the fused module only imports that name and never uses it. The fused source
and the three imported helpers it does use are unchanged. The 65,536 boundary
therefore cannot alone explain the post-6d32 onset; it can expose an
interaction with changed query inputs.

## Gate A — compiled safe-BMM cross-image fingerprint

Run the same probe on one GPU in:

```text
OLD = ghcr.io/yatesdr/glm52-serve@sha256:6d32a0c3a64962078c74c86485da1090c9f01b287c70009e4761359ed063c338
NEW = ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-prod-ready-20260724
```

Use a read-only bind for `harness/`, a writable artifact directory, the image's
`/opt/venv/bin/python`, and no model arguments:

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  --entrypoint /opt/venv/bin/python \
  -v "$PWD/harness:/proof:ro" -v "$OUT:/out" \
  "$OLD" /proof/v20_safe_query_bmm_cross_image_probe.py \
  --output /out/safe-bmm-old.jsonl

docker run --rm --gpus '"device=0"' --ipc=host \
  --entrypoint /opt/venv/bin/python \
  -v "$PWD/harness:/proof:ro" -v "$OUT:/out" \
  "$NEW" /proof/v20_safe_query_bmm_cross_image_probe.py \
  --output /out/safe-bmm-new.jsonl

python3 harness/v20_compare_safe_query_bmm_fingerprints.py \
  "$OUT/safe-bmm-old.jsonl" "$OUT/safe-bmm-new.jsonl"
```

Required validity signal:

```text
status = PASS
reference_stable = true
```

Interpretation:

- `operator_changed=true`: the post-6d32 compiled BMM changed numerical
  output; the candidate remains live.
- `post_quant_changed=true`: the difference survives the actual static FP8
  boundary.
- `retrieval_ids_changed=true`: the synthetic downstream selected-id set also
  changes. This is strong causal support, but `false` does not exonerate the
  operator because this is not the model's learned key distribution.
- `operator_changed=false`: this candidate is exonerated for the tested
  production geometries.

## Gate B — #171 deep physical-index decode-versus-extend proof

Run on the NEW image only:

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  --entrypoint /opt/venv/bin/python \
  -v "$PWD/harness:/proof:ro" -v "$OUT:/out" \
  "$NEW" /proof/v20_nvfp4_decode_extend_high_index_probe.py \
  --output /out/nvfp4-decode-extend-high-index.jsonl
```

The 12 cases cover DCP4-local physical depths corresponding to 100k, 150k,
250k, and 475k global contexts. Each writes valid 368-byte NVFP4/FP8-RoPE
records at deep physical slots and compares:

```text
split-K decode -> explicit dequantized-record oracle
single-pass extend -> same oracle
extend -> decode
```

Interpretation:

- Any extend failure with decode passing directly rejects #171.
- Both paths passing removes high-index address/record math as #171's defect,
  but does not by itself qualify the route under learned model distributions.
- Any decode failure makes the probe/harness invalid for a routing decision;
  preserve the JSONL and stop.

## Prepared code paths

- No-#171 integration candidate:
  `workspace/vllm-v20-staged-query-no171`, head `b8534c4a5fad`.
- Experimental precision-preserving implementation:
  `workspace/vllm-v20-safe-query-fp8-precision`,
  branch `fix/v20-safe-query-fp8-precision-20260725`, commit `1cfec8e72`.

The precision branch keeps regular FP32/tensor-core execution as the default
and requests PEDANTIC accumulation only when the BF16 absorbed query is about
to be requantized to FP8. It is committed locally for reproducibility but is
not a PR until Gate A establishes that this numeric delta is real.

## Stop condition

Do not build or boot a model from either candidate from this handoff. Return
the three JSONL files and comparator JSON to Sol. Those results choose the
single consolidated model candidate.
