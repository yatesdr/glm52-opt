# v20 NF3 staged-projection diagnostic gate

Date: 2026-07-25

## Question

Does the v20-only generated-token MLA projection route cause the remaining
deep-needle/finalization regression relative to the known-good v19 NF3
deployment?

This is a **diagnostic**, not the proposed production fix.  v20 intentionally
introduced compact native MXFP8 weights and fused small-M projection.  A
passing staged discriminator says that seam is causal; it does not say all of
the v20 work should be reverted.

This gate changes one semantic seam as a bundle:

- current v20: native MXFP8 absorbed query/value BMM, with fused query
  assembly when qualified;
- discriminator: materialized BF16 absorbed weights, no fused query, staged
  BMMs for both query and value projection.

Model weights, NF3 routing, NVFP4 KV, FP8 RoPE, DCP, MTP, wire mode,
SparkInfer, FlashInfer, CUDA extensions, launcher, prompt bytes, and sampling
seeds remain fixed.

The bundle contains three separable changes:

1. native MXFP8 query BMM instead of a materialized-BF16 `torch.bmm`;
2. native MXFP8 value BMM instead of a materialized-BF16 `torch.bmm`;
3. fused query assembly and static E4M3 conversion.

SparkInfer's own tests require item 3 to be byte-identical to native BMM plus
staged assembly.  They do not require either native BMM to be byte-identical
to the v19 materialized-BF16 reduction.  The no-model decomposition probe must
therefore run before interpreting a model A/B:

```text
harness/v20_nf3_absorbed_projection_probe.py
sha256 ecd99b8890c8d8d044d90a9d8301f6ecacac2be0a419cd69db190e13f845e390
```

Its expected invariant is zero changed bytes for native fused assembly versus
native BMM plus staged assembly.  Nonzero v19/native differences after the
E4M3 boundary isolate the BMM reduction as a model-visible change.

## Why both fast paths must be disabled

`VLLM_B12X_ABSORB_BMM=0` is not a complete discriminator on v20. It
materializes the weights but `_try_fused_mla_query()` then selects the later
BF16 fused-query kernel for generated-token batches. The candidate mode
therefore disables native absorption and fused query assembly together.

The CPU proof:

```text
harness/v20_nf3_projection_route_cpu_proof.py
sha256 6101c1b50e2677820d2aa81243e6b06268867cf4c59e8f8fdc88be12763aac49
```

pins the exact v19/v20 sources, verifies the test-coverage gap, verifies both
candidate gates, and produces a deterministic production-K=192 one-BF16-ULP
reduction-order witness. It is a mechanism proof, not the causal verdict.

## Candidate source

```text
base vLLM: 551719766029e78824a30d97ae6ac63917405b5f
branch:    fix/v20-nf3-legacy-projection
commit:    d367318c9a74ddc4d79de0ab6db81e9aab9b81dc
```

Files:

```text
vllm/envs.py
  input  652f67a93ade31c8a078797d894ae7b93b915b9948edb9b3e4f6926f9755beb3
  output 844fd2fe01a3311ca58aa004945fcc4f4df01921e4da44f8197e7e5636eef340

vllm/model_executor/layers/attention/mla_attention.py
  input  96a37e550aa64e8e2ce7f7761f55bbdafaa58b018ae0c49e53a6c37f5aa1f3e4
  output 2ab1bc3e2b52f1d664c5f7624f18f5a42e2f933443796a8cd7dce7522a6b1e96
```

The patch adds:

```text
VLLM_B12X_MLA_PROJECTION_MODE=auto|legacy
```

`auto` preserves current behavior. `legacy` is the discriminator.

## Build while the current process remains live

Use the exact 5517197 base plus the already-qualified launcher layer:

```text
workspace/blackwell-v20-p2p-launcher/
  Dockerfile.v20-5517197-nf3-legacy-projection
    sha256 0d81d703581b48446421de841c7df1b961975e7113d8738253873e37901cdc3b
  build-v20-5517197-nf3-legacy-projection.sh
    sha256 e27ebddef386f8ba6ee808e660db54a1254cc9370219b881daf14fea99aaf926
  patches/0001-fix-mla-add-staged-projection-compatibility-mode.patch
    sha256 768dac3e0a191073ea08afae7451a72744a85c725a2f2ac52cbe0b2de851f30e
```

The build is a Python-only patch layer. It must not rebuild or replace any
CUDA, C++, SparkInfer, FlashInfer, or model byte. The wrapper fails closed on
base input hashes, patch hash, output hashes, launcher hash, `py_compile`,
mode, and labels. It does not push unless `PUSH=1`.

## One model-down window

1. Preserve the current process identity and evidence, then stop it.
2. With GPUs free, run the no-model projection comparator:

   ```text
   harness/v20_nf3_absorbed_projection_probe.py
   sha256 ecd99b8890c8d8d044d90a9d8301f6ecacac2be0a419cd69db190e13f845e390
   ```

   Require the summary to be complete, native fusion versus native staged
   output to remain byte-identical, and record separately whether the native
   query BMM survives the FP8 boundary, whether BF16 fusion preserves the
   legacy FP8 bytes, and whether the native value BMM changes BF16 bytes.
3. Boot the derived image with the exact current NF3 compose. Change only the
   image identity. The image already sets
   `VLLM_B12X_MLA_PROJECTION_MODE=legacy`; verify the effective environment
   and the log line:

   ```text
   Using materialized BF16 MLA projection weights and staged BMMs
   ```

   The log must not contain:

   ```text
   Serving MLA absorbed projections directly from the B12X MXFP8 pack
   ```
4. Hard boot gates:

   - NVFP4 MLA KV and `KV_FP8_ROPE=1` remain active;
   - TP4/DCP4/MTP3 and the exact `/model` checkpoint remain active;
   - pool is at least 500,000 tokens;
   - API healthy, RC 0, no illegal access, OOM, Xid, assertion, or fallback.
5. Run the same fixed prompt bytes and sampling seeds used for the current
   5517197 evidence. Record `prompt_sha256`, all response fields,
   `finish_reason`, usage, cached tokens, and MTP acceptance.

   Fail-closed order:

   ```text
   100k x3 -> 250k x3 -> 350k x3 -> 450k x3 -> 475k x3
   ```

   Every row must be cold (`cached_tokens=0`), `finish_reason=stop`, and exact
   non-empty finalized `content == "738216"`. An answer present only in
   reasoning is still a promotion failure.

## Verdict

- **CONFIRMED:** all fixed rows exact while the current image's matched rows
  are not. Do **not** ship `legacy` as the answer. Use the no-model
  decomposition and a query-only/value-only discriminator, if still needed,
  to choose the smallest forward fix:

  - correct the native kernel's numerical contract;
  - narrow native eligibility only for the failing projection/format;
  - or retain fused assembly with a materialized BF16 query or value weight.

  Present the causal result, memory cost, and measured speed cost to Derek
  before selecting one.
- **REFUTED:** any matched row still fails. Preserve the result, do not tune,
  and remove this seam from the long-context causal path.
