# v20 GLM sparse-indexer precision: permanent-fix plan

Date: 2026-07-26

## Decision

`SPARKINFER_NSA_TOPK_SELECTION_POLICY=bounded_compat` is a compatibility
fallback and a causal discriminator, not the intended v20 fix.  It proved that
changing indexer selection can recover the frozen long-context needle, but it
also restores a bounded v19-era policy whose relationship to the checkpoint's
full-precision selector is not defined.

The permanent contract is:

1. preserve v20's exact top-k policy;
2. make the FP8 score field a faithful approximation to the checkpoint's
   full-precision Q/K score field;
3. remove `bounded_compat` from the production configuration after the native
   path passes the cold ladder.

## Source finding

The GLM reference indexer computes post-RoPE Q/K dot products in full
precision, applies per-head ReLU and learned head weights, and then takes an
exact top-k.  The current vLLM accelerated path instead:

- applies RoPE and immediately quantizes each 128-wide Q row to E4M3 with one
  UE8M0 scale (`fused_indexer_q_rope_quant`);
- quantizes each raw 128-wide K row the same way while inserting it into the
  indexer cache (`indexer_k_quant_and_cache_kernel`);
- runs exact top-k over that quantized score field.

Neither Q nor K receives outlier-conditioning before the one-scale-per-row
quantizer.

DeepSeek's optimized DSA reference applies the same normalized Sylvester
Hadamard transform to Q and K before FP8 quantization.  Since the transform is
orthonormal, it preserves every full-precision dot product:

```text
(Q H) (K H)^T = Q H H^T K^T = Q K^T
```

It changes only how the vectors occupy FP8's dynamic range.  This makes it a
candidate v20-native precision fix rather than a selector-policy rewind.

The omission is visible in both accelerated implementations:

- GLM/DeepSeek-V3.2 constructs the complete post-RoPE 128-D Q/K rows in
  `deepseek_v2.py`, then sends Q directly to
  `per_token_group_quant_fp8`/`fused_indexer_q_rope_quant` and K directly to
  `indexer_k_quant_and_cache`.
- DeepSeek-V4 constructs its indexer compressor with `rotate=True`, stores that
  value as `DeepseekCompressor.rotate`, but never reads it.  Its fused Q
  quantizer likewise has no conditioning argument.

This makes the proposed change the completion of an already-declared
compressor contract, not a request to restore the historical selector.

## Evidence boundary

This is a source-grounded hypothesis, not yet a model fix.

A deterministic synthetic production-shape CPU fixture showed:

- full-precision transformed scores agree with the untransformed oracle;
- Hadamard reduced the worst observed FP8 score error;
- top-k recall did **not** improve consistently on arbitrary synthetic
  activations.

That result rejects synthetic sufficiency.  The hypothesis proceeds only
through a real checkpoint-activation proof.

## Real-activation discriminator

Diagnostic branch:

```text
diag/v20-indexer-hadamard-activation-proof-20260726
ec12ccdd1 test(indexer): capture real prequant activations
```

Pinned trace image:

```text
glm52-serve:v20-20260726-indexer-prequant-trace
sha256:08465d95d895ade73d4dd2155a5bd6c89afba3c8c4def605348132e8942ad165
```

The trace runs one cold frozen 350k request on the current v20 base with MTP0.
An opaque custom op, armed only after boot, records for indexer layer 0:

- every real post-RoPE BF16 K chunk;
- the final post-RoPE BF16 Q row;
- the learned head weights;
- v20's runtime exact top-k output.

The trace is about 90 MiB and does not retain a second GPU cache.  The offline
CPU comparator computes on identical bytes:

1. full-precision checkpoint oracle;
2. current raw-vector E4M3/UE8M0 scores and exact top-k;
3. Hadamard-conditioned E4M3/UE8M0 scores and exact top-k;
4. separate 64-D RoPE/NoPE E4M3 scales and exact top-k.
5. four 32-D E4M3 groups with four UE8M0 scale bytes, which fit in the
   existing 132-byte cache record but require a new reader contract.

It also checks that its raw-FP8 selection reproduces the runtime set.  This
closes the gap between a convenient synthetic distribution and actual GLM
activations.  The split-scale cell costs four additional scale bytes per
cached token (136 B versus the current 132 B index record) but requires no
orthogonal transform, so the same proof ranks the two strongest native fixes
without another model boot.

## Go/no-go criteria

Proceed to a model patch only if the real activation proves all of:

1. CPU raw-FP8 reproduces the runtime top-k (or any difference is explained by
   DCP index mapping);
2. transformed full-precision scores remain numerically identical within the
   expected FP32 accumulation tolerance;
3. at least one native precision mode materially improves top-k agreement with
   the full-precision oracle;
4. the best score/rank in the known 40% needle window improves rather than
   regresses;
5. the selected fix has a measurable advantage sufficient to justify its
   memory/compute cost (Hadamard launch/compute versus four extra scale bytes).

If these fail, both precision-conditioning candidates are rejected and the
real trace becomes the input to the next investigation: score accumulation or
DCP global candidate merge.

## Implementation sequence if it passes

1. **Causal prototype.** Implement only the winning real-activation mode behind
   a new `attention_config.indexer_fp8_conditioning` field.  The field is
   server-static and participates in `AttentionConfig.compute_hash`, so
   transformed and untransformed cache semantics cannot mix.  For Hadamard,
   disable the fused Q quantizer, transform post-RoPE Q/K, and always consume
   the tensor returned by Hadacore: its internal padding path may return a new
   allocation when the K-row count is odd.  For split scales, change the Q/K
   quantization contract and index-cache stride from one 128-D scale to two
   64-D scales.
2. **Frozen model gate.** Run 250k control plus 350k x3 on the exact prompt and
   sampling hashes.  No bounded selector is enabled.
3. **Cold ladder.** Require finalized content at 50k, 150k, 250k, 300k, 350k,
   475k, with both `content` and reasoning fields audited.
4. **Production kernel.** Fuse the winning mode into
   `fused_indexer_q_rope_quant` and `indexer_k_quant_and_cache_kernel`; avoid
   intermediate buffers and launches.  Consume the existing
   `DeepseekCompressor.rotate` flag in the DeepSeek-V4 compressor path as part
   of the same cache contract.  Preserve the unfused implementation as a test
   oracle.
5. **PR.** Submit the precision-conditioning change with operator fingerprints,
   real-activation comparison, cold ladder, and performance/capacity data.
   `bounded_compat` remains a separately documented fallback and is absent
   from the promoted compose.

## Prepared causal prototype

The disabled-by-default GLM prototype is isolated from the trace branch:

```text
worktree: workspace/vllm-v20-indexer-wht-prototype
branch:   fix/v20-indexer-fp8-conditioning-prototype-20260726
commit:   ea5cb33c5 feat(indexer): prototype post-RoPE FP8 conditioning
```

It adds the server-static `attention_config.indexer_fp8_conditioning` field,
applies Hadacore only after interleaved RoPE and full 128-D concatenation, and
forces the unfused Q path while enabled so it can serve as the causal oracle
for later fused kernels.  It remains disabled until the captured real
activation comparison selects it.

## Work after quality

The first consolidated memory candidate reached 3.35 GiB available KV memory,
enough for an estimated 450,560 tokens but short of 480k's 3.57 GiB
requirement.  Once quality is restored, reclaim the remaining approximately
0.22 GiB, then measure prefill/decode.  Neither memory nor throughput work is
allowed to obscure the selector acceptance gate.
