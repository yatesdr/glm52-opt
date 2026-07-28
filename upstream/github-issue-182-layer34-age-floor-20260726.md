### Layer-34 result: precision is not sufficient; a minimal explicit history floor is proof-positive

I completed the real-activation proof at layer 34, the first layer where the
known-good and failing selection traces diverged around the deep needle. This
changes the permanent-fix direction again: selector quantization contributes
error, but it is not the primary reason the needle is absent.

#### Frozen reproduction

- image lineage: current v20 `0c79e41/e603f74`, with a diagnostic-only
  activation trace;
- TP4/DCP4/MTP0, MNBT 3072, NVFP4 MLA KV, FP8 RoPE;
- frozen `fail-350k-r1` prompt SHA-256:
  `f0d1c16d816b777f27a3882d9e6b5ef056852684ea155fb11dd845f9e1654ab5`;
- rendered prompt: 343,727 tokens, `cached_tokens=0`;
- evaluated query: absolute position 343,726;
- eligible committed history: 340,992 rows;
- actual needle offset: 137,496; proof window: `137,490 +/- 24`;
- score order: FP32 Q.K, per-head ReLU, learned head weights, then top-2,048.

The production K-cache writer was checked again: its E4M3 values and scale
bytes matched `per_token_group_quant_fp8` byte-for-byte on 4,096 real K rows.

#### Quantization/precision result

| Path | Recall@2048 vs BF16 | False negatives | Needle best rank |
|---|---:|---:|---:|
| BF16 Q + BF16 K oracle | 100.000% | 0 | 4,410 |
| current E4M3/UE8M0 Q + K | 96.045% | 81 | 4,774 |
| BF16 Q + current FP8 K | 95.947% | 83 | 4,560 |
| current FP8 Q + BF16 K | 99.365% | 13 | 4,603 |
| E4M3 with exact FP32 per-vector scales | 95.801% | 86 | 4,539 |
| per-vector INT8 + FP32 scale | 98.291% | 35 | 4,374 |

The K side dominates the top-k disagreement, and INT8 materially improves
proxy recall, but every path still excludes the needle from top-2,048. Most
importantly, the full BF16 oracle also excludes it. Exact FP32 scales therefore
do not provide a fix, and widening then reranking with the same GLM indexer
score cannot help unless the candidate set exceeds roughly 4.4k at this layer.

This is consistent with the layer-0 result (BF16 needle rank approximately
12k): v20 is exact for its proxy, but unconstrained global top-2,048 is not the
selection distribution this checkpoint needs for deep retrieval.

#### Bounds-safe policy proof

I then evaluated deterministic chronological coverage policies on the same
captured tensors. The smallest proof-positive rule was:

1. split eligible history into two chronological halves;
2. require at least 64 exact-score winners from each half;
3. fill every remaining slot with the exact global winners.

At layer 34:

- current BF16 exact position quartiles: `[7, 10, 456, 1575]`;
- only 17/2,048 entries come from the older half;
- the two-half/64-floor policy yields `[11, 53, 450, 1534]`;
- it includes real needle-local token 137,485;
- it changes only 47/2,048 BF16 choices.

Using the production FP8 proxy, the same policy yields
`[12, 52, 437, 1547]` and also includes token 137,485.

At layer 0 the policy is a literal no-op: the current exact selector already
has far more than 64 winners in each half (`[1222, 205, 237, 384]` by
quartile), so all 2,048 selections remain unchanged. This adaptive behavior is
important: it adds historical coverage only when the learned scores would
otherwise collapse almost the entire budget into recent context.

This is not a recreation of the v19 overflow:

- no coarse-bucket truncation;
- no scan-order-dependent 4,096-candidate cap;
- every selected entry is an exact-score winner within either its required
  chronological region or the global fill;
- deterministic, bounds-safe, and straightforward to include in the compile
  key.

#### Next gate

The next candidate is therefore an explicit server-static
`two_half_min_64`-style policy, with `exact` remaining the default. It should
be implemented independently of `bounded_compat`, then tested in this order:

1. frozen 250k control;
2. all three frozen cold 350k failures;
3. randomized cold 50k--475k ladder;
4. KLD/finalization, throughput, KV-pool, graph-capture, and fatal-signature
   gates.

If the causal boot fails, the policy proof is only local evidence and the
candidate is discarded. If it passes, this is a reviewable permanent v20
design: explicit minimum historical coverage with exact selection, rather
than historical overflow compatibility.

Evidence SHA-256:

- layer-34 precision report:
  `8c363b985aa794273810bc81aba6b781f8a22a3d12011cfb42fd481cf67ef5ab`;
- layer-34 age-policy report:
  `d568f7576fd263f3d799e4956584a7be8de843cfdf66f7343a5c9099cad068bb`;
- layer-0 age-policy control:
  `acac64e53fe2fb9c648e2b5e1eba4fb53f73a4c64f0c55db4623c3df2e4c9df`;
- age-policy proof program:
  `034f79e913c340463b7f3c7321d9ee73f669471d296e2b28650726a26736106f`;
- hardened capture helper:
  `6c6ad64b298cacb621d3b38404f39c74a967bcd9a1535d22ba0a1f5ba0e642a3`.
