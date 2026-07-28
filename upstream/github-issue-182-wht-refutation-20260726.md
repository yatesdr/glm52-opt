### Permanent-fix update: real-activation proof refutes WHT as the remedy

The bounded compatibility selector remains a useful causal discriminator and
restores the known-good quality distribution, but it is **not** the proposed
permanent v20 default. I ran the next proof against activations captured from
the frozen, cold `fail-350k-r1` request to test the more principled hypothesis:
apply the same normalized WHT-128 to Q and K after interleaved RoPE, before
fresh FP8 scale calculation and quantization.

#### Reproduction and proof boundary

- rendered prompt: 343,727 tokens;
- actual needle token offset: 137,496;
- evaluated query: final active query row at absolute position 343,726;
- eligible history: 340,992 rows (the in-flight tail was excluded);
- selector: full GLM score order — FP32 dot product, per-head ReLU, learned
  head weights, then exact top-2,048;
- K quantization: the production `indexer_k_quant_and_cache` output was
  byte-compared with `per_token_group_quant_fp8` on 4,096 captured K rows.
  Both FP8 values and scales were byte-identical.

This is an operator-level proof on one real failing layer/query. It does not
replace the end-to-end cold ladder.

#### Result

| Path | Recall@2048 vs BF16 oracle | False negatives | Score RMSE |
|---|---:|---:|---:|
| Direct FP8 (current) | 92.773% | 148 | 5.871 |
| Normalized Hadacore WHT + FP8 | 84.961% | 308 | 7.304 |
| FP32 Sylvester WHT + FP8 | 84.766% | 312 | 8.347 |

The orthogonal transform approximately preserves the unquantized ordering
(Hadacore BF16 recall 98.730%), but it makes the production FP8 selection
materially worse on this captured failing row. Therefore WHT must not be
shipped as the retrieval fix on the strength of the reference-path analogy.

There is a second, stronger result: the needle neighborhood is absent from the
top-2,048 even under the full BF16/FP32 oracle. Its best rank is approximately
12,018 before FP8 (12,267 with direct FP8; 11,396 with Hadacore WHT FP8).
Consequently, merely increasing precision—or selecting a wider set and
reranking it with the same indexer proxy—cannot recover this layer/query
unless the candidate set exceeds roughly 12k and the final score changes.

#### Instrumentation correction

The first runtime-selection comparison is discarded. The trace recorded
`topk_indices[-1]` from a fixed 3,072-row workspace even though the final
active chunk had 2,735 rows, so it captured a stale workspace row. The
activation tensors themselves are valid. The trace fix now records
`topk_indices[active_rows - 1]` and stores `active_rows`, `buffer_rows`, and
`selected_row` in schema v2 (`d0beb42dd`).

#### Current permanent-fix direction

The evidence now separates two facts:

1. v20's exact selector is a better implementation of the quantized proxy;
2. that proxy's unconstrained top-2,048 allocation is not sufficient for the
   checkpoint's deep-retrieval behavior.

The next proof is at the first observed end-to-end divergence (layer 34), where
the known-good path selected five needle-local positions and exact v20 selected
none. I am comparing two explicit, bounds-safe designs:

- deterministic age/diversity coverage that reserves a calibrated fraction of
  the 2,048 budget across older context regions while keeping the remainder
  exact;
- broader candidate selection followed by a more model-aligned rerank (not the
  same indexer proxy).

`bounded_compat` remains diagnostic/compatibility-only while that proof is in
progress. The permanent patch will be the smallest deterministic policy that
passes the frozen 250k control, all three frozen 350k failures, randomized cold
50k–475k retrieval, KLD, throughput, and KV-capacity gates.

Evidence SHA-256:

- GPU proof report:
  `3581c5f6c6f9bed6ac484647ed19f1fb7ac7fa238648ad57344ad334988fc410`;
- proof program:
  `815bd80dd38b5fb2f3ab4f918b4b1132462a1ffd44433c2855d5ccd5978f669d`;
- corrected trace commit: `d0beb42dd`.
