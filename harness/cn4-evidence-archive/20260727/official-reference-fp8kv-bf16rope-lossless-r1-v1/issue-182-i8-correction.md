### Correction: `i8_ring` is rank-consistent, not numerically lossless

I characterized `i8_ring` too broadly in the preceding update. The source and
its existing codec proof show:

- one signed-INT8 value per activation;
- one FP32 scale per 128 values;
- `scale = amax / 127`;
- round-to-nearest with saturation to `[-127, 127]`;
- pre-BF16 absolute error bounded by `amax / 254`;
- identical owner/peer materialization from the same payload.

It is therefore rank-consistent and materially higher precision than the prior
E4M3 transport, but not bit-exact BF16 for arbitrary activations. The existing
four-rank integrity proof uses integer-valued tensors chosen so the expected
collective is exact; it does not prove general numerical equivalence to raw
BF16.

This correction strengthens the current experiment. Restoring `i8_ring` while
holding FP8-DS-MLA, BF16 RoPE, the official scorer, and the request constant is
a genuine precision discriminator:

- if retrieval remains exact, the causal set narrows to main-KV/RoPE
  representation;
- if retrieval fails, block-INT8 wire rounding is a causal contributor, after
  which codec loss must be separated from routing/collective defects.

The preceding causal recovery result itself is unchanged.
