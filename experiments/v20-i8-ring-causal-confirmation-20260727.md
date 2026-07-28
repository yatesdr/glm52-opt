# v20 block-INT8 DCP wire causal confirmation — 2026-07-27

## Question

After the frozen 350k r1 request recovered under the combined
`fp8_ds_mla`/BF16-RoPE/raw-BF16-wire posture, does restoring the production
`i8_ring` DCP transport cause the request to fail again?

This is a one-row causal confirmation. It is not a claim that block-INT8 is
bit-exact or that transport precision never contributes to another prompt.

## Held constant

- derived image:
  `glm52-serve:v20-official-reference-runtime-stride-chunk16-75715e51`;
- image ID:
  `sha256:899e64cc6098407d1e41bca8db53f70ea60f31009b812872e4690540798ded1a`;
- official GLM BF16/FP32 scorer and production exact top-k;
- NF3 hybrid checkpoint;
- `fp8_ds_mla`, 656-byte cache record, BF16 RoPE;
- TP4/DCP4, MTP0, eager execution, no prefix caching;
- MNBT 3,072, max model length 360,000, GMU 0.950;
- query split, owner merge, and CKV prefetch disabled;
- immutable prompt and deterministic decoding parameters.

The intended changed variable relative to the preceding recovered run was:

```text
F8_DMA: 0 -> i8_ring
```

The launched compose is pinned at:

```text
harness/cn4-evidence-archive/20260727/
  official-reference-fp8kv-bf16rope-i8ring-r1-v1/compose.launched.yaml

sha256 6716295d99c15bc04ee414d247eac0674adfa224bf0e439c575cf1867c9ffe20
```

## Result

| Field | Result |
|---|---|
| frozen row | `fail-350k-r1` |
| prompt SHA-256 | `f0d1c16d816b777f27a3882d9e6b5ef056852684ea155fb11dd845f9e1654ab5` |
| prompt tokens | 343,727 |
| verdict | `EXACT` / gate `PASS` |
| content | `738216` |
| completion tokens | 4 |
| finish reason | `stop` |
| cached tokens | 0 |
| elapsed | 689.0 s |
| container | healthy, zero restarts |
| measured KV pool | 491,769 tokens |

Pinned evidence:

```text
harness/cn4-evidence-archive/20260727/
  official-reference-fp8kv-bf16rope-i8ring-r1-v1/

run.log
  sha256 43b0e4b07d8f7fcb80cc9cf3a16f5ee798bc9e26bcdb0c834f06c088d120395a
results/summary.json
  sha256 a80eec20e5291a1d209ab04a4e1d5985fae8d2b4a5070f379bb86421cc21f3ec
results/rows.json
  sha256 dc909bd4d12cf2d3785732a6801718dd4967796d8c5508ee8ae2fac561318641
results/resp-fail-350k-r1.json
  sha256 2357b5d55cec49f009fbe7b1fed4d9d357b1a09792d832c2f5f4f616b6bb5112
container.log
  sha256 6b7107430d05db263598570cd474fb5fc944e0a7920e4cc0fded84510a17313f
container-inspect.json
  sha256 1b5eea8a109b89a55a44ea200861ace971c7ba78d1795e89e26d02684e8cd267
```

## Interpretation

`i8_ring` is sufficient for this frozen row when the main cache is
`fp8_ds_mla` with BF16 RoPE. It is therefore not the minimal cause of this
row's failure in the compact v20 posture.

The transport is a rank-consistent block-INT8 codec, not a lossless byte
transport. It uses an FP32 scale per 128 values and has a pre-BF16 absolute
error bound of `amax / 254`. This result shows that its measured error does
not cross the retrieval boundary for this row under the held cache posture;
it does not establish bit equality for arbitrary activations.

Historical evidence is consistent:

- the later block-INT8 field test recovered the then-failing 300k and 350k
  rows 3/3 each;
- the original v19 E4M3 `ag`/`ring` failures are not evidence against
  block-INT8 because those are different codecs;
- v19 also used uncalibrated NVFP4+FP8-RoPE and could recover under BF16 wire.

That last point prevents overclaiming the dormant calibrated-scale support as
the unique v19-to-v20 source differential. It can still be the clean minimal
quality restoration for v20 if it repairs the compact posture.

## Next causal cell

The next test holds the official scorer, exact top-k, 368-byte
NVFP4+FP8-RoPE record, `i8_ring`, and execution posture constant and enables
only the calibrated per-layer outer scales shipped default-off by vLLM
PR #145:

```text
VLLM_NVFP4_MLA_SCALES_FILE=
  /opt/vllm/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json
```

Scale-file SHA-256:

```text
efd7e23ac1ace6da9dcd9046c46bca5cca68ed5e89cd648b5f8bc1d51eafebb2
```

The file declares 78 finite positive layer scales, latent width 512, and
denominator 2,688. The exact SparkInfer build includes
`latent_scale_identity` in both prefill and decode compile-cache keys. The
test also uses a fresh compile namespace.

An r1 recovery would select calibrated NVFP4 scaling as the leading minimal
quality fix, but the frozen four-row gate and healthy-posture layer trace
remain mandatory before a regression-wide or production claim.
