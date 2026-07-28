# GLM-5.2 v20 NVFP4 scaling KLD comparison

Date: 2026-07-28

## Result

Lower KLD is better.

| Configuration | Runs | Mean KLD | Sample SD | Min | Max |
|---|---:|---:|---:|---:|---:|
| Historical `oldest_boundary`, uncalibrated NVFP4 | 3 | 0.16044075 | 0.00297924 | 0.15700885 | 0.16236257 |
| A: exact + current online weights + static #145 scales | 3 | 0.14622770 | 0.00468791 | 0.14180147 | 0.15113949 |
| B: exact + current online weights + dynamic per-token scales | 3 | 0.13903565 | 0.00201006 | 0.13672545 | 0.14038452 |
| C: exact + quality-first MXFP8 weights + static #145 scales | 3 | 0.13378422 | 0.00126327 | 0.13232554 | 0.13452094 |
| D: exact + quality-first MXFP8 weights + dynamic per-token scales | 3 | **0.13326164** | **0.00212500** | 0.13117311 | 0.13542133 |

Relative mean differences:

| Comparison | Absolute delta | Relative delta |
|---|---:|---:|
| Static calibrated minus historical bounded | -0.01421305 | -8.86% of bounded |
| B minus A: dynamic-scale effect under current weights | -0.00719205 | -4.92% of A |
| C minus A: quality-first weight effect under static scaling | -0.01244348 | -8.51% of A |
| D minus B: quality-first weight effect under dynamic scaling | -0.00577401 | -4.15% of B |
| D minus C: dynamic-scale effect under quality-first weights | -0.00052258 | -0.39% of C |
| D minus historical bounded | -0.02717911 | -16.94% of bounded |

The matched dynamic-minus-static per-run deltas were:

```text
-0.00574516
-0.01075497
-0.00507602
```

All three signs favor dynamic per-token scaling in the A/B comparison. The
paired mean delta was `-0.00719205`; its sample SD was `0.00310366`.

The matched D-minus-C per-run deltas were:

```text
-0.00133044
+0.00091515
-0.00115243
```

These values interleave: two favor dynamic and one favors static. The paired
mean was `-0.00052258` (`-0.39%` of C) with sample SD `0.00124828`.
At n=3 this does not distinguish the two scaling policies at run level.
The quality-first online-MXFP8 membership is the dominant additional gain.

## Per-run values

| Configuration | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Historical `oldest_boundary` | 0.15700885 | 0.16195084 | 0.16236257 |
| A: current weights + static calibrated | 0.14574215 | 0.15113949 | 0.14180147 |
| B: current weights + dynamic per-token | 0.13999698 | 0.14038452 | 0.13672545 |
| C: quality-first weights + static calibrated | 0.13452094 | 0.13450618 | 0.13232554 |
| D: quality-first weights + dynamic per-token | 0.13319050 | 0.13542133 | 0.13117311 |

## Contract and interpretation

All rows use:

- the GLM-5.2 NF3 hybrid checkpoint;
- the `nvfp4_nf3_hybrid` quantization backend;
- `nvfp4_ds_mla` KV cache with FP8 RoPE;
- TP4/DCP1, eager execution;
- the same 2,048 token IDs and 2,047 scored output positions;
- BF16 reference logits SHA256
  `87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063`;
- KLD direction `KL(BF16_reference || candidate)`;
- cleanup runner SHA256
  `d1dc1a63b9889e881f3bd899638d0ec65a1a1079132f6a207a600d9cba845405`.

A/B and C/D are each strict same-image scaling comparisons on image ID:

```text
sha256:db82fdcb5756d4a547853ba1330538bdd8a3dc0c6443c29bc49ba77b69b51cd1
```

The matrix additionally proves the writer compile identity for every run:

```text
static calibrated: per_token_scale=false
dynamic per-token: per_token_scale=true
```

A/B use the prior online quantization policy:

```text
{"linear":{"weight":"mxfp8"},
 "shared_experts":{"weight":"mxfp8"},
 "ignore":["re:^model\\.layers\\.0\\.",
           "re:.*\\.self_attn\\.indexer\\.",
           "re:.*\\.mlp\\.gate$",
           "model.layers.78.eh_proj",
           "lm_head"]}
```

C/D use Destroyed's quality-first policy:

```text
{"linear":{"weight":"mxfp8"},
 "ignore":["re:.*\\.fused_qkv_a_proj$",
           "re:.*\\.q_a_proj$",
           "re:.*kv_a_proj_with_mqa",
           "re:.*\\.mlp\\.gate$",
           "model.layers.78.eh_proj",
           "lm_head"]}
```

That policy keeps the listed projections, gates, draft `eh_proj`, `lm_head`,
and shared experts in their checkpoint precision. It is a deployment
configuration result layered on top of the dynamic-record implementation,
not a change to the record ABI or either upstream PR.

The historical `oldest_boundary` row uses the same KLD contract on an older
image ID:

```text
sha256:43e5a48781ee5cf40a92cc494749b21306b72280bd1a875721a45422323f2599
```

It is useful historical context, not a strict single-variable comparison
with the two scaling rows.

This shallow cell is not selector-sensitive: the context contains 2,048
tokens and the selector budget is also 2,048. Therefore the improvement
must not be attributed to `exact` versus `oldest_boundary` selection.
It measures the complete shallow inference distribution. The A/B row supports
the dynamic record path as a no-regression/improvement result under the prior
weight policy. C/D show that the quality-first weight membership contributes
the larger additional reduction and that static/dynamic KLD is
indistinguishable at this sample size under that membership. The frozen 350k
causal gate and randomized 50k–475k ladder provide the deep-context evidence
for dynamic scaling.

The qualified runner records per-run means, logs, configs, and compile proofs,
but does not retain the 2,047 individual per-position KLD values. Therefore
the block-bootstrap comparison requested in the follow-up characterization
spec cannot be reconstructed from this archive and is not claimed here.

## Evidence

Current matched static/dynamic evidence:

```text
harness/cn4-evidence-archive/20260728/
  nvfp4-dynamic-token-scale-kld-n3-v1/
```

Aggregate summary SHA256:

```text
996c2a58940bae4c16424a2176e5f9605fe4b2fd94f6f288e79a0b31a06ab579
```

Matched quality-first C/D evidence:

```text
harness/cn4-evidence-archive/20260728/
  kld-quality-weight-arms-DC-v1/
```

Aggregate summary SHA256:

```text
7f574369662bd3f12c4a16658ed63d6e13495c64e6e266e965e7608a2e902ffd
```

Derived runner SHA256:

```text
0f16b955aedaecad6b98a0d8581148810ca750f90ad0cf16addc566070ec32db
```

Historical bounded evidence:

```text
harness/cn4-evidence-archive/20260727/kld-pr84-final-n3/
```

Historical aggregate summary SHA256:

```text
e9bc81d775e6830b8b3101943a6bba714b50ffd65aab8255c4a5d07acb663ff3
```
