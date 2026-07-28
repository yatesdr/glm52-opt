## Causal update: shared upstream representation precision recovers frozen 350k row

The official GLM BF16/FP32 scorer is not sufficient to restore the frozen
deep-retrieval failure when it reads activations produced by the original
compressed main-attention posture. An exact token-inclusion trace showed why:
the ticket value did not enter the selected top-2,048 candidates until sparse
layers 62–74, too late to steer the answer.

I therefore held the official scorer and execution posture constant and
changed the shared inputs that produce the hidden-state trajectory:

| Component | Failed official-reference posture | Recovered posture |
|---|---|---|
| main MLA KV | `nvfp4_ds_mla` | `fp8_ds_mla` |
| RoPE field | compact FP8 | BF16 |
| DCP wire | `i8_ring` | raw BF16 |
| selector/scorer | official BF16/FP32 + exact top-k | unchanged |
| TP/DCP, MTP | 4/4, 0 | unchanged |
| execution | eager, prefix cache off | unchanged |

Frozen `fail-350k-r1` result:

| Field | Value |
|---|---|
| prompt SHA-256 | `f0d1c16d816b777f27a3882d9e6b5ef056852684ea155fb11dd845f9e1654ab5` |
| prompt tokens | 343,727 |
| verdict | `EXACT` |
| content | `738216` |
| completion tokens | 4 |
| finish reason | `stop` |
| cached tokens | 0 |
| elapsed | 708.8 s |
| container | healthy, zero restarts |

This is causal evidence for the **combined changed upstream trajectory
posture** on this frozen row. It does not yet identify which changed input is
necessary, and it is not yet a production configuration: the measured
491,769-token pool is sufficient for the causal request but below the
500k-at-480k promotion floor.

The next discriminator restores `i8_ring` while retaining `fp8_ds_mla` and
BF16 RoPE. Correction to my initial characterization: `i8_ring` is
rank-consistent block-INT8 transport, but it is numerically lossy. Each
128-value block uses one FP32 scale and signed-INT8 rounding, with pre-BF16
absolute error bounded by `amax / 254`. This is therefore a genuine precision
A/B as well as an implementation-integrity check. The subsequent cell
restores the compact NVFP4/RoPE representation. Each cell will record both
retrieval and the first sparse layer where the ticket tokens enter top-2,048,
so additive margin erosion can be distinguished from a single binary culprit.

Pinned local evidence:

```text
harness/cn4-evidence-archive/20260727/official-reference-fp8kv-bf16rope-lossless-r1-v1/
summary.json  db505691d4a6025eea8789e10d19e372133386fc990990d5845da4e866052cb9
rows.json     828ee3ae1684757a9dac6cd585883c12da4d53cc25c606c561919aced9b0bc14
response      0481d7f9cb209afa7ec5d70df820f777edd539ca9469306ed58ba56a4716ef3d
run.log       688cd35059df720aebef73d22e8f1f65a10224ef140d4ed0d1812bd9a384ba53
```
