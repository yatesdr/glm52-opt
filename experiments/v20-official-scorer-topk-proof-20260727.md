# GLM-5.2 v20 official-scorer / production-top-k proof

Date: 2026-07-27

Status: **production top-k cleared; full raw-input scorer reconstruction in
progress**

## Question

Luke requested a direct composition test:

1. construct the reference GLM indexer score field on real frozen 350k
   activations;
2. pass those same FP32 scores to the production vLLM/SparkInfer top-k kernel;
3. compare selected indices and values with `torch.topk`, allowing differences
   only where an explicitly characterized cutoff tie permits them.

This report records the first completed portion and its exact boundary.

## Frozen input

The proof used the immutable layer-0 trace from the cold frozen
`fail-350k-r1` request:

| Field | Value |
|---|---|
| Rendered prompt tokens | 343,727 |
| Query absolute position | 343,726 |
| Indexer heads | 32 |
| Head dimension | 128 |
| Selected tokens | 2,048 |
| Trace chunks | 112 |
| Trace manifest SHA-256 | `8e6bf39b5fda414f9120fc998641002a842d06cb81db736f707768b5f7d3f526` |
| Q SHA-256 | `d18e1254afd57e41b043bd38ce7777e682f5e2e85b9902feb721e0dfa1f63de1` |
| K SHA-256 | `6bf83dc96fbb86be528f54ebf6ed07c58afc10b12db869a95a0d36d47660a0f0` |
| Learned-weight output SHA-256 | `396cabf75d88877a86c87601dd53832f976032ffc2f07ba187477160b2a6b5be` |

The trace schema captures Q/K after vLLM has applied indexer K normalization,
GLM interleaved RoPE, and full 128-D concatenation, but before FP8
quantization. The proof then reconstructs the remaining Hugging Face score
order literally:

```text
Q.float @ K.float.T
* 128**-0.5
ReLU per head
* learned head weights.float
* 32**-0.5
sum over heads
causal mask
top-2048
```

TF32 was disabled. The resulting 343,727-element FP32 score row has SHA-256:

```text
09577efde901ff04cd94afd387024fd94b9905b11c63dbb204c6afd9c05924a5
```

## Result

The identical FP32 score row was selected through `torch.topk` and two
production SparkInfer entrypoints:

| Selector | Index set vs `torch.topk` | Values by index | Verdict |
|---|---|---|---|
| `run_row_topk` | bit-exact | bit-exact | PASS |
| `run_tiled_topk` (`BLOCK_Q=32`, `BLOCK_K=256`) | bit-exact | bit-exact | PASS |

All three paths produced:

```text
canonical indices SHA-256:
ce4855112d29431193b6a56e712d132d7439df94c4d885c235273a462979b62d

canonical values SHA-256:
790dbd06de4fcace07b20655fec89492658fe13d79606091a0dc903132de7be1
```

The cutoff score was `5.745027542114258`. There were 2,047 strict winners and
one cutoff token, so the result has no cutoff-tie ambiguity.

## Finding

The production exact top-k kernel does not change the selected set or values
when it receives this real FP32 score row. The long-context retrieval loss is
therefore upstream of exact top-k: score construction, cache semantics, DCP
composition, or an earlier model-trajectory divergence.

This does **not** yet prove that the score row itself matches the official GLM
implementation. The older trace begins after K normalization and interleaved
RoPE, and its learned head weights came from vLLM's fused BF16
`wk + weights_proj` projection.

The Hugging Face implementation instead keeps `indexer.weights_proj` in FP32
and evaluates that projection separately. The checkpoint stores the
layer-34 `weights_proj` tensor in BF16, but Hugging Face upcasts the module and
input to FP32 for the projection. The next trace will preserve raw K,
pre-RoPE Q, positions, K-norm parameters, final hidden state, and the separate
projection weights so this difference can be measured directly.

## Pins

```text
proof JSON:
f40236b83746093538e5ba4aca18a40d583b3e811bf28767f177aa88268a31ed

proof harness:
86d7236f41ad9e2f25d0deb31560e341899f80329e4260dc776211e03c77f8ba

production tiled_topk.py:
57d5ec00c70e60024bd94bb6afbb9174144c9c07cf88e5568a996ab511bc24c5

base image:
voipmonitor/vllm@sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0
```

Evidence:

- `harness/cn4-evidence-archive/20260727/official-scorer-topk/post-rope-production-topk-v1.json`
- `harness/v20_glm_reference_scorer_production_topk_proof.py`

Comms record: `proofs#217`.
