### Latest-v20 integration: clean selector PR passes the frozen causal gate

The minimal `oldest_boundary` implementation has now passed the decisive
end-to-end gate on the newer topology-calibrated v20/NF3 base, not only on the
earlier causal image.

#### Exact stack

```text
base:
  voipmonitor/vllm@sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0
  vLLM 0c79e41db4 / SparkInfer e603f74bb6

derived image:
  glm52-serve:v20-20260726-oldest-boundary-pr-candidate
  sha256:43e5a48781ee5cf40a92cc494749b21306b72280bd1a875721a45422323f2599

selector commit:
  1a71afe0 fix(indexer): preserve oldest threshold-boundary candidates

installed tiled_topk.py:
  b15bab73f1fcd6434f712f6fc99ec5369104969cb9157ae473926bf40d72e23b
```

The image contains no other code overlay. Runtime remained TP4/DCP4/MTP3,
NVFP4 MLA KV, FP8 RoPE, `i8_ring`, MNBT 3,072, and max length 480,000.

#### Safety and capacity

The model-free 3,072-row production-profile geometry passed with every count
correct and every index in bounds:

```text
6d32434593a932026ad16fdde2aded4f5e1b45c584cadc52452edf4397e6b23d
```

The first boot at GMU 0.974 correctly stopped before serving because the newer
reusable CUDA-graph pool now charges its complete 1.03-GiB measured high-water
mark. The boot itself calculated 0.9848 as the equivalent posture. With that
single configuration correction:

```text
health:             healthy
restart count:       0
KV pool:             545,280 tokens
max model length:    480,000
max concurrency:     1.14x
```

The actual production graph capture used 0.19 GiB versus the conservative
1.03-GiB estimate. That 0.83-GiB estimator gap is a separate memory-reclaim
opportunity, not part of this selector fix.

#### Frozen end-to-end result

The exact 250k known-good control and all three byte-frozen 350k prompts that
stock v20 missed returned exact finalized content:

| Cell | Prompt tokens | Cached | Finish | Completion | Final content |
|---|---:|---:|---|---:|---|
| 250k control | 245,497 | 0 | stop | 4 | `738216` |
| 350k-r1 | 343,727 | 0 | stop | 4 | `738216` |
| 350k-r2 | 343,727 | 0 | stop | 4 | `738216` |
| 350k-r3 | 343,727 | 0 | stop | 4 | `738216` |

Verdict:

```text
CONFIRMED — every frozen stock-FAIL prompt recovered
```

Pinned summary:

```text
dda7bddd33919d0947bcf45e0731c7fe07e1d4918944781fca9928cafe1d18f6
```

The current-base randomized cold 50k--475k ladder is now running. KLD and
matched throughput remain promotion gates, but the causal relationship and
current-base compatibility are established.
