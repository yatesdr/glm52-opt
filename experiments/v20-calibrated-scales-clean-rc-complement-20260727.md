# V20 calibrated NVFP4-scale clean-RC complement

Date: 2026-07-27

## Question

The official GLM BF16/FP32 scorer plus exact top-k recovered the frozen
343,727-token r1 only after the calibrated per-layer NVFP4 MLA outer scales
from vLLM PR #145 were enabled. Does the clean runtime-stride RC also recover
with its stock accelerated scorer and the same scales?

This complement distinguishes two materially different endpoints:

1. **Stock scorer passes:** PR #145 calibration is sufficient on the clean RC.
   No indexer accuracy patch is justified by this row.
2. **Stock scorer fails:** calibrated main-KV inputs and the official scorer
   are jointly sufficient, while neither clean official scoring alone nor
   the tested scales-only path is sufficient. The accelerated indexer then
   needs a deployable precision improvement identified by offline replay.

## Provenance correction

The earlier scales-only miss is not a clean-RC complement. Its archived
container identity is:

```text
image:       sha256:2566f905f13252c514a0f96c177ba982bd16321943927966310bf8c7c92d94b7
vLLM:       551719766029e78824a30d97ae6ac63917405b5f
SparkInfer: be0edcaae6f5d284bb29a82325aba7a0ead6960f
```

That SparkInfer pin predates the runtime-stride fix in `c3828fd`. Its
`ABSENT` result remains valid for that image, but cannot establish an
interaction on the current clean RC.

Archived inspect SHA-256:

```text
88b7deba7fe9e48139c8784f696b1659b49ac147d53c21db8374e8aba9253e0a
```

## Valid joint recovery

The passing joint cell used:

```text
image:       sha256:899e64cc6098407d1e41bca8db53f70ea60f31009b812872e4690540798ded1a
vLLM:       0c79e41db41f250ccdfc4be92d171960a5787f73
SparkInfer: c3828fd7f807ce237a9ac36ef033659e6f6b6dd3
scorer:      official BF16/FP32, exact top-k
KV:          nvfp4_ds_mla, FP8 RoPE, 368-byte record
wire:        i8_ring
scales:      efd7e23ac1ace6da9dcd9046c46bca5cca68ed5e89cd648b5f8bc1d51eafebb2
execution:   TP4/DCP4/MTP0, eager, prefix cache off, MNBT 3072
```

Runtime gates:

- healthy, restart count 0;
- `latent_scale_identity=0` on all four ranks;
- GPU KV pool 837,697 tokens at max-model-len 360,000;
- zero fatal signatures.

Frozen result:

```text
fail-350k-r1  EXACT  gate=PASS  finish=stop  out=4
prompt_tok=343727  cached=0  elapsed=667s  content=738216
```

Evidence:

```text
harness/cn4-evidence-archive/20260727/
  official-reference-nvfp4kv-fp8rope-i8ring-scales-r1-v1/
```

Key hashes:

```text
rows.json:       d1f3c7c8628739fb748ab316f2ff9628971776a9937422261c1455e65b4a31e4
response:        7669c61ef77273d5b696251b3f5869109ac42b373f1b57a0353b29b709f33277
container.log:   458165b678b75a90f8c92fe91ee5c5c81627b6f186815c9211d40734ecb3876c
container inspect:
                 2cd91ee11d4469f8ad263163d1d86a7677b14b5e98129acdc5a461757665c74d
```

## Clean-RC complement contract

The stock and official-reference images share the exact source pins:

```text
stock image:     sha256:e288bc87717df765769052decfe716c63af87ef54e6b09e7d93e6b85ff8f9dae
reference image: sha256:899e64cc6098407d1e41bca8db53f70ea60f31009b812872e4690540798ded1a
vLLM:            0c79e41db41f250ccdfc4be92d171960a5787f73
SparkInfer:      c3828fd7f807ce237a9ac36ef033659e6f6b6dd3
```

The complement holds the passing posture fixed and removes only the official
reference-scorer mode, selecting the stock accelerated scorer with exact
top-k. It uses a fresh compile namespace and the same frozen r1.

Compose:

```text
compose/glm52-v20-runtime-stride-stock-nvfp4-scales-r1-20260727.yaml
SHA-256:
bb99caf70ebcdc17f46f6d1a2e3381879bc9fb205d1158d4730e5842f7ac5ed7
eager/prefix-off invocation wrapper:
docker/vllm-official-reference-eager
SHA-256:
eecb84231e30482cd26128dfff932fb255bcfb360cdc90f68ddc9bf4998c9ced
```

The first launch failed closed before engine construction because the stock
launcher always supplied CUDA-graph options while `GRAPH=0`; the reference
image's wrapper had previously added `--enforce-eager`. The failed launch
made no model request and is archived under `failed-graph0-launch/`. The
corrected compose read-only mounts that already-pinned invocation wrapper and
uses a second fresh cache namespace. This is an execution-posture correction,
not a model or scorer change.

## Claim discipline after the result

One r1 pass is a causal discriminator, not promotion evidence. Whichever arm
wins still requires:

1. healthy-posture layer-entry trace;
2. frozen 250k control plus all three 350k rows;
3. randomized cold 50k--475k ladder;
4. KLD n=3, performance, capacity, graph/restart/cache gates.

## Result

The clean-RC stock accelerated scorer passed:

```text
fail-350k-r1  EXACT  gate=PASS  finish=stop  out=4
prompt_tok=343727  cached=0  elapsed=327s  content=738216
```

Runtime gates:

```text
image:                  sha256:e288bc87717df765769052decfe716c63af87ef54e6b09e7d93e6b85ff8f9dae
health / restarts:      healthy / 0
scorer:                 stock accelerated, exact policy
reference mode:         unset
latent_scale_identity:  0 on all four ranks
scale JSON:             efd7e23ac1ace6da9dcd9046c46bca5cca68ed5e89cd648b5f8bc1d51eafebb2
KV pool:                775,778 tokens
fatal signatures:       0
```

Evidence:

```text
harness/cn4-evidence-archive/20260727/
  runtime-stride-stock-nvfp4-scales-r1-v1/
```

Key hashes:

```text
rows.json:       1cdd5b1fa6c3438616f175509b3b5b5300ba3b69dfe9907a7858e0bd7989bda1
response:        0b9ecd3c82b1f89eb797dcd9691b3a8f406e6be27fcdc73ea83a5268185efea0
container.log:   4b3669a70aecadb173b9f3207f64982c856f9f5fbe88a1bbca451eecfbc01f20
container inspect:
                 266e7c36f2245bc96e3063258a46b6873dfd2e771319f07ea19a90b23bea969f
```

## Interpretation

The calibrated NVFP4 MLA outer scales are sufficient for this frozen deep
retrieval row on the clean stock accelerated scorer. The official reference
scorer is not required for the measured recovery, so there is no evidence
from this row supporting an indexer accuracy patch.

The minimal measured endpoint preserves:

- v20's bounds-safe exact selector;
- the stock accelerated FP8 indexer;
- the compact 368-byte NVFP4+FP8-RoPE cache record;
- `i8_ring`;
- existing vLLM/SparkInfer runtime-stride RC pins.

The configuration change is to enable and canonicalize the shipped PR #145
scale file for this checkpoint. This conclusion remains scoped to the causal
r1 until the healthy trace, complete frozen gate, randomized ladder, and
promotion evidence finish.
