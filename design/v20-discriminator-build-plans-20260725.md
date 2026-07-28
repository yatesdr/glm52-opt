# v20 needle-regression discriminator build plans

Date: 2026-07-25  
Owner: Sol  
Runtime owner: Fable  
Status: staged; build only after the no-model gates choose a candidate

## Fixed control

Both candidates start from the source state represented by the live
`fa71a0c1` image:

```text
image tag:
  ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-prod-ready-20260724
local image id:
  sha256:fa71a0c1e06e29db88364dcaa047c09c37662fda105551a988a2d09e54fdec86
current GPU KV pool:
  501,504 tokens
```

No candidate changes the compose, model checkpoint, i8_ring wire mode,
maximum length, DCP geometry, graph sizes, or offload configuration.

## Candidate A: remove #171 only

Purpose: restore the pre-#171 compact-NVFP4 MTP verifier route while retaining
all later query, memory, offload, and SparkInfer changes.

```text
worktree: workspace/vllm-v20-staged-query-no171
commit:   b8534c4a5fad9500c0aebd0ef5e293672033fc4e
input b12x_mla_sparse.py:
  885401fc5dc10dcfa02fe6fc524763bb9a44150e4cf743c1e6f9a3460002da74
output b12x_mla_sparse.py:
  585bf3e008d4149db94684050649cb2bb0daf72ca8e90144346157b1cd354ff0
Dockerfile:
  docker/Dockerfile.v20-no171-discriminator-20260725
```

This is a Python-only derived image. The Dockerfile checks the live image's
input byte, copies exactly one production file, compiles it, checks the output
byte, and verifies the old route expression.

Suggested build:

```bash
docker build \
  -f docker/Dockerfile.v20-no171-discriminator-20260725 \
  -t ghcr.io/yatesdr/glm52-serve:v20-no171-discriminator-20260725 \
  .
```

### Expected KV-pool effect

#171 reduces the maximum verifier scratch rows from 64 to 16 at MNS16/MTP3,
recovering at least 99.38 MiB/GPU:

```text
tmp_output  96.00 MiB
tmp_lse      0.38 MiB
output       3.00 MiB
```

Removing it gives those bytes back. At the measured approximately 8 KiB per
KV token, the first-order estimate is:

```text
501,504 - (99.38 MiB / 8 KiB) = approximately 488,784 tokens
```

Allocator rounding and graph profiling can move the observed result, so this
is a prediction, not a fit proof. It likely remains above 480k but below the
500k preferred floor, with only about 1.8% maximum-length margin.

## Candidate B: retain #171 and preserve FP32 safe-query reductions

Purpose: retain the current compact-NVFP4 verifier qualification and add a
format-boundary precision option. Both modes retain tensor-core-eligible
`CUBLAS_COMPUTE_32F`; only the BF16 absorbed query that is immediately
requantized to FP8 sets
`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`. The exact incoming cuBLAS
math mode is restored after the call.

```text
worktree: workspace/vllm-v20-safe-query-fp8-accum-with171
branch:   fix/v20-safe-query-fp8-accum-with171-20260725
commit:   5a8204db178f4aa6c47ffc46c40ef11ddd497586
base:     e16288f5d006726b0492981d3db5627fc5d9f70e
```

Pinned changed production bytes:

```text
vllm/model_executor/layers/attention/mla_attention.py
  2db31d7d5ec71fa93567207819dd2e325d2201784553d126850b3d8b3c0dda36
csrc/libtorch_stable/attention/mla/safe_query_bmm.cu
  4055e6b0f59a66cfc16a26a103e9c25a5218434543bafc84f7b6c97e1814a954
csrc/libtorch_stable/ops.h
  191bec362d52c5f54ede3f8c117f7871611da6811949b6e8139dc598f9abf87e
csrc/libtorch_stable/torch_bindings.cpp
  416fdfca209d0b3a5f0134138cfcc8c255dd75cc7e8d3e71424de060e221e5c3
```

Static proof:

```text
harness/v20_safe_query_accum_static_proof.py
sha256 298e8afbf3b6b0df338294ffe6be92d0e11ecaeebfb1076d5ab250a4130c5620
```

This candidate changes the compiled stable-libtorch operator schema and
therefore **cannot** be constructed as a Python overlay on `fa71a0c1`.
Build the exact committed vLLM source with the current gilded-gnosis v20 build
recipe, then layer the same already-pinned runtime/SparkInfer files used by
`Dockerfile.v20-prod-ready-20260724`. Fail the build unless:

1. all four production hashes above match;
2. the stable extension contains `safe_mla_query_bmm`;
3. Python reports the four-argument schema with `bool precise=False`;
4. the regular and precise CUDA tests both pass, including graph replay and
   exact restoration of the incoming cuBLAS math mode;
5. the #171 backend byte remains
   `885401fc5dc10dcfa02fe6fc524763bb9a44150e4cf743c1e6f9a3460002da74`.

The branch is build-ready at the source level. An executable full-image recipe
must use the maintainer's current CUDA build base; the final runtime image does
not contain the compiler needed to replace `_C_stable_libtorch.abi3.so`.

### Expected KV-pool effect

The precision option adds no persistent tensor, workspace, or graph-sized
buffer. Since this candidate retains #171, its first-order expectation is the
current 501,504-token pool, with no systematic loss. Runtime profiling must
still measure this. The reduction guard can change cuBLAS's selected algorithm,
so the dedicated GPU proof must report its latency before a model boot:

```text
harness/v20_safe_query_accum_gpu_proof.py
sha256 22f1c412b0f548b33c9448c047af77d99e38a9139bff784bbf14929acd6f8ea9
```

## Source prediction versus the observed sharp cliff

Neither Candidate A nor Candidate B alone predicts a hard transition between
roughly 60k and 70k:

- #171's verifier route is active at every tested depth.
- safe-query compute mode is active at every tested depth.

The field boundary instead coincides exactly with the production fused
indexer's runtime merge switch:

```text
serial:       local sequence <= 16,384
cooperative:  local sequence >= 16,385
DCP4 global crossover: 65,536
```

However, the fused file and the helpers it actually uses are byte/AST
equivalent between SparkInfer `ffa922b` and `a93df671`. Therefore the
cooperative arm is not a standalone post-6d32 source regression. It is a
high-value **interaction** candidate: changed query numerics may expose a
latent merge-arm defect that the serial path does not.

Run the pinned no-model crossover discriminator before choosing a model build:

```text
harness/v20_fused_indexer_16384_crossover_probe.py
sha256 edf2c93be7df939a5d9fb48b0850d2184a2e4de6dfe1c60b0bd39229e0b17a2e
```

If that probe fails, fix or safely disable the cooperative arm first and do
not spend a model boot on Candidate A or B in isolation. If it passes, Gate A
chooses between the two pinned model candidates.
