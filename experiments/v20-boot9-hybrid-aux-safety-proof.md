# v20 Boot 9 — hybrid shared-expert/resident-grid source-fix proof

Date: 2026-07-23  
Operator: Fable  
Objective: prove the diagnosed v20 CUDA-concurrency defect at the unchanged
480,000-token production geometry, then qualify the same process

## Finding

The v20 image branch `3e731bc0` does not contain the sibling-release fix
`a11733f5` (`fix(moe): order shared experts before resident grids`). During the
SparkInfer namespace migration, the plan-level auxiliary-stream capability was
also removed from the new SparkInfer line. vLLM retained a conservative helper
that returns false for unqualified B12X plans, but no runner code consulted it.

Consequently, current v20 scheduling does this for GLM hybrid MoE layers:

1. enqueue the routed Grid188/W4A16 resident-grid kernel on the main stream;
2. enqueue the shared-expert GEMMs on an auxiliary stream; and
3. wait for the auxiliary stream only after both were submitted.

The hybrid routed kernels use device-wide software barriers and require
exclusive residency. Auxiliary GEMM work can occupy resources required by a
barrier participant. This is an invalid launch-order contract, not a model
length, KV-memory, DCP-route, or graph-cap problem.

The runtime evidence agrees with the source history:

- Boot 7 localized the first fault to the production decode-speculator eager
  forward at M=9; the identical profiling descriptor passed.
- M=9 is the first W4A16 fallback descriptor. Boot 8 added cooperative
  admission to that fallback and progressed farther before failing.
- Smaller descriptors still use the hybrid/Grid188 resident path, which Boot
  8 left concurrent with the shared-expert stream.
- The sibling v20 branch already carries the broader ordering/capability
  design, but it was absent from image commit `3e731bc0`.

## Minimal fix

Local vLLM commit:

```text
bf1b32cf7a70810b047d683cfb1b599b6e7304bd
fix(moe): gate unsafe hybrid shared-expert overlap
```

Patch artifact:

```text
patches/v20-hybrid-shared-expert-aux-safety/
  0001-fix-moe-gate-unsafe-hybrid-shared-expert-overlap.patch
SHA-256 90b28e8b63fec21da5cb6a86e7a0b608214138295eca7bf0ee2b0e48404239b0
```

Production change: 30 added lines across four files. Tests add 50 lines and
remove two old import lines.

The patch adds a quant-method capability queried by `SharedExperts` before it
selects its auxiliary CUDA stream. `NvFp4Nf3HybridMoEMethod` returns false for
that capability at every token count because both its Grid188 and per-tier
fallback launches require exclusive residency. Other quantization methods
keep the existing default and behavior.

For the GLM hybrid target and MTP layers, the resulting order is:

```text
shared expert on main stream -> routed resident grid on main stream
```

This does not disable MTP, CUDA graphs, i8-ring, DCP4, A2A, graph cap64,
MNS16, KV quantization, DRAM offload, NVMe offload, or 480k context. The
expected tradeoff is a small decode-latency cost from serializing the shared
expert with the routed expert for this hybrid method. Measure it at C1/C8/C16.

## Byte pins

Verify these inputs in the Boot 8 image before applying the patch:

```text
b8d3f53f46614f2db1b4cf9be058296f84191e2437887fb0dd3dfc42bd8c5f48  fused_moe_method_base.py
531a271b0f9280dbc26a0638598fec613a6f6a35e9c43ae65238e622493eda1b  moe_runner.py
5d1cc3158e2eddc5b3bfe88ecd50a390e18e9fe0c58fde0060c873783ab813b6  shared_experts.py
a8b4e19c5e776ece1d6c7ff2c48da236d1bd4032a3399f7b6a9563955c99f61b  nvfp4_nf3_hybrid.py
```

Require these outputs in the built image:

```text
c6a86a640a0ab67f3668d6aa1a669d4ff4996bac34b0a6d368d70b4f1a07f2f2  fused_moe_method_base.py
23713a84fcbd193098726164030a4b6060099f48e804c989b4475f6e5e891c15  moe_runner.py
8b7f6df20d7c2a1fc49a46a59582d3a784b5be4e29f5507781bdf92b094f567e  shared_experts.py
904aaa0bd62051fa462704ea9211393ad8b0a8da5a95d44ba9d5eec9dadfc760  nvfp4_nf3_hybrid.py
```

Retain every Boot 8 pin, including the W4A16 cooperative patch, CKV
profiling-generation reset, MRV2 pool fix, aligned indexer, INT8 wire mode,
DRAM connector fix, and bounded NVMe tier.

## Gate 0 — build proof

Build from the exact Boot 8 image/source. Apply only the patch above. Run:

```bash
python -m pytest -q \
  tests/model_executor/layers/fused_moe/test_shared_experts_stream.py \
  tests/quantization/test_nvfp4_nf3_hybrid.py
```

At minimum require these new tests to pass if unrelated fixtures in the full
files are unavailable:

```text
test_aux_stream_respects_expert_kernel_capability
test_hybrid_moe_rejects_shared_expert_aux_stream
```

Also require Python import/compile, `git diff --check`, clean patch
application, and the four runtime output hashes above.

## Gate 1 — one 480k proof boot

Use the unchanged Boot 8 production geometry:

```text
MAX_MODEL_LEN=480000
TP=4 / DCP=4 / MTP=3
MAX_NUM_SEQS=16
MAX_NUM_BATCHED_TOKENS=3072
graph cap=64
GMU=0.980
A2A threshold=64
SPARKINFER_PCIE_DMA_FP8=i8_ring
VLLM_PCIE_DMA_FP8=i8_ring
DRAM offload=64,000,000,000 bytes
NVMe acceptance cap=8,589,934,592 bytes
```

Enable the already-baked descriptor diagnostic for this proof:

```text
VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS=1
```

Keep `CUDA_LAUNCH_BLOCKING` unset. Do not change max length, GMU, MNS, MNBT,
graph cap, A2A route, allocator configuration, cache volumes, or offload
limits.

PASS requires:

1. profiling and production target captures complete;
2. profiling and production prefill-speculator captures complete;
3. production decode-speculator M=16 through M=1 all complete;
4. the former M=9 tuple explicitly reports `warmup_forward PASS` and proceeds
   through fresh inputs, prewarm, and full capture;
5. the API reaches serving state;
6. liveness returns `4` and an ordinary request reports `finish_reason=stop`
   with nonzero MTP acceptance;
7. `RestartCount=0`, with stable container ID and `StartedAt`; and
8. no illegal access, OOM, Xid, assertion, EngineDead, worker exit, or 5xx.

The diagnostic synchronization changes timing, so reaching serving is source
proof but not final production qualification. If it passes, keep the process
up only long enough for liveness/MTP evidence, then perform the clean
qualification boot below.

On failure, stop and preserve the complete descriptor trace and inspect JSON.
Do not tune or start a ladder. A different failing CUDA launch would require
`CUDA_LAUNCH_BLOCKING=1` on this exact source and geometry before another fix.

## Gate 2 — clean qualification and promotion

After Gate 1 passes, restart the exact same image and Compose with only
`VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS` removed. Run on that same process:

1. decode C1, C4, C8, and C16, recording aggregate/per-user throughput and MTP
   acceptance;
2. unique-prefix needles at 300k, 350k, and 475k;
3. bounded NVMe fill, turnover, promotion, and persistence proof;
4. 16 overlapping unique-prefix 50k requests proving overflow beyond the GPU
   pool into DRAM/NVMe;
5. cold 8k and 50k prefill with prefix-cache miss evidence; and
6. final liveness, container identity, restart count, tier-capacity inventory,
   and fatal-signature audit.

If all gates pass, perform the planned identical persistence restart, prove a
retained NVMe entry can be promoted, rerun liveness plus one MTP request and a
475k needle, and leave that exact v20 process online.

## PR rule

Do not publish this as a ready PR from local/static evidence alone. After the
480k diagnostic boot and clean qualification agree with the mechanism, open a
draft against the branch that actually builds the v20 image and include the
Boot 7/8/9 descriptor evidence. Move it to ready only after clean runtime
qualification.

