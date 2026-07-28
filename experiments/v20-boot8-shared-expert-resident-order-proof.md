# v20 Boot 8 — shared-expert / resident-grid ordering proof

> **SUPERSEDED — DO NOT RUN.** Source review after this draft found that M=9
> crosses from the already-cooperative fused-micro path into SparkInfer's
> route-packed W4A16 fused body. That body contains whole-grid software
> barriers but its two launch sites omit cooperative admission. The replacement
> one-boot proof is `v20-boot8-w4a16-cooperative-grid-proof.md`. This vLLM
> scheduling patch remains local and unsubmitted because PR #150 previously
> disproved the broader shared-expert-overlap attribution.

Status: **ready for one discriminating CN3 boot**  
Patch commit: `2e852193a1ed1f1d47b3e6e20eb0a068797946f5`  
Image source under test: vLLM `3e731bc043d23ec21277fb76d3e15fe6da91b23b`, SparkInfer `1a88b389a8d14f26dbe4c157965938cfd8f1bf51`

## Finding

Boot 7 localized the first CUDA fault to the decode speculator's eager MTP
forward at `num_tokens=9`, after input preparation and before graph capture.
The MTP layer executes a GLM MoE block at that point.

The exact v20 source has an unsafe submission order:

1. `SharedExperts.maybe_sync_shared_experts_stream()` makes the auxiliary
   stream wait for its input, but does **not** enqueue the shared-expert GEMMs.
2. `MoERunner._apply_quant_method()` launches the routed MoE first.
3. Only after that launch does it enqueue shared-expert work on the auxiliary
   stream and make the consumer wait for it.

The hybrid routed kernels include resident-grid/software-barrier launches.
Submitting the auxiliary GEMMs behind that grid allows both streams to compete
for device admission in the unsafe order. This is the same concurrency class
identified on v19; the v20 image contains SparkInfer's cooperative fix for one
micro-MoE launch, but it does not correct vLLM's generic submission order or
cover every hybrid per-tier resident-grid launch.

The earlier DCP arithmetic lead is not supported by the source. DCP partitions
each request's KV sequence; it does not divide nine requests as `ceil(9/4)`.
Also, request counts 10 and 11 have the same unequal ceil/floor property and
passed Boot 7.

## Minimal fix

The patch changes only the shared-expert scheduling seam:

- enqueue shared-expert work during the existing pre-gate/pre-router stream
  synchronization point;
- allow it to overlap gate and router work;
- join the auxiliary stream immediately before routed-expert submission; and
- retain the existing output-lifetime `record_stream` protection.

This guarantees that routed resident grids and shared-expert GEMMs never run
concurrently. It does **not** disable MTP, CUDA graphs, Grid188, DCP4, A2A,
INT8 ring, DRAM offload, or NVMe offload.

Expected tradeoff: shared experts no longer overlap the routed expert kernel.
They still overlap gate/router work, so this is narrower than
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`. Measure decode after the boot passes;
stability is the first gate.

Runtime files:

| File | image input SHA-256 | patched output SHA-256 |
|---|---|---|
| `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | `531a271b0f9280dbc26a0638598fec613a6f6a35e9c43ae65238e622493eda1b` | `bb6c0c92842cc530c9462f232b6352de9915284f94d1390c279a3036ee032a41` |
| `vllm/model_executor/layers/fused_moe/runner/shared_experts.py` | `5d1cc3158e2eddc5b3bfe88ecd50a390e18e9fe0c58fde0060c873783ab813b6` | `a637d522e208f23dc4b84b672fccf85c8251cad8911e33848e1b25f30e5d0dee` |

Patch artifact:

`patches/v20-shared-expert-resident-order/0001-fix-moe-join-shared-experts-before-resident-grids.patch`

SHA-256: `a67d2081c423f1477f555d2ed863c608f3e91dd7291194c8407d74853c5a15f9`

The test-file change is source-review coverage and need not be copied into the
runtime image.

## Build gate

1. Start from the exact Boot 7 image/source stack.
2. Verify both runtime input hashes above before applying anything.
3. Apply the patch cleanly or copy only the two patched runtime files.
4. Verify both output hashes in the built image.
5. Keep all Boot 7 patches, image pins, and configuration unchanged.
6. Keep `VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS=1` so descriptor-level evidence
   remains available.
7. Remove `CUDA_LAUNCH_BLOCKING=1` if it was added for the separate diagnostic
   boot. Do **not** add `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`; that would hide
   whether this ordering fix works.

Do not add draft CKV-reset PR #169 to this boot. It is an independent real
lifecycle fix, but adding it here would introduce a second variable.

## Gate 1 — boot proof

Use the unchanged production candidate geometry:

- TP4 / DCP4 / MTP3;
- max model length 480,000;
- max sequences 16;
- CUDA-graph cap 64;
- the same GMU and MRV2 pool-reuse patch used by Boot 7;
- A2A, `i8_ring`, DRAM offload, and the staged NVMe tier unchanged.

PASS requires all of the following:

1. Profiling target, prefill-speculator, and decode-speculator captures finish.
2. Production target and prefill-speculator captures finish.
3. Production decode-speculator descriptors 16 through 1 finish all diagnostic
   stages.
4. The former failure tuple
   `num_tokens=9,num_reqs=9,uniform_token_count=1,warmup_forward` is `PASS`.
5. The API reaches healthy/serving state.
6. `RestartCount=0`, unchanged container ID/`StartedAt`, and no illegal access,
   EngineDead, worker exit, assertion, OOM, or Xid.

FAIL closed: on the first diagnostic failure, save the complete first-run log
and inspect JSON, then stop. Do not tune MNS, graph cap, GMU, DCP route, or
memory settings in the same boot.

## Gate 2 — minimal functional proof

Only after Gate 1 passes:

1. Run a clean liveness request that returns `4`.
2. Run one ordinary short MTP request and confirm `finish_reason=stop`.
3. Run one decode cell at concurrency 1 and one at concurrency 16.
4. Confirm nonzero MTP acceptance, no errors, and no restart.

Record aggregate and per-user decode throughput. A regression is useful for
the final performance decision but does not negate the crash fix if correctness
and stability pass.

## Gate 3 — continue the combined qualification

If Gates 1 and 2 pass, continue the already-planned single-boot ladder rather
than restarting:

1. bounded NVMe fill/eviction and persistence evidence;
2. 300k, 350k, and 475k needle tests;
3. 16 x 50k unique-prefix overflow/concurrency stress;
4. prefill and decode throughput cells; and
5. final liveness plus `RestartCount`, container ID, and error-signature audit.

## Static proof already completed

- The three changed source blobs are byte-identical between the PR base and
  image commit `3e731bc0`; this is an exact patch, not a speculative rebase.
- Regression coverage asserts auxiliary work is enqueued without an early
  consumer wait, then joined before routed experts.
- Output-lifetime coverage retains the consumer-stream `record_stream` rule.
- `git diff --check` passes.

GPU runtime proof is intentionally delegated to this one CN3 boot.
