## Summary

Launch the two barrier-bearing fused W4A16 MoE grids cooperatively.

Both `W4A16FusedMoeKernel` and `W4A16FusedMoeHybridKernel` synchronize every
CTA between FC1, activation, output initialization and FC2. Their planners
already cap the grid to `SM count * blocks_per_sm`, but the launch itself did
not request whole-grid cooperative admission.

This patch adds `cooperative=True` to those two launch sites. It does not
change the standalone W4A16 GEMM, launch geometry, tile selection,
quantization, routing or output math.

## Why

A software whole-grid barrier requires every participating CTA to be
resident. A bounded grid alone does not guarantee that: unrelated work on
another stream can occupy an SM after only part of the fused grid has been
admitted, leaving resident CTAs waiting for peers that cannot be scheduled.

SparkInfer already uses cooperative admission for the fused-micro W4A16 decode
path. The route-packed fallback entered above that micro range and the
two-tier hybrid path use the same barrier pattern but were missing the launch
contract.

The production workload that motivated the audit runs shared-expert and other
auxiliary-stream work beside MTP decode graph capture. This PR is deliberately
not presented as the fix for that workload's eventual CUDA illegal-address
failure: launch-blocking subsequently localized that separate failure to an
MLA query-absorption BMM, which has its own vLLM fix. Cooperative admission is
an independent correctness requirement visible directly in these fused
kernels.

## Validation

Completed:

- forward-port onto current SparkInfer `master`;
- `python -m py_compile` for both changed files;
- `git diff --check`;
- exact patched W4A16 source was byte-verified in the GLM-5.2 v20 candidate
  that completed profiling and production decode capture at sizes 16 through
  1, with 624/624 diagnostic boundaries passing and no CUDA/cuBLAS/OOM/Xid
  failure.

The engine result is combined-stack evidence, not an isolated A/B proof of
this change.

This PR adds a focused SM120 GPU regression:

```bash
pytest -q tests/moe/test_fused_moe.py \
  -k test_run_w4a16_m9_graph_replay_with_prequeued_aux_work
```

It captures the first route-packed size above the fused-micro range using the
GLM shard geometry, prequeues large BF16 GEMMs on an auxiliary stream, replays
the W4A16 graph concurrently, and checks finite, nonzero, stable output.

The test cannot run on the contributor's macOS host because CUTLASS DSL and
SM120 CUDA are unavailable. This PR remains draft until the focused GPU test
and the ongoing decode-throughput qualification are recorded.

## Tradeoff

Cooperative admission can alter scheduling latency when unrelated stream work
is queued. It does not serialize the streams globally or disable
shared-expert overlap; it only requires the complete bounded W4A16 grid to be
admitted as a unit, which is the execution contract required by its software
barriers.

## AI assistance

OpenAI Codex assisted with source tracing, forward-porting, regression-test
design and PR drafting. The submitted code and runtime evidence are being
reviewed by the human contributor and target-system operator before the draft
is marked ready.
