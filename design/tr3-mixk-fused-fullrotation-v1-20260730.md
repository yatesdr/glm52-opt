# Mixed-K Trellis full-rotation fusion v1

Status: implemented and promoted. The operator matrix passed 21/21
bit-exact cells, the corrected full-model image passed the complete quality
and performance gates, and the self-contained package is published at
`ghcr.io/yatesdr/glm52-serve@sha256:a2b233f60329f22c7e541406b8972b3d8dc46c5c7d104d0aa3b4fee26374870e`.

## Problem

The 3.25-bpw checkpoint partitions every routed-expert layer into two native
Trellis tiers:

- K=3: 192 of 256 experts
- K=4: 64 of 256 experts

The current vLLM integration invokes the complete SparkInfer
`trellis_moe` plan once per tier. Each invocation independently:

1. maps and packs global top-k routes into tier-local routes;
2. rotates the route inputs using that tier's `suh` tables;
3. runs the persistent FC1, activation, and FC2 phases;
4. applies inverse H128, `svh`, router weights, and the FP32 top-k sum.

The two FP32 tier outputs are then added and converted to BF16. This is
numerically valid, but duplicates launch/scheduling/reduction work in every
one of the 75 routed layers.

## Existing mechanism that can be extended

SparkInfer already contains `W4A16FusedMoeHybridKernel` and
`build_w4a16_tier_local_map`. The descriptor is:

```text
global expert -> (tier << 8) | tier-local expert
```

That kernel runs two heterogeneous weight layouts through one persistent
grid, but its current contract explicitly excludes Trellis full-rotation
semantics. It was written for ordinary NVFP4+NF3 W4A16 decode. Calling it
unchanged for EXL3 would omit trained rotations and is not an acceptable
optimization.

## Proposed exact Trellis extension

Add a separate fail-closed full-rotation hybrid variant. Do not weaken the
existing hybrid contract.

The fused core receives both prepared Trellis tiers, the global route IDs,
and the immutable descriptor map. For each live route it:

1. resolves tier and local expert exactly once;
2. applies that tier/expert's gate and up `suh` tables to the raw input;
3. performs the identical FP16 H128 input rotations used by the single-tier
   implementation;
4. dispatches FC1 and FC2 to the existing tier-specific K=3/K=4 child
   kernels;
5. applies that tier/expert's compact intermediate rotation table.

The K3 and K4 expert ids are interleaved in global expert order. The fused
path therefore must not materialize a second global `[E,3I]` table. It uses
the same immutable descriptor to index the existing contiguous per-tier
rotation tables:

```text
(tier, local_expert) = descriptor[global_expert]
rotation_row = tier_rotations[tier][local_expert]
```

FC2 continues to emit one FP16 row per route. One paired hybrid top-k kernel
then applies inverse H128 and the tier-specific `svh` table.

To preserve the current mixed-K arithmetic ordering, the reduction keeps two
FP32 accumulators per output element:

```text
tier0 = sum(routes in original order where tier == 0)
tier1 = sum(routes in original order where tier == 1)
output = tier0 + tier1
```

This matches the current `tier0_output; tier1_output; tier0.add_(tier1)`
association, rather than interleaving both tiers into one accumulator.

## Compile and runtime contracts

- Decode-only v1: `1 <= live_m <= 32`, `topk == 8`, `int32` global routes.
- Exactly two non-empty Trellis tiers.
- Both tiers must be `trellis3_t256`, projection-major, with identical H/I,
  activation, tile geometry, device, input dtype, and rotation ABI.
- Trellis bit widths are explicit compile-key fields and must match packed
  tensor widths.
- The descriptor map must cover every global expert exactly once.
- No rank-local fallback. Selection is a static environment/config decision
  and therefore rank-invariant.
- Fresh compilation must report zero local-memory spill; otherwise the
  candidate is rejected before launch.
- Graph capture/replay must be bit-stable.
- The existing two-plan path remains the fallback when any eligibility
  condition is false.

## Memory plan

The fused route path requires one set of:

- route-major gate/up FP16 rotations;
- FC1 and activation buffers;
- per-route FP16 FC2 output;
- FC1/FC2 split-K scratch;
- one FP32 final output.

It does not need two simultaneously live plan arenas or the vLLM-level
second-tier FP32 result. The first implementation may reuse the existing
shared arena for safety; a separately proven arena compaction can follow.

The first full-model candidate violated this memory intent by retaining the
per-tier tables and also building a global `[E,3I]` copy (~57 MiB/GPU over
75 target layers). It served and benchmarked, but failed a cold 250k request
on a 36 MiB transient allocation. That candidate is rejected. The corrected
implementation reads the existing tier tables directly and its synthetic
production-geometry output remains bit-exact.

## Acceptance order

1. Static/source contracts and CPU descriptor/reduction-order proof.
2. CUDA synthetic GLM geometry:
   - sequential two-plan oracle versus fused mixed-K;
   - K3-only, K4-only, mixed, invalid-route, partial-batch cases;
   - eager repeatability and CUDA-graph replay;
   - exact output preferred; otherwise report maximum absolute/relative
     error and do not advance without review.
3. Operator timing at M=1/2/4/8/16/24/32.
4. One CN4 model boot:
   - short output smoke;
   - matched MTP0 C1;
   - matched MTP3 concurrency ladder;
   - KLD;
   - frozen cold needle rows.
5. Only after quality passes: prefill and memory work.

## Non-goals

- No selector, KV record, RoPE, DCP, transport, or checkpoint semantics
  change.
- No broad flag matrix.
- No attempt to use the existing non-rotation hybrid kernel as a quality
  shortcut.

## Final selector and promotion result

Direct operator timing placed the heterogeneous fused/serial crossover at
M=8. The promoted runtime therefore uses the fused path for M<=8 and the
exact serial tier path above it. The final Graph64 compose provides explicit
capture coverage through concurrency 64 without extending the fused
arithmetic beyond its proven winning range.

The promoted CN4 result exposed 500,224 GPU KV tokens and measured:

- MTP0 C1: 34.84 tok/s;
- MTP3 C1/C8/C16: 65.47/154.44/190.64 tok/s;
- cold 55k prefill: 1,325 tok/s;
- exact cold retrieval at 50k, 250k, 350k, and 475k;
- KLD over 2,047 evaluated positions: 0.0959706378925062.

The complete accepted and rejected candidate ledger is
`experiments/tr3-325-fused-route-results-20260730.md`.
