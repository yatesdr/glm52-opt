# GLM-5.2 TR3 3.25-bpw fused routed-MoE experiment

Date: 2026-07-30  
Host: CN4 only  
Status: **promotion gates passed** for the GPU-only C1-C16 balanced profile;
exact image, compose, KLD, cold deep retrieval, and ordered stress are pinned

## Frozen baseline

- Image:
  `ghcr.io/yatesdr/glm52-serve@sha256:b53d5d551937a0580848101dfc5df9b7fb2638419cfa6da0fa35d0a2d339fe2e`
- Model:
  `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@e2b03576cd103e6ad322a1e091e5d0e2d0529073`
- TP4/DCP4/MTP3, graph 32, MNBT 3072, maximum model length 480,000,
  GMU 0.9688
- Dynamic NVFP4 MLA KV, FP8 RoPE, exact selector, PXB + `i8_ring`
- GPU KV pool: 532,992 tokens
- MTP3 C1: 52.77 tok/s
- MTP3 C32: 120.04 tok/s
- Cold 55k prefill: 1,320 tok/s server-side (1,314 tok/s wall-derived)
- KLD: 0.0959706378925062 over 2,047 evaluated positions
- Frozen cold 350k needle: exact

Rollback:

```bash
cd /home/derek/glm52-tr3-325-public-20260729
MODEL_PATH=/home/derek/models/GLM-5.2-EXL3-TR3-3.25bpw \
CACHE_PATH=/home/derek/glm52-tr3-325-public-20260729/cache-v4-validation \
docker compose -f compose.yaml up -d --force-recreate
```

## Profiler localization

Evidence:
`harness/cn4-evidence-archive/20260730/tr3-325-mixk-mtp0-profile-v1/`

- Rank-0 decode graph envelope: 31.253784 ms.
- All 75 routed layers execute two complete Trellis pipelines, one for the
  checkpoint's 192 K3 experts and one for its 64 K4 experts.
- 150 Trellis kernel calls total 11.362 ms:
  - K3: 75 calls, 5.621536 ms, 74.95 us mean.
  - K4: 75 calls, 5.740535 ms, 76.54 us mean.
- Route support is likewise duplicated: two route packs and two top-k sums per
  routed layer.

## Candidate

The candidate performs one global route pack and one heterogeneous K3/K4
Trellis launch, followed by a two-accumulator FP32 top-k reduction that
preserves the production association:

```text
result = sum(K3 routes in original order) + sum(K4 routes in original order)
```

It is selected only for `1 <= M <= 8`. Larger decode shapes and every prefill
shape retain the prior serial two-tier implementation.

Pinned source outputs:

| File | SHA-256 |
|---|---|
| `w4a16/kernel.py` | `8fb83a73be4a3ea7ad0b2093cad674130dee84183b01c8d8187249b87be0feae` |
| `trellis_moe/__init__.py` | `dc977562db1dd394cef3df8163daf0a0eabe8057452f7659d39e1e03b9155427` |
| `trellis_moe/_impl.py` | `17a58dc97f29ba4792a63af26e0dee30aac5e39d32ee562c4ee45c3b303c0ca9` |
| `trellis_moe/api.py` | `33a910eb86648ef333d374bfd283cbd06fbcb4e1777b910346a949a4d331c929` |
| mixed-K v6 patch | `f337457c29063f0467516a69ded808196e029c65f3359493226dddd4e3f422a1` |
| patched `vllm/.../exl3.py` | `ddcb1ae7f21c72119a0c24c376ed3f9d2f09e49a49f272af25e6cfcf4fc98ec5` |

Candidate 3 local manifest:
`sha256:23a28752d0feab5faf5de5a99d0ac9767a4fc504a677b7c8da47bd039139af10`

## Pre-boot gates

- CPU descriptor, tier-local rotation lookup, and reduction-order proof:
  4/4 pass.
- Public lazy API import: pass.
- Exact production geometry: H=6144, I=512, top-k=8, K3=192, K4=64.
- Shapes M=1/2/4/8/16/24/32, route modes K3-only/K4-only/mixed:
  21/21 bit-exact; route outputs and final outputs have maximum absolute
  difference 0.
- CUDA graph replay: bit-exact at every measured shape.
- Compiled kernel: 139 registers/thread, 43,136 bytes shared memory,
  0 local-memory bytes, 1 block/SM.

Two-hundred-replay CUDA-graph timings:

| M | Serial ms | Fused ms | Speedup | Saved ms/layer |
|---:|---:|---:|---:|---:|
| 1 | 0.169966 | 0.096436 | 1.7625x | 0.073530 |
| 2 | 0.176910 | 0.103677 | 1.7064x | 0.073233 |
| 4 | 0.205238 | 0.121808 | 1.6849x | 0.083430 |
| 8 | 0.346070 | 0.263955 | 1.3111x | 0.082115 |
| 16 | 0.547996 | 0.583565 | 0.9390x | -0.035569 |
| 24 | 0.940834 | 0.920737 | 1.0218x | 0.020097 |
| 32 | 1.229655 | 1.212949 | 1.0138x | 0.016706 |

The production selector is therefore deliberately capped at M=8.

## Integration-gate finding

The first packaged no-model import caught that the new implementation existed
in `_impl.py` and `api.py` but had not been added to the package's lazy
`META.entry_points`. That would have failed the model boot when
`_load_sparkinfer_trellis()` accessed `plan_hybrid`. The public exports and
type-checking imports were added, then the public-surface exact-geometry gate
was rerun and passed 21/21.

The first full-model boot then exposed a second, distinct contract boundary:
the 75 target transformer layers use the validated `((K3,192),(K4,64))`
signature, but the MTP draft layer is `((K3,256),)`. Candidate 1 stopped
fail-closed before serving because its initial guard required every mixed-K
runtime to have the fused two-tier signature. Candidate 2 now selects fusion
only for the exact proven K3/K4 geometry; the K3-only draft layer retains the
unchanged exact serial path. This changes neither weights nor arithmetic for
the draft layer.

Candidate 2 then reached serving and produced two useful positive signals:

- GPU KV pool 544,256 tokens versus the 532,992-token baseline (+11,264,
  +2.1%);
- warm MTP3 C1 55.52 and 65.75 tok/s; C32 123.87 tok/s versus 120.04
  (+3.2%). MTP acceptance varied between prompts, so the C1 values are not an
  isolated kernel-cost measurement.

It was nevertheless rejected. The cold 50k retrieval control passed exactly,
but the following 250k cell failed with a 36 MiB transient CUDA allocation
when GPU 3 had only 38.69 MiB free. Candidate 2 had introduced a duplicate
global `[E,3I]` FP16 rotation table in every target layer (~57 MiB/GPU total).
The failed request and subsequent collective wait are archived at:

`cn4:/home/derek/glm52-tr3-325-fused-m8-20260730/evidence/candidate2-eaaf31a4/`

Checkpoint metadata proves the 192 K3 and 64 K4 expert sets are interleaved,
so they cannot be represented as two contiguous views of a global table.
Candidate 3 therefore removes the global copy: the fused activation epilogue
uses the existing immutable global-to-tier-local descriptor to read the
original per-tier rotation tables directly. Its production-geometry operator
gate again passed 21/21 bit-exact cases, graph replay, and the performance
table above.

Candidate 3 then completed the full model boot:

- fused runtime initialized on all four ranks for the target signature and
  retained serial fallback for the K3-only draft signature;
- CUDA-graph capture completed and the API became healthy;
- GPU KV pool: 551,680 tokens (+18,688 / +3.5% versus baseline), clearing the
  secondary 550k target;
- cold 50k retrieval: exact `738216`, `cached_tokens=0`.

The following cold 250k request failed at 245,506 context before completion.
This was not a fused-MoE allocation. SparkInfer r9 selected its adaptive
two-level exact indexer fold and allocated two transient candidate slabs with
bare `torch.empty` calls. The first 72 MiB slab succeeded; the second
(`fold_indices`, `sparkinfer/attention/nsa_indexer/paged.py:959`) failed with
56.69 MiB physically free. The process was therefore rejected at the memory
safety gate even though the fused implementation itself loaded and passed its
short-context proof.

The repository already documents this allocation-class defect:
`design/b12x-indexer-two-level-fold-scratch-cure.md`. The r9 adaptive planner
bounds the transient to 256 MiB but does not reserve it during vLLM memory
profiling, so a bounded request is still not a fit proof on a saturated GPU.

Candidate 4 changes only the fold schedule:

```text
SPARKINFER_INDEXER_TWO_LEVEL_FOLD=0
```

This selects SparkInfer's exact, pre-reserved streaming-carry path. It does
not change scorer arithmetic, selector output, the fused routed-MoE kernel, or
the KV layout. The purpose of the single rerun is to prove the fused candidate
at long context without the independently demonstrated r9 late allocation.
Any deep-prefill cost will be measured after the 250k safety gate passes.

Candidate 4 completed that decisive gate:

- GPU KV pool: 552,192 tokens;
- cold 250k cell: exact `738216`, `cached_tokens=0`, 245,505 context tokens;
- completion: 65 tokens, clean finalization;
- wall time: 302.94 seconds;
- no OOM or engine error.

The same image and memory posture therefore fails with adaptive two-level
folding and passes with exact streaming carry. This causally isolates the
late-allocation failure from the fused routed-MoE change. The 302.94-second
wall time also shows why `SPARKINFER_INDEXER_TWO_LEVEL_FOLD=0` is a safety
control rather than the final optimized posture: the permanent path must make
the parallel fold's workspace part of the profiled/reserved peak, or otherwise
remove its late transient, without giving up the recovered deep-prefill speed.

Matched candidate-4 MTP3 decode sweep (256 output tokens/request, deterministic
nonce family, zero-context):

| Concurrency | Aggregate tok/s | MTP acceptance | Finalized |
|---:|---:|---:|:---:|
| 1 | 59.92 | 0.5649 | yes |
| 2 | 87.50 | 0.6007 | yes |
| 4 | 109.56 | 0.5756 | yes |
| 8 | 146.15 | 0.5254 | yes |
| 16 | 64.14 | 0.5638 | yes |
| 24 | 90.48 | 0.5744 | yes |
| 32 | 116.71 | 0.5661 | yes |

The fused selector applies through M=8 and those cells are healthy. M=16 and
above deliberately retain the serial implementation; their cliff is therefore
a separate graph/scheduling target rather than evidence against this kernel.
The C32 point is below the 120.04 baseline by 2.8%, while MTP acceptance differs,
so it is not an isolated kernel regression.

The 552,192-token pool nevertheless failed the required post-decode stress
sequence. Immediately after the C1-through-C32 sweep, a cold 55k prefill
request reached the shared-expert down projection and failed a 36 MiB CUDA
allocation on GPU 1 with only 33.12 MiB physically free. The allocator reported
660.23 MiB reserved-but-unallocated. This allocation is independent of both
the fused routed-MoE kernel and the two-level indexer fold.

Therefore:

- the fused code remains valid;
- exact streaming carry fixes the 250k indexer-fold crash;
- the 552,192-token / GMU 0.9688 memory posture is not production-safe after
  maximum-concurrency decode activity;
- the secondary 550k target is rejected unless another persistent allocation
  is reclaimed. The next boot must explicitly reduce the KV pool/headroom and
  repeat the post-decode cold-prefill stress, rather than treating a clean boot
  as a fit proof.

Candidate 5 is the final evidence-based memory posture:

```text
GPU_MEMORY_UTILIZATION=0.9675
SPARKINFER_INDEXER_TWO_LEVEL_FOLD=auto
```

It restores the fast r9 fold but converts the fusion's persistent-memory
savings into transient headroom. The expected KV pool is roughly 535–537k:
above the 532,992 baseline, but below the disproven 550k target. Acceptance is
fail-closed and ordered to reproduce both failures: maximum-concurrency decode,
then cold 55k prefill, then cold 250k/350k retrieval.

Candidate 5 loaded with a 535,040-token GPU KV pool and passed the first three
ordered stress gates:

- MTP3 C32: 112.22 tok/s, acceptance 0.5706, 8,192 generated tokens;
- immediate cold 55k: 1,308 server / 1,301 wall tok/s, `cached_tokens=0`;
- immediate cold 250k: exact `738216`, `cached_tokens=0`, 245,506 context
  tokens, 297.72 seconds.

The subsequent cold 350k request failed in the same adaptive-fold allocation
site as candidate 3:

```text
sparkinfer/attention/nsa_indexer/paged.py:959
fold_indices = torch.empty(...)
requested: 120.00 MiB
physically free: 116.69 MiB
allocator reserved-but-unallocated: 498.68 MiB
```

This is a useful fail-closed result, not a near-pass. Reducing GMU until one
particular late allocation happens to fit would leave the request-time
workspace outside vLLM's profiled peak and remain sensitive to request order
and fragmentation. Candidate 5 is therefore rejected despite passing the
preceding stress sequence.

The final conclusion is now sharper:

- the M<=8 fused routed-MoE implementation loads, graph-captures, serves, and
  passes its operator and live decode gates;
- its persistent-memory saving is real (+2,048 KV tokens at the safe-minded
  candidate-5 posture);
- the stock r9 adaptive exact-fold path has an independent unprofiled
  request-time workspace defect at deep context;
- no production promotion can use adaptive folding until that workspace is
  reserved/profiled, reused from an existing profiled allocation, or made to
  fall back before allocation based on physically guaranteed headroom.

## Model gates

### Candidate 6: profile-accounted adaptive fold

Candidate 6 replaces the stock request-time fold allocations with a fixed
rank-invariant reservation in SparkInfer's existing paged-attention scratch.
The reservation is included in vLLM's memory-profile peak and the request path
borrows typed views from it. `auto` falls back to exact streaming carry when a
request exceeds the reserved candidate-row capacity; `force` fails closed.

Pre-boot gates:

- 11 focused CPU planner/reservation tests passed (one CUDA-only skip);
- reserved parallel-fold output matched the reference exactly on GPU;
- zero-budget streaming fallback matched the reference exactly on GPU;
- all modified deployed-source bytes were pinned and verified by the image
  build.

Live startup evidence on image manifest
`sha256:a2b233f60329f22c7e541406b8972b3d8dc46c5c7d104d0aa3b4fee26374870e`:

- fused serving runtime selected on all four ranks:
  `K3=192, K4=64, fused_max_m=8`;
- MTP draft correctly used the exact serial path for its homogeneous K3 tier;
- the profiler included 233.25 MiB of sparse-DCP transient scratch;
- GPU KV pool: 484,608 tokens;
- C32 stress: 32/32 requests, 8,192/8,192 generated tokens, no errors,
  108.82 tok/s, 0.5713 MTP acceptance.

The immediate cold 55k request then failed in the shared-expert down
projection while allocating 36.00 MiB on GPU 1 with 35.12 MiB physically
free. This is distinct from the removed fold allocation. The request was the
first real prefill in a fresh compile namespace and the JIT monitor recorded
post-engine-start disk-cache misses for both `UnifiedPrefillMGKernel` and the
paged sparse-indexer kernels. At failure the allocator held 663.75 MiB
reserved-but-unallocated.

Candidate 6 therefore establishes both sides of the current boundary:

- the fused routed-MoE and profile-accounted fold mechanisms load and pass
  their direct/live gates;
- the clean-cache startup contract still misses first-prefill kernel warmup
  and/or allocator headroom, so it is not production-safe yet.

The next confirmation must distinguish transient first-JIT fragmentation from
steady-state residency using the now-populated disk cache, then close the
clean-cache contract by prewarming the missing signatures or reserving a
measured transient margin. Merely retrying the request is not a promotion
gate.

### Candidate 7: warmed-cache memory discriminator

Candidate 7 restarted the exact candidate-6 image, configuration, GMU, and
cache volume. The only changed state was that the first boot had populated
the compile cache.

- startup fell from 474.05 seconds to 78.85 seconds;
- the KV pool remained exactly 484,608 tokens;
- an immediate cold 55k request passed at 1,187 server tok/s
  (`cached_tokens=0`);
- C32 then passed 32/32 at 117.91 tok/s with 0.5616 MTP acceptance;
- a second cold 55k request immediately after C32 failed on the same 36 MiB
  shared-expert down-projection output, this time with 36.69 MiB physically
  free and 615.63 MiB reserved-but-unallocated.

This rules out first-JIT compilation as the sufficient cause. The shared
expert implementation uses an auxiliary CUDA stream for decode-sized batches
up to `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` (default 256) and records
the output onto the consumer stream. The failure only appears after C32 has
exercised that path; 55k prefill itself is above the threshold and runs
shared experts on the main stream. The evidence is consistent with
cross-stream allocator fragmentation/residency left by decode, which prevents
the subsequent main-stream 36 MiB output from finding a usable block despite
hundreds of MiB being reserved.

The next narrow control sets the existing threshold to 8. This preserves
shared-expert overlap for the fused M<=8 latency-sensitive cells but keeps
C16/C24/C32 on the main stream, where upstream already expects overlap to
become less useful. Acceptance remains C32 followed immediately by cold 55k;
the change is rejected if either stability or throughput regresses materially.

### Candidate 8: shared-expert overlap threshold 8

Candidate 8 set only:

```text
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=8
```

It produced two useful results:

- GPU KV pool increased from 484,608 to 502,272 tokens because the C32
  auxiliary-stream peak left the memory profile;
- C32 passed 32/32 at 119.26 tok/s and 0.5725 MTP acceptance, within 0.7% of
  the 120.04 baseline.

The immediate cold 55k request still failed on the same 36 MiB main-stream
shared-expert output. GPU 1 had 31.12 MiB physically free and the allocator
reported 610.38 MiB reserved-but-unallocated. This rejects auxiliary-stream
overlap as the sufficient cause. Decode creates a fragmented small-allocation
steady state even when C32 itself uses the main stream.

The next safe control converts the candidate's 22,272 KV tokens above the
480k contract into roughly 160–175 MiB of physical margin by lowering GMU to
0.9671. This leaves max model length unchanged and initially avoids introducing
`expandable_segments`; that allocator mode is incompatible with the
`OffloadingConnector` used by some other deployment profiles and must only be
considered after proving the active qualification compose is GPU-only.

### Candidate 9: 480k floor plus physical margin

Candidate 9 kept the threshold-8 shared-expert policy and lowered only GMU:

```text
GPU_MEMORY_UTILIZATION=0.9671
```

The server exposed 480,768 GPU KV tokens, remained above the declared
480,000-token model contract, and passed the short ordered stress pair:

- C32: 118.82 tok/s, 32/32 finalized, 8,192 generated tokens, 0.5734 MTP
  acceptance;
- immediate cold 55k: 1,318 server / 1,311 wall tok/s, 54,211 prompt tokens,
  `cached_tokens=0`.

The subsequent cold 250k request failed before producing a model answer, so
this is not a retrieval-quality verdict. GPU 1 failed the same 36.00 MiB
shared-expert down-projection output allocation with only 9.12 MiB physically
free while PyTorch held 804.39 MiB reserved-but-unallocated. The engine
restarted once after the worker timeout.

Lowering GMU enough to preserve the short pair is therefore not sufficient
for a sequential deep-prefill production history. The failure is specifically
allocator contiguity: the requested allocation is much smaller than the
aggregate allocator-owned free space but cannot be satisfied from a
contiguous reusable block.

### Candidate 10: expandable allocator, GPU-only validation posture

The active CN4 qualification compose has no `OffloadingConnector`: it mounts
only the model and compile-cache volumes and has no KV-transfer/offload
configuration. Candidate 10 therefore adds the allocator mode PyTorch
recommends for this exact fragmentation signature:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This is a single-variable retry of candidate 9. It is intentionally scoped to
the GPU-only validation posture. It must not be copied into deployments using
vLLM's `OffloadingConnector`, which explicitly rejects expandable allocator
segments.

Candidate 10 booted healthy with zero restarts and a 480,512-token KV pool.
The exact candidate-9 stress sequence then passed:

- C32: 117.94 tok/s, 32/32 finalized, 8,192 generated tokens, 0.5628 MTP
  acceptance;
- immediate cold 55k: 1,321 server / 1,314 wall tok/s, 54,210 prompt tokens,
  `cached_tokens=0`;
- immediate cold 250k: exact `738216`, 245,504 context tokens,
  `cached_tokens=0`, 301.63 seconds.
- cold 350k: exact `738216`, 343,736 context tokens, `cached_tokens=0`,
  464.00 seconds;
- cold 475k: exact `738216`, 466,495 context tokens, `cached_tokens=0`,
  675.22 seconds.

Candidate 9 failed the same 250k arm after the same C32/cold55 history, while
candidate 10 changed only the allocator mode. This is causal evidence that
allocator fragmentation—not the fused routed-MoE kernel, fold workspace, or
retrieval path—caused the remaining 36 MiB transient allocation failure in
the GPU-only qualification profile. The full ordered one-boot history passed
with zero restarts:

```text
C32 -> cold55 -> cold250 -> cold350 -> cold475
```

The 350k and 475k arms also validate the exact streaming-carry fallback above
the reserved two-level fold row budget. The fused route, profile-accounted
fold scratch, and allocator policy are therefore stable through the declared
480k context contract in this GPU-only profile.

The subsequent MTP3 decode matrix ran on the same process after all deep
requests:

| concurrency | aggregate tok/s | finalized | MTP acceptance |
|---:|---:|---:|---:|
| 1 | **63.25** | 1/1 | 0.6111 |
| 2 | 83.84 | 2/2 | 0.5727 |
| 4 | 108.54 | 4/4 | 0.5504 |
| 8 | **151.54** | 8/8 | 0.5766 |
| 16 | 65.46 | 16/16 | 0.5812 |
| 24 | 91.07 | 24/24 | 0.5937 |
| 32 | 113.13 | 32/32 | 0.5658 |

C1 improved 19.9% over the pinned 52.77 tok/s baseline. C1 through C8 use
the fused mixed-K route. C16 and above use the exact serial tier path by
design because the direct operator benchmark measured the fusion crossover
at M=8; their clean but lower rows isolate the next performance opportunity
to the M16+ serial/cudagraph path rather than the new fused kernel.

The cold prefill matrix on the same post-stress process was:

| target | actual prompt tokens | server tok/s | cached tokens |
|---:|---:|---:|---:|
| 8k | 7,919 | 1,351 | 0 |
| 55k | 54,210 | **1,321** | 0 |
| 64k | 63,086 | 1,305 | 0 |
| 128k | 126,203 | 1,240 | 0 |
| 250k | 246,546 | 833 | 0 |

The 55k result clears the pinned 1,320 tok/s promotion bar while preserving
the exact deep-context gates. A final cold 50k needle row also passed:
49,100 context tokens, `cached_tokens=0`, exact `738216`, 38.07 seconds.

### Candidate 11: MTP0 kernel-cost control

The exact candidate image was booted without speculative decoding and measured
with three matched 1,024-token C1 samples:

| Run | MTP0 C1 tok/s |
|---:|---:|
| 1 | 34.84 |
| 2 | 34.87 |
| 3 | 34.81 |
| **Mean** | **34.84** |

All three requests finalized cleanly. The population standard deviation is
0.02 tok/s (sample standard deviation 0.03), so this is a stable measurement
of the non-speculative route rather than an MTP-acceptance artifact.

Evidence:
`harness/cn4-evidence-archive/20260730/tr3-325-fused-final/candidate11-mtp0-a2b233f6/`

### Candidate 12: C16 route localization and Graph-64 proof

The operator probe established the exact production mapping:

```text
MTP3 verification rows = concurrency * 4
C16 => M=64
```

At M=64 the existing exact serial block-M=8 Trellis plan measured 0.8638 ms,
versus 1.8356 ms for the prefill-oriented block-M=64 plan, with bit-identical
output. The C16 cliff was therefore not evidence that the fused M<=8 kernel
needed to expand; it was a missing decode graph/route envelope.

Candidate 12 extended the exact serial decode window and CUDA graph to M=64.
The matched live matrix was:

| Concurrency | Candidate 12 tok/s |
|---:|---:|
| 1 | 66.20 |
| 2 | 90.85 |
| 4 | 116.36 |
| 8 | 146.81 |
| 16 | **190.34** |
| 24 | 94.46 |
| 32 | 121.89 |

C16 improved from candidate 10's 65.46 to 190.34 tok/s. C24 and C32 are
sanity rows outside the chosen Graph-64 C1-C16 envelope.

Candidate 12 was not promotable: after C32 it failed a cold 250k request while
allocating 48 MiB with only 23 MiB physically free. This proved the route fix,
but rejected its all-shapes graph/memory posture.

Evidence:
`harness/cn4-evidence-archive/20260730/tr3-325-fused-final/candidate12-c16-window-graph64-a2b233f6/`

### Candidate 13: balanced production profile

Candidate 13 retained the proven M=64 serial decode route but captured only
the five production shapes needed for MTP3 C1/C2/C4/C8/C16:

```text
4, 8, 16, 32, 64
```

It also selected exact streaming carry instead of reserving the parallel
two-level-fold slab and explicitly exposed 1,954 KV blocks. The resulting
GPU KV pool is **500,224 tokens**. This is above the 480k declared context
contract while preserving request-time headroom. An attempted higher-memory
posture is not accepted: the tightest observed deep-prefill headroom in the
passing profile was approximately 102 MiB, and candidate 12 had already
demonstrated a 48 MiB terminal allocation failure.

The decisive ordered gate passed:

```text
C16 192.03 tok/s -> C32 122.51 tok/s -> cold 250k exact
```

The complete promotion matrix on the same process was:

| Concurrency | Aggregate tok/s | MTP acceptance | Finalized |
|---:|---:|---:|:---:|
| 1 | **65.47** | 0.6513 | 1/1 |
| 2 | 84.30 | 0.5714 | 2/2 |
| 4 | 110.79 | 0.5447 | 4/4 |
| 8 | **154.44** | 0.5790 | 8/8 |
| 16 | **190.64** | 0.5868 | 16/16 |
| 24 | 94.57 | 0.5607 | 24/24 |
| 32 | 123.06 | 0.5661 | 32/32 |

Cold prefill, with prefix-cache deltas proving every sample was cold:

| Target | Actual tokens | Server tok/s | Cached tokens |
|---:|---:|---:|---:|
| 8k | 7,916 | 1,379 | 0 |
| 55k | 54,208 | 1,325 | 0 |
| 64k | 63,084 | 1,308 | 0 |
| 128k | 126,204 | 1,240 | 0 |
| 250k | 246,547 | 833 | 0 |

Frozen cold retrieval:

| Target | Actual context | Result | Cached tokens | Wall time |
|---:|---:|:---:|---:|---:|
| 50k | 49,099 | exact `738216` | 0 | 38.00 s |
| 250k | 245,505 | exact `738216` | 0 | 295.71 s |
| 350k | 343,735 | exact `738216` | 0 | 462.94 s |
| 475k | 466,494 | exact `738216` | 0 | 674.11 s |

The final maximum-load sequence also passed:

```text
C32 115.45 tok/s, 32/32 finalized
-> immediate cold 350k, exact 738216, cached_tokens=0, 463.54 s
```

No worker restarted and the final container error scan was empty.

Prescribed MTP0/eager KLD smoke:

| Metric | Result |
|---|---:|
| Mean KL(reference || candidate) | **0.0959706378925062** |
| Evaluated positions | 2,047 |
| Promotion ceiling | 0.0959706378925062 |
| Result | **PASS** |

The runner confirmed the exact image digest, model revision, DCP1/MTP0,
eager execution, exact selector, dynamic per-token NVFP4 MLA KV, and FP8 RoPE.
The KLD result is exactly equal to the prior accepted TR3-3.25 baseline.

Final evidence:

- `harness/cn4-evidence-archive/20260730/tr3-325-fused-final/candidate13-balanced-500224-a2b233f6/`
- `harness/cn4-evidence-archive/20260730/tr3-325-fused-final/candidate13-mtp3-promotion-500224-a2b233f6/`
- `harness/cn4-evidence-archive/20260730/tr3-325-fused-final/candidate13-kld-mtp0-a2b233f6/`

Key KLD artifact pins:

```text
f6aeba2963f2ee43610e23710b4d5a311f054aa7f8e9ca41ffc013e64404cc0d  run.log
937a2c87bab1abe02420867e085b630e8d40e9812c782ba1e5d765b44dcd1b20  summary.json
```

## Final promoted composition

- Image:
  `ghcr.io/yatesdr/glm52-serve@sha256:a2b233f60329f22c7e541406b8972b3d8dc46c5c7d104d0aa3b4fee26374870e`
- Model:
  `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@e2b03576cd103e6ad322a1e091e5d0e2d0529073`
- Compose:
  `compose/glm52-v20-r9-tr3-325-c16-balanced-shareable-20260730.yaml`
- TP4/DCP4/MTP3, exact selector, dynamic NVFP4 MLA KV, FP8 RoPE,
  PXB + `i8_ring`, MNBT 3072, max length 480k.
- Fused heterogeneous mixed-K routed-MoE through M=8.
- Exact serial block-M=8 decode route and graph coverage through M=64.
- Exact streaming-carry indexer folding.
- Expandable CUDA allocator, GPU-only cache posture.
- 500,224-token GPU KV pool.

This is a configuration-level promotion on the already pinned image. No new
source change or PR is required for candidate 13.

### Image pedigree

The immutable image labels were re-read from the deployed digest:

- Base image:
  `voipmonitor/vllm:gilded-gnosis-v20-r9`,
  digest
  `sha256:8246024490670e43af6ccdc3df9c6dd0a084119f4507b7ac35a86f5a1c6c33c3`.
- vLLM integration:
  `4247d6765398fd42de3c108a8d991b2634fe88d1`; locked review heads
  145, 172, 175, 179, 184, 185, 190, and 189.
- SparkInfer integration:
  `f9be2724953a5b412d19c20482aeb0a64fbd5d2a`; locked review heads
  81, 49, 86, and 87.
- EXL3 runtime:
  `brandonmmusic-max/exllamav3@704aefd743b390af4bd0fb429d1906f9b964c7d8`.
- Overlay additions:
  mixed-K v6 route integration, exact M<=8 heterogeneous fused routed-MoE,
  profile-accounted SparkInfer fold scratch, allocation-free production PCIe
  DMA fusion, and the bounded NVMe-offload vLLM #165 implementation. The final
  GPU-only compose does not enable the offload connector.
- Record ABI:
  `nvfp4_ds_mla:fp8-rope-368:dynamic-token-v1`.

## Final comparison

| Metric | Frozen unfused baseline | Fused M<=8 / Graph-32 (candidate 10) | **C1-C16 balanced (candidate 13)** |
|---|---:|---:|---:|
| GPU KV tokens | 532,992 | 480,512 | **500,224** |
| MTP0 C1 tok/s | — | 34.84 | **34.84** (same exact image) |
| MTP3 C1 tok/s | 52.77 | 63.25 | **65.47** |
| MTP3 C2 tok/s | — | 83.84 | **84.30** |
| MTP3 C4 tok/s | — | 108.54 | **110.79** |
| MTP3 C8 tok/s | — | 151.54 | **154.44** |
| MTP3 C16 tok/s | — | 65.46 | **190.64** |
| MTP3 C24 tok/s | — | 91.07 | **94.57** |
| MTP3 C32 tok/s | 120.04 | 113.13 | **123.06** |
| Cold prefill 8k tok/s | — | 1,351 | **1,379** |
| Cold prefill 55k tok/s | 1,320 | 1,321 | **1,325** |
| Cold prefill 64k tok/s | — | 1,305 | **1,308** |
| Cold prefill 128k tok/s | — | 1,240 | **1,240** |
| Cold prefill 250k tok/s | — | 833 | **833** |
| Cold needles 50/250/350/475k | 350k exact | all exact | **all exact** |
| KLD, 2,047 positions | 0.0959706379 | 0.0959706379 | **0.0959706379** |
| Ordered C32→cold350 stress | not recorded | not run in this order | **PASS** |

Relative to the frozen unfused baseline, candidate 13 improves MTP3 C1 by
24.1%, C32 by 2.5%, and cold-55k prefill by 0.4%. Relative to the prior fused
Graph-32 posture, it removes the C16 cliff (+191.2%) and recovers 19,712 KV
tokens while retaining the same quality result.

## Shareable configuration controls

These are the measured controls exposed in the final compose. Values in the
“validated setting” column form one coherent profile; changing one invalidates
the aggregate promotion result until the ordered stress gate is repeated.

| Control | Validated setting | Measured purpose/effect |
|---|---|---|
| `VLLM_EXL3_MIXK_FUSED_MAX_M` | `8` | Uses the bit-exact heterogeneous fused kernel only where its direct operator timing wins. Fusion regressed at M=16, so extending this is not an optimization. |
| `VLLM_EXL3_TRELLIS_MAX_M` | `64` | Keeps the fast exact serial block-M=8 decode plan through MTP3 C16 (`M=64`). |
| `GRAPH` | `64` | Captures the C16 decode shape. This removed the 65.46 tok/s C16 cliff and reached 190.64 tok/s. |
| `COMPILATION_CONFIG_JSON` capture sizes | `[4,8,16,32,64]` | Captures only MTP3 C1/C2/C4/C8/C16. Avoids the memory cost of unused intermediate graphs. |
| `SPARKINFER_INDEXER_TWO_LEVEL_FOLD` | `0` | Selects exact streaming carry and avoids the parallel-fold reservation/late-allocation failure. All 50k–475k needles passed. |
| `NUM_GPU_BLOCKS_OVERRIDE` | `1954` | Exposes exactly 500,224 KV tokens. Higher tested memory postures failed ordered decode→deep-prefill stress. |
| `GPU_MEMORY_UTILIZATION` | `0.9671` | Profiled baseline used with the explicit safe KV-block count. Do not raise independently. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Causally fixed post-decode deep-prefill fragmentation in this GPU-only posture. Incompatible with vLLM `OffloadingConnector`. |
| `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` | `8` | Preserves low-M overlap while keeping C16+ shared-expert work on the main stream. |
| `F8_DMA` | `i8_ring` | Measured CN4 PCIe transport. The complete KLD and deep-needle suite passed in this posture. |
| `NCCL_P2P_LEVEL` | `PXB` | Matches CN4 topology. Other systems should select this from their actual `nvidia-smi topo -m` result rather than copying blindly. |
| `DCP_TOPK_OWNER_MERGE` | `0` | Retains the measured PCIe prefill posture; exact retrieval passes with it disabled. |
| `MAX_BATCHED_TOKENS` | `3072` | Preserves the validated prefill traffic/compute balance. |
| `KV_CACHE_DTYPE` / `KV_FP8_ROPE` | `nvfp4_ds_mla` / `1` | Capacity-oriented KV record used by all promotion gates. |
| `VLLM_NVFP4_MLA_DYNAMIC_SCALE` | `1` | Enables the self-contained dynamic per-token outer scale; KLD remained 0.0959706378925062. |

## Launch and rollback

Launch the promoted profile:

```bash
MODEL_PATH=/path/to/GLM-5.2-EXL3-TR3-3.25bpw \
CACHE_PATH=/path/to/persistent/compile-cache \
docker compose \
  -f compose/glm52-v20-r9-tr3-325-c16-balanced-shareable-20260730.yaml \
  up -d
```

The performance-safe rollback is the frozen unfused profile at the top of this
report. On CN4 its existing command is:

```bash
cd /home/derek/glm52-tr3-325-public-20260729
MODEL_PATH=/home/derek/models/GLM-5.2-EXL3-TR3-3.25bpw \
CACHE_PATH=/home/derek/glm52-tr3-325-public-20260729/cache-v4-validation \
docker compose -f compose.yaml up -d --force-recreate
```

That rollback returns to image digest
`sha256:b53d5d551937a0580848101dfc5df9b7fb2638419cfa6da0fa35d0a2d339fe2e`,
the 532,992-token pool, 52.77 tok/s MTP3 C1, 120.04 tok/s C32, exact cold
350k retrieval, and KLD 0.0959706378925062.
