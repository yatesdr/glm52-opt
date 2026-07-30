# TR3 MTP0/C1 decode-gap investigation — 2026-07-29

## Scope and target

CN4, four RTX PRO 6000 Blackwell GPUs, TP4/DCP4, MTP0/C1.  The objective is
to explain and reduce the TR3 decode gap without trading away its measured
quality or context capacity.  The acceptance floor is 45 tok/s; the goal is
the fastest result supported by causal evidence.

Measured end-to-end controls:

| Arm | MTP0/C1 decode | Step time |
|---|---:|---:|
| NF3 production | 45.901 tok/s | 21.786 ms |
| TR3 r9 dynamic-KV | 35.324 tok/s | 28.309 ms |
| Gap to explain | — | 6.523 ms |

Evidence:

- NF3: `harness/cn4-evidence-archive/20260728/pr86-pr189-cleanup/dynamic-mtp0.jsonl`
- TR3: `harness/cn4-evidence-archive/20260728/r9-tr3-dynamic/decode-mtp0/n10.jsonl`

## Discriminator 1: routed-expert Trellis kernel

Probe:
`harness/tr3_trellis_decode_profile.py`
(`sha256:7591900c6d1069a23d77b8c143284b8a9fc4da447cc43370f26d8f2a69612949`
at execution).

It uses the exact GLM TP4 routed-expert geometry: M=1, H=6144, local I=512,
256 experts and top-k 8.

| Operator | CUDA-graph median |
|---|---:|
| TR3 Trellis full-rotation MoE | 71.30 us/layer |
| NF3 production hybrid MoE | 69.25 us/layer |
| Difference across 75 routed layers | about 0.154 ms |

The Trellis profile is finite and eager/graph bit-exact.  A safe cooperative
grid sweep from 64 through 188 blocks found the current grid-188 setting on
the optimal 160–188 plateau.

This probe repeats one synthetic layer, so its weights remain hot in L2.  The
later full-model trace measured the same Trellis fused kernel at 86.05
us/layer while streaming distinct layer weights.  Therefore the probe rules
out launch geometry and the grid choice, but it does **not** prove that the
full-model Trellis/NF3 weight-streaming costs differ by only 0.154 ms.  A
matched full-model NF3 trace is required for that attribution.

CN4 artifact:
`/home/derek/tr3-trellis-profile-20260729/out/baseline.json`.

## Discriminator 2: BF16 dense/shared projections versus B12X MXFP8

Source inspection showed that EXL3 deliberately returns
`UnquantizedLinearMethod` for non-EXL3 projections, while NF3 production uses
an online MXFP8 overlay on eligible dense/shared projections.  This made the
overlay a plausible explanation, but the exact-shape measurement refuted it.

Probe:
`harness/tr3_dense_decode_profile.py`
(`sha256:fafc2a25130b4185b9a8fe81cd76ad87add21e859079c8a9075541711cd015f7`
at execution).

| TP4 M=1 projection | Count | BF16 | B12X MXFP8 |
|---|---:|---:|---:|
| fused QKV-A | 78 | 13.152 us | 20.128 us |
| Q-B | 78 | 12.832 us | 12.768 us |
| O projection | 78 | 16.592 us | 20.240 us |
| shared gate/up | 75 | 11.712 us | 17.536 us |
| shared down | 75 | 11.456 us | 11.408 us |
| dense gate/up | 3 | 18.176 us | 22.144 us |
| dense down | 3 | 14.912 us | 17.008 us |

Weighted hot-weight operator accounting:

- BF16: 5.158 ms/model pass
- MXFP8: 6.433 ms/model pass
- MXFP8 change: **1.275 ms slower**

All MXFP8 outputs were finite with cosine similarity 0.99925–0.99937 against
the BF16 operator output.  This is an operator timing result, not a KLD
acceptance claim.  It proves that MXFP8's arithmetic/launch overhead loses
when repeatedly reading one hot weight.  It does not measure the full-model
benefit of streaming smaller packed weights across 78 distinct layers.  The
full trace below makes that distinction material, so a broad overlay is not
yet accepted or rejected.

CN4 artifact:
`/home/derek/tr3-dense-profile-20260729/out/baseline.json`.

## Full-runtime TR3 trace

The exact r9/TR3 MTP0/DCP4 posture captured two steady decode iterations on
all four ranks.  Rank-to-rank timing is consistent.  Rank 0 reported:

| Full-runtime phase or kernel family | Time/token |
|---|---:|
| Main graph GPU wall | 25.54–25.93 ms |
| Full steady engine step | about 29.5 ms |
| Trellis fused MoE kernel only | 6.454 ms |
| Trellis route/top-k named support kernels | 1.391 ms |
| `nvjet` dense/shared projection kernels | 6.912 ms |
| DCP head gather + LSE reduce | 2.587 ms |
| fused all-reduce + RMS norm | 2.136 ms |
| unified sparse MLA decode | 1.113 ms |
| steady metadata build on GPU | 1.173 ms |
| logits | 0.354 ms |

The two graph replays began 30.97 ms apart while profiling.  Trace processing
adds overhead, so the frozen n=10 throughput measurement remains the rate
control; the trace is used for attribution.

Evidence:

- `harness/cn4-evidence-archive/20260729/tr3-decode-gap-full-runtime-mtp0-v1/`
- rank-0 summary SHA-256
  `258e5bba46ebc5b456a1ecb4d4f63c51728aed14d2ed7f231460a02abf81ae23`
- rank-0 trace SHA-256
  `599ffcd8663545edcdbd11234dd679feddde7a8d7ad1843b130d0558aa7d4d8f`

## Matched NF3 full-runtime control

The measured NF3 image was then captured with the same TP4/DCP4, MTP0/C1,
PXB, i8-ring and dynamic-KV settings.  Both traces contain two steady decode
iterations.  The NF3 launcher unconditionally enabled asynchronous
scheduling, whereas the r9 TR3 launcher honored the requested synchronous
posture, so the comparison below is deliberately limited to GPU graph and
kernel attribution rather than host scheduling.

| Full-runtime family | NF3 ms/token | TR3 ms/token | TR3 delta |
|---|---:|---:|---:|
| Main graph GPU wall | 19.000 | 23.040 | **+4.040** |
| Routed-MoE core | 5.683 | 6.854 | +1.171 |
| Route/top-k support | 0.286 | 0.992 | +0.706 |
| Dense/shared projection family | 7.410 | 8.383 | +0.973 |
| DCP head gather + LSE reduce | 2.586 | 2.587 | +0.001 |
| PCIe all-reduce family | 2.221 | 2.433 | +0.212 |
| Sparse MLA decode/merge | 1.183 | 1.363 | +0.180 |
| Sparse indexer | 0.462 | 0.465 | +0.003 |

The family rows can overlap in execution and therefore are not additive wall
time.  They are nevertheless matched kernel sums over the same two decode
iterations.  DCP and the sparse indexer are effectively identical, directly
refuting PCIe/DCP as the primary C1 gap.  The routed-expert core plus its
route support is 1.877 ms/token slower, and the dense/shared projection family
is 0.973 ms/token slower.  These two weight-path families explain most of the
4.040 ms graph-wall delta.  The remaining difference is distributed among
all-reduce, sparse attention and smaller runtime kernels.

At 23.040 ms, the TR3 graph alone has a 43.4 tok/s physical ceiling before
host work, so the 45 tok/s acceptance floor cannot be reached solely through
host scheduling.  At least one GPU weight-path optimization is required.

Evidence:

- `harness/cn4-evidence-archive/20260729/nf3-decode-gap-full-runtime-mtp0-v1/`
- rank-0 NF3 summary SHA-256
  `25aeb933c339373cbd026c5483733bddb019f835ee14aa59771ddd91a560d57e`
- rank-0 NF3 trace SHA-256
  `6ff9508724cc74a0f24bfe45bcefed6e6d8bc2a0706145225b9f2c30872b5166`

## Current conclusion and next code line

The two full traces close the broad search:

1. The C1 gap is predominantly GPU weight execution, not PCIe/DCP.
2. The first code target is the TR3 routed-expert path: fuse or eliminate the
   extra route packing/top-k-sum work and reduce distinct-layer Trellis weight
   streaming/decode cost.
3. The second target is the unquantized dense/shared projection path.  Any
   packed-MXFP8 experiment must preserve the TR3 KLD advantage and therefore
   needs an operator proof followed by the fixed-window KLD gate.

The downloaded 3.25-bpw checkpoint is evaluated before source changes.  It
may improve KLD but uses the same runtime seams, so its KLD result and the
kernel optimization are independent decisions.
