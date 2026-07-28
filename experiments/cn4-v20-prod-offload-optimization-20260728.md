# CN4 v20 production/offload optimization — 2026-07-28

## Objective

Run the pinned v20 dynamic-NVFP4 image on CN4 with:

- prefix caching enabled;
- 64 GB aggregate DRAM KV offload;
- a bounded 1 TB filesystem tier on the Intel NVMe;
- the exact selector and dynamic per-token NVFP4 MLA-KV quality fix;
- the best already-supported TP4/DCP4 PCIe posture; and
- a larger GPU-resident KV pool without disabling required features.

CN3 is production and must not be changed or restarted.

## Pinned runtime

```text
ghcr.io/yatesdr/glm52-serve@sha256:fa6365fba88eee8b34b8e6d14dedc79eb3f43bda2bb787d8e71b014421bcd929
```

The image contains:

- SparkInfer dynamic-scale PR #86 (`0d9aead9`);
- vLLM dynamic-scale PR #189 (`91dff5a9`);
- bounded filesystem-manager commit `95488c388`;
- exact top-k selection;
- dynamic FP32 per-token NVFP4 scale in the 368-byte MLA record; and
- offload metadata-lag fix `c29debe6` plus the regression test covering
  missing block IDs and missing keys.

No source or image change is required for this exercise.

## Measured starting point

CN4 exact-image baseline, prefix caching enabled, no active offload connector:

| Setting/result | Value |
|---|---:|
| GPU memory utilization | 0.9848 |
| CUDA graph limit | 64 |
| GPU KV | 522,240 tokens / 3.93 GiB |
| 8k cold prefill | 1,382 tok/s |
| 64k cold prefill | 1,315 tok/s |
| C1 decode | 75.3 tok/s |
| C4 aggregate decode | 133.3 tok/s |

Benchmark artifact:

```text
cn4:/home/derek/llm-inference-bench/results/v20-prod-share-cn4-minimal-20260728/results.json
```

## Where the missing GPU KV went

The limiting-rank startup ledger at GMU 0.9848 reported:

- weights: about 84.93 GiB;
- peak activation: 2.84 GiB;
- non-Torch: about 0.76–0.79 GiB;
- estimated MRV2 graph memory: about 1.07 GiB;
- actual retained graph memory: about 0.21 GiB; and
- GPU KV: 3.93 GiB / 522,240 tokens.

The graph estimator therefore held back roughly 0.86 GiB/GPU more than the
captured graph pool used. vLLM printed the corrective recommendation itself:

```text
current gpu_memory_utilization=0.9848 is equivalent to 0.9736
without CUDA graph profiling; increase to 0.9960
```

The candidate does **not** follow the `0.9960` recommendation. That GMU leaves
too little unmanaged headroom. Instead, the image exposes vLLM's
`--kv-cache-memory-bytes` through a fail-closed launcher input and uses the
minimum post-capture recommendation across all four ranks:

```text
KV_CACHE_MEMORY_BYTES=4938417470  # 4.60 GiB
GPU_MEMORY_UTILIZATION=0.9848
```

This keeps CUDA graphs, prefix caching, MTP, and every quality feature.

The historical 550,144-token run is not a clean prefix-enabled comparison:
it used `enable_prefix_caching=False` and also measured a lower 84.76 GiB
weight footprint. It remains useful evidence that the record format can
expose a larger pool, but the new boot measurement is authoritative.

## Selected production posture

The staged Compose is:

```text
compose/glm52-v20-cn4-prod-offload-20260728.yaml
sha256:767ab28a118b47fa12ae4a6e9aeb494ad1ee1ab6afe0ee67cac7fc6047b9e570
```

Key settings:

| Area | Selected value | Basis |
|---|---|---|
| GMU | 0.9848 | retained production envelope |
| GPU KV bytes | 4,938,417,470 | minimum post-capture recommendation across ranks |
| Prefix caching | enabled | required production feature |
| DRAM offload | 64,000,000,000 bytes | operator requirement; known CN3 form |
| Intel NVMe cap | 1,000,000,000,000 bytes | operator-approved bounded capacity |
| NVMe threads | 16 read / 16 write | existing CN3-tested configuration |
| NVMe locality | LOCAL | existing CN3-tested configuration |
| MNBT | 3072 | higher PCIe prefill result than 2048 |
| Owner merge | 0 | correctness-equivalent; community PCIe4 regression when on |
| Query split | 0 | avoids transient pressure in the 480k posture |
| Indexer shards | 0 | measured TP4 posture |
| CKV prefetch | 0 | avoids the 233.25 MiB transient |
| Topology | auto (expected PXB) | measured CN4 probe |
| Wire | auto (expected i8_ring) | rank-consistency validated |
| Dynamic NVFP4 scale | 1 | retrieval/KLD quality fix |
| Destroyed MXFP8 membership | 0 | preserves the larger KV pool |
| Network | Docker default bridge | avoids a project-specific subnet |

The filesystem tier uses the fresh namespace:

```text
/nvme-kv/glm52-v20-dynamic-prod
```

This prevents mixing old static-scale and new dynamic-scale cache semantics.

## KLD state

The dynamic implementation previously passed matched n=3 KLD:

| Membership | Mean KLD | Sample SD |
|---|---:|---:|
| production NF3/MXFP8 | 0.139036 | 0.002010 |
| optional quality-first MXFP8 | 0.133262 | 0.002125 |

Reference direction is `KL(BF16 reference || candidate)`, 2,047 positions per
run. Two exact release-image confirmation attempts on 2026-07-28 failed
closed during first-prompt JIT before emitting any `summary.json`; they are
invalid harness runs, not measured quality regressions. Preserve:

```text
cn4:/home/derek/kld-prod-release-20260728/
```

The final image is derived directly from the n=3-tested `db82fdcb...` image.
All 13 installed files changed by the SparkInfer/vLLM dynamic-scale commits
were hashed in both images. Their sorted manifest hashes are identical:

```text
base  e796e428da85554f85a74a20fb223890b7c112460cee77be5bf0a5c9e836f332
final e796e428da85554f85a74a20fb223890b7c112460cee77be5bf0a5c9e836f332
```

The derived layer changes only the filesystem manager, launcher hook, and
auto-probe scripts. Therefore the valid n=3 KLD result transfers byte-for-byte
to the release image even though the redundant release-image harness run is
currently defective.

## CN4 acceptance result

The candidate reached healthy serving on 2026-07-28:

| Result | Value |
|---|---:|
| Derived image ID | `sha256:cacf3304e586906aa504aab966f2eed6e82e34f6700bad06ac65e69313f37cdc` |
| Compose SHA-256 | `767ab28a118b47fa12ae4a6e9aeb494ad1ee1ab6afe0ee67cac7fc6047b9e570` |
| GPU KV | **617,728 tokens / 4.60 GiB** |
| Gain over baseline | **+95,488 tokens / +18.3%** |
| Maximum 480k concurrency | 1.29x |
| DRAM tier | 64,000,000,000-byte active shared mmap |
| DRAM primary capacity | 7,816 blocks |
| NVMe tier | enabled, 0 / 1,000,000,000,000 bytes initially |
| NVMe namespace | `/nvme-kv/glm52-v20-dynamic-prod/_model_a1f720d91ee6_r0` |
| Health | Docker healthy; local and LAN `/health` returned 200 |
| Final restart count | 0 |
| Minimal inference | passed |

The legacy 64 GB mmap was proven to have no open file descriptors, deleted,
and replaced by one active mmap belonging to the new engine ID. The prior
Compose project network was removed; the accepted container uses Docker's
built-in bridge and creates no project subnet.

The final graph-capture log reported 1.07 GiB retained and idle GPU headroom
of 939–1,064 MiB. This is tighter than the earlier 0.21 GiB incremental graph
measurement implied, so the 617,728-token result is accepted as a CN4
candidate after a real inference smoke, not yet as a universal hardware
default. Operators should derive the exact byte value from their own
limiting-rank ledger.

No GPU frequency setting was changed during this deployment. The persistent
`nvidia-powercap.service` applied the established `-lgc 0,2600` graphics-clock
ceiling and the intentional 300 W power cap to all four cards at boot. The
3090 MHz value reported by `clocks.max.graphics` is the hardware maximum, not
evidence that the active 2600 MHz ceiling was removed.

## PR review follow-ups

Independent review of SparkInfer #86 and vLLM #189 left four pre-merge items:

1. Make dynamic/static record semantics part of an immutable external-cache
   ABI identity, or add an explicit record marker and stale/mixed-cache
   rejection test.
2. Correct the scale claim: per-token scaling positions the largest group
   scale near E4M3 maximum; it does not prove every smaller group scale is
   non-subnormal.
3. Add focused vLLM tests for mutual exclusion, incompatible
   SparkInfer signatures/layouts, and both writer call sites; run pre-commit.
4. Measure matched static-versus-dynamic decode at MTP0.

KLD attribution must remain matched:

| Comparison | Mean KLD | Incremental interpretation |
|---|---:|---|
| Exact + static #145 | 0.146228 | baseline |
| Exact + dynamic per-token scaling | 0.139036 | **-4.92%**, matched dynamic-scale gain |
| Quality MXFP8 + static | 0.133784 | separate membership mode |
| Quality MXFP8 + dynamic | 0.133262 | -0.39%; likely within n=3 variance |

Use “dynamic per-token KV activation/latent scaling,” not “dynamic KV weight
assignment.” Bounded selectors remain diagnostic compatibility paths, not
production defaults.
