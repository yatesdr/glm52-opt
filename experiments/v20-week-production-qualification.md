# v20 week-production qualification decision

Date: 2026-07-22  
Target: CN3, GLM-5.2 TP4/DCP4, MTP3  
Objective: one qualification boot, one identical persistence restart, then
leave the passing second boot in production for the week

## Decision

Do not run another root-cause boot ladder tonight. Qualify one conservative
v20 profile that preserves the block-INT8 prefill win, keeps 16-request
capacity, avoids both presently unqualified boundaries, and maximizes useful
GPU KV without relying on the new MRV2 allocator patch.

Use the pre-MRV2 candidate image:

```text
ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-vllm3e731bc-si1a88b38-int8-nvme-mtpfix
```

This is the v20 base plus:

- the merged DRAM OffloadingConnector readiness fix `c29debe6` (PR #134),
  which prevents the previously reproduced `_build_store_jobs` assertion;
- PR #69 block-INT8 `i8_ring` wire support;
- PR #165 bounded filesystem/NVMe eviction; and
- PR #166 padded block-table correction.

Do **not** use the `...mrv2fix` image or commit `982cda45` for this production
qualification. Its accounting behavior is proven, but it returned only 11,520
additional GPU KV tokens and has not completed a serving boot. The stock v20
MRV2 estimate is conservative, which is preferable for a week-long production
deployment.

## Fixed production candidate

```text
TP/DCP/MTP:                         4 / 4 / 3
MAX_NUM_SEQS:                       16
MAX_CUDAGRAPH_CAPTURE_SIZE:         32
MAX_NUM_BATCHED_TOKENS:             3072
MAX_MODEL_LEN:                      479744
GPU_MEMORY_UTILIZATION:             0.978
VLLM_DCP_A2A_MAX_TOKENS:            16
VLLM_DCP_A2A_LARGE_BACKEND:         ag_rs
VLLM_PCIE_DMA_FP8:                  i8_ring
SPARKINFER_PCIE_DMA_FP8:            i8_ring
KV cache:                           nvfp4_ds_mla + KV_FP8_ROPE=1
DRAM primary offload:               64000000000 bytes
filesystem/NVMe secondary limit:    8589934592 bytes
filesystem namespace:               fresh acceptance namespace
PYTORCH_CUDA_ALLOC_CONF:            expandable_segments:False
PYTHONHASHSEED:                      0
```

Keep every other reviewed candidate setting and source byte unchanged.

### Why these values

- **`i8_ring` stays fully enabled.** Its large TP all-reduces remain the prefill
  transport. The DCP routing settings are a separate communication axis.
- **A2A16 is the supported hybrid route used by the independently successful
  TP4/DCP4 v20 configuration.** MTP3 batches above 16 rows use AG/RS instead of
  the unqualified large B12X CUDA-IPC channel. MTP and concurrency remain on.
- **479,744 loses only 256 advertised tokens** but avoids the odd 1,875 logical
  versus 1,876 padded local page-table boundary. PR #166 remains in the image
  but its width-trimming branch becomes a no-op at this aligned length.
- **Graph cap32 does not reduce `MAX_NUM_SEQS=16`.** It keeps the field-proven
  graph envelope and lets larger verification batches use the supported
  non-full-graph path. The tradeoff may be lower C16 decode throughput; measure
  it in this boot rather than accepting another capture-risk boot.
- **GMU0.978 is the measured conservative point.** CN3 measured 592,640 GPU KV
  tokens at GMU0.978/MNS8/cap32 before the crash. MNS16 may consume somewhat
  more static scratch, so require at least 500,000 rather than predicting the
  exact result. This still fits 479,744 and leaves DRAM/NVMe to absorb
  concurrency overflow.
- **The 8 GiB NVMe limit is production-safe for this week.** It is large enough
  to prove bounded turnover and persisted reuse, and small enough to exercise
  eviction quickly. Do not change it to 64 GiB after qualification; keeping
  the exact proven limit avoids a third boot and an unqualified production
  mutation. The 64 GB DRAM tier remains the main overflow capacity.

## Boot budget: exactly two

### Boot 1 — qualification

Use the fixed profile above. If boot fails, restore v19. Do not change a knob
and retry v20 during this window.

After health, run the existing combined acceptance gates in this order on the
same process:

1. source-byte and image verification, process fingerprint, GPU KV pool;
2. bounded NVMe fill/turnover using the fresh 8 GiB namespace;
3. cold 8k and 50k prefill, with prefix-cache miss evidence;
4. decode C1, C8, and C16 at ctx0 plus C8 at ctx50k;
5. unique-prefix needles at 300k, 350k, and 475k;
6. 16 overlapping unique-prefix 50k requests to exceed the GPU pool and force
   DRAM/NVMe offload; and
7. final health, logs, process fingerprint, cache inventory, and metrics.

Do not stop after the boot gate. The point of this choice is to spend the
remaining first boot characterizing the exact production profile.

### Boot 2 — identical restart and promotion proof

Restart only the vLLM service without changing the image, Compose, cache
limits, namespace, JIT cache, or source bytes. Prove persisted NVMe promotion
with the saved replay anchor, then run one clean liveness request. If it passes,
leave this exact second boot online as production.

No third boot is planned.

## Hard acceptance

Production PASS requires all of the following:

- GPU KV pool at least 500,000 tokens and max length 479,744 accepted;
- in-image scheduler bytes contain merged `c29debe6`, and the INT8, NVMe, and
  PR #166 output pins match their reviewed values;
- startup and runtime `RestartCount` remain zero except for the one controlled
  Boot1-to-Boot2 replacement;
- all required requests complete without worker death, CUDA illegal access,
  OOM, Xid, assertion, 5xx, watchdog action, or offload-store mismatch;
- bounded NVMe fill and eviction pass at 8 GiB;
- persisted replay produces a real external-cache hit after restart;
- needles 300k/350k/475k all return `738216` with `finish_reason=stop`;
- 16/16 overflow requests pass and demand exceeds the recorded GPU pool;
- cold prefill is no more than 15% below the v19 `i8_ring` reference;
- decode cells are recorded; a C16 regression attributable to graph cap32 or
  AG/RS is acceptable only by Derek's explicit review before promotion; and
- Boot 2 remains healthy and responsive after replay.

Any correctness, stability, needle, offload, or persistence failure means v19
rollback. Do not waive those gates for performance. A throughput-only hold is
a review decision, not permission for an automatic configuration retry.

## Deferred engineering after production is restored

Investigate and repair these outside the production window:

1. B12X CUDA-IPC DCP behavior above 16 rows;
2. the 480,000 padded page-table boundary with PR #166;
3. MRV2 global graph-pool reuse in PR #168; and
4. cap64 versus cap32 high-concurrency decode performance.

None is required to retain `i8_ring`, DRAM offload, bounded NVMe eviction,
MNS16, deep-context retrieval, or a 500k-plus GPU KV pool in the selected
week-production profile.
