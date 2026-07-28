# v20 Boot 9 — aligned-length go-live

Date: 2026-07-23  
Operator: Fable  
Objective: boot and qualify v20 on CN3 without another source rebuild or
configuration ladder

## Decision

Use the exact Boot 8 image and Compose. Change exactly one serving value:

```text
MAX_MODEL_LEN: 480000 -> 479744
```

Do not change the image, GMU, MNS, MNBT, graph cap, A2A threshold, i8-ring
settings, KV dtype, CKV gather, DRAM tier, NVMe tier, prefix caching, allocator
configuration, mounted source, or cache volumes. Keep
`VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS` and `CUDA_LAUNCH_BLOCKING` unset.

Boot 8 image identity:

```text
tag:    ...-int8-nvme-bt1876-w4a16coop-ckvreset
digest: sha256:5f0c7b43daaa...
```

Record the complete local tag and digest from CN3 before launch; the shortened
values above are report identifiers, not sufficient pull pins.

## Why this is the next production boot

With block size 64 and DCP4, v20 derives a rank-local page-table width from:

```text
ceil(max_model_len / (block_size * DCP))
```

The model runner then aligns backend-facing rows to a 128-token boundary:

```text
480000: ceil(480000 / 256) = 1875 -> aligned width 1876
479744: ceil(479744 / 256) = 1874 -> aligned width 1874
```

PR #166 repaired the indexer's expanded table at the 1875/1876 seam, but Boot
6 proved that repairing that one consumer was not sufficient. Other v20
consumers still derive logical and padded dimensions independently. The one
independently successful v20 TP4/DCP4/MTP3 configuration explicitly uses
479744 and records `480000` as a tensor-crash boundary. Every CN3 failure so
far retained exactly 480000; it is the only such boundary not tested away.

This is a 256-token advertised-context reduction (0.053%). It preserves the
475k qualification needle, MTP3, MNS16, cap64, i8-ring, DRAM offload, NVMe
offload, and the measured 555,520-token Boot 8 GPU KV pool envelope.

## Preflight

Before launch:

1. verify the Boot 8 image digest and all six Boot 8 runtime byte pins;
2. produce and preserve a Compose diff proving the only change is max length;
3. verify CN3 has no old vLLM container or CUDA process;
4. verify no stale `/dev/shm/vllm_offload_*.mmap` remains; and
5. verify the staged NVMe acceptance namespace is still empty and resides on
   the intended ext4 NVMe filesystem.

Abort before launch if the Compose diff contains any other change.

## Boot gate

Allow active compilation to finish. Abort only on a real process exit,
traceback, OOM, Xid, fatal assertion, or 15 minutes with no log, compiler, or
GPU activity.

PASS requires:

1. target, prefill-speculator, and all production decode-speculator captures
   complete;
2. the API reaches serving state;
3. deterministic liveness returns `4`;
4. one normal request returns `finish_reason=stop` with nonzero MTP acceptance;
5. `RestartCount=0`, with stable container ID and `StartedAt`; and
6. no illegal access, OOM, Xid, EngineDead, assertion, worker exit, or 5xx.

Record the GPU KV pool. The hard floor remains 500,000 tokens.

## Same-process qualification

If the boot gate passes, do not restart or tune. Immediately run on that same
process:

1. decode at concurrency 1, 4, 8, and 16;
2. unique-prefix needles at 300k, 350k, and 475k;
3. bounded NVMe fill, turnover, and promotion at the staged 8 GiB limit;
4. 16 overlapping unique-prefix 50k requests, proving demand exceeds the GPU
   pool and DRAM/NVMe offload is exercised;
5. cold 8k and 50k prefill with prefix-cache miss deltas; and
6. final liveness, process fingerprint, restart count, fatal-signature audit,
   and tier-capacity inventory.

If all gates pass, perform the already planned identical persistence restart,
prove an NVMe-retained entry can be restored/promoted, run liveness plus one
MTP request and one 475k needle, and leave that exact second v20 process online.

## Failure action

If Boot 9 fails at the same production capture boundary, stop after preserving
the complete first-run log and inspect JSON. Do not run a knob ladder. The next
engineering boot must use this exact 479744 configuration with both existing
descriptor diagnostics and `CUDA_LAUNCH_BLOCKING=1`; that boot is diagnostic,
not a go-live retry.

