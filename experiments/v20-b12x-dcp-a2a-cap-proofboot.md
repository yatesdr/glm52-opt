# v20 B12X DCP A2A large-batch proof boot

Date: 2026-07-22  
Operator: Fable  
Target: CN3, GLM-5.2 TP4/DCP4, MTP3

## Decision

Run one boot with the exact Boot 4 image and profile, changing only:

```text
VLLM_DCP_A2A_MAX_TOKENS=16
```

The failed value was `64`. Keep `VLLM_DCP_A2A_LARGE_BACKEND=ag_rs`.

Do not add `CUDA_LAUNCH_BLOCKING`, change max model length, remove an overlay,
alter graph sizes, or run a configuration ladder in this boot.

## Why this is the highest-information next boot

The independently successful TP4/DCP4 v20 configuration in
`/Users/derek/Downloads/message-6.txt` uses the same `a2a` DCP backend but caps
the small-batch route at 16 tokens. CN3 uses 64. In the v20 code:

```text
num_mqa_tokens <= VLLM_DCP_A2A_MAX_TOKENS
    -> B12X CUDA-IPC DCP query gather and LSE reduce-scatter

num_mqa_tokens > VLLM_DCP_A2A_MAX_TOKENS
    -> VLLM_DCP_A2A_LARGE_BACKEND (ag_rs here)
```

With MTP3, each active request can contribute four verifier rows. The failed
MNS16/cap64 profile therefore admits 32- and 64-row graph buckets to the B12X
CUDA-IPC channel. The successful user's cap of 16 sends those buckets through
the established AG/RS path. CN3's earlier MNS8/cap32 attempt retained the
64-token A2A limit, so it did not test this distinction.

The SparkInfer tests at source commit `1a88b38` cover channel lifetime and
layout with CPU fakes whose configured capacity is four rows; they do not
provide a real multi-GPU 32/64-row proof. The repeated CN3 failure begins after
the target capture and surfaces at the first synchronizing operation in the
production speculator capture, which is consistent with an asynchronous
target-side collective fault.

This also distinguishes the candidate overlays cleanly:

- PR #69 changes the TP all-reduce wire codec. A 64-row MTP tensor is below
  its 6 MiB DMA threshold, so block-INT8 is not selected here.
- PR #165 changes filesystem-tier capacity/eviction bookkeeping and performs
  no NVMe store or load during graph capture.
- PR #166 is in the indexer path, but remains byte-identical in this proof.
- `982cda45` remains present, so the already-proven MRV2 accounting correction
  and 555,520-token result are not conflated with this route test.

## Exact boot

Use the Boot 4 image and compose:

```text
ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-vllm3e731bc-si1a88b38-int8-nvme-mrv2fix
local image ID: sha256:89a27fe2b4c0...
```

Retain all of these:

```text
TP4 / DCP4 / MTP3
MAX_MODEL_LEN=480000
MAX_NUM_SEQS=16
MAX_CUDAGRAPH_CAPTURE_SIZE=64
MAX_NUM_BATCHED_TOKENS=3072
GPU_MEMORY_UTILIZATION=0.980
VLLM_DCP_A2A_LARGE_BACKEND=ag_rs
VLLM_PCIE_DMA_FP8=i8_ring
SPARKINFER_PCIE_DMA_FP8=i8_ring
NVMe tier=8 GiB acceptance namespace
PR #69 + PR #165 + PR #166 + 982cda45
```

Change only the existing compose entry:

```diff
- VLLM_DCP_A2A_MAX_TOKENS=64
+ VLLM_DCP_A2A_MAX_TOKENS=16
```

Before boot, record the compose hash, image ID, and the same five in-image byte
pins recorded for Boot 4. Keep the NVMe acceptance namespace pristine.

## Gate 1

PASS requires:

1. all four workers finish MRV2 profiling, KV allocation, target capture, and
   speculator capture;
2. the API becomes healthy;
3. the reported KV pool remains at least 500,000 tokens;
4. `RestartCount` remains zero and container ID/`StartedAt` stay unchanged;
5. no illegal access, worker death, `EngineDeadError`, OOM, or 5xx occurs; and
6. one deterministic MTP request returns HTTP 200 with a correct non-empty
   completion.

Stop after Gate 1 and report. Do not proceed into NVMe or performance gates
until the attribution is reviewed.

## Interpretation

### PASS

A pass is causal evidence that the crash requires admitting a graph above 16
rows to the B12X CUDA-IPC DCP path. It exonerates the INT8 and NVMe overlays and
shows that 480k plus PR #166 can boot. The supported production containment is
to retain MNS16 and graph cap64 while using the hybrid A2A16/AG-RS route. This
does not disable MTP, reduce maximum concurrency, or remove the 32/64 graphs;
it changes only the DCP transport for batches above 16 rows.

After a PASS, measure decode at concurrency 1/4/8/16 because MTP3 crosses the
route boundary above roughly four active requests. A source patch should then
target or guard the B12X DCP channel's unqualified large-batch path; do not
alter PR #69, #165, #166, or `982cda45` to explain this failure.

### FAIL with the identical illegal access

The A2A-large-batch hypothesis is falsified. Do not try another memory or graph
size. The next discriminator is the only remaining capture-time code delta
against the successful user's base: the `MAX_MODEL_LEN=480000` padded
block-table seam plus PR #166. Prepare one exact-base 479,744 boot without the
PR #166 overlay; do not combine that test with any other change.

