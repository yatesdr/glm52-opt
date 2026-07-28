# v20 production-candidate build and promotion handoff

Date: 2026-07-23  
Owner: Derek  
Operator: Fable

## Objective

Build once, boot once with the intended production configuration, qualify that
same running process, and leave it online if every gate passes. This is not a
configuration or patch ladder.

Budget approximately 60–75 minutes of production downtime: roughly 17 minutes
to boot, five cold long-context needles, the sentinel performance cells, and
the 16 x 50k overlapping stress. An actively compiling healthy boot may extend
that estimate; do not restart it merely for exceeding the estimate.

## 1. Exact source inputs

### vLLM

Use the evidence-backed production tree:

```text
/Users/derek/glm52-opt/workspace/vllm-v20-cn3-prod-proven
branch: integration/v20-cn3-prod-proven-20260723
head:   dcb83079f
```

Required history:

```text
dcb83079f  CPU policy proof for native-versus-flattened MTP indexer routing
87d987e01  native SM120+B12X MTP3 indexer path (extracted from PR #139)
95488c388  bounded filesystem/NVMe KV tier (PR #165)
af9d01cf1  voipmonitor v20 final-candidate integration
```

`af9d01cf1` already contains #166, #167, #169, #172, and #173. Do not
reapply their older local predecessors.

PR #171 is deliberately absent. The independently validated 9/9 deep-needle
configuration used the upstream `auto` verifier route with only the native
MTP3 indexer correction. Keep #171 on its separate experimental branch for a
later A/B; do not combine it with this promotion.

Push the production branch for the build without rebasing, squashing, or
resolving onto a different base. Record the remote commit SHA used by the
builder.

### Upstream PR topology

Do not open the production integration branch as one omnibus PR. Its base is
the exact image integration commit, while maintainers review topic changes
against `local-inference-lab/vllm:dev/gilded-gnosis`.

Use these two independent paths:

```text
NVMe capacity:
  existing PR #165
  head patch is identical to production commit 95488c388

Native B12X MTP3 indexer:
  local branch fix/gg-b12x-native-mtp3-indexer-20260723
  head de394d28e
  base 4a4299c4b (current dev/gilded-gnosis)
```

The native-indexer branch preserves Brandon Music's original authorship and
extracts only that fix from broad EXL3 PR #139, followed by a separate CPU
policy test. Its PR description must reference #139 and must not claim the
original implementation as ours.

Critical vLLM byte pins:

```text
653edbf4b393e2acd6204bf4664c300eaee9e959656040864491c94548b4cc60  vllm/v1/kv_offload/tiering/fs/manager.py
23ad50cb6017e54f1c9fee76b0d999e3544d584e9377fc776041bfa3d2d2b821  vllm/v1/attention/backends/mla/indexer.py
df63fc6f6a72f9b5b6cfacc25a5ead2d83f28f98c21df1b52961bf143f0a16f2  vllm/v1/attention/backends/mla/b12x_mla_sparse.py
392c13256154b9ba66db732df5b8e2ea69dcfac65e5b603e06c4bd7330bc8d3e  vllm/v1/worker/gpu/model_runner.py
6506fd5580bdfde622369943eb9b910ec0459e60911e53952de5f6a3659ec0db  vllm/model_executor/layers/attention/mla_attention.py
4f4c2f2b3acb34396a1e323385bd89aae2a57e70e98a74b20b0ae125af13d385  csrc/libtorch_stable/attention/mla/safe_query_bmm.cu
```

### SparkInfer

Use the already-built final-candidate source unchanged:

```text
repo:   voipmonitor/b12x
branch: build/sparkinfer-v20-final-candidate-20260723
commit: ffa922b0c06e5c45ed1344bdc5260cc9c7e85c9a
```

This contains #71, #72, and #73. Do not reapply #46 or #69.

### Build method — digest-derived runtime image

The historical full-build recipe currently references unavailable
`glm-kimi-cu132-*-base-20260626` images. Do not rebuild those bases and do not
wait for Docker Hub credentials. The delta from `af9d01cf1` is Python-only:

```text
vllm/v1/attention/backends/mla/indexer.py
vllm/v1/kv_offload/tiering/fs/manager.py
```

Build a reproducible derived image from the already-published final candidate:

```text
base:
voipmonitor/vllm@sha256:b90ea60347a45cf8d964f32b663dc2302cf7aab97c529b87f794f364bdc126f5

Dockerfile:
/Users/derek/glm52-opt/docker/Dockerfile.v20-prod-proven
```

Run from the exact `dcb83079f` source checkout so the two `COPY` inputs come
from that commit:

```bash
docker build --pull \
  -f /path/to/Dockerfile.v20-prod-proven \
  -t ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-prodproven-20260723 \
  /path/to/vllm-v20-cn3-prod-proven
```

This is a baked image layer, not a bind mount or a runtime overlay. All CUDA,
C++, SparkInfer, NCCL, FlashInfer, InstantTensor, and system bytes remain those
of the already validated base. The Dockerfile compiles the two Python files,
checks their exact output hashes plus the unchanged Python pins, verifies the
safe query-BMM symbol is embedded in the stable-libtorch extension, and records
the base and patch revisions in OCI labels. Loading that extension requires
`libcuda.so.1`, so actual `torch.ops` registration is checked after launch,
when the NVIDIA driver is present.

### Historical full-build recipe

Use:

```text
repo:   local-inference-lab/blackwell-llm-docker
branch: build/gilded-gnosis-v20-final2-20260723
commit: de9ea1b
script: build-gilded-gnosis-v20-final-cu132.sh
```

Retain this only as the eventual full rebuild once its private/pruned base
images are restored. It is not required for the present candidate because no
compiled source changed. Do not set `BUILD_BASE_IMAGE=1` during this promotion
window.

No Python source bind mounts and no diagnostic overlays are permitted.

## 2. Build gate (no GPU boot)

Before starting the service:

1. Record the image ID and repository digest.
2. Verify the base digest, patch revision `dcb83079f`, embedded vLLM base
   revision `af9d01cf1`, and SparkInfer commit labels.
3. Verify the five installed Python byte pins above inside the image. Verify
   the `safe_query_bmm.cu` pin against the recorded `af9d01cf1` source/build
   provenance; the `.cu` source need not be installed in site-packages.
4. Statically verify `_C_stable_libtorch.abi3.so` contains the
   `safe_mla_query_bmm` symbol. Do not import the extension during
   `docker build`: the build environment has no `libcuda.so.1`.
5. Verify SparkInfer normalizes `i8_ring` to the block-INT8 ring wire.
6. Verify the image contains no CUDA-graph diagnostic patch and no
   `CUDA_LAUNCH_BLOCKING`.
7. Push this exact post-gate image to GHCR and record its GHCR repository
   digest before booting. The pinned base is public but hosted by a third party;
   copying the complete image to our registry prevents a later Docker Hub prune
   from making the accepted production artifact unrecoverable. Boot by the
   recorded GHCR digest, not a mutable tag.

Any mismatch is a build failure. Do not boot it.

## 3. One production-config boot

Start from `deploy/glm52-prod-v20.yaml`, changing only the image pin and the
items below.

Required posture:

```text
TP4 / DCP4 / MTP3
max_model_len=480000
max_num_seqs=16
max_cudagraph_capture_size=64
gpu_memory_utilization=0.980
wire=i8_ring
DRAM offload=64,000,000,000 bytes
NVMe capacity=64,000,000,000 bytes aggregate
              (one KV-transfer manager/rank with kv_parallel_size=1)
```

Configuration requirements:

- Set `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=auto` explicitly. Do not force
  `0` or `1`. The stock final-candidate logic must recognize genuine MTP
  verifier batches and use its automatic decode route, matching the
  independently validated configuration.
- Set `VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=524288` explicitly. This is the
  current v20 default and sizes the transient CKV communication arena to cover
  the full 480k production context. The independent 9/9 needle deployment also
  used this value, but that result does not isolate this performance setting as
  a correctness requirement. Spelling it out makes the arena and its memory
  cost visible in the deployment record.
- Set `SPARKINFER_PCIE_DMA_FP8=i8_ring`. Keep any compatibility alias set to
  the same value; never mix wire-mode strings.
- Remove the stale `VLLM_USE_B12X_PCIE_DMA` variable.
- Keep diagnostics, `CUDA_LAUNCH_BLOCKING`, and profiler probes disabled.
- Use a fresh v20 JIT/AOT cache namespace, not a v19 or earlier-v20 cache.
- Use a fresh NVMe model/config namespace. Do not delete another namespace.
- Keep `PYTHONHASHSEED=0`, `ipc: host`, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`.
- Keep the 64 GB aggregate production NVMe capacity. This deployment uses one
  EngineCore `TieringOffloadingManager` with `kv_parallel_size=1`, so the
  manager owns only the `_r0` namespace; TP4/DCP4 does not multiply the
  filesystem limit. Do not reduce it to 16 GB based on TP rank count. #165's
  bounded-capacity algorithm was runtime-proven at 8 GiB, and this boot is for
  production promotion rather than another isolated capacity experiment.
  Treat the 64 GB scale and restart persistence as not yet independently
  proven; this promotion boot checks initialization, writes, and stability but
  does not claim either additional proof.

Do not alter GMU, graph sizes, MNS, A2A cap, weight quantization, KV format,
reasoning effort, or offload sizes after launch.

Do not copy the independent EXL3 deployment's fixed
`--kv-cache-memory-bytes=5118000000` value. Its 3.0-bpw checkpoint has a
different resident weight footprint and its 640,000-token pool is not a valid
capacity promise for this MXFP8/NVFP4/NF3 checkpoint.

## 4. Boot acceptance

The single process may proceed only when all conditions hold:

- API reaches healthy/serving state.
- `RestartCount` remains zero and container identity is unchanged.
- No illegal access, cuBLAS failure, OOM, Xid, assertion, worker death, or
  `EngineDead`.
- Production target and speculator graph capture completes.
- A GPU-enabled process inside the running container can import both `torch`
  and `vllm._C_stable_libtorch`, and
  `hasattr(torch.ops._C, "safe_mla_query_bmm")` is true. This is the runtime
  registration check that cannot run in the driverless Docker build.
- GPU KV pool is at least 500,000 tokens and max context remains 480,000.
- The boot log reports
  `KV_FP8_ROPE=1 kv_gmem_stride=368 kv_cache_dtype=nvfp4_ds_mla`; a
  432-byte/BF16-RoPE run is a different quality and capacity configuration and
  cannot qualify this candidate.
- The log reports `use_flattening=False` for SM120+B12X with MTP3
  (`next_n=4`).
- Runtime/source introspection confirms
  `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=auto`, automatic verifier routing is
  enabled but not forced, and `nvfp4_ds_mla` has no format-specific override.
- Runtime introspection reports PCIe DMA enabled, backend `b12x`, wire
  `i8_ring`, with no fallback warning.
- The filesystem tier reports capacity enabled at 64,000,000,000 bytes.
- Exactly one `Created TieringOffloadingManager` line is present, the
  `KVTransferConfig` reports `kv_parallel_size=1`, and only the `_r0`
  filesystem namespace exists.
- A health request returns the expected answer.

Failure means stop immediately, preserve the complete first-run log and inspect
JSON, and restore v19. Do not tune or retry during this window.

## 5. Minimum promotion suite on the same running process

### Quality

Run genuinely cold, salted needles at:

```text
50k, 250k, 300k, 350k, 475k
```

For every request:

- confirm `cached_tokens=0`;
- diagnostically search `content`, `reasoning`, `reasoning_content`, and the
  serialized response message so retrieval and response-field behavior cannot
  be confused;
- require `content` to be a non-empty string containing the exact needle.
  Finding the value only in a reasoning field is retrieval PASS but
  finalization FAIL, and therefore promotion FAIL;
- require `finish_reason=stop`;
- record prompt and completion token counts.

All five must pass retrieval and finalization. The 250k and 300k cells are
intentional: those were the first depths where the previous engine retrieved
the value into `reasoning` but returned empty `content`. Also run the existing
arithmetic/coherence canary once.

### Performance

Run only the production sentinel cells:

- cold prefill at 8k and 55k with prefix-cache metric deltas;
- decode at C1, C8, and C16.

Acceptance floors:

```text
55k cold prefill hard floor: >= 1,350 tok/s
55k cold prefill target:     >= 1,395 tok/s
C1 decode:        >= 56.9 aggregate tok/s
C8 decode:        >= 114.5 aggregate tok/s
C16 decode:       >= 148.5 aggregate tok/s
```

Run two genuinely cold 55k samples and report both plus their median. A median
from 1,350 through 1,394 is a promotion PASS with a performance warning; below
1,350 is a hard failure. The 1,395 target is retained, but the only prior v20
measurement was 1,432 with CUDA-graph diagnostics enabled, so its 2.7% margin
is not a defensible single-sample correctness gate.

### Offload stability

Run the existing 16 x 50k unique-prefix overlapping load:

- 16/16 requests complete with `finish_reason=stop`;
- `RestartCount` stays zero;
- container ID and `StartedAt` remain unchanged;
- no `_build_store_jobs`, assertion, OOM, or engine-death signature;
- record the count of `Created TieringOffloadingManager` log lines and list
  matching `_r[0-9]` namespaces; require one manager and `_r0` only;
- post-load health succeeds.

Do not artificially fill 64 GB of NVMe during this promotion boot. Confirm
that
the bounded manager initialized, files are being written in the fresh
namespace, and physical usage remains below the configured bound. The ordered
8 GiB capacity proof already established the eviction invariant. Record
explicitly that a full 64 GB turnover and restart-persistence/promotion test
remain unproven and are deferred; do not describe this boot as proving them.

## 6. Promotion

If every gate passes:

1. Keep this exact container running; do not restart into a different compose.
2. Record the final image digest, source commits, compose hash, KV pool,
   throughput cells, needle table, offload result, container ID, and
   `RestartCount`.
3. Push/publish the already-tested image digest if it was built locally.
4. Mark this compose and digest as the v20 rollback target.

Keep verifier routing on unmodified `auto` throughout promotion. PR #171 and
forced route modes remain a later scheduled A/B after production is stable;
that experiment does not belong in this boot.
