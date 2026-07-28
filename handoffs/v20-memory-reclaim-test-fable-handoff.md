# v20 memory-reclaim candidate — Fable handoff

Date: 2026-07-24  
Operator target: CN4  
Goal: one production-shaped boot that validates the newest Festr stack, restores
enough honest KV capacity for a 480k endpoint, and preserves i8-ring, DRAM
offload, and bounded NVMe eviction.

## 1. Exact candidate

Published base:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm992b874-sia93df67-fi801d57a-cu132-20260724
sha256:adddafd2b1749729fdf2d2ca23818c7c39f2a95e6fb05edd98657251913b83f2
vLLM:       992b874cf7ae504616bbb1d2d4f7a7355be6972b
SparkInfer: a93df671cc7b33734f499b57228e542c3d3c3697
```

The new SparkInfer revision includes the widened, exact long-context top-k
path. Do not replace its compiled artifacts or mount an older SparkInfer tree.

Integration source:

```text
https://github.com/yatesdr/vllm-opt
branch: integration/v20-memory-reclaim-20260724
code revision: 7373bb24c881fa05af57d7eaf8aa7b4e9f2d2ddb
```

The integration branch adds, in order:

1. PR #165 bounded filesystem-tier capacity.
2. The field-proven SM120/B12X native-MTP flattening gate and CPU coverage.
3. A conflict-resolved forward port of PR #154 over vLLM `992b874cf`.

The PR #154 port keeps Martin Vit's implementation and authorship. Its only
merge conflict was adjacent unit-test coverage; both old and new tests were
preserved. Production semantics remain the narrow PR #154 contract:
`B12X_MLA_SPARSE` releases MXFP8 `kv_b_proj` source storage only after the
materialized absorbed pair exists; other backends/formats remain default-off.

Image recipe:

```text
docker/Dockerfile.v20-memory-reclaim-test
```

Launch recipe:

```text
deploy/glm52-v20-memory-reclaim-test.yaml
```

## 2. What changed in the launch policy

The compose deliberately changes only three memory controls:

```text
VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=480000
cudagraph_capture_sizes=[1,2,4,8,16,32,64]
gpu_memory_utilization=0.976
```

Rationale:

- PR #154 previously reclaimed about 290 MiB/GPU with unchanged throughput,
  MTP acceptance, and KLD.
- Bounding the CKV arena to the actual endpoint saves about 38.9 MiB/GPU.
- Dropping intermediate graph sizes removes persistent graph allocations while
  retaining exact power-of-two service points through the required cap of 64.
- GMU 0.976 returns 0.4 percentage points of the 96 GiB device budget to
  runtime headroom compared with 0.980.

Do not add `VLLM_B12X_ABSORB_BMM=1`, disable CUDA-graph memory profiling, or
switch the large DCP backend merely to inflate the displayed KV pool. Those
would introduce performance/routing changes or make the accounting dishonest.

## 3. Build gate — no GPU boot

Clone or update the integration branch, then prove the pin before building:

```bash
git fetch origin integration/v20-memory-reclaim-20260724
git checkout integration/v20-memory-reclaim-20260724
test "$(git rev-parse HEAD)" = \
  7373bb24c881fa05af57d7eaf8aa7b4e9f2d2ddb
test -z "$(git status --porcelain)"
```

From that vLLM checkout, build using the Dockerfile from `glm52-opt`:

```bash
docker build \
  --file /path/to/glm52-opt/docker/Dockerfile.v20-memory-reclaim-test \
  --tag ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-memory-reclaim-test-20260724 \
  .
```

The Dockerfile is fail-closed. It checks all five input Python bytes, the
untouched MRV2 model-runner byte, and the `safe_mla_query_bmm` symbol before
copying anything. It then compiles and checks all five output bytes. Any
non-zero build result is a hard stop; do not weaken a hash to make it build.

Run the focused CPU proof inside the candidate before loading model weights:

```bash
docker run --rm --gpus all \
  --entrypoint /opt/venv/bin/python \
  --volume /path/to/glm52-opt/harness/v20_memory_reclaim_unit_proof.py:/proof.py:ro \
  ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-memory-reclaim-test-20260724 \
  /proof.py
```

Required result:

```json
{"non_b12x_owner_inert": true, "reload_rematerialization": true, "source_parameters_released": true, "verdict": "PASS"}
```

After success:

```bash
docker inspect \
  ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-memory-reclaim-test-20260724 \
  --format '{{json .Config.Labels}}'
docker push \
  ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-memory-reclaim-test-20260724
docker buildx imagetools inspect \
  ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-memory-reclaim-test-20260724
```

Replace the Compose image tag with the pushed `@sha256:...` digest before boot.

## 4. CN4 preflight

Set the existing CN4 paths; do not invent new storage locations:

```bash
export GLM52_MODEL_DIR=/actual/checkpoint/path
export GLM52_CACHE_DIR=/actual/writable/cache/path
export GLM52_NVME_DIR=/actual/ext4-or-xfs/nvme/path
```

Then check:

```bash
findmnt -T "$GLM52_NVME_DIR"
df -h "$GLM52_NVME_DIR" /dev/shm
nvidia-smi
docker compose \
  -f /path/to/glm52-opt/deploy/glm52-v20-memory-reclaim-test.yaml \
  config --quiet
```

Preflight fails if the NVMe filesystem is vfat, `/dev/shm` cannot hold the
64 GB DRAM tier, any GPU is missing, or Compose does not resolve.

## 5. One-boot acceptance

Launch once. The acceptance compose intentionally has `restart: "no"` and no
autoheal so a failed first process cannot consume a hidden retry.

```bash
docker compose \
  -f /path/to/glm52-opt/deploy/glm52-v20-memory-reclaim-test.yaml \
  up -d
```

Hard boot requirements:

- container identity and `StartedAt` stay unchanged; restart count remains 0;
- no OOM, illegal access, cuBLAS error, assertion, EngineDead, Xid, or dead
  worker;
- CUDA-graph profiling remains enabled;
- all profiling and production MTP captures complete;
- API becomes healthy and returns a finalized non-empty response;
- GPU KV pool is at least 480,000 tokens. The target is at least 500,000; a
  480k–499,999 result is usable but should be reported as thin, not silently
  retuned during this boot.

Record these exact memory lines:

```text
model weights / non-torch memory
MRV2 captured / retained / additional
Available KV cache memory
GPU KV cache size
Maximum concurrency at 480,000
```

Also capture the resolved engine configuration and prove:

```text
wire mode = i8_ring
CKV gather max = 480000
capture sizes = 1,2,4,8,16,32,64
GMU = 0.976
MNS = 16
max model length = 480000
```

## 6. Qualification on the same live process

Do not reboot between cells.

1. Cold prefill at 8k and 55k; require prefix-cache query deltas and zero hits.
2. Decode C1/C4/C8/C16 at ctx0, plus C16 at ctx16k and ctx50k.
3. Cold needles at 50k, 250k, 350k, and 475k. Search `content`, `reasoning`,
   `reasoning_content`, and the serialized assistant message for the needle,
   but require non-empty finalized `content` at every depth.
4. Bounded NVMe turnover using the existing ordered inotify monitor. Count
   completed plus temporary files and require high-water `<=64,000,000,000`.
5. 16×50k unique-prefix overlapping stress; all requests must finish, with no
   restart and no offload assertion.
6. Final health request on the same container identity.

Report measured prefill/decode, MTP acceptance, KV pool, DRAM/NVMe capacity
high-water, prefix-cache miss evidence, and all four needle rows. Do not claim
the expected reclaim until the measured before/after pool and resident-memory
lines support it.

## 7. Fail policy

Stop after the first hard boot failure. Do not run a GMU ladder in the same
window. Preserve the complete log, inspect JSON, image digest, file hashes,
container ID, `StartedAt`, and restart count, then return the evidence for
source diagnosis.
