# GLM-5.2 v19 production serving image + compose

Drop-in serving for the GLM-5.2 MXFP8-NVFP4-NF3 hybrid checkpoint on 4× RTX PRO 6000 (96 GB) nodes.

## Image
`ghcr.io/yatesdr/glm52-serve:v19-20260719` — **public**.
- **Base (pinned by digest):** `voipmonitor/vllm@sha256:41078d5f…` = `gilded-gnosis-v19-vllm7ea567a-b12x4cfa530-fi801d57a-cu132-20260718` (David / local-inference-lab).
- **Only delta vs stock v19:** `serve-glm52-v16.sh` patched to
  1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` — required for KV `OffloadingConnector` compatibility.
  2. build + append `--kv-transfer-config` from `OFFLOAD_ENABLE` / `DRAM_OFFLOAD_BYTES` (or an explicit `KV_TRANSFER_CONFIG`).
  3. clean stale `/dev/shm/vllm_offload_*.mmap` on start.

## Validated config (defaults)
TP4 / DCP4 / MTP3, MNBT3072, `max-model-len` 480000, `KV_FP8_ROPE=1` (368 B compact KV), reworked hybrid one-grid
Grid188, `--load-format safetensors`, GMU **0.975**, prefix caching ON, `restart: unless-stopped`.
- **GPU KV pool ≈ 680–710k tokens** (best-ever; recovers + exceeds old prod).
- **KV offload:** GPU hot tier + **64 GB DRAM warm tier** (`OffloadingConnector` / `TieringOffloadingSpec`, ~2.0M
  tokens). NVMe (3rd) tier intentionally **not** enabled — it faults (`OSError EFAULT`) on v19's newer `fs` tier API.

## Deploy on a new node
```bash
# 1. copy the checkpoint dir to the node
# 2. (optional) create .env to override any tunable:
cat > .env <<EOF
MODEL_DIR=/path/to/GLM-5.2-hybrid
GPU_MEMORY_UTILIZATION=0.975
MAX_NUM_SEQS=16
DRAM_OFFLOAD_BYTES=64000000000
PORT=5001
EOF
# 3. bring it up
docker compose up -d
```
Tunables (env / `.env`): `MODEL_DIR`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `DRAM_OFFLOAD_BYTES`,
`OFFLOAD_ENABLE` (0 disables offload), `PORT`, `SERVED_MODEL_NAME`, `CONTAINER_NAME`.

## Known caveats
- **Offload connector under extreme load:** at ~32 concurrent streams + eviction, vLLM's `OffloadingConnector` can
  hit `AssertionError` in `_build_store_jobs` and the engine restarts (`restart: unless-stopped` recovers it).
  Stable under typical load (≤8 concurrent). Set `OFFLOAD_ENABLE=0` to run GPU-only if you ever push very high
  concurrency before the upstream connector bug is fixed.
- **First boot is slow** (~15 min: safetensors weight load + compile). Subsequent boots reuse the AOT compile cache.
- **Rollback:** old prod (`davidyoung/…lowbit-kv:v1.3` + `glm52-prodshape` compose) is retained.
