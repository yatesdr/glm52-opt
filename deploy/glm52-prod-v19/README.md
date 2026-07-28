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
Grid188, `--load-format safetensors`, **GMU 0.970**, prefix caching ON, `restart: unless-stopped`,
**KV offload ON** (GPU hot + 64 GB DRAM warm tier).
- **GPU KV pool = 644,864 tokens** (1.34× @ 480k) + DRAM warm tier. Full ladder (conc 1–16, 8k/55k prefill,
  needle @50k/@350k) passes clean: **0 errors, 0 engine restarts** across a ~20 min sustained run.
- **GMU is the ceiling with offload at 480k.** 0.975 boots but **OOMs on the first real prefill** (36 MiB short;
  offload forces `expandable_segments:False`, so ~400 MiB fragmentation is unreclaimable). ≤0.957 can't fit one
  480k request (boot crashloop). **0.970 is the validated ceiling** — KV fits 480k *and* offload GPU→DRAM stores
  keep enough headroom. If you raise GMU past 0.970, you must set `OFFLOAD_ENABLE=0` (GPU-only) or it OOMs.
- **The earlier conc-1 "store hang" was a symptom of 0.975 memory pressure, not a connector deadlock** — it does
  not reproduce at 0.970 (55k prefill + 16-way burst run clean). `kv_buffer_size` (offload GPU staging buffer) is
  left at the 1 GB default; shrinking it does **not** buy headroom (vLLM sizes the pool to the GMU target
  regardless) and only lowers DRAM-transfer throughput.
- **Follow-up to push pool higher:** offload requires `expandable_segments:False` (fragmentation cap). Making
  `expandable_segments` conditional on `OFFLOAD_ENABLE` only helps the GPU-only path. Getting well above ~670k
  *with* offload needs an upstream fix to the offload/expandable-segments incompatibility.

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
- **Offload + GMU are coupled — don't touch one without the other.** Offload is stable at GMU ≤ 0.970. Raising
  GMU above 0.970 with offload on OOMs on the first real prefill; if you need a bigger GPU pool than 0.970 gives
  (644,864 tokens), you must turn offload off (`OFFLOAD_ENABLE=0`) to run higher GMU GPU-only.
- **`_build_store_jobs` assertion at ~32 concurrent.** Not hit at our `MAX_NUM_SEQS=16` ceiling; `restart:
  unless-stopped` recovers it if you ever push there. Don't raise `MAX_NUM_SEQS` past ~16 with offload on.
- **First boot is slow** (~15 min: safetensors weight load + compile). Subsequent boots reuse the AOT compile cache.
- **GMU is context-length-sensitive.** If you change `MAX_MODEL_LEN`, re-tune `GPU_MEMORY_UTILIZATION`: too high →
  OOM on first real prefill, too low → boot fails with `estimated maximum model length < MAX_MODEL_LEN`.
- **Rollback:** old prod (`davidyoung/…lowbit-kv:v1.3` + `glm52-prodshape` compose) is retained.
