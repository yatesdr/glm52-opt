# GLM-5.2 MXFP8-NVFP4-NF3 hybrid serving image.
# Stock local-inference-lab v19 base (pinned by digest) + one patched serve script for KV-offload compatibility.
# Base tag: voipmonitor/vllm:gilded-gnosis-v19-vllm7ea567a-b12x4cfa530-fi801d57a-cu132-20260718
FROM voipmonitor/vllm@sha256:41078d5fac36b802e0f3798057a4b383c24296cb8570f6f4fc138c34d8cfa303
# Only delta vs stock v19: serve-glm52-v16.sh patched to
#  (a) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False  (KV OffloadingConnector compatibility)
#  (b) build+append --kv-transfer-config from OFFLOAD_ENABLE / DRAM_OFFLOAD_BYTES (or explicit KV_TRANSFER_CONFIG)
#  (c) clean stale /dev/shm/vllm_offload_*.mmap on start
COPY serve-glm52-v16.sh /usr/local/bin/serve-glm52-v16.sh
RUN chmod +x /usr/local/bin/serve-glm52-v16.sh
LABEL org.opencontainers.image.title="glm52-serve" \
      org.opencontainers.image.description="GLM-5.2 MXFP8-NVFP4-NF3 hybrid TP4/DCP4 serving (v19 base + offload serve patch)" \
      org.opencontainers.image.source="https://github.com/yatesdr/glm52-opt" \
      org.opencontainers.image.base.name="voipmonitor/vllm:gilded-gnosis-v19-vllm7ea567a-b12x4cfa530-fi801d57a-cu132-20260718"
