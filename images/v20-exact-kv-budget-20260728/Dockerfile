FROM ghcr.io/yatesdr/glm52-serve@sha256:fa6365fba88eee8b34b8e6d14dedc79eb3f43bda2bb787d8e71b014421bcd929

COPY launcher-kv-cache-memory-bytes.patch /tmp/launcher-kv-cache-memory-bytes.patch

RUN patch --batch --forward --fuzz=0 -p1 -d / \
      < /tmp/launcher-kv-cache-memory-bytes.patch
RUN bash -n /usr/local/bin/serve-glm52-v16.sh
RUN grep -F -- '--kv-cache-memory-bytes' /usr/local/bin/serve-glm52-v16.sh

LABEL org.opencontainers.image.description="GLM-5.2 v20 dynamic NVFP4 production image with fail-closed exact KV-byte launcher input"
LABEL local.glm52.kv-cache-memory-input="KV_CACHE_MEMORY_BYTES"

