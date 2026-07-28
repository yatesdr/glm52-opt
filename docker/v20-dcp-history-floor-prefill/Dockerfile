FROM voipmonitor/vllm@sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0

ARG ENVS_SHA256
ARG INDEXER_SHA256

COPY envs.py /opt/venv/lib/python3.12/site-packages/vllm/envs.py
COPY sparse_attn_indexer.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py

RUN test "$(sha256sum /opt/venv/lib/python3.12/site-packages/vllm/envs.py | cut -d' ' -f1)" = "${ENVS_SHA256}" \
 && test "$(sha256sum /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py | cut -d' ' -f1)" = "${INDEXER_SHA256}" \
 && /opt/venv/bin/python -m py_compile \
      /opt/venv/lib/python3.12/site-packages/vllm/envs.py \
      /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py

LABEL org.opencontainers.image.base.digest="sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0" \
      org.opencontainers.image.source.commit="0c79e41db41f250ccdfc4be92d171960a5787f73" \
      org.opencontainers.image.patch="v20-dcp-indexer-history-floor-prefill-owner"
