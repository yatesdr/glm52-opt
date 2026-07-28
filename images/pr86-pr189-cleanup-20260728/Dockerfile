# syntax=docker/dockerfile:1.7
FROM glm52-v20-prod-exact-kv:20260728

COPY --from=sparkinfer sparkinfer/attention/_shared/mla/kv_cache.py \
    /opt/venv/lib/python3.12/site-packages/sparkinfer/attention/_shared/mla/kv_cache.py

COPY --from=vllm vllm/model_executor/layers/mla.py \
    /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mla.py
COPY --from=vllm vllm/model_executor/layers/mla_cache_format.py \
    /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mla_cache_format.py
COPY --from=vllm vllm/envs.py \
    /opt/venv/lib/python3.12/site-packages/vllm/envs.py
COPY --from=vllm vllm/v1/attention/backends/mla/b12x_mla_sparse.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py
COPY --from=vllm \
    vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py \
    /opt/venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py
COPY --from=vllm vllm/v1/kv_offload/config.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/kv_offload/config.py
COPY --from=vllm vllm/v1/kv_offload/file_mapper.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/kv_offload/file_mapper.py

LABEL local.glm52.dynamic.sparkinfer.cleanup.commit="0ddd13b4fdbb6a287581aec55fcf9dbbb7e52fd3" \
      local.glm52.dynamic.vllm.cleanup.commit="b57062274c3f53bec69b431bfae7230977f5f10c" \
      local.glm52.dynamic.cache-abi="server-static namespace keyed"
