# syntax=docker/dockerfile:1

FROM davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv@sha256:99ae7b28bb7069b9f7a96f75ea815be56266d2cccf7808d4c497340bb8658bd5

LABEL org.opencontainers.image.title="glm52-opt" \
      org.opencontainers.image.description="Full-spec 480k GLM-5.2 TP4/DCP4 serving stack" \
      org.opencontainers.image.source="https://github.com/yatesdr/glm52-opt" \
      org.opencontainers.image.licenses="Apache-2.0"

# Exactly seven live v1.4 overlays. Do not copy the rest of the source bundle:
# three files were absent from production and four are superseded below.
COPY docker/overlays/v14/b12x/gemm/block_fp8_linear.py /opt/venv/lib/python3.12/site-packages/b12x/gemm/block_fp8_linear.py
COPY docker/overlays/v14/b12x/moe/fused/w4a16/kernel.py /opt/venv/lib/python3.12/site-packages/b12x/moe/fused/w4a16/kernel.py
COPY docker/overlays/v14/b12x/moe/fused/w4a16/route_pack.py /opt/venv/lib/python3.12/site-packages/b12x/moe/fused/w4a16/route_pack.py
COPY docker/overlays/v14/hybrid_loader.py /opt/venv/lib/python3.12/site-packages/hybrid_loader.py
COPY docker/overlays/v14/vllm/v1/attention/backends/utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/utils.py
COPY docker/overlays/v14/vllm/v1/sample/ops/topk_topp_sampler.py /opt/venv/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_sampler.py
COPY docker/overlays/v14/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py

# The five stage-3/phase-2 files present in the verified production container.
COPY patches/phase2-fullcontext/overlays/vllm/v1/attention/backends/mla/b12x_mla_sparse.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py
COPY patches/stage3-packed-ckv/overlays/vllm/v1/attention/ops/common.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/common.py
COPY patches/phase2-fullcontext/overlays/vllm/v1/worker/gpu_worker.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py
COPY patches/stage3-packed-ckv/overlays/vllm/model_executor/layers/attention/mla_attention.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/mla_attention.py
COPY patches/phase2-fullcontext/overlays/b12x/distributed/pcie_dma.py /opt/venv/lib/python3.12/site-packages/b12x/distributed/pcie_dma.py

COPY LICENSE /usr/share/licenses/glm52-opt/LICENSE
COPY docker/overlays/v14/NOTICE.md /usr/share/doc/glm52-opt/v14-overlays-NOTICE.md
COPY docker/overlay-md5.txt /usr/share/glm52-opt/overlay-md5.txt
RUN md5sum -c /usr/share/glm52-opt/overlay-md5.txt

# Verbatim defaults from the shipped 480k production service. In particular,
# PYTORCH_CUDA_ALLOC_CONF is intentionally absent: expandable segments are not
# part of the validated memory posture.
ENV PYTHONHASHSEED=0 \
    HYBRID_TC_DECODE=1 \
    HYBRID_NF3_TC_DECODE=1 \
    HYBRID_HETERO_DECODE=1 \
    HYBRID_TIER=both \
    HYBRID_KEPT=b12x_nf3 \
    HYBRID_NF3=b12x_nf3 \
    HYBRID_B12X_MAX_TOKENS=2048 \
    HYBRID_MXFP8_NATIVE=1 \
    B12X_EMPTY_CACHE_AFTER_WARMUP=1 \
    VLLM_USE_AOT_COMPILE=1 \
    VLLM_DISABLE_COMPILE_CACHE=1 \
    B12X_DCP_PREFILL_TRANSPORT=ckv \
    B12X_CKV_PHASE2_NCCL=1 \
    B12X_CKV_HEADROOM_PROBES=0 \
    B12X_DCP_GATHER_FP8=0 \
    B12X_DCP_RS_RING=0 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_DEVICE_MAX_CONNECTIONS=32 \
    CUTE_DSL_ARCH=sm_120a \
    OMP_NUM_THREADS=16 \
    SAFETENSORS_FAST_GPU=1 \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_LEVEL=SYS \
    NCCL_PROTO=LL,LL128,Simple \
    XDG_CACHE_HOME=/cache/jit \
    TRITON_CACHE_DIR=/cache/jit/triton \
    VLLM_USE_B12X_FP8_GEMM=1 \
    VLLM_USE_B12X_MOE=1 \
    VLLM_USE_B12X_SPARSE_INDEXER=1 \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    B12X_MOE_FORCE_A16=1 \
    VLLM_DCP_GLOBAL_TOPK=1 \
    VLLM_DCP_SHARD_DRAFT=1 \
    VLLM_ENABLE_PCIE_ALLREDUCE=1 \
    VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
    VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
    B12X_DENSE_SPLITK_TURBO=1 \
    B12X_MLA_SM120_PREFILL_F16_ROPE=1 \
    VLLM_PCIE_DMA_FP8=ag \
    B12X_PCIE_DMA_FP8=ag \
    VLLM_USE_B12X_DCP_A2A=1 \
    VLLM_DCP_A2A_MAX_TOKENS=16 \
    VLLM_DCP_A2A_LARGE_BACKEND=ag_rs \
    VLLM_DCP_PROJECT_BEFORE_MERGE=1 \
    VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS=1024 \
    SWEEP_PROFILE_NAME=v1_3_fast647_a2a16_dcp4_mnbt3072_mtp3

# Stale offload mmaps can consume the entire shared-memory filesystem. Keep
# this wrapper byte-for-byte equivalent to the production compose sequence.
ENTRYPOINT ["/bin/sh", "-c", "rm -f /dev/shm/vllm_offload_*.mmap; exec vllm serve \"$@\"", "--"]

CMD ["/model", "--served-model-name", "GLM-5.2", "--host", "0.0.0.0", "--port", "5001", "--trust-remote-code", "--tensor-parallel-size", "4", "--decode-context-parallel-size", "4", "--dcp-comm-backend", "a2a", "--dcp-kv-cache-interleave-size", "1", "--kv-cache-dtype", "nvfp4_ds_mla", "--kv-transfer-config", "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":56000000000,\"secondary_tiers\":[]}}", "--attention-backend", "B12X_MLA_SPARSE", "--moe-backend", "b12x", "--load-format", "safetensors", "-cc.pass_config.fuse_allreduce_rms=True", "--gpu-memory-utilization", "0.980", "--max-model-len", "480000", "--max-num-seqs", "8", "--max-num-batched-tokens", "3072", "--num-gpu-blocks-override", "2340", "--max-cudagraph-capture-size", "32", "--async-scheduling", "--enable-chunked-prefill", "--enable-prefix-caching", "--enable-auto-tool-choice", "--tool-call-parser", "glm47", "--reasoning-parser", "glm45", "--default-chat-template-kwargs", "{\"reasoning_effort\":\"high\"}", "--hf-overrides", "{\"use_index_cache\":true,\"index_topk_pattern\":\"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS\"}", "--speculative-config", "{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"moe_backend\":\"b12x\",\"draft_sample_method\":\"probabilistic\"}"]
