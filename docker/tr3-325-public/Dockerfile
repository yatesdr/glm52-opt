# syntax=docker/dockerfile:1
#
# Public GLM-5.2 EXL3/TR3 3.25-bpw validation image.
#
# Model weights (mounted at /model; not redistributed in this image):
# https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw
#
# Pedigree:
#   upstream image:
#     voipmonitor/vllm:gilded-gnosis-v20-vllm34f26c2-side7739a-fi801d57a-cu132-20260728-r9
#     sha256:8246024490670e43af6ccdc3df9c6dd0a084119f4507b7ac35a86f5a1c6c33c3
#   upstream vLLM integration:     34f26c2848e4ec99c4bec80bf8d46bb5f5de367e
#   upstream SparkInfer integration:
#                                  de7739aa7ebc8a52eb1f1997367eee1d0a6bab79
#   included by upstream r9:
#     vLLM #189                    b57062274c3f53bec69b431bfae7230977f5f10c
#     SparkInfer #86               0ddd13b4fdbb6a287581aec55fcf9dbbb7e52fd3
#     SparkInfer #87               c24a01441f87c0eccd1390aee2a170c535340894
#   overlays added here:
#     mixed-K EXL3 v6              sha256:f337457c29063f0467516a69ded808196e029c65f3359493226dddd4e3f422a1
#       (v6 retains v4's persistent output buffer, uses one exact routed-MoE
#        pack/launch at M<=8, and reuses tier-local rotation tables without a
#        duplicate global copy; larger decode and all prefill stay serial)
#     SparkInfer fused mixed-K v1  source files pinned below
#       (exact K3/K4 hybrid Trellis path; public planned API)
#     SparkInfer profiled exact-fold scratch v1
#       (reserves the adaptive two-level top-k candidate workspace before
#        vLLM sizes the KV pool; removes request-time fold allocations)
#     B12X PCIe DMA fusion v1      sha256:330db3e8d26213ce6a09328e69e9fd217b92e9e6d2fd0d78d1c49e513f71d107
#       (production-size fused AR+RMS reduces in-place; generic AR unchanged)
#     bounded filesystem PR #165   a1f5cc6cd0bcbcedfc607f98afbd8883ea3a3d5a
#
# The mixed-K path is required by this checkpoint's per-expert 3/4-bpw
# layout. PR #165 makes the optional filesystem/NVMe tier honor its configured
# byte bound. Every runtime default below remains overridable by Compose.

ARG BASE_IMAGE=voipmonitor/vllm@sha256:8246024490670e43af6ccdc3df9c6dd0a084119f4507b7ac35a86f5a1c6c33c3
FROM ${BASE_IMAGE}

USER root

ARG BASE_EXL3_SHA256=9236ed6baa84ba4205fc328fd0c9cfc3f6cb3db07d1640deead88c207d3671b7
ARG MIXK_PATCH_SHA256=f337457c29063f0467516a69ded808196e029c65f3359493226dddd4e3f422a1
ARG PATCHED_EXL3_SHA256=ddcb1ae7f21c72119a0c24c376ed3f9d2f09e49a49f272af25e6cfcf4fc98ec5
ARG BASE_W4A16_KERNEL_SHA256=d6a03340a22ae4b9d34e964e8030c67e2dee9b8fbcc0614cd325a0b18b59f4fb
ARG PATCHED_W4A16_KERNEL_SHA256=8fb83a73be4a3ea7ad0b2093cad674130dee84183b01c8d8187249b87be0feae
ARG BASE_TRELLIS_INIT_SHA256=21c109ad779a365ca2a03c6744680e6a3efaef14c92e5b94de5db48d7866102a
ARG PATCHED_TRELLIS_INIT_SHA256=dc977562db1dd394cef3df8163daf0a0eabe8057452f7659d39e1e03b9155427
ARG BASE_TRELLIS_IMPL_SHA256=bd933ed2a4913bc56bb6b2ba8d42276482905d559affa92fe9f5668260d1d8b7
ARG PATCHED_TRELLIS_IMPL_SHA256=17a58dc97f29ba4792a63af26e0dee30aac5e39d32ee562c4ee45c3b303c0ca9
ARG BASE_TRELLIS_API_SHA256=4e39ce79d8a741f4a61dcb316a52cca6e2d56d1dc4734d5bc07fe84a90bb2329
ARG PATCHED_TRELLIS_API_SHA256=33a910eb86648ef333d374bfd283cbd06fbcb4e1777b910346a949a4d331c929
ARG BASE_INDEXER_PAGED_SHA256=b4be8ba9cad3ad08c724bc445f23bd3c10bd17ae43478e79cea597976fc53972
ARG PATCHED_INDEXER_PAGED_SHA256=c1f01f3fbc7731e82f69af7d9ec9cf100bf9650a7728ad3b1383e49a85b34697
ARG BASE_INDEXER_SCRATCH_SHA256=8d4cd238be5063c0c7d56f9337edbe12b5207708ed74940377ed3c4672549736
ARG PATCHED_INDEXER_SCRATCH_SHA256=236d287580a07b618ae2385184a9cb970b78bc12751104339b2d8f401e6aba3a
ARG INDEXER_FOLD_POLICY_SHA256=44138ebd939b56fa732322f84a89cec47d7b1190ae25331500c091db6f5bee88
ARG BASE_CUSTOM_ALLREDUCE_SHA256=ac3d5dcd4bf1f933e98576ca2c0c0ff64a9436d27c1d790c144012346f6d43b5
ARG PCIE_FUSION_PATCH_SHA256=330db3e8d26213ce6a09328e69e9fd217b92e9e6d2fd0d78d1c49e513f71d107
ARG PATCHED_CUSTOM_ALLREDUCE_SHA256=e68dfe9805ec9df2ca1acf7e6bc0835e2cd475993a0f7ccbdeca5686db3012bd
ARG BASE_PCIE_DMA_SHA256=245654168cecf9cc0263e6f7d9577bb098a59a941dece6880a73077a91269975
ARG BASE_FS_MANAGER_SHA256=ffb5904cba86f5703e2dfb274690e9da3b0f35e6130ab4f6730c43a573601b3d
ARG BOUNDED_FS_MANAGER_SHA256=653edbf4b393e2acd6204bf4664c300eaee9e959656040864491c94548b4cc60

COPY docker/tr3-325-public/patch_exl3_mixk.py /opt/glm52-overlays/patch_exl3_mixk.py
COPY docker/tr3-325-public/patch_pcie_dma_inplace_fusion.py /opt/glm52-overlays/patch_pcie_dma_inplace_fusion.py
COPY docker/tr3-325-public/bounded_fs_manager.py /opt/glm52-overlays/bounded_fs_manager.py
COPY docker/tr3-325-public/overlays/sparkinfer/moe/_shared/kernels/w4a16/kernel.py /opt/glm52-overlays/sparkinfer/w4a16_kernel.py
COPY docker/tr3-325-public/overlays/sparkinfer/moe/trellis_moe/__init__.py /opt/glm52-overlays/sparkinfer/trellis_init.py
COPY docker/tr3-325-public/overlays/sparkinfer/moe/trellis_moe/_impl.py /opt/glm52-overlays/sparkinfer/trellis_impl.py
COPY docker/tr3-325-public/overlays/sparkinfer/moe/trellis_moe/api.py /opt/glm52-overlays/sparkinfer/trellis_api.py
COPY docker/tr3-325-public/overlays/sparkinfer/attention/nsa_indexer/fold_policy.py /opt/glm52-overlays/sparkinfer/indexer_fold_policy.py
COPY docker/tr3-325-public/overlays/sparkinfer/attention/nsa_indexer/paged.py /opt/glm52-overlays/sparkinfer/indexer_paged.py
COPY docker/tr3-325-public/overlays/sparkinfer/attention/nsa_indexer/scratch.py /opt/glm52-overlays/sparkinfer/indexer_scratch.py

RUN set -eu; \
    exl3=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py; \
    custom_ar=/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/custom_all_reduce.py; \
    pcie_dma=/opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/pcie_dma.py; \
    w4a16_kernel=/opt/venv/lib/python3.12/site-packages/sparkinfer/moe/_shared/kernels/w4a16/kernel.py; \
    trellis_init=/opt/venv/lib/python3.12/site-packages/sparkinfer/moe/trellis_moe/__init__.py; \
    trellis_impl=/opt/venv/lib/python3.12/site-packages/sparkinfer/moe/trellis_moe/_impl.py; \
    trellis_api=/opt/venv/lib/python3.12/site-packages/sparkinfer/moe/trellis_moe/api.py; \
    indexer_fold_policy=/opt/venv/lib/python3.12/site-packages/sparkinfer/attention/nsa_indexer/fold_policy.py; \
    indexer_paged=/opt/venv/lib/python3.12/site-packages/sparkinfer/attention/nsa_indexer/paged.py; \
    indexer_scratch=/opt/venv/lib/python3.12/site-packages/sparkinfer/attention/nsa_indexer/scratch.py; \
    fs_manager=/opt/venv/lib/python3.12/site-packages/vllm/v1/kv_offload/tiering/fs/manager.py; \
    test "$(sha256sum "${exl3}" | awk '{print $1}')" = "${BASE_EXL3_SHA256}"; \
    test "$(sha256sum "${custom_ar}" | awk '{print $1}')" = "${BASE_CUSTOM_ALLREDUCE_SHA256}"; \
    test "$(sha256sum "${pcie_dma}" | awk '{print $1}')" = "${BASE_PCIE_DMA_SHA256}"; \
    test "$(sha256sum "${w4a16_kernel}" | awk '{print $1}')" = "${BASE_W4A16_KERNEL_SHA256}"; \
    test "$(sha256sum "${trellis_init}" | awk '{print $1}')" = "${BASE_TRELLIS_INIT_SHA256}"; \
    test "$(sha256sum "${trellis_impl}" | awk '{print $1}')" = "${BASE_TRELLIS_IMPL_SHA256}"; \
    test "$(sha256sum "${trellis_api}" | awk '{print $1}')" = "${BASE_TRELLIS_API_SHA256}"; \
    test ! -e "${indexer_fold_policy}"; \
    test "$(sha256sum "${indexer_paged}" | awk '{print $1}')" = "${BASE_INDEXER_PAGED_SHA256}"; \
    test "$(sha256sum "${indexer_scratch}" | awk '{print $1}')" = "${BASE_INDEXER_SCRATCH_SHA256}"; \
    test "$(sha256sum "${fs_manager}" | awk '{print $1}')" = "${BASE_FS_MANAGER_SHA256}"; \
    test "$(sha256sum /opt/glm52-overlays/patch_exl3_mixk.py | awk '{print $1}')" = "${MIXK_PATCH_SHA256}"; \
    test "$(sha256sum /opt/glm52-overlays/patch_pcie_dma_inplace_fusion.py | awk '{print $1}')" = "${PCIE_FUSION_PATCH_SHA256}"; \
    test "$(sha256sum /opt/glm52-overlays/bounded_fs_manager.py | awk '{print $1}')" = "${BOUNDED_FS_MANAGER_SHA256}"; \
    test "$(sha256sum /opt/glm52-overlays/sparkinfer/w4a16_kernel.py | awk '{print $1}')" = "${PATCHED_W4A16_KERNEL_SHA256}"; \
    test "$(sha256sum /opt/glm52-overlays/sparkinfer/trellis_init.py | awk '{print $1}')" = "${PATCHED_TRELLIS_INIT_SHA256}"; \
    test "$(sha256sum /opt/glm52-overlays/sparkinfer/trellis_impl.py | awk '{print $1}')" = "${PATCHED_TRELLIS_IMPL_SHA256}"; \
    test "$(sha256sum /opt/glm52-overlays/sparkinfer/trellis_api.py | awk '{print $1}')" = "${PATCHED_TRELLIS_API_SHA256}"; \
    test "$(sha256sum /opt/glm52-overlays/sparkinfer/indexer_fold_policy.py | awk '{print $1}')" = "${INDEXER_FOLD_POLICY_SHA256}"; \
    test "$(sha256sum /opt/glm52-overlays/sparkinfer/indexer_paged.py | awk '{print $1}')" = "${PATCHED_INDEXER_PAGED_SHA256}"; \
    test "$(sha256sum /opt/glm52-overlays/sparkinfer/indexer_scratch.py | awk '{print $1}')" = "${PATCHED_INDEXER_SCRATCH_SHA256}"; \
    install -m 0644 /opt/glm52-overlays/sparkinfer/w4a16_kernel.py "${w4a16_kernel}"; \
    install -m 0644 /opt/glm52-overlays/sparkinfer/trellis_init.py "${trellis_init}"; \
    install -m 0644 /opt/glm52-overlays/sparkinfer/trellis_impl.py "${trellis_impl}"; \
    install -m 0644 /opt/glm52-overlays/sparkinfer/trellis_api.py "${trellis_api}"; \
    install -m 0644 /opt/glm52-overlays/sparkinfer/indexer_fold_policy.py "${indexer_fold_policy}"; \
    install -m 0644 /opt/glm52-overlays/sparkinfer/indexer_paged.py "${indexer_paged}"; \
    install -m 0644 /opt/glm52-overlays/sparkinfer/indexer_scratch.py "${indexer_scratch}"; \
    /opt/venv/bin/python /opt/glm52-overlays/patch_exl3_mixk.py; \
    /opt/venv/bin/python /opt/glm52-overlays/patch_pcie_dma_inplace_fusion.py; \
    install -m 0644 /opt/glm52-overlays/bounded_fs_manager.py "${fs_manager}"; \
    test "$(sha256sum "${exl3}" | awk '{print $1}')" = "${PATCHED_EXL3_SHA256}"; \
    test "$(sha256sum "${custom_ar}" | awk '{print $1}')" = "${PATCHED_CUSTOM_ALLREDUCE_SHA256}"; \
    test "$(sha256sum "${pcie_dma}" | awk '{print $1}')" = "${BASE_PCIE_DMA_SHA256}"; \
    test "$(sha256sum "${w4a16_kernel}" | awk '{print $1}')" = "${PATCHED_W4A16_KERNEL_SHA256}"; \
    test "$(sha256sum "${trellis_init}" | awk '{print $1}')" = "${PATCHED_TRELLIS_INIT_SHA256}"; \
    test "$(sha256sum "${trellis_impl}" | awk '{print $1}')" = "${PATCHED_TRELLIS_IMPL_SHA256}"; \
    test "$(sha256sum "${trellis_api}" | awk '{print $1}')" = "${PATCHED_TRELLIS_API_SHA256}"; \
    test "$(sha256sum "${indexer_fold_policy}" | awk '{print $1}')" = "${INDEXER_FOLD_POLICY_SHA256}"; \
    test "$(sha256sum "${indexer_paged}" | awk '{print $1}')" = "${PATCHED_INDEXER_PAGED_SHA256}"; \
    test "$(sha256sum "${indexer_scratch}" | awk '{print $1}')" = "${PATCHED_INDEXER_SCRATCH_SHA256}"; \
    test "$(sha256sum "${fs_manager}" | awk '{print $1}')" = "${BOUNDED_FS_MANAGER_SHA256}"; \
    /opt/venv/bin/python -m py_compile "${exl3}" "${custom_ar}" "${w4a16_kernel}" "${trellis_init}" "${trellis_impl}" "${trellis_api}" "${indexer_fold_policy}" "${indexer_paged}" "${indexer_scratch}" "${fs_manager}"; \
    grep -Fq '# === EXL3-MIXK-PATCH v6 (exact fused M<=8, tier-local rotations) ===' "${exl3}"; \
    ! grep -Fq 'return accum.to(x.dtype)' "${exl3}"; \
    grep -Fq '# === B12X-PCIE-DMA-INPLACE-FUSION v1 ===' "${custom_ar}"; \
    grep -Fq 'self._pcie_dma.all_reduce(inp, out=inp)' "${custom_ar}"; \
    grep -Fq 'out = torch.empty_like(inp)' "${pcie_dma}"; \
    /opt/venv/bin/python -c \
      "from sparkinfer.moe import trellis_moe as t; assert all(hasattr(t,n) for n in ('HybridPlan','HybridBinding','plan_hybrid','bind_hybrid','build_tier_local_map','run_hybrid'))"; \
    /opt/venv/bin/python -c \
      "import inspect; from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager as m; assert 'max_cache_size_bytes' in inspect.signature(m).parameters"

# Quality/capacity posture: dynamic per-token NVFP4 MLA KV with FP8 RoPE.
# Production geometry: TP4/DCP4/MTP3, 3,072-token scheduler, graph width 32.
# Community-confirmed PCIe4/PCIe3 defaults avoid the owner-merge, query-split,
# indexer-shard, and CKV-prefetch regressions. The companion Compose leaves
# the host-specific NCCL P2P level and compressed wire selection explicit.
ENV MODEL_FAMILY=glm52-exl3 \
    MODEL=/model \
    MODEL_REVISION=e2b03576cd103e6ad322a1e091e5d0e2d0529073 \
    SERVED_MODEL_NAME=GLM-5.2-EXL3-TR3-3.25bpw \
    TP=4 \
    DCP=4 \
    MTP=3 \
    GRAPH=32 \
    MAX_NUM_SEQS=32 \
    MAX_BATCHED_TOKENS=3072 \
    MAX_MODEL_LEN=480000 \
    GPU_MEMORY_UTILIZATION=0.9690 \
    MOE_MODE=a16 \
    MOE_BACKEND=b12x \
    MTP_MOE_BACKEND=triton \
    MTP_DRAFT_SAMPLE_METHOD=greedy \
    QUANTIZATION=exl3 \
    ONLINE_QUANT=none \
    KV_CACHE_DTYPE=nvfp4_ds_mla \
    KV_FP8_ROPE=1 \
    VLLM_NVFP4_MLA_DYNAMIC_SCALE=1 \
    VLLM_NVFP4_MLA_SCALES_FILE="" \
    VLLM_EXL3_TRELLIS_MIN_M=1 \
    VLLM_EXL3_TRELLIS_MAX_M=32 \
    VLLM_EXL3_TRELLIS_BLOCK_M=8 \
    VLLM_EXL3_MIXK_FUSED_MAX_M=8 \
    VLLM_EXL3_PREFILL_TRELLIS=1 \
    VLLM_EXL3_PREFILL_BLOCK_M=64 \
    VLLM_EXL3_PREFILL_CHUNK=128 \
    DCP_QUERY_SPLIT=0 \
    DCP_CKV_GATHER=1 \
    DCP_CKV_GATHER_MAX_TOKENS=140000 \
    DCP_TOPK_OWNER_MERGE=0 \
    DCP_INDEXER_SHARDS=0 \
    DCP_CKV_PREFETCH_DEPTH=0 \
    DCP_CKV_PREFETCH_WORKSPACE_MIB=1024 \
    PCIE_DMA_MIN_BYTES=6MB \
    VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=120KB \
    VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB \
    SPARKINFER_INDEXER_TWO_LEVEL_FOLD=auto \
    SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB=256 \
    ASYNC_SCHEDULING=0 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_DEVICE_MAX_CONNECTIONS=32 \
    OMP_NUM_THREADS=16 \
    SAFETENSORS_FAST_GPU=1

ENTRYPOINT ["/usr/local/bin/serve-gilded-gnosis.sh"]

LABEL org.opencontainers.image.title="GLM-5.2 EXL3/TR3 3.25-bpw v20-r9 fused routed-MoE runtime" \
      org.opencontainers.image.description="Immutable v20-r9 plus fail-closed mixed-K EXL3 v6, exact low-M fused routed-MoE, allocation-free production PCIe DMA fusion, and reviewed TR3/NVFP4 defaults" \
      org.opencontainers.image.source="https://github.com/yatesdr/glm52-opt" \
      org.opencontainers.image.base.name="docker.io/voipmonitor/vllm" \
      org.opencontainers.image.base.digest="sha256:8246024490670e43af6ccdc3df9c6dd0a084119f4507b7ac35a86f5a1c6c33c3" \
      ai.glm52.model.repository="https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw" \
      ai.glm52.model.revision="e2b03576cd103e6ad322a1e091e5d0e2d0529073" \
      ai.glm52.exl3.mixk.patch="v6" \
      ai.glm52.exl3.mixk.patch.sha256="f337457c29063f0467516a69ded808196e029c65f3359493226dddd4e3f422a1" \
      ai.glm52.exl3.output.sha256="ddcb1ae7f21c72119a0c24c376ed3f9d2f09e49a49f272af25e6cfcf4fc98ec5" \
      ai.glm52.exl3.mixk.fused-max-m="8" \
      ai.glm52.sparkinfer.w4a16.kernel.sha256="8fb83a73be4a3ea7ad0b2093cad674130dee84183b01c8d8187249b87be0feae" \
      ai.glm52.sparkinfer.trellis.init.sha256="dc977562db1dd394cef3df8163daf0a0eabe8057452f7659d39e1e03b9155427" \
      ai.glm52.sparkinfer.trellis.impl.sha256="17a58dc97f29ba4792a63af26e0dee30aac5e39d32ee562c4ee45c3b303c0ca9" \
      ai.glm52.sparkinfer.trellis.api.sha256="33a910eb86648ef333d374bfd283cbd06fbcb4e1777b910346a949a4d331c929" \
      ai.glm52.sparkinfer.indexer-fold-scratch="profiled-v1" \
      ai.glm52.sparkinfer.indexer-fold-budget-mib="256" \
      ai.glm52.sparkinfer.indexer-fold-policy.sha256="44138ebd939b56fa732322f84a89cec47d7b1190ae25331500c091db6f5bee88" \
      ai.glm52.sparkinfer.indexer-paged.sha256="c1f01f3fbc7731e82f69af7d9ec9cf100bf9650a7728ad3b1383e49a85b34697" \
      ai.glm52.sparkinfer.indexer-scratch.sha256="236d287580a07b618ae2385184a9cb970b78bc12751104339b2d8f401e6aba3a" \
      ai.glm52.pcie.dma.fusion="inplace-v1" \
      ai.glm52.pcie.dma.fusion.patch.sha256="330db3e8d26213ce6a09328e69e9fd217b92e9e6d2fd0d78d1c49e513f71d107" \
      ai.glm52.pcie.dma.fusion.output.sha256="e68dfe9805ec9df2ca1acf7e6bc0835e2cd475993a0f7ccbdeca5686db3012bd" \
      ai.glm52.pcie.dma.generic.sha256="245654168cecf9cc0263e6f7d9577bb098a59a941dece6880a73077a91269975" \
      ai.glm52.offload.vllm.pr="165" \
      ai.glm52.offload.vllm.commit="a1f5cc6cd0bcbcedfc607f98afbd8883ea3a3d5a" \
      ai.glm52.offload.fs-manager.sha256="653edbf4b393e2acd6204bf4664c300eaee9e959656040864491c94548b4cc60" \
      ai.glm52.cuda.graph.max="32" \
      ai.glm52.kv.record="nvfp4_ds_mla:fp8-rope-368:dynamic-token-v1"
