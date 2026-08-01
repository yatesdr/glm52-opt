# Production-candidate overlay for GLM-5.2 EXL3/TR3 3.36 bpw on CN4's
# four-GPU 2+2 PCIe topology.
#
# Immutable base: Gilded Gnosis v20 r17. The base preserves vLLM #222,
# SparkInfer #105/#110, its audited source composition, launchers, LMCache,
# and cache ABI. This overlay replaces exactly five runtime files:
#   * r17-lifecycle i8_hier PCIe collective (Python + packaged CUDA source)
#   * bounded block-64 one-grid mixed-K EXL3 prefill planning
#   * grouped block-64 FC2 scheduling that shares each decoded weight fragment
#     across a pair of independent, exact M8 arithmetic tiles
FROM voipmonitor/vllm@sha256:d1008eb2bce2947110010fcf52b715b49d54ed3bf62a6b1e0a0b698774157727

COPY exl3.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py
COPY pcie_dma.py /opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/pcie_dma.py
COPY pcie_dma.cu /opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/pcie_dma.cu
COPY kernel.py /opt/venv/lib/python3.12/site-packages/sparkinfer/moe/_shared/kernels/w4a16/kernel.py
COPY mixed_trellis.py /opt/venv/lib/python3.12/site-packages/sparkinfer/moe/_shared/kernels/w4a16/mixed_trellis.py
COPY r17-tr3-336-installed-files.sha256 /usr/share/glm52-r17-overlay/installed-files.sha256
COPY r17-tr3-336-overlay-provenance.json /usr/share/glm52-r17-overlay/provenance.json

# r17's helper is deliberately fail-closed for unknown transports. Extend only
# the two GLM-5.2 transport allowlists; the shared runtime-env helper already
# propagates F8_DMA unchanged to vLLM, B12X, and SparkInfer.
RUN set -eux; \
    old='0|ag|ring|a2a|i8|i8_ring|i8_a2a|mx|mx_ring|mx_a2a'; \
    new='0|ag|ring|a2a|i8|i8_ring|i8_hier|i8_a2a|mx|mx_ring|mx_a2a'; \
    for script in /usr/local/bin/serve-glm52-v16.sh /usr/local/bin/serve-glm52-v19.sh; do \
      test "$(grep -Fc "$old" "$script")" -eq 1; \
      sed -i "s/$old/$new/" "$script"; \
      test "$(grep -Fc "$new" "$script")" -eq 1; \
    done; \
    sha256sum -c /usr/share/glm52-r17-overlay/installed-files.sha256; \
    /opt/venv/bin/python -m py_compile \
      /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py \
      /opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/pcie_dma.py \
      /opt/venv/lib/python3.12/site-packages/sparkinfer/moe/_shared/kernels/w4a16/kernel.py \
      /opt/venv/lib/python3.12/site-packages/sparkinfer/moe/_shared/kernels/w4a16/mixed_trellis.py

LABEL org.opencontainers.image.title="GLM-5.2 r17 TR3-3.36 exact paired-FC2 prefill + i8_hier" \
      org.opencontainers.image.description="Gilded Gnosis r17 with bounded block-64 mixed-K one-grid EXL3 prefill, bit-exact paired-M8 FC2 weight reuse, and topology-specific r17-lifecycle i8_hier" \
      org.opencontainers.image.base.name="voipmonitor/vllm@sha256:d1008eb2bce2947110010fcf52b715b49d54ed3bf62a6b1e0a0b698774157727" \
      local-inference.overlay.cache-namespace="r17-tr3-336-onegrid-f0d1f989-i8hier-9c792af-v4" \
      local-inference.overlay.vllm.commit="f0d1f989396aa7acebf3c1ad3704131d60c8a80e" \
      local-inference.overlay.sparkinfer.commit="9c792afbd1fe29fd46efda8b36374d947de4e109" \
      local-inference.overlay.runtime-files="5" \
      local-inference.overlay.superseded-overlays="none"
