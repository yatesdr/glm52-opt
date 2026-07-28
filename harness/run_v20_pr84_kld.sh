#!/usr/bin/env bash
set -euo pipefail

# Fail-closed 2,048-token BF16-reference KLD gate for SparkInfer PR #84.
# This is a shallow no-regression check. Because index_topk=2,048, it is not
# selector-sensitive; pair it with the frozen 350k and randomized 475k gates.

MODE="${1:-}"
case "${MODE}" in
  smoke|full|summarize) ;;
  *)
    echo "usage: $0 {smoke|full|summarize}" >&2
    exit 2
    ;;
esac

IMAGE="${IMAGE:-glm52-serve:v20-20260726-oldest-boundary-pr-candidate}"
MODEL_DIR="${MODEL_DIR:-/home/claude/LLM/GLM-5.2-hybrid}"
KLD_ROOT="${KLD_ROOT:-/home/derek/kld-pr84}"
CACHE_ROOT="${CACHE_ROOT:-/home/derek/glm52-kld-pr84-cache}"
RUNS="${RUNS:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SUMMARIZER="${SUMMARIZER:-${KLD_ROOT}/summarize_v20_pr84_kld.py}"
RUNNER_FILE="${RUNNER_FILE:-prefill_kld_fallback.py}"

EXPECTED_IMAGE_FILE_SHA="b15bab73f1fcd6434f712f6fc99ec5369104969cb9157ae473926bf40d72e23b"
EXPECTED_RUNNER_SHA="${EXPECTED_RUNNER_SHA:-e3958eb8b2f603a8a33e42b851fbaaa0f059e16c69881610c0e6d8a7a7776341}"
EXPECTED_REF_SHA="87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063"
EXPECTED_MANIFEST_SHA="985120136741037918bcd4dc8da9813c1f6268b35a730302f99cf6b3eebb7606"

HF_OVERRIDES='{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
QUANT_CONFIG='{"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"},"ignore":["re:^model\\.layers\\.0\\.","re:.*\\.self_attn\\.indexer\\.","re:.*\\.mlp\\.gate$","model.layers.78.eh_proj","lm_head"]}'
LLM_EXTRA_JSON="$(QUANT_CONFIG="${QUANT_CONFIG}" python3 - <<'PY'
import json
import os

print(json.dumps({
    "decode_context_parallel_size": 1,
    "moe_backend": "b12x",
    "enforce_eager": True,
    "quantization_config": json.loads(os.environ["QUANT_CONFIG"]),
}, separators=(",", ":")))
PY
)"

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha() {
  local path="$1"
  local expected="$2"
  [[ -f "${path}" ]] || {
    echo "missing required file: ${path}" >&2
    exit 2
  }
  local actual
  actual="$(sha256_file "${path}")"
  [[ "${actual}" == "${expected}" ]] || {
    echo "SHA-256 mismatch: ${path}" >&2
    echo "expected=${expected}" >&2
    echo "actual=${actual}" >&2
    exit 2
  }
}

preflight() {
  [[ "${RUNS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "RUNS must be a positive integer" >&2
    exit 2
  }
  [[ -d "${MODEL_DIR}" ]] || {
    echo "missing model directory: ${MODEL_DIR}" >&2
    exit 2
  }
  [[ "${RUNNER_FILE}" != */* ]] || {
    echo "RUNNER_FILE must be a basename under KLD_ROOT" >&2
    exit 2
  }
  require_sha "${KLD_ROOT}/${RUNNER_FILE}" "${EXPECTED_RUNNER_SHA}"
  require_sha \
    "${KLD_ROOT}/reference-logits/logits_0.safetensors" \
    "${EXPECTED_REF_SHA}"
  require_sha \
    "${KLD_ROOT}/reference-logits/manifest.json" \
    "${EXPECTED_MANIFEST_SHA}"
  [[ -f "${SUMMARIZER}" ]] || {
    echo "missing summarizer: ${SUMMARIZER}" >&2
    exit 2
  }

  docker image inspect "${IMAGE}" >/dev/null
  local image_file_sha
  image_file_sha="$(
    docker run --rm --entrypoint sha256sum "${IMAGE}" \
      /opt/venv/lib/python3.12/site-packages/sparkinfer/attention/nsa_indexer/tiled_topk.py |
      awk '{print $1}'
  )"
  [[ "${image_file_sha}" == "${EXPECTED_IMAGE_FILE_SHA}" ]] || {
    echo "unexpected tiled_topk.py in ${IMAGE}: ${image_file_sha}" >&2
    exit 2
  }

  local compute_pids
  compute_pids="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
      sed '/^[[:space:]]*$/d' |
      sort -u
  )"
  [[ -z "${compute_pids}" ]] || {
    echo "GPUs are not free; refusing to overlap KLD with compute PIDs:" >&2
    echo "${compute_pids}" >&2
    exit 3
  }

  mkdir -p "${KLD_ROOT}/results" "${CACHE_ROOT}"
}

run_one() {
  local policy="$1"
  local run="$2"
  local out="${KLD_ROOT}/results/${policy}/run${run}"
  local name="glm52-pr84-kld-${policy//_/-}-run${run}"
  local cache="${CACHE_ROOT}/${policy}"

  if [[ -f "${out}/summary.json" ]]; then
    python3 "${SUMMARIZER}" validate "${out}" >/dev/null
    echo "KLD_SKIP_VALID policy=${policy} run=${run} out=${out}"
    return
  fi

  mkdir -p "${out}" "${cache}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  cat >"${out}/config.json" <<EOF
{
  "policy": "${policy}",
  "run": ${run},
  "image": "${IMAGE}",
  "image_id": "$(docker image inspect "${IMAGE}" --format '{{.Id}}')",
  "model_dir": "${MODEL_DIR}",
  "reference_logits": "${KLD_ROOT}/reference-logits",
  "reference_logits_sha256": "${EXPECTED_REF_SHA}",
  "runner_sha256": "${EXPECTED_RUNNER_SHA}",
  "context_length": 2048,
  "max_num_batched_tokens": 512,
  "gpu_memory_utilization": ${GPU_MEMORY_UTILIZATION},
  "kv_cache_dtype": "nvfp4_ds_mla",
  "quantization": "nvfp4_nf3_hybrid",
  "online_quantization": "nf3-mxfp8",
  "kv_fp8_rope": true,
  "tensor_parallel_size": 4,
  "decode_context_parallel_size": 1,
  "enforce_eager": true
}
EOF

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) KLD_START policy=${policy} run=${run}"
  if ! docker run --rm --name "${name}" \
    --gpus all \
    --ipc=host \
    --network=host \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=1048576:1048576 \
    -v "${MODEL_DIR}:/model:ro" \
    -v "${KLD_ROOT}:/kld:rw" \
    -v "${cache}:/cache:rw" \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
    -e CUTE_DSL_ARCH=sm_120a \
    -e TORCH_CUDA_ARCH_LIST=12.0a \
    -e OMP_NUM_THREADS=16 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e SAFETENSORS_FAST_GPU=1 \
    -e INSTANTTENSOR_BACKEND=BUFFERED \
    -e HF_HOME=/kld/huggingface \
    -e XDG_CACHE_HOME=/cache/jit \
    -e TRITON_CACHE_DIR=/cache/jit/triton \
    -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    -e SPARKINFER_COMPILE_CACHE_DIR=/cache/sparkinfer \
    -e PYTHONPATH=/kld/pydeps \
    -e KV_FP8_ROPE=1 \
    -e NF3_GRID188=1 \
    -e VLLM_NF3_GRID188_DECODE=1 \
    -e VLLM_USE_B12X_FP8_GEMM=1 \
    -e VLLM_USE_B12X_MOE=1 \
    -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
    -e VLLM_USE_B12X_WO_PROJECTION=1 \
    -e VLLM_USE_B12X_MHC=1 \
    -e VLLM_USE_B12X_ABSORB_BMM=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e B12X_MOE_FORCE_A8=0 \
    -e B12X_MOE_FORCE_A16=1 \
    -e B12X_MLA_SM120_UNIFIED=1 \
    -e B12X_DENSE_SPLITK_TURBO=1 \
    -e B12X_W4A16_TC_DECODE=1 \
    -e B12X_W4A8_TINY_DECODE=1 \
    -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
    -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
    -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
    -e VLLM_PCIE_DMA_FP8=0 \
    -e B12X_PCIE_DMA_FP8=0 \
    -e NCCL_P2P_LEVEL=PXB \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_PROTO=LL,LL128,Simple \
    -e LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4 \
    -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4 \
    -e SPARKINFER_NSA_TOPK_SELECTION_POLICY="${policy}" \
    -e HF_OVERRIDES="${HF_OVERRIDES}" \
    -e LLM_EXTRA_JSON="${LLM_EXTRA_JSON}" \
    --entrypoint bash \
    "${IMAGE}" \
    -lc "set -euo pipefail; unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE; \
      /opt/venv/bin/python \"/kld/${RUNNER_FILE}\" \
        --model /model \
        --tokenizer /model \
        --reference-logits /kld/reference-logits \
        --context-length 2048 \
        --stride 512 \
        --max-windows 1 \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
        --dtype bfloat16 \
        --kv-cache-dtype nvfp4_ds_mla \
        --load-format instanttensor \
        --max-model-len 4096 \
        --max-num-batched-tokens 512 \
        --max-num-seqs 1 \
        --quantization nvfp4_nf3_hybrid \
        --attention-backend B12X_MLA_SPARSE \
        --hf-overrides \"\${HF_OVERRIDES}\" \
        --llm-extra-json \"\${LLM_EXTRA_JSON}\" \
        --kld-chunk-rows 32" \
    2>&1 | tee "${out}/prefill_dcp1.log"; then
    echo "KLD_RUN_FAILED policy=${policy} run=${run} out=${out}" >&2
    exit 1
  fi

  python3 "${SUMMARIZER}" validate "${out}"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) KLD_DONE policy=${policy} run=${run}"
}

if [[ "${MODE}" == "summarize" ]]; then
  python3 "${SUMMARIZER}" summarize "${KLD_ROOT}" --expected-runs "${RUNS}"
  exit
fi

preflight

if [[ "${MODE}" == "smoke" ]]; then
  run_one exact 1
  run_one oldest_boundary 1
  echo "KLD_SMOKE_PASS"
  exit
fi

for policy in exact oldest_boundary; do
  for run in $(seq 1 "${RUNS}"); do
    run_one "${policy}" "${run}"
  done
done
python3 "${SUMMARIZER}" summarize "${KLD_ROOT}" --expected-runs "${RUNS}"
