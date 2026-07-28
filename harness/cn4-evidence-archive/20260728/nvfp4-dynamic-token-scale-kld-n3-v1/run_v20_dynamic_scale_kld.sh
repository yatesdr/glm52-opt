#!/usr/bin/env bash
set -euo pipefail

# Matched n=3 BF16-reference KLD comparison for NVFP4 MLA scale semantics.
# Both arms use one image and exact top-k. The only intentional difference is:
#   static_calibrated: the pinned per-layer outer-scales artifact
#   dynamic_per_token: the in-record per-token FP32 outer scale

MODE="${1:-}"
case "${MODE}" in
  smoke|full|summarize) ;;
  *)
    echo "usage: $0 {smoke|full|summarize}" >&2
    exit 2
    ;;
esac

IMAGE="${IMAGE:-glm52-serve:v20-nvfp4-dynamic-token-scale-20260727}"
MODEL_DIR="${MODEL_DIR:-/home/claude/LLM/GLM-5.2-hybrid}"
ASSET_ROOT="${ASSET_ROOT:-/home/derek/kld-pr84}"
KLD_ROOT="${KLD_ROOT:-/home/derek/kld-dynamic-scale-20260728}"
CACHE_ROOT="${CACHE_ROOT:-/home/derek/glm52-kld-dynamic-scale-cache}"
RUNS="${RUNS:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SUMMARIZER="${SUMMARIZER:-${KLD_ROOT}/summarize_v20_dynamic_scale_kld.py}"
COMPILE_PROVER="${COMPILE_PROVER:-${KLD_ROOT}/prove_nvfp4_writer_compile.py}"
RUNNER_FILE="${RUNNER_FILE:-prefill_kld_fallback_cleanup.py}"
STATIC_SCALES_FILE="/opt/vllm/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json"

EXPECTED_IMAGE_ID="sha256:db82fdcb5756d4a547853ba1330538bdd8a3dc0c6443c29bc49ba77b69b51cd1"
EXPECTED_RUNNER_SHA="d1dc1a63b9889e881f3bd899638d0ec65a1a1079132f6a207a600d9cba845405"
EXPECTED_REF_SHA="87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063"
EXPECTED_MANIFEST_SHA="985120136741037918bcd4dc8da9813c1f6268b35a730302f99cf6b3eebb7606"
EXPECTED_STATIC_SCALES_SHA="efd7e23ac1ace6da9dcd9046c46bca5cca68ed5e89cd648b5f8bc1d51eafebb2"
EXPECTED_TOPK_SHA="284bd167a971cc6c992c8b2b3ce120000185ef6ffe93be845036e098bfc834f2"

HF_OVERRIDES='{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
QUANT_CONFIG='{"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"},"ignore":["re:^model\\.layers\\.0\\.","re:.*\\.self_attn\\.indexer\\.","re:.*\\.mlp\\.gate$","model.layers.78.eh_proj","lm_head"]}'
LLM_EXTRA_JSON="$(
  QUANT_CONFIG="${QUANT_CONFIG}" python3 - <<'PY'
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
  require_sha "${ASSET_ROOT}/${RUNNER_FILE}" "${EXPECTED_RUNNER_SHA}"
  require_sha \
    "${ASSET_ROOT}/reference-logits/logits_0.safetensors" \
    "${EXPECTED_REF_SHA}"
  require_sha \
    "${ASSET_ROOT}/reference-logits/manifest.json" \
    "${EXPECTED_MANIFEST_SHA}"
  [[ -f "${SUMMARIZER}" ]] || {
    echo "missing summarizer: ${SUMMARIZER}" >&2
    exit 2
  }
  [[ -f "${COMPILE_PROVER}" ]] || {
    echo "missing compile prover: ${COMPILE_PROVER}" >&2
    exit 2
  }

  local image_id
  image_id="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
  [[ "${image_id}" == "${EXPECTED_IMAGE_ID}" ]] || {
    echo "unexpected image ID: ${image_id}" >&2
    exit 2
  }
  docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}' \
    | grep -q '"local.glm52.dynamic.sparkinfer.commit":"0d9aead9"'
  docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}' \
    | grep -q '"local.glm52.dynamic.vllm.commit":"91dff5a9"'

  local image_hashes
  image_hashes="$(
    docker run --rm --entrypoint sha256sum "${IMAGE}" \
      "${STATIC_SCALES_FILE}" \
      /opt/venv/lib/python3.12/site-packages/sparkinfer/attention/nsa_indexer/tiled_topk.py
  )"
  grep -q "^${EXPECTED_STATIC_SCALES_SHA}  ${STATIC_SCALES_FILE}$" \
    <<<"${image_hashes}"
  grep -q "^${EXPECTED_TOPK_SHA}  /opt/venv/lib/python3.12/site-packages/sparkinfer/attention/nsa_indexer/tiled_topk.py$" \
    <<<"${image_hashes}"

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

write_compile_proof() {
  local cache="$1"
  local expected="$2"
  local output="$3"
  docker run --rm \
    -v "${cache}:/scan:ro" \
    -v "${COMPILE_PROVER}:/prove_nvfp4_writer_compile.py:ro" \
    --entrypoint /opt/venv/bin/python \
    "${IMAGE}" \
    /prove_nvfp4_writer_compile.py /scan "${expected}" >"${output}"
}

run_one() {
  local policy="$1"
  local run="$2"
  local out="${KLD_ROOT}/results/${policy}/run${run}"
  local name="glm52-dynamic-scale-kld-${policy//_/-}-run${run}"
  local cache="${CACHE_ROOT}/${policy}/run${run}"
  local dynamic_flag static_file static_sha expected_compile
  local -a scale_env

  case "${policy}" in
    static_calibrated)
      dynamic_flag=false
      static_file="${STATIC_SCALES_FILE}"
      static_sha="${EXPECTED_STATIC_SCALES_SHA}"
      expected_compile=false
      scale_env=(-e "VLLM_NVFP4_MLA_SCALES_FILE=${STATIC_SCALES_FILE}")
      ;;
    dynamic_per_token)
      dynamic_flag=true
      static_file=null
      static_sha=null
      expected_compile=true
      scale_env=(-e VLLM_NVFP4_MLA_DYNAMIC_SCALE=1)
      ;;
    *)
      echo "invalid policy: ${policy}" >&2
      exit 2
      ;;
  esac

  if [[ -f "${out}/summary.json" ]]; then
    python3 "${SUMMARIZER}" validate "${out}" >/dev/null
    echo "KLD_SKIP_VALID policy=${policy} run=${run} out=${out}"
    return
  fi

  mkdir -p "${out}" "${cache}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  python3 - "${out}/config.json" \
    "${policy}" "${run}" "${IMAGE}" "${EXPECTED_IMAGE_ID}" \
    "${MODEL_DIR}" "${ASSET_ROOT}/reference-logits" \
    "${dynamic_flag}" "${static_file}" "${static_sha}" <<'PY'
import json
import pathlib
import sys

(
    output, policy, run, image, image_id, model_dir, reference_logits,
    dynamic_flag, static_file, static_sha,
) = sys.argv[1:]
config = {
    "policy": policy,
    "run": int(run),
    "image": image,
    "image_id": image_id,
    "model_dir": model_dir,
    "reference_logits": reference_logits,
    "reference_logits_sha256": "87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063",
    "reference_manifest_sha256": "985120136741037918bcd4dc8da9813c1f6268b35a730302f99cf6b3eebb7606",
    "runner_sha256": "d1dc1a63b9889e881f3bd899638d0ec65a1a1079132f6a207a600d9cba845405",
    "context_length": 2048,
    "max_num_batched_tokens": 512,
    "gpu_memory_utilization": 0.90,
    "kv_cache_dtype": "nvfp4_ds_mla",
    "quantization": "nvfp4_nf3_hybrid",
    "online_quantization": "nf3-mxfp8",
    "kv_fp8_rope": True,
    "tensor_parallel_size": 4,
    "decode_context_parallel_size": 1,
    "enforce_eager": True,
    "selector_policy": "exact",
    "dynamic_per_token": dynamic_flag == "true",
    "static_scales_file": None if static_file == "null" else static_file,
    "static_scales_sha256": None if static_sha == "null" else static_sha,
}
pathlib.Path(output).write_text(json.dumps(config, indent=2) + "\n")
PY

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
    -v "${ASSET_ROOT}:/kld-assets:ro" \
    -v "${ASSET_ROOT}/huggingface:/hf-cache:rw" \
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
    -e HF_HOME=/hf-cache \
    -e XDG_CACHE_HOME=/cache/jit \
    -e TRITON_CACHE_DIR=/cache/jit/triton \
    -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    -e SPARKINFER_COMPILE_CACHE_DIR=/cache/sparkinfer \
    -e PYTHONPATH=/kld-assets/pydeps \
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
    -e SPARKINFER_NSA_TOPK_SELECTION_POLICY=exact \
    -e HF_OVERRIDES="${HF_OVERRIDES}" \
    -e LLM_EXTRA_JSON="${LLM_EXTRA_JSON}" \
    "${scale_env[@]}" \
    --entrypoint bash \
    "${IMAGE}" \
    -lc "set -euo pipefail; unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE; \
      /opt/venv/bin/python \"/kld-assets/${RUNNER_FILE}\" \
        --model /model \
        --tokenizer /model \
        --reference-logits /kld-assets/reference-logits \
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

  write_compile_proof \
    "${cache}" "${expected_compile}" "${out}/writer-compile-proof.json"
  python3 "${SUMMARIZER}" validate "${out}"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) KLD_DONE policy=${policy} run=${run}"
}

if [[ "${MODE}" == "summarize" ]]; then
  python3 "${SUMMARIZER}" summarize "${KLD_ROOT}" --expected-runs "${RUNS}"
  exit
fi

preflight

if [[ "${MODE}" == "smoke" ]]; then
  run_one static_calibrated 1
  run_one dynamic_per_token 1
  echo "KLD_SMOKE_PASS"
  exit
fi

for policy in static_calibrated dynamic_per_token; do
  for run in $(seq 1 "${RUNS}"); do
    run_one "${policy}" "${run}"
  done
done
python3 "${SUMMARIZER}" summarize "${KLD_ROOT}" --expected-runs "${RUNS}"
