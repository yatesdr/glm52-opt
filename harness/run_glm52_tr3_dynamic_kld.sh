#!/usr/bin/env bash
set -euo pipefail

# Matched n=3 BF16-reference KLD qualification for the EXL3-TR3 candidate.
# This deliberately uses DCP1, MTP0, eager execution, exact selection, the
# same 2,048-token sample/reference as the NF3 dynamic-scale study, and one
# fresh compile/cache directory per run.

MODE="${1:-}"
case "${MODE}" in
  smoke|full|summarize) ;;
  *)
    echo "usage: $0 {smoke|full|summarize}" >&2
    exit 2
    ;;
esac

IMAGE="${IMAGE:-glm52-exl3-tr3-dynamic:20260728}"
EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:a5608e0b4a2fcdaec476de79fbe5cf2f6e9ce2ecf30bf2dfe0c1314d97c6666e}"
MODEL_DIR="${MODEL_DIR:-/home/derek/models/GLM-5.2-EXL3-TR3-3.0bpw}"
MODEL_REVISION="${MODEL_REVISION:-9297b9f1d53af5c67cffa01e30cc071a1ff7144b}"
ASSET_ROOT="${ASSET_ROOT:-/home/derek/kld-pr84}"
KLD_ROOT="${KLD_ROOT:-/home/derek/glm52-tr3-qualification-20260728/results/kld-n3}"
CACHE_ROOT="${CACHE_ROOT:-/home/derek/glm52-tr3-qualification-20260728/cache-kld}"
TOOLS_ROOT="${TOOLS_ROOT:-/home/derek/glm52-tr3-qualification-20260728/kld-tools}"
RUNS="${RUNS:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
RUNNER_FILE="${RUNNER_FILE:-prefill_kld_fallback_cleanup.py}"
SUMMARIZER="${SUMMARIZER:-${TOOLS_ROOT}/summarize_glm52_tr3_dynamic_kld.py}"
COMPILE_PROVER="${COMPILE_PROVER:-${TOOLS_ROOT}/prove_nvfp4_writer_compile.py}"

EXPECTED_RUNNER_SHA="d1dc1a63b9889e881f3bd899638d0ec65a1a1079132f6a207a600d9cba845405"
EXPECTED_REF_SHA="87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063"
EXPECTED_MANIFEST_SHA="985120136741037918bcd4dc8da9813c1f6268b35a730302f99cf6b3eebb7606"
EXPECTED_COMPILE_PROVER_SHA="b15350166c897dc56f6844e26d1cedcb2a45e4335701825b24e61b1ecd36cfe1"
EXPECTED_SUMMARIZER_SHA="0957b776b81e1c166b50f4b5661e05ea508dd4b542e3b09a454fe73865ff9477"

HF_OVERRIDES='{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
LLM_EXTRA_JSON='{"decode_context_parallel_size":1,"moe_backend":"b12x","enforce_eager":true}'

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
  require_sha "${COMPILE_PROVER}" "${EXPECTED_COMPILE_PROVER_SHA}"
  require_sha "${SUMMARIZER}" "${EXPECTED_SUMMARIZER_SHA}"

  local image_id
  image_id="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
  [[ "${image_id}" == "${EXPECTED_IMAGE_ID}" ]] || {
    echo "unexpected image ID: ${image_id}" >&2
    exit 2
  }
  docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}' \
    | grep -q '"ai.glm52.tr3.vllm_dynamic_merge":"42779a70eee6cafb2b2d3353620fafd2f13cfb5e"'
  docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}' \
    | grep -q '"ai.glm52.tr3.vllm_bounded_fs":"a1f5cc6cd0bcbcedfc607f98afbd8883ea3a3d5a"'
  docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}' \
    | grep -q '"ai.glm52.tr3.sparkinfer_dynamic_merge":"7ef6d26e"'

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
  mkdir -p "${KLD_ROOT}/results/dynamic_per_token" "${CACHE_ROOT}"
}

write_compile_proof() {
  local cache="$1"
  local output="$2"
  docker run --rm \
    -v "${cache}:/scan:ro" \
    -v "${COMPILE_PROVER}:/prove_nvfp4_writer_compile.py:ro" \
    --entrypoint /opt/venv/bin/python \
    "${IMAGE}" \
    /prove_nvfp4_writer_compile.py /scan true >"${output}"
}

run_one() {
  local run="$1"
  local out="${KLD_ROOT}/results/dynamic_per_token/run${run}"
  local name="glm52-tr3-dynamic-kld-run${run}"
  local cache="${CACHE_ROOT}/run${run}"

  if [[ -f "${out}/summary.json" ]]; then
    python3 "${SUMMARIZER}" validate "${out}" >/dev/null
    echo "KLD_SKIP_VALID run=${run} out=${out}"
    return
  fi
  mkdir -p "${out}" "${cache}"
  docker rm -f "${name}" >/dev/null 2>&1 || true

  python3 - \
    "${out}/config.json" "${run}" "${IMAGE}" "${EXPECTED_IMAGE_ID}" \
    "${MODEL_DIR}" "${MODEL_REVISION}" <<'PY'
import json
import pathlib
import sys

output, run, image, image_id, model_dir, model_revision = sys.argv[1:]
config = {
    "policy": "dynamic_per_token",
    "run": int(run),
    "image": image,
    "image_id": image_id,
    "model_dir": model_dir,
    "model_revision": model_revision,
    "reference_logits_sha256": "87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063",
    "reference_manifest_sha256": "985120136741037918bcd4dc8da9813c1f6268b35a730302f99cf6b3eebb7606",
    "runner_sha256": "d1dc1a63b9889e881f3bd899638d0ec65a1a1079132f6a207a600d9cba845405",
    "context_length": 2048,
    "max_num_batched_tokens": 512,
    "gpu_memory_utilization": 0.90,
    "kv_cache_dtype": "nvfp4_ds_mla",
    "quantization": "exl3",
    "exl3_trellis_min_m": "auto",
    "kv_fp8_rope": True,
    "dynamic_per_token": True,
    "static_scales_file": None,
    "tensor_parallel_size": 4,
    "decode_context_parallel_size": 1,
    "enforce_eager": True,
    "selector_policy": "exact",
}
pathlib.Path(output).write_text(json.dumps(config, indent=2) + "\n")
PY

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) KLD_START run=${run}"
  docker run --rm --name "${name}" \
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
    -e FLASHINFER_CUDA_ARCH_LIST=12.0f \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e OMP_NUM_THREADS=16 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e SAFETENSORS_FAST_GPU=1 \
    -e HF_HOME=/hf-cache \
    -e XDG_CACHE_HOME=/cache/jit \
    -e TRITON_CACHE_DIR=/cache/jit/triton \
    -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    -e SPARKINFER_COMPILE_CACHE_DIR=/cache/sparkinfer \
    -e PYTHONPATH=/kld-assets/pydeps \
    -e KV_FP8_ROPE=1 \
    -e VLLM_NVFP4_MLA_SCALES_FILE= \
    -e VLLM_NVFP4_MLA_DYNAMIC_SCALE=1 \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_USE_B12X_FP8_GEMM=1 \
    -e VLLM_USE_B12X_MOE=1 \
    -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
    -e VLLM_USE_B12X_WO_PROJECTION=1 \
    -e VLLM_USE_B12X_MHC=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e B12X_MOE_FORCE_A16=1 \
    -e B12X_MLA_SM120_UNIFIED=1 \
    -e B12X_DENSE_SPLITK_TURBO=1 \
    -e B12X_W4A16_TC_DECODE=1 \
    -e B12X_W4A8_TINY_DECODE=1 \
    -e VLLM_EXL3_TRELLIS_MIN_M= \
    -e VLLM_EXL3_TRELLIS_MAX_M=32 \
    -e VLLM_EXL3_TRELLIS_BLOCK_M=8 \
    -e VLLM_EXL3_PREFILL_CHUNK=128 \
    -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
    -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
    -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
    -e VLLM_PCIE_DMA_FP8=0 \
    -e B12X_PCIE_DMA_FP8=0 \
    -e NCCL_P2P_LEVEL=PXB \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_PROTO=LL,LL128,Simple \
    -e SPARKINFER_NSA_TOPK_SELECTION_POLICY=exact \
    -e HF_OVERRIDES="${HF_OVERRIDES}" \
    -e LLM_EXTRA_JSON="${LLM_EXTRA_JSON}" \
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
        --load-format safetensors \
        --max-model-len 4096 \
        --max-num-batched-tokens 512 \
        --max-num-seqs 1 \
        --quantization exl3 \
        --attention-backend B12X_MLA_SPARSE \
        --hf-overrides \"\${HF_OVERRIDES}\" \
        --llm-extra-json \"\${LLM_EXTRA_JSON}\" \
        --kld-chunk-rows 32" \
    2>&1 | tee "${out}/prefill_dcp1.log"

  write_compile_proof "${cache}" "${out}/writer-compile-proof.json"
  python3 "${SUMMARIZER}" validate "${out}"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) KLD_DONE run=${run}"
}

if [[ "${MODE}" == "summarize" ]]; then
  python3 "${SUMMARIZER}" summarize "${KLD_ROOT}" --expected-runs "${RUNS}"
  exit
fi

preflight
if [[ "${MODE}" == "smoke" ]]; then
  run_one 1
  echo "KLD_SMOKE_PASS"
  exit
fi
for run in $(seq 1 "${RUNS}"); do
  run_one "${run}"
done
python3 "${SUMMARIZER}" summarize "${KLD_ROOT}" --expected-runs "${RUNS}"
