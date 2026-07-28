#!/usr/bin/env bash
# Resolve the two host-dependent communication settings, then hand control
# unchanged to Festr's canonical GLM-5.2 launcher.
#
# Explicit values always win:
#   NCCL_P2P_LEVEL=PXB|PHB|SYS|...  skips the topology probe.
#   F8_DMA=0|i8|i8_ring|i8_a2a|... skips the wire probe.
#
# Auto values:
#   NCCL_P2P_LEVEL=auto  measures verified NCCL traffic, then falls back to
#                         the conservative static topology derivation.
#   F8_DMA=auto           races i8/i8_ring/i8_a2a and accepts only modes
#                         whose output passes the configured numeric and
#                         cross-rank consistency checks. If none verify,
#                         serving falls back to uncompressed F8_DMA=0.
set -euo pipefail

log() {
  printf '[glm52-auto] %s\n' "$*" >&2
}

die() {
  log "FATAL: $*"
  exit 2
}

python_bin="${GLM52_PYTHON:-/opt/venv/bin/python3}"
nccl_probe="${GLM52_NCCL_PROBE:-/usr/local/bin/nccl_p2p_probe.py}"
nccl_static="${GLM52_NCCL_STATIC:-/usr/local/bin/derive_nccl_p2p_level.py}"
wire_probe="${GLM52_WIRE_PROBE:-/usr/local/bin/wire_mode_probe.py}"
canonical_launcher="${GLM52_CANONICAL_LAUNCHER:-/usr/local/bin/serve-gilded-gnosis.sh}"
calibration_cache="${XDG_CACHE_HOME:-/cache}/glm52-auto"

case "${DESTROYED_MXFP8:-0}" in
  0)
    ;;
  1)
    export ONLINE_QUANT=custom
    export QUANTIZATION_CONFIG_JSON='{"linear":{"weight":"mxfp8"},"ignore":["re:.*\\.fused_qkv_a_proj$","re:.*\\.q_a_proj$","re:.*kv_a_proj_with_mqa","re:.*\\.mlp\\.gate$","model.layers.78.eh_proj","lm_head"]}'
    log "DESTROYED_MXFP8=1: quality-first MXFP8 membership enabled"
    log "this mode consumes more VRAM; MAX_MODEL_LEN remains operator-controlled"
    ;;
  *)
    die "DESTROYED_MXFP8 must be 0 or 1"
    ;;
esac

case "${NCCL_P2P_LEVEL:-auto}" in
  auto|"")
    level=""
    if level="$("${python_bin}" "${nccl_probe}" \
        --devices "${GPUS:-${CUDA_VISIBLE_DEVICES:-}}" \
        --cache-dir "${calibration_cache}/nccl-p2p")"; then
      log "NCCL_P2P_LEVEL=auto resolved by measured probe: ${level}"
    elif level="$("${python_bin}" "${nccl_static}" \
        --devices "${GPUS:-${CUDA_VISIBLE_DEVICES:-}}")"; then
      log "NCCL measured probe unavailable; static topology selected: ${level}"
    else
      level=""
      log "NCCL topology selection unavailable; leaving NCCL_P2P_LEVEL unset"
    fi
    if [[ -n "${level}" ]]; then
      export NCCL_P2P_LEVEL="${level}"
    else
      unset NCCL_P2P_LEVEL
    fi
    ;;
  *)
    log "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL} is explicit"
    ;;
esac

alias_wire="${SPARKINFER_PCIE_DMA_FP8:-${VLLM_PCIE_DMA_FP8:-${B12X_PCIE_DMA_FP8:-}}}"
requested_wire="${F8_DMA:-auto}"

case "${requested_wire}" in
  auto|i8_auto|"")
    if [[ -n "${alias_wire}" && "${alias_wire}" != "auto" && \
          "${alias_wire}" != "i8_auto" ]]; then
      resolved_wire="${alias_wire}"
      log "wire mode supplied through a legacy alias: ${resolved_wire}"
    elif resolved_wire="$("${python_bin}" "${wire_probe}" --family i8 \
        --cache-dir "${calibration_cache}/wire-mode")"; then
      log "F8_DMA=${requested_wire:-auto} resolved by verified probe: ${resolved_wire}"
    else
      resolved_wire=0
      log "no compressed INT8 mode verified; falling back to F8_DMA=0"
    fi
    ;;
  *)
    resolved_wire="${requested_wire}"
    if [[ -n "${alias_wire}" && "${alias_wire}" != "${resolved_wire}" ]]; then
      die "conflicting wire settings: F8_DMA=${resolved_wire}, legacy alias=${alias_wire}"
    fi
    log "F8_DMA=${resolved_wire} is explicit"
    ;;
esac

export F8_DMA="${resolved_wire}"
export SPARKINFER_PCIE_DMA_FP8="${resolved_wire}"
export VLLM_PCIE_DMA_FP8="${resolved_wire}"
export B12X_PCIE_DMA_FP8="${resolved_wire}"

[[ -x "${canonical_launcher}" ]] || die "canonical launcher is missing: ${canonical_launcher}"
log "resolved NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-default} F8_DMA=${F8_DMA}"
log "delegating to ${canonical_launcher}"
exec "${canonical_launcher}" "$@"
