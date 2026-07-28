#!/bin/bash
# Auto-calibration entrypoint for GLM-5.2 v20 serving (direct vllm serve).
#
# Usage (compose):  entrypoint: ["/usr/local/bin/serve-with-autocal.sh"]
#                   command:    [/model, --port=5001, ...vllm serve args...]
#
# Before vllm starts it:
#   1. NCCL_P2P_LEVEL=auto/unset -> measured probe (real NCCL traffic,
#      verified, hang-guarded, cached) -> static topology derivation ->
#      NCCL defaults. Explicit values always win.
#   2. Wire transport: in the NVFP4 posture (--kv-cache-dtype=nvfp4_ds_mla
#      + KV_FP8_ROPE=1) with no explicit wire mode set, races the INT8
#      family (i8 / i8_ring / i8_a2a) with verified traffic and exports the
#      winner as SPARKINFER_PCIE_DMA_FP8 / VLLM_PCIE_DMA_FP8 /
#      B12X_PCIE_DMA_FP8. Explicit values (any of the three envs, incl.
#      0/off) are respected verbatim. i8_auto / mx_auto request a race
#      explicitly. The codec family is never chosen automatically.
#   3. Dynamic per-token NVFP4 KV scaling: defaults ON in the NVFP4
#      posture (VLLM_NVFP4_MLA_DYNAMIC_SCALE unset -> 1). Explicit 0 wins.
#      Other KV dtypes are untouched.
#   4. Clears stale offload mmaps from /dev/shm (ipc:host), then
#      exec vllm serve "$@".
set -u

_autocal_log() { echo "[autocal] $*" >&2; }

# ---- Posture detection from the real serve args + env -----------------------
_kv_dtype=""
for _arg in "$@"; do
  case "${_arg}" in
    --kv-cache-dtype=*) _kv_dtype="${_arg#--kv-cache-dtype=}" ;;
  esac
done
[ -z "${_kv_dtype}" ] && _kv_dtype="${KV_CACHE_DTYPE:-}"
_nvfp4_posture=0
if [ "${_kv_dtype}" = "nvfp4_ds_mla" ] && [ "${KV_FP8_ROPE:-0}" = "1" ]; then
  _nvfp4_posture=1
fi
_autocal_log "posture: kv-cache-dtype=${_kv_dtype:-unset} KV_FP8_ROPE=${KV_FP8_ROPE:-0} nvfp4=${_nvfp4_posture}"

# ---- 1. NCCL_P2P_LEVEL ------------------------------------------------------
case "${NCCL_P2P_LEVEL:-auto}" in
  auto|"")
    LEVEL=""
    if LEVEL=$(python3 /usr/local/bin/nccl_p2p_probe.py \
                 --devices "${CUDA_VISIBLE_DEVICES:-}" \
                 2> >(sed 's/^/[autocal] /' >&2)); then
      _autocal_log "NCCL_P2P_LEVEL=auto -> measured ${LEVEL}"
    elif LEVEL=$(python3 /usr/local/bin/derive_nccl_p2p_level.py \
                 --devices "${CUDA_VISIBLE_DEVICES:-}" \
                 2> >(sed 's/^/[autocal] /' >&2)); then
      _autocal_log "NCCL_P2P_LEVEL=auto -> probe unavailable, static ${LEVEL}"
    else
      _autocal_log "NCCL_P2P_LEVEL=auto -> derivation unavailable, NCCL defaults"
      LEVEL=""
    fi
    if [ -n "${LEVEL}" ]; then export NCCL_P2P_LEVEL="${LEVEL}"; else unset NCCL_P2P_LEVEL; fi
    ;;
  *)
    _autocal_log "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL} explicit; respected"
    ;;
esac

# ---- 2. Wire transport ------------------------------------------------------
# Single source across historical env spellings; first set one wins.
_wire="${SPARKINFER_PCIE_DMA_FP8:-${VLLM_PCIE_DMA_FP8:-${B12X_PCIE_DMA_FP8:-${F8_DMA:-}}}}"
if [ -z "${_wire}" ] && [ "${_nvfp4_posture}" = "1" ]; then
  _autocal_log "NVFP4 posture, wire unset -> i8 family race"
  _wire="i8_auto"
fi
_export_wire() {
  export SPARKINFER_PCIE_DMA_FP8="$1" VLLM_PCIE_DMA_FP8="$1" B12X_PCIE_DMA_FP8="$1"
}
case "${_wire}" in
  i8_auto|mx_auto)
    FAMILY="${_wire%_auto}"
    if MODE=$(python3 /usr/local/bin/wire_mode_probe.py --family "${FAMILY}" \
                2> >(sed 's/^/[autocal] /' >&2)); then
      _autocal_log "wire=${_wire} -> measured ${MODE}"
    else
      case "${FAMILY}" in i8) MODE="i8_ring" ;; mx) MODE="mx_ring" ;; esac
      _autocal_log "wire probe unavailable -> family default ${MODE}"
    fi
    _export_wire "${MODE}"
    ;;
  "")
    : ;;  # non-NVFP4 posture, nothing requested: leave stack defaults
  *)
    _autocal_log "wire=${_wire} explicit; respected"
    _export_wire "${_wire}"
    ;;
esac

# ---- 3. Dynamic per-token NVFP4 KV scale ------------------------------------
if [ -z "${VLLM_NVFP4_MLA_DYNAMIC_SCALE:-}" ] && [ "${_nvfp4_posture}" = "1" ]; then
  export VLLM_NVFP4_MLA_DYNAMIC_SCALE=1
  _autocal_log "VLLM_NVFP4_MLA_DYNAMIC_SCALE unset + NVFP4 posture -> 1"
fi

# ---- 4. shm cleanup + exec --------------------------------------------------
rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null || true
_autocal_log "starting: vllm serve $*"
exec vllm serve "$@"
