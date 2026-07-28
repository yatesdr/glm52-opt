#!/usr/bin/env bash
set -euo pipefail

# Derive the quality-first weight-quantization KLD matrix from the already
# qualified dynamic/static runner without duplicating its long fail-closed
# Docker invocation.  The source runner is byte-pinned.  The derived runner is
# retained beside its results so the exact executed program remains auditable.
#
# Arm D runs first: quality-first weights + dynamic per-token KV scale.
# Arm C runs second: quality-first weights + static #145 KV scales.

MODE="${1:-}"
case "${MODE}" in
  smoke|full|summarize) ;;
  *)
    echo "usage: $0 {smoke|full|summarize}" >&2
    exit 2
    ;;
esac

BASE_RUNNER="${BASE_RUNNER:-/home/derek/kld-dynamic-scale-20260728/run_v20_dynamic_scale_kld.sh}"
BASE_ROOT="${BASE_ROOT:-/home/derek/kld-dynamic-scale-20260728}"
KLD_ROOT="${KLD_ROOT:-/home/derek/kld-quality-weight-dynamic-scale-20260728}"
CACHE_ROOT="${CACHE_ROOT:-/home/derek/glm52-kld-quality-weight-cache}"
EXPECTED_BASE_SHA="ac8e57f67194a3e64a779eed54494810e81cca6459283e50baf242d19c714ce0"
DERIVED_RUNNER="${KLD_ROOT}/run_v20_quality_weight_kld.derived.sh"

QUALITY_QUANT_CONFIG='{"linear":{"weight":"mxfp8"},"ignore":["re:.*\\.fused_qkv_a_proj$","re:.*\\.q_a_proj$","re:.*kv_a_proj_with_mqa","re:.*\\.mlp\\.gate$","model.layers.78.eh_proj","lm_head"]}'
BASE_QUANT_CONFIG='{"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"},"ignore":["re:^model\\.layers\\.0\\.","re:.*\\.self_attn\\.indexer\\.","re:.*\\.mlp\\.gate$","model.layers.78.eh_proj","lm_head"]}'

actual_base_sha="$(sha256sum "${BASE_RUNNER}" | awk '{print $1}')"
if [[ "${actual_base_sha}" != "${EXPECTED_BASE_SHA}" ]]; then
  echo "base KLD runner SHA-256 mismatch" >&2
  echo "expected=${EXPECTED_BASE_SHA}" >&2
  echo "actual=${actual_base_sha}" >&2
  exit 2
fi

mkdir -p "${KLD_ROOT}" "${CACHE_ROOT}"

# These are intentionally literal, count-checked substitutions.  Any drift in
# the pinned base runner makes the derivation fail rather than silently running
# a different experiment.
BASE_QUANT_CONFIG="${BASE_QUANT_CONFIG}" \
QUALITY_QUANT_CONFIG="${QUALITY_QUANT_CONFIG}" \
python3 - "${BASE_RUNNER}" "${DERIVED_RUNNER}" <<'PY'
import os
import pathlib
import sys

source_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
text = source_path.read_text()

replacements = (
    (
        "QUANT_CONFIG='" + os.environ["BASE_QUANT_CONFIG"] + "'",
        "QUANT_CONFIG='" + os.environ["QUALITY_QUANT_CONFIG"] + "'",
    ),
    (
        '"online_quantization": "nf3-mxfp8",',
        '"online_quantization": "custom_quality_first_mxfp8",',
    ),
    (
        "for policy in static_calibrated dynamic_per_token; do",
        "for policy in dynamic_per_token static_calibrated; do",
    ),
    (
        "run_one static_calibrated 1\n  run_one dynamic_per_token 1",
        "run_one dynamic_per_token 1\n  run_one static_calibrated 1",
    ),
)

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"expected one derivation target, found {count}: {old[:80]!r}"
        )
    text = text.replace(old, new)

output_path.write_text(text)
PY
chmod 0755 "${DERIVED_RUNNER}"

derived_sha="$(sha256sum "${DERIVED_RUNNER}" | awk '{print $1}')"
printf 'QUALITY_KLD_DERIVATION base_sha256=%s derived_sha256=%s\n' \
  "${actual_base_sha}" "${derived_sha}"
printf 'QUALITY_KLD_CONFIG %s\n' "${QUALITY_QUANT_CONFIG}"

export KLD_ROOT CACHE_ROOT
export SUMMARIZER="${BASE_ROOT}/summarize_v20_dynamic_scale_kld.py"
export COMPILE_PROVER="${BASE_ROOT}/prove_nvfp4_writer_compile.py"

exec "${DERIVED_RUNNER}" "${MODE}"
