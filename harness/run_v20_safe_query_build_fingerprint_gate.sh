#!/bin/bash
# Post-image, pre-push GPU gate. This is intentionally not a Dockerfile RUN:
# Docker builds do not have access to libcuda or a GPU.
set -euo pipefail

IMAGE=${1:?image tag or digest required}
STABLE_SHA256=${2:?measured stable-libtorch sha256 required}
SOURCE_COMMIT=${3:?exact source commit required}
OUT=${4:?fresh output directory required}
WAIVER=${5:-}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HARNESS=$ROOT/harness
REFERENCE=$HARNESS/references/safe_mla_query_bmm_sm120_cu132_pedantic_v1.jsonl
META=$HARNESS/references/safe_mla_query_bmm_sm120_cu132_pedantic_v1.meta.json
PROBE=$HARNESS/v20_safe_query_build_fingerprint_probe.py
COMPARATOR=$HARNESS/v20_safe_query_build_fingerprint_gate.py

expect_sha() {
  local file=$1 expected=$2 actual
  actual=$(sha256sum "$file" | awk '{print $1}')
  [ "$actual" = "$expected" ] || {
    echo "SHA mismatch: $file expected=$expected observed=$actual" >&2
    exit 2
  }
}

expect_sha "$PROBE" \
  6b3b6ebe2db2ead13533755daaa007afed8ccdc612e2612aa0a839f820853b96
expect_sha "$COMPARATOR" \
  59a8105b3c35443caed066f8bb289d883b23ea9f149d8424c771df7467733114
expect_sha "$REFERENCE" \
  08ae9da7501debee3dfb4144371f9f9c7929828047d57e737da17b610ca60084
expect_sha "$META" \
  d3e8a41a9028e8357e08f660637cb1426ff3cda0807a9356446bf5cd59490e94

[ ! -e "$OUT" ] || {
  echo "output path already exists: $OUT" >&2
  exit 2
}
mkdir -p "$OUT"
docker image inspect "$IMAGE" >"$OUT/image-inspect.json"

docker run --rm --gpus '"device=0"' --ipc=host \
  --entrypoint /opt/venv/bin/python \
  -v "$HARNESS:/proof:ro" -v "$OUT:/out" \
  "$IMAGE" /proof/v20_safe_query_build_fingerprint_probe.py \
  --call-mode precise \
  --expected-stable-sha256 "$STABLE_SHA256" \
  --output /out/observed.jsonl \
  2>&1 | tee "$OUT/probe.log"

gate=(
  python3 "$COMPARATOR"
  --observed "$OUT/observed.jsonl"
  --reference "$REFERENCE"
  --reference-meta "$META"
  --source-commit "$SOURCE_COMMIT"
  --output "$OUT/gate-report.json"
)
if [ -n "$WAIVER" ]; then
  gate+=(--waiver "$WAIVER")
fi
"${gate[@]}" 2>&1 | tee "$OUT/gate.log"
