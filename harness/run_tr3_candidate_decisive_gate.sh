#!/usr/bin/env bash
# Fail-closed promotion discriminator for an already-running TR3 candidate.
#
# Order is deliberate: exercise the MTP3 M=64 route, then maximum configured
# concurrency, then demand a cold 250k retrieval without restarting the model.
# This reproduces the request history that rejected Candidate 12.
set -euo pipefail

BASE=${BASE:-http://127.0.0.1:8000}
MODEL=${MODEL:-GLM-5.2-EXL3-TR3-3.25bpw}
CONTAINER=${CONTAINER:-glm52-v20-r9-tr3-325}
DECODE_BENCH=${DECODE_BENCH:-/home/derek/decode_bench.py}
NEEDLE_BENCH=${NEEDLE_BENCH:-/home/derek/glm52-tr3-325-fused-m8-20260730/v20_gate2_needle_ladder.py}
OUT=${OUT:?set OUT to a new evidence directory}

mkdir -p "${OUT}"
STARTED_AT=$(docker inspect --format '{{.State.StartedAt}}' "${CONTAINER}")

capture_state() {
  local label=$1
  docker inspect "${CONTAINER}" >"${OUT}/${label}-container-inspect.json" 2>&1 || true
  docker inspect \
    --format 'status={{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} restarts={{.RestartCount}} started={{.State.StartedAt}}' \
    "${CONTAINER}" >"${OUT}/${label}-container-state.txt" 2>&1 || true
  nvidia-smi \
    --query-gpu=index,memory.total,memory.used,memory.free,temperature.gpu,power.draw,clocks.sm \
    --format=csv,noheader,nounits >"${OUT}/${label}-nvidia-smi.csv" 2>&1 || true
}

on_exit() {
  local rc=$?
  capture_state final
  docker logs --since "${STARTED_AT}" "${CONTAINER}" >"${OUT}/container.log" 2>&1 || true
  sha256sum "${OUT}"/* >"${OUT}/SHA256SUMS" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT

assert_decode_row() {
  local path=$1
  local expected_concurrency=$2
  python3 -c '
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert len(rows) == 1, rows
row = rows[0]
assert row["concurrency"] == expected, row
assert row["requests_ok"] == expected, row
assert row["effective_peak_concurrency"] == expected, row
assert row["all_finalized"] is True, row
assert row["errors"] == [], row
assert row["server_generation_tokens"] > 0, row
' "${path}" "${expected_concurrency}"
}

capture_state initial
curl -fsS --max-time 10 "${BASE}/health" >"${OUT}/health.txt"

python3 "${DECODE_BENCH}" \
  --base "${BASE}" --model "${MODEL}" --concurrency 16 --output-tokens 32 \
  | tee "${OUT}/decode-c16-warmup.log"
assert_decode_row "${OUT}/decode-c16-warmup.log" 16

python3 "${DECODE_BENCH}" \
  --base "${BASE}" --model "${MODEL}" --concurrency 16 --output-tokens 256 \
  | tee "${OUT}/decode-c16.log"
assert_decode_row "${OUT}/decode-c16.log" 16
capture_state post-c16

python3 "${DECODE_BENCH}" \
  --base "${BASE}" --model "${MODEL}" --concurrency 32 --output-tokens 256 \
  | tee "${OUT}/decode-c32.log"
assert_decode_row "${OUT}/decode-c32.log" 32
capture_state post-c32

python3 "${NEEDLE_BENCH}" \
  --base "${BASE}" --model "${MODEL}" --out "${OUT}/needle-250k" \
  --only 250000 --skip-side-checks \
  | tee "${OUT}/needle-250k.log"

python3 -c '
import json
import pathlib
import sys

summary = json.loads((pathlib.Path(sys.argv[1]) / "summary.json").read_text())
assert summary["verdict"] == "PASS", summary
assert len(summary["cells"]) == 1, summary
cell = summary["cells"][0]
assert cell["verdict"] == "PASS", cell
assert cell["cached_tokens"] in (0, None), cell
' "${OUT}/needle-250k"

capture_state passed
printf 'PASS: C16 -> C32 -> cold-250k completed with no harness failure\n'
