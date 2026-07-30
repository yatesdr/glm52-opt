#!/usr/bin/env bash
# Full fail-closed MTP3 promotion gate for a healthy TR3 serving process.
#
# This script never starts or restarts the server. It preserves request order,
# records every cold-cache metric, and rejects any container restart.
set -euo pipefail

BASE=${BASE:-http://127.0.0.1:8000}
MODEL=${MODEL:-GLM-5.2-EXL3-TR3-3.25bpw}
CONTAINER=${CONTAINER:-glm52-v20-r9-tr3-325}
DECODE_BENCH=${DECODE_BENCH:-/home/derek/decode_bench.py}
PREFILL_BENCH=${PREFILL_BENCH:-/home/derek/glm52-tr3-325-fused-m8-20260730/prefill_bench.py}
NEEDLE_BENCH=${NEEDLE_BENCH:-/home/derek/glm52-tr3-325-fused-m8-20260730/v20_gate2_needle_ladder.py}
OUT=${OUT:?set OUT to a new evidence directory}

mkdir -p "${OUT}"
STARTED_AT=$(docker inspect --format '{{.State.StartedAt}}' "${CONTAINER}")
START_RESTARTS=$(docker inspect --format '{{.RestartCount}}' "${CONTAINER}")
test "${START_RESTARTS}" = "0"

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

assert_container_stable() {
  test "$(docker inspect --format '{{.State.Status}}' "${CONTAINER}")" = "running"
  test "$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${CONTAINER}")" = "healthy"
  test "$(docker inspect --format '{{.RestartCount}}' "${CONTAINER}")" = "${START_RESTARTS}"
  test "$(docker inspect --format '{{.State.StartedAt}}' "${CONTAINER}")" = "${STARTED_AT}"
}

on_exit() {
  local rc=$?
  capture_state final
  docker logs --since "${STARTED_AT}" "${CONTAINER}" >"${OUT}/container.log" 2>&1 || true
  sha256sum "${OUT}"/* >"${OUT}/SHA256SUMS" 2>/dev/null || true
  exit "${rc}"
}
trap on_exit EXIT

assert_decode_rows() {
  local path=$1
  local expected_csv=$2
  python3 -c '
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = [int(value) for value in sys.argv[2].split(",")]
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert [row["concurrency"] for row in rows] == expected, rows
for row, concurrency in zip(rows, expected):
    assert row["requests_ok"] == concurrency, row
    assert row["effective_peak_concurrency"] == concurrency, row
    assert row["all_finalized"] is True, row
    assert row["errors"] == [], row
    assert row["server_generation_tokens"] > 0, row
' "${path}" "${expected_csv}"
}

assert_prefill() {
  local path=$1
  python3 -c '
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
rows = [line for line in text.splitlines() if line.startswith("RESULT ")]
assert len(rows) == 1, rows
row = rows[0]
cached = re.search(r"cached_tokens=(\d+)", row)
server = re.search(r"server_prefill_tok_s=(\d+)", row)
assert cached and int(cached.group(1)) == 0, row
assert server and int(server.group(1)) > 0, row
' "${path}"
}

assert_exact_needle_dir() {
  local directory=$1
  local expected_count=$2
  python3 -c '
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
expected_count = int(sys.argv[2])
summary = json.loads((directory / "summary.json").read_text())
assert summary["verdict"] == "PASS", summary
assert len(summary["cells"]) == expected_count, summary
for row in summary["cells"]:
    assert row["verdict"] == "PASS", row
    assert row["cached_tokens"] in (0, None), row
for path in sorted(directory.glob("cell-*k.json")):
    cell = json.loads(path.read_text())
    assert cell["content"].strip().replace(",", "") == "738216", (
        path,
        cell["content"],
    )
    assert all(cell["checks"].values()), (path, cell["checks"])
' "${directory}" "${expected_count}"
}

capture_state initial
curl -fsS --max-time 10 "${BASE}/health" >"${OUT}/health.txt"

python3 "${DECODE_BENCH}" \
  --base "${BASE}" --model "${MODEL}" \
  --concurrency 1,2,4,8,16,24,32 --output-tokens 256 \
  | tee "${OUT}/decode-mtp3-c1-c32.log"
assert_decode_rows "${OUT}/decode-mtp3-c1-c32.log" "1,2,4,8,16,24,32"
assert_container_stable
capture_state post-decode-matrix

for tokens in 8000 55000 64000 128000 250000; do
  python3 "${PREFILL_BENCH}" \
    --base "${BASE}" --model "${MODEL}" --tokens "${tokens}" \
    --label "candidate13-cold-${tokens}" \
    | tee "${OUT}/prefill-cold-${tokens}.log"
  assert_prefill "${OUT}/prefill-cold-${tokens}.log"
  assert_container_stable
done
capture_state post-prefill-matrix

python3 "${NEEDLE_BENCH}" \
  --base "${BASE}" --model "${MODEL}" --out "${OUT}/needle-main" \
  --only 50000 350000 475000 \
  | tee "${OUT}/needle-main.log"
assert_exact_needle_dir "${OUT}/needle-main" 3
assert_container_stable
capture_state post-needle-main

# Final production-history discriminator: max concurrency followed immediately
# by a new, cold deep retrieval on the same process.
python3 "${DECODE_BENCH}" \
  --base "${BASE}" --model "${MODEL}" --concurrency 32 --output-tokens 256 \
  | tee "${OUT}/stress-decode-c32.log"
assert_decode_rows "${OUT}/stress-decode-c32.log" "32"
assert_container_stable

python3 "${NEEDLE_BENCH}" \
  --base "${BASE}" --model "${MODEL}" --out "${OUT}/stress-needle-350k" \
  --only 350000 --skip-side-checks \
  | tee "${OUT}/stress-needle-350k.log"
assert_exact_needle_dir "${OUT}/stress-needle-350k" 1
assert_container_stable

capture_state passed
printf 'PASS: complete MTP3 promotion matrix and C32 -> cold-350k stress\n'
