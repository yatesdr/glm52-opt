#!/bin/bash
# v20 baseline suite — Gate 3 characterization on an already-running process.
#
# Run ON the serving host. Strictly SEQUENTIAL: overlapping requests would corrupt
# cold-prefill numbers and confuse concurrency cells. Does not restart or reconfigure
# anything, so it is safe on a process we have been told to leave undisturbed.
#
# Fills the library gaps identified 2026-07-25:
#   1. v20 decode at ctx16k (never measured — decode_bench only ever ran ctx0)
#   2. prefill 8k/55k reproduction, to check Sol's 1364 / 1115-1221 tok/s independently
#   4. MTP acceptance per cell (both benches report acceptance deltas)
#
# Usage: bash run_v20_baseline_suite.sh <outdir>
set -uo pipefail
OUT=${1:-$HOME/v20-baseline-$(date -u +%Y%m%dT%H%M%SZ)}
BASE=${BASE:-http://localhost:5001}
mkdir -p "$OUT"
say() { echo "=== $* ===" ; }

say "identity before" | tee -a "$OUT/suite.log"
docker inspect a5982029f831 --format 'running={{.State.Running}} restarts={{.RestartCount}} started={{.State.StartedAt}}' \
  2>&1 | tee -a "$OUT/suite.log"

for tok in 8000 55000; do
  say "prefill cold ${tok}" | tee -a "$OUT/suite.log"
  python3 prefill_bench.py --base "$BASE" --tokens "$tok" --label "v20-fa71a0c1" \
    2>&1 | tee "$OUT/prefill-${tok}.log" | tee -a "$OUT/suite.log"
done

for ctx in 0 16000; do
  say "decode ctx${ctx} C1,C4,C8,C16" | tee -a "$OUT/suite.log"
  python3 decode_bench.py --base "$BASE" --concurrency 1,4,8,16 --output-tokens 256 \
    --context-tokens "$ctx" 2>&1 | tee "$OUT/decode-ctx${ctx}.log" | tee -a "$OUT/suite.log"
done

say "identity after" | tee -a "$OUT/suite.log"
docker inspect a5982029f831 --format 'running={{.State.Running}} restarts={{.RestartCount}} started={{.State.StartedAt}}' \
  2>&1 | tee -a "$OUT/suite.log"
say "SUITE COMPLETE" | tee -a "$OUT/suite.log"
