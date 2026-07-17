#!/bin/bash
# Packed-CKV Stage-3 acceptance (brief rev 2 gate matrix).
# Usage: run_ckv_acceptance.sh {parity|ckv}
#   parity: env unset -> must match 964 baseline +-3%
#   ckv:    B12X_DCP_PREFILL_TRANSPORT=ckv -> bands 1253/1109
set -u
MODE=${1:?parity|ckv}
cd ~/glm52-wintest
docker rm -f glm52-wintest >/dev/null 2>&1
if [ "$MODE" = ckv ]; then
  EXTRA="CKV_TRANSPORT=ckv"
else
  EXTRA=""
fi
env $EXTRA MAXLEN=64000 BLOCKS=400 MNBT=3072 FP8_MODE=ring GATHER_FP8=1 RS_RING=1 PROF=1 \
  docker compose -f docker-compose.ckv.yml up -d
for i in $(seq 1 90); do
  docker logs glm52-wintest 2>&1 | grep -q "Application startup complete" && break
  docker logs glm52-wintest 2>&1 | grep -qE "CUDA out of memory|EngineCore failed|initialization failed" && { echo BOOT_FAILED; exit 1; }
  sleep 15
done
docker logs glm52-wintest 2>&1 | grep -q "Application startup complete" || { echo BOOT_TIMEOUT; exit 1; }
docker logs glm52-wintest 2>&1 | grep -E "GPU KV cache size|CKV|ckv" | tail -5
python3 ~/bench/prefill_bench.py --tokens 4000  --label ckvacc_${MODE}_warm >/dev/null 2>&1
python3 ~/bench/prefill_bench.py --tokens 8000  --label ckvacc_${MODE}_8k  | tail -1
python3 ~/bench/prefill_bench.py --tokens 55000 --label ckvacc_${MODE}_55k | tail -1
python3 ~/bench/quality_gate.py 2>&1 | tail -2
python3 ~/bench/quality_gate_fp8_ext.py --depth-tokens 60000 2>&1 | tail -1
echo "=== profiler ==="
docker logs glm52-wintest 2>&1 | grep "B12X_DCP_PROF summary" | head -4
