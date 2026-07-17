#!/bin/bash
# Phase-2 server gates (Sol design §8). Usage: run_p2_gates.sh {gate1|gate2|gate3}
set -u
G=${1:?gate1|gate2|gate3}
cd ~/glm52-wintest
docker rm -f glm52-wintest >/dev/null 2>&1
case $G in
  gate1) ENV="CKV_TRANSPORT=ckv CKV_P2=1 MAXLEN=64000 BLOCKS=400 MNBT=3072 FP8_MODE=ring PROF=1";;
  gate2) ENV="CKV_TRANSPORT=ckv CKV_P2=1 CKV_ESCROW=1 CKV_PROBES=1 MAXLEN=480000 BLOCKS=2340 MNBT=3072 FP8_MODE=ring PROF=0";;
  gate3) ENV="CKV_TRANSPORT=ckv CKV_P2=1 CKV_ESCROW=1 MAXLEN=480000 BLOCKS=2340 MNBT=3072 FP8_MODE=ring PROF=1";;
esac
env $ENV docker compose -f docker-compose.ckv-p2.yml up -d
for i in $(seq 1 120); do
  docker logs glm52-wintest 2>&1 | grep -q "Application startup complete" && break
  docker logs glm52-wintest 2>&1 | grep -qE "CUDA out of memory|EngineCore failed|initialization failed|Packed-CKV.*(fatal|failed)" && { echo BOOT_FAILED; docker logs glm52-wintest > ~/bench/p2-$G-fail.log 2>&1; exit 1; }
  docker ps -q -f name=glm52-wintest | grep -q . || { echo CONTAINER_DEAD; exit 1; }
  sleep 15
done
docker logs glm52-wintest 2>&1 | grep -q "Application startup complete" || { echo BOOT_TIMEOUT; exit 1; }
docker logs glm52-wintest 2>&1 | grep -E "GPU KV cache size|Packed-CKV|escrow|probe" | tail -8
M0=$(curl -s localhost:5001/metrics | grep -E "prefix_cache_(queries|hits)_total|prompt_tokens_cached" | head -6)
case $G in
  gate1)
    python3 ~/bench/prefill_bench.py --tokens 4000  --label p2g1_warm >/dev/null 2>&1
    python3 ~/bench/prefill_bench.py --tokens 8000  --label p2g1_8k  | tail -1
    python3 ~/bench/prefill_bench.py --tokens 55000 --label p2g1_55k | tail -1
    python3 ~/bench/quality_gate.py 2>&1 | tail -2
    docker logs glm52-wintest 2>&1 | grep "B12X_DCP_PROF summary" | head -2;;
  gate2)
    python3 ~/bench/prefill_bench.py --tokens 470000 --label p2g2_480k 2>&1 | tail -1
    docker logs glm52-wintest 2>&1 | grep -iE "escrow|probe" | tail -10;;
  gate3)
    python3 ~/bench/prefill_bench.py --tokens 4000   --label p2g3_warm >/dev/null 2>&1
    python3 ~/bench/prefill_bench.py --tokens 55000  --label p2g3_55k  | tail -1
    python3 ~/bench/prefill_bench.py --tokens 470000 --label p2g3_480k 2>&1 | tail -1
    python3 ~/bench/quality_gate.py 2>&1 | tail -2
    python3 ~/bench/quality_gate_fp8_ext.py --deep 2>&1 | tail -3
    ~/hfenv/bin/python ~/bench/llm_decode_bench.py --port 5001 --concurrency 1 >/dev/null 2>&1
    python3 -c "import json; d=json.load(open(\"benchmark_results.json\")); [print(f\"decode ctx={r.get(chr(39)+\"context_tokens\"+chr(39))}: {r.get(chr(39)+\"aggregate_tps\"+chr(39)):.1f}\") for r in d.get(\"results\",[]) if r.get(\"aggregate_tps\")]" 2>/dev/null || true
    docker logs glm52-wintest 2>&1 | grep "B12X_DCP_PROF summary" | head -2;;
esac
echo "=== cache deltas ==="
echo "BEFORE: $M0" | head -3
curl -s localhost:5001/metrics | grep -E "prefix_cache_(queries|hits)_total|prompt_tokens_cached" | head -6
