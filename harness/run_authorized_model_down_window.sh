#!/bin/bash
# THE AUTHORIZED MODEL-DOWN WINDOW — everything that cannot run alongside production.
#
# Requires Derek's explicit authorization: step 1 STOPS the running fa71a0c1 process.
#
# Three workloads, ordered so the cheap conclusive ones come first and a failure stops early:
#   A. numeric gate with GPUs free   — M=3072 across all four modes + accum/handle proof
#                                      (unreachable with the model up: cuBLAS handle failure during
#                                       graph capture, then OOM at 48 MiB vs 39.8 MiB free)
#   B. causal boot of the discriminator — 100k x3 and 150k x3 fixed-seed cold cells (Sol dev#52)
#   C. pricing + deep ladder          — only if B's decisive cells all come back EXACT
#
# Everything measurable alongside production is already done and archived; nothing here duplicates it.
#
# Usage: ACCURATE_IMAGE=<tag> ACCURATE_STABLE_SHA256=<sha256> \
#   bash run_authorized_model_down_window.sh <compose.yaml> [outdir]
set -euo pipefail

COMPOSE=${1:?compose file required}
OUT=${2:-$HOME/window-$(date -u +%Y%m%dT%H%M%SZ)}
BASE=${BASE:-http://localhost:5001}
PROOF=$HOME/proof-harness
POUT=$OUT/proofs
DISC=ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-pedantic-discriminator-20260725
CURRENT=ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-prod-ready-20260724
ACCURATE=${ACCURATE_IMAGE:?ACCURATE_IMAGE is required}
ACC_SHA=${ACCURATE_STABLE_SHA256:?ACCURATE_STABLE_SHA256 is required}
POOL_FLOOR=500000
mkdir -p "$OUT" "$POUT"
log() { echo "$@" | tee -a "$OUT/window.log"; }
expect_sha() {
  local file=$1 expected=$2 actual
  actual=$(sha256sum "$file" | awk '{print $1}')
  [ "$actual" = "$expected" ] || {
    log "SHA MISMATCH: $file expected=$expected actual=$actual"
    exit 10
  }
}
probe() {  # probe <image> <mode> <expected-stable-sha> <outfile>
  docker run --rm --gpus '"device=0"' --ipc=host --entrypoint /opt/venv/bin/python \
    -v "$PROOF:/proof:ro" -v "$POUT:/out" "$1" \
    /proof/v20_safe_query_reduction_equivalence_probe.py \
    --mode "$2" --expected-stable-sha256 "$3" --output "/out/$4" 2>&1 | tail -3 | tee -a "$OUT/window.log"
}

log "=== 0. fail-closed preflight (NO process changes) ==="
[ -f "$COMPOSE" ] || { log "compose missing: $COMPOSE"; exit 10; }
expect_sha "$PROOF/v20_safe_query_reduction_equivalence_probe.py" \
  aa0ecad9cb1eb2b539d6455dc4077e2180853d8a3ca2668e1437b14edb5e3b06
expect_sha "$PROOF/v20_compare_safe_query_reduction_equivalence.py" \
  205552e06301d1a3ebb52fe703104cca8733e43f0399d07683ff309c4a573f50
expect_sha "$PROOF/v20_safe_query_accum_gpu_proof.py" \
  22f1c412b0f548b33c9448c047af77d99e38a9139bff784bbf14929acd6f8ea9
for image in "$CURRENT" "$DISC" "$ACCURATE"; do
  docker image inspect "$image" >/dev/null
done
COMPOSE_IMAGES=$(docker compose -f "$COMPOSE" config --images)
if ! grep -Fxq "$DISC" <<<"$COMPOSE_IMAGES"; then
  log "compose does not resolve to discriminator image: $DISC"
  log "resolved images:"
  log "$COMPOSE_IMAGES"
  exit 10
fi
mapfile -t BEFORE_CIDS < <(docker compose -f "$COMPOSE" ps -q)
[ "${#BEFORE_CIDS[@]}" -gt 0 ] || {
  log "compose has no running container to stop"
  exit 10
}
for cid in "${BEFORE_CIDS[@]}"; do
  docker inspect "$cid" \
    --format 'before: id={{.Id}} image={{.Config.Image}} restarts={{.RestartCount}} started={{.State.StartedAt}}' \
    | tee -a "$OUT/window.log"
done

reclaim_shm() {  # orphaned vllm offload mmaps survive container death and eat /dev/shm
  local before after
  before=$(df -h /dev/shm | awk 'NR==2{print $3}')
  docker run --rm -v /dev/shm:/hostshm alpine sh -c 'rm -f /hostshm/vllm_offload_*.mmap' >/dev/null 2>&1 || true
  after=$(df -h /dev/shm | awk 'NR==2{print $3}')
  log "reclaimed /dev/shm: ${before} used -> ${after} used"
}

log "=== 1. STOP the production process (AUTHORIZED) ==="
docker compose -f "$COMPOSE" down 2>&1 | tail -3 | tee -a "$OUT/window.log"
sleep 8
[ -z "$(docker compose -f "$COMPOSE" ps -q)" ] || {
  log "compose still has running containers after down"
  exit 11
}
reclaim_shm
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$OUT/window.log"

log "=== 2A. numeric gate, GPUs free: full 54 cases incl. M=3072, graphs ENABLED ==="
probe "$CURRENT" legacy-current  9e2608a49dfe7953bf822b244530053799a3ac99c5136f1b33b39d6fe91f78fa equiv-w-current.jsonl
probe "$DISC"    legacy-pedantic fcf056af6607bb4fd0174fda977e45e2ebe69e8c09c0dab4844adc7bf33c635d equiv-w-pedantic.jsonl
probe "$ACCURATE" accurate-regular "$ACC_SHA" equiv-w-accurate-regular.jsonl
probe "$ACCURATE" accurate-precise "$ACC_SHA" equiv-w-accurate-precise.jsonl
log "--- four-mode comparator (pin 205552e0) ---"
python3 "$PROOF/v20_compare_safe_query_reduction_equivalence.py" \
  "$POUT/equiv-w-current.jsonl" "$POUT/equiv-w-pedantic.jsonl" \
  "$POUT/equiv-w-accurate-regular.jsonl" "$POUT/equiv-w-accurate-precise.jsonl" \
  2>&1 | tee -a "$OUT/window.log"
log "--- accum/handle proof (pin 22f1c412), regular->precise->regular restoration ---"
docker run --rm --gpus '"device=0"' --ipc=host --entrypoint /opt/venv/bin/python \
  -v "$PROOF:/proof:ro" -v "$POUT:/out" "$ACCURATE" \
  /proof/v20_safe_query_accum_gpu_proof.py --output /out/accum-handle-proof.jsonl \
  2>&1 | tail -5 | tee -a "$OUT/window.log"

log "=== 2B. boot the discriminator (exact fa71 compose, image tag only differs) ==="
docker compose -f "$COMPOSE" up -d 2>&1 | tail -3 | tee -a "$OUT/window.log"
for i in $(seq 1 120); do
  curl -fsS -m 5 "$BASE/health" >/dev/null 2>&1 && break
  sleep 15
done
curl -fsS -m 5 "$BASE/health" >/dev/null
CID=$(docker ps --format '{{.ID}}' --filter ancestor="$DISC" | head -1)
[ -n "$CID" ] || {
  log "no running container found for discriminator image: $DISC"
  exit 12
}
docker inspect "$CID" --format 'container={{.Id}} started={{.State.StartedAt}} restarts={{.RestartCount}}' \
  2>&1 | tee -a "$OUT/window.log"

log "=== 2C. KV pool floor gate — STOP if under $POOL_FLOOR (no GMU raise, per runbook) ==="
POOL=$(docker logs "$CID" 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+' \
  | tail -1 | grep -oE '[0-9,]+' | tr -d , || true)
log "pool=${POOL:-unreadable} floor=$POOL_FLOOR"
[ -n "${POOL:-}" ] && [ "$POOL" -ge "$POOL_FLOOR" ] || { log "POOL FLOOR MISS — stopping before quality cells."; exit 2; }

log "=== 3. decisive causal cells: 100k x3 and 150k x3, fixed seeds, prompt_sha256 recorded ==="
cd "$HOME"
python3 v20_needle_duplication_onset_probe.py --base "$BASE" --heads natural \
  --depths 100000 150000 --reps 3 --max-tokens 2000 \
  --fixed-head-seed 20260725 --sampling-seed 20260725 --out "$OUT/decisive" \
  2>&1 | tee -a "$OUT/window.log"
if ! python3 -c '
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
ok = len(rows) == 6 and all(row.get("verdict") == "EXACT" for row in rows)
raise SystemExit(0 if ok else 1)
' "$OUT/decisive/rows.jsonl"; then
  log "DECISIVE CELLS NOT EXACT 6/6 — PEDANTIC is not sufficient. Stopping; returning evidence."
  exit 3
fi
log "all decisive cells EXACT — the compute mode is causal"

log "=== 4. deep ladder 250k / 350k / 475k (250k matches 6d32's known-good cell) ==="
python3 v20_gate2_needle_ladder.py --base "$BASE" --out "$OUT/ladder" --only 250000 350000 475000 \
  2>&1 | tee -a "$OUT/window.log"

log "=== 5. price PEDANTIC against the measured fa71 baseline curve ==="
log "baseline cold prefill server tok/s: 8k 1411 | 16k 1301 | 32k 1305 | 55k 1222 | 100k 1151 | 200k 1051 | 300k 975"
for t in 8000 55000; do python3 prefill_bench.py --base "$BASE" --tokens "$t" --label pedantic-disc 2>&1 | tee -a "$OUT/window.log"; done
for c in 0 16000; do python3 decode_bench.py --base "$BASE" --concurrency 1,4,8,16 --output-tokens 256 --context-tokens "$c" 2>&1 | tee -a "$OUT/window.log"; done

log "=== 6. final identity audit ==="
docker inspect "$CID" --format 'restarts={{.RestartCount}} started={{.State.StartedAt}}' | tee -a "$OUT/window.log"
FATALS=$(docker logs "$CID" 2>&1 | grep -ciE 'OOM|illegal|cuBLAS|assert|EngineDead|Xid' || true)
log "fatal signatures in log: $FATALS"
[ "$FATALS" -eq 0 ] || exit 4
log "=== WINDOW COMPLETE ==="
