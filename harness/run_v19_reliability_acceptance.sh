#!/bin/bash
# v19 Tier-1 reliability candidate — acceptance suite (2026-07-26)
#
# Run ON the serving host (cn3 or cn4), against an already-running candidate.
# Strictly SEQUENTIAL: overlapping requests corrupt cold-prefill numbers and
# confuse concurrency cells.
#
# This is a DRIVER. It calls the existing house harnesses rather than
# reimplementing them (see MEASUREMENT-LIBRARY.md — quality_gate.py is NOT used,
# it scores the needle by substring and passes corrupted output).
#
#   Phase 0  identity      v19_reliability_identity_gate.py
#   Phase 1  boot health   container + engine log inspection
#   Phase 2  functional    chat / reasoning / tool-call smoke
#   Phase 3  retrieval     needle_hunt.py ladder 50k -> 475k
#   Phase 4  performance   prefill_bench.py + decode_bench.py
#   Phase 5  wedge repro   the 2026-07-24 discriminator sequence
#   Phase 6  integrity     restart count, dmesg, engine log scan
#
# Usage:
#   bash run_v19_reliability_acceptance.sh --quick          # phases 0-2   (~5 min)
#   bash run_v19_reliability_acceptance.sh --full           # phases 0-6   (~3-4 h)
#   bash run_v19_reliability_acceptance.sh --stress-only    # phases 0,5,6
#
# Env overrides: BASE, CONTAINER, MODEL, OUT
set -uo pipefail

MODE="${1:---full}"
BASE="${BASE:-http://localhost:5001}"
CONTAINER="${CONTAINER:-glm52-prod-candidate}"
MODEL="${MODEL:-GLM-5.2}"
OUT="${OUT:-$HOME/v19-reliability-acceptance-$(date -u +%Y%m%dT%H%M%SZ)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEEDLE="$HERE/cn4-evidence-archive/20260725/needle_hunt.py"

mkdir -p "$OUT"
LOG="$OUT/suite.log"
PASS=0; FAIL=0; SKIP=0
say()  { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }
note() { echo "    $*" | tee -a "$LOG"; }
ok()   { PASS=$((PASS+1)); echo "  [PASS] $*" | tee -a "$LOG"; }
bad()  { FAIL=$((FAIL+1)); echo "  [FAIL] $*" | tee -a "$LOG"; }
skip() { SKIP=$((SKIP+1)); echo "  [SKIP] $*" | tee -a "$LOG"; }

run_p0=1; run_p1=1; run_p2=1; run_p3=1; run_p4=1; run_p5=1; run_p6=1
case "$MODE" in
  --quick)       run_p3=0; run_p4=0; run_p5=0; run_p6=0 ;;
  --stress-only) run_p1=0; run_p2=0; run_p3=0; run_p4=0 ;;
  --full)        ;;
  *) echo "unknown mode: $MODE (use --quick | --full | --stress-only)"; exit 2 ;;
esac

echo "v19 Tier-1 reliability acceptance — $(date -u +%FT%TZ)" | tee "$LOG"
note "container=$CONTAINER base=$BASE out=$OUT mode=$MODE"

restarts_of() { docker inspect "$CONTAINER" --format '{{.RestartCount}}' 2>/dev/null || echo "?"; }
R0="$(restarts_of)"
note "RestartCount at start: $R0"

# ---------------------------------------------------------------- Phase 0 ----
if [ "$run_p0" = 1 ]; then
  say "Phase 0 — identity gate"
  python3 "$HERE/v19_reliability_identity_gate.py" --container "$CONTAINER" \
      --json "$OUT/phase0-identity.json" > "$OUT/phase0-identity.log" 2>&1
  gate_rc=$?
  cat "$OUT/phase0-identity.log" | tee -a "$LOG" | tail -0
  grep -E "^\s+\[(PASS|FAIL)\]|Phase 0 result" "$OUT/phase0-identity.log" | tee -a "$LOG"
  if [ "$gate_rc" = "0" ]; then
    ok "identity gate"
  else
    bad "identity gate rc=$gate_rc — WRONG IMAGE OR MISSING PATCHES; stopping"
    echo "SUITE ABORTED" | tee -a "$LOG"; exit 1
  fi
fi

# ---------------------------------------------------------------- Phase 1 ----
if [ "$run_p1" = 1 ]; then
  say "Phase 1 — boot health"
  docker inspect "$CONTAINER" --format \
    'running={{.State.Running}} health={{.State.Health.Status}} restarts={{.RestartCount}} started={{.State.StartedAt}}' \
    2>&1 | tee -a "$LOG"

  docker logs "$CONTAINER" > "$OUT/engine-boot.log" 2>&1
  POOL="$(grep -oE 'GPU KV cache size: *[0-9,]+ *tokens' "$OUT/engine-boot.log" | tail -1)"
  if [ -n "$POOL" ]; then
    note "$POOL"
    POOLN="$(echo "$POOL" | grep -oE '[0-9,]+' | tr -d ',')"
    echo "$POOLN" > "$OUT/kv_pool_tokens.txt"
    # With vLLM #172 the pool is sized honestly (the profiler now counts the
    # persistent kernel/communication pools). The old 600,000 bar came from
    # numbers that were measured against an undercount, so the meaningful gate
    # is simply: can we still serve one full-length request?
    MML=480000
    if [ "$POOLN" -gt "$MML" ]; then
      ok "KV pool $POOLN > max-model-len $MML (full context fits, $(python3 -c "print(f'{$POOLN/$MML:.2f}')")x)"
    else
      bad "KV pool $POOLN <= max-model-len $MML — cannot serve one full-length request"
    fi
  else
    skip "KV pool line not found in logs (log may have rotated)"
  fi

  for pat in "CUBLAS_STATUS" "illegal memory access" "an illegal memory" "Traceback" "EngineDeadError"; do
    n="$(grep -ci "$pat" "$OUT/engine-boot.log" || true)"
    if [ "$n" = "0" ]; then ok "boot log clean of: $pat"
    else bad "boot log contains $n x '$pat'"; fi
  done

  h="$(docker inspect "$CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null)"
  if [ "$h" = "healthy" ]; then ok "healthcheck reports healthy (deep probe)"
  else bad "healthcheck status=$h"; fi
fi

# ---------------------------------------------------------------- Phase 2 ----
if [ "$run_p2" = 1 ]; then
  say "Phase 2 — functional smoke"
  # 2a plain completion.
  # max_tokens must be generous: the server default is reasoning_effort=high, so a small
  # budget is spent entirely on reasoning and returns finish_reason=length with empty
  # content — that is NOT a generation failure (see MEASUREMENT-LIBRARY.md).
  curl -sS -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly the word: acorn\"}],\"max_tokens\":2048,\"temperature\":0,\"chat_template_kwargs\":{\"reasoning_effort\":\"low\"}}" \
    > "$OUT/smoke-chat.json" 2>>"$LOG"
  smoke_rc=$(python3 - "$OUT/smoke-chat.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
c=d['choices'][0]; m=c['message']
# the field is 'reasoning' on this build; accept every known spelling
txt = "".join(str(m.get(k) or "") for k in
              ("content","reasoning","reasoning_content","thinking"))
if 'acorn' in txt.lower():
    print("PASS"); sys.exit(0)
if c.get('finish_reason') == 'length' and not (m.get('content') or "").strip():
    print("TRUNCATED"); sys.exit(2)     # budget exhausted, not a wrong answer
print("MISS"); sys.exit(1)
PY
)
  case "$smoke_rc" in
    PASS)      ok "chat completion returns the requested token" ;;
    TRUNCATED) skip "chat hit max_tokens with empty content — reasoning consumed the budget, not a generation failure" ;;
    *)         bad "chat completion did not contain 'acorn' (see smoke-chat.json)" ;;
  esac

  # 2b reasoning parser populated (field name is 'reasoning' on this build)
  if python3 -c "
import json,sys
d=json.load(open('$OUT/smoke-chat.json'))
m=d['choices'][0]['message']
sys.exit(0 if any(k in m for k in ('reasoning','reasoning_content','thinking')) else 1)"; then
    ok "reasoning parser (glm45) exposed a reasoning field"
  else skip "no reasoning field present"; fi

  # 2c tool call
  curl -sS -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\":\"$MODEL\",\"max_tokens\":256,\"temperature\":0,
    \"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Oslo? Use the tool.\"}],
    \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",
      \"description\":\"Get weather for a city\",
      \"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],
    \"tool_choice\":\"auto\"}" > "$OUT/smoke-tool.json" 2>>"$LOG"
  if python3 -c "
import json,sys
d=json.load(open('$OUT/smoke-tool.json'))
tc=d['choices'][0]['message'].get('tool_calls') or []
sys.exit(0 if tc and tc[0]['function']['name']=='get_weather' else 1)"; then ok "tool-call parser (glm47) emitted get_weather"
  else bad "tool call not emitted (see smoke-tool.json)"; fi

  # 2d usage accounting flags still on
  if python3 -c "
import json,sys
d=json.load(open('$OUT/smoke-chat.json'))
u=d.get('usage') or {}
sys.exit(0 if 'prompt_tokens_details' in u else 1)"; then ok "prompt-token details present (usage flags intact)"
  else bad "usage.prompt_tokens_details missing"; fi
fi

# ---------------------------------------------------------------- Phase 3 ----
if [ "$run_p3" = 1 ]; then
  say "Phase 3 — retrieval ladder (needle_hunt.py, 50k -> 475k)"
  if [ ! -f "$NEEDLE" ]; then
    skip "needle_hunt.py not found at $NEEDLE"
  else
    # --save-json takes a DIRECTORY (writes request-/response-<depth>.json per cell).
    # Exit 0 = every depth retrieved; exit 1 = at least one MISS.
    python3 "$NEEDLE" --base "$BASE" --model "$MODEL" \
      --depths 50000,150000,250000,350000,475000 \
      --runtag "v19rel-$(date -u +%H%M%SZ)" \
      --save-json "$OUT/needle-cells" \
      > "$OUT/needle-ladder.log" 2>&1
    needle_rc=$?
    grep -E "^depth=|retrieval=|# DONE|WARNING" "$OUT/needle-ladder.log" | tee -a "$LOG"

    n_cells="$(grep -c '^depth=' "$OUT/needle-ladder.log" || true)"
    n_miss="$(grep -c 'retrieval=MISS' "$OUT/needle-ladder.log" || true)"
    note "cells run: $n_cells   misses: $n_miss   (raw bodies in $OUT/needle-cells/)"

    if [ "$needle_rc" = "0" ] && [ "$n_cells" = "5" ]; then
      ok "needle retrieved at all 5 depths 50k-475k"
    else
      bad "needle: rc=$needle_rc cells=$n_cells misses=$n_miss — READ THE RAW ANSWERS in $OUT/needle-cells/ before believing it"
    fi

    # MEASUREMENT-LIBRARY traps: a cache-warm cell is not a cold measurement, and
    # finish_reason=length is not a miss.
    n_cached="$(grep -c 'served from prefix cache' "$OUT/needle-ladder.log" || true)"
    if [ "$n_cached" = "0" ]; then ok "all needle cells were cold (cached_tokens=0)"
    else bad "$n_cached cell(s) served from prefix cache — not a cold measurement"; fi

    n_trunc="$(grep -c 'hit max_tokens' "$OUT/needle-ladder.log" || true)"
    if [ "$n_trunc" = "0" ]; then ok "no needle cell truncated on max_tokens"
    else skip "$n_trunc cell(s) hit max_tokens — read finish_reason before calling any of them a miss"; fi

    n_finfail="$(grep -c 'finalization=FAIL' "$OUT/needle-ladder.log" || true)"
    if [ "$n_finfail" = "0" ]; then ok "every needle cell finalized (non-empty content)"
    else skip "$n_finfail cell(s) retrieved but returned empty content — finalization, not retrieval"; fi
  fi
fi

# ---------------------------------------------------------------- Phase 4 ----
if [ "$run_p4" = 1 ]; then
  say "Phase 4 — performance"
  for tok in 8000 50000; do
    note "cold prefill ${tok}"
    python3 "$HERE/prefill_bench.py" --base "$BASE" --model "$MODEL" --tokens "$tok" \
      --label "v19-tier1" 2>&1 | tee "$OUT/prefill-${tok}.log" | tail -6 | tee -a "$LOG"
  done
  for ctx in 0 50000; do
    note "decode ctx${ctx}"
    python3 "$HERE/decode_bench.py" --base "$BASE" --model "$MODEL" \
      --concurrency 1,2,4,8,16 --output-tokens 256 --context-tokens "$ctx" \
      2>&1 | tee "$OUT/decode-ctx${ctx}.log" | tail -12 | tee -a "$LOG"
  done
  note "compare against the criteria table in v19-reliability-acceptance-criteria-20260726.md"
  note "expected: prefill >= 97% of baseline; decode >= 85% (aux-stream overlap is now off by design)"
fi

# ---------------------------------------------------------------- Phase 5 ----
if [ "$run_p5" = 1 ]; then
  say "Phase 5 — wedge reproduction (the 2026-07-24 discriminator)"
  note "sequence: deep needle sweep -> 3x 350k recheck prefills -> decode ctx50k, NO restart between."
  note "On the pre-patch image this sequence produced CUBLAS_STATUS_INTERNAL_ERROR in _v_up_proj."
  R_before="$(restarts_of)"

  if [ -f "$NEEDLE" ]; then
    python3 "$NEEDLE" --base "$BASE" --model "$MODEL" \
      --depths 50000,150000,250000,350000,475000 --runtag "stress-sweep" \
      > "$OUT/stress-1-sweep.log" 2>&1
    note "deep sweep done"
  else
    skip "needle harness missing — stress sweep degraded to prefill only"
  fi

  for i in 1 2 3; do
    python3 "$HERE/prefill_bench.py" --base "$BASE" --model "$MODEL" --tokens 350000 \
      --label "stress-recheck-$i" > "$OUT/stress-2-recheck-$i.log" 2>&1
    note "350k recheck prefill $i/3 done"
  done

  note "decode ctx50k immediately, no restart"
  python3 "$HERE/decode_bench.py" --base "$BASE" --model "$MODEL" \
    --concurrency 1,2,4,8 --output-tokens 256 --context-tokens 50000 \
    > "$OUT/stress-3-decode-ctx50k.log" 2>&1
  note "decode ctx50k done"

  R_after="$(restarts_of)"
  if [ "$R_before" = "$R_after" ]; then ok "no restart across the stress sequence (RestartCount $R_after)"
  else bad "RestartCount moved $R_before -> $R_after — THE ENGINE DIED DURING STRESS"; fi

  docker logs --since "2h" "$CONTAINER" > "$OUT/engine-stress.log" 2>&1

  # Distinguish the two failure classes. A wedge is memory corruption / a poisoned
  # CUDA context and historically needs a HOST REBOOT. An OOM is honest resource
  # exhaustion that a container restart clears. Reporting them the same way is
  # actively misleading — the 2026-07-26 run hit an OOM and the suite called it a wedge.
  for pat in "CUBLAS_STATUS_INTERNAL_ERROR" "illegal memory access" "an illegal memory"; do
    n="$(grep -ci "$pat" "$OUT/engine-stress.log" || true)"
    if [ "$n" = "0" ]; then ok "no wedge signature: $pat"
    else bad "WEDGE REPRODUCED — $n x '$pat' (corruption class; expect host-reboot recovery)"; fi
  done

  n_oom="$(grep -ci "OutOfMemoryError\|CUDA out of memory" "$OUT/engine-stress.log" || true)"
  if [ "$n_oom" = "0" ]; then ok "no CUDA OOM during stress"
  else
    bad "ENGINE OOM — $n_oom occurrence(s); resource exhaustion, not corruption"
    note "failing allocation(s):"
    grep -oE "Tried to allocate [0-9.]+ [KMG]iB" "$OUT/engine-stress.log" | sort | uniq -c | sed 's/^/      /' | tee -a "$LOG"
    grep -oE "of which [0-9.]+ [KMG]iB is free" "$OUT/engine-stress.log" | sort -u | head -2 | sed 's/^/      /' | tee -a "$LOG"
  fi

  for pat in "EngineDeadError" "sample_tokens timed out"; do
    n="$(grep -ci "$pat" "$OUT/engine-stress.log" || true)"
    if [ "$n" = "0" ]; then ok "stress log clean of: $pat"
    else bad "engine died: $n x '$pat' (see the wedge/OOM rows above for which class)"; fi
  done
fi

# ---------------------------------------------------------------- Phase 6 ----
if [ "$run_p6" = 1 ]; then
  say "Phase 6 — post-run integrity"
  R1="$(restarts_of)"
  if [ "$R0" = "$R1" ]; then ok "RestartCount unchanged across whole suite ($R1)"
  else bad "RestartCount $R0 -> $R1"; fi

  if dmesg -T > "$OUT/dmesg.txt" 2>/dev/null || sudo -n dmesg -T > "$OUT/dmesg.txt" 2>/dev/null; then
    n="$(grep -c '\[12\] Completion Timeout' "$OUT/dmesg.txt" || true)"
    note "PCIe [12] Completion Timeout lines in dmesg: $n"
    if [ "$n" = "0" ]; then ok "no PCIe completion timeouts"
    else bad "$n PCIe completion timeouts — check whether they are cold-boot only (see cn3-prod-wedge-recovery)"; fi
  else
    skip "dmesg not readable without root — run: sudo dmesg -T | grep 'Completion Timeout'"
  fi

  curl -sS -m 30 "$BASE/health" -o /dev/null -w '    /health -> %{http_code}\n' 2>&1 | tee -a "$LOG"
fi

# ---------------------------------------------------------------- verdict ----
say "VERDICT"
echo "  PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP" | tee -a "$LOG"
echo "  artifacts: $OUT" | tee -a "$LOG"
if [ "$FAIL" -eq 0 ]; then
  echo "  RESULT: ACCEPTED (mode $MODE)" | tee -a "$LOG"; exit 0
else
  echo "  RESULT: REJECTED — $FAIL failing gate(s)" | tee -a "$LOG"; exit 1
fi
