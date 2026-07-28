#!/bin/bash
# Delivery-guarantee tests for comms. Runs against an isolated COMMS_ROOT copy.
set -uo pipefail
SP=/private/tmp/claude-501/-Users-derek-glm52-opt/9d65d6da-13b1-4d16-9999-5d0e0dd9d967/scratchpad
T=$SP/t; rm -rf "$T"; mkdir -p "$T"
export COMMS_ROOT=$T
C="/Users/derek/glm52-opt/comms/bin/comms"
mkdir -p "$T/channels" "$T/hooks"; cp /Users/derek/glm52-opt/comms/hooks/* "$T/hooks/" 2>/dev/null
pass=0; fail=0
ck(){ if [ "$2" = "$3" ]; then echo "  PASS $1"; pass=$((pass+1)); else echo "  FAIL $1: expected [$3] got [$2]"; fail=$((fail+1)); fi; }

echo "== setup =="
COMMS_AGENT=fable $C open dev --owner fable --purpose t >/dev/null
COMMS_AGENT=fable $C open prod --owner sol --purpose t >/dev/null

echo "== T1: global monotonic unique ids across channels =="
COMMS_AGENT=fable $C send --channel dev  --to sol --type fyi --subject m1 --body b --no-notify >/dev/null
COMMS_AGENT=fable $C send --channel prod --to sol --type fyi --subject m2 --body b --no-notify >/dev/null
COMMS_AGENT=fable $C send --channel dev  --to sol --type fyi --subject m3 --body b --no-notify >/dev/null
ids=$(cat "$T"/channels/*/log.jsonl | python3 -c 'import sys,json;print(",".join(str(json.loads(l)["id"]) for l in sys.stdin))')
ck "ids unique+monotonic" "$(printf '%s' "$ids" | tr ',' '\n' | sort -n | tr '\n' ',' )" "1,2,3,"
ck "next_id persisted" "$(python3 -c 'import json;print(json.load(open("'$T'/registry.json"))["next_id"])')" "4"

echo "== T2: message sent AFTER a read still gets delivered (the original bug) =="
COMMS_AGENT=sol $C inbox --for sol >/dev/null            # advances cursors
ck "inbox clear after read" "$(COMMS_AGENT=sol $C inbox --for sol --quiet-if-clear | wc -l | tr -d ' ')" "0"
COMMS_AGENT=fable $C send --channel dev --to sol --type fyi --subject m4-after-read --body b --no-notify >/dev/null
got=$(COMMS_AGENT=sol $C inbox --for sol --peek --quiet-if-clear | grep -c 'm4-after-read')
ck "post-read message delivered" "$got" "1"

echo "== T3: ack is per channel#id, not global int =="
COMMS_AGENT=fable $C send --channel dev  --to sol --type handoff --subject need-ack-dev  --body b --ack --no-notify >/dev/null
COMMS_AGENT=fable $C send --channel prod --to sol --type handoff --subject need-ack-prod --body b --ack --no-notify >/dev/null
COMMS_AGENT=sol $C inbox --for sol >/dev/null
n=$(COMMS_AGENT=sol $C inbox --for sol --json | python3 -c 'import sys,json;print(json.load(sys.stdin)["count"])')
ck "2 ack-required remain after read" "$n" "2"
devid=$(python3 - <<EOF
import json
print([json.loads(l)["id"] for l in open("$T/channels/dev/log.jsonl") if json.loads(l)["subject"]=="need-ack-dev"][0])
EOF
)
COMMS_AGENT=sol $C ack "dev#$devid" >/dev/null
left=$(COMMS_AGENT=sol $C inbox --for sol --json | python3 -c 'import sys,json;d=json.load(sys.stdin);print(",".join(i["ref"] for i in d["items"]))')
ck "only the dev item cleared" "$left" "prod#$(python3 -c "
import json
print([json.loads(l)['id'] for l in open('$T/channels/prod/log.jsonl') if json.loads(l)['subject']=='need-ack-prod'][0])")"

echo "== T4: bare-id ack resolves to the owning channel =="
COMMS_AGENT=sol $C ack 5 >/dev/null 2>&1 || true
COMMS_AGENT=sol $C inbox --for sol --json >/dev/null
ck "bare ack accepted" "$?" "0"

echo "== T5: concurrent sends never collide (20 parallel) =="
for i in $(seq 1 20); do COMMS_AGENT=fable $C send --channel dev --to sol --type fyi --subject "p$i" --body b --no-notify >/dev/null & done; wait
tot=$(cat "$T"/channels/*/log.jsonl | wc -l | tr -d ' ')
uniq=$(cat "$T"/channels/*/log.jsonl | python3 -c 'import sys,json;print(len({json.loads(l)["id"] for l in sys.stdin}))')
ck "all messages have unique ids under concurrency" "$uniq" "$tot"

echo "== T6: doctor passes on healthy state, fails on injected corruption =="
# State-integrity checks only: hook-installed/agents-md describe the real repo, not this temp root.
integrity(){ $C doctor --json | python3 -c '
import sys,json
c={x["check"]:x["ok"] for x in json.load(sys.stdin)["checks"]}
print("ok" if all(c[k] for k in ("unique-ids","id-counter","cursors","renders")) else "bad")'; }
ck "state integrity ok when healthy" "$(integrity)" "ok"
python3 - <<EOF
import json
p="$T/channels/dev/log.jsonl"; ls=open(p).read().splitlines()
m=json.loads(ls[-1]); m["id"]=1; ls[-1]=json.dumps(m)   # inject a duplicate id
open(p,"w").write("\n".join(ls)+"\n")
EOF
ck "duplicate id detected" "$(integrity)" "bad"
$C doctor --repair >/dev/null; ck "state integrity ok after --repair" "$(integrity)" "ok"
$C doctor >/dev/null 2>&1; ck "doctor exits non-zero while hook unregistered here" "$?" "1"

echo "== T7: repair preserves message count and per-channel order =="
tot2=$(cat "$T"/channels/*/log.jsonl | wc -l | tr -d ' ')
ck "no messages lost by repair" "$tot2" "$tot"
ord=$(python3 -c 'import json;ids=[json.loads(l)["id"] for l in open("'$T'/channels/dev/log.jsonl")];print(ids==sorted(ids))')
ck "dev log still id-ordered" "$ord" "True"

echo "== T8: hook emits a valid codex envelope only when items are open =="
export COMMS_AGENT=sol
HOOKOUT=$(printf '{"hook_event_name":"UserPromptSubmit","cwd":"/Users/derek/glm52-opt","prompt":"x"}' | COMMS_ROOT=$T /Users/derek/glm52-opt/comms/hooks/inject_inbox.py)
ok=$(printf '%s' "$HOOKOUT" | python3 -c '
import sys,json
d=json.load(sys.stdin); h=d["hookSpecificOutput"]
print("ok" if h["hookEventName"]=="UserPromptSubmit" and "comms inbox" in h["additionalContext"] else "bad")' 2>/dev/null)
ck "hook envelope valid (live comms root)" "$ok" "ok"
out2=$(printf '{"hook_event_name":"UserPromptSubmit","cwd":"/Users/derek/other-project","prompt":"x"}' | /Users/derek/glm52-opt/comms/hooks/inject_inbox.py)
ck "hook silent outside the repo" "$(printf '%s' "$out2" | wc -c | tr -d ' ')" "0"

echo "== T9: hook never blocks, even if codex sends no stdin =="
S=$( { time ( (sleep 6; echo '{}') | COMMS_ROOT=$T /Users/derek/glm52-opt/comms/hooks/inject_inbox.py >/dev/null ) ; } 2>&1 | grep -o 'cpu .*total' | tail -1 )
touch "$T/.hook.log"; before=$(wc -l < "$T/.hook.log")
( (sleep 6; echo '{}') | COMMS_ROOT=$T /Users/derek/glm52-opt/comms/hooks/inject_inbox.py >/dev/null & ) ; sleep 4
after=$(wc -l < "$T/.hook.log")
ck "hook returned before stdin closed (logged within 4s)" "$([ "$after" -gt "$before" ] && echo yes || echo no)" "yes"

echo "== T10: inbox --quiet-if-clear prints nothing when clear (hook silence contract) =="
COMMS_AGENT=sol $C inbox --for sol >/dev/null
COMMS_AGENT=sol $C ack $(COMMS_AGENT=sol $C inbox --for sol --json | python3 -c 'import sys,json;print(" ".join(i["ref"] for i in json.load(sys.stdin)["items"]))') >/dev/null 2>&1 || true
ck "silent when clear" "$(COMMS_AGENT=sol $C inbox --for sol --peek --quiet-if-clear | wc -c | tr -d ' ')" "0"

echo "== T11: Stop hook delivers pending items instantly, with the standing order =="
STOP=/Users/derek/glm52-opt/comms/hooks/stop_await_instruction.py
SIN='{"hook_event_name":"Stop","cwd":"/Users/derek/glm52-opt","session_id":"s-t11","stop_hook_active":false}'
COMMS_AGENT=fable $C send --channel dev --to sol --type handoff --ack --subject "T11 pending" --body x --no-notify >/dev/null
out=$(printf '%s' "$SIN" | COMMS_ROOT=$T COMMS_STOP_POLL_S=3 $STOP)
ck "blocks with the item" "$(printf '%s' "$out" | python3 -c '
import sys,json;d=json.load(sys.stdin);r=d["reason"]
print("ok" if d["decision"]=="block" and "T11 pending" in r and "STANDING BY" in r else "bad")')" "ok"

echo "== T12: standing-by rounds are capped, then the turn is allowed to end =="
COMMS_AGENT=sol $C inbox --for sol >/dev/null
COMMS_AGENT=sol $C ack $(COMMS_AGENT=sol $C inbox --for sol --json | python3 -c 'import sys,json;print(" ".join(i["ref"] for i in json.load(sys.stdin)["items"]))') >/dev/null 2>&1 || true
rm -f "$T/.stop-state.json"
SIN2='{"hook_event_name":"Stop","cwd":"/Users/derek/glm52-opt","session_id":"s-t12","stop_hook_active":true}'
r1=$(printf '%s' "$SIN2" | COMMS_ROOT=$T COMMS_STOP_POLL_S=1 COMMS_STOP_MAX_IDLE=2 $STOP | python3 -c 'import sys,json;print(json.load(sys.stdin)["decision"])' 2>/dev/null)
r2=$(printf '%s' "$SIN2" | COMMS_ROOT=$T COMMS_STOP_POLL_S=1 COMMS_STOP_MAX_IDLE=2 $STOP | python3 -c 'import sys,json;print(json.load(sys.stdin)["decision"])' 2>/dev/null)
r3=$(printf '%s' "$SIN2" | COMMS_ROOT=$T COMMS_STOP_POLL_S=1 COMMS_STOP_MAX_IDLE=2 $STOP | wc -c | tr -d ' ')
ck "round 1 holds him" "$r1" "block"
ck "round 2 holds him" "$r2" "block"
ck "round 3 releases (cap reached)" "$r3" "0"

echo "== T13: a real prompt clears the standing-by streak =="
rm -f "$T/.stop-state.json"
printf '%s' "$SIN2" | COMMS_ROOT=$T COMMS_STOP_POLL_S=1 COMMS_STOP_MAX_IDLE=2 $STOP >/dev/null
n1=$(python3 -c "import json;print(json.load(open('$T/.stop-state.json'))['s-t12']['idle'])")
printf '{"hook_event_name":"UserPromptSubmit","cwd":"/Users/derek/glm52-opt","session_id":"s-t12","prompt":"hi"}' \
  | COMMS_ROOT=$T /Users/derek/glm52-opt/comms/hooks/inject_inbox.py >/dev/null
n2=$(python3 -c "
import json;print(json.load(open('$T/.stop-state.json')).get('s-t12',{}).get('idle','cleared'))")
ck "streak counted before prompt" "$n1" "1"
ck "streak cleared by prompt" "$n2" "cleared"

echo "== T14: hold file releases Sol's session immediately =="
COMMS_ROOT=$T $C hold --reason test >/dev/null
ck "hold -> no block" "$(printf '%s' "$SIN" | COMMS_ROOT=$T COMMS_STOP_POLL_S=30 $STOP | wc -c | tr -d ' ')" "0"
COMMS_ROOT=$T $C hold --release >/dev/null

echo
echo "RESULT: $pass passed, $fail failed"
[ "$fail" = 0 ]
