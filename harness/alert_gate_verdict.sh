#!/bin/bash
# Speak a gate verdict on Derek's Mac: 3 Glass bells, then ONE short clause.
#
# FAILS CLOSED. An earlier version announced "A16 confirmed" for a REFUTED 0/3 run: the JSON parse
# died on a compile-time SyntaxError, so both counts were empty strings, and `[ "$R" = "$N" ]`
# compared empty-to-empty and evaluated TRUE. Never infer success from an unreadable verdict --
# validate the fields first and say so out loud when they cannot be read.
#
#   usage: alert_gate_verdict.sh <label> <summary.json path or - for stdin>
set -uo pipefail
LABEL="${1:?usage: alert_gate_verdict.sh <label> <summary.json|->}"
SRC="${2:--}"

if [ "$SRC" = "-" ]; then JSON=$(cat); else JSON=$(cat "$SRC" 2>/dev/null); fi

# No backslashes inside f-string expressions; keep it parseable on any python3.
PARSED=$(printf '%s' "$JSON" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    r = d.get("recovered"); n = d.get("fail_count"); c = d.get("control_ok")
    if not isinstance(r, int) or not isinstance(n, int):
        raise ValueError("counts not ints")
    print(str(r) + " " + str(n) + " " + str(c))
except Exception:
    print("UNREADABLE")
' 2>/dev/null)

LINE=""
case "$PARSED" in
  UNREADABLE|"")
    LINE="$LABEL finished. Verdict unreadable. Check the terminal." ;;
  *)
    set -- $PARSED
    R="${1:-}"; N="${2:-}"; C="${3:-}"
    # Every field must be positively valid before any verdict is spoken.
    if ! printf '%s' "$R" | grep -qE '^[0-9]+$' || ! printf '%s' "$N" | grep -qE '^[0-9]+$' \
       || [ "$N" -eq 0 ]; then
      LINE="$LABEL finished. Verdict unreadable. Check the terminal."
    elif [ "$C" = "False" ]; then
      LINE="$LABEL invalid. Control regressed."
    elif [ "$C" != "True" ]; then
      LINE="$LABEL finished. Control state unknown. Check the terminal."
    elif [ "$R" -eq "$N" ]; then
      LINE="$LABEL confirmed. $R of $N recovered."
    else
      LINE="$LABEL refuted. $R of $N recovered."
    fi ;;
esac

# ONE bell, not three: 2026-07-26, three bells upset Derek's dog.
afplay /System/Library/Sounds/Glass.aiff 2>/dev/null
say -v Samantha "$LINE" 2>/dev/null || say "$LINE" 2>/dev/null
echo "SPOKEN: $LINE"
