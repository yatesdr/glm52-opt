#!/usr/bin/env python3
"""Codex SessionStart / UserPromptSubmit hook — inject Sol's open comms items into his context.

Delivery must not depend on anyone remembering to check a file, so this runs on every turn.

Contract (verified against codex-cli 0.144.5 embedded schemas):
  stdin  : {"hook_event_name": "...", "cwd": "...", ...}   (JSON; may be absent/slow)
  stdout : {"hookSpecificOutput": {"hookEventName": "<Event>", "additionalContext": "<text>"}}
  silent when the inbox is clear, so quiet turns stay quiet.

Hard rules, learned the hard way:
  * NEVER block — stdin is read with a timeout, the CLI call has a timeout. A hook that hangs
    hangs Sol's whole turn.
  * ALWAYS exit 0 — a comms problem must never break his session.
  * Log every invocation to comms/.hook.log so `comms doctor` can prove the path is live.
"""
import json, os, select, subprocess, sys, datetime, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent.parent   # .../glm52-opt
CLI = REPO / "comms" / "bin" / "comms"                         # tool from this checkout
COMMS = pathlib.Path(os.environ.get("COMMS_ROOT", REPO / "comms"))   # state follows COMMS_ROOT
LOG = COMMS / ".hook.log"
STDIN_WAIT_S = 2.0
CLI_TIMEOUT_S = 8.0
LOG_CAP = 200_000

BANNER = ("[comms inbox — open coordination items from Fable. Handle or reply before other work; "
          "clear each with `comms ack <channel#id>`, reply with "
          "`comms send --channel <c> --to fable --type <t> --subject ... --body ...`]\n")

def log(note):
    try:
        line = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {note}\n"
        if LOG.exists() and LOG.stat().st_size > LOG_CAP:
            LOG.write_text("".join(LOG.read_text().splitlines(keepends=True)[-500:]))
        with LOG.open("a") as f: f.write(line)
    except Exception:
        pass

def read_stdin_json():
    """Bounded read: if codex sends nothing (or is slow), carry on rather than hang."""
    try:
        if not select.select([sys.stdin], [], [], STDIN_WAIT_S)[0]: return {}
        return json.loads(sys.stdin.read() or "{}")
    except Exception:
        return {}

def reset_standing_by(session):
    """A real prompt from Derek means the standing-by streak is over; clear it for this session."""
    p = COMMS / ".stop-state.json"
    try:
        s = json.loads(p.read_text())
        if s.pop(session, None) is not None: p.write_text(json.dumps(s, indent=2) + "\n")
    except Exception:
        pass

def main():
    d = read_stdin_json()
    event = d.get("hook_event_name") or "UserPromptSubmit"
    cwd = d.get("cwd") or os.environ.get("PWD") or str(REPO)
    if event == "UserPromptSubmit" and d.get("session_id"):
        reset_standing_by(str(d["session_id"]))
    # Repo-scoped: never leak Sol's coordination inbox into Derek's other codex projects.
    if not str(pathlib.Path(cwd).resolve()).startswith(str(REPO)):
        log(f"{event} skip out-of-repo cwd={cwd}"); return
    try:
        r = subprocess.run([sys.executable, str(CLI),
                            "inbox", "--for", "sol", "--peek", "--quiet-if-clear"],
                           cwd=str(REPO), capture_output=True, text=True, timeout=CLI_TIMEOUT_S,
                           env={**os.environ, "COMMS_AGENT": "sol"})
        out = (r.stdout or "").strip()
    except subprocess.TimeoutExpired:
        log(f"{event} FAIL comms inbox timed out after {CLI_TIMEOUT_S}s"); return
    except Exception as e:
        log(f"{event} FAIL {type(e).__name__}: {e}"); return
    if not out:
        log(f"{event} clear"); return
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event,
                                             "additionalContext": BANNER + out}}))
    log(f"{event} injected {len(out)} chars, {out.count('] from fable')} item(s)")

if __name__ == "__main__":
    try: main()
    except Exception as e:
        log(f"FAIL unhandled {type(e).__name__}: {e}")
    sys.exit(0)   # never block a turn
