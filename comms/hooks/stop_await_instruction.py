#!/usr/bin/env python3
"""Codex Stop hook — keep Sol reachable at the end of a turn instead of parking silently.

Sol can only be woken from inside a live codex session (brew install, one Sol, no app-server
daemon), so this hook is the wake path: when his turn would end, it long-polls comms and either
hands him what arrived or holds him in an explicit STANDING BY state.

Contract (verified against codex-cli 0.144.5 embedded schemas):
  stdin  : stop.command.input — has hook_event_name, cwd, session_id, stop_hook_active, turn_id
  stdout : {"decision": "block", "reason": "<continuation prompt>"}  to keep him going
           nothing at all                                            to let the turn end
Codex requires a non-empty `reason` with decision:block ("Stop hook returned decision:block
without a non-empty reason") and ignores a block with no prompt.

Cost model: waiting happens HERE, not in the model. A poll costs zero tokens; only an actual
delivery (or an idle standing-by round) costs a turn. That's why the poll window is generous.

Escape hatches, because a running Stop hook holds Sol's prompt:
  * `comms hold`  -> returns immediately and stops polling (Derek reclaims the session)
  * Ctrl-C in his session
  * COMMS_STOP_MAX_IDLE consecutive idle rounds -> let him park normally
"""
import json, os, select, subprocess, sys, time, datetime, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
CLI = REPO / "comms" / "bin" / "comms"                  # the tool always comes from this checkout
# ...but the state it reads/writes follows COMMS_ROOT, so the hook is testable in isolation
# and always agrees with whatever root the CLI is using.
COMMS = pathlib.Path(os.environ.get("COMMS_ROOT", REPO / "comms"))
LOG = COMMS / ".hook.log"
STATE = COMMS / ".stop-state.json"
HOLD = COMMS / ".sol-hold"
ORDER = REPO / "comms" / "standing-order.md"             # Derek's wording lives with the code

POLL_S = float(os.environ.get("COMMS_STOP_POLL_S", 60))       # long-poll window (tokens: free)
TICK_S = 2.0
MAX_IDLE = int(os.environ.get("COMMS_STOP_MAX_IDLE", 20))     # consecutive standing-by rounds
STDIN_WAIT_S = 2.0
CLI_TIMEOUT_S = 8.0
# Announce idleness to Fable so she can assign work instead of Sol burning a turn a minute
# saying "standing by" (observed 11 consecutive rounds, 2026-07-25).
ANNOUNCE_EVERY_S = float(os.environ.get("COMMS_STOP_ANNOUNCE_EVERY_S", 900))


def poll_window(round_n):
    """Back off while idle: a longer hook wait costs zero tokens but saves a whole model turn.

    HARD CEILING: codex kills a hook at the `timeout` in its hooks.json entry, and that timeout is
    part of the trust hash — raising it untrusts the hook until someone re-accepts it. Trusted
    timeout is 75s, so windows stay <=60s. Idle turn-burn is instead solved by announcing
    availability to Fable, which converts idleness into assigned work.
    """
    return min(POLL_S, 60.0)


def log(note):
    try:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with LOG.open("a") as f: f.write(f"{ts} Stop {note}\n")
    except Exception:
        pass


def read_stdin_json():
    try:
        if not select.select([sys.stdin], [], [], STDIN_WAIT_S)[0]: return {}
        return json.loads(sys.stdin.read() or "{}")
    except Exception:
        return {}


def load_state():
    try: return json.loads(STATE.read_text())
    except Exception: return {}


def save_state(s):
    try: STATE.write_text(json.dumps(s, indent=2) + "\n")
    except Exception: pass


def idle_count(session):
    return int(load_state().get(session, {}).get("idle", 0))


def set_idle(session, n, announced_at=None):
    s = load_state()
    prev = s.get(session, {})
    s[session] = {"idle": n, "at": time.time(),
                  "announced_at": announced_at if announced_at is not None else prev.get("announced_at", 0)}
    save_state(s)


def announce_idle(session, round_n):
    """Post 'standing by, available' to Fable as Sol, so her Monitor wakes her and she can
    assign work. Rate-limited per session; returns True if a message was sent."""
    st = load_state().get(session, {})
    if time.time() - float(st.get("announced_at") or 0) < ANNOUNCE_EVERY_S:
        return False
    body = (f"Standing by and available for assignment — no comms items pending, "
            f"standing-by round {round_n}. Session {session[:8]}. Held at end of turn by the "
            f"Stop hook, so anything you send lands within one poll window (<=3 min) without "
            f"Derek relaying. If you have measurement, source-review, or drafting work that "
            f"does not need a restart, send it and I will pick it up.")
    try:
        subprocess.run([sys.executable, str(CLI), "send", "--channel", "meta", "--to", "fable",
                        "--type", "status", "--subject", "Sol standing by — idle, available for assignment",
                        "--body", body, "--no-notify",
                        "--last", f"idle since round 1 (session {session[:8]})",
                        "--in-flight", "nothing", "--next", "awaiting assignment",
                        "--blocking", "no work queued"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=CLI_TIMEOUT_S,
                       env={**os.environ, "COMMS_AGENT": "sol"})
        log(f"announced idle to fable (round {round_n}, session {session[:8]})")
        return True
    except Exception as e:
        log(f"FAIL idle announce: {type(e).__name__}")
        return False


def inbox_text():
    """Sol's open items, or '' when clear. --peek so the hook never consumes his read receipt."""
    try:
        r = subprocess.run([sys.executable, str(CLI),
                            "inbox", "--for", "sol", "--peek", "--quiet-if-clear"],
                           cwd=str(REPO), capture_output=True, text=True, timeout=CLI_TIMEOUT_S,
                           env={**os.environ, "COMMS_AGENT": "sol"})
        return (r.stdout or "").strip()
    except Exception as e:
        log(f"FAIL comms inbox: {type(e).__name__}"); return ""


def standing_order():
    try:
        t = ORDER.read_text()
        if "<!-- BEGIN STANDING ORDER -->" in t:
            return t.split("<!-- BEGIN STANDING ORDER -->")[1].split("<!-- END STANDING ORDER -->")[0].strip()
        return t.strip()
    except Exception:
        return ("You are being held at the end of your turn by the comms Stop hook. Handle any "
                "items below, ack them, and reply on the channel. If none: standing by — do not "
                "start new work, do not touch CN3, propose rather than act.")


def block(reason):
    print(json.dumps({"decision": "block", "reason": reason}))


def main():
    d = read_stdin_json()
    cwd = d.get("cwd") or os.environ.get("PWD") or str(REPO)
    session = str(d.get("session_id") or "unknown")
    if not str(pathlib.Path(cwd).resolve()).startswith(str(REPO)):
        return                                        # not our repo: let the turn end
    if HOLD.exists():
        log(f"hold file present; releasing session {session[:8]}"); return

    # Announce availability before the wait, so work can arrive *during* this poll window.
    round_n = idle_count(session) + 1
    if not inbox_text() and announce_idle(session, round_n):
        set_idle(session, round_n - 1, announced_at=time.time())

    # Deliver immediately if something is already waiting.
    items = inbox_text()
    deadline = time.time() + poll_window(round_n)
    while not items and time.time() < deadline:
        if HOLD.exists():
            log(f"hold during poll; releasing {session[:8]}"); return
        time.sleep(TICK_S)
        items = inbox_text()

    if items:
        set_idle(session, 0)
        log(f"delivered {len(items)} chars to {session[:8]}")
        block(standing_order() + "\n\n--- comms items ---\n" + items +
              "\n--- end comms items ---\n"
              "Handle these now, then `comms ack <channel#id>` each one and reply on the channel.")
        return

    n = round_n
    if n > MAX_IDLE:
        set_idle(session, 0)
        log(f"idle cap {MAX_IDLE} reached; parking {session[:8]}")
        return                                        # let him park; Derek's prompt is free
    set_idle(session, n)
    log(f"standing by {n}/{MAX_IDLE} for {session[:8]} (poll {int(poll_window(n))}s)")
    block(standing_order() +
          f"\n\nNo comms items pending (checked for {int(poll_window(n))}s; standing-by round {n} "
          f"of {MAX_IDLE}). Your availability has been posted to Fable on the meta channel, so if "
          f"she has work it will arrive within a poll window without Derek relaying. Reply in one "
          f"line that you are standing by, and stop.")


if __name__ == "__main__":
    try: main()
    except Exception as e:
        log(f"FAIL unhandled {type(e).__name__}: {e}")
    sys.exit(0)      # never break his session
