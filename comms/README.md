# comms — Fable ⇄ Sol coordination bus

Per-channel coordination between Fable (Claude Code) and Sol (codex), both on Derek's Mac Studio.
One folder per area, isolated logs, **one aggregated inbox** so nothing hides in a quiet channel.
Replaces the flat `fable-sol-comms.md` (kept as the pre-2026-07-25 archive).

Design + rationale: `design/fable-sol-comms-v2.md`. Health gate: `comms doctor`.

## Everyday use

```bash
comms inbox                      # your open items across all channels, newest first
comms ack dev#12                 # clear an item you've handled (bare `12` also works)
comms send --channel dev --to fable --type handoff \
           --subject "..." --body "..." [--ref k=v] [--reply ID] [--ack]
comms status                     # dashboard: every channel's owner / state / open items
comms channels                   # channel list
comms handoff --channel dev --to fable --note "yours for model-readiness"
comms open <name> --owner <a> --purpose "..." [--policy "..."]
comms doctor                     # is the delivery path sound? (non-zero exit if not)
```

Identity = `$COMMS_AGENT` (`fable` or `sol`), or `--as <agent>`. **There is no default** — the CLI
exits with an error if neither is set, and refuses a send whose `to` equals its `from`. Both guards
exist because the original default of `fable` silently recorded Sol's `dev#47` as coming *from*
Fable: a wrong `from` field corrupts the audit trail, which is the one thing this ledger is for.
`comms doctor` flags any self-addressed message still in the logs.

Add `comms/bin` to `PATH` (or alias `comms=/Users/derek/glm52-opt/comms/bin/comms`) so it's a bare
command, and `export COMMS_AGENT=<you>` in your shell profile.

Message types: `handoff · handoff-owner · question · evidence · status · ack · fyi`.
`--last/--in-flight/--next/--blocking` attach a status block that shows on the dashboard.
`--ack` marks a message **ack-required** — it stays in the recipient's inbox until they
`comms ack <channel#id>`. That's our read receipt, and it's what fixes the
"already delivered, re-requested" problem. Reading (`comms inbox`) advances a cursor but does
**not** clear an ack-required item.

## Watching it (Derek)

```bash
comms board                      # live dashboard: channels, open items, recent traffic, stale warnings
comms watch                      # live tail of messages as they're sent
comms feed -n 20                 # recent traffic, one shot
```

Or read the rendered files without running anything: `STATUS.md`, `FEED.md`, `INBOX-*.md`.
`comms send` fires a macOS notification for ack-required messages (`COMMS_NOTIFY=0` to silence),
and `comms nudge` re-notifies about ack-required items left longer than `--stale-min` (default
30 min, rate-limited to once per hour per item) — cron it for a standing safety net.

## Layout

```
comms/
  bin/comms            # the CLI (stdlib python3, no deps)                    [tracked]
  hooks/inject_inbox.py# codex context injection for Sol                      [tracked]
  registry.json        # channels: owner, status, purpose, policy             [runtime, git-ignored]
  STATUS.md            # rendered dashboard                                   [runtime]
  INBOX-fable.md / INBOX-sol.md   # rendered per-agent open items             [runtime]
  FEED.md              # rendered all-channel feed                            [runtime]
  acks.<agent>.json    # explicit acks, keyed "channel#id"                    [runtime]
  .hook.log            # every hook invocation (proof the path is live)       [runtime]
  channels/<name>/
    log.jsonl          # append-only source of truth                          [runtime]
    thread.md          # rendered, newest-first                               [runtime]
    cursor.<agent>     # last-read message id = read receipt                  [runtime]
```

Only `bin/`, `hooks/`, and this README are tracked in git; live logs and renders are local
runtime state (see repo `.gitignore`).

## Reliability

Message ids are **global and monotonic**, allocated under an `flock` and self-healed from the
logs, so a lost counter write can't cause collisions. All mutations are serialized on
`comms/.lock` and written atomically (tmp + `os.replace`). Acks are keyed `channel#id` so an ack
in one channel can never clear another's item.

This matters because the first implementation got it wrong: the id counter was never persisted,
every message was id 1, and a message sent after the recipient had read a channel was **silently
dropped**. `comms doctor` now proves the invariants instead of assuming them:

```
comms doctor                # PASS/FAIL per check, non-zero exit on failure
comms doctor --repair       # renumber duplicate ids, remap acks, reset stale cursors, re-render
comms doctor --json         # machine-readable, for cron
```

Checks: unique ids · counter ahead of the logs · cursors sane · renders current · hook script
present · hook registered with codex · hook has actually fired · AGENTS.md carries the
instruction · no stale ack-required items.

## Codex auto-injection (the push to Sol)

Codex 0.144.5 reads project hooks from `<repo>/.codex/hooks.json`. `comms install-hook` registers
`comms/hooks/inject_inbox.py` for `SessionStart` and `UserPromptSubmit`; it emits
`{"hookSpecificOutput": {"hookEventName": …, "additionalContext": …}}`, so Sol's open items land
in his context before he starts work — no habit required.

The script never blocks (bounded stdin + CLI timeouts), always exits 0, stays silent when the
inbox is clear, stays silent outside this repo, and logs every invocation to `.hook.log`.

**Enable (one time, Sol/Derek):**

```bash
comms install-hook           # writes .codex/hooks.json  (--print to preview, --uninstall to remove)
```

Codex trust-gates hooks: start a codex session in this repo and accept the trust prompt, then
`comms doctor` should show `hook-fired PASS`. If the hook doesn't fire after trusting, check
whether hooks are enabled globally (`hooks.state` in `~/.codex/config.toml`) — `.hook.log` tells
you whether the script ran at all, which separates "codex didn't call it" from "it ran and
stayed quiet".

Belt-and-suspenders: `AGENTS.md` also tells Sol to run `comms inbox` before starting, so the
channel works even with the hook off.

**Trust is per hook entry** (`[hooks.state]` in `~/.codex/config.toml`, keyed
`<file>:<event>:<group>:<index>`). Adding a hook doesn't untrust the others, but each new one needs
its own acceptance in an interactive codex session. `codex exec` silently skips untrusted hooks —
`.hook.log` is how you tell "not trusted yet" from "ran and stayed quiet".

## Waking Sol (the Stop hook)

`comms/hooks/stop_await_instruction.py` runs when Sol's turn would end. Items waiting → it blocks
the stop and hands them over, so he works through them instead of parking. Nothing waiting → it
long-polls for `COMMS_STOP_POLL_S` (default 60s), then holds him in an explicit STANDING BY round,
up to `COMMS_STOP_MAX_IDLE` (default 20) rounds before letting him park.

Waiting happens in the hook, not the model: **a poll costs no tokens**; only a delivery or an idle
round costs a turn. The catch is that a running Stop hook holds his prompt:

```bash
comms hold [--reason "..."]   # release his session now; turns end normally
comms hold --release          # re-arm the wake path
```

Ctrl-C in his session works too. A real prompt from Derek clears the standing-by streak.

The continuation wording is **Derek's**, in `comms/standing-order.md` between the BEGIN/END
markers — edit it there, no code change. It defines what standing by means (propose, don't act;
CN3 off limits), so an idle wake can't turn into Sol inventing work.

**Hard limit worth knowing:** codex can only be woken from inside a live session. No daemon
injection without OpenAI's standalone install, and codex has no scheduler of its own. If his
session is closed, nothing wakes him — stale items escalate to Derek via `comms nudge` instead.

## Fable side

Fable watches with the Monitor tool and uses `comms send` / `comms inbox` instead of appending to
the old `.md`.
