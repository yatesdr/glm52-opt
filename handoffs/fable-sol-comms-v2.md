# Fable ⇄ Sol comms v2 — as built

**Author:** Fable · **Built:** 2026-07-25 · **Status:** live; `comms doctor` is the health gate

Replaces the flat append-only `fable-sol-comms.md` (477 KB, ~7,900 lines — kept as the
pre-2026-07-25 archive) with a per-channel bus under `comms/`. This document describes what
**exists and is verified**, the decisions behind it, and the few things still open.

Supersedes `design/fable-sol-comms-v2-proposal.md`.

## Why

Four failures cost us cycles on the v20 effort:

1. **Sol missed messages.** Codex doesn't watch files, so he only saw the log when he remembered
   to open it. Twice he re-requested work already delivered (C1 decode, Proof 3).
2. **No read receipts** — neither side knew a message landed, so we hedged and repeated.
3. **One interleaved stream.** prod, dev, code-prep, patches, fabric, proofs all in one file:
   finding the open item in *one* area meant scrolling past every other area.
4. **Assumed sync.** Appends were believed to reach Sol; they didn't.

(4) turned out not to be transport at all — both agents run on Derek's Mac Studio, so the file
was always readable locally. The misses were Sol *not reading it* (now fixed by context
injection) and integration commits living in nested worktree repos, which is a code-repo problem
tracked separately.

## Design in one paragraph

**Isolate by channel, aggregate the inbox, push into context.** One folder per coordination area
so each thread stays clean; one rendered inbox per agent that pulls the *open* items from every
channel so nothing hides in a quiet corner; and delivery to Sol by **codex hook injection** —
his open items are placed in his context at session start and on every prompt, rather than
relying on him to check a file.

## Layout

```
comms/
  bin/comms                  # the CLI (stdlib python3, no deps)              [tracked]
  hooks/inject_inbox.py      # codex context injection for Sol                [tracked]
  README.md                  # operator docs                                  [tracked]
  registry.json              # channels: owner, status, purpose, policy       [runtime, git-ignored]
  STATUS.md                  # rendered dashboard                             [runtime]
  INBOX-fable.md / INBOX-sol.md   # rendered per-agent open items             [runtime]
  FEED.md                    # rendered all-channel feed, newest first        [runtime]
  channels/<name>/
    log.jsonl                # append-only source of truth                    [runtime]
    thread.md                # rendered, newest-first archive                  [runtime]
    cursor.<agent>           # last-read message id = the read receipt        [runtime]
  acks.<agent>.json          # explicit acks, keyed "channel#id"              [runtime]
  .hook.log                  # every hook invocation (proof the path is live) [runtime]
.codex/hooks.json            # codex hook registration for this repo          [tracked]
```

Runtime state is git-ignored so coordination metadata never pollutes model commits, but it lives
at a stable path both agents reference directly. No git-sync, no socket, no network — there are
no machines to bridge.

### Channel = the unit of coordination

- **`owner`** — who is driving it now. Models what we already do live ("Sol leads the fabric
  experiment; hands `dev` back to Fable for model-readiness"). `comms handoff` transfers it and
  logs the transfer as an ack-required message.
- **`status`** — active / paused / archived. Channels are opened on demand and archived when
  done, so the active set stays small.
- **`policy`** — e.g. `prod` = protect CN3, read-only unless incident; `dev` = CN4, experiment
  freely. Encodes the CN3/CN4 rule we already live by.

### Message envelope (one JSON line per message)

```json
{"id": 142, "channel": "dev", "ts": "2026-07-25T13:20:00Z",
 "from": "fable", "to": "sol", "type": "handoff|handoff-owner|question|evidence|status|ack|fyi",
 "subject": "Gate 0 build PASS", "in_reply_to": 141, "needs_ack": true,
 "refs": {"image": "sha256:fa71a0c1…", "compose": "glm52-v20-prod-ready-20260724.yaml"},
 "status": {"last": "…", "in_flight": "…", "next": "…", "blocking": "sol on X"},
 "body": "markdown text"}
```

- **`id` is global and monotonic** across channels, so cross-references work: a `patches` message
  can point at `dev#142`. Isolation for organization, not walls.
- **`refs`** makes image hashes and artifact paths first-class instead of buried in prose.
- **`status`** formalizes Sol's `last / in-flight / next / blocking` footer so the dashboard can
  show live state per channel.

## Reliability — what makes delivery trustworthy

The v1 implementation had a defect that reproduced the exact failure the bus exists to prevent:
`send` allocated an id but never persisted the counter, so **every message in every channel was
id 1**. Consequences: after Sol read a channel once, later messages compared `1 > 1` and were
**silently invisible**; one `comms ack 1` cleared items in all seven channels; and cross-channel
references were impossible. Fixed, with guards so it cannot recur:

| Property | How it is guaranteed |
|---|---|
| Unique, monotonic ids | Allocated under an `flock`, persisted in the same critical section as the append, and **self-healed** from the logs (`max(counter, max_id_in_logs + 1)`) so a lost write can never collide |
| Concurrent writers safe | All mutations serialized on `comms/.lock`; verified with 20 parallel sends |
| No torn reads | Every file write is tmp + `os.replace`; a bad JSONL line is skipped, never fatal |
| Acks can't cross channels | Ack keys are `channel#id`; `comms ack 12` resolves the owning channel, `comms ack dev#12` is explicit |
| Read receipts | Per-channel cursor of the last-read global id; `--peek` reads without consuming |
| Delivery is provable | `comms doctor` checks all of the above plus the hook path; exits non-zero on failure |

### `comms doctor`

Run it any time, cron it if you like. Checks: unique ids · counter ahead of the logs · cursors
sane · renders current · hook script present and executable · hook registered with codex · hook
has actually fired (from `.hook.log`) · `AGENTS.md` carries the instruction · no ack-required item
older than `--stale-min` (default 30). `--repair` fixes legacy damage: renumbers duplicate ids in
timestamp order preserving per-channel order, remaps ack keys, resets stale cursors, re-renders.

Tested: 15 checks against an isolated `COMMS_ROOT`, including the regression that a message sent
*after* a read is still delivered, ack isolation across channels, 20-way concurrent sends, and
doctor's own detection/repair of an injected duplicate id.

## Unattended operation

**Sol.** Codex 0.144.5 reads project hooks from `<repo>/.codex/hooks.json` (the same shape
Homebrew ships in its own repo). `comms install-hook` registers `comms/hooks/inject_inbox.py` for
`SessionStart` and `UserPromptSubmit`; the script emits
`{"hookSpecificOutput": {"hookEventName": …, "additionalContext": …}}` — verified against the
schemas embedded in the codex binary. So Sol's open items are in his context before he does
anything, with no habit required. Hard rules in the script, both learned from failures during
bring-up:

- **Never block.** stdin is read with a 2 s timeout, the CLI call with 8 s. An early version read
  stdin unbounded and hung a codex turn.
- **Always exit 0.** A comms problem must never break his session.
- **Repo-scoped.** It checks the hook's `cwd` and stays silent outside `/Users/derek/glm52-opt`,
  so it never leaks into Derek's other codex projects.
- **Silent when clear**, so quiet turns stay quiet.
- **Logs every invocation** to `comms/.hook.log`, which is what makes `doctor`'s `hook-fired`
  check meaningful rather than aspirational.

Codex trust-gates hooks: the first codex session in this repo prompts to trust them, and that
acceptance is a one-time human action by design. Belt-and-suspenders, `AGENTS.md` also tells Sol
to run `comms inbox` before starting, so the channel works even with the hook disabled.

**Derek.** `comms send` fires a macOS notification for ack-required messages (`COMMS_NOTIFY=0`
disables), so a handoff reaches him away from the terminal. `comms nudge` re-notifies about
ack-required items left past `--stale-min`, rate-limited to once per hour per item — cron or
launchd it for a standing safety net.

## Waking each other

Seeing a message and being woken by one are different problems, and the second is asymmetric.

**Codex can only be woken from inside a live session.** With a Homebrew install and one Sol, the
managed app-server daemon (which would allow `turn/start` injection over a control socket) is
unavailable — it requires OpenAI's standalone install at `~/.codex/packages/standalone/current/`.
Codex also has no scheduling of its own: no cron subcommand, and no job/timer method anywhere in
the app-server protocol. So **if Sol's session is closed, nothing can wake him** — that's a
platform limit, not a design choice.

Within a live session, the `Stop` hook is the wake path
(`comms/hooks/stop_await_instruction.py`). When his turn would end:

- **Items waiting** → `{"decision": "block", "reason": …}` hands them over as his continuation, so
  he works through them instead of parking.
- **Nothing waiting** → it long-polls comms for `COMMS_STOP_POLL_S` (default 60s) and then holds
  him in an explicit **STANDING BY** round, up to `COMMS_STOP_MAX_IDLE` (default 20) consecutive
  rounds before letting him park normally.

The waiting happens in the hook, not the model, so a poll costs **zero tokens** — only a real
delivery or an idle standing-by round costs a turn. The tradeoff is that a running Stop hook holds
his prompt: `comms hold` (or Ctrl-C) releases it immediately, and `comms hold --release` re-arms.
A real prompt from Derek clears the standing-by streak for that session.

The continuation text is Derek's, not mine: `comms/standing-order.md` between the BEGIN/END
markers, editable without touching code. It tells Sol he's held by the hook, how to treat each
message type, and — when nothing is pending — that standing by means proposing rather than acting,
with CN3 off limits.

**Codex hook trust is per entry**, keyed `<file>:<event>:<group>:<index>` with a hash of that
entry, recorded under `[hooks.state]` in `~/.codex/config.toml`. Adding a hook therefore does not
untrust the others, but each new one needs its own acceptance in an interactive session before it
will run. `codex exec` silently skips untrusted hooks, so absence of injection in a headless run
means "not trusted yet", not "broken" — check `comms/.hook.log` to tell the two apart.

**Fable's side** is easier: a Monitor on the comms log wakes me on any new message while my session
lives. Beyond my session there is no built-in local scheduler either (`CronCreate` is session-only,
and cloud routines can't reach this Mac), so a session-independent Fable wake needs a launchd job
invoking headless `claude -p`. Until that exists, stale items escalate to Derek via `comms nudge`.

## Watching it

| want | command |
|---|---|
| live dashboard (channels · open items · recent traffic · stale warnings) | `comms board` |
| live tail of new messages as they're sent | `comms watch [--channel C]` |
| recent traffic, one shot | `comms feed -n 20` |
| current state without running anything | read `comms/STATUS.md`, `comms/FEED.md` |

The two agent sessions stay in their own terminals; the board is the third window that shows the
traffic between them. (No tmux on this box, so it's three windows rather than one split screen.)

## CLI

```
comms send --channel dev --to sol --type handoff --subject "…" \
           [--body "…" | --body-file f | --body-file -] [--ref k=v]… [--reply ID] [--ack]
           [--last …] [--in-flight …] [--next …] [--blocking …] [--notify|--no-notify]
comms inbox [--channel C] [--for AGENT] [--peek] [--json] [--quiet-if-clear]
comms ack ID…                # 12 or dev#12
comms status [--channel C] · comms channels · comms feed [-n N]
comms watch [--channel C] [--from ID] [--interval S] · comms board [--interval S] [--tail N]
comms open NAME --owner A --purpose "…" [--policy "…"]
comms handoff --channel C --to AGENT [--note "…"] · comms archive --channel C
comms doctor [--json] [--repair] [--stale-min N] · comms nudge [--force] · comms render
comms install-hook [--print] [--uninstall]
```

Identity is `$COMMS_AGENT` (`fable` | `sol`), or `--as AGENT`. Add `comms/bin` to `PATH`, or
alias `comms=/Users/derek/glm52-opt/comms/bin/comms`.

`send` appends to the channel log and regenerates that channel's `thread.md`, both `INBOX-*.md`,
`STATUS.md`, and `FEED.md`. `inbox` advances the read cursor unless `--peek`.

## Decisions taken (previously open questions)

| question | decision |
|---|---|
| Same machine? | Yes — both on the Mac Studio, so `comms/` is a shared local directory. No git-sync, no socket, no network. |
| Location | `/Users/derek/glm52-opt/comms/` at the repo root, runtime state git-ignored. Stable path both agents reference. |
| Does codex support push? | Yes — project `.codex/hooks.json`, `SessionStart` + `UserPromptSubmit`, `additionalContext`. Registered. Trust acceptance is one interactive codex run. |
| Migrate the old 7,900-line `.md` into channels as history? | **No.** It stays as a read-only archive; re-importing it would flood the channels and the inbox with settled traffic. New coordination starts in the channels. |
| Dashboard: file or shared URL? | Local (`STATUS.md` + `comms board`). An HTML artifact is easy to add later if a shareable URL is ever wanted. |
| `attachments/` folder with sha256 in the envelope | **Dropped as speculative.** `refs` already carries hashes and paths; artifacts live where they're produced in the repo. Re-add when something actually needs it. |

## Channels (live registry)

| channel | owner | area | policy |
|---|---|---|---|
| `prod` | fable | CN3 production: deploys, incidents | protect CN3; read-only unless incident |
| `dev` | sol | CN4 qualification + model-readiness | CN4 dev; experiment freely |
| `code-prep` | sol | image builds, source integration, Dockerfiles, byte-pinning | — |
| `patches` | sol | PR drafts / upstream #76 #154 #165 #168 #171 | — |
| `fabric` | sol | PCIe/NCCL cross-IIO topology investigation | no-model; Sol-led |
| `proofs` | fable | no-model GPU discriminators: peer/collective matrices, microprobes | — |
| `meta` | fable | process/coordination (comms system, conventions) | — |

`owner` is a single agent — there is no "shared" value. Where we previously said "shared"
(`prod`, `proofs`), the registry names the agent who answers for it; the channel's `policy` line
is what constrains behaviour.

## Still open for Sol

1. **Accept the hook trust prompt** on your next codex session in this repo, then confirm
   `comms doctor` shows `hook-fired PASS`. Until then your inbox arrives via `comms inbox`.
2. **Channel ownership** — the table above reflects the registry. Correct any of it with
   `comms handoff`, and say if any channel should split or merge.
3. **`dev` handback.** `dev#1` reports Gate 0 PASS and Gate 1 serving with PXB, KV pool 507,904
   (clears the 500k production floor; 6d32a0c3 only made 487k). Gate 2 (needle ladder
   50k→475k) resumes when you hand `dev` back to Fable.
