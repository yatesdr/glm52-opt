# Proposal: Fable ⇄ Sol comms v2 — a `comms/` folder of channels

> **SUPERSEDED 2026-07-25 by `design/fable-sol-comms-v2.md`** (as-built). This file is the
> original proposal, kept for the reasoning trail. Where the two differ, the as-built doc is
> correct: ids/acks/locking work differently from the sketch below, the transport questions are
> decided, `attachments/` and the old-log migration were dropped, and the codex push is a
> project `.codex/hooks.json` hook rather than an `AGENTS.md` habit.

**Author:** Fable · **Date:** 2026-07-25 · **Status:** superseded by the as-built doc

## Why change

`fable-sol-comms.md` (one flat append-only file) carried the whole v20 effort and its audit
trail is valuable — but four failures have cost us cycles:

1. **Sol misses messages.** Codex doesn't watch files, so Sol only sees the log when he
   remembers to open it. Twice he re-requested work (C1 decode, Proof 3) already delivered
   in the file.
2. **Sync is manual/lossy.** The file is Mac-local; my appends didn't reliably reach Sol,
   his integration commits weren't in my repo's git at all. Derek hand-relays.
3. **No read receipts** — neither side knows a message landed, so we hedge and repeat.
4. **Everything is in one stream.** We coordinate on prod, dev, code-prep, patches, fabric,
   proofs… all interleaved in ~7,900 lines. Finding the open item in *one* area means
   scrolling past every other area.

Problem 4 is the one this revision targets: **isolate coordination by area, but keep a single
place to see everything pending.**

## Two design principles

1. **Isolate by channel, aggregate the inbox.** One folder per coordination area so each
   thread stays clean; a single rendered inbox pulls the *open* items from every channel so
   nothing hides in a quiet corner. Isolation for reading, aggregation for acting.
2. **Push = context injection, not "please check the file."** Codex loads `AGENTS.md`/project
   docs each session; whatever is in that loaded surface is what Sol reliably sees. So the
   inbox is a rendered file his harness already ingests. The transport (files now, socket
   later) is secondary to *where the open items surface*.

---

## Structure

```
comms/
  README.md              # how this works (short; also the doc codex loads)
  registry.json          # channel list: name, owner, status(active|paused|archived), purpose, policy
  STATUS.md              # rendered dashboard: every channel's owner / state / open items / blocking
  INBOX-sol.md           # rendered: UNACKED items across ALL channels, grouped by channel, newest first
  INBOX-fable.md         # same, for me
  channels/
    prod/                # cn3 production — policy: protect, read-only unless incident
      log.jsonl          #   append-only messages (source of truth)
      thread.md          #   rendered newest-first (browsable archive)
      cursor.sol         #   last-read message id for Sol in this channel  (read-receipt substrate)
      cursor.fable
    dev/                 # cn4 qualification + model-readiness
    code-prep/           # image builds, source integration, Dockerfiles, byte-pinning
    patches/             # PR drafts / upstream (#76 #154 #165 #168 #171 …)
    fabric/              # PCIe/NCCL topology investigation
    proofs/              # no-model GPU discriminators (peer/collective matrices, microprobes)
    meta/                # process/coordination (e.g. this proposal)
  attachments/           # shared artifacts referenced by messages (sha256 recorded in the envelope)
```

Channels are **created on demand** (an investigation spins up `fabric/`, a build effort
`code-prep/`) and **archived** when done, so the active set stays small. `registry.json` is
the index.

### Channel = the unit of coordination

Each channel carries the things that were implicit before:

- **`owner`** — who is currently driving it. This models exactly what we do live: *"Sol is
  lead for the fabric experiment; hands `dev` back to Fable for model-readiness."* Ownership
  is explicit and logged, so there's never ambiguity about who acts next.
- **`status`** — active / paused / archived.
- **`policy`** — e.g. `prod` = protect cn3, read-only unless incident; `dev` = experiment freely.
  Encodes the cn3/cn4 rule we already live by.

### Message envelope (one JSON line in a channel's `log.jsonl`)

```json
{"id": 142, "channel": "dev", "ts": "2026-07-25T13:20:00Z",
 "from": "fable", "to": "sol", "type": "handoff|question|evidence|status|ack|fyi|handoff-owner",
 "subject": "Gate 0 build PASS", "in_reply_to": 141, "needs_ack": true,
 "refs": {"image": "sha256:fa71a0c1…", "artifacts": [{"path": "…/v3.jsonl", "sha256": "eb8b4e49…"}],
          "see_also": ["patches#88"]},
 "status": {"last": "…", "in_flight": "…", "next": "…", "blocking": "sol on X"},
 "body": "markdown text"}
```

- **Global monotonic `id`** across channels so cross-references work — a `patches` message can
  point at `dev#142` via `see_also`. Isolation for organization, not walls.
- **`refs`** makes hashes/artifacts first-class instead of buried in prose.
- **`status`** formalizes Sol's existing `last / in-flight / next / blocking` footer so the
  dashboard can show live state per channel.

---

## `comms` CLI (thin wrapper both agents call)

```
comms send --channel dev --to sol --type handoff --subject "…" \
           [--ref image=sha256:…] [--reply 141] [--ack-required] --body-file msg.md
comms inbox                 # UNACKED for me across all channels, grouped by channel; marks read
comms inbox --channel dev   # just one channel
comms ack 141
comms channels              # registry: each channel's owner / status / #open
comms status [--channel …]  # dashboard render
comms open fabric --owner sol --purpose "PCIe/NCCL topology" --policy "no-model, Sol-led"
comms handoff --channel dev --to fable            # transfer ownership, logged as a message
comms archive --channel proofs
```

`send` = append to the channel log + regenerate that channel's `thread.md` + regenerate both
`INBOX-*.md` and `STATUS.md`. `inbox` = advance my read-cursor (the read receipt). For me
(Fable) it's a straight swap for grepping the `.md`; I still watch with the Monitor tool.

## How Sol stops missing things (codex injection)

- **`INBOX-sol.md`** holds only unacked items, across all channels, newest first — short by
  construction, never the 7,900-line scroll. One line in Sol's `AGENTS.md`: *"Before starting
  a task, run `comms inbox` and ack what you read."* Codex loads `AGENTS.md` each session, so
  the habit rides in automatically.
- If codex exposes a **notify hook**, point it at `comms inbox --peek` for a terminal poke.
  Without it, the per-session `comms inbox` covers the gap.

## The "live document" — dashboard

`comms status` renders `STATUS.md` (a local file, or a self-contained HTML Artifact if you
want a shareable URL): a **board of channels**, each row = owner · status · current open item ·
who's blocking whom. Top of the page = the aggregated unacked queue. This is the at-a-glance
"where does everything stand across prod/dev/patches/…" we don't have today.

## Transport — decided: shared local directory

Fable and Sol both run on Derek's Mac Studio, so **`comms/` is just a directory both processes
read and write directly.** No git-sync, no push/pull, no socket, no network — those existed only
to bridge machines, and there are none to bridge. This also means the "sync was lossy" failure
was never really transport: the file was always readable locally; the misses were Sol *not
reading it* (fixed by injection) and the code **integration commits** living in nested worktree
repos (a code-repo problem, separate from this channel).

**Location:** `/Users/derek/glm52-opt/comms/` at the repo root, **git-ignored** so coordination
metadata never pollutes model commits, but present locally for both agents to reference by a
stable path. (A sibling `/Users/derek/comms/` outside the repo is equally fine if you'd rather
keep it fully decoupled from the model tree — your call.)

Because we're co-located, a lightweight **local push** is easy if wanted: `comms send` can also
fire a macOS notification (`osascript -e 'display notification'`) and/or ring the terminal bell,
so Derek and a live Sol session get poked immediately — on top of the reliable per-session
`comms inbox`. A socket/WebSocket service would only matter if we ever went multi-host; it isn't
needed and isn't in the plan.

## Rollout

- **Phase 1 (≈1–2h, no cn4):** `comms/` folder, `comms` CLI over per-channel `log.jsonl`,
  cursors+acks, `INBOX-*.md` + `STATUS.md` renders, one line into Sol's `AGENTS.md`. Seed the
  channels below and migrate the existing `.md` into them as history.
- **Phase 2 (opt-in):** socket service behind the same CLI for live push + browser dashboard.
- **Phase 3 (opt-in):** chat bridge (Slack/Discord) for Derek's phone.

### Seed channels (maps our current work)

| channel | area | current owner |
|---|---|---|
| `prod` | cn3 production, deploys, incidents | shared (protect) |
| `dev` | cn4 qualification + model-readiness | Fable (paused, Sol booting) |
| `code-prep` | image builds, source integration, Dockerfiles | Sol |
| `patches` | PR drafts / upstream #76 #154 #165 #168 #171 | Sol |
| `fabric` | PCIe/NCCL topology investigation | **Sol (active)** |
| `proofs` | no-model GPU discriminators | shared |
| `meta` | process/coordination | Fable |

### Open questions for Derek/Sol
1. ~~Same machine?~~ **Answered: yes, both on the Mac Studio → shared local `comms/` directory.**
2. Does codex expose a **notify hook** / reliably reload `AGENTS.md` per session? (Decides whether
   Sol's inbox is fully automatic or leans on the one-line `comms inbox` habit + macOS notification.)
3. Dashboard as a **local file** or a **shared Artifact/URL**?
4. Do these seven seed channels match how you think about the work, or should any split/merge?
5. `comms/` at the repo root (git-ignored) or a sibling `/Users/derek/comms/` outside the model tree?

I can stand up Phase 1 without touching cn4 whenever you want it.
