# Task: Implement packed-CKV DCP prefill transport (rev 2)

**For:** Sol
**Requested by:** Derek, 2026-07-17
**Reviewer:** Fable (design gate + code gate + boot execution)
**Basis:** `glm52-prefill-breakthrough-proposal.md` §3, §8 — now IN THIS
FOLDER (md5 ea365b7f). Rev 2 addresses all 8 findings of your gate review.

## Objective

Implement the packed-CKV gather / ownership-inversion prefill path from
your report, gated by `B12X_DCP_PREFILL_TRANSPORT={query,ckv}` with
`query` (current path, byte-identical behavior) as the hard default.

**Scope cut (your finding 3): `auto` / dynamic crossover is OUT of v1.**
Transport is a PROCESS-LEVEL choice made once at startup. In `ckv` mode
the fp8-query staging and DCP-RS ring must never initialize (their lazy
allocations are on paths `ckv` bypasses — state in the design note how
you guarantee that). This removes transport-stack coexistence from v1
entirely; the hybrid selector is a later phase, gated on v1's numbers.
Consequence: v1 targets the ≤64k test profile where CKV always wins the
byte equation. Full-context (480k) sizing is still REQUIRED in the
design note (see Memory), but v1 acceptance runs at the test profile.

## Patch base and permitted output set (your finding 2)

The tested baseline is the **v14eq composite stack**. Exact base files
and md5s — your patches must be produced AGAINST these bytes:

| File you may modify | Base | md5 |
|---|---|---|
| `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` | `sol-workspace/current-patches/b12x_mla_sparse.py` | f4462905 |
| `vllm/v1/attention/ops/common.py` | `sol-workspace/current-patches/common.py` (collective-safe rev) | 255bde14 |
| `b12x/distributed/pcie_dma.py` | `sol-workspace/current-patches/pcie_dma.py` | 0cb86590 |
| `vllm/model_executor/layers/attention/mla_attention.py` | `sol-workspace/v14-overlays/overlays/vllm/.../mla_attention.py` (v1.4 overlay) | 998654b5 |

No other file may be modified. If the design genuinely needs another
seam, stop and raise it at the design gate with the reason.
`sol-workspace/pristine-v13/` is REFERENCE ONLY (pristine v1.3 for
understanding stock behavior) — do not patch against it.

Note the base `common.py` (255bde14) already contains: module logger,
`B12X_DCP_RS_WIRE` gate, RS-phase profiler hooks, and the
**group-vote pattern for collective-safe init** (added tonight after a
rank-divergent ring init deadlocked a full-context boot: rank 0 OOM'd
its slab and fell back to NCCL while ranks 1–3 entered the ring). Your
CKV communicator init MUST use the same pattern: local init → MIN
all-reduce vote over the DCP group → all-or-nothing adoption, losers
`close()` + discard. Any per-rank runtime routing decision must be
provably rank-invariant (deterministic in shape/dtype/config only).

## Communicator contract (your finding 3)

- ONE process-wide CKV communicator, fixed capacity, created once —
  never per layer, never per observed byte size. (The RS ring's
  payload-size keying built a fresh ~96 MB slab per distinct tail-chunk
  length; found and fixed tonight — don't repeat it.)
- Capacity sized from config (max eligible context at the chosen
  profile), stated in the design-note bill.
- Byte-preserving wire (records are already quantized): construct with
  `fp8="0"` if you reuse `PCIeDmaAllReduce`, or justify a new schedule.

## Memory (your findings 3+4)

The design note must show a **peak-coexistence bill**, not per-route
bills: every buffer alive simultaneously during an eligible chunk in
`ckv` mode — gathered-CKV region, LOCAL query (3072 × 16 × 576 × 2
= 54 MiB — the backend still needs it contiguous), remap tables, live
attention scratch, communicator slab + IPC scratch — at BOTH the 64k
test profile and 480k/609,280-token pool. Your own numbers to beat:
480k CKV + local Q = 222.5 MiB > the 216 MiB gathered-Q region, and
267.8 MiB at the full pool — so state explicitly what layout v1 uses,
what it costs in KV-pool tokens at 2380 blocks, and what `ckv` mode
does when the layout does NOT fit: **fail at startup with a clear
message — never a runtime fallback** (runtime transport divergence is
the deadlock class we just fixed).

All simultaneous workspace views from ONE `get_simultaneous` call
(offset-0 packing aliases separate calls — comment at
b12x_mla_sparse.py:903). Lazy-alloc + `empty_cache()` idiom for
anything outside the workspace. No hot-path allocations. IPC slabs are
non-PyTorch memory — `empty_cache()` cannot reclaim them; count them
separately in the bill.

## Transport semantics — normative (your finding 5)

Define in the design note, and implement exactly:

1. `query` (default, env unset or unrecognized value → `query` + one
   startup warning): behavior byte-identical to base. Unknown values
   must NOT boot a half-configured mode.
2. `ckv`: eligibility = the existing workspace-gather gate (sparse
   B12X AG/RS prefill, 1025–3072 rows, not capturing, DCP > 1) AND
   pure-prefill chunks only — mixed prefill/decode rows and the ≤16-row
   a2a path are untouched. Ineligible chunks in `ckv` mode take the
   UNMODIFIED current path (this is chunk-shape-deterministic and
   therefore rank-invariant — state why in the note).
3. Multi-request chunks: packing traverses each request's block table;
   define handling for uneven per-rank context lengths, partial
   interleave tails (dcp-kv-cache-interleave-size 1), invalid/padded
   top-k entries, and prefix-cache holes (cached blocks present on some
   ranks). Each gets a sentence + a CPU test.
4. The remap: global top-k logical IDs → gathered-CKV physical
   offsets. Specify the table layout and its lifetime.

## Stage 2 — correctness harnesses (your finding 6)

1. Collective: byte-exact comparison of the gathered stream vs direct
   concatenation of per-rank shards, using arbitrary byte records
   (not floats — the wire is byte-preserving; there is nothing to
   accumulate). 4 ranks, include uneven tails.
2. **End-to-end ownership-inversion equivalence (the real proof):**
   CPU reference comparing baseline (query-AG → local-KV attention →
   LSE merge) vs proposed (local-Q → gathered-KV attention) on the
   same random inputs — final local-head outputs and LSE equal to
   fp32 tolerance. Must cover: uneven rank lengths, multi-request
   block tables, noncontiguous blocks, invalid top-k entries.
3. Remap unit test: known topk IDs → gathered-view reads equal direct
   shard reads (4 ranks, interleave 1, with holes).

## Instrumentation (your finding 7)

- Extend `_DcpPhaseProf` with tags `ckv_pack`, `ckv_ag`, `ckv_remap`.
- **Route-neutral summary trigger:** the summary must fire on an
  eligible-chunk counter at the dispatch seam, not on the query-gather
  counter (in `ckv` mode the query gather never runs and the current
  trigger would never fire).
- Counters logged with the summary (timing alone can't prove the kill
  conditions): per-route chunk counts, packed bytes gathered, record
  count, wire bytes, local head count, selected-entry count per chunk.
  One line per rank, same self-disabling discipline as the base class.

## Acceptance (your finding 8 — one baseline, three bands)

Baseline: **964 tok/s @ 55k** (v14eq composite, 64k test profile,
MAXLEN=64000 BLOCKS=400 MNBT=3072 FP8_MODE=ring GATHER_FP8=1 RS_RING=1
PROF=1). All runs cold per the paired protocol (prefill_bench.py's
random first block; prefix-cache metric deltas recorded).

| Band | 55k cold prefill | Meaning |
|---|---|---|
| CONFIRM | ≥ 1,253 (≥ +30%) | mechanism proven; proceed to phase 2 (auto/crossover, full-ctx layout) |
| INCONCLUSIVE | +15% … +30% (1,109–1,253) | profile decides: if `ckv_ag` ≤ 7 ms/layer late-55k and query-AG/RS absent, the transport works and the loss is elsewhere — Fable review picks the next step |
| KILL | < +15% (< 1,109) | per your report §8: traffic matches query-sized, attention work ~4x, or RS remains |

Additional gates, all mandatory: query-mode parity (env unset: 55k
within ±3% of 964 and byte-identical code path), full quality-gate
suite (standard + fp8_ext), decode C1 smoke (no regression vs the
config's own baseline — Fable supplies it at test time), §3.4
signatures visible in the profiler line.

## Staged delivery — gates unchanged

1. **Design note** (review gate): peak-coexistence memory bill (64k +
   480k), communicator contract, remap scheme + table layout, transport
   semantics per the normative list, rank-invariance argument, workspace
   `get_simultaneous` layout, startup-failure behavior.
2. **CPU harnesses** (correctness gate): the three tests above,
   committed with outputs.
3. **Integration patch** (code gate): overlay files + unified diff
   against the pinned bases + md5 manifest + boot/bench instructions.

## Server isolation (unchanged)

You do NOT have server access. Do not SSH to cn3 or place any load on
it. Everything is mirrored in `sol-workspace/` (pristine v1.3 extracts,
current patches — common.py now at the collective-safe rev 255bde14 —
harness, full v1.4 checkout). Need another file? List it in the design
note; Fable mirrors it. Your checks are local-only: pyflakes, ast.parse,
CPU harnesses, md5 manifest. Fable runs the in-image import check,
deploy verification, boots, benches, and gates.

## Required reading order

1. `window2-runbook.md` §5b — tonight's complete failure ledger (two of
   its bugs are now constraints in this brief).
2. `glm52-prefill-breakthrough-proposal.md` §3, §5.3, §8 (your report).
3. `sol-workspace/current-patches/` — the patch idiom (env gates, lazy
   alloc, profiler, group vote).
