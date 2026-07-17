# Packed-CKV DCP prefill transport — Stage 1 design note

**Status:** design gate only; no harness or integration code has been written.
**Target:** the pinned v14eq composite in
`../glm-5/sol-packed-ckv-implementation-brief.md` rev 2.
**Transport version:** v1, process-level `query` or `ckv`; no dynamic crossover.

## 1. Decision summary

The `ckv` route keeps each rank's 16 absorbed query heads local, packs the
rank-local 368-byte `nvfp4_ds_mla` records referenced by the active requests,
all-gathers those bytes into a rank-major temporary paged cache, remaps the
global logical top-k IDs into that temporary cache, runs a separate 16-head
B12X extend plan, projects with the rank-local `W_UV`, and returns the local
16-head result directly. It does not perform query all-gather, cross-rank LSE
merge, or output reduce-scatter.

The design adds a copy-only `PCIeDmaAllGather` channel in `pcie_dma.py`
instead of using `PCIeDmaAllReduce`. A copy-only all-gather needs three
receive/forward slots at world size four; the all-reduce object allocates six
reduce-plus-gather slots and FP8-related state that packed CKV cannot use.
The specialized object therefore halves the IPC slab and has no numerical
operation capable of changing a packed record.

The v1 packing layout is request-major and block-padded. It is deliberately
simple enough to prove on CPU:

```text
for each source rank:
    request 0's virtual blocks, then request 1's, ...
    each virtual block contributes one full 64-record local page

gathered cache:
    [rank 0 packed pages][rank 1 packed pages]
    [rank 2 packed pages][rank 3 packed pages]
```

Duplicate prefix-cache blocks referenced by two requests are copied twice in
v1. That avoids a cross-rank physical-block dictionary in the correctness
path. The 64k acceptance profile still fits at its configured maximum of
eight 64k requests. The corresponding eight-request 480k bound does not fit
and is rejected at startup. Physical-block deduplication is a phase-2/full-
context layout, not an implicit v1 fallback.

No fifth overlay file is needed.

## 2. Pinned base and intended touch map

| File | Base md5 | Intended change |
|---|---:|---|
| `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` | `f4462905` | env parsing, fixed-capacity checks, metadata buffers, pack/remap kernels, 16-head plan, CKV workspace views, singleton initialization, profiler counters |
| `vllm/v1/attention/ops/common.py` | `255bde14` | hard-disarm the optional DCP RS ring when process transport is `ckv` |
| `b12x/distributed/pcie_dma.py` | `0cb86590` | copy-only fixed-capacity `PCIeDmaAllGather` with `close()` and exact contracts |
| `vllm/model_executor/layers/attention/mla_attention.py` | `998654b5` | route dispatch, preserve local tuple Q, local projection, and bypass LSE merge/RS |

`query` mode does not enter any new pack, remap, CKV planning, communicator,
or projection branch. Existing query code remains in place rather than being
refactored through a shared abstraction.

## 3. Process-level transport semantics

The environment is normalized once per process:

- unset or `query`: `query`;
- exactly `ckv`: `ckv`;
- every other value: `query` plus one warning per process.

In `query`, all existing feature gates retain their current meanings and the
forward control flow is the pinned base.

In `ckv`, construction performs rank-invariant configuration validation. The
strict v1 production geometry is:

- B12X sparse MLA, `nvfp4_ds_mla`, 368-byte records;
- TP4/DCP4, PCP1, interleave size 1;
- 16 local heads, 64 current gathered heads, head dimension 576;
- latent/value dimensions 512/256;
- block size 64, top-k 2048, MNBT 3072;
- non-DBO workspace operation.

Any mismatch is a startup error that names the failed field and tells the
operator to use `query`. There is no partial CKV mode.

### 3.1 Eligible chunks

An invocation uses CKV only when all of the following are true:

1. process transport is `ckv`;
2. the pinned workspace-gather route would otherwise be eligible: sparse
   B12X large-row AG/RS, 1025–3072 rows, persistent BF16 `W_UV`, DCP > 1,
   non-DBO, not A2A/B12X-small, and not stream capture;
3. `num_decode_tokens == 0`;
4. `num_prefill_tokens == num_mqa_tokens` and is nonzero.

The decision uses only process configuration, scalar tensor shapes, and
metadata counts constructed identically on all DCP ranks. It does not inspect
rank-local packed counts or allocation success. Therefore every rank enters
the same collective sequence.

Decode, MTP verification, the <=16-row A2A route, small/tail prefills,
captured work, and mixed prefill/decode invocations remain query-owned.

### 3.2 No coexistence with the old lazy transport allocations

There is one tension in the rev-2 wording that needs an explicit gate ruling.
It requires both (a) mixed/ineligible chunks to take the current path and (b)
the FP8-query staging and DCP-RS ring to *never* initialize in a `ckv`
process. A 1025–3072-row mixed batch can satisfy the old workspace gate and
would initialize both optional overlays if they remain armed.

This design gives the no-coexistence rule precedence:

- `b12x_mla_sparse.py` treats FP8 workspace-query gathering as armed only
  when process transport is `query`;
- `common.py` treats the optional DCP RS ring as armed only when process
  transport is `query`;
- an ineligible `ckv` invocation otherwise executes the existing BF16/NCCL
  query path and unchanged attention/LSE mathematics.

Thus `_fp8_staging_pool`, `_fp8_dequant_noPE`, `_fp8_dequant_RoPE`, and the
RS-ring singleton remain uninitialized for the lifetime of a `ckv` process,
including a mixed batch. Query mode is unaffected. Approval of this
interpretation is required at the design gate; retaining the two optional
wire optimizations for a large mixed batch would contradict the explicit
no-coexistence requirement.

## 4. Logical and physical record layout

For v1's fixed interleave `I=1`, DCP world `W=4`, and local page size 64,
one virtual block spans `64 * 4 = 256` global logical token IDs.

For request `r` with global sequence length `L[r]`:

```text
blocks[r]   = ceil(L[r] / 256)
req_base[r] = sum(blocks[j] for j < r)
B           = sum(blocks[r])       # packed blocks per source rank
```

`req_base` is an int32 vector of `num_reqs + 1` entries. Its CPU-pinned and
GPU buffers are allocated once by the metadata builder at maximum sequence
count and reused. `L[r]` comes from the global, CPU-side sequence-length
metadata used by the indexer, not `dcp_local_seq_lens`. An upper bound may
pad the last page but may never be below the actual length.

Each source rank packs `B` full local pages. For packed block
`req_base[r] + vb`, it reads physical block `block_table[r, vb]` from that
rank's native cache and copies all `64 * 368` bytes. Physical block IDs may
be arbitrary and noncontiguous. Copying full pages makes the production
collective length equal on every rank despite uneven final token ownership.
Records after the logical causal tail are never remapped and therefore are
not read by attention.

A one-byte validity entry accompanies every packed block. A negative or
padded block-table entry produces a zero-filled packed page and validity 0.
The local record pages and validity bytes form the communicator's source
payload. Received record portions go to the query workspace; received
validity portions go to a small scratch tail. This allows each destination
rank to test the *owner's* block validity rather than assuming all four local
block tables contain the same physical holes.

The B12X cache view is contiguous:

```text
uint8 gathered_records[4, B, 64, 368]
```

It is exposed to the existing kernel as
`[4 * B, 64, 368]`, preserving the required native page stride of
`64 * 368` bytes. The validity table is not part of the kernel cache view.

### 4.1 Top-k remap

For query row `t`, request ID `r`, and global top-k logical ID `g`:

```text
virtual_block = g // 256
within_virtual = g % 256
owner = within_virtual % 4
local_offset = within_virtual // 4
packed_block = req_base[r] + virtual_block

gathered_slot = (
    (owner * B + packed_block) * 64 + local_offset
)
```

The entry is valid only if:

- `g >= 0`;
- `virtual_block < blocks[r]`;
- the owner's gathered validity byte for `packed_block` is 1;
- `g` is below that query row's global causal length.

The remap kernel preserves the relative order of valid global top-k entries,
compacts them to the front of the existing `page_table_1` row, fills the
remainder with `-1`, and writes the exact valid count to the existing
`nsa_cache_seqlens` buffer. No new 3072x2048 table is allocated. The input
global top-k buffer and output remap buffer are already separate 24 MiB
resident tensors in the pinned base.

Invalid top-k IDs remain invalid. A top-k ID whose owner reports a missing
block is dropped and increments a profiler-only missing-block counter; that
counter must be zero in acceptance. Prefix-cached, noncontiguous blocks are
ordinary valid block-table entries and are copied exactly like newly
allocated blocks.

The gathered cache and validity table live only through the local attention
launch. They are dead before projection and are overwritten by projection
staging on the same ordered CUDA stream.

## 5. Attention and projection ownership

The base keeps 64-head decode and query-prefill plans untouched. `ckv` adds
one 16-head *extend-only* scratch plan:

```text
q:                 [T, 16, 576] BF16 contiguous
selected_indices:  [T, 2048] int32
kv_cache:           [4 * B, 64, 368] uint8 contiguous
output:             [T, 16, 512] BF16
```

The plan is prewarmed for the same row specializations as the base plan. The
pack and remap Triton kernels are also prewarmed. Temporary prewarm tensors
are released and followed by `empty_cache()` before serving. The CKV IPC
communicator remains lazy and does not load the extension at module import.

The CKV kernel is called without cross-rank LSE output because every local
head has seen the complete selected CKV set. Projection uses the rank-local
contiguous `W_UV` view already validated against the persistent rank-major
weight gather:

```text
[16, T, 512] @ [16, 512, 256] -> [16, T, 256]
```

Projection has no runtime allocation. After attention has consumed CKV:

1. copy the 16-head attention result to a 48 MiB head-major prefix of the
   now-dead query workspace;
2. expose a 24 MiB head-major output in the now-dead local-Q prefix of raw
   scratch;
3. run `torch.bmm(..., out=...)` with local `W_UV`;
4. return its `[T,16,256]` token-major/head-major-strided view directly.

The outer MLA layer sets neither `workspace_gather_used` nor any DCP merge
flag for this result. It does not call `_cp_lse_common`,
`cp_lse_ag_out_rs[_into]`, or any output collective.

Arithmetic stays approximately constant:

```text
base: 64 heads * approximately (topk / 4) local winners
CKV:  16 heads * topk global winners
```

## 6. Workspace plan and lifetime

The existing one-call workspace allocation is retained:

```text
q_workspace:     3072 * 64 * 576 * 2 = 216 MiB
scratch_storage: 3072 * 64 * 512 * 2 = 192 MiB
total:                                      408 MiB
```

Both views still come from the existing single `get_simultaneous` call. CKV
creates only `narrow`/`view` aliases of those returned tensors.

### Eligible CKV phase

| Lifetime | Query workspace | Raw scratch |
|---|---|---|
| pack + all-gather | gathered record destination, up to capacity | local packed payload plus local validity; gathered validity at tail |
| remap + attention | gathered records remain live | contiguous local Q (54 MiB), 16-head attention scratch/output (48 MiB), gathered validity tail |
| local projection | 16-head head-major projection input (48 MiB); CKV dead | projected output (24 MiB) plus still-live attention input (48 MiB); Q and validity dead |

All simultaneously live slices are explicitly disjoint. The 16-head plan is
bound only to its 48 MiB attention-scratch slice, not to the 192 MiB base
plan view.

### Ineligible/query phase

The alternate CKV aliases are not used. The existing query layout remains
`q_workspace = 216 MiB` plus `scratch_storage = 192 MiB`, so small, mixed,
decode, MTP, and capture behavior retain their original storage contracts.

There are no hot-path `torch.empty`, `empty_like`, `cat`, data-dependent
indexing allocations, or workspace growth operations. Metadata buffers,
request bases, validity space, remap output, profiler counters, and event
pairs are all fixed before execution. The communicator is the sole lazy
resident allocation and is created once at the first eligible call.

## 7. Copy-only communicator contract

`PCIeDmaAllGather` is a separate class in the permitted `pcie_dma.py`.
Construction takes:

```text
exchange_group, device, max_local_record_bytes, max_local_metadata_bytes
```

It calls `_load_extension()` only from the constructor. It uses raw
`dma_copy`, flags, counters, streams, and events; it never calls a reduction,
quantization, or dequantization kernel. Packed input/output dtype is uint8.

The public eager-only call accepts preallocated contiguous local record and
metadata sources, preallocated disjoint global record and metadata outputs,
and a rank-length tuple. It validates:

- CUDA device/current device and exact DCP group/rank;
- world size four;
- uint8, positive strides, capacity bounds, and 256-byte slab alignment;
- record bytes divisible by 368 and metadata counts consistent with pages;
- source/destination disjointness, except an explicitly supported exact
  local-owner record slice alias;
- no current-stream capture;
- identical deterministic rank-length tuple on all callers by construction.

The generalized schedule permits unequal rank lengths for the Stage-2 proof.
Production's full-page packing gives equal lengths.

For world size `W`, ring step `k=0..W-2` forwards owner
`(rank-k) mod W` to the next rank and receives owner `(rank-k-1) mod W`.
Step 0 reads the caller's local source; later steps forward the preceding
receive slot. Each received payload is split by copy into its rank-major
record and metadata destinations. A final neighbor handshake prevents the
next layer/call from overwriting a slot still read by its successor. The main
stream drains copy and flag streams before returning.

The slab is:

```text
FLAG_SLOTS * FLAG_STRIDE
+ (world_size - 1) * align_up(max_local_payload_bytes, 256)
```

There is no FP8 stage. `close()` destroys streams/events, closes the IPC
mapping/allocation, and makes later calls fail.

### 7.1 Collective-safe singleton initialization

The CKV channel is one process-wide singleton keyed only by device, DCP group,
and the fixed startup capacity—not by layer or actual payload length.

At first eligible use every rank performs:

1. local construction, catching failure into a local `None`;
2. one MIN all-reduce vote over the DCP device group;
3. adopt only if all four votes are 1;
4. otherwise close every locally successful channel, store a permanent
   failed sentinel, call `empty_cache()`, and raise the same clear fatal CKV
   error on all ranks.

Success is likewise followed once by `empty_cache()`. There is no NCCL/query
runtime fallback after a failed vote. All ranks reach initialization on the
same first eligible layer because eligibility and capacity are rank-invariant.

## 8. Capacity and startup failure

For request-major full-page packing:

```text
max_blocks_per_request = ceil(max_model_len / 256)
max_B = max_num_seqs * max_blocks_per_request
global_record_capacity = 4 * max_B * 64 * 368
local_payload_capacity = max_B * 64 * 368 + max_B validity bytes
```

At startup, `ckv` checks:

- global record capacity <= the 216 MiB query workspace;
- local packed payload plus gathered validity <= raw scratch during pack;
- 54 MiB local Q + 48 MiB local attention scratch + gathered validity <=
  raw scratch during attention;
- 48 MiB projection input <= query workspace and 24+48 MiB projection
  output/input coexistence <= raw scratch.

Failure raises before model execution with the required/available bytes,
`max_model_len`, `max_num_seqs`, DCP/interleave, and the instruction to use
`B12X_DCP_PREFILL_TRANSPORT=query`. No communicator is allocated after a
capacity failure.

## 9. Peak-coexistence memory bill

MiB below means 2^20 bytes. The input split query tuple and logical top-k
table are existing activations/buffers, but they are included because they
are live with CKV. “Incremental resident” distinguishes new HBM from aliases
inside the already resident 408 MiB workspace.

### 9.1 Acceptance profile: MAXLEN=64,000, max sequences=8, MNBT=3072

`max_B = 8 * ceil(64,000 / 256) = 2,000` blocks per source rank.

| Component | Peak MiB | Incremental resident? |
|---|---:|---|
| Existing query workspace | 216.000 | no; reused |
| Existing raw attention scratch | 192.000 | no; reused |
| Max gathered CKV records (`512,000 * 368`) | 179.688 | no; view inside query workspace |
| Max local packed staging (`128,000 * 368`) | 44.922 | no; view inside raw scratch |
| Local contiguous query | 54.000 | no; view inside raw scratch |
| 16-head latent attention output/scratch | 48.000 | no; view inside raw scratch |
| Local projected output | 24.000 | no; later alias inside raw scratch |
| Split local Q source | 54.000 | existing activation |
| Logical top-k input | 24.000 | existing buffer |
| Remapped top-k output | 24.000 | existing `page_table_1` buffer |
| Validity + request-base metadata | <0.010 | fixed workspace/metadata views |
| Copy-only IPC slab, flags included | 134.797 | **yes** |
| Device counters/events/streams | <0.1 known tensor bytes; driver handles not measurable locally | yes, bounded |

The pack/gather peak uses 179.688 MiB of query workspace and about 44.93
MiB of raw scratch. The attention peak uses 54 + 48 + <0.01 = about 102.01
MiB of raw scratch. Neither grows the 408 MiB workspace.

At the actual late-55k single-request chunk:

```text
B = ceil(55,000 / 256) = 215
gathered record bytes = 19.316 MiB
local source bytes    = 4.829 MiB
wire bytes per rank   = 3 * 4.829 = 14.487 MiB (+ tiny validity bytes)
```

The communicator remains sized once for the 64k profile maximum.

The current query-only optional residents do not coexist:

- current code-computable FP8 query staging/dequant buffers: 45.469 MiB;
- current fixed-capacity FP8 RS object: about 144.031 MiB IPC slab plus
  49.500 MiB FP8 stage = 193.531 MiB.

Thus the specialized CKV channel's 134.797 MiB is about 104.2 MiB below
those two code-computable optional residents. The failure ledger's larger
observed 450–550 MiB delta includes allocator/IPC/runtime effects not visible
from tensor shapes; the boot remains ground truth. The test profile already
has about 5 GiB more slack than the full-context profile.

### 9.2 480k and the 609,280-token/2380-block pool

The supplied full configuration still has `max_num_seqs=8`. V1's safe
request-major bound is therefore:

```text
max_B = 8 * ceil(480,000 / 256) = 15,000 blocks/rank
global CKV region = 3,840,000 * 368 = 1,347.656 MiB
local staging = 336.914 MiB
copy-only IPC slab = 1,010.773 MiB
```

Both workspace requirements exceed 216/192 MiB, so v1 `ckv` rejects this
configuration at startup before allocating the 1,010.8 MiB slab. This is the
intended full-profile behavior for phase 1.

For comparison, the two useful lower bounds for phase 2 are:

| Hypothetical bounded layout | CKV region | Local stage | Fixed CKV slab | Equivalent KV-pool cost |
|---|---:|---:|---:|---:|
| One 480k request (`max_num_seqs=1`) | 168.457 MiB | 42.114 MiB | 126.374 MiB | 18,466 global tokens, or 73 blocks |
| Deduplicated physical 2380-block pool (609,280 records globally) | 213.828 MiB | 53.457 MiB | 160.402 MiB | 23,438 global tokens, or 92 blocks |

The conversion uses 78 attention layers and per-GPU bytes per global pool
token of `368 * 78 / 4 = 7,176`. Reserving the pool-capped channel literally
would change 2380 blocks/609,280 tokens to about 2288 blocks/585,728 tokens,
still 1.22x a 480k stream. This is an accounting equivalence only: the
failure ledger showed that reducing a lazy PyTorch KV allocation does not
necessarily create device-free IPC memory. A full-context phase must prove
the allocation order/headroom in a separate boot; v1 makes no fit claim.

The one-request layout demonstrates that CKV plus local Q can coexist by
placing Q in raw scratch rather than beside CKV in the 216 MiB region. The
pool-capped comparison demonstrates the future deduplicated upper bound:
213.828 MiB fits the query region with 2.172 MiB to spare, while 54+48 MiB
fits raw scratch. V1 does not implement that deduplication.

### 9.3 Profiler memory

`B12X_DCP_PROF=0` creates no profiler resources. With profiling enabled for
the acceptance boot, three additional fixed event-pair pools are created for
`ckv_pack`, `ckv_ag`, and `ckv_remap`, plus fixed device scalar counters.
They are test-only and self-disable after one summary. The design does not
claim a full-context fit with profiling enabled.

## 10. Rank-invariance and deadlock argument

The following values are identical across DCP ranks:

- normalized process transport and strict geometry;
- MNBT and actual row count;
- decode/prefill counts and capture state for the invocation;
- global sequence-length upper bounds, request order, and `req_base`;
- packed page count `B` and record/metadata byte lengths;
- communicator capacity and ring step/owner sequence.

Physical block numbers and validity bytes may differ, but they affect only
payload content, never routing or collective lengths. A missing owner page is
represented by validity 0 rather than a local fallback. Local communicator
allocation success is converted into a group MIN vote. Consequently no rank
can enter CKV while another enters NCCL/query, and no rank can issue a
different number or size of ring steps.

## 11. Instrumentation design

`_DcpPhaseProf._PHASES` becomes:

```text
gather, proj, rs, ckv_pack, ckv_ag, ckv_remap
```

The event pool remains fixed and every start/stop retains the current
self-disabling exception behavior.

The summary trigger moves from `stop("gather")` to a dispatch-seam method
called exactly once for every otherwise workspace-eligible layer invocation,
before choosing query versus CKV. It therefore fires in either process mode.
The existing `B12X_DCP_PROF_CALLS` meaning remains “eligible layer calls”
(1200 is about 15 chunks x 78 layers), despite the historical variable name.

The one summary line per rank includes:

- query and CKV eligible-call counts;
- per-phase count, total, and mean milliseconds;
- total and mean packed global bytes;
- total and mean record count;
- total and mean per-rank wire bytes;
- local head count (16 for CKV);
- accumulated selected-entry count and mean per CKV call;
- missing-owner-block count, required to be zero.

Selected-entry and missing-block totals are accumulated by the remap kernel
into profiler-owned preallocated device scalars, avoiding a per-layer
`.item()` synchronization or reduction allocation. Summary synchronization
happens once, as in the current profiler.

The acceptance signature for an all-CKV eligible window is:

```text
gather n=0, rs n=0, ckv_ag n>0,
local_heads=16, missing_blocks=0,
wire bytes proportional to sum ceil(L[r]/256), not query_rows*64.
```

## 12. Stage-2 CPU proof plan

No integration work starts until this note is approved. The next gate will
contain three allocation-free CPU tests and captured output:

1. **Ring schedule:** four arbitrary uint8 payloads with unequal record
   counts, zeros, `0xff`, NaN-like bit patterns, and non-aligned tails.
   Simulate every owner/step/forward slot and compare the final compact
   stream byte-for-byte with direct rank concatenation. Repeat the schedule
   to exercise the no-overwrite handshake model.
2. **Ownership inversion:** random FP32 Q/K/V for four ranks and local head
   groups. Compare baseline query-AG + owner-local attention + stable LSE
   merge against local-Q + gathered-KV attention. Assert local output and
   LSE within declared FP32 tolerance. Cases include uneven lengths,
   multiple requests, arbitrary/noncontiguous physical blocks, shared
   prefix blocks, and invalid top-k entries.
3. **Remap/read equivalence:** four ranks, interleave 1, block size 64,
   request-major pack, known IDs around every 256-token boundary, partial
   tails, noncontiguous block IDs, `-1` entries, and owner-specific holes.
   Every valid gathered read must equal the direct owner-shard byte record;
   invalid/hole entries must remain `-1` and reduce valid counts.

## 13. Design-gate decisions requested

1. Approve the specialized three-slot copy-only communicator rather than a
   six-slot `PCIeDmaAllReduce(fp8="0")`; it is exact and saves half the IPC
   slab.
2. Approve hard-disarming the FP8-query staging and optional RS ring for the
   whole `ckv` process. Large mixed batches then use the existing BF16/NCCL
   query mathematics, satisfying no-coexistence but not retaining those two
   optional wire optimizations.
3. Approve request-major duplicate packing for v1 and the resulting startup
   rejection of the supplied max-sequences=8/full-480k configuration.
   Physical-block deduplication/pool-capped full context remains phase 2.
