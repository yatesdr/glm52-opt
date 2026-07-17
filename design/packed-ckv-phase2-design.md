# Expanded-charter design: full-context packed CKV and compute profiling

**Author:** Sol, 2026-07-17  
**Status:** design gate; no server action authorized  
**Basis:** `sol-charter-expansion.md`, fit boots 2–5 in
`window2-runbook.md`, packed-CKV Stage 3 commit `860810d`

## 1. Decisions

1. Full-context phase 2 must not allocate the fixed 160 MiB CE receive slab.
   Its first implementation should gather byte records with NCCL directly
   into the existing 216 MiB query workspace. The current NCCL communicator
   is already resident; any incremental internal allocation remains a boot
   measurement, not an accounting assumption.
2. Normal calls retain Stage 3's request-major active-page packing. At one
   480k request this gathers 168.457 MiB and stages 42.114 MiB locally.
3. A physical-pool fallback handles prefix sharing or multi-request aliasing
   that makes the logical request-major page count exceed workspace capacity.
   It gathers the native physical pool and owner-specific block tables, then
   remaps top-k IDs through the owning rank's table. It does not gather stale
   pages into attention because only valid table entries are selected.
4. The first full-context target is `BLOCKS=2340`, not another block-cut
   ladder. It preserves 599,040 global tokens, 1.248x a 480k stream.
   A per-process 192 MiB direct-CUDA headroom escrow is armed after warmup,
   held through the first CKV attention, then freed before the first TP/MoE
   transient. The boot passes only if two group-min probes read at least
   150 MiB device-free after release.
5. The measured DCP1 profile is now the short-context recommendation:
   `MAXLEN=64k`, `BLOCKS=1200`, 76,800-token capacity, and 1,621/1,879 tok/s
   at 8k/55k. DCP2 is the unmeasured middle profile; DCP4 remains the only
   480k-capable profile. Bands are based on aggregate live context.
6. DCP1 re-rates the packed-CKV ceiling from the old 1,740 comm-only ledger
   to about 1,700--1,800 tok/s after CKV's residual transport cost. The
   acceptance profiler must explain the gap to the measured 1,879 DCP1 cell.
7. The 22.6 ms profiler is a separate default-off measurement patch. It must
   tag registered/custom-op runtime boundaries and the TP allreduce runtime;
   it must also split the DCP LSE gather/correction and workspace phases that
   the old comm-only ledger left in its remainder. It must not place
   synchronizing timers around every decoder operation.

## 2. Empirical constraints

The full-context fit boots replace the old allocator ledger:

| Profile | Device-free/transient fact | Meaning for phase 2 |
|---|---|---|
| 2380, RS vote fell back | 10.7 MiB free; 36 MiB MoE transient failed | Removing a slab alone is not proof of fit |
| 2340, RS ring resident | 22.69 MiB free; 24 MiB transient missed by 1.3 MiB | Large raw slabs allocate, but float transients do not |
| 2300, RS ring resident | 36 MiB transient still missed by 11 MiB | Forty blocks returned only about 2 MiB usable free |
| 2340, RS off | 36 MiB stock TP transient missed by 5 MiB; PyTorch allocation grew about 210 MiB | Accounting-equivalent free space can be absorbed elsewhere |

Therefore:

- no block cut is credited toward transient headroom;
- explicit residents are counted, but only `cudaMemGetInfo`/driver data at
  the same execution phase can pass the 150 MiB gate;
- failure before the memory probe is automatically below the gate;
- no full-context throughput or quality number is meaningful until the
  headroom gate passes.

## 3. Why Stage 3 rejects 480k

Stage 3 uses the deliberately safe request bound:

```text
B_config = max_num_seqs * ceil(max_model_len / 256)
```

At 480k and eight sequences this is 15,000 pages per source rank:

```text
global records = 4 * 15,000 * 64 * 368 = 1,413,120,000 bytes
local records  =     15,000 * 64 * 368 =   353,280,000 bytes
CE slab        = flags + 3 * align(local records + metadata)
               = about 1,010.8 MiB
```

The 216/192 MiB workspaces cannot hold that bound, so the Stage 3 startup
refusal is correct. Phase 2 changes the capacity proof and collective; it
does not weaken Stage 3 in place.

## 4. Phase-2 layouts

### 4.1 Active request-major layout

For each request `r` with global length `L[r]`:

```text
blocks[r] = ceil(L[r] / 256)
B = sum(blocks[r])
```

Packing and remapping are unchanged from Stage 3. Each source rank packs one
local `[64,368]` page for every request virtual block. Owner validity bytes
remain authoritative. The only collective change is:

```text
NCCL all_gather_into_tensor(local_records,
                            gathered_records_in_query_workspace)
NCCL all_gather_into_tensor(local_validity,
                            gathered_validity_in_raw_scratch)
```

Both destinations and both sources are fixed views from the existing single
`get_simultaneous` workspace borrow. There is no hot allocation and no CE
slab. Equal lengths follow from global request metadata and remain
rank-invariant.

The raw record capacity of the 216 MiB destination is:

```text
B_records = floor(216 MiB / (4 * 64 * 368)) = 2,404 pages/rank
```

Phase 2 deliberately caps the active route one page lower, at
`B_active=2,403`. At 2,404 pages the local record prefix leaves only 4,096
bytes before the fixed 54 MiB local-Q offset, less than the 12,020 bytes of
local plus gathered validity. At 2,403 it leaves 27,648 bytes, enough for the
12,015 validity bytes. This retains Stage 3's simple disjointness proof and
does not reduce the `P=2340,R<=8` no-sharing bound.

At the full-context acceptance request:

```text
B = ceil(480,000 / 256) = 1,875
global records = 480,000 * 368 = 176,640,000 bytes = 168.457 MiB
local records  = 120,000 * 368 =  44,160,000 bytes =  42.114 MiB
wire/rank      = 3 * local = 132,480,000 bytes = 126.343 MiB
```

The 54 MiB local query and 48 MiB 16-head attention scratch retain their
Stage 3 offsets. Validity is consumed by remap before the local-Q prefix
overwrites it.

This collective is source-feasible on the pinned stack. Its current query
workspace path already calls `torch.distributed.all_gather_into_tensor` with
caller-owned input/output views, and `sparse_attn_indexer.py` contains the
same caller-owned PyNCCL/process-group pattern. Phase 2 may use either direct
PyNCCL or the process-group call, but must reject startup if neither is
available. It must not call `GroupCoordinator.all_gather`, whose base path
allocates a new output tensor.

### 4.2 Why physical-pool capacity alone is not a complete proof

Without prefix sharing, all but the last virtual block of each request are
full and consume one physical page on every DCP rank. With `P` physical
pages/rank and `R` requests:

```text
B <= P + R
```

For `P=2340, R<=8`, the bound is 2,348 and fits:

```text
global records = 221,200,384 bytes = 210.953 MiB
local records  =  55,300,096 bytes =  52.738 MiB
```

Prefix sharing breaks that proof: several request/logical blocks can alias
one physical page. Request-major packing would duplicate the page and `B`
could exceed `P+R` despite low physical occupancy. Phase 2 must not assume
block IDs or holes are identical across DCP ranks.

### 4.3 Physical-pool fallback

If runtime-global `B > B_active`, every rank takes a deterministic pool
route:

1. gather the complete native local cache `[P,64,368]` directly into the
   query workspace as `[4*P,64,368]`;
2. gather each rank's current block-table rows into a fixed tiny table
   `[4,num_reqs,table_width]`;
3. for global logical ID `g`, compute owner/local offset as in Stage 3, then
   load `physical = owner_tables[owner,request,virtual_block]`;
4. accept only `0 <= physical < P` and map to
   `(owner * P + physical) * 64 + local_offset`;
5. preserve stable compaction and exact valid counts.

At `P=2340`:

```text
global physical pool = 220,446,720 bytes = 210.234 MiB
query-workspace spare = 6,045,696 bytes = 5.766 MiB
```

Eight 1,875-wide int32 block tables from four owners consume only 240,000
bytes. Their buffers fit raw scratch and are dead before local-Q creation.

The pool route is a correctness/capacity fallback, not the normal fast path.
Repeatedly gathering the whole pool on small-context chunks would erase the
cumulative CKV traffic advantage. Its route count and bytes must appear in
the profiler summary. Its first activation also emits one WARNING per process
with logical `B`, `P`, and gathered bytes so an operationally common pool
route is visible even when the profiler is disabled.

### 4.4 Route invariance and failure behavior

The phase-2 process choices are read once at import/startup. Proposed gates:

```text
B12X_DCP_PREFILL_TRANSPORT=ckv
B12X_CKV_PHASE2_NCCL=1
```

Unset keeps Stage 3 exactly. Phase-2 startup requires NVFP4 geometry,
TP4/DCP4/interleave1, MNBT3072, non-DBO workspaces, and an explicit physical
block override whose gathered pool fits the query workspace.

For an eligible call, `B`, `P`, request order, table width, and the
active-vs-pool comparison are identical across ranks. Invalid owner table
entries affect only remap content. NCCL initialization/operation failure is
fatal on every rank; there is no CE/query runtime fallback.

The mixed/decode/small paths remain Stage 3-ineligible and take the current
non-CKV path. The phase-2 boot must prove those paths fit with FP8 query
staging and the optional RS slab still hard-disarmed.

## 5. Peak memory and the 150 MiB gate

### 5.1 Explicit residents at `BLOCKS=2340`

| Component | MiB | New resident? |
|---|---:|---|
| Existing query workspace | 216.000 | no |
| Existing raw scratch | 192.000 | no |
| Active 480k gathered records | 168.457 | alias in query workspace |
| Active local pack | 42.114 | alias in raw scratch |
| Pool fallback gathered records | 210.234 | alias in query workspace |
| Local Q | 54.000 | alias in raw scratch |
| 16-head attention scratch/output | 48.000 | alias in raw scratch |
| Block-table/route metadata | below 0.5 | small fixed buffers |
| FP8 query staging | 0 | hard-disarmed |
| DCP RS ring/stage | 0 | hard-disarmed |
| CE CKV slab | 0 | phase-2 NCCL route |
| Headroom escrow before first CKV completion | 192.000 | temporary direct `cudaMalloc`; freed once |
| NCCL incremental internals | unknown | boot measurement |

### 5.2 Empirical lower bound and enforced escrow

The fit-boot-3 comparison starts at 22.69 MiB device-free with the optional
FP8 query resident (45.469 MiB) and fixed-capacity RS object (193.531 MiB)
present. Removing both and adding no permanent large resident predicts about
261.7 MiB free after first-path initialization. Fit boot 5 prevents treating
that number as a fit proof: removing the RS object caused PyTorch allocation
to grow by about 210 MiB. Starting instead from fit boot 5 and crediting only
the removed 45.469 MiB FP8 resident predicts roughly 76 MiB, below the hard
bar. The honest empirical interval is therefore about 76--262 MiB.

Phase 2 closes that interval with a headroom escrow rather than an accounting
credit:

1. after graph/kernel/sampler warmup and the existing allocator-cache
   release, allocate exactly 192 MiB with direct `cudaMalloc`, outside the
   PyTorch caching allocator;
2. group-vote the allocation and fail startup on every rank if any rank
   cannot hold it;
3. retain the pointer while request-dependent buffers, the first NCCL CKV
   gathers, the sparse-attention plan, and projection execute;
4. at the end of the first CKV attention/projection, synchronize once,
   `cudaFree` the escrow, and immediately take Probe A before the first
   attention TP allreduce or MoE transient;
5. take Probe B at entry to the next decoder layer, after one full layer's TP
   and MoE paths have run. Freeing a successful 192 MiB direct allocation
   returns driver memory rather than a PyTorch-reserved segment.

The escrow does not manufacture capacity: if context-scaled buffers need its
bytes, the boot fails while the escrow is held. That is the intended hard
behavior. Relative to fit boot 3, replacing 239.000 MiB of optional residents
with the 192 MiB escrow predicts about 69.7 MiB free while held and 261.7 MiB
after release. The falsifiable prediction is therefore that `BLOCKS=2340`
reaches Probe A and reports 240--265 MiB group-min free; the acceptance rule
remains the lower, non-negotiable 150 MiB value.

The escrow is a process singleton, not one allocation per layer. The exact
post-warmup arm seam is the deployed v1.4-equivalent `gpu_worker.py`; release
and both probes are shared process state called by the packed-CKV backend.
Unset phase-2 transport allocates no escrow.

### 5.3 Probe record and gate

The one-time probes record, on every rank:

```text
driver free/total
torch memory_allocated/reserved
route and actual B/P bytes
probe sequence number
group MIN driver-free
```

The escrow-arm line separately records driver-free before and after the
direct allocation plus the local allocation vote, so a failure while the
192 MiB reservation is still held remains diagnosable without reaching a
probe.

Probe A occurs after the first CKV gather/attention/projection has completed
and synchronized, before its first MoE/TP transient. Probe B occurs at the
next layer entry, after one full layer has initialized its recurring paths.
If Probe A is below 150 MiB, Probe B is below 150 MiB, or the first layer
fails before either probe, phase 2 is killed. The performance boot runs with
probe logging disabled, but retains the already-proven escrow/release state
machine. Escrow allocation, group vote, release, and probe failures are all
fatal; there is no unescrowed runtime fallback.

## 6. DCP posture by workload band

With block size 64, DCP world `D`, and `P` local pages/GPU:

```text
global KV capacity = P * 64 * D
required local pages for aggregate context C = ceil(C / (64 * D))
```

Capacities:

| Local pages | DCP1 | DCP2 | DCP4 |
|---:|---:|---:|---:|
| 1200 | 76,800 | 153,600 | 307,200 |
| 1875 | 120,000 | 240,000 | 480,000 |
| 2340 | 149,760 | 299,520 | 599,040 |
| 2380 | 152,320 | 304,640 | 609,280 |

Required pages:

| Aggregate live context | DCP1 | DCP2 | DCP4 |
|---:|---:|---:|---:|
| 64,000 | 1,000 | 500 | 250 |
| 120,000 | 1,875 | 938 | 469 |
| 240,000 | 3,750 | 1,875 | 938 |
| 480,000 | 7,500 | 3,750 | 1,875 |

Recommended profiles:

- **DCP1, short/decode-heavy, measured:** at most 64k configured/aggregate
  context with `BLOCKS=1200`, giving a 76,800-token pool (1.20x 64k).
  The physics boot measured 1,621 tok/s at 8k and 1,879 at 55k. It removes
  query gather, LSE gather/correction, output reduce-scatter, and decode-side
  DCP merge. This is now a shipping candidate, not a projected experiment,
  subject to the normal quality/decode acceptance record.
- **DCP2, middle, measurement required:** above 64k through about 240k
  aggregate context. `P=1200` supplies 153,600 tokens; `P=1875` supplies
  exactly 240,000 with no concurrency reserve, so a 240k service profile
  should use a slightly larger explicit pool. Pair DCP ranks within the two
  local P2P/PLX pairs if vLLM group construction permits. Run direct 64k and
  120k cold cells; do not interpolate throughput from DCP1/DCP4.
- **DCP4, long/concurrent:** above about 240k through a 480k stream. Use
  `BLOCKS=2340` for 599,040-token capacity and phase-2 packed CKV. This is the
  only proposed profile that preserves the stated 480k service envelope.

These are aggregate live-token bands. Eight 40k requests are a 320k workload
and belong in DCP4 even though every individual request is “short.” Prefix
cache reuse changes physical occupancy and must be reported separately.

The new physics cell also changes the performance ledger. DCP4's measured
964 tok/s at late 55k versus DCP1's 1,879 is a 1.95x ratio: total DCP cost is
about 49% of wall, not merely the 45% previously attributed to gather, RS,
and projection. Packed CKV removes the DCP output merge but retains a smaller
CKV gather, packing/remap, and staging cost. Its revised late-55k ceiling is
therefore 1,700--1,800 tok/s. The unexplained difference to 1,879 is a direct
target for the expanded profiler, not generic “compute” overhead.

## 7. The 22.6 ms measurement patch

### 7.1 Required exclusive accounting

The patch records nested CUDA-event phases:

```text
layer_total
  mla_total
    indexer                 (F layers only)
    attention_path
      dcp_query_ag          (query route only)
      ckv_pack/ag/remap/stage
      sparse_attn
      dcp_project
      dcp_lse_ag
      dcp_lse_correct
      dcp_output_rs
    o_proj_total
      tp_ar_attention
  moe_total
    tp_ar_moe
```

Exclusive values are derived after the one-shot sync:

```text
attention_local = attention_path - every tagged DCP/CKV phase
mla_prepare = mla_total - indexer - attention_path - o_proj_total
o_proj_gemm = o_proj_total - tp_ar_attention
moe_compute = moe_total - tp_ar_moe
norm_residual_other = layer_total - mla_total - moe_total
```

The exclusive sum must reproduce `layer_total` within event-rounding error.
The mean measured `layer_total` must reproduce the DCP1 per-layer ledger
within five percent; otherwise the instrumentation perturbs the workload and
its component numbers are invalid.

Counters in the same one-line/rank summary:

- F-layer and S-layer calls;
- rows and accumulated context bucket;
- both TP allreduce counts and bytes;
- DCP LSE-gather bytes, correction rows/heads, and output-RS bytes;
- sparse selected-entry count;
- route-pack rows/experts if already exposed cheaply;
- unaccounted milliseconds and percentage.

Exactly one attention TP allreduce and one MoE TP allreduce are expected per
ordinary profiled layer. A different count is a finding and disables ordinal
classification rather than silently relabeling calls.

### 7.2 Runtime design

Proposed gate:

```text
B12X_COMPUTE_PROF=1
B12X_COMPUTE_PROF_CALLS=<fixed layer-call target>
```

Default off imports constants only and allocates no events/tensors. Enabled
mode preallocates all event pairs and fixed counters, records on the current
stream, synchronizes once at summary, logs one line/rank, then permanently
disarms. Any internal exception also disarms without affecting serving.

Tags must sit at runtime/custom-op boundaries. In particular,
`dcp_lse_correct` wraps `correct_attn_out`, not the entire DCP merge, so the
previously hidden sanitize/correction kernels are distinct from LSE gather,
projection, and output RS. Tags must not call
`cuda.synchronize`, `.item()`, allocate events, or grow a list on the hot
path. If a B12X kernel uses auxiliary streams, its public call must establish
a current-stream completion dependency before the end event is recorded.

### 7.3 Compute-profiler patch surface requiring a design gate

| File | Base/known md5 | Purpose |
|---|---|---|
| new `vllm/model_executor/layers/compute_phase_profiler.py` | new | shared fixed event pool, nesting, counters, summary |
| `vllm/model_executor/layers/mla.py` | `afd7453cfe9e8478f6f09e6e47697b75` in both mirrored v13/v17 | indexer, MLA total, and output-projection boundaries |
| `vllm/model_executor/models/deepseek_v2.py` | deployed md5 must be byte-verified; mirror candidates differ | layer/MoE boundaries and F/S identity |
| Stage 3 `mla_attention.py` | `61f0b3a70d1618c87b2c90ff7384ae19` | attention path and query/CKV route boundary |
| Stage 3 `b12x_mla_sparse.py` | `5092bf945c33cfdb59b124342410d141` | sparse kernel plus existing CKV phase boundaries and selected counts |
| Stage 3 `common.py` | `c0adc0b61a86da1383b0abe016fc1c8b` | split DCP LSE gather/correction from output RS |
| Stage 3 `pcie_dma.py` | `fc796fa9af58d5b63bce85f8d5c195e8` | TP allreduce bytes/counts classified by active profiler region |

The new profiler module plus `mla.py` and `deepseek_v2.py` are outside the
original four-file Stage 3 set. The expanded charter authorizes proposing
them, but implementation waits for Fable to confirm the deployed
`deepseek_v2.py` bytes and approve this seven-file surface.

### 7.4 Phase-2 integration surface

The transport/memory patch is reviewed separately from the measurement-only
profiler:

| File | Base/known md5 | Purpose |
|---|---|---|
| Stage 3 `b12x_mla_sparse.py` | `5092bf945c33cfdb59b124342410d141` | active/pool layouts, direct NCCL gathers, remap, escrow release/probes |
| Stage 3 `mla_attention.py` | `61f0b3a70d1618c87b2c90ff7384ae19` | pool-route metadata and deterministic dispatch |
| Stage 3 `pcie_dma.py` | `fc796fa9af58d5b63bce85f8d5c195e8` | process-singleton direct-CUDA escrow helper; CE gather remains untouched |
| v1.4-equivalent `gpu_worker.py` | `0829a65484d4dd14c385366291e7a25c` in the mirror | arm escrow after post-warmup cache release |

No source file is shared with an unverified base. Fable must byte-confirm the
deployed `gpu_worker.py` before Gate C even though the mirror is expected to
be the active overlay.

## 8. Staged gates

### Gate A — this design

Review the no-slab NCCL choice, active/pool route proof, 150 MiB probes, DCP
bands, 192 MiB escrow, revised 1,700--1,800 ceiling, and the two separately
listed integration/profiler surfaces.

### Gate B — CPU/source harnesses

1. Shared-prefix layout: construct owner-specific aliased block tables where
   logical `B > B_active`; prove the pool remap reads the same 368-byte
   records as direct owner reads.
2. Active layout: prove NCCL-style rank concatenation plus validity produces
   the exact Stage 3 gathered stream for uneven tails/holes.
3. Route determinism: randomized identical global metadata on four ranks must
   select the same active/pool route and byte lengths.
4. Escrow state machine: fake CUDA runtime/group objects prove one allocation
   per process, all-rank vote, exactly-one free, A/B probe ordering, fatal
   failure, and zero activity when phase 2 is unset.
5. Profiler state machine: fake events prove nested/exclusive math, F/S
   counts, exactly-two TP classification, overflow self-disable, and zero
   allocation when disabled.

### Gate C — integration package

Pinned overlays, unified diff, MD5 manifest, pyflakes, `ast.parse`, CPU
harness output, and boot instructions. Fable performs in-image import.

### Server gates — Fable only

1. **64k collective isolation:** same Stage 3 CKV code, change only CE to
   phase-2 NCCL. Require byte/quality parity, `ckv_ag <= 7 ms` late-55k, and
   report the result against the revised 1,700--1,800 target.
2. **480k memory:** `BLOCKS=2340`, one 480k cold request, 192 MiB escrow and
   probes enabled. Require both group-min probes at least 150 MiB. No
   unescrowed or block-cut retry if it fails.
3. **480k acceptance:** probes off; cold throughput, both quality gates,
   decode C1, prefix-cache deltas, phase signature, and no DCP output merge.
4. **DCP posture:** preserve the measured DCP1 1,621@8k/1,879@55k row, then
   run DCP2 64k and 120k cells. Only measured rows enter the shipping matrix.
5. **Compute profiler:** DCP1 8k/64k-safe profile. Require tagged total within
   five percent of the uninstrumented 22.6 ms ledger before choosing the next
   kernel project.

## 9. Falsifiable outcomes

- **Phase 2 confirm:** both probes show at least 150 MiB group-min free after
  the one-time 192 MiB escrow release, no permanent CKV/RS/FP8 slab, full
  480k request completes, query AG and output RS are absent, and quality is
  green.
- **Memory kill:** first layer fails or either probe is below 150 MiB. Do not
  remove the escrow or trade additional blocks for float crumbs.
- **Transport kill:** NCCL CKV exceeds 7 ms/layer at late55k or initializes a
  large new resident that breaks the headroom gate.
- **Pool-route kill:** shared-prefix CPU reads differ from direct owner reads,
  or any rank chooses a different route/length.
- **Profiler kill:** tagged totals perturb/reproduce the layer ledger by worse
  than five percent, TP call classification is not exactly one per region, or
  event resources appear with the env unset.
- **Ceiling check:** late-55k packed CKV below 1,700 tok/s is a measured gap
  to the DCP1 physics cell, not evidence that 1,740 remains a hard compute
  ceiling. Use the expanded phase tags to name that gap before choosing the
  next kernel project.

The boot remains ground truth in every case.
