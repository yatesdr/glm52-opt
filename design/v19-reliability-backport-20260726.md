# v19 reliability backport — source review and minimal patch set

Date: 2026-07-26
Author: Fable (source review only — no changes made to cn3 or cn4)
Scope: identify the smallest set of v20 fixes that address the v19 prod memory-wedge
failures, and confirm each one against the *actual bytes running on cn3*.

---

## 1. What is actually running on cn3 right now

Read directly from the live container (read-only inspection, no restart, no writes):

```text
container    glm52-prod            Up 47h (healthy), RestartCount=0, started 2026-07-24 16:39 UTC
image        ghcr.io/yatesdr/glm52-serve@sha256:ca8481687f71…
             tag gilded-gnosis-v19-int8-block-patched
compose      /home/claude/glm52-prod-ring.yaml   (host file, not the repo copy)
vllm         0.11.2.dev280+gilded.gnosis.v19.vllm7ea567a.b12x4cfa530.fi801d57a.cu132.20260718
b12x         0.30.2  (pure-Python package + .cu compiled at runtime into
                      TORCH_EXTENSIONS_DIR=/cache/int8ext_baked_a826ef58)
model        GLM-5.2 hybrid — hidden 6144, 64 heads, 78 layers, kv_lora_rank 512, v_head_dim 256
runtime      TP4 / DCP4, B12X_MLA_SPARSE, nvfp4_ds_mla KV, KV_FP8_ROPE=1, MTP num_spec=3,
             GMU 0.970, max-model-len 480000, MNS 16, MNBT 3072, async-scheduling,
             OffloadingConnector 64 GB DRAM (secondary_tiers: []), i8_ring wire mode,
             PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

The v19 source base is vLLM fork commit **`7ea567a2`**. The v20 line is based on
`992b874cf` → selected head `89b4a98d1`. Everything below is stated as a delta against
`7ea567a2`, and every "applies clean" claim was verified by extracting `7ea567a2` into a
scratch tree and running `git apply --check`.

---

## 2. Root cause of the v19 wedge — found, and it is a one-file fix

### 2.1 The contract v19 declares but never honors

The prod attention backend declares three layout requirements:

```text
/opt/venv/.../vllm/v1/attention/backends/mla/b12x_mla_sparse.py
  979:  self.force_contiguous_mla_bmm_input  = True
  980:  self.force_contiguous_mla_bmm_weight = True
  981:  self.force_contiguous_mla_bmm_output = True
```

and the consumer that is supposed to act on them **does not exist in v19**:

```text
$ grep -c force_contiguous_mla_bmm .../vllm/model_executor/layers/attention/mla_attention.py
0
```

So B12X_MLA_SPARSE asks for contiguous BMM operands, and vLLM silently ignores the
request. The unguarded call sits at:

```text
mla_attention.py:1476    torch.bmm(x, self.W_UV, out=out.transpose(0, 1))
```

Both operands are strided views. `x` is `attn_out.view(...).transpose(0,1)`; `out` is a
view of `mqa_output_slice`. Under `VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=1` those buffers
come from the workspace manager, which packs every borrowed view from offset 0 — i.e.
**tightly mapped, with a live neighbour immediately after the end of each view.**

### 2.2 That is the exact frame that crashed

`a2a-cublas-crash-spec.md` records the CN3 failure:

```text
mla_attention.py:1476  _v_up_proj
  torch.bmm(x, self.W_UV, out=out.transpose(0, 1))
RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling
  cublasGemmStridedBatchedEx(... CUDA_R_16BF ...)
```

Same file, same line number, same op. The upstream fix — Martin Vit's
**vLLM #136, `b3ea2e8f` "[GG] Fix MLA BMM layout contract for cuBLAS read-ahead"** —
states the mechanism directly:

> Some CUDA BMM algorithms read a full tile beyond strided tensor bounds. Backends with
> tightly mapped buffers opt into contiguous operands so those accesses stay inside the
> logical allocation.

This is an **out-of-bounds tile access against a tightly packed workspace**, not allocator
fragmentation. That reframes the whole incident chain and it matches the observed
symptoms end to end:

| Observed on cn3 2026-07-24 | Explained by |
|---|---|
| `CUBLAS_STATUS_INTERNAL_ERROR`, one rank only | OOB access is data/shape dependent; only the rank whose tile crosses the boundary faults |
| Non-reproducible from a clean arena | Whether the read lands outside the *allocation* (vs. inside a neighbouring view) depends on where the allocator placed the workspace that boot |
| Corrupted output before any crash (the 500-storm) | An OOB **write** into the adjacent workspace view corrupts results without raising |
| Wedged CUDA context → NCCL init hang → PCIe `[12] Completion Timeout` → host reboot required | Sticky illegal-access context poisoning; container restarts cannot clear it |

The "fragmentation under `expandable_segments:False`" hypothesis in the crash spec is
consistent with the data but is not the root cause — it only explains *why the fault
surfaces intermittently*, which the OOB explanation covers as well.

**This is the single highest-value backport and it is 37 lines in one file with no
b12x dependency.**

### 2.3 Second confirmed corruption path: draft and target share one workspace

`WorkspaceManager` in v19 has one buffer per ubatch and no notion of lanes. The MTP
draft runs the same B12X_MLA_SPARSE backend through `speculator.propose()` /
`speculator.capture()` and calls `get_simultaneous()` on **the same lane-0 workspace the
target's captured graphs are bound to**. A draft-side call that grows the workspace frees
the buffer the target's captured graph nodes point at.

`a8b59fbe "fix(workspace): isolate speculative execution buffers"` adds a lane dimension
and wraps every draft-side entry point (`load_model`, `set_attn`, `init_cudagraph_manager`,
`capture`, `propose`, `warmup_capacity_kernels`, `compute_capacities`) in
`use_workspace_lane(1)`. Prod runs `VLLM_USE_V2_MODEL_RUNNER=1` **and** MTP3, which is
exactly the condition that turns on 2 lanes.

---

## 3. Recommended minimal backport (Tier 1)

**7 Python files — 6 vLLM + 1 b12x — and zero compiled-artifact changes.** Verified: the vLLM
patches apply clean to `7ea567a2`, all 7 files compile under the image's py3.12, and there is no
reference to any b12x symbol missing from the deployed `b12x 0.30.2`. The b12x change is pure
Python (the kernel is CuTe-DSL JIT-compiled at runtime), so there is no wheel or CUDA rebuild.

> **Revision 2, 2026-07-26.** The original draft of this section included the v20 capability
> gate (`93735960` + `e5b6cabb`), which *disables* the shared-experts aux-stream overlap and
> costs ~11% decode. That was the wrong call: the overlap is worth keeping, and the hazard has
> a proper fix. See §3.1. The gate is no longer part of the recommended set.

| # | Change | Title | Files | Why for v19 |
|---|---|---|---|---|
| 1 | `b3ea2e8f` | [GG] Fix MLA BMM layout contract for cuBLAS read-ahead (#136) | `mla_attention.py` | Honors the contract v19's own backend declares. Fixes the exact crash frame. **Primary fix.** |
| 2 | `a8b59fbe` | fix(workspace): isolate speculative execution buffers | `workspace.py`, `gpu_worker.py`, `gpu/model_runner.py`, `gpu/warmup.py` | Stops the MTP draft from aliasing/reallocating the target's captured workspace |
| 3 | `ef7cae43` | fix(dcp): preallocate packed A2A graph buffers (#130) | `dcp_alltoall.py` | Retained A2A staging pair is allocated *before* CUDA capture instead of from the shared graph pool, so two descriptors can't alias one address. `is_vllm_cudagraph_capture_active` already exists in v19. 4 lines. |
| 4 | *new* | b12x: cooperative launch for the W4A16 barrier kernels | `b12x/moe/fused/w4a16/kernel.py` | Makes the shared-experts overlap **safe instead of disabled**. See §3.1. |

### 3.1 Why the MoE hazard gets a cooperative launch, not a gate

`W4A16FusedMoeKernel._grid_barrier` is a spin-wait sense-reversal barrier across `grid_x` CTAs:

```python
old_count = atomic_add_global_i32(count_addr, 1)
if old_count == grid_x - 1:   # last arrival releases
    st_global_i32(count_addr, 0); threadfence(); red_add_global_release_i32(sense_addr, 1)
else:
    while sense == old_sense:                      # spins until released
        sense = ld_global_acquire_i32(sense_addr)
```

If fewer than `grid_x` CTAs are resident — because shared-expert GEMMs on the aux stream hold
some SMs — the arrived CTAs spin forever on a peer that will never be scheduled. That is the
"async cuBLAS concurrency hazard", and it surfaces later, in whatever CUDA op comes next.

The grid is **already** sized for co-residency. From `_fused_grid_x`'s own docstring:

> "pick the fewest full persistent waves that fit the co-residency cap ... while staying
> **<= the cap so the cooperative barrier never deadlocks**"

with `cap = sms * blocks_per_sm`. So the code already assumes whole-grid residency and even
calls the barrier "cooperative" — it simply never asked CUDA to guarantee it. Adding
`cooperative=True` converts that unenforced assumption into a CUDA-enforced one. It should cost
nothing when the grid already fits, and it makes the launch wait rather than deadlock when it
doesn't.

**There is direct precedent in the shipped tree.** `b12x/moe/fused/dynamic.py` already passes
`cooperative=True` at its resident-grid launch, with this comment:

> "A regular launch can admit only part of the grid while auxiliary stream work occupies the
> remaining SMs, leaving resident CTAs spinning and the unscheduled CTAs unable to arrive.
> Cooperative launch makes the all-CTA residency contract explicit."

The Grid188/unified path was fixed. The w4a16 path — which is exactly what prod runs under
`B12X_MOE_FORCE_A16=1` — was missed. This backport applies the same fix to both w4a16 launches
whose kernels use the barrier: `W4A16FusedMoeKernel` and `W4A16FusedMoeHybridKernel` (the
latter composes two `W4A16FusedMoeKernel` tiers and shares their barrier offsets).

**Known remaining gap:** `b12x/moe/fused/micro.py` has the same pattern and is still
non-cooperative. It is the native-NVFP4 micro-decode path, reachable only when
`quant_mode ∈ {nvfp4, w4a8_nvfp4}`; `B12X_MOE_FORCE_A16=1` routes prod to `w4a16`, so it is not
reachable here. Fix it before changing `B12X_MOE_FORCE_A16`.

The capability gate remains available with **no image change at all** —
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` sets `self._stream = None` and produces the identical
`NO_OVERLAP` outcome for a single-quant-method deployment. It is the emergency fallback, not
the plan.

Diffstat of the combined stack:

```text
 vllm/model_executor/layers/attention/mla_attention.py              | 37 +++++-
 vllm/model_executor/layers/fused_moe/fused_moe_method_base.py      | 14 +++
 vllm/model_executor/layers/fused_moe/runner/moe_runner.py          |  5 ++
 vllm/model_executor/layers/fused_moe/runner/shared_experts.py      |  3 +
 vllm/model_executor/layers/quantization/nvfp4_nf3_hybrid.py        | 10 +++
 vllm/v1/attention/ops/dcp_alltoall.py                              | 15 +++-
 vllm/v1/worker/gpu/model_runner.py                                 | 99 ++++++------
 vllm/v1/worker/gpu/warmup.py                                       |  4 +-
 vllm/v1/worker/gpu_worker.py                                       |  8 +-
 vllm/v1/worker/workspace.py                                        | 66 ++++++---
 10 files changed, 192 insertions(+), 69 deletions(-)
```

Because b12x and the baked INT8 extension are untouched, this ships as a **thin overlay
layer over the existing v19 image** — copy 10 `.py` files, no CUDA rebuild, no
recompilation of the `a826ef58` INT8 extension, `TORCH_EXTENSIONS_DIR` unchanged.

### Costs to measure on CN4 (not yet measured)

- **#4 + #5 disable the shared-experts aux stream for prod's config.** The compose comment
  records that dropping `VLLM_DISABLE_SHARED_EXPERTS_STREAM` was worth ~11% decode, so
  expect to give that back. This is a deliberate perf-for-stability trade and it is the
  main thing to decide. See §6 for how to price it today with no code change.
- **#1 adds a staging copy in `_v_up_proj`** when `force_contiguous_mla_bmm_output` is set:
  a `(16, B, 256)` bf16 temp per call. ~512 KiB at decode B=64; ~24 MiB transient at
  prefill B=3072. Same size every call, so allocator-friendly under
  `expandable_segments:False`, but prefill throughput must be re-measured.
- **First boot recompiles** the changed b12x CuTe kernels into `/cache/jit`; later boots are normal.
- **The workspace lane fix adds a second workspace buffer** (the draft lane). Sized during profiling, so the
  KV pool will shrink somewhat from the current 644,864 tokens. Record the exact delta.

---

## 4. Tier 2 — worth taking if Tier 1 boots clean

| Commit | Title | Status |
|---|---|---|
| `d6b49f4cd` | fix(mla): trim padded block tables during MTP expansion | Applies clean to v19. ~10 lines in `mla/indexer.py`. Fixes a shape mismatch on the nonuniform-decode MTP path at the 480k / block-64 / DCP4 boundary — prod's exact geometry. Turns an occasional engine-killing exception into correct behaviour. Clean crash, not a wedge, so it is availability rather than corruption. |

---

## 5. Deliberately excluded, with reasons

### 5.1 `4781731c` fix(pcie): isolate target and draft graph channels — **defer**

This is real and it does apply to prod. Verified on the live image:

- `VLLM_PCIE_ONESHOT_SINGLE_CHANNEL` defaults to `True` in v19 and the compose does not
  override it → target and draft graphs share one PCIe oneshot channel.
- `dcp_alltoall.py:115` hardcodes `single_channel=True` for the DCP A2A pool.
- Worse, deployed `b12x/distributed/pcie_dcp_a2a.py` `for_stream()` will, during capture,
  hand a **new stream key an arbitrary existing channel** rather than a fresh one — so a
  second graph manager silently aliases the first one's A2A staging buffers.

But the vLLM-side fix calls `pool.capture(stream=…)` on the DCP A2A pool, and
`PCIeDCPA2APool.capture` **does not exist in the deployed b12x 0.30.2** (nor do
`_all_channels` / `_capture_channel_stack`). Applying commit 4 as-is raises `AttributeError`
at graph capture — it will not boot.

The b12x-side delta is small and pure Python (~50 changed lines in one file, no CUDA),
and it exists in `workspace/b12x-int8-v19-concurrency/b12x/distributed/pcie_dcp_a2a.py`.
So this is portable — it is just no longer a *minimal* change, because it means shipping a
modified b12x alongside modified vLLM. Recommend as a **separate second boot on CN4** after
Tier 1 is proven, not bundled with it.

Do **not** flip `VLLM_PCIE_ONESHOT_SINGLE_CHANNEL=0` by env alone: without the code change,
each oneshot channel allocates `max_num_batched_tokens × hidden × 2` = 3072 × 6144 × 2 =
**36 MiB per channel** instead of the 84 KiB the fix reduces it to. Multi-channel without
the sizing fix costs memory on a box already at GMU 0.970.

### 5.2 `83579ac7` [GG] Fix MRV2 CUDA graph and sparse DCP memory profiling (#131) — defer

Applies clean to v19, but needs b12x `checkpoint_channels()` / `rollback_channels()`, both
missing from the deployed b12x. Same situation as §5.1. Its value is profiling accuracy and
graph-channel reclaim (capacity), not the wedge.

### 5.3 Already fixed in v19 — take no action

- **PR #133, offload `_build_store_jobs` assertion** (`2705bb2b`). The live image already
  has `advance_stored_idx(self, num_chunks_by_group)` and the
  `assert len(offload_keys) == len(offload_block_ids)` is gone. Confirmed on cn3.

### 5.4 Not applicable to v19 prod

- **vLLM #165 bounded filesystem KV tier** — prod runs `secondary_tiers: []`; there is no
  NVMe tier to bound.
- **vLLM #175 sparse-indexer query split** — at TP4/DCP4 each query-split group has world
  size 1. Keep `VLLM_DCP_QUERY_SPLIT=0` as prod already does.
- **v20 `w4a16-cooperative-grid`** — SparkInfer-side (`sparkinfer/moe/.../w4a16/kernel.py`),
  a different package layout from the deployed `b12x 0.30.2`. Not a drop-in.
- **`00861b94d` aligned block-table width, `b7a89de43` CKV prefetch reset,
  `0995a688d` capacity-pitched DCP workspace (#167)** — all fail both forward and reverse
  apply against v19; they are written against v20 context and would need hand-porting.
  None of them map to an observed v19 failure. Leave them in v20.
- **#154 (absorbed MXFP8 `kv_b_proj`), #168 (MRV2 graph-pool reuse), #171 (compact NVFP4
  MTP verifier)** — these are capacity/perf work, not stability. Out of scope for a
  minimal reliability patch.

---

## 6. Two things that need no code at all

**a) The healthcheck is the biggest MTTR win and it is a compose edit.**
On 2026-07-24 the engine returned 500s for ~50 minutes while
`urlopen('/health')` kept answering 200, so autoheal stayed blind. Replacing the
healthcheck with a real 1-token completion turns a ~50-minute outage into a ~90-second
one. This is independent of every patch above and is worth doing first. It does require a
prod restart, so it needs a window.

**b) The shared-experts-stream trade can be priced today with an env var.**
Tier-1 items 4 and 5 are functionally equivalent, for prod's config, to setting
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` — that env var exists in v19 (`envs.py:2089`) and
short-circuits the same code path. So the ~11% decode question can be answered on CN4 with
a config change and no patched image at all.

Worth noting *why* they are equivalent here: the capability hook in `b12x_moe.py` (the hunk
we drop) resolves through
`tp_moe_plan_supports_aux_stream_overlap`, which only returns `True` for the native-NVFP4
micro-decode band (`quant_mode ∈ {nvfp4, w4a8_nvfp4}`, ≤7 tokens). Prod runs
`B12X_MOE_FORCE_A16=1` → `quant_mode == "w4a16"` → that helper would return `False` in every
case anyway. Porting the b12x side would buy prod nothing.

---

## 7. Suggested CN4 sequence when dev frees up

1. **Boot A — baseline control.** Current v19 image, prod compose, unmodified. Establish
   prefill/decode/needle numbers and KV pool size on cn4's hardware.
2. **Boot B — env-only.** Boot A + `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`. Prices the
   decode cost of Tier-1 items 4+5 before writing any image.
3. **Boot C — Tier 1 overlay.** All five commits as a 10-file overlay layer. Confirm:
   boot clean, KV pool delta vs. Boot A, prefill/decode vs. Boot A/B, full needle ladder
   (50k → 475k) unchanged.
4. **Stress the actual failure.** The crash spec's run 1 followed *four back-to-back
   near-max-context prefills*, and run 2 without that barrage did not reproduce. So the
   discriminator is: needle sweep to 475k + 3× 350k recheck prefills, then immediately the
   decode ctx=50k concurrency matrix, no restart between. Baseline image should be able to
   reproduce the fault; Tier 1 should not.
5. **Boot D — optional.** Tier 1 + the b12x `pcie_dcp_a2a.py` channel-isolation delta +
   `4781731c`. Separate boot, separate verdict.

Promotion to cn3 only after 3 and 4 are green, in a scheduled window, with the deepened
healthcheck landing in the same change.

---

## 8. Verification notes

Everything asserted about the running system was read from cn3 read-only
(`docker inspect`, `docker exec … grep`, `tar` of the site-packages tree for offline diff).
No container was restarted, no file on cn3 or cn4 was modified, and cn4 was not touched at
all while Sol is working there.

Patch-application results were produced by extracting `7ea567a2` from
`workspace/gilded-gnosis` into a scratch tree and running `git apply --check`:

```text
b3ea2e8f   APPLIES CLEAN      93735960   APPLIES CLEAN (and clean without the b12x_moe.py hunk)
ef7cae43   APPLIES CLEAN      e5b6cabb   APPLIES CLEAN
a8b59fbe   APPLIES CLEAN      4781731c   applies clean, but missing b12x API at runtime
83579ac7   applies clean, but missing b12x API at runtime
b9ed50ca   CONFLICT (test file)
d6b49f4cd  APPLIES CLEAN
00861b94d / b7a89de43 / 0995a688d   neither forward nor reverse — need hand-porting
```

Missing-symbol audit against the deployed `b12x 0.30.2`:

```text
tp_moe_plan_supports_aux_stream_overlap   MISSING   (avoided by dropping the b12x_moe.py hunk)
PCIeDCPA2APool.capture / _all_channels /
  _capture_channel_stack                  MISSING   (blocks 4781731c)
checkpoint_channels / rollback_channels   MISSING   (blocks 83579ac7)
```

After applying the Tier-1 stack, `grep` over `vllm/` finds **no** reference to any of those
symbols, and all 10 modified files pass `py_compile`.
