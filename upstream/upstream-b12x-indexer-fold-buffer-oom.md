# b12x/SparkInfer: paged indexer two-level fold buffers bypass the scratch reservation → OOM at deep context

**Filed:** 2026-07-26 · **Reporter:** Derek Yates (cn3, GLM-5.2 TP4/DCP4)
**Component:** `b12x/attention/indexer/paged.py` — `index_topk_fp8`
**Severity:** engine-killing OOM at long context under arena pressure

---

## Summary

The paged indexer's **two-level fold** path allocates its three working buffers with bare
`torch.empty` / `torch.full` instead of taking them from the pre-reserved indexer scratch.
At maximum context width this is ~**384 MiB per call**, allocated fresh on every invocation,
only at deep context.

Because the allocation never passes through `plan_indexer_scratch` →
`current_workspace_manager().get_simultaneous(...)`, vLLM's memory profiler does not reserve
for it. The KV cache is therefore sized as though the memory were free, and the indexer later
demands it back at exactly the moment the arena is most loaded.

**Every other buffer in this kernel comes from the scratch.** The *legacy* carry path,
immediately below in the same function, correctly uses
`scratch.get_indexer_contiguous_candidate_buffers()`. Only the newer two-level path bypasses it.

## Affected versions

| tree | ref | status |
|---|---|---|
| `b12x` (deployed) | 0.30.2 | **affected** |
| `b12x` master | `24335002` | **affected** |
| SparkInfer (v20 line) | `1a88b38` | **affected** |

The allocation block is identical across all three apart from one line-wrap. SparkInfer's
scratch does reserve `fold_carry_chunks` — but that is the *legacy* carry double-buffer
(`carry_buf_values` / `carry_buf_indices`), not the two-level fold buffers. The gap was
inherited, not closed.

## The code

`b12x/attention/indexer/paged.py` (0.30.2 line numbers; SparkInfer ≈ +33):

```python
    width_tokens = page_table_width * page_size
    ...
    if not output_physical_slots and width_tokens >= 2 * _TWO_LEVEL_SLICE_TOKENS:
        ...
        total_slices = base
        fold_values = torch.empty(                                    # ~192 MiB
            (q_rows * total_slices, topk), dtype=torch.float32, device=q_fp8.device
        )
        fold_indices = torch.empty(                                   # ~192 MiB
            (q_rows * total_slices, topk), dtype=torch.int32, device=q_fp8.device
        )
        fold_lengths = torch.full(
            (q_rows,), total_slices * topk, dtype=torch.int32, device=q_fp8.device
        )
    carry_buf_values = None
    carry_buf_indices = None
    if num_chunks > 1 and not two_level_slices:
        # <-- the legacy path does it correctly:
        carry_buf_values, carry_buf_indices = scratch.get_indexer_contiguous_candidate_buffers()
```

`_TWO_LEVEL_MAX_SLICES = 32` bounds `total_slices`, so the requirement is finite and knowable
at plan time — it just isn't planned for.

## Observed failure

GLM-5.2 hybrid, TP4/DCP4, `max-model-len` 480,000, GMU 0.970, `B12X_MLA_SPARSE`,
`nvfp4_ds_mla` KV, `expandable_segments:False` (required by the KV offload tier).

All four ranks simultaneously, at a 475k-token request:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB.
  vllm/model_executor/layers/sparse_attn_indexer.py:1006  _run_b12x_paged_topk
    → b12x/attention/indexer/paged.py:771  index_topk_fp8
        fold_indices = torch.empty(...)
GPU 2: 179.12 MiB free · 92.14 GiB allocated by PyTorch
       546.26 MiB reserved by PyTorch but unallocated
```

The engine died; the container restarted and recovered.

## The accounting gap

| | GiB |
|---|---|
| profiled steady state (weights 85.44 + activation 2.34 + non-torch 0.07 + graphs 0.96 + KV 4.27) | 93.08 |
| device total | 94.97 |
| profiled free headroom | **1.89** |
| actually in use at OOM | 94.66 |
| **runtime allocation beyond the profile** | **1.58** |

The profiler under-reserves by ~1.58 GiB at deep context, of which these buffers are a large
part. With only 1.89 GiB of headroom, a loaded arena leaves 179 MiB against a 192 MiB request.

## Reproduction

The defect is **sequence-dependent**, which is why it hides. The identical 475k request
succeeded against a clean arena and failed after a decode benchmark had loaded it:

1. Boot with a long `max-model-len` (we used 480,000) and GMU high enough that the KV pool
   consumes most of the profiled headroom.
2. Run a decode benchmark sweep (we used concurrency 1→16 at ctx 0 and ctx 50k) to load and
   fragment the allocator arena.
3. **Immediately**, with no restart, issue a request at maximum context width (≥475k).
4. Observe the OOM at `paged.py` `fold_indices`.

Under `expandable_segments:False` the 546 MiB of reserved-but-unallocated memory cannot be
coalesced to satisfy the 192 MiB contiguous request.

## Impact

- Engine-killing OOM at deep context on saturated deployments.
- Silent over-sizing of the KV cache: the pool is advertised larger than the device can
  actually sustain, so the failure appears as an unexplained crash rather than a capacity error.
- `expandable_segments:True`, which PyTorch's own message suggests, is unavailable to
  deployments using the KV offload tier (incompatible).
- **The v20 line is exposed.** This will surface there as an unexplained deep-context crash
  with no obvious connection to the indexer.

## Suggested fix — with an important caveat

**A naive static reservation is worse than the bug.** We implemented it and measured it before
proposing it. The fold buffers scale as `q_rows x total_slices x topk`, and those factors are
**anti-correlated in practice**: at deep context the prefill chunker feeds *small* q chunks
(the observed 192 MiB implies `q_rows ~ 819` at `total_slices = 30`), while large `q_rows`
only occurs at short context where `total_slices` is small. The joint worst case never occurs,
but a static reservation must assume it.

Measured on our geometry (`topk = 2048`, `S_max = 30`, page size 64, 480k context):

| sizing basis | reservation |
|---|---|
| observed runtime demand at 475k | **384 MiB** |
| static reservation at `max_q_rows = 3072` | **1,440 MiB** |
| static reservation at the `profile_q_rows` cap (4096) | **1,920 MiB** |

`profile_q_rows` is capped by `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (512) / supertile `tile_k`
(32768) = 4096, so a correct worst-case reservation is ~5x the real demand. On our box that
would cost ~189,000 KV tokens — far more than the ~60,000 the workaround costs.

So the reservation must **not** be sized on `max_q_rows` alone. Either bound it by the true
joint maximum of `q_rows x total_slices` (which requires knowing what the chunker can hand the
indexer at depth), or retain a grow-on-demand cached buffer so the footprint converges on the
actual high-water mark rather than the theoretical worst case.

If you do reserve statically, the mechanism is:

1. `scratch.py::_indexer_paged_scratch_layout` — add regions sized
   `max_q_rows × S_max × topk` (float32 and int32) plus `max_q_rows` int32 for lengths, where
   `S_max = min(_TWO_LEVEL_MAX_SLICES, ceil(max_page_table_width * page_size / _TWO_LEVEL_SLICE_TOKENS))`.
   Skip entirely when `width_tokens < 2 * _TWO_LEVEL_SLICE_TOKENS`, so short-context
   deployments pay nothing.
2. `scratch.py::_materialize_indexer_paged_scratch` — carve the views.
3. Add a `get_two_level_fold_buffers(*, row_count, total_slices)` accessor with the same
   capacity-check style as `get_indexer_contiguous_topk_buffers`.
4. `paged.py` — take the buffers from the accessor; `fold_lengths.fill_(total_slices * topk)`
   since the buffer is now reused rather than freshly created.

The caps already carry every input needed (`max_q_rows`, `topk`, `max_page_table_width`), so
this is a planning omission rather than a design problem.

Note the buffers must join the **existing** `get_simultaneous` call rather than a second one —
the workspace manager packs each call from offset 0, so a separate call would alias.

We verified the mechanical part of this is sound — a CPU proof confirmed every pre-existing
scratch offset is byte-identical, the new regions append cleanly with no overlap, and the
reserved capacity matches the runtime slice count exactly across widths from 32k to 480k. The
blocker is purely the sizing basis, not the layout surgery.

## Workaround

Bound the KV cache explicitly so the unreserved transient always fits — e.g.
`--kv-cache-memory-bytes`, sized to leave ≥ ~2.3 GiB of headroom at this geometry. This costs
KV capacity (~60k tokens for us) and is a hand-tuned number, but it is deterministic and needs
no code change.
