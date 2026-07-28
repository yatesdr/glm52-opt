# Cure: reserve the two-level fold buffers through the indexer scratch

Date: 2026-07-26
Status: design complete, not implemented — needs a boot to validate
Relates to: `v19-oom-rootcause-and-boot-proposal-20260726.md` (the mitigation)

---

## The defect

`b12x/attention/indexer/paged.py:769-777`, inside `index_topk_fp8`:

```python
fold_values  = torch.empty((q_rows * total_slices, topk), torch.float32, device=...)
fold_indices = torch.empty((q_rows * total_slices, topk), torch.int32,   device=...)
fold_lengths = torch.full((q_rows,), total_slices * topk, torch.int32,   device=...)
```

Every other buffer in this kernel comes from the pre-reserved indexer scratch. These three do
not. At maximum context width the pair is ~**384 MiB**, allocated fresh on every call, at the
deepest contexts, when the arena is most loaded.

Consequences:

1. **Invisible to the memory profiler.** vLLM sizes the KV pool from a profiling run that never
   reserves these bytes, so the pool is over-sized by exactly the amount the indexer will later
   demand. Measured on cn3: profiled steady state 93.08 GiB, but 94.66 GiB actually in use at
   475k — **1.58 GiB of unreserved runtime allocation**, of which this is a large part.
2. **Allocation churn.** Repeated 192 MiB alloc/free under `expandable_segments:False` (mandatory
   for the offload tier) cannot be coalesced; 546 MiB was observed stranded as
   reserved-but-unallocated at the moment of failure.

Observed failure, all four ranks:

```text
torch.OutOfMemoryError: Tried to allocate 192.00 MiB.
  paged.py:771 → fold_indices = torch.empty(...)
  179 MiB free · 546 MiB reserved but unallocated
```

**Present in unmodified `b12x 0.30.2`** — the old v19 prod image has it too. This is not a
regression introduced by the reliability backport.

## Why the scratch route is the right cure

The infrastructure already exists and is already wired correctly:

```text
vllm/model_executor/layers/sparse_attn_indexer.py:1307  _reserve_b12x_paged_indexer_scratch
  → b12x plan_indexer_scratch(B12XIndexerScratchCaps(...))
  → plan.shapes_and_dtypes()
  → current_workspace_manager().get_simultaneous(*...)      # ONE call — AGENTS.md contract #3
```

Everything reserved this way is allocated during warmup, is therefore counted by the memory
profiler, and the KV pool is sized around it automatically. Adding the fold buffers to that plan
makes the reservation **correct and self-sizing**, replacing the hand-tuned
`--kv-cache-memory-bytes` mitigation.

The caps already carry every input needed to size it: `max_q_rows`, `topk`,
`max_page_table_width`.

## Change set (4 edits, all in b12x)

**1. `scratch.py::_indexer_paged_scratch_layout` (~line 818)** — add three regions:

```text
two_level_fold_values   (max_q_rows * S_max, topk)  float32
two_level_fold_indices  (max_q_rows * S_max, topk)  int32
two_level_fold_lengths  (max_q_rows,)               int32

where S_max = min(_TWO_LEVEL_MAX_SLICES,
                  ceil(max_page_table_width * PAGED_INDEX_PAGE_SIZE / _TWO_LEVEL_SLICE_TOKENS))
```

`_TWO_LEVEL_MAX_SLICES = 32` already bounds this, so the reservation is finite and known at plan
time. Skip the regions entirely when the two-level path cannot engage
(`width_tokens < 2 * _TWO_LEVEL_SLICE_TOKENS`), so short-context deployments pay nothing.

**2. `scratch.py::_materialize_indexer_paged_scratch` (~line 1190)** — carve the three views out
of the single scratch storage, matching the existing pattern.

**3. `scratch.py::B12XIndexerPagedScratch`** — add an accessor mirroring
`get_indexer_contiguous_topk_buffers`:

```python
def get_two_level_fold_buffers(self, *, row_count: int, total_slices: int):
    # validate against reserved capacity, raise with the same message style
    # return (values[:row_count*total_slices], indices[:row_count*total_slices], lengths[:row_count])
```

**4. `paged.py` (~line 769)** — replace the three `torch.empty`/`torch.full` calls with the
accessor, and `fold_lengths.fill_(total_slices * topk)` since the buffer is now reused.

## What this does and does not do

- **Does**: make the 384 MiB reserved, profiled, and allocated once. Removes the runtime OOM and
  the fragmentation churn. Lets the KV pool auto-size correctly and retires the hand-tuned flag.
- **Does not**: recover memory. The bytes are still needed; they just move from an unaccounted
  runtime spike to an accounted reservation. Expect the auto-sized pool to land near the
  mitigation's ~513k, not back at 573k.
- **Separate issue**: the 0.53 GiB of contiguous absorbed MLA weights from `#136`. Recovering
  that is what would return the pool toward ~580k. Still unexplained at the mechanism level.

## Risks

| risk | note |
|---|---|
| capacity check too tight → runtime raise | `S_max` must be computed from the same constants `paged.py` uses; a mismatch turns an OOM into an exception. Assert equality in a unit test. |
| short-context regression | gate the reservation on the same `width_tokens` condition so nothing changes below 32,768 tokens |
| contract #3 violation | the buffers must join the **existing** `get_simultaneous` call, not a second one — otherwise views alias |
| hot-path change | `index_topk_fp8` runs every layer at every deep-context step; a sizing bug is loud but a stride bug would be silent. Needs the needle ladder to validate, not just a boot. |

## Validation plan

1. CPU-only: unit-test `S_max` agreement between the layout planner and `paged.py`.
2. Boot: confirm the KV pool auto-sizes *down* by roughly the reservation and that
   `--kv-cache-memory-bytes` can be removed.
3. Needle ladder 50k→475k — this path computes top-k indices, so a stride error would corrupt
   retrieval. This is the gate that matters.
4. The Phase-5 stress sequence, which is what exposed the defect.

## Upstream

This is a genuine defect in `b12x`/SparkInfer, not a local integration issue, and is worth
reporting with the cn3 evidence: the allocation site, the 1.58 GiB unreserved-runtime figure,
and the OOM trace.
