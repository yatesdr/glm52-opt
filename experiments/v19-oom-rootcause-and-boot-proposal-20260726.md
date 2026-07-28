# v19 candidate OOM — root cause and proposal for tonight's remaining boot

Date: 2026-07-26
Status: root cause identified; one boot requested

---

## 1. Root cause — an unprofiled 384 MiB runtime allocation

`b12x/attention/indexer/paged.py:755-777`, in `index_topk_fp8`:

```python
if not output_physical_slots and width_tokens >= 2 * _TWO_LEVEL_SLICE_TOKENS:
    ...
    total_slices = base                        # grows with context, capped at 32
    fold_values  = torch.empty((q_rows * total_slices, topk), torch.float32, ...)   # 192 MiB
    fold_indices = torch.empty((q_rows * total_slices, topk), torch.int32,   ...)   # 192 MiB
```

Two facts make this the failure:

1. **It bypasses the scratch manager.** The *legacy* carry path immediately below uses
   `scratch.get_indexer_contiguous_candidate_buffers()` — pre-reserved memory. The two-level
   fold path allocates with bare `torch.empty`, so vLLM's memory profiler never sees it and
   the KV pool is sized as though it does not exist.
2. **It scales with context width.** `width_tokens = page_table_width * page_size`, and the
   gate `width_tokens >= 2 * _TWO_LEVEL_SLICE_TOKENS` (32,768) means it only engages at deep
   context. `_TWO_LEVEL_MAX_SLICES = 32` caps it, so the pair is bounded at ~384 MiB.

Observed failure, all four ranks simultaneously:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB.
  b12x/attention/indexer/paged.py:771 in index_topk_fp8 → fold_indices = torch.empty(...)
  179 MiB free · 92.14 GiB allocated by PyTorch · 546 MiB reserved but unallocated
```

## 2. The accounting gap, quantified

| | GiB |
|---|---|
| profiled steady state (weights 85.44 + act 2.34 + non-torch 0.07 + graphs 0.96 + KV 4.27) | **93.08** |
| device total | 94.97 |
| **profiled free headroom** | **1.89** |
| actually in use at OOM | 94.66 |
| **runtime allocation beyond the profile** | **1.58** |

The engine consumes ~1.58 GiB at 475k that the profiler never reserved. The 384 MiB fold pair
is part of it. With only 1.89 GiB of headroom, a loaded arena leaves 179 MiB — and a 192 MiB
request fails.

This is why the **same 475k depth passed in Phase 3 and failed in Phase 5**: Phase 3 ran against
a clean arena; Phase 5 ran immediately after the Phase 4 decode benchmarks.

## 3. Both images have this bug

`b12x 0.30.2` is **unmodified between the old prod image and the candidate** — our overlay
touches only `moe/fused/w4a16/kernel.py`. The indexer path is byte-identical.

Profiled headroom is also effectively identical:

| | weights | act | graphs | KV | total | headroom |
|---|---|---|---|---|---|---|
| old v19 prod | 85.02 | 2.22 | 0.95 | 4.80 | 93.00 | 1.97 |
| candidate r2 | 85.44 | 2.34 | 0.96 | 4.27 | 93.08 | 1.89 |

So the old image is exposed to the same OOM, with the **additional** cuBLAS wedge the candidate
fixes. Neither is stable today; the candidate is strictly the better base.

One material difference already demonstrated: when the candidate died it **restarted and
recovered unaided in ~20 min**. The 2026-07-24 wedge on the old image required a *host reboot*.

## 4. Proposed fix — bound the KV pool so the transient always fits

Single change, no code, fully deterministic:

```text
--kv-cache-memory-bytes=4101693767      # 3.82 GiB  (currently auto-sized to 4.27 GiB)
```

| | now | proposed |
|---|---|---|
| KV | 4.27 GiB | **3.82 GiB** |
| pool | 573,696 tok | **~513,000 tok** |
| concurrency @480k | 1.20× | **1.07×** — full context still fits |
| free headroom | 1.89 GiB | **2.34 GiB** |
| margin over observed 1.58 GiB overshoot | 1.20× | **1.48×** |

Why 3.82 GiB and not another value:

- **0.75 GiB back → pool 472,930**, below `max-model-len` 480,000. Breaks full context. Rejected.
- **0.30 GiB back → headroom 2.19 GiB**, only 0.61 GiB over the observed overshoot. Too thin
  against fragmentation (`expandable_segments:False` cannot coalesce; 546 MiB was stranded).
- **0.45 GiB back** is the largest give-back that keeps full context with ~1.5× margin.

`expandable_segments:True`, which the OOM message suggests, is **not available** — the compose
documents that the offload tier is incompatible with it.

This is a **mitigation, not a cure**: the 384 MiB is still unprofiled and still allocated per
call. It is chosen because it is deterministic, needs no code change, and can be reasoned about
exactly — the right properties for a single boot that must land stable.

## 5. Proposed boot plan

1. Edit one line in `deploy/glm52-v19-reliability-20260726.yaml` — add `--kv-cache-memory-bytes`.
   No other flag changes. Same image (`glm52-serve:v19-reliability-r2-20260726`), no rebuild.
2. Recreate the container. JIT cache is warm — expect ~10-15 min, not 22.
3. Confirm `GPU KV cache size` lands near 513,000 and headroom ≈ 2.34 GiB.
4. Re-run **Phase 5 only** (`--stress-only`, ~1.5 h): the exact sequence that failed —
   deep sweep to 475k → 3× 350k rechecks → decode ctx50k, no restart between.
5. Accept only if: no OOM, no wedge signature, `RestartCount` unchanged, pool > 480,000.

Autoheal stays off during the run; `restart=unless-stopped` stays on so a failure self-recovers.

## 6. What this does not fix (follow-ups)

- **The real fix** is routing the fold buffers through the scratch manager, or caching them
  module-level like `#130`'s `_DCP_A2A_GRAPH_BUFFERS`, so they are profiled and reserved once.
  That is a b12x change in a hot path and should not ride along on a stability boot. It is also
  an upstream-worthy bug report against b12x.
- **The 0.53 GiB weight growth** from `#136`'s contiguous absorbed weights is still unexplained
  at the mechanism level (`max_memory_allocated` says it is live, refcounting says it should be
  freed). Recovering it would return the pool to ~580k *and* keep the new headroom. Unresolved,
  not proposed for tonight.

## 7. Risk

| risk | assessment |
|---|---|
| pool 513k too small for real traffic | 1.07× at full 480k; at typical 50k context supports ~10 concurrent of the 16 slots |
| headroom still insufficient | 1.48× margin over the worst observed demand; if it recurs, the next lever is trimming cudagraph capture sizes |
| boot fails | image unchanged and already proven to boot; only a serve flag differs |
| rollback | stop candidate, `docker compose -f /home/claude/glm52-prod-ring.yaml up -d` |
