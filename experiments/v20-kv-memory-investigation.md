# v20 KV-pool memory regression — investigation facts (Fable → Sol)

Status: **MRV2 graph-pool lifetime defect isolated; local fix `982cda45`, runtime proof pending**
Operator: Fable · Date: 2026-07-22 · Context: CN3 v20 combined acceptance window

Facts only. What we measured, what the code says, what we ruled out, and the open question.

## 1. Symptom

v20 candidate refused to boot at the reviewed profile (GMU 0.970, `max_model_len=480000`):

```text
ValueError: To serve at least one request with the model's max seq len (480000),
(3.57 GiB KV cache is needed, which is larger than the available KV cache memory (3.25 GiB).
Based on the available memory, the estimated maximum model length is 435968.
```

Clean config-fit failure at KV allocation. **Not a crash** — no cuBLAS, no Xid, no assertion;
workers exited gracefully. Deterministic: the container then restart-looped into the identical
failure (`restart: unless-stopped`), which we stopped manually.

## 2. Measured deltas, v19 prod vs v20 candidate (identical serving profile)

| Quantity | v19 (`gilded-gnosis-v19-int8-block-patched`) | v20 candidate | Δ |
|---|---|---|---|
| GMU | 0.970 | 0.970 | — |
| Weights / rank | 85.02 GiB | **85.44 GiB** | **+0.42** |
| Peak activation | 2.22 GiB | (not separately printed) | — |
| non-torch | 0.07 GiB | — | — |
| CUDAGraph accounted | 0.95 GiB | **1.08 GiB captured** (0.72 retained-as-non-torch, 0.36 additional) | **+0.49** |
| B12X sparse-DCP transient | **not counted** | **+233.25 MiB** included in profile peak | **+0.23** |
| **Available KV cache memory** | **4.80 GiB** | **3.25 GiB** | **−1.55 GiB** |
| **GPU KV pool** | **644,864 tokens** (1.34× @480k) | would be ~435,968 | **−32%** |

Token density is consistent across both (~134k tokens/GiB), so the KV format is unchanged —
the entire pool delta comes from available memory, not from KV record size.

Relevant v20 log lines:

```text
[mla_attention.py:1047] Including 233.25 MiB of B12X sparse DCP transient memory in the profile peak
[gpu_worker.py:576]     Available KV cache memory: 3.25 GiB
[gpu_worker.py:591]     CUDA graph memory profiling is enabled (default since v0.21.0).
                        The current --gpu-memory-utilization=0.9700 is equivalent to
                        --gpu-memory-utilization=0.9662 without CUDA graph [profiling]
[model_runner.py:953]   Estimated MRV2 CUDA graph memory: 0.36 GiB additional
                        (1.08 GiB captured, 0.72 GiB retained and counted as non-torch)
```

## 3. Root cause: v20 added *measurement*, not extra graphs

This is the central finding and it is code-verified, not inferred:

- **v19's `vllm/v1/worker/gpu/model_runner.py` contains zero occurrences of the MRV2 estimator**
  (`grep -c "MRV2 CUDA graph memory"` → `0`). v20 introduced it.
- The v20 estimator does **not create graphs**. It redirects every existing `CUDAGraphWrapper` and
  `BreakableCUDAGraphWrapper` instance into a temporary profiling pool, performs a dry-run capture,
  measures `start_free − end_free`, subtracts the retained pool size, restores the original pools,
  and returns the remainder to be reserved.
- The in-code comment states the gap being closed:

  > "A CUDA graph private pool can retain physical pages after its graph objects are destroyed.
  > `memory_profiling` observes those pages as non-torch memory, so only return the remaining
  > capture cost here."

**Conclusion: v19 captured the same graphs but never measured the retained private-pool pages, so
its memory profile under-counted and the unreserved slack was handed to the KV cache. v19's 644,864
was therefore partly borrowed from memory CUDA graphs were actually using. v20's 3.25 GiB is the
honest figure.**

Three independent corroborations:
1. v19 **OOM'd at GMU 0.975** on the first real deep prefill — consistent with thin real headroom.
2. v20 release notes: *"TP6 exposes lower KV token capacity than older unsafe estimates due to
   accounting for MRV2 graph and sparse-DCP transient memory."*
3. v20 release notes: *"correctness no longer depends on allocator slack."*

## 4. Ruled out

- **Not** the INT8 wire (PR #69), NVMe eviction (PR #165), or the MTP indexer fix (PR #166) — all
  four baked seams byte-verified against Sol's pins; failure is in KV budgeting, before any of them.
- **Not** a KV-format change — token density identical (~134k tokens/GiB).
- **Not** a crash or concurrency fault — graceful worker exit, explicit ValueError.
- **Not** stale `/dev/shm` — verified 107.3 GB free and 0 CUDA procs before each boot. (Separately:
  v19's 64 GB offload mmap *does* survive `compose down` as a root-owned file and must be removed
  explicitly, or a subsequent 64 GB tier cannot allocate. Operationally relevant, unrelated to this.)

## 5. Levers, with honest assessment

| Lever | Frees real memory? | Notes |
|---|---|---|
| Trim `cudagraph_capture_sizes` `[1,2,4,8,16,24,32,40,48,56,64]` → `[1,2,4,8,16,32,64]` | **Yes** | est. 0.3–0.4 GiB. Size **64 must stay**: `max_num_seqs=16` × MTP(1+3) = 64-token verification batches — the path carrying v20's decode gain |
| Raise GMU | **No** | re-takes the slack v20 deliberately reserved; re-creates the v19 margin that OOM'd at 0.975, now with graph usage proven in use |
| Reduce MNBT 3072→2048 | Yes (peak activation) | deviates from the reviewed profile; would make prefill numbers non-comparable to the v19 baseline |
| Disable CUDA-graph memory profiling | **No** | only stops counting; reverts to v19's unsafe accounting |
| Lower `max_model_len` | n/a | trades max context for concurrency headroom |

**Unrecoverable floor:** ~0.65 GiB (weights +0.42, sparse-DCP +0.23) is genuine new usage, not
accounting. Exact v19 parity (644,864) is therefore unlikely to be safely reachable on v20.

## 6. Open questions for Sol

1. **Is the +0.42 GiB weight growth expected** between `vllm2167295/si6a92bcc` and
   `vllm3e731bc/si1a88b38`? Nothing in the audit predicted it. Is it MTP/`eh_proj` handling, a quant
   path change, or something reclaimable?
2. **Is the 0.72 GiB "retained and counted as non-torch" double-counted?** The estimator subtracts it
   from its own return value on the assumption `memory_profiling` already sees it. If both paths
   reserve it, we are losing ~0.72 GiB twice.
3. **Can the MRV2 graph set be reduced for our shape** without losing the 64-token MTP verification
   graph — e.g. are the intermediate capture sizes (24/40/48/56) reachable at `max_num_seqs=16`, or
   are they dead weight for this configuration?
4. **Is the 233 MiB sparse-DCP transient per-rank and permanent**, or a peak that could be pooled
   with another allocation?
5. **Your audit predicted "no expected KV-pool change from the MRV2 accounting update under our
   1024/1025 threshold configuration."** Empirically the pool dropped 32%. What did the threshold
   analysis assume that the runtime does differently?

## 7. Operator state at time of writing

- v19 prod is **down**; window is open; v19 rollback armed (image + warm cache, ~12–15 min).
- Retrying v20 at **GMU 0.980** to establish whether it boots and what pool it yields
  (projected ~4.2 GiB ≈ ~563k tokens — clears the 3.57 GiB needed for 480k, short of the 600k goal).
- Business goal for the night: v20 in prod with **≥600k pool** and similar-or-better prefill/decode.
  On the honest accounting, ≥600k with `max_model_len=480000` may not be reachable without
  re-introducing the over-commitment v20 exists to prevent. That tradeoff is unresolved.

## 8. 2026-07-22 code resolution: retained pool was not reusable

The 0.72 GiB question in section 6 exposed the boot crash. MRV2 used a fresh
`graph_pool_handle()` for profiling, subtracted memory retained by that pool,
then restored all managers and wrappers to the different global production
pool. The later production capture therefore could not reuse the capacity
whose cost MRV2 had subtracted. At GMU 0.980 this left a 0.72 GiB hidden
capture shortfall beside the 544,000-token KV pool.

The complete size-64 target and speculative graph set already captured and
synchronized successfully during profiling. That falsifies the initial
intrinsic-MTP-route diagnosis. The local fix profiles into the same global
pool used by production; see `v20-mrv2-graph-pool-reuse-fix.md` and commit
`982cda45`. One exact-profile CN3 boot is still required for runtime proof.
