# OffloadingConnector `_build_store_jobs` assertion (v19)

Prepared for Sol. Facts only — observed behavior, exact stack, and source; no proposed root cause.

## Summary

The v19 `OffloadingConnector` crashes `EngineCore` with an `AssertionError` in
`_build_store_jobs` when KV blocks are evicted (active KV demand exceeds the GPU pool). The
engine dies (`EngineDeadError`); `restart: unless-stopped` recovers it after a full reboot.

## Environment

- Image: `voipmonitor/vllm:gilded-gnosis-v19-vllm7ea567a-b12x4cfa530-fi801d57a-cu132-20260718`
  (vLLM `v0.11.2.dev280+gilded.gnosis.v19`).
- Model: GLM-5.2 hybrid, TP4 / DCP4 / MTP3, `max-model-len 480000`, `MAX_NUM_SEQS 16`, GMU 0.970.
- Connector: `OffloadingConnector`, `spec_name=TieringOffloadingSpec`, `kv_role=kv_both`,
  `cpu_bytes_to_use=64000000000` (64 GB DRAM tier), `secondary_tiers=[]`, `kv_buffer_size=1e9`.
- GPU KV pool: **644,864 tokens**.
- Wire-mode independent: reproduced with `B12X_PCIE_DMA_FP8` = `0` (bf16), `ag` (E4M3), and `i8`
  (INT8). The fault is in the scheduler/connector, not the DMA all-reduce.

## Exact stack (2026-07-19 20:09:23Z)

```
vllm/v1/engine/core.py:666  step_with_batch_queue
  scheduler_output = self.scheduler.schedule(...)
vllm/v1/core/sched/scheduler.py:1188  schedule
  meta = self._build_kv_connector_meta(self.connector, scheduler_output)
vllm/v1/core/sched/scheduler.py:1210  _build_kv_connector_meta
  return connector.build_connector_meta(scheduler_output)
vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:157  build_connector_meta
  return self.connector_scheduler.build_connector_meta(scheduler_output)
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1157  build_connector_meta
  store_jobs=self._build_store_jobs(scheduler_output),
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:975  _build_store_jobs
  assert len(offload_keys) == len(offload_block_ids)
AssertionError
```

Result: `EngineCore encountered a fatal error` → `vllm.v1.engine.exceptions.EngineDeadError`.
In-flight requests return HTTP 500; container restarts (`RestartCount` increments; ~full reboot).

## Source at the assertion (`offloading/scheduler.py`, ~lines 963–975)

The two operands are sliced differently — `offload_keys` by chunk index, `offload_block_ids` by a
strided slice of `block_ids` (last block of each chunk):

```python
offload_keys = group_state.offload_keys[start_chunk_idx:num_chunks]
# A block_id of 0 means either a sliding window / SSM skip
# or a stale entry that was zeroed out — skip it either way.
offload_block_ids = group_state.block_ids[
    start_chunk_idx * blocks_per_chunk
    + blocks_per_chunk
    - 1 : num_chunks * blocks_per_chunk : blocks_per_chunk
]
assert len(offload_keys) == len(offload_block_ids)
```

(The in-code comment notes `block_id == 0` entries are meant to be skipped "either way"; the two
slices are compared for equal length before that filtering.)

## Observed triggers

- **2026-07-19 ~20:09Z:** LIL decode bench at **concurrency 16 × context 50k** (≈ 16 × 50k =
  800k tokens requested against a 644,864-token pool → eviction). Crashed mid-run.
- **Earlier this session:** a `MAX_NUM_SEQS=32` / ~32-concurrent sweep crashed with the same
  assertion.
- **Not observed** when active KV demand stays below the pool (e.g., conc 16 × context 16k =
  256k < 644,864 ran clean; conc ≤ 8 × 50k = 400k ran clean). I.e., it correlates with the
  eviction/store path being exercised, not with concurrency per se.

## Recovery

`restart: unless-stopped` restarts the container after the crash. Reboot is a full cold start
(~5–10 min: safetensors load + graph capture). No manual intervention needed, but every trip is
a multi-minute outage of that engine.

## Not established here

- The precise invariant that is violated (why the chunk-indexed `offload_keys` length and the
  strided `offload_block_ids` length diverge under eviction) — stated only as the observed
  correlation above plus the source structure; not root-caused.
- Whether it reproduces without the DRAM tier, with a different `blocks_per_chunk`, or under
  sliding-window/SSM paths specifically.
