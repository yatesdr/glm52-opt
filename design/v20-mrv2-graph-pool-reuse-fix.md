# v20 MRV2 CUDA-graph pool reuse fix

Date: 2026-07-22  
Status: code-proven; one CN3 runtime acceptance boot pending

## Verdict

The v20 boot failure is a CUDA-graph memory-pool accounting/lifetime bug in
MRV2, not an intrinsic MTP size-64 kernel failure and not PR #69, #165, or
#166.

MRV2 successfully captured the complete target and speculative graph set
before KV allocation. It reported:

```text
0.36 GiB additional (1.08 GiB captured, 0.72 GiB retained and counted as non-torch)
```

The code then destroyed those graphs, allocated the 544,000-token KV pool,
and captured the same graph set again. The second capture failed
asynchronously; CUDA surfaced the prior failure at `make_dummy()`.

## Root cause in code

`GPUModelRunner.profile_cudagraph_memory()` did this:

```python
profiling_pool = current_platform.graph_pool_handle()
```

Every production graph manager and wrapper normally uses:

```python
current_platform.get_global_graph_pool()
```

MRV2 therefore profiled into a disposable private pool but later captured
production graphs into another pool. It subtracted the 0.72 GiB retained by
the profiling pool from the future reservation even though the production
pool could not reuse that capacity. The resulting budget reserved only the
0.36 GiB remainder for a production capture whose measured gross requirement
was 1.08 GiB: a 0.72 GiB hidden shortfall.

This explains all observed facts:

- profiling the full graph set succeeds and synchronizes;
- actual capture fails only after the full KV pool is allocated;
- CUDA reports an asynchronous illegal access at an innocent later copy;
- a different user's graph-cap-32/MNS-8 configuration works because its graph
  footprint is smaller;
- the failure does not require a defect in INT8 wire, NVMe eviction, or the
  indexer block-table fix.

PyTorch documents that captures given the same pool token may share a graph
private memory pool. Production vLLM already relies on one global token for
its managers and wrappers. The profiling graphs are destroyed before the
production graphs are captured, so these two generations never replay
concurrently.

Reference:
<https://docs.pytorch.org/docs/main/notes/cuda.html#graph-memory-management>

## Minimal fix

Profile into the same persistent global pool production will use:

```diff
- profiling_pool = current_platform.graph_pool_handle()
+ profiling_pool = current_platform.get_global_graph_pool()
```

Now the retained 0.72 GiB remains reusable by the later identical capture,
so MRV2's `gross - retained` reservation is valid. Reserving the full 1.08
GiB would also avoid the crash, but would sacrifice roughly another 0.72 GiB
of KV capacity unnecessarily.

Artifacts:

- base: `3e731bc043d23ec21277fb76d3e15fe6da91b23b`
- branch: `fix/mrv2-reuse-profile-pool-20260722`
- commit: `982cda453bf50a280381400c4294315dc04fbee8`
- input `model_runner.py` SHA-256:
  `cbe6e868b8ea901b9d53f90d2548f97eea53be00f028ea2b90cb9103805d7543`
- patched `model_runner.py` SHA-256:
  `2eab8362e2ce3e1004941988347b9921072053d52198b6f88be2d98d03cdd779`

The patch is one production-line behavior change plus explanatory comments
and a regression test. The test fails if MRV2 creates a disposable pool and
asserts that target, wrapper, lazy-wrapper, cleanup, and restored state all
use the production pool.

Local checks passed:

- `git diff --check`
- Python bytecode compilation of both changed files
- exact base and input/output byte pins

The repository's Python environment is not installed on this Mac, so the
pytest case requires the image/CI environment.

## One-boot CN3 acceptance

Do not run the superseded routing ladder. Build the combined v20 candidate
with commit `982cda45`, preserving the exact previously failed profile:

- TP4/DCP4, MTP3
- MNS 16 and graph cap 64
- `max_model_len=480000`
- GMU 0.980
- existing INT8, NVMe, and indexer patches unchanged

One boot is sufficient. PASS requires:

1. MRV2 again reports the graph-memory tuple; record all three values.
2. KV allocation remains at or near 544,000 tokens. The fix does not free the
   retained 0.72 GiB for KV; it preserves it as reusable production graph-pool
   capacity. A large pool regression means accounting changed unexpectedly.
3. Production target and speculative CUDA-graph capture completes on all four
   workers.
4. The API becomes healthy and returns a correct deterministic response.
5. `RestartCount` stays zero and logs contain no illegal access, CUDA OOM,
   worker death, or `EngineDeadError`.

This single boot directly exercises both the former failure and the intended
memory reuse. Only if it fails should we spend another boot distinguishing
allocator reuse from a separate post-profile fault.
