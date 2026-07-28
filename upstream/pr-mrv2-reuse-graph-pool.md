> **Scope status — updated 2026-07-24:** The exact patched proof boot changed
> MRV2 accounting as intended and recovered 11,520 KV tokens. It did not fix
> the separate illegal access; launch-blocking later localized that fault to
> the MLA query BMM and the independent safe-query correction booted cleanly.
> Review this PR only as an MRV2 allocator-consistency/capacity change, not as
> a decode-crash fix.
>
> The exact delta has now been forward-ported over vLLM `992b874cf` plus the
> production integration at `e16288f5d006726b0492981d3db5627fc5d9f70e`.
> Its stable patch-id is identical to PR #168:
> `6e224cef7ddae739c459874ac506c5850ca22c5d`. Current-base runtime
> qualification remains pending.
>
> The target base already contains PR #172's functional startup-profiling
> change as `cb27f671d`. That work profiles persistent startup allocations
> before sizing KV and avoids duplicating the synthetic compile workspace.
> It does not make the profiling and production CUDA-graph captures share one
> allocator pool. PR #168 is therefore complementary, not a duplicate: it
> corrects the lifetime and reuse of capacity retained by the profiling pool.

## Summary

Reuse the production CUDA graph pool while MRV2 profiles graph memory.

MRV2 currently captures into a fresh private pool, destroys the profiling
graphs, subtracts memory retained by that pool from the future graph-memory
reservation, and then captures production graphs into a different global
pool. Capacity retained by the profiling pool is therefore counted as
resident non-Torch memory but cannot be shared with production capture. The
patch makes MRV2's `gross - retained` calculation and later pool selection
internally consistent.

The hardware result below proves this inconsistency was real and the patch
changed its accounting, but also proves it was **not** the cause of the
production-capture illegal access being investigated.

The behavioral change is one line:

```diff
- profiling_pool = current_platform.graph_pool_handle()
+ profiling_pool = current_platform.get_global_graph_pool()
```

No graph, MTP, DCP, capture-size, quantization, or memory-profiling feature is
disabled. Profiling graphs are still synchronized, destroyed, and have their
external graph channels rolled back before production capture.

## Root-cause proof from the code

An independent reviewer can validate the reasoning in
`vllm/v1/worker/gpu/model_runner.py`:

1. `profile_cudagraph_memory()` assigns every target/speculator manager and
   every eager/lazy wrapper to `profiling_pool`.
2. It calls both `self.cudagraph_manager.capture(...)` and
   `self.speculator.capture()` before computing the gross measurement.
3. `_cleanup_cudagraph_memory_profile()` begins and ends with accelerator
   synchronization, clears target/speculator/wrapper graphs, and frees the
   minimal profiling KV state.
4. Only after that cleanup does MRV2 compute
   `retained_pool_size = start_free - free_after_cleanup` and return
   `gross_cuda_graph_size - retained_pool_size`.
5. Before this patch, `profiling_pool` came from a fresh
   `graph_pool_handle()`. Production managers and wrappers are constructed
   with `get_global_graph_pool()` in `cudagraph_utils.py`, `cuda_graph.py`, and
   `breakable_cudagraph.py`.

Therefore the subtraction is a stronger future-headroom guarantee if the
future capture can reuse the pool whose retained capacity was subtracted. The
patched hardware result shows that this correction alone is insufficient to
prevent the separate production-capture failure.

The regression test fails if profiling calls `graph_pool_handle()` and checks
that the target manager, existing wrapper, lazily-created wrapper, cleanup,
and restored state all use the production pool.

Review byte pins on the current `dev/gilded-gnosis` target:

| File | Base SHA-256 | Patched SHA-256 |
|---|---|---|
| `vllm/v1/worker/gpu/model_runner.py` | `cbe6e868b8ea901b9d53f90d2548f97eea53be00f028ea2b90cb9103805d7543` | `2eab8362e2ce3e1004941988347b9921072053d52198b6f88be2d98d03cdd779` |
| `tests/v1/cudagraph/test_breakable_cudagraph.py` | `e7f5b222dc9f70de392e2e0a18b42c953c6eb74217934f2cb2b3dd0a41829d94` | `7af4d27f0aaf9426cfa40b21aacef7c5608a41fb572825f70673ccc33dd4261b` |

## Operator evidence

Source: CN3 operator report, 2026-07-22. The image under test was:

```text
ghcr.io/yatesdr/glm52-serve@sha256:7e51a7cf...
gilded-gnosis-v20-vllm3e731bc-si1a88b38-int8-nvme-mtpfix
```

It contained vLLM `3e731bc`, SparkInfer `1a88b38`, INT8 wire PR #69,
filesystem-tier PR #165, and indexer padding PR #166. The MRV2 pool fix in
this PR was not present.

The size-64 MRV2 profiling pass completed and emitted:

```text
Estimated MRV2 CUDA graph memory: 0.36 GiB additional
(1.08 GiB captured, 0.72 GiB retained and counted as non-torch)
```

Because this log is emitted after both target and speculator capture and after
the synchronized cleanup, it proves the complete profiled graph set finished;
the later failure is not evidence that the size-64 graph set is intrinsically
uncapturable.

Three clean-start boots were recorded:

| Boot | Configuration | Available KV | KV pool | Result |
|---|---|---:|---:|---|
| 1 | GMU .970, MNS 16, graph 64 | 3.25 GiB | ~435,968 estimated | Clean max-length fit failure: 480k required 3.57 GiB |
| 2 | GMU .980, MNS 16, graph 64 | 4.14 GiB | 544,000 | Illegal memory access during production speculative graph capture |
| 3 | GMU .978, MNS 8, graph 32 | 4.41 GiB | 592,640 | Identical illegal memory access at the identical site |
| 4 | GMU .980, MNS 16, graph 64, pool-reuse patch | 4.14 GiB | 555,520 | Identical illegal memory access at the identical site |

Boots 2 and 3 failed on all four workers through:

```text
spec_decode/autoregressive/speculator.py
  -> decode_cudagraph_manager.capture()
  -> cudagraph_utils.py: prepare_inputs_to_capture()
  -> input_batch.py: make_dummy()
  -> CUDA error: an illegal memory access was encountered
```

`make_dummy()` performs a host-to-device copy and is a synchronization point;
it reports a preceding asynchronous CUDA failure rather than necessarily being
the faulting operation.

The cap-32 differential motivated this patch. Reducing graph demand increased
available KV from 4.14 to 4.41 GiB and the allocator expanded KV from 544,000
to 592,640 tokens. Production capture gained no usable post-KV headroom and
failed in the same place. That was consistent with the two-pool hypothesis,
but Boot 4 is the decisive falsifier for that hypothesis as the crash cause.

At stand-down CN3 had zero containers and CUDA processes, 107.3 GB free in
`/dev/shm`, and 829 GB free on the NVMe filesystem. This rules out a stale
worker, shared-memory exhaustion, or filesystem-capacity failure for these
boots.

## Patched hardware result and falsification

The exact Boot 2 profile was rerun with image-local patch commit `982cda45`
(the same source bytes as this forward port):

```text
TP4 / DCP4 / MTP3
max_num_seqs=16
max CUDA graph capture size=64
max_model_len=480000
gpu_memory_utilization=0.980
```

The patch was byte-verified in-image. It changed the MRV2 tuple from
`1.08/0.72/0.36 GiB` captured/retained/additional to
`1.08/0.66/0.42 GiB`, and the KV pool changed from 544,000 to 555,520 tokens.
All four ranks agreed. This proves the new pool selection was active and
affected allocator state.

Production speculator capture then failed two seconds after starting with the
same illegal-address stack as Boots 2 and 3. Profiling speculator capture had
succeeded five minutes earlier. The shortfall hypothesis is therefore
falsified: same pool, changed accounting, three memory budgets, and two graph
caps all produce the invariant fault.

The remaining fault is an asynchronous illegal access from work preceding the
reported `InputBatch.make_dummy()` synchronization point. Production differs
from profiling chiefly in using the full KV/block-table state and in running
kernel warmup between the two passes.

## Validation

- `git diff --check`
- Python bytecode compilation of both changed files
- regression test added for pool identity and cleanup ordering
- CN3 differential evidence above obtained without this patch
- patched CN3 exact-profile capacity gate: **PASS, +11,520 KV tokens**
- the same boot's independent query-BMM crash: **FAIL, not fixed by this PR**

This PR must not be promoted using the boot-crash justification. Its
independent justification is graph-pool/accounting consistency plus the
measured capacity recovery; the consolidated current-base capture remains the
final compatibility gate.

## AI assistance disclosure

The root-cause analysis, patch, regression-test update, and PR write-up were
prepared with assistance from OpenAI Codex and reviewed by the operator/author.
