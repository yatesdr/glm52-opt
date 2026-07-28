# v20 Boot 8 candidate — reset CKV prefetch state at KV-cache replacement

## Status

Prepared source fix. Do not start this boot until the Boot 7 diagnostic log is
sealed; its first failing `CG_DIAG` boundary remains useful independent
evidence. This candidate is intended to be the next and only proof boot if
Boot 7 is consistent with a target-forward/ordering fault.

## Root cause found in source

The B12X CKV layer-prefetch path stores each layer's KV-cache tensor in the
class-level `_all_layer_kv_caches` registry. The July 17 prefetch correctness
fix deliberately preserved that registry between scheduler steps under an
explicit **stable cache pointers** assumption.

MRV2 CUDA-graph memory profiling, added July 20, violates that assumption:

1. `_init_minimal_kv_cache_for_profiling()` binds a temporary, minimal KV
   cache to the already-constructed attention implementations.
2. CUDA-graph profiling forwards populate the class-level CKV registry with
   those temporary cache tensors.
3. `_cleanup_cudagraph_memory_profile()` clears runner-owned caches and graph
   state, but did not clear the CKV registry.
4. Production `initialize_kv_cache()` rebinds `layer.kv_cache` without
   reconstructing the attention implementations or their class state.
5. On the first production forward, layer L may therefore prefetch L+1 from
   the old profiling-cache generation before L+1 has replaced its registry
   entry. The prefetch is asynchronous, so its illegal access can surface at
   the next synchronizing H2D copy in speculator input preparation.

This matches the observed phase split: the profiling capture succeeds, the
production capture fails, and memory/graph/A2A/indexer-width changes do not
move the fault.

It also explains the external TP4 comparison: its compose leaves
`KV_FP8_ROPE` unset. With `nvfp4_ds_mla`, that keeps synchronous CKV gather but
disables the layer-prefetch side stream. CN3 sets `KV_FP8_ROPE=1` for the
368-byte cache record, which enables the affected prefetch path.

## Fix

Commit:

```text
ce1746b7 fix(mrv2): reset CKV prefetch cache bindings
```

Patch:

```text
patches/v20-ckv-prefetch-profile-reset/0001-fix-mrv2-reset-CKV-prefetch-cache-bindings.patch
SHA-256 cdd456974545baf5381c9bb1cdd104bf86e63390ff5c26ddffa98027cf16d3b8
```

The cleanup path invokes an optional attention-implementation lifecycle hook
before releasing the temporary cache. B12X sparse MLA uses the hook to clear:

- the layer-to-cache registry;
- the cross-layer completion event;
- the ping-pong workspace index.

The fix does **not** disable CKV gather, CKV prefetch, i8-ring, MTP, CUDA
graphs, A2A, DRAM offload, or NVMe offload. The first forward after the cache
replacement primes the empty registry using the existing synchronous gather;
subsequent forwards recover normal layer prefetch.

Production source change: 30 added lines across two files. Tests add 56 lines
to two existing suites.

## Local proof

- Patch applies cleanly to the exact Boot 7/Boot 6 combined source.
- Reverse application restores a clean tree.
- Python source parsing/compile checks passed.
- `git diff --check` passed.
- Two focused unit tests were added:
  - backend reset drops cache/event/buffer state;
  - MRV2 cleanup invokes the reset before emptying `layer.kv_cache`.

The local Mac does not have the repository-mandated `uv` environment or
PyTorch, so pytest was not claimed locally. Run the focused tests in the
build image.

## Byte pins against the current combined source

Inputs:

```text
model_runner.py       2eab8362e2ce3e1004941988347b9921072053d52198b6f88be2d98d03cdd779
b12x_mla_sparse.py    e06b35c88db6691c11cdef0e1a134746060682f596478365661847f681d4e0bb
```

Expected outputs after the patch:

```text
model_runner.py       526ac1643d9cbbf6e03fe505fdc64e4ab0c78bb898eb0b7013926e18a38ccf17
b12x_mla_sparse.py    ee08b603a266b752791ac3a811b23eb0680d9834d84d97323fcc11280f4e927d
```

The Boot 6 aligned indexer and all SparkInfer/int8 files are unchanged.

## Focused test command in the image

```bash
.venv/bin/python -m pytest -q \
  tests/v1/cudagraph/test_breakable_cudagraph.py \
  tests/v1/attention/test_b12x_mla_dcp_workspace.py
```

At minimum require the two new tests by name to pass if the full files need
unsupported GPU fixtures filtered out.

## One proof boot

Use the Boot 6 production candidate as the input so the diagnostic barriers
do not alter ordering. Retain all production requirements:

- TP4 / DCP4 / MTP3;
- max model length 480000;
- max sequences 16 and graph cap 64;
- `KV_FP8_ROPE=1`, CKV gather enabled;
- `i8_ring`;
- MRV2 pool-reuse fix;
- aligned indexer block table;
- DRAM plus bounded NVMe offload.

Remove `VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS`; do not set
`CUDA_LAUNCH_BLOCKING=1`.

Gate 1 passes only if the server reaches ready state with:

- no illegal access;
- no `EngineDeadError` or worker death;
- `RestartCount=0` and unchanged container ID/`StartedAt`;
- the expected KV pool recorded;
- one deterministic liveness request returning correctly.

On failure, stop. Do not tune another configuration. Preserve the full log
and correlate it with Boot 7's first failing diagnostic boundary.

