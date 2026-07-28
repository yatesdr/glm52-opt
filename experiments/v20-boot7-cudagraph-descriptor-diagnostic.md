# v20 Boot 7 — CUDA-graph descriptor fault localization (Fable handoff)

## Objective

Localize the invariant asynchronous CUDA illegal access to the exact graph
manager, descriptor, and capture stage in one boot. This is a diagnostic
build, not a production fix and not an upstream PR.

Do not run a configuration ladder. Do not combine this with
`CUDA_LAUNCH_BLOCKING=1`.

## Exact diagnostic source

- Local commit: `f0021cc306cf0d78174665654653432d0c424db6`
- Changed production file:
  `vllm/v1/worker/gpu/cudagraph_utils.py`
- Expected Boot 6 input SHA-256:
  `c82a956cb3df0827c9aae090a05fa3b2326601ae0f5323808ca1669c26a6ef91`
- Boot 7 diagnostic output SHA-256:
  `001365a061da3e5aeaa8bd3e7f04a41eac7b58de9020afbf4ae2d29be5b6a3f3`
- Patch:
  `patches/v20-cudagraph-descriptor-diagnostic/0001-debug-cudagraph-localize-asynchronous-capture-faults.patch`
- Patch SHA-256:
  `1fdb1ceb6ef175b82d6c3e147c0c7295df914e8c13084c3c1999eb6f64e042e4`

Local proof:

```text
3 passed
py_compile: passed
git diff --check: passed
reverse apply check: passed
```

## What the patch does

With `VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS=1`, every graph descriptor is
synchronized immediately after these safe boundaries:

1. `warmup_inputs`
2. `warmup_forward`
3. `fresh_capture_inputs`
4. `piecewise_capture`
5. `b12x_prewarm`
6. `full_capture`
7. `manager_exit`

The synchronizations occur outside any nested `torch.cuda.graph(...)` block.
Each boundary emits a structured record:

```text
CG_DIAG BEGIN label=... stage=... desc=BatchExecutionDescriptor(...)
CG_DIAG PASS  label=... stage=... desc=BatchExecutionDescriptor(...)
CG_DIAG FAIL  label=... stage=... desc=BatchExecutionDescriptor(...)
```

The `label` distinguishes target profiling (`profiling_cuda_graph_memory`),
target production (`capturing_cuda_graphs`), and the prefill/decode speculator
managers. Because the same managers are called once during profiling and once
during production, chronology plus the target labels gives a direct A/B.

When the flag is absent, the helper performs no synchronization.

## Gate 0 — build and byte proof

Use the exact Boot 6 image as the input:

```text
…-int8-nvme-mrv2fix-bt1876
local image sha256:427d4122804e…
```

Before applying the patch, require `cudagraph_utils.py` to equal the input pin
`c82a956c...`. Apply the patch cleanly, then require the output pin
`001365a0...`. Preserve all Boot 6 pins, especially:

- PR #69 INT8 wire files;
- PR #165 filesystem manager;
- PR #166 aligned `indexer.py` = `d419af9e...`;
- MRV2 `model_runner.py` = `2eab8362...`.

Add exactly one environment setting:

```yaml
- VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS=1
```

Explicitly ensure this is absent:

```yaml
CUDA_LAUNCH_BLOCKING
```

## Gate 1 — one diagnostic boot

Retain the complete Boot 6 runtime profile unchanged:

- `MAX_MODEL_LEN=480000`;
- TP4 / DCP4 / MTP3;
- `MAX_NUM_SEQS=16`;
- graph cap 64;
- GMU 0.980;
- A2A cap 16 with large backend `ag_rs`;
- `i8_ring`;
- unchanged DRAM and bounded NVMe configuration;
- unchanged cache volumes.

This boot may take somewhat longer because the diagnostic deliberately drains
all CUDA work at every boundary. That slowdown is expected and is not a
performance result.

Capture the full unfiltered log. In a second view, watch:

```bash
docker logs -f CONTAINER 2>&1 | grep --line-buffered -E \
  'CG_DIAG|Capturing model for speculator|illegal memory access|Traceback'
```

## Stop condition and evidence

On the first `CG_DIAG FAIL` or any other CUDA error:

1. stop; do not retry or tune anything;
2. preserve the entire log and inspect JSON;
3. report the last 20 `CG_DIAG PASS` records;
4. report the first `CG_DIAG FAIL` record and 80 surrounding lines;
5. search the profiling portion for the identical descriptor and report every
   stage that passed there;
6. record container ID, `StartedAt`, `RestartCount`, image ID, and all byte
   pins.

The most important discriminator is the first failing tuple:

```text
(label, stage, cg_mode, num_tokens, num_reqs, uniform_token_count,
 max_req_tokens, num_active_loras)
```

### Interpretation

- Failure in `production target / warmup_forward` or `b12x_prewarm`: the eager
  target forward launches the bad operation.
- Failure in `production target / full_capture`: the captured target forward
  launches it.
- All production-target stages pass, then speculator `manager_entry` fails:
  target manager/context teardown launched or exposed it.
- Speculator `warmup_inputs` fails after its own `manager_entry` passed: the
  fault is truly in speculator dummy-input preparation, not deferred target
  work.
- Profiling and production fail on different stages for the same descriptor:
  preserve both sequences; their state/lifetime difference is the next source
  seam.

## If the diagnostic boot reaches serving

Do not call that a production qualification. The synchronization changed
capture ordering and may have masked a missing fence. Leave the server up,
run only one deterministic liveness request, preserve the complete log, and
report. The PASS itself would identify ordering as the defect class; the next
patch would reduce the diagnostic barriers to the single required fence.

## CUDA_LAUNCH_BLOCKING follow-up

Use `CUDA_LAUNCH_BLOCKING=1` only if Boot 7 identifies a failing descriptor and
stage but the normal traceback still does not identify the operation within
that stage. That would be one separate targeted boot. Combining it with Boot 7
would change two scheduling mechanisms and make the result harder to trust.
