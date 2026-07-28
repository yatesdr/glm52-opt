# v20 MRV2 pool-reuse fix — proof boot result (Fable → Sol)

Date: 2026-07-22 · Boot 4 · Operator: Fable
Image: `ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-vllm3e731bc-si1a88b38-int8-nvme-mrv2fix`
(local only, `sha256:89a27fe2b4c0…`; not pushed)

**Outcome: Gate 1 FAILED. Stopped immediately per orders. No retry, no configuration ladder.**

## 1. Headline

**The fix is provably active and provably changed the memory accounting — and the crash is
unchanged, at the identical site.**

`982cda45` did exactly what it was designed to do. It did not fix the boot failure.

## 2. What the fix changed (it works)

| Signal | Boot 2 (pre-fix) | Boot 4 (with `982cda45`) | |
|---|---|---|---|
| MRV2 captured | 1.08 GiB | 1.08 GiB | same |
| MRV2 **retained** | 0.72 GiB | **0.66 GiB** | **−0.06** |
| MRV2 **additional** | 0.36 GiB | **0.42 GiB** | **+0.06** |
| Available KV | 4.14 GiB | 4.14 GiB | same |
| **GPU KV pool** | 544,000 | **555,520** | **+11,520** |

The retained/additional split moved in the predicted direction: profiling now shares the production
pool, so less is stranded as non-torch and more is correctly returned as reservable. **Pool came out
11,520 tokens above your estimate.** All four ranks reported identical values.

Against Derek's revised floor, **555,520 clears ≥500,000 by 11%**. Memory is not the blocker.

## 3. What did not change

Production speculator capture faulted 2 seconds after starting, all 4 workers, byte-identical
stack to boots 2 and 3:

```text
gpu_worker.py:804                     compile_or_warm_up_model()
  model_runner.py:1002                capture_model()
  spec_decode/autoregressive/speculator.py:125   self.decode_cudagraph_manager.capture(
  spec_decode/autoregressive/cudagraph_utils.py:74   super().capture(create_forward_fn, ...)
  cudagraph_utils.py:493              forward_fn = create_forward_fn(desc, warmup=False)
  spec_decode/autoregressive/cudagraph_utils.py:51  prepare_inputs_to_capture(
  cudagraph_utils.py:790              input_batch = InputBatch.make_dummy(
  input_batch.py:151                  torch.from_numpy(num_scheduled_tokens).to(device=device)
→ torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

Timeline (rank-uniform):

```text
22:59:36  speculator.py:92  "Capturing model for speculator..."   ← PROFILING pass
22:59:40  model_runner.py:961  MRV2 estimate printed              ← profiling capture SUCCEEDED
22:59:40  Available KV 4.14 GiB / KV pool 555,520
23:00-04  Capturing CUDA graphs (PIECEWISE) 11/11 complete
          sparkinfer fused_indexer cute-compile ×26 per rank, clean
23:04:43  speculator.py:92  "Capturing model for speculator..."   ← PRODUCTION pass
23:04:45  illegal memory access, all 4 workers
```

## 4. The inference this forces

**The pool-reuse defect was real, but it is not the cause of the crash.** Three independent reasons:

1. **Profiling and production now share one pool, and production still faults.** The shortfall
   mechanism no longer exists, so it cannot be what kills the production capture.
2. **The identical speculator capture succeeds at 22:59:36 and fails at 23:04:43** with *more*
   reservable memory than the pre-fix run had. Same code path, same shapes, same pool — different
   outcome depending only on which pass it is.
3. **`cudaErrorIllegalAddress` is not `cudaErrorMemoryAllocation`.** Across all three crashing boots
   the error has been an illegal address, never a CUDA OOM. Memory exhaustion in this path raises
   OOM. An out-of-bounds access does not become an OOM no matter how tight the budget is — and this
   boot had a *looser* budget than boot 2 and still faulted.

The faulting statement (`input_batch.py:151`) is a synchronizing H2D copy, so it is almost certainly
reporting an **async fault from earlier work**, not the true site. The most recent GPU work before it
is the first-pass speculator capture and the sparkinfer `fused_indexer` cute-kernels compiled in
between. Suggest looking upstream of the reported line rather than at it.

## 5. Boot ledger (all four)

| # | Config | Avail KV | KV pool | Result |
|---|---|---|---|---|
| 1 | GMU 0.970 / MNS16 / cap64 | 3.25 GiB | ~435,968 est | `ValueError`: 480k needs 3.57 GiB. Graceful |
| 2 | GMU 0.980 / MNS16 / cap64 | 4.14 GiB | 544,000 | illegal memory access @ spec capture |
| 3 | GMU 0.978 / MNS8 / cap32 | 4.41 GiB | 592,640 | illegal memory access, same site |
| 4 | **+`982cda45`**, GMU 0.980 / MNS16 / cap64 | 4.14 GiB | **555,520** | illegal memory access, same site |

Four boots, three distinct memory budgets (4.14 / 4.41 / 4.14 GiB), two distinct graph caps
(64 / 32), one MRV2 accounting change — **the same fault every time.** The fault is invariant to
every memory lever tried.

## 6. Build verification (for the record)

All five seams byte-verified in-image before boot:

| Pin | SHA-256 | |
|---|---|---|
| PR#69 `pcie_dma.py` / `.cu` | `5a6e6a0e…` / `70f4be32…` | unchanged |
| PR#165 `manager.py` | `653edbf4…` | unchanged |
| PR#166 `indexer.py` | `40572670…` | unchanged |
| **`982cda45` `model_runner.py`** | **`2eab8362e2ce3e1004941988347b9921072053d52198b6f88be2d98d03cdd779`** | **matches your pin exactly** |

Also confirmed: `get_global_graph_pool` present (2 refs), old `graph_pool_handle()` gone (0 refs),
`1b8e7f8f` absent from worktree history (0 matches). Profile was the exact previously-failed one —
TP4/DCP4, MTP3, MNS 16, cap 64, GMU 0.980, 480,000, i8_ring, NVMe 8 GiB, **zero diagnostic env vars**.

## 7. State

- **Stopped, no retry.** CN3 clear: 0 containers, 0 CUDA procs, `/dev/shm` 100 G free.
- **NVMe acceptance namespace still pristine** (0 entries) — Gate 2's precondition intact.
- **Evidence:** `~/glm52-test-artifacts/v20-window-evidence/v20-mrv2fix-FAILED.log` (2,147 lines)
  and `v20-mrv2fix-FAILED-inspect.json`. Container `0d497ada7660`, image `89a27fe2b4c0…`,
  StartedAt `2026-07-22T22:48:14Z`, **RestartCount 0**.
- **v19 prod: DOWN by choice.** Rollback armed and unused — `glm52-prod-ring.yaml`, image `ca8481…`,
  warm cache, ~12–15 min to serve.
- Gates 2, 3a, 3b, 3c, 4: **NOT REACHED.** No NVMe, throughput, needle or stress data exists.

## 8. Suggested next step (Sol's call)

**Update after comparison with the independently successful TP4/DCP4
configuration:** use the one-variable proof in
`v20-b12x-dcp-a2a-cap-proofboot.md`. Keep the Boot 4 image and profile and change
only `VLLM_DCP_A2A_MAX_TOKENS=64` to `16`. The successful configuration routes
32/64-row buckets to AG/RS; every CN3 failure admitted them to B12X CUDA-IPC
A2A. CN3's cap-32 attempt retained the 64-token A2A limit and therefore did not
test this distinction.

The paragraph below is retained as the original operator recommendation. Do
not run `CUDA_LAUNCH_BLOCKING` before the route discriminator above.

The memory question is closed — 555,520 clears the floor. What remains is a genuine illegal access in
the **production** speculator capture pass that survives every memory lever. I'd look at what differs
between the profiling pass (succeeds) and the production pass (faults) other than the pool: KV cache
now allocated and bound, attention metadata built against real cache shapes, and the sparkinfer
`fused_indexer` kernels JIT-compiled in between. Happy to run any single instrumented boot you
specify — including `CUDA_LAUNCH_BLOCKING=1` to move the report to the true fault site, which would
cost one boot (~17 min) and likely localise this directly.
