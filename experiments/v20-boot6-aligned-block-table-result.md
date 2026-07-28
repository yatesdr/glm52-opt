# v20 Boot 6 — aligned MTP block-table (PR #166 @ `c9a2e28d`): result (Fable → Sol)

Date: 2026-07-23 · Boot 6 · Operator: Fable
Image: `…-int8-nvme-mrv2fix-bt1876`, `sha256:427d4122804e…`
Compose: `glm52-v20-tonight.yaml` sha256 `2e83cf88428e0992…` (Boot 5 = `0d11500a6f1d42b1…`)

**Outcome: Gate 1 FAILED. Identical illegal access, identical site, identical 2-second delay.**

Stopped per your Gate 1 FAIL instruction. No retry. No parameter ladder — GMU, graph cap, MNS,
A2A and max length all untouched.

## 1. Gate 0 passed byte-exact

| Pin | SHA-256 | |
|---|---|---|
| PR#69 `pcie_dma.py` / `.cu` | `5a6e6a0e…` / `70f4be32…` | unchanged |
| PR#165 `manager.py` | `653edbf4…` | unchanged |
| **PR#166 `indexer.py`** | **`d419af9e1e84a1fa316246d0cb930f6559dcb5859a8e71bb54240dbf41fb0cde`** | **matches your Boot 6 output pin** |
| `982cda45` `model_runner.py` | `2eab8362…` | unchanged |

Provenance verified end to end without a repo fetch:
patch file = `c274ebe26dbc4d2e…` (matches your pin and its own manifest) → input `indexer.py` =
`4057267018b9…` (matches your stated Boot 5 input) → `git apply` clean, tests hunk skipped →
output `d419af9e…`. Byte-identical to PR head `c9a2e28d`.

Semantic check in-image (no GPU):

```text
alignment: PASS   a(1875,64)=1876   a(1874,64)=1874   a(1876,64)=1876
```

**The compile key did move as you predicted.** Boot 6 ran a *fresh* SparkInfer `fused_indexer`
compile round (26 kernels/rank from `compile-start number=1`), not a cache hit — so the failed
odd-width kernel variant is definitively not in play.

Only one line differed from the Boot 5 compose: the image tag. `MAX_MODEL_LEN=480000` retained.

## 2. The fix is real and measurable — and the crash is unchanged

| Signal | Boot 5 (1875) | **Boot 6 (1876)** | Δ |
|---|---|---|---|
| MRV2 captured | 1.09 GiB | 1.09 GiB | — |
| MRV2 retained | 0.75 | 0.69 | −0.06 |
| MRV2 additional | 0.34 | 0.40 | +0.06 |
| **Available KV** | 4.19 GiB | **4.13 GiB** | **−0.06** |
| **KV pool** | 562,432 | **554,496** | **−7,936** |
| Max concurrency @480k | 1.17× | 1.16× | |

The −7,936 tokens is the widened buffer being paid for in real memory — the expected cost of the
fix, not a regression. Still **11% above the 500k floor.**

## 3. Identical failure

```text
00:09:57  speculator.py:92  "Capturing model for speculator..."   ← PROFILING pass — SUCCEEDED
00:10:01  MRV2 estimate printed
00:10:02  Available KV 4.13 GiB / pool 554,496 / 1.16x
00:10-14  PIECEWISE 11/11 complete; 26 fresh cute-kernels/rank compiled clean
00:14:23  speculator.py:92  "Capturing model for speculator..."   ← PRODUCTION pass
00:14:25  illegal memory access, all 4 workers                     ← 2 seconds, as in Boots 2-5
```

Stack frames byte-identical to Boots 2, 3, 4 and 5 — `gpu_worker.py:804` → `model_runner.py:1002`
→ `speculator.py:125` → `autoregressive/cudagraph_utils.py:74` → `cudagraph_utils.py:493` →
`autoregressive/cudagraph_utils.py:51` → `cudagraph_utils.py:790` → `input_batch.py:151`.

## 4. Two negative results worth having

- **`Target sizes` never appeared** — 0 occurrences. The Python shape error the old revision
  removed stayed removed.
- **Your new `ValueError("MTP block-table row width…")` guard never fired** — 0 occurrences.

**Caveat on the guard, stated precisely:** absence of the guard is *not* proof the widths matched.
The guard lives in `_prepare_decode_tensors`, and the fault occurs in
`prepare_inputs_to_capture` → `InputBatch.make_dummy`, which is upstream. I cannot tell from this
log whether the guard was reached and passed, or simply never reached. Treat it as
"did not fire", not as "widths verified equal at runtime."

## 5. Boot ledger (all six)

| # | Distinguishing lever | Avail KV | KV pool | Result |
|---|---|---|---|---|
| 1 | GMU 0.970 baseline | 3.25 GiB | ~435,968 est | `ValueError`, graceful |
| 2 | GMU 0.980 | 4.14 GiB | 544,000 | illegal access @ spec capture |
| 3 | MNS8 / cap32 / GMU 0.978 | 4.41 GiB | 592,640 | illegal access, same site |
| 4 | + MRV2 pool reuse `982cda45` | 4.14 GiB | 555,520 | illegal access, same site |
| 5 | + A2A cap 16 → AG/RS | 4.19 GiB | 562,432 | illegal access, same site |
| 6 | **+ aligned block table 1876 (`c9a2e28d`)** | **4.13 GiB** | **554,496** | **illegal access, same site** |

**Six boots. Five distinct memory budgets, two graph caps, the MRV2 pool-reuse fix, the DCP route,
and now the block-table width contract — the same fault, the same frames, the same 2-second delay,
every single time.** Every hypothesis tested so far has been active and measurable in the logs, and
none has moved the failure.

## 6. Now excluded

- Memory availability (`cudaErrorIllegalAddress`, never `cudaErrorMemoryAllocation`; faults across
  4.13 / 4.14 / 4.19 / 4.41 GiB)
- MRV2 graph-pool lifetime
- Graph-size / MNS tuning
- B12X DCP A2A large-batch route + IPC staging size
- Indexer block-table row-width alignment at the 480k seam
- Stale kernel cache (fresh compile key this boot)

Still unexercised at the fault point per your review: PR #69 (below its 6 MiB DMA threshold),
PR #165 (no NVMe I/O during capture).

## 7. State

- **Stopped, no retry, no ladder.** CN3 clear: 0 containers, 0 CUDA procs, `/dev/shm` 100 G free.
- **NVMe acceptance namespace still pristine (0 entries)** — Gate 2 precondition intact across all
  six boots.
- **Evidence:** `~/glm52-test-artifacts/v20-window-evidence/v20-boot6-bt1876-FAILED.log` and
  `v20-boot6-bt1876-FAILED-inspect.json`. Container `ed3c9233fba8…`, image `427d4122804e…`,
  StartedAt `2026-07-22T23:58:24Z`, **RestartCount 0**.
- **Gates 2, 3a, 3b, 3c, 4: NOT REACHED.** No NVMe, throughput, needle or stress data exists.
- **v19 prod: DOWN by choice.** Rollback armed and unused — `glm52-prod-ring.yaml`, `ca8481…`,
  warm cache, ~12–15 min to serve.

## 8. Ready for the instrumented build

Your stated next artifact is the source-instrumented build that synchronizes immediately after each
target graph descriptor. I'm ready to build and run it the moment you hand it over — same
verification discipline, one boot, no ladder.

Two operator notes that may help you scope it:

1. **The profiling pass survives and the production pass dies, every time, 2 s in.** Whatever
   instrumentation you add, the highest-value comparison is the same descriptor sequence across
   those two passes rather than the production pass alone — the difference is where the answer is.
2. **`CUDA_LAUNCH_BLOCKING=1` remains untried** at your direction. If your instrumented build
   doesn't localize it in one boot, that's still available and costs ~17 min.

## 9. Operational flag for Derek (not a Sol item)

Six boots consumed; prod has now been down by choice since 21:09Z (~3 h). No user traffic observed
since 19:57:50Z the previous evening. The window remains open and the rollback is armed and unused,
so nothing is at risk yet — but the decision tree calls for v19 rollback if the window expires
mid-test, and that call is Derek's to make.
