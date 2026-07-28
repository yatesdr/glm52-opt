# v20 Boot 5 — B12X DCP A2A cap discriminator: result (Fable → Sol)

Date: 2026-07-22 · Boot 5 · Operator: Fable
Image: `sha256:89a27fe2b4c0…` (Boot 4 image, unchanged, not rebuilt)
Compose: `glm52-v20-tonight.yaml` sha256 `0d11500a6f1d42b1…` (Boot 4 = `b5cfa8e8de9a0bbc…`)

**Outcome: Gate 1 FAILED with the identical illegal access.
Per your spec §"FAIL with the identical illegal access": the A2A-large-batch hypothesis is falsified.**

Stopped immediately. No retry, no memory or graph-size ladder.

## 1. The change was exactly one line

```diff
- - VLLM_DCP_A2A_MAX_TOKENS=64
+ - VLLM_DCP_A2A_MAX_TOKENS=16
```

`diff` against the Boot 4 compose returns that single hunk and nothing else.
`VLLM_DCP_A2A_LARGE_BACKEND=ag_rs` **was already set** in our compose (line 84) — it was never a
variable, so no change was needed there. Container env confirmed at runtime:
`VLLM_DCP_A2A_MAX_TOKENS=16`, `VLLM_DCP_A2A_LARGE_BACKEND=ag_rs`, `VLLM_USE_B12X_DCP_A2A=1`.

Image identical (not rebuilt); all five byte pins re-verified in-image before boot:
`5a6e6a0e` / `70f4be32` / `653edbf4` / `40572670` / `2eab8362`.

## 2. The knob was provably live — this was not a no-op boot

| Signal | Boot 4 (cap 64) | **Boot 5 (cap 16)** | Δ |
|---|---|---|---|
| MRV2 captured | 1.08 GiB | 1.09 GiB | +0.01 |
| MRV2 retained | 0.66 GiB | 0.75 GiB | +0.09 |
| MRV2 additional | 0.42 GiB | 0.34 GiB | −0.08 |
| **Available KV** | 4.14 GiB | **4.19 GiB** | **+0.05** |
| **GPU KV pool** | 555,520 | **562,432** | **+6,912** |
| Max concurrency @480k | 1.15× | **1.17×** | |

The +0.05 GiB is the mechanism from the `envs.py` docstring appearing in measurement:

> *"Also bounds the B12X DCP IPC staging allocation, which scales linearly with this value."*

Cap 64 → 16 shrank the IPC staging buffer and the KV allocator absorbed the slack. **The
route/staging change unambiguously took effect and the crash was unaffected.**

## 3. Identical failure

```text
23:35:16  speculator.py:92  "Capturing model for speculator..."   ← PROFILING pass — SUCCEEDED
23:35:20  MRV2 estimate printed
23:35:21  Available KV 4.19 GiB / KV pool 562,432 / 1.17x
23:35-40  PIECEWISE 11/11 complete; sparkinfer cute-kernels clean
23:40:17  speculator.py:92  "Capturing model for speculator..."   ← PRODUCTION pass
23:40:19  illegal memory access, all 4 workers                     ← 2 seconds, same as Boot 4
```

Stack frames byte-identical to Boots 2, 3 and 4:

```text
gpu_worker.py:804  compile_or_warm_up_model → model_runner.py:1002 capture_model
 → speculator.py:125 capture → autoregressive/cudagraph_utils.py:74 capture
 → cudagraph_utils.py:493 create_forward_fn → autoregressive/cudagraph_utils.py:51
 → cudagraph_utils.py:790 prepare_inputs_to_capture → input_batch.py:151 make_dummy
→ torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

Same 2-second delay from production-capture start to fault. Same rank-uniform pattern.

## 4. Boot ledger (all five)

| # | Config | Avail KV | KV pool | Result |
|---|---|---|---|---|
| 1 | GMU 0.970 / MNS16 / cap64 / A2A64 | 3.25 GiB | ~435,968 est | `ValueError`, graceful |
| 2 | GMU 0.980 / MNS16 / cap64 / A2A64 | 4.14 GiB | 544,000 | illegal access @ spec capture |
| 3 | GMU 0.978 / MNS8 / cap32 / A2A64 | 4.41 GiB | 592,640 | illegal access, same site |
| 4 | +`982cda45`, GMU 0.980 / MNS16 / cap64 / A2A64 | 4.14 GiB | 555,520 | illegal access, same site |
| 5 | **A2A16 + ag_rs**, GMU 0.980 / MNS16 / cap64 | **4.19 GiB** | **562,432** | illegal access, same site |

**Five boots. Four memory budgets, two graph caps, one accounting fix, one DCP route change — the
same fault, the same site, the same 2-second delay, every time.** Nothing tried so far moves it.

## 5. What is now excluded

- **Memory availability** — falsified at Boot 4 (`cudaErrorIllegalAddress`, never
  `cudaErrorMemoryAllocation`; faults across 4.14 / 4.19 / 4.41 GiB budgets).
- **MRV2 graph-pool lifetime** — fix present and active, crash unchanged.
- **Graph-size / MNS tuning** — cap 64 and cap 32 both fault.
- **B12X DCP A2A large-batch route** — cap 16 forces 32/64-row buckets to AG/RS, staging allocation
  demonstrably shrank, crash unchanged. **Falsified by your own criterion.**

Still not exercised at the fault point, per your code review: PR #69 (below its 6 MiB DMA
threshold), PR #165 (no NVMe store/load during capture).

## 6. Next discriminator — one caveat before I run it

Your spec's FAIL branch prescribes: *"Prepare one exact-base 479,744 boot without the PR #166
overlay; do not combine that test with any other change."*

**Flagging that as written this is two changes, not one:** dropping `max_model_len`
480,000 → 479,744 **and** removing the PR #166 indexer overlay. If it boots, we won't know which
of the two was responsible, and PR #166 was specifically the 480k block-table trim — so the two are
entangled by design and a pass would leave the attribution ambiguous.

Two ways to split it, both one-variable, ~17 min each:

- **5a — keep PR #166, set 479,744.** Tests the padded block-table seam alone. If it passes, PR #166
  is exonerated and the 480,000 boundary is the fault.
- **5b — keep 480,000, drop PR #166.** Tests the overlay alone against the unpatched indexer base.
  Note this may simply reproduce the pre-#166 480k failure mode rather than a clean result.

I'd run **5a first** — it needs no rebuild (env/flag only), so it's the cheaper and faster of the
two, and PR #166 exists precisely because 480k needed the trim, which makes the boundary the more
likely carrier.

**Not proceeding without your call.** Say the word and I'll run whichever you want — including the
combined test exactly as originally specified if you'd rather accept the ambiguity for speed.

## 7. State

- **Stopped, no retry.** CN3 clear: 0 containers, 0 CUDA procs, `/dev/shm` 100 G free.
- **NVMe acceptance namespace still pristine (0 entries)** — Gate 2 precondition intact.
- **Evidence:** `~/glm52-test-artifacts/v20-window-evidence/v20-boot5-a2a16-FAILED.log` (724 KB) and
  `v20-boot5-a2a16-FAILED-inspect.json`. Container `4e0ffc9035c6…`, image `89a27fe2b4c0…`,
  StartedAt `23:21:03Z`, **RestartCount 0**.
- **v19 prod: DOWN by choice.** Rollback armed and unused — `glm52-prod-ring.yaml`, `ca8481…`,
  warm cache, ~12–15 min to serve.
- Gates 2, 3a, 3b, 3c, 4: **NOT REACHED.** No NVMe, throughput, needle or stress data exists.

## 8. Silver lining on the pool question

562,432 tokens at MNS16 / cap64 / GMU 0.980 is the **largest v20 pool measured at the production
profile** — above Boot 4's 555,520 and 12.5% above the 500k floor. Whatever fixes the capture crash,
the cap-16 setting looks worth keeping for the memory alone. It would need decode measured at
C1/4/8/16 first, since with MTP3 the route boundary sits near four active requests and AG/RS now
carries everything above it.
