# v20 acceptance window — operator report (Fable → Sol)

Date: 2026-07-22 · Window 21:09Z–22:45Z · Operator: Fable
Image under test: `ghcr.io/yatesdr/glm52-serve@sha256:7e51a7cf…`
(tag `gilded-gnosis-v20-vllm3e731bc-si1a88b38-int8-nvme-mtpfix`
= v20 decode base + PR#69 i8_ring + PR#165 NVMe eviction + PR#166 MTP indexer fix)

**Outcome: HOLD. v20 never reached serving. Three boots, three failures.**
No acceptance gate beyond Gate 1 was executed — no NVMe, needle, throughput, or stress data exists.

## 1. Headline for Sol

**Your reported-successful combination did not reproduce on CN3.**

> reported: *successful: MNS 8, graph cap 32, GMU .978* / *failed: MNS 16, graph cap 64, GMU .980*

On CN3, **MNS 8 / cap 32 / GMU 0.978 failed** with the identical illegal-memory-access signature at
the identical site. Either your successful run already carried `982cda45`, or some other variable
differs between our setups. Worth reconciling before the next attempt.

## 2. All three boots

| # | Config | Available KV | **KV pool** | Result |
|---|---|---|---|---|
| 1 | GMU 0.970, MNS 16, cap 64 | 3.25 GiB | ~435,968 (est) | ❌ `ValueError`: 3.57 GiB needed for 480k > 3.25 available. Clean fit failure, graceful exit |
| 2 | GMU 0.980, MNS 16, cap 64 | 4.14 GiB | **544,000** (1.13× @480k) | ❌ CUDA **illegal memory access** during graph capture |
| 3 | GMU 0.978, **MNS 8, cap 32** | **4.41 GiB** | **592,640** | ❌ CUDA **illegal memory access**, same site |

Crash site identical in boots 2 and 3, all 4 workers:

```text
spec_decode/autoregressive/speculator.py
  → decode_cudagraph_manager.capture()
  → cudagraph_utils.py:493  prepare_inputs_to_capture()
  → input_batch.py:151      make_dummy()
  → torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

Note the reported line is a synchronizing `.copy_()`; CUDA surfaces async faults there, so it is
very likely **not** the true fault site.

## 3. The finding that corroborates your MRV2 pool-reuse diagnosis

**Reducing graph memory made the crash *no better*, and arguably worse — exactly as your
hidden-shortfall model predicts.**

Going cap 64 → cap 32 freed graph memory, which the KV allocator promptly absorbed:
available KV rose 4.14 → **4.41 GiB**, pool 544,000 → **592,640**. The production capture was
therefore left with *no more* headroom than before, and still faulted. The shortfall does not
respond to graph-size tuning because the reserved capacity is in a **different pool** — which is
precisely the defect `982cda45` addresses.

**Implication: MNS 8 / cap 32 is not a viable workaround on CN3. The pool-reuse
fix is the next direct proof; it remains runtime-unproven until that boot.**

## 4. Useful side result: the pool question is effectively answered

Derek revised the acceptance floor to **≥500k tokens** (from 600k), accepting that v19's 644,864 was
partly errant accounting. Against that floor:

- **592,640 tokens @ GMU 0.978 / cap 32 already clears it by ~19%** — measured, not projected.
- `982cda45` does **not** return the retained 0.72 GiB to KV. It keeps that
  capacity in the global graph pool and makes it reusable by production
  capture. Expected KV pools therefore remain roughly **544,000 at
  cap-64/GMU 0.980** and **592,640 at cap-32/GMU 0.978**; the expected change is
  successful capture, not a larger KV allocation.

So memory is no longer the blocker. **The graph-pool lifetime defect is the only thing standing
between us and a v20 acceptance run.**

## 5. Corrections to my earlier analysis

- I hypothesised an **intrinsic MTP-route bug** at 64-wide capture. **Your evidence falsifies it** —
  the full size-64 target and speculative graph set captured and synchronized during profiling. I
  was wrong; the capture path is sound and the fault is memory availability, not indexing.
- I attributed the v19→v20 pool drop entirely to honest re-accounting. That remains true for
  ~0.65 GiB (weights +0.42, sparse-DCP +0.23), but **0.72 GiB of it was a defect, not accounting.**

## 6. Operationally relevant, unrelated to the defect

**v19's 64 GB offload mmap survives `docker compose down` as a root-owned file in `/dev/shm`.**
It must be removed with elevated privileges before any subsequent 64 GB tier can allocate, or the
next boot fails at offload init (the original EFAULT signature). This bit us at teardown and I now
clear it explicitly before every boot. Recommend folding a privileged cleanup into the entrypoint or
the runbook.

## 7. State at stand-down

- **v20: not serving.** CN3 cleared, 0 containers, GPUs free, `/dev/shm` released.
- **v19 prod: DOWN.** Rollback armed and unused — `glm52-prod-ring.yaml`, image `ca8481…`,
  warm cache present, ~12–15 min to serve. Derek holds ~12 h of window; prod is down by choice.
- **NVMe acceptance namespace `/nvme-kv/glm52-v20-acceptance`: still pristine.** No inference ever
  ran, so Gate 2's fresh-namespace precondition is intact for the next attempt.
- **Evidence:** `~/glm52-test-artifacts/v20-window-evidence/` — v19 baseline inspect/metrics/log plus
  all three failed-boot logs with timestamps.

## 8. Next step (needs Sol)

Bake `982cda45` (`workspace/vllm-v20-mrv2-pool-reuse`) into the candidate
image and boot the **exact failed production profile: MNS 16 / cap 64 / GMU
0.980**. This directly proves the fix against boot 2 and is the configuration
wanted in production. Do not spend another boot on MNS 8 / cap 32 first.
