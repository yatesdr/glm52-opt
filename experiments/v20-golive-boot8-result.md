# v20 go-live candidate (Boot 8) — W4A16 cooperative grid + CKV profile reset: result

Date: 2026-07-23 · Boot 8 · Operator: Fable
Image: `…-int8-nvme-bt1876-w4a16coop-ckvreset`, `sha256:5f0c7b43daaa…`
Compose: `glm52-v20-tonight.yaml` sha256 `cda7e0b4955c44e8…`

**Outcome: boot gate FAILED. Same illegal access, same site. Stopped, no tuning ladder.**
Qualification suite not started. v20 is **not** online.

## 1. Gate 0 — all six byte checks passed

| File | Input | Output | |
|---|---|---|---|
| SparkInfer `w4a16/kernel.py` | `7b15236d…` ✓ | `7bae99df…` ✓ | both match runbook |
| vLLM `model_runner.py` | `2eab8362…` ✓ | `526ac164…` ✓ | both match runbook |
| vLLM `b12x_mla_sparse.py` | `e06b35c8…` ✓ | `ee08b603…` ✓ | both match runbook |

Patch hashes verified first: `1a76f60f…` (W4A16) and `cdd45697…` (CKV) — both match. Clean apply,
test hunks skipped. Carried forward unchanged: PR#69 `5a6e6a0e`/`70f4be32`, PR#165 `653edbf4`,
PR#166 `d419af9e`.

Both patches confirmed *live* at runtime, not merely on disk: the MRV2 log line moved
`model_runner.py:961` → **`:972`**, and `cudagraph_utils` frames moved `493/790` → **`550/858`**.

## 2. Production geometry as specified

Compose diff vs Boot 7 was exactly three lines: image tag, diagnostics env deleted, A2A restored
**16 → 64**. Verified live in-container: `VLLM_DCP_A2A_MAX_TOKENS=64`, `i8_ring` (both vars),
`KV_FP8_ROPE=1`, `SPARKINFER_PRINT_COMPILE_PROGRESS=1` retained.
`CUDA_LAUNCH_BLOCKING` and `VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS`: **0 occurrences**.
Unchanged: TP4/DCP4/MTP3, 480,000, MNS16, MNBT 3072, cap 64, GMU 0.980, `nvfp4_ds_mla`, CKV gather,
prefix caching, `expandable_segments:False`, `ipc: "host"`, DRAM 64 GB, NVMe 8 GiB.
NVMe namespace empty, on `/dev/nvme0n1p2` ext4; zero stale `/dev/shm/vllm_offload_*.mmap`.

## 3. Timeline and failure

```text
01:10:36  boot
01:25:16  profiling speculator capture — SUCCEEDED
01:26:58  MRV2 1.08 / 0.66 / 0.42 · Available KV 4.14 GiB · pool 555,520 · 1.16x
01:26:58  profiling decode capture reached 16/16
01:31:26  PRODUCTION speculator capture
01:31:28  illegal memory access, all 4 workers
```

Stack identical to Boots 2–7 (line numbers shifted by the CKV patch):

```text
gpu_worker.py:804 → model_runner.py:1013 capture_model → speculator.py:125
 → autoregressive/cudagraph_utils.py:74 → cudagraph_utils.py:550
 → autoregressive/cudagraph_utils.py:51 → cudagraph_utils.py:858 prepare_inputs_to_capture
 → input_batch.py:151 make_dummy → torch.from_numpy(...).to(device=device)
→ torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

## 4. The one signal that changed — and its limits

Production decode-capture progress before the fault, from the tqdm bar:

| Boot | Diagnostics | Production decode bar at fault |
|---|---|---|
| 7 | ON (CG_DIAG) | **3/16** — CG_DIAG placed the true failure at the **8th** descriptor, size **9** |
| 8 | OFF | **6/16** |

**Boot 8 got measurably further into production decode capture than Boot 7.** That is consistent
with the M=9 fault being fixed and a *later, different* descriptor now failing.

**I am not claiming that as proven.** Two reasons for caution:

1. The tqdm bar is coarse and flush-timed — in Boot 7 it read 3/16 while CG_DIAG showed seven
   descriptors actually complete. Bar position is a lower bound, not a descriptor index.
2. Without `CG_DIAG` this boot cannot name the failing descriptor at all, so the runbook's stated
   PASS criterion — *"proven fixed when production decode capture passes size 9 and continues
   through sizes 8 to 1"* — **cannot be evaluated from this log.**

The comparison is suggestive and worth acting on, not conclusive.

## 5. Memory (both patches visible in accounting)

| | Boot 7 | **Boot 8** |
|---|---|---|
| MRV2 captured/retained/additional | 1.09 / 0.69 / 0.40 | **1.08 / 0.66 / 0.42** |
| Available KV | 4.13 GiB | **4.14 GiB** |
| **KV pool** | 554,496 | **555,520** |

Pool +1,024 tokens; clears the 500k floor by 11%. The retained/additional shift is the CKV
profiling-generation reset doing what it says.

Boot was slower than predecessors (MRV2 at +16:22 vs +11:30) because the W4A16 kernel change
invalidated the SparkInfer compile key — 66+ kernels recompiled vs ~26. Expected, allowed per
runbook §2, not a performance result.

## 6. State

- **Stopped, no ladder.** CN3 clear: 0 containers, 0 CUDA procs.
- **NVMe acceptance namespace still pristine (0 entries)** — precondition intact across all 8 boots.
- **Evidence:** `~/glm52-test-artifacts/v20-window-evidence/v20-golive-FAILED.log` and
  `v20-golive-FAILED-inspect.json`. Container `ef83555fe17d…`, image `5f0c7b43daaa…`,
  StartedAt `01:10:36Z`.
- **Qualification suite never started.** No liveness, MTP, decode, needle, NVMe, or stress data.
- **v19 prod: DOWN by choice.** Rollback armed and unused — `glm52-prod-ring.yaml`, `ca8481…`,
  warm cache, ~12–15 min to serve.

## 7. Recommended next step

**Re-run this exact image with `VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS=1` and nothing else changed.**

The diagnostic build is already baked into this image (`cudagraph_utils.py` = `001365a0…`), so this
costs no rebuild — one env var, one boot. It would answer the only open question directly: whether
size 9 now passes, and which descriptor and stage fail instead. Boot 7 proved that instrumentation
localizes this in a single attempt.

Without it, the next fix would be aimed at an unidentified descriptor.

## 8. Note on the diagnostic in a production image

`cudagraph_utils.py` in this candidate is still the Boot 7 diagnostic revision. It is a no-op with
the env unset — which is how Boot 8 ran — but it is not the clean Boot 6 source (`c82a956c…`).
Before anything from this line goes to production for real, that file should be reverted or the
diagnostic accepted deliberately. Flagging now rather than at promotion time.
