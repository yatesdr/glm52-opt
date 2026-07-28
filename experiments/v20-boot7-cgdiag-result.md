# v20 Boot 7 — CUDA-graph descriptor diagnostic: LOCALIZED (Fable → Sol)

Date: 2026-07-23 · Boot 7 · Operator: Fable
Image: `…-int8-nvme-bt1876-cgdiag-f0021cc3`, `sha256:9de799c74fb3…`
Compose: `glm52-v20-tonight.yaml` sha256 `c18fda1019008371…`

**The diagnostic worked. One boot, exact localization.**

## 1. The first failing tuple

```text
CG_DIAG FAIL
  label   = capturing_decode_cuda_graphs      ← DECODE speculator manager
  stage   = warmup_forward                    ← eager forward, NOT captured
  cg_mode = CUDAGraphMode.FULL (2)
  num_tokens            = 9
  num_reqs              = 9
  uniform_token_count   = 1
  max_req_tokens        = None
  num_active_loras      = 0
```

All four workers, same tuple, 00:39:05Z. **Exactly one distinct failing tuple in the entire log**
(3 FAIL records, all identical modulo rank).

The diagnostic's own `torch.accelerator.synchronize()` at the `warmup_forward` boundary is what
raised it — i.e. the illegal access is launched by that stage's work, not inherited from earlier.

## 2. The A/B you asked for (item 5)

Two full rounds ran. Chronology separates them cleanly:

| Round | Time | Managers |
|---|---|---|
| **Profiling** | 00:35:34 → 00:36:56 | `profiling_cuda_graph_memory` → prefill spec → decode spec → MRV2/KV printed |
| **Production** | 00:37:42 → 00:39:05 | `capturing_cuda_graphs` (target) → prefill spec → decode spec → **FAIL** |

**The identical descriptor `num_tokens=9, num_reqs=9, uniform_token_count=1` passed every stage
during profiling:**

```text
PROFILING  decode-spec, tok=9:  warmup_inputs PASS → warmup_forward PASS
                                → fresh_capture_inputs PASS → b12x_prewarm PASS
                                → full_capture PASS
PRODUCTION decode-spec, tok=9:  warmup_inputs PASS → warmup_forward FAIL
```

Profiling captured **all 16 decode sizes (1…16)** without a single failure.

## 3. Production sequence — descending, clean until exactly 9

TP0, production decode-speculator manager. Stage order per size is
`warmup_inputs → warmup_forward → fresh_capture_inputs → b12x_prewarm → full_capture`:

| tok | inputs | forward | fresh_inputs | b12x_prewarm | full_capture |
|---|---|---|---|---|---|
| 16 | PASS | PASS | PASS | PASS | PASS |
| 15 | PASS | PASS | PASS | PASS | PASS |
| 14 | PASS | PASS | PASS | PASS | PASS |
| 13 | PASS | PASS | PASS | PASS | PASS |
| 12 | PASS | PASS | PASS | PASS | PASS |
| 11 | PASS | PASS | PASS | PASS | PASS |
| 10 | PASS | PASS | PASS | PASS | PASS |
| **9** | **PASS** | **FAIL** | — | — | — |

Progress bar corroborates: `Capturing decode CUDA graphs (FULL): 3/16` — it died on the 8th of 16.

**Zero FAILs in the production target manager (`capturing_cuda_graphs`) and zero in the production
prefill speculator manager.** Both completed fully before the decode manager was entered.

## 4. What this rules in and out

Mapping onto your interpretation table, this is a case your table doesn't quite cover — it is
**not** the target manager and **not** input preparation:

- **Not the target forward.** All production target stages passed, eager and captured.
- **Not deferred target teardown.** The prefill speculator manager ran entirely between the target
  manager and the failure, clean.
- **Not speculator dummy-input preparation.** `warmup_inputs` for tok=9 **passed** with a
  synchronize immediately after it.
- **Not size 32/64.** The failing graph is **9 tokens**, the 8th of 16 — small. Every earlier
  hypothesis about large-batch behaviour (A2A cap, graph cap 64, MNS) was aimed at the wrong end of
  the size range.
- **Not a capture-mode problem.** The failure is in the **eager** `warmup_forward`, outside any
  `torch.cuda.graph(...)` block.

**The fault is a real illegal access launched by the decode speculator's eager forward at
`num_tokens=9, num_reqs=9, uniform_token_count=1`, in the production round only, after the same
descriptor succeeded in the profiling round.**

## 5. Operator observation on why 9 (hypothesis, not a finding)

Offered only as a lead: with DCP world size 4, 9 = 2·4+1 — the first size below 10 where a
per-rank split leaves a remainder of 1 with **three** ranks holding 2 and one holding 3, and the
first descending size where `ceil(9/4)=3` while `floor(9/4)=2`. Sizes 16…10 all passed. If a
per-rank shard length or a padded index is computed from a floor/ceil pair somewhere in the decode
speculator or DCP gather path, 9 is a plausible first crossing. **I have not verified this** — the
size-boundary arithmetic is yours to confirm against the source.

The profiling-vs-production difference for the identical descriptor is the other half: whatever
state exists in production but not profiling (KV cache allocated and bound, real cache shapes,
target graphs already captured into the shared pool) is what turns the same shape lethal.

## 6. Gate 0 verification (for the record)

| Pin | SHA-256 | |
|---|---|---|
| PR#69 `pcie_dma.py` / `.cu` | `5a6e6a0e…` / `70f4be32…` | unchanged |
| PR#165 `manager.py` | `653edbf4…` | unchanged |
| PR#166 `indexer.py` | `d419af9e…` | unchanged |
| MRV2 `model_runner.py` | `2eab8362…` | unchanged |
| **`cudagraph_utils.py`** | **`001365a061da3e5aeaa8bd3e7f04a41eac7b58de9020afbf4ae2d29be5b6a3f3`** | **matches your output pin** |

Chain: patch `1fdb1ceb…` (matches your pin + manifest) → input extracted from the Boot 6 image
`c82a956c…` (**matches your expected input pin**) → clean apply → `001365a0…`.

Exactly one env added: `VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS=1` (verified live).
**`CUDA_LAUNCH_BLOCKING` absent — 0 occurrences in the compose**, as instructed.
Runtime profile unchanged from Boot 6 in every respect.

Memory for continuity: MRV2 `1.09 / 0.69 / 0.40`, Available KV 4.13 GiB, **pool 554,496**,
1.16× @480k — identical to Boot 6, confirming the diagnostic did not perturb allocation.

## 7. State

- **Stopped, no retry, no ladder.** CN3 clear: 0 containers, 0 CUDA procs, `/dev/shm` 100 G free.
- **NVMe acceptance namespace still pristine (0 entries)** across all seven boots.
- **Evidence:** `~/glm52-test-artifacts/v20-window-evidence/v20-boot7-cgdiag.log` (611 KB,
  **1,162 CG_DIAG records**, 3 FAIL) and `v20-boot7-cgdiag-inspect.json`.
  Container `166974596e08…`, image `9de799c74fb3…`, StartedAt `2026-07-23T00:25:28Z`.
- **Note on RestartCount:** the container auto-restarted once under `restart: unless-stopped`
  before I tore it down (RC went 0 → 1 at 00:39:36Z). **The evidence files were captured from the
  first, failing run before the restart** — the log contains the complete first-failure sequence.
  The restart is an artifact of my teardown timing, not a second fault.
- **v19 prod: DOWN by choice.** Rollback armed and unused.

## 8. Recommended next step

Per your own guidance, `CUDA_LAUNCH_BLOCKING=1` is now the indicated follow-up **only if** you
cannot identify the operation within `warmup_forward` from source. We now have the descriptor and
the stage, which was the precondition you set. One targeted boot, ~20 min with the diagnostic
slowdown, and it would name the kernel.

Standing by — I have not run it and will not without your instruction.
