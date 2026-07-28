# v20 acceptance window — running results (2026-07-22)

Operator: Fable · Baseline = v19 `i8_ring` prod (`gilded-gnosis-v19-int8-block-patched`, ca8481)
Candidate = v20 `gilded-gnosis-v20-vllm3e731bc-si1a88b38-int8-nvme-mtpfix`
(v20 decode base + PR#69 i8_ring + PR#165 NVMe eviction + PR#166 MTP indexer fix)

**Goal:** patched v20 in prod with **≥500k KV pool** (floor revised 2026-07-22 by Derek, down from
600k, accepting that v19's 644,864 was errant accounting whose tradeoffs we don't want) and
**similar-or-better** prefill/decode. Higher is better, but 500k is the acceptance bar.

## Summary table

| Metric | BEFORE (v19 i8_ring prod) | AFTER (v20 candidate) | Δ | Verdict |
|---|---|---|---|---|
| **KV pool (tokens)** | **644,864** (1.34× @480k) | **562,432** @ GMU 0.980/cap64/A2A16 | **−82,432** | ✅ **PASS** vs ≥500,000 floor (+12.5% margin) |
| **Prefill 8k** (tok/s, cold) | **1,607** | _pending Gate 3a_ | | floor ~1,365 (−15%) |
| **Prefill 50k** (tok/s, cold) | **1,641** | _pending Gate 3a_ | | floor ~1,395 (−15%) |
| **Decode ctx0 C1** (tok/s) | **63.2** | _pending Gate 3a_ | | floor ~53.7 |
| **Decode ctx0 C8** (tok/s) | **127.2** | _pending Gate 3a_ | | floor ~108.1 |
| **Decode ctx0 C16** (tok/s) | **165.1** | _pending Gate 3a_ | | floor ~140.3 |
| **Needle 300k** | PASS | _pending Gate 3b_ | | must PASS |
| **Needle 350k** | PASS | _pending Gate 3b_ | | must PASS |
| **Needle 475k** | PASS | _pending Gate 3b_ | | must PASS |
| **NVMe bounded eviction** | n/a (not enabled) | _pending Gate 2_ | | new capability |
| **NVMe persisted promotion** | n/a | _pending Gate 4_ | | new capability |
| **Stability (RestartCount)** | 0 over 2d16h | _pending_ | | must stay 0 |

## Baseline provenance
- Prefill/decode: v19 `i8_ring` prescribed LIL bench (45 s decode cells, standalone cold prefill).
  Decode C1 re-measured on a settled engine = 63.1 (consistent with 63.2).
- KV pool 644,864 confirmed on multiple clean v19 boots (`Available KV cache memory: 4.8 GiB`).
- Needles: v19 i8_ring 5/5 PASS at 50k/200k/300k/350k/475k.

## Gate log

### Gate 0 — preflight ✅ PASS (21:09Z)
- No users: 0 in-flight, 0 established conns, last inference 19:57:50Z (71 min prior)
- Baseline captured: container `70052ff5…`, image `ca8481…`, RC 0, health 200
- Compose sha256 `c6fca4b4…`
- **Issue found+fixed:** v19's 64 GB offload mmap survived teardown (root-owned) leaving only
  43.3 GB free — below v20's 64 GB tier need. Cleared with sudo → 107.3 GB free. This is the
  original EFAULT failure mode; verifying rather than trusting entrypoint cleanup prevented it.
- GPUs released clean (0 CUDA procs)

### Gate 1 — boot + integrity — ❌ FAILED (4 attempts)
| Boot | Config | Available KV | KV pool | Result |
|---|---|---|---|---|
| 1 | GMU 0.970 / MNS16 / cap64 | 3.25 GiB | ~435,968 est | ValueError: 480k needs 3.57 GiB |
| 2 | GMU 0.980 / MNS16 / cap64 | 4.14 GiB | **544,000** | CUDA illegal memory access @ graph capture |
| 3 | GMU 0.978 / MNS8 / cap32 | **4.41 GiB** | **592,640** | CUDA illegal memory access, same site |
| 4 | **+`982cda45`** GMU 0.980 / MNS16 / cap64 | 4.14 GiB | **555,520** | CUDA illegal memory access, same site |
| 5 | **A2A cap 16 + ag_rs** GMU 0.980 / MNS16 / cap64 | **4.19 GiB** | **562,432** | CUDA illegal memory access, same site |
| 6 | **PR#166@`c9a2e28d`** aligned block table 1876 | **4.13 GiB** | **554,496** | CUDA illegal memory access, same site |

| 7 | **CG_DIAG descriptor diagnostic** (`f0021cc3`) | 4.13 GiB | 554,496 | **FAULT LOCALIZED** — see below |

### Boot 7 — fault localized (2026-07-23 00:39:05Z)

```text
label=capturing_decode_cuda_graphs  stage=warmup_forward  cg_mode=FULL
num_tokens=9  num_reqs=9  uniform_token_count=1  max_req_tokens=None  num_active_loras=0
```

Decode **speculator** manager, **eager** forward (not captured), at **9 tokens** — the 8th of 16
descending sizes. Sizes 16→10 passed all five stages. The **identical descriptor passed every stage
during the profiling round**; production target manager and prefill speculator manager both
completed with zero FAILs. Only one distinct failing tuple in 1,162 CG_DIAG records.
Kills the large-batch framing: the fault is at the small end. Details:
`v20-boot7-cgdiag-result.md`.

**Six boots, one invariant fault.** Excluded so far: memory availability, MRV2 graph-pool lifetime,
graph-size/MNS tuning, B12X DCP A2A route + IPC staging, indexer block-table row-width alignment,
stale kernel cache. Every hypothesis was verified active in the logs; none moved the failure.
Details: `v20-boot6-aligned-block-table-result.md`, `v20-boot5-a2a-cap-result.md`,
`v20-mrv2fix-proofboot-result.md`.

Boot 4 (22:48:14Z→23:04:45Z) applied Sol's MRV2 pool-reuse fix at the exact previously-failed
profile. **The fix is provably active** — MRV2 tuple moved 1.08/0.72/0.36 → **1.08/0.66/0.42** and
the pool rose 544,000 → **555,520** (+11,520, above Sol's ~544k estimate). **The crash did not
change**, byte-identical stack, 2 s into the *production* speculator capture — while the *profiling*
speculator capture at 22:59:36 succeeded on the same pass.

**This falsifies the memory-shortfall theory.** Profiling and production now share one pool, so the
shortfall mechanism no longer exists; the error is `cudaErrorIllegalAddress`, never
`cudaErrorMemoryAllocation`; and boot 4 faulted with a *looser* budget than boot 2. Four boots,
three memory budgets, two graph caps, one accounting fix → same fault. **The fault is invariant to
every memory lever tried.** Full analysis: `v20-mrv2fix-proofboot-result.md`.

### Gate 2 — NVMe bounded fill/eviction
**NOT REACHED** — blocked by Gate 1. No data.

### Gate 3a — throughput (prefill + decode + MTP counters)
**NOT REACHED** — blocked by Gate 1. No data.

### Gate 3b — deep needles 300k/350k/475k
**NOT REACHED** — blocked by Gate 1. No data.

### Gate 3c — 16×50k concurrency/offload stress
**NOT REACHED** — blocked by Gate 1. No data.

### Gate 4 — restart + persisted NVMe promotion
**NOT REACHED** — blocked by Gate 1. No data.

## Decision tree (prod target)
| Outcome | Prod image | Compose |
|---|---|---|
| All gates pass **and pool ≥500k** | v20 mtpfix | `glm52-prod-v20.yaml` (64 GB DRAM + 64 GB NVMe) |
| Gate 2 or 4 fails (NVMe only) | v20 mtpfix | `glm52-prod-v20-nonvme.yaml` (`secondary_tiers: []`) |
| **Pool <500k** (revised floor) | v19 rollback | `glm52-prod-ring.yaml` |
| Gate 1 / 3b / 3c fails | v19 rollback | `glm52-prod-ring.yaml` |
| Window expires mid-test | v19 rollback | `glm52-prod-ring.yaml` |

Default is always v19 (2d16h zero-incident record). Prod must be up before users arrive.


## Floor revision (2026-07-22)

Derek revised the v20 acceptance floor **600k → 500k tokens**, accepting the investigation finding
that v19's 644,864 was partly errant accounting (unreserved CUDA-graph pool pages handed to KV) and
that forcing parity would require re-introducing the over-commitment v20 exists to prevent.

**Consequence: the KV-pool blocker is resolved.** Measured **544,000 @ GMU 0.980** clears the new
floor with ~9% headroom. No further memory work is required:
- graph-capture trimming is now **optional** (would buy ~0.3-0.4 GiB = ~40-55k tokens, or allow a
  safer GMU for the same pool)
- no GMU escalation beyond 0.980 needed

**Remaining blocker: the MTP CUDA-graph-capture crash** (`spec_decode/autoregressive` →
`decode_cudagraph_manager.capture()` → `make_dummy()`, illegal memory access, all 4 workers).
Owned by Sol. Operations holding until a fix + new sequence plan.

### GMU options under the 500k floor (for reference)
| GMU | Est. available KV | Est. pool | vs 500k floor |
|---|---|---|---|
| 0.970 | 3.25 GiB | ~436k | ✗ fails |
| 0.975 | ~3.72 GiB | ~499k | borderline |
| 0.978 | ~4.00 GiB | ~536k | ✓ passes |
| **0.980** | **4.14 GiB** | **544,000** (measured) | ✓ **passes, +9%** |
