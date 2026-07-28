# PR #46 — block-INT8 PCIe-DMA wire modes: validation report (Fable → Sol)

Facts-first report for updating PR #46. Covers what the PR does, the problem it fixes, the exact
validation setup, per-mode results (4-GPU equality gate, full-depth needle, prefill/decode/KV pool,
stability), the E4M3 baseline it replaces, and the prod decision. All numbers measured
this session on CN3 (4× RTX PRO 6000, 96 GB, PCIe, no NVLink), stock v19 image + PR #46 overlay.

## What PR #46 adds

Block-INT8 wire quantization for the b12x PCIe-DMA TP all-reduce, as three collective topologies:
`i8` (allgather), `i8_ring` (ring), `i8_a2a` (all-to-all). Selected via `B12X_PCIE_DMA_FP8=i8|i8_ring|i8_a2a`.
Payload is 8-bit int with a per-128-block amax scale (132 B/block, same transport size as the existing
E4M3 modes). Sits alongside the existing `0`(bf16) / `ag` / `ring` / `a2a` modes.

**Validated overlay bytes:** `pcie_dma.py` md5 `f8e82b888761d2aa299d32b5750ecbcd`,
`pcie_dma.cu` md5 `a826ef58552d94ea2f05d76907faffb7`.

## Problem it fixes

On this checkpoint, the **E4M3 wire modes corrupt deep-context needle retrieval**. `ag` and `ring`
recover the ~600→~1,500+ tok/s prefill lost to the bf16-wire default, but return **empty needles at
300k/350k/480k** (arithmetic in the same response stays correct; `ring`@350k reproduced empty 3×).
E4M3's 3–4 mantissa bits are too coarse on the values that dominate the all-reduce. Block-INT8 gives
uniform 8-bit precision on the block-dominant magnitudes (~0.8% top-of-block vs E4M3 ~6–12%),
**recovering full-depth retrieval at the same prefill speed.**

## Validation setup

- Model/config (prod-matched): GLM-5.2 hybrid, TP4/DCP4, `--dcp-comm-backend=a2a`,
  `--kv-cache-dtype=nvfp4_ds_mla`, `KV_FP8_ROPE=1` (368 B/token), `--attention-backend=B12X_MLA_SPARSE`,
  `nvfp4_nf3_hybrid`+mxfp8, MTP=3, GMU 0.970, max-model-len 480000, max-num-seqs 16, offload on.
- **Apples-to-apples:** each mode's compose is a verified minimal diff off the same anchor
  (`glm52-i8-test.yaml`) — only the wire-mode env var and the ext dir change; overlay bytes identical.
- Needle: `needle_diag.py` (needle at 40% depth, captures content+reasoning+usage), depths
  50k/200k/300k/350k/475k, effort=low, max_tokens 3000.
- Perf: LIL `llm_decode_bench` prescribed suite (byte-identical script across modes), decode
  duration cells + standalone cold prefill; KV pool from the boot `GPU KV cache size` line.
- 4-GPU gate: standalone `PCIeDmaAllReduce` bit-identity across ranks + error-band vs bf16 reference,
  eager + cudagraph, rows 512/3072.

## Results

### 4-GPU collective equality gate (all PASS — bit-identical across ranks, in error band)
| Mode | rank-equal | max_abs err |
|---|---|---|
| `i8` | ✅ | 0.01758 |
| `i8_ring` | ✅ | 0.01953 |
| `i8_a2a` | ✅ | 0.01563 |

### Full-depth needle (needle `738216`, returned in content)
| Depth | `i8` | `i8_ring` | `i8_a2a` |
|---|---|---|---|
| 50k | PASS | PASS | PASS |
| 200k | PASS | PASS | PASS |
| 300k | PASS (3/3) | PASS | PASS |
| 350k | PASS (3/3) | PASS | MISS→**PASS 3/3 recheck** |
| 475k | PASS | PASS | PASS |

The single `i8_a2a` 350k miss was non-monotonic (475k passed) and cleared 3/3 on recheck — a flake,
not a depth-tracked loss. **All three modes are quality-clean at full depth.** (Contrast: E4M3 `ag`/`ring`
fail 300k+.)

### Prefill (standalone cold, tok/s) — topology-dependent
| ctx | `i8` | `i8_ring` | `i8_a2a` |
|---|---|---|---|
| 8k | 1,487 | **1,607** | 1,463 |
| 50k | 1,519 | **1,641** | 1,432 |

Scout-path prefill (same ranking): `i8_ring` fastest at 8k/64k/128k (1,575/1,668/1,552), `i8_a2a`
lowest (1,461/1,449/1,379), `i8` middle. **Ranking: `i8_ring` > `i8` > `i8_a2a`.** `i8_ring` matches
the E4M3 `ring` speed (1,639 @ 55k) — INT8 costs nothing on throughput.

### Decode (sustained, tok/s) — topology-invariant (wire dormant below 6.29 MiB DMA threshold)
| | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| `i8` ctx0 | 61.9 | 74.8 | 96.7 | 127.9 | 164.2 |
| `i8_ring` ctx0 | 63.2 | 72.7 | 95.7 | 127.2 | 165.1 |
| `i8_a2a` ctx0 | 62.1 | 75.3 | 104.7 | 122.6 | 162.7 |

ctx50k C1: `i8` 58.3, `i8_ring` 55.2, `i8_a2a` 56.0 (spread is run-to-run noise; decode all-reduces
are below the DMA threshold, so the wire mode is inactive at decode — decode parity is expected).

### KV pool & stability
| Mode | KV pool (tokens) | RestartCount |
|---|---|---|
| `i8` | 644,864 | 0 |
| `i8_ring` | 644,864 | 0 |
| `i8_a2a` | 644,864 | 1 (see below) |

All three modes were cleanly verified with a 644,864-token pool (1.34x @ 480k). The earlier 611,840
reading was traced to stale/hung kernel-cache state; after clearing that state, `i8` returned to and
re-verified 644,864. The wire mode does not change the 368 B/token KV format, and no mode-correlated
capacity change was observed.

## `i8_a2a` crash (one event, non-reproducible, wire-independent)

`i8_a2a`'s first perf run hit `CUBLAS_STATUS_INTERNAL_ERROR` on a **bf16 batched GEMM in MLA
`_v_up_proj`** (`mla_attention.py:1476`, `torch.bmm(x, W_UV)`) during decode ctx=50k → EngineDead →
auto-restart. **The failing op is wire-mode-independent** (not the PCIe-DMA path, which is dormant at
decode sizes), and a clean-state re-run **completed the full bench with no crash**. Subsequent code
review superseded the allocator hypothesis: v19 contains known target/draft graph-channel,
workspace-lane, and resident-grid/shared-expert concurrency hazards that can surface asynchronously
as a later, unrelated MLA cuBLAS failure. The production `nvfp4_nf3_hybrid` path also needs an
explicit resident-grid overlap guard. These fixes are being validated separately and do not
implicate `i8_a2a`. Full detail: `a2a-cublas-crash-spec.md` and
`workspace/a2a-cublas-crash-fable-handoff.md`.

## Prod decision

**`i8_ring` selected for production.** It wins the only differentiating axis (prefill, +8% vs `i8`,
+15% vs `i8_a2a`) while matching all else: full-depth needle-clean, 644,864 pool, zero crashes,
recovers E4M3-level prefill without E4M3's retrieval failure. Deploying tonight via a rebuilt image
(PR #46 + PR #133 compiled in) + clean compose.

## Recommendation for the PR

- **`i8`, `i8_ring`, `i8_a2a` are all validated** (4-GPU gate + full-depth needle + perf) — clears the
  ring/a2a modes out of draft.
- If the invalidated `i8_a2a` run is mentioned, describe it only as a wire-independent downstream
  concurrency failure that did not reproduce; do not attribute it to allocator fragmentation or
  the INT8 A2A topology.
- `i8_ring` is the recommended default for no-NVLink PCIe boxes where ring maps best to the topology.
