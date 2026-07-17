# Complete test results

All measurements on: 4x RTX PRO 6000 Blackwell Workstation 96 GB (SM120),
PCIe **Gen3** x16 host with two PLX switch pairs (GPU0-1 / GPU2-3), no
NVLink, cross-root-complex P2P ~2–6 GB/s, intra-pair ~9.5–11 GB/s.
Model: GLM-5.2 753B, MXFP8/NVFP4/NF3 hybrid checkpoint, 78 layers,
TP4 + DCP4 (except where noted), `nvfp4_ds_mla` KV, MTP-3.
"Cold" = unique random first prompt block every run (prefix-cache-proof);
server-side tok/s from `/metrics` deltas.

## 1. Prefill progression (55k-token cold prefill)

| Config | 8k | 55k | Quality gates |
|---|---:|---:|---|
| Stock v1.3, `B12X_PCIE_DMA_FP8=ag` | — | ~640 | pass |
| v1.3 + `B12X_PCIE_DMA_FP8=ring` | — | 685 | pass |
| v1.3 + fp8 DCP gather + fp8-ring RS (stage 1) | 932 | 955 | pass (incl. deep-needle@95%, JSON echo) |
| v1.4-equiv + stage 1 | 935 | 964 | pass |
| v1.4-equiv + stage 1, parity check (CKV patch mounted, disabled) | 959 | 966 | pass — proves patch is inert when off |
| **v1.4-equiv + packed-CKV (stage 3)** | **1,506** | **1,696** | **pass** |
| DCP1 (no context parallelism; ¼ KV capacity) | 1,621 | 1,879 | not gated (physics reference) |

Notes:
- Test profile for the 932–1,696 rows: `MAXLEN=64000 BLOCKS=400
  MNBT=3072` (102,400-token pool). Stage-1 numbers were additionally
  validated at full 480k/599k-pool boots up to first-prefill (see §5).
- The DCP1 row used `BLOCKS=1200` (76,800-token pool — DCP1 keeps full KV
  on every rank).

## 2. Decode (single-user C1, duration-mode, ignore_eos, same bench tool throughout)

| Config | ctx 0 | ctx 16k | ctx 32k |
|---|---:|---:|---:|
| v1.3 baseline (window-1 record) | 51.7 | — | 51.2 |
| v1.4-equiv (+ stage-1 prefill patches) | 66.5 | 64.2 | 63.3 |
| v1.4-equiv + packed-CKV active | 67.4 | 66.1 | 65.0 |

- v1.4's decode gain (+29%) comes from its heterogeneous W4A16 decode
  kernel + MTP/DCP sync fixes; our prefill patches are decode-neutral
  (identical numbers within noise, by design — decode rides the ≤16-row
  a2a path and small oneshot allreduces that none of our patches touch).
- MTP acceptance observed 0.39–0.61 on prose (content-dependent).

## 3. Per-layer phase profiler ledgers (3072-token chunks, 55k prefill, all 4 ranks within 2%)

Query-transport stack (stage 1):

| Phase | ms/layer/chunk | share of wall |
|---|---:|---:|
| DCP query all-gather (fp8) | 13.9–14.2 | ~34% |
| DCP output reduce-scatter (fp8 ring) | 4.0–4.1 | ~10% |
| project-before-merge | 0.48 | ~1% |
| everything else (attn, MoE, TP allreduces) | ~22.6 | ~55% |

Packed-CKV transport (stage 3) — same chunk geometry:

| Phase | ms/layer/chunk |
|---|---:|
| ckv_pack | 0.26 |
| ckv_ag (byte all-gather) | 0.53–0.83 |
| ckv_remap | 0.32–0.44 |
| ckv_stage (head-major staging copy) | 0.05 |
| **total CKV transport** | **~1.4** (was 18.6) |

Signature counters at acceptance: routes ckv=1200 query=0; gather/rs/proj
n=0; missing_blocks=0; local_heads=16; mean wire bytes/rank/chunk 4.18 MB
(query transport moved ~54+ MB for the same work).

DCP1 physics interpretation: 964 → 1,879 (1.95x) when DCP is removed
entirely means total DCP cost ≈ 49% of wall — more than the 45% the
comm-only profile attributed. The CKV result (1,696) lands within ~10% of
that no-DCP ceiling; the residual gap is a named profiler target, not
generic overhead.

## 4. Quality gates (every "pass" above = all of these green)

1. needle retrieval @55k tokens (benign phrasing), exact match
2. multi-step arithmetic, exact integer
3. long-generation coherence (length + trigram-repetition bound)
4. deep needle @95% depth of 60k, exact match
5. JSON echo: nested numeric payload byte-exact round trip
(60k-depth variants at the test profile; 200k deep variant reserved for
full-context acceptance.)

CPU-level proofs (no GPU): ownership-inversion equivalence vs baseline
LSE-merge attention — max output error 2.98e-08, max LSE error 2.38e-07
(fp32-emulated arithmetic); byte-exact ring collective schedule incl.
slot-reuse; remap/read equivalence incl. holes, tails, aliasing. See
`harness/`.

## 5. Full-context (480k) memory campaign — the failure ledger

These failures are as valuable as the wins. All at 480k max-len,
v1.4-equiv + stage-1 patches, chasing a first prefill that survives.

| Boot | Config | Outcome |
|---|---|---|
| A | BLOCKS=2380 (609,280-token pool) | RS-ring slab OOM on rank 0 only (50 MiB short) → rank-divergent fallback → **cross-rank collective mismatch → deadlock** (fixed: group-vote init, see §6) |
| B | BLOCKS=2380 + collective-safe vote | clean group fallback (fix verified live), then 36 MiB MoE transient OOM, 10.7 MiB free |
| C | BLOCKS=2340 | ring vote PASSED all 4 ranks (freed blocks DID become slab-usable on v1.4); 24 MiB transient missed by 1.3 MiB |
| D | BLOCKS=2300 | 36 MiB transient missed by 11 MiB — **marginal return of block cuts ≈ 2 MiB device-free per 40 blocks** |
| E | BLOCKS=2340, RS ring disabled (~194 MiB "freed" on paper) | same 36 MiB transient missed by 5 MiB; **PyTorch allocation GREW ~210 MiB** vs boot A |

Memory model distilled (the load-bearing findings):

1. **Slabs vs transient float.** Large single allocations (a 144 MiB ring
   slab) get served; small runtime transients (24–36 MiB) starve. They
   draw from different pools: freed KV blocks return as slab-usable
   memory but NOT as transient float.
2. **Context-scaled absorption.** At 480k max-len, workspace/plan buffers
   scale with context and absorb what block cuts free — accounting
   equivalences are not fit proofs. Only same-phase `cudaMemGetInfo`
   counts.
3. Consequence: the phase-2 design allocates ZERO new permanent residents
   (NCCL gather into existing workspaces) and proves headroom with a
   192 MiB direct-cudaMalloc escrow + dual ≥150 MiB group-min probes
   instead of arithmetic. See `design/packed-ckv-phase2-design.md`.

## 6. Collective-safety findings (both found the hard way)

1. **Per-rank fallback on collective-resource init is a deadlock, not a
   fallback.** If one rank fails to build a custom collective and falls
   back to NCCL while its peers enter the custom ring, the job hangs at
   the watchdog. Fix pattern (now in both patch sets): local init →
   1-element MIN all-reduce vote over the group → all-or-nothing adoption;
   losers close and discard. Any runtime routing decision must be provably
   rank-invariant (derivable from shape/dtype/config only).
2. **Keying a communicator by payload size builds one slab per distinct
   tail-chunk length.** Key by capacity ceiling; let smaller payloads ride
   the same instance.

## 7. Other findings

- **Triton kernels can't read module globals** (only `tl.constexpr`
  parameters) — and NO static check catches it before first GPU compile:
  not pyflakes, not ast, not imports, not CPU harnesses. Dry-compile new
  kernels before boot. (`design/field-fix1-triton-constexpr.md`)
- **DCP posture bands** (block 64): DCP1 caps capacity at pool/4 but
  removes ~49% of prefill wall — it is a legitimate short-context profile
  (≤64k) at 1,621–1,879 tok/s. DCP4 is the only 480k-capable posture.
  DCP2 is unmeasured; do not interpolate.
- **Cold vs cache-effective throughput**: prefix caching + KV offload can
  make warm "prefill" numbers arbitrary. Any number worth trusting comes
  with prefix-cache metric deltas. Our bench seeds a random first block
  every run; the chained block hashes make the entire prompt unique.
- 8k-class community prefill claims we could verify were TP8 (8-GPU) or
  cache-warm numbers — see `design/breakthrough-analysis.md` for the
  evidence audit.
