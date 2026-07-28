# v19 Tier-1 reliability candidate — acceptance criteria

Candidate: `glm52-serve:v19-reliability-r2-20260726`
(also tagged `ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v19-reliability-r2-20260726`)
Built on cn3, 2026-07-26. Base `sha256:ca8481687f71…` (`gilded-gnosis-v19-int8-block-patched`).
Compose: `deploy/glm52-v19-reliability-20260726.yaml`
Suite: `harness/run_v19_reliability_acceptance.sh`

## What changed, and therefore what we are testing for

Seven pure-Python files (6 vLLM + 1 b12x). b12x version pin 0.30.2 unchanged, no CUDA or wheel
rebuild, baked INT8 extension `a826ef58` untouched, every `vllm serve` flag and every container
env var identical to prod.

**The shared-experts aux stream stays ON.** It is worth ~11% decode and the fix makes it safe
rather than disabling it:

- `b12x/moe/fused/w4a16/kernel.py` — `W4A16FusedMoeKernel` and `W4A16FusedMoeHybridKernel`
  now launch with `cooperative=True`. Their `_grid_barrier` is a spin-wait sense-reversal
  barrier over `grid_x` CTAs; a normal launch can admit only part of the grid while
  shared-expert GEMMs hold the remaining SMs, and the admitted CTAs then spin forever on a
  peer that will never be scheduled. `grid_x` is *already* sized `<= sms*blocks_per_sm`
  (`_fused_grid_x`: "staying <= the cap so the cooperative barrier never deadlocks"), so the
  code already assumed co-residency — it just never asked CUDA to guarantee it.
  `b12x/moe/fused/dynamic.py` already does exactly this for the Grid188/unified path; w4a16,
  which is what we run under `B12X_MOE_FORCE_A16=1`, had been missed.

One behaviour *is* expected to change and must not be read as a regression:

- **The KV pool shrinks slightly.** The MTP draft now gets its own workspace lane instead of
  aliasing the target's. That second buffer is sized during profiling and comes out of the pool.

Everything else — prefill, decode, retrieval quality, tool calling — should be **unchanged**.
First boot is slower: the changed b12x kernel invalidates its CuTe-DSL JIT entries and
recompiles once into `/cache/jit`.

---

## Run modes

| Mode | Phases | Time | Use |
|---|---|---|---|
| `--quick` | 0–2 | ~5 min | Confirm the right image booted and serves. Run this first, always. |
| `--full` | 0–6 | ~3–4 h | The real acceptance run. |
| `--stress-only` | 0, 5, 6 | ~1.5 h | Re-run just the wedge discriminator. |

```sh
bash harness/run_v19_reliability_acceptance.sh --quick
bash harness/run_v19_reliability_acceptance.sh --full
```

---

## Gate table

### Phase 0 — identity (blocking; the suite aborts if any row fails)

| # | Criterion | Pass |
|---|---|---|
| 0.1 | Container running, `RestartCount == 0` | exact |
| 0.2 | All 7 backported files match the recorded sha256 (6 vLLM + 1 b12x) | exact, all 7 |
| 0.3 | All vLLM patch markers present | exact |
| 0.4 | w4a16 has **2** `cooperative=True` launches; `dynamic.py` still has its pre-existing one | exact |
| 0.5 | `_v_up_proj` contains the `force_contiguous_mla_bmm_output` branch | exact |
| 0.6 | Backend still declares all 3 `force_contiguous_mla_bmm_*` flags | exact (3) |
| 0.7 | `VLLM_PCIE_ONESHOT_SINGLE_CHANNEL` default still `1` | exact — channel isolation deliberately not backported |
| 0.8 | `b12x` version pin `0.30.2` (one file patched, no wheel rebuild) | exact |
| 0.9 | No reference anywhere in `vllm/` to a b12x symbol absent from 0.30.2 | zero hits |
| 0.10 | `TORCH_EXTENSIONS_DIR == /cache/int8ext_baked_a826ef58` | exact |
| 0.11 | `B12X_PCIE_DMA_FP8 == i8_ring` | exact |
| 0.12 | vLLM build string still `…gilded.gnosis.v19.vllm7ea567a.b12x4cfa530…` | substring |
| 0.13 | Capability gate ABSENT from `vllm/` and `VLLM_DISABLE_SHARED_EXPERTS_STREAM` unset/0 | exact — overlap must stay ON |

Rationale for 0.7/0.9: the PCIe target/draft channel-isolation fix (`4781731c`) and #131 both
need b12x APIs that 0.30.2 does not have. If either ever shows up in this image, the boot will
`AttributeError` at graph capture. These rows are the tripwire.

### Phase 1 — boot health

| # | Criterion | Pass | Notes |
|---|---|---|---|
| 1.1 | Cold boot reaches healthy | < 45 min | prod start_period is 2600 s. **First boot recompiles the changed b12x CuTe kernels** — expect longer, still inside the window |
| 1.2 | Healthcheck status `healthy` | exact | this is the new deep probe — validated at 0.67 s against the live engine |
| 1.3 | `GPU KV cache size` | **≥ 600,000 tokens** | prod baseline **644,864**. Record the exact number. |
| 1.4 | Boot log free of `CUBLAS_STATUS`, `illegal memory access`, `Traceback`, `EngineDeadError` | zero | |

If 1.3 lands below 600,000, do **not** raise GMU to compensate. Record it and decide separately —
GMU 0.970 is the stable ceiling with offload on, and 0.975 OOMs on the first real prefill.

### Phase 2 — functional

| # | Criterion | Pass |
|---|---|---|
| 2.1 | Chat completion returns the requested token | exact |
| 2.2 | `reasoning_content` field present (glm45 parser) | present or justified skip |
| 2.3 | Tool call emits `get_weather` (glm47 parser) | exact |
| 2.4 | `usage.prompt_tokens_details` present | exact |

### Phase 3 — retrieval (the quality gate that actually matters)

Uses `needle_hunt.py` (Sol's, retrieval-vs-finalization split). **Not** `quality_gate.py` —
per `MEASUREMENT-LIBRARY.md` it scores by substring and passes degenerate repetition.

| # | Criterion | Pass |
|---|---|---|
| 3.1 | Needle retrieved at 50k, 150k, 250k, 350k, 475k | **5/5**, no regression vs v19 |
| 3.2 | No degenerate repetition in any answer | manual read of the raw answers |
| 3.3 | Any empty answer explained by `finish_reason`/`completion_tokens` before being called a miss | reviewer judgement |

v19 is the reference here: it passes the full ladder today, and deep retrieval is precisely what
the fp8 wire modes broke. **Any retrieval regression is an automatic reject** — this backport
touches the MLA output projection, so retrieval is the highest-risk surface.

### Phase 4 — performance

Baselines below are the documented v19 figures. Prefer a same-host control run
(`--quick` on the *current* prod image) when the box allows it, since cn3 and cn4 differ
(cn4's switch links train x8, cn3 x16).

| Metric | v19 baseline | Accept | Source |
|---|---|---|---|
| Cold prefill 8k | 1,463 tok/s | ≥ 97% (≥ 1,419) | `a2a-cublas-crash-spec.md` run 2 |
| Cold prefill 50k | 1,432 tok/s | ≥ 97% (≥ 1,389) | same |
| Prefill (i8_ring) | ~1,640 tok/s | ≥ 97% | prod compose note |
| Decode ctx0 C1 | 62.1 tok/s | ≥ 97% (≥ 60.2) | run 2 |
| Decode ctx0 C16 | 162.7 tok/s | ≥ 97% (≥ 157.8) | run 2 |
| Decode ctx50k C1 | 56.0 tok/s | ≥ 97% (≥ 54.3) | run 2 |
| Decode ctx50k C8 | 121.4 tok/s | ≥ 97% (≥ 117.8) | run 2 |

**No throughput loss is expected anywhere**, because the shared-experts overlap is retained.
The two things a shortfall would point at:

- **Decode below 97%** → the cooperative launch is costing something. Most likely the grid no
  longer fits co-residency at some shape and CUDA is serializing the launch. Check
  `_fused_grid_x` against `sms * blocks_per_sm` for the failing shape. This would be a real
  finding, not a tuning knob.
- **Prefill below 97%** → the staging copy added in `_v_up_proj` by `#136`
  (a `(16, B, 256)` bf16 temp; ~24 MiB transient at prefill B=3072).

**If the cooperative launch itself fails** (a hard error rather than a slowdown), the fallback is
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` in the compose — no rebuild, costs the ~11% decode, and
removes the concurrency entirely. That is the emergency lever, not the plan.

### Phase 5 — wedge reproduction (the discriminator)

This is the whole point of the exercise. The sequence is the one that produced the
2026-07-24 failure, taken verbatim from `a2a-cublas-crash-spec.md`:

```text
deep needle sweep to 475k
  -> 3x 350k recheck prefills
    -> decode ctx=50k, concurrency 1,2,4,8
       with NO restart anywhere in between
```

Run 1 of that sequence crashed with `CUBLAS_STATUS_INTERNAL_ERROR` in `_v_up_proj`.
Run 2, from a clean arena without the prefill barrage, did not.

| # | Criterion | Pass |
|---|---|---|
| 5.1 | `RestartCount` unchanged across the whole sequence | exact |
| 5.2 | No `CUBLAS_STATUS_INTERNAL_ERROR` in the engine log | zero |
| 5.3 | No `illegal memory access` | zero |
| 5.4 | No `EngineDeadError` / `sample_tokens timed out` | zero |
| 5.5 | No HTTP 500 from any request in the sequence | zero |
| 5.6 | No MoE grid-barrier hang: no request stalls with GPUs pinned at ~100% and zero token output | zero |
| 5.7 | No `too many blocks in cooperative launch` / `cudaErrorCooperativeLaunchTooLarge` | zero |

5.6 is the signature of the hazard this build's b12x change targets: a spin-wait `_grid_barrier`
deadlock shows as busy GPUs with no forward progress, not as a clean exception. 5.7 is the
failure mode of the fix itself — it would mean some shape produces `grid_x > sms*blocks_per_sm`,
contradicting `_fused_grid_x`. Either one is a reject.

A clean pass here is **suggestive, not conclusive** — the original fault was itself
non-reproducible on the second attempt. What makes it meaningful is the mechanism: the
backend declared a contiguity contract, v19 ignored it, and the ignored call is the exact
frame that faulted. Phase 5 is the behavioural check on top of that structural argument.

Strongest possible evidence, if a box can be spared: run Phase 5 against the **unpatched**
v19 image first. If the wedge reproduces there and not on the candidate, that closes the loop.

### Phase 6 — integrity

| # | Criterion | Pass |
|---|---|---|
| 6.1 | `RestartCount` unchanged across the entire suite | exact |
| 6.2 | PCIe `[12] Completion Timeout` in dmesg | 0 under serving load |
| 6.3 | `/health` returns 200 at the end | exact |

On 6.2: a burst during graph capture (peak P2P) was observed on cn3 after the 2026-07-24 reboot
and was benign. Timeouts under *real serving load* are the failure signal. cn4 trains x8 and is
noisier than cn3 here — judge cn4 numbers against cn4 history, not cn3's.

---

## Overall decision rule

**Accept for prod promotion** when:

- Phase 0 clean (blocking), and
- Phases 1–3 clean, with retrieval **5/5** and no degenerate output, and
- Phase 4 within the bands above, and
- Phase 5 clean, and
- Phase 6 clean.

**Accept with a follow-up** if everything passes except a decode shortfall between 90% and 97%.
Ship it — the wedge cost a host reboot and ~80 minutes of prod downtime, which outweighs a few
percent of decode — and open the cooperative-launch grid-sizing question separately.

**Reject** on any retrieval regression, any Phase 5 fault, or a KV pool below 600,000 tokens.

**Fall back, don't reject,** if Phase 5 shows a residual MoE grid-barrier hang (5.6) or the
cooperative launch errors (5.7): set `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` in the compose and
re-run. That trades the ~11% decode for removing the concurrency entirely, needs no rebuild, and
still keeps the three vLLM structural fixes — which address a *different* mechanism (the OOB
strided BMM) and are independent of the MoE path.

### Known remaining gap

`b12x/moe/fused/micro.py` also has a `_resident_grid_barrier` and a non-cooperative launch. It is
the native-NVFP4 micro-decode path, reachable only when `quant_mode ∈ {nvfp4, w4a8_nvfp4}`; prod
runs `B12X_MOE_FORCE_A16=1` → `w4a16`, so it is not reachable in this configuration. **Fix it
before changing `B12X_MOE_FORCE_A16`.**

## Promotion steps once accepted

Four edits to `deploy/glm52-v19-reliability-20260726.yaml`, listed in its own header:
`restart: unless-stopped`, re-add `labels: [autoheal=true]`, rename the container to
`glm52-prod`, and bring back the autoheal service. Keep the deep healthcheck — it is the
durable fix for the 500-storm detection gap, and by promotion time it will have been observed
for a full acceptance run before it is ever allowed to trigger autoheal.

## Rollback

The prod image is untouched and still present on cn3
(`ghcr.io/yatesdr/glm52-serve@sha256:ca8481687f71…`). Rollback is: stop the candidate stack,
`docker compose -f /home/claude/glm52-prod-ring.yaml up -d`. No data migration, no cache
invalidation — the candidate shares `TORCH_EXTENSIONS_DIR` and the JIT/AOT cache with prod
because the compiled artifacts are identical.
