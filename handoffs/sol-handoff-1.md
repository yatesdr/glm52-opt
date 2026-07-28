# Sol Handoff #1 — v20 memory-reclaim candidate: deep-context degeneration + low throughput

**Date:** 2026-07-24
**Operator:** Fable (CN4)
**Status:** Acceptance boot PASSED; **qualification FAILED** — degenerate generation at ≥150k context. Ladder stopped at first failures per fail-policy, evidence preserved.

---

## TL;DR for Sol

The memory-reclaim image **boots clean and is hardware-stable**, but **regresses deep-context coherence vs the prior v20 build**:

- **50k needle: PASS** (correct `738216`, non-empty content).
- **150k needle: DEGENERATE** — 2 tokens (`"The"`), empty content, MISS.
- **250k needle: DEGENERATE** — repetition (`"40 40 40 40 …"` ×117), empty content, MISS.
- The **prior v20 (`6d32a0c3`, use_flattening build) PASSED 150k** with the identical harness/needle. So this is a **regression introduced by this candidate**, not the harness or the platform.

Prime suspect: **PR #154 forward-port (absorbed `kv_b_proj` source release)** — the failure is depth-triggered and attention-shaped, and #154 is the change that releases MLA attention-source storage. Secondary: the native-MTP3 flattening gate interaction. Throughput is also very low (see §5) but appears to be **separate** (present on the prior build too → platform/Gen3, not this candidate).

---

## 1. Exact candidate under test

```
image        ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-memory-reclaim-test-20260724
imageID      sha256:0567498e6d790e6fcd294be431381b71c03409049b0fd635462c1b1623ec2b91
base         voipmonitor/vllm@sha256:adddafd2b1749729fdf2d2ca23818c7c39f2a95e6fb05edd98657251913b83f2
integ rev    7373bb24c881fa05af57d7eaf8aa7b4e9f2d2ddb
patchset     #165, native-mtp3-gate, pr154-forward-port
container    45296b92a3d6   StartedAt 2026-07-24T22:00:10Z   RestartCount 0 (no hidden retries)
```

Built on CN4 from the byte-pinned worktree; **Dockerfile fail-closed checks all passed** (6 input hashes + `safe_mla_query_bmm` symbol + 5 output hashes). **CPU proof PASSED:**
```json
{"non_b12x_owner_inert": true, "reload_rematerialization": true, "source_parameters_released": true, "verdict": "PASS"}
```

Resolved runtime config (verified from logs): `gpu_memory_utilization=0.976`, `max_model_len=480000`, `max_num_seqs=16`, `kv_cache_dtype=nvfp4_ds_mla`, wire `i8_ring`, `VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=480000`, cudagraph sizes `[1,2,4,8,16,32,64]`.

## 2. Acceptance boot — PASS (but KV thin)

Booted clean, served, no OOM / no EngineCore fail / no reset.

```
Available KV cache memory: 3.64 GiB
GPU KV cache size:         481,792 tokens
Max concurrency @480,000:  1.00x        <-- clears >=480k, MISSES >=500k target => THIN
per-GPU memory: 85.22 weight / 2.87 peak-act / 0.61 non-torch / 0.38 CUDAgraph / 3.64 KV
```
Reclaim landed (weight 85.22 GiB vs ~85.5 without #154), but nets ~same KV as the prior build at 0.980 (487,424 tok). Log notes ~0.9 GiB left on the table (≈500k+ reachable at GMU~0.985). **Not retuned per your protocol** — flagged thin.

## 3. THE FAILURE — degenerate generation at depth

Needle = `738216` at position 0.40, unique per-depth prefix (defeats prefix cache; `cached=0` confirmed), harness reads content/reasoning/reasoning_content/whole-message.

| depth | ctx | completion | finish | secs | retrieval | finalization | output |
|------|-----|-----------|--------|------|-----------|--------------|--------|
| 50,000  | 49,141  | 73  | stop | 130 | **PASS** (content) | **PASS** | `738216` ✓ |
| 150,000 | 147,371 | **2**   | stop | 497 | MISS | FAIL(empty) | content=`''`, reasoning_tail=`'The'` |
| 250,000 | 245,602 | 117 | stop | 978 | MISS | FAIL(empty) | content=`''`, reasoning_tail=`'40 40 40 40  40 40 …'` |

- 50k is fully correct. At **≥150k the decode collapses**: 150k emits `"The"` then stops; 250k emits pure repetition (`"40 " × 117`). Both leave `content` empty and never surface the needle anywhere.
- Depth-monotonic collapse (worse as context grows) is the classic signature of **KV/attention corruption at long context**, not a sampling/temperature or harness issue.
- **Regression proof:** prior v20 `6d32a0c3` (same harness, same needle, GMU 0.980) returned `retrieval=PASS content='738216'` at 150k. This candidate does not.

## 4. Ruled out (so Sol can focus on code)

- **Harness / prefix cache:** identical harness PASSED 150k on the prior build; `cached=0` every depth (unique prefixes).
- **Hardware stability / power / thermal:** the box ran 50k+150k+250k deep prefills with **zero resets** (boot_id unchanged ~2h) at 300W + **locked 2600 MHz** (see §6); peak GPU temp **76°C**. Not a crash/reset.
- **OOM / boot:** none — RestartCount 0, container never exited, KV allocated fine.
- **Clock/power config:** 50k passes at the same 300W/2600 profile; failure is depth-triggered, not clock-triggered.

## 5. Throughput problem (likely separate — platform)

Effective prefill (context ÷ total request time; decode negligible at 150k/250k):

| depth | ctx | secs | prefill tok/s |
|------|-----|------|---------------|
| 50k  | 49,141  | 130 | 378 |
| 150k | 147,371 | 497 | 297 |
| 250k | 245,602 | 978 | 251 |

```
instantaneous prefill chunk-rate:  ~307 tok/s steady (231 samples), bursts to 614
decode (generation) throughput:    11.7 tok/s peak (50k), collapses to ~0 at degenerate depths
MTP spec-decode: mean acceptance 2.90 / draft accept 63.3%   (spec path itself healthy)
```
- Expected ballpark on this class of card (ref: jcartu EXL3 study, same 4× RTX PRO 6000): ~1,780–1,945 tok/s prefill @128k, ~105 tok/s decode single-seq. **CN4 is ~5–6× low on prefill and ~10× low on decode.**
- Prior build showed **similar prefill times** (150k ≈ 494–609s), so **prefill throughput is not specific to this candidate** — points to the **CN4 platform**: Xeon W-2195 / C422 SAGE is **PCIe Gen3** (idle drops to Gen1 via ASPM, trains to **Gen3 x16 under load** — confirmed). TP4/DCP4 + i8_ring PCIe collectives are Gen3-bandwidth-bound; per-token decode collectives explain the very low decode rate. Worth confirming whether decode is expected to be this collective-bound at Gen3, or if there's an image-side decode regression on top.

## 6. Platform / config context (CN4)

- **Board:** ASUS WS C422 SAGE, **Xeon W-2195** (Skylake-W, Gen3). GPUs: 4× RTX PRO 6000 Blackwell **Workstation Edition** (600W parts), PCIe **Gen3 x16 under load** (x8-downtrain from the old board is GONE on this board).
- **Stability fix in place:** the WS cards silent-reset under prefill at any `-pl` (transient di/dt from sprinting to ~2850 MHz). Fix = **locked clocks** `nvidia-smi -lgc 0,2600` + `-pl 300` → steady ~2170–2587 MHz, no sprint, **no resets** across the full sweep (2200/2400/2600 all stable; 150k prefill 502/510/494s respectively — prefill is clock-insensitive). Persisted in `nvidia-powercap.service`.
- **Known-unclean platform bits (may or may not matter):** `[Firmware Bug]: APEI: Invalid physical address in GAR` (hardware-error reporting partly broken → "silent" resets can't be fully trusted), **Intel ME 11.12.0.1622** (older than BIOS's expected build; no flashing path yet). EDAC UE/CE = 0. NVMe (Samsung 990 EVO Plus 2TB boot + Intel U.2 1.5T ext4 KV tier) clean.

## 7. Suspected code areas (for Sol)

1. **PR #154 absorbed `kv_b_proj` source release (top suspect).** Failure is depth-triggered decode collapse — exactly where released MLA source storage would bite if the absorbed pair isn't fully materialized/valid for long-context blocks. Check the release ordering vs. `B12X_MLA_SPARSE` long-context paged/gather path; verify the reclaim guard holds beyond the CKV gather bound and across chunked-prefill boundaries (>chunk, >graph capture sizes).
2. **`VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=480000` interaction.** Failure begins at 150k (well under 480k), but confirm the gather-arena bounding + #154 release don't interact at the 150k/250k block boundaries.
3. **native-MTP3 flattening gate.** Spec accept is healthy (2.90), but the flattening gate + #154 together are the two deltas over `6d32a0c3`; a differential build (this candidate minus #154) would isolate it fast.
4. **Decode throughput** (11 tok/s) — separate track: confirm Gen3 PCIe-collective-bound vs. an image-side decode regression.

## 8. Fastest isolation experiment

Rebuild **this exact stack minus the PR #154 forward-port** (keep #165 + flattening gate + CKV bound + GMU 0.976 + cudagraph set), and re-run 50k/150k/250k. If 150k returns to PASS → #154 is the cause. If still degenerate → flattening-gate/CKV path.

## 9. Evidence on CN4 (preserved)

```
/home/derek/needle-ladder-sol.log                 # full ladder log (verdicts + tails)
/home/derek/needle-out-sol-2600/                  # request-*.json + response-*.json per depth (50k/150k/250k)
docker logs glm52-v20-reclaim                     # full engine log (mem lines, throughput, MTP, config)
image sha256:0567498e…  base sha256:adddafd2…  rev 7373bb24…
```

**Recommended next:** #8 differential build. The hardware side (stable profile 300W+2600, thermals) is solved and reproducible — the remaining blocker is the deep-context correctness regression, then the throughput.
