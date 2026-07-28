# CN4 tuning + benchmark report — prod v19 vs cn3, post-ME-flash

**Date:** 2026-07-25
**Operator:** Fable (autonomous run while Derek away)
**Box:** CN4 — ASUS WS C422 SAGE, Xeon W-2195, 4× RTX PRO 6000 WS, Seasonic PX-1600
**Image benchmarked:** prod v19 `ca8481…` (the exact image cn3 runs) — chosen as the known-good, needle-passing baseline to isolate platform vs image.

---

## Executive summary

1. **✅ The Intel ME flash (11.12.0.1622 → 11.12.98.2655) FIXED the transient resets.** A full-boost **unlocked** prefill soak (~24 min at ~2850 MHz — the exact clock that used to silent-reset the box) ran with **zero resets**. The clock lock is no longer required for stability; I've set the profile back to cn3-parity (300W + unlocked). *(The APEI firmware-bug line persists — the ME update did not clear it — so silent-reset classification still isn't fully trustworthy; recommend a longer soak before declaring victory.)*

2. **⚠️ CN4 cannot reach cn3's throughput by tuning — it's ~5–6× slower on the *identical* image + wire config + same Gen3 fabric.** Both prefill and decode are **clock- and power-insensitive** (identical at 2600-lock, 2850, and 3090). So clock/power tuning does not close the gap. The bottleneck is the **PCIe collective path**, a hardware/topology characteristic of this board — needs a code-side or topology fix, not tuning.

3. **✅ Retrieval works on v19** (unlike Sol's memory-reclaim image — see `sol-handoff-1.md`): 8k/55k/128k all `retrieval=PASS`, non-empty content.

**Best tuning profile found:** `-pl 300` + **unlocked clocks** (no `-lgc`) — matches cn3, stable post-flash, and gives cn4's max achievable throughput (which the collective bottleneck still caps well below cn3). Persisted in `nvidia-powercap.service`.

---

## 1. Benchmark results (prod v19 on cn4) vs cn3

| metric | **cn3 (target)** | **cn4 (prod v19)** | ratio |
|---|---|---|---|
| KV pool @480k | 645,888 tok (1.34x) | **644,864 tok (1.34x)** | ✅ = |
| prefill (chunk rate) | ~1843 tok/s | **~307 tok/s** (peak 614) | **6.0× slow** |
| prefill 8k / 55k / 128k (effective) | — | 415 / 392 / 325 tok/s | — |
| decode (single-seq) | ~118 tok/s | **57–66 tok/s** | **~2× slow** |
| MTP mean acceptance | 3.05 | **3.21** | ✅ (cn4 better) |
| needle 8k/55k/128k | PASS | **PASS** (non-empty content) | ✅ |
| needle 250k | — | prefill >900s → harness timeout (speed, not correctness) |

KV pool matches cn3 exactly → **the platform is not memory/config-limited; Sol's image's 481k was image-specific.**

## 2. Tuning sweeps — what moved the needle (nothing, on throughput)

- **Clock (2200 / 2400 / 2600 / unlocked-2850):** 150k prefill = 502 / 510 / 494s; decode = 56 / — / 66 / 66 tok/s. **Prefill and decode are both clock-insensitive** → the critical path is *collective/latency-bound, not compute-bound*.
- **Power (150 / 200 / 250 / 300W):** pre-flash, only the *reset threshold* changed (all reset under prefill except via clock-lock); throughput unchanged. Post-flash, 300W unlocked is stable.
- **Clock-lock (2600) was purely a *stability* lever** (pre-flash, to stop the sprint-to-2850 transient). Post-ME-flash it's unnecessary.

**Conclusion: there is no clock/power profile that closes the throughput gap to cn3.** The gap is upstream of anything tunable from the operator side.

## 3. Where the throughput gap is (for Sol)

Ruled out (identical between cn4 and cn3): **image** (both prod v19 `ca8481`), **wire mode** (both `B12X_PCIE_ONESHOT_DMA` + PYNCCL, same crossover config `oneshot=65536/fused=86016/DMA=6291456`; **no fallback** on either), **MTP** (cn4 3.21 ≥ cn3 3.05), **PCIe link width** (both x16 Gen3 under load; cn4 switch uplinks x16 8GT/s — no downtrain on the new board), **KV pool** (identical).

Remaining difference = **the interconnect topology**:
```
cn4 topo:  GPU0-1 PIX, GPU2-3 PIX, cross-pair = NODE   (dual PEX 8747 switches; cross-pair traverses switch->host-bridge->host-bridge)
cn3 topo:  GPU0-1 PIX, GPU2-3 PIX, cross-pair = PHB    (single PCIe host bridge)
```
NODE is a strictly worse rung than PHB. On the C422 SAGE, GPU0/1 sit behind one PEX 8747 and GPU2/3 behind another; **cross-pair all-reduce traffic takes an extra switch + host-bridge hop and the two GPUs on each switch share a single x16 Gen3 uplink.** The `B12X_PCIE_ONESHOT_DMA` all-reduce is small-message, **latency-bound** (oneshot ≤64KB) — exactly the regime where the extra switch-store-and-forward hop hurts, and it explains why raising clocks does nothing. This is the leading hypothesis for the ~6× prefill / ~2× decode gap.

Secondary observation: cn4 emits many **`Triton kernel JIT compilation during inference … latency spike`** warnings (`_build_prefill_chunk_metadata_kernel`, `_gather_shared_paged_supertile_kernel`, `_map_global_topk_to_gathered_ckv_kernel`, `_pack_topk_routes_post_prefix_kernel`). One-time per shape, but worth confirming the AOT/warmup coverage matches cn3's.

## 4. Recommended next steps (Sol)

1. **Measure the collective directly:** P2P bandwidth+latency (p2pBandwidthLatencyTest / nvbandwidth) and a bare all-reduce microbench on cn4 vs cn3. If cn4's cross-pair latency is ~N× cn3's, that N should track the throughput ratio → confirms topology.
2. **Topology mitigation:** pin the DCP/TP rank→GPU mapping so the *heaviest* collective stays intra-switch (PIX) and cross-switch traffic is minimized; test `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE` / DMA-min tuning for the NODE path; consider whether an NCCL ring ordered to the switch topology beats oneshot-DMA on this fabric.
3. **BIOS:** confirm no MMIO/ACS/relaxed-ordering setting is throttling P2P DMA across the PEX switches on the C422 SAGE (ACS on can force P2P through the root complex).
4. **Warmup:** extend AOT/warmup to cover the JIT'd prefill shapes and re-measure.

## 5. Stability / hardware state (for the record)

- ME `11.12.98.2655` ✓ (flashed by Sol). APEI GAR firmware-bug **still present** (1 in dmesg) — unrelated to ME, hardware-error reporting still partly blind.
- Post-flash: unlocked full-boost prefill (~2850 MHz, ~250W/GPU) + decode, **no reset** over ~24 min. Peak GPU temp during the earlier sweep ≤76°C.
- Profile persisted: `nvidia-powercap.service` = `-pm 0 / -pl 300` (clock lock removed). If any instability recurs, re-add `-lgc 0,2600` (proven-stable throttle) as the fallback.
- Evidence: `/home/derek/prefill-bench.log`, `/home/derek/prefill-bench-json/`, `docker logs glm52-prod`, cn3 comparison via its logs.

## 6. Related

- **Sol's v20 memory-reclaim image is broken for deep context** — degenerate generation at ≥150k (regression from `6d32a0c3`), prime suspect PR #154 `kv_b_proj` reclaim. Full detail + isolation experiment in **`sol-handoff-1.md`**.
