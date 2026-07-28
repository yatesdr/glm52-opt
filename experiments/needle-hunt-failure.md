# Needle-retrieval failures under fp8 PCIe-DMA modes (deep context)

Prepared for Sol. Facts only — measured results and configuration, no interpretation.

## Scope

Deep-context needle retrieval was tested against the GLM-5.2 MXFP8-NVFP4-NF3 hybrid checkpoint
on CN3 (4× RTX PRO 6000, 96 GB) while varying one flag at a time. This documents which modes
passed and which failed the needle check, at which depths.

## Environment (constant across all runs below)

- Image: `voipmonitor/vllm:gilded-gnosis-v19-vllm7ea567a-b12x4cfa530-fi801d57a-cu132-20260718`
  (vLLM `v0.11.2.dev280+gilded.gnosis.v19`).
- Model: GLM-5.2 hybrid, TP4 / DCP4 / MTP3.
- Serving flags: `--gpu-memory-utilization 0.970`, `--max-model-len 480000`,
  `--kv-cache-dtype nvfp4_ds_mla`, `KV_FP8_ROPE=1`, `--max-num-seqs 16`,
  `--max-num-batched-tokens 3072`, `--attention-backend B12X_MLA_SPARSE`, `--moe-backend b12x`,
  `--dcp-comm-backend a2a`, KV offload ON (`OffloadingConnector` / `TieringOffloadingSpec`,
  64 GB DRAM tier). GPU KV pool 644,864 tokens.
- Test tool: `quality_gate.py --depth-tokens N`. It embeds a fixed needle value (`738216`) at
  token depth N and reports three independent checks: `needle` (retrieval of `738216`),
  `arithmetic`, `coherence`. `GATE: PASS` requires all three.

## Modes referenced

- `B12X_PCIE_DMA_FP8` (alias `VLLM_PCIE_DMA_FP8`): fp8 wire mode for the b12x PCIe all-reduce
  (`b12x/distributed/pcie_dma.py`, class `PcieRingAllReduce`). Accepted values `0`, `ag`, `ring`,
  `a2a`. Per the docstring in that file: `ag` quantizes the allgather phase only (one rounding);
  `ring` quantizes every reduce-scatter hop and the allgather payload; `0` = bf16 (no fp8).
- `ag_rs`: a value of `--dcp-comm-backend` / `DCP_BACKEND` (DCP4 collective backend). It is a
  distinct flag from `B12X_PCIE_DMA_FP8`. **Not tested for deep-needle retrieval — no data.**

## Results

Retrieval outcome by mode and depth (needle check only; `''` = empty retrieval):

| `B12X_PCIE_DMA_FP8` | 50k | 150k | 300k | 350k | 480k |
|---|---|---|---|---|---|
| `0` (bf16) | PASS | not run | not run | **PASS** | not run |
| `ag` | not run | not run | not run | **FAIL (`''`)** | **FAIL (`''`)** |
| `ring` | PASS | not run | not run | **FAIL (`''`) ×3** | **FAIL (`''`)** |

### Exact recorded output

**bf16 (`B12X_PCIE_DMA_FP8=0`)**
```
[PASS] needle@50000:  '738216'   | arithmetic PASS | coherence PASS (181 words) | GATE: PASS
[PASS] needle@350000: '738216'   | arithmetic PASS | coherence PASS (184 words) | GATE: PASS
```

**`ag` (`B12X_PCIE_DMA_FP8=ag`)** — GMU 0.970, offload on, a2a
```
[FAIL] needle@350000: ''         | arithmetic PASS | coherence FAIL (110 words) | GATE: FAIL
[FAIL] needle@480000: ''         | arithmetic PASS | coherence PASS (178 words) | GATE: FAIL
```

**`ring` (`B12X_PCIE_DMA_FP8=ring`)** — GMU 0.970, offload on, a2a
```
[PASS] needle@50000:  '738216'   | arithmetic PASS | coherence PASS (193 words) | GATE: PASS
[FAIL] needle@350000: ''         | arithmetic PASS | coherence PASS (182 words) | GATE: FAIL
  recheck 1: needle@350000: ''   | arithmetic PASS | coherence PASS (186 words) | GATE: FAIL
  recheck 2: needle@350000: ''   | arithmetic PASS | coherence PASS (195 words) | GATE: FAIL
[FAIL] needle@480000: ''         | arithmetic PASS | coherence PASS (178 words) | GATE: FAIL
```

## Additional facts

- In every recorded fp8 failure above, the `arithmetic` check returned the correct value
  (`13444`) in the same response where the `needle` check returned `''`.
- `ring` @350k failure was reproduced 3 times (initial + 2 rechecks); all returned `''`.
- bf16 and fp8 runs were on identical serving flags except `B12X_PCIE_DMA_FP8`.
- Prefill throughput on the same runs (55k cold, C1): bf16 1,327 tok/s; `ag` 1,458 tok/s;
  `ring` 1,639 tok/s (server-side Prometheus, `quality_gate`/LIL bench).
- Per `RESULTS.md` §5, the prior production config (v1.3 + v1.4 overlay + stage-3) ran
  `B12X_PCIE_DMA_FP8=ag` and passed a needle at 95% depth of a 200k-token context; that config
  was not validated beyond 200k depth.

## Not tested (no data)

- `ag_rs` (`--dcp-comm-backend`) at any depth.
- `B12X_PCIE_DMA_FP8=a2a` (the third fp8 mode) at any depth.
- Depths 150k and 300k for any mode (the boundary between the 50k pass and 350k fail is unmeasured).
- `ag` at 50k.
- bf16 at 480k.
