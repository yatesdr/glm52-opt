# glm52-opt — GLM-5.2 prefill/decode optimization on 4x RTX PRO 6000 (PCIe, no NVLink)

Patches, designs, proofs, and reproduction steps that took **GLM-5.2
(753B hybrid-quant) cold prefill from 640 → 1,696 tok/s (2.65x)** and
decode from 51.7 → 67 tok/s on a 4x RTX PRO 6000 Blackwell box with
**PCIe Gen3 + PLX switches** — the worst fabric anyone runs this model on.
If it works here, it works on your Gen4/Gen5 rig.

## Headline results (55k-token cold prefill, quality gates green)

| Stack | tok/s | Delta |
|---|---:|---|
| Stock v1.3 serving stack | 640 | baseline |
| + fp8 wire mode (`B12X_PCIE_DMA_FP8=ring`) | 685 | +7% |
| + fp8 DCP query gather + fp8-ring output reduce-scatter (stage 1, ours) | 964 | +51% |
| + **packed-CKV ownership inversion** (stage 3, ours) | **1,696** | **+165%** |
| DCP1 physics ceiling (no DCP at all, ¼ KV capacity) | 1,879 | reference |

Decode: 51.7 → **67 tok/s** single-user (v1.4 decode overlays; our prefill
work is decode-neutral, verified). Both numbers cold, prefix-cache-proof,
with needle/deep-needle/arithmetic/JSON/coherence gates passing.

## The core idea (packed-CKV ownership inversion)

MLA models like GLM-5.2 store ONE head-independent 368-byte compressed KV
record per token, but have 64 query heads. Standard DCP prefill ships the
**head-multiplied query tensor** (rows × 64 heads × 576 B) to every KV
shard, then reduce-scatters the output. We invert ownership: **queries stay
local, the compressed KV records are gathered instead** — 5–10x fewer wire
bytes at typical contexts, no output reduce-scatter at all, byte-preserving
(no new quantization error). Measured per layer: 18.6 ms of DCP
communication → **1.4 ms**. Full analysis: `design/breakthrough-analysis.md`.

## Repo map

- `RESULTS.md` — every measurement, the per-phase profiler ledgers, and
  the memory-model findings (worth reading before tuning anything).
- `REPRODUCTION.md` — step-by-step: image, checkpoint, mounts, env, boots,
  acceptance criteria.
- `AGENTS.md` — written for AI assistants working with this repo: exact
  contracts, md5 pins, failure modes, porting guide.
- `patches/stage1-fp8-gather-rs/` — fp8 DCP query gather + collective-safe
  fp8-ring reduce-scatter + the phase profiler (3 overlay files).
- `patches/stage3-packed-ckv/` — the packed-CKV transport (4 overlay
  files + unified-diff manifest). Includes the Triton constexpr fix.
- `patches/phase2-fullcontext/` — **the shipped configuration**: packed-CKV
  at full 480k context, with the escrow/probe memory contract that makes it
  fit. Overlays + gate checks + manifest.
- `design/` — the full design/review chain: mechanism analysis, v1 and
  phase-2 design notes, gate verdicts, field fixes. The process is part of
  the point: every stage was design-gated, CPU-proven, then boot-verified.
- `harness/` — cold-safe prefill bench, quality gates, acceptance scripts,
  and the CPU proof suites (byte-exact collective schedule, ownership-
  inversion equivalence, remap reads, escrow state machine, profiler
  state machine).
- `compose/` — docker-compose profiles for the acceptance configurations.

## Status / roadmap

- **CONFIRMED + shipped:** the full stack at **480k context** with a
  599,040-token KV pool — 1,509 tok/s @ 55k and 1,126 tok/s on a cold
  463k-token request, quality gates green to 200k depth, +30% decode,
  tiered KV cache (DRAM warm tier + NVMe cold tier). See `RESULTS.md` §8
  and `patches/phase2-fullcontext/`.
- **In progress:** prebuilt image on GHCR + a one-line compose, so the
  shipped 480k configuration boots without assembling overlays by hand
  (today `compose/` covers the 64k acceptance profiles only).
- **Planned:** a measurement patch decomposing the residual ~21 ms/layer
  compute, to pick the next kernel target; DCP2 posture cells; evaluation
  of trellis-quant checkpoints (1M-token pools).

## Provenance & attribution

Built on the excellent serving stack lineage of
[davidsyoung/vllm-glm52](https://github.com/davidsyoung/vllm-glm52)
(Apache-2.0) and [vLLM](https://github.com/vllm-project/vllm)
(Apache-2.0), serving the
[madeby561 GLM-5.2 MXFP8/NVFP4/NF3 hybrid checkpoint](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid).
Our overlay files are derivative works of those Apache-2.0 sources and are
published under Apache-2.0 (`LICENSE`). The reduce-scatter/gather ideas
build on the b12x PCIe-DMA collective work in that lineage; the packed-CKV
ownership inversion was independently derived from profiler data (a
similar unpublished result was reported in the community — see
`design/breakthrough-analysis.md` §1 for the honest evidence trail).

Hardware context: 4x RTX PRO 6000 Blackwell Workstation (96 GB, SM120),
TP4 + DCP4, `nvfp4_ds_mla` 4-bit KV, MTP-3 speculation. Results measured
on PCIe **Gen3** x16 with dual PLX switch pairs and NO NVLink — expect
equal or better on newer fabrics.
