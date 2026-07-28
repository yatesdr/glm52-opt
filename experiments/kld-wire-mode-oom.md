# Wire-mode KLD capture — memory/threshold bind (Fable → Sol, before another boot)

Facts only. The goal, the exact setup, the two failures, the fundamental tension, and the open
questions. Nothing has a clean config yet — asking for your read before spending another ~12-min boot.

## Goal

Measure `KL(BF16 reference || candidate)` for the **DMA wire mode** axis so we can compare
**block-INT8 (`i8`) vs E4M3 (`ag`)** quality on the exact same model. The needle result already
shows INT8 passes 300k/350k where E4M3 fails; the KLD is meant to *quantify* the margin. Per your
guidance we measure both legs ourselves (don't borrow Discord numbers), and we're using the
FP8-KV-cache calibration's ~0.188 (compact-368B NVFP4-KV, bf16 wire) only as a context anchor.

## Setup (as run)

- Runner: `prefill_kld_fallback.py` (offline `LLM(...)`, `prompt_logprobs`/`return_prompt_logits`,
  full vocab), against `festr2/GLM-5.2-BF16-KLD-Reference-Logits-20260708` (1 window, 2048 tokens,
  154,880 vocab). Tokenized `first16` matched the reference manifest — the prompt is correct.
- Model/config (production-matched): `/model` (madeby561/GLM-5.2 hybrid), `--quantization
  nvfp4_nf3_hybrid` + our mxfp8 quant-config via `--llm-extra-json`, `--kv-cache-dtype nvfp4_ds_mla`,
  `KV_FP8_ROPE=1`, `--attention-backend B12X_MLA_SPARSE`, TP4, `--max-model-len 4096`,
  `--max-num-seqs 1`, MTP not set. `expandable_segments:True`, no offload connector.
- Overlays per leg: INT8 = int8 `.py`+`.cu`, F8_DMA=i8; E4M3 = rank-consistency `.py`, F8_DMA=ag.

## The two failures (both identical)

| Attempt | MNBT | GMU | Result |
|---|---:|---:|---|
| 1 | 2048 | 0.85 | `ValueError: No available memory for the cache blocks` (after weights+compile, at KV profiling) |
| 2 | 1024 | 0.85 | same `No available memory for the cache blocks` |

The full-vocab prompt-logit capture buffer scales with MNBT and, at GMU 0.85, the profiled peak
leaves **zero** room for KV blocks. Each attempt is a full weight load (~305 s) + a torch.compile
(~149 s, recompiles when MNBT changes the compile range) before it dies — ~11–12 min per failed boot.

## The fundamental tension (this is the crux)

`hidden_size = 6144`. The per-chunk TP all-reduce is `[MNBT, 6144] bf16`, and the b12x DMA path only
engages **above the 6.29 MiB (6,291,456 B) threshold** (boot log: `DMA min=6291456`):

| MNBT | all-reduce size | DMA engages? |
|---:|---:|---|
| 512 | **6.00 MiB** | **NO** (below threshold) |
| 640 | 7.50 MiB | yes |
| 768 | 9.00 MiB | yes |
| 1024 | 12.00 MiB | yes (but OOM'd) |
| 3072 (prod) | 36.00 MiB | yes |

So:
- The **official KLD protocol's MNBT=512** (the harness default that fits memory at GMU 0.74) is
  **below the DMA threshold → it would not exercise the fp8 wire at all** (all wire modes read
  identical). It was designed for the KV-cache/model-quant axis at bf16 wire, not the wire axis.
- To engage the wire we need **MNBT ≥ 640**, but MNBT ≥ ~1024 OOMs the logit capture at GMU 0.85.
- The viable window is narrow: **MNBT 640–768 + a higher GMU**, unverified.

## Open questions for you

1. **Config:** is `MNBT 640 (or 768) + GMU ~0.92` the right next single-shot, or is there a lower-
   footprint capture path? The error literally says "increase GMU," and the buffer is the full-vocab
   `prompt_logprobs`. Is there a supported way to capture the 2047×154,880 logits with a smaller peak
   (e.g. fewer positions per forward, `kld-chunk-rows` affecting capture not just compute, or a
   return-logits mode that streams)?
2. **Representativeness:** does a 640-token-chunk all-reduce faithfully represent the production
   MNBT=3072 wire effect? My reasoning says the fp8 penalty is ~MNBT-invariant above the threshold
   (per-128-block amax, blocks don't span chunks), so 640 ≈ 3072 — do you agree, or is there a
   reduce-scatter cross-chunk accumulation effect I'm missing?
3. **Eager vs compiled:** the calibration you saw ran **eager**; my run went through torch.compile
   (149 s, `compile range (1, MNBT)`). Does the KLD need eager for validity, or is compiled fine?
4. **Is it even worth it?** The needle A/B already isolates E4M3 precision (INT8 3/3 vs E4M3 0/3 at
   300k/350k, plus 475k). Is the wire-mode KLD necessary for the PR, or is the needle result
   sufficient and this is belt-and-suspenders not worth the config fight?
5. Is there an **established wire-mode KLD protocol** distinct from the KV-config one? Everything in
   the repo (`bench-glm52-v14-kld-keypoints.sh`, MNBT 512) targets the KV/quant axis at bf16 wire.

## What I have staged

- `~/kld/kld_leg.sh <leg> <F8_DMA> <overlay>` (parameterized runner), `~/kld/prefill_kld_fallback.py`,
  `~/kld/pydeps` (datasets), festr2 reference at `~/kld/glm52_bf16_ref/reference-logits/` (verified).
- Box is free (GPUs idle); no KLD leg currently running.
