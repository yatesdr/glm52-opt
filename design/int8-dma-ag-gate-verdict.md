# Block-INT8 PCIe-DMA adversarial gate verdict

Verdict: **CUDA SERVING PATH RUNS AND DEEP RETRIEVAL PASSES 6/6 AT 300K/350K.
ROOT CAUSE IS ISOLATED TO E4M3 PRECISION. THROUGHPUT, KLD, AND THE STANDALONE
FOUR-RANK GATE REMAIN BEFORE PRODUCTION CLEARANCE.**

Reviewed artifact: `patches/int8-ag-rank-consistent/` against the pinned v19
B12X Python and CUDA inputs.

## Field result (2026-07-19)

The `i8` server completed three independent runs at each failing E4M3 depth:

| Target total context | Runs | Result | Finish reason |
|---:|---:|---|---|
| 300k | 3/3 | `738216`, pass | `stop` on all runs |
| 350k | 3/3 | `738216`, pass | `stop` on all runs |

`RestartCount` remained zero throughout. E4M3 had reproduced its failures
three times, so the matched 3/3 INT8 result is not a one-run recovery.

Because INT8 and E4M3 use the same topology, 128-value block, FP32 scale
overhead, and 132-byte wire layout, this isolates the retrieval failure to
E4M3's numerical representation rather than DMA routing, bandwidth reduction,
or generic one-byte wire compression.

## What is proved locally

- Python parses and the mode aliases normalize to `i8`.
- INT8 is admitted only to the all-gather compression branch; reduce-scatter
  and a2a retain their existing behavior.
- Owner and peers call the same selected dequantization entry point.
- The CUDA source contains the signed-INT8 quantize/dequantize kernels and
  both pybind registrations.
- The CPU proof checks 132 bytes per block, saturation, the `amax / 254`
  pre-BF16 error bound, and identical owner/peer materialization on zero,
  normal, smooth, and outlier-heavy inputs.

## What remains unproved

- eager and graph-replay collective ordering;
- the proposed four-GPU max-error and RMSE bands;
- throughput parity with E4M3 `ag`;
- matched BF16/E4M3/INT8 KLD; and
- the remaining standard acceptance cells outside the reported deep ladder.

The successful serving runs prove that the CUDA extension compiled/imported,
the INT8 kernels executed through the real model path, and deep retrieval
recovered. Outlier-heavy 128-value blocks can still suppress small values, so
KLD and the complete quality suite remain useful margin measurements.

## Historical comparison caveat

The v1.3 image lineage likely ran the E4M3 DMA path: it includes an earlier
autotuned dispatcher, its release docs name the active FP8 DMA wire, and the
recorded `ag`/`ring` speed difference supports activation. The preserved
quality record, however, stops at 128k total context publicly and 200k total
context in this project's phase-2 gate. It has no matched 300k or 350k cell.

Therefore the current failure proves that E4M3 is unsafe for the required v19
350k gate. It does not by itself prove that v19 regressed from a successful
v1.3 350k cell.

## Remaining gates

1. On the live idle INT8 server, run the matched cold 55k throughput cell with
   prefix-cache deltas and effective `wire_mode=int8-ag` evidence.
2. Run `test_pcie_dma_int8_gpu.py` if the standalone eager/graph equality gate
   was not already captured.
3. When the box is free, measure BF16, E4M3 `ag`, and INT8 against one local
   KLD harness, identical prompts/configuration and BF16 as the reference.
4. Complete the historical 200k@95%, arithmetic, JSON, coherence, and normal
   operational gates before a production flip.

Use a locally measured E4M3 KLD cell; a Discord number with a different image,
prompt set, or serving configuration is not a valid differential.
