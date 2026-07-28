# FP8 rank-consistency adversarial gate verdict

Verdict: **FOUR-GPU COLLECTIVE GATE PASSED; DEEP-CONTEXT SERVER GATE FAILED.
DO NOT USE AS A NEEDLE-RETRIEVAL FIX OR ENABLE FP8 IN PRODUCTION.**

Reviewed artifact: `patches/fp8-rank-consistent/`, against pinned v19 input
MD5 `96e07e55c3843766999b88e184ce06dd`.

## Field disposition (2026-07-19)

The four-GPU equality gate passed: the patched `ag` result was bit-identical
across ranks.  The matched server ladder then produced:

| Target total context | Result | Finish reason | Completion tokens |
|---:|---|---|---:|
| 50k | needle hit | `stop` | 92 |
| 200k | needle hit | `stop` | 71 |
| 300k | empty content / miss | `stop` | 19 |
| 350k | empty content / miss | `length` | 3,000 |

The 300k cell is a real retrieval miss rather than output-budget exhaustion.
The 350k cell is additionally a degenerate length-cap failure.  Passing 200k
does not prove an improvement over the unpatched v19 path because that exact
boundary was not measured before the patch.

The historical v1.3 comparison does not establish a 300k+ regression.  The
checked-out v1.3 branch lacks the large-DMA dispatcher, but commit `9ee73133d`
on the v1.3 image lineage does construct `PCIeDmaAllReduce` and lets it read
`B12X_PCIE_DMA_FP8`; the release documentation and measured `ag`/`ring`
throughput difference are also evidence that this path was active in the
image.  The preserved quality records cover 128k total context in the public
v1.3 battery and 200k total context in this project's full-context gate.  They
do not contain an E4M3 result at 300k or 350k total context.  v19 changes the
dispatcher from an autotuned crossover to a fixed 6 MiB floor, but this alone
does not prove that identical full-chunk tensors route differently.

The patch remains a valid fix for the replicated-output invariant.  It is not
the root-cause fix for the observed needle regression.

## Finding under review

The FP8 all-reduce modes do not currently return one replicated value.  The
owner keeps its pre-wire BF16 reduced shard while peers dequantize the FP8
payload.  Because each rank owns a different shard, every rank returns a
different activation tensor.

The proposed patch locally materializes the owner shard from the exact payload
already sent to peers.  It applies to `ag`, `ring`, and `a2a`.

## Adversarial challenges

### 1. Is rank equality actually required for an approximate collective?

Yes.  Approximation may change the value relative to FP32/BF16, but a
tensor-parallel all-reduce represents a replicated activation.  If rank `r`
evaluates the next local weight shard on `x_r` while another rank evaluates on
`x_s`, the subsequent summed result no longer represents one model function
evaluated at a common input.  Ordinary ring reduction-order differences do not
justify this: a completed shard is normally copied from one owner so all ranks
receive the same bits for that shard.

### 2. Could the proposed local store race the broadcast?

No source-level write/write or read/write race was found.  The payload-ready
event is recorded after quantization and before local materialization.  The
copy stream and local dequantization then read the same immutable payload.
The local store writes the BF16 output; the copy stream writes peer scratch.
The main stream orders subsequent uses of the local output after the store.

This conclusion must still be exercised under CUDA graph replay by the GPU
gate.

### 3. Does `ring` have a valid final payload to round-trip?

Yes.  On the last reduce-scatter hop, `dma_dequant_add_quant` writes both the
final FP8 payload/scale and the pre-wire BF16 owner output.  The first
all-gather step already forwards that payload.  The patch reads the same stage
slot and overwrites the BF16 owner output with its dequantized value.

### 4. Does `ag` accidentally add a second FP8 rounding?

No.  The new operation is dequantization only.  The owner is materialized from
the one payload that peers already receive.  Payload creation and PCIe bytes
are unchanged.

### 5. Does `a2a` need the same change despite lacking server failure data?

Yes at the collective-contract level.  It has the identical asymmetry after
the owner accumulation and broadcast quantization.  Extending the fix avoids
leaving a known defect in an untested mode, but `a2a` remains separately gated
and must not be inferred safe from `ag` results.

### 6. Could uniform FP8 materialization make model quality worse?

Yes.  The legacy owner kept one-quarter of its output in higher precision;
after the patch, every shard on every rank is wire-rounded.  Rank consistency
restores the collective abstraction but does not guarantee lower KLD or a
passing needle gate.  This is the central unresolved risk and why the verdict
does not clear a server boot in FP8 without the four-GPU equality gate first.

Because rank equality passes while the 300k and 350k needle gates fail, the
remaining cause is E4M3-path numerical loss or another version-dependent
interaction exposed by that lossy path.  The evidence does not yet distinguish
a codec limit from a change in which tensor shapes/call sites are compressed.
The correct response is to retain BF16, not weaken the quality gate.

### 7. Could the empty needle result be only a harness artifact?

Possibly.  The current harness records only `message.content`; it discards
hidden reasoning and `finish_reason`.  It also labels the target total context
as the needle depth even though the needle is placed at about 40%.  These are
reporting defects, but they do not invalidate the matched BF16-versus-FP8
causal result.  The server gate must capture full choice diagnostics.

### 8. Is the patch safely inert in production BF16 mode?

At source level, yes.  Both additions are inside FP8-only branches.  No
configuration default or normalization changed.  The first patched-image boot
must nevertheless keep both aliases at `0` and repeat the existing acceptance
ladder before any FP8 test boot.

## Local evidence

- Dependency-free schedule proof: PASS for `ag`, `ring`, and `a2a` at world
  sizes 2, 4, and 8.  Legacy outputs have `world` unique values; patched
  outputs have one.
- Source contract: PASS against the pinned v19 input and reviewed overlay.
- Python AST/compile gate: PASS for the overlay and all checks.
- Patch dry-run: PASS against the pinned v19 source.
- CUDA compile/import: not applicable; no CUDA source changed, but in-image
  Python import is still required.
- Four-GPU eager/graph equality: PASS in the target four-GPU environment.
- Deep-context server quality: FAIL at 300k and 350k target total context.

## Disposition

1. Keep `VLLM_PCIE_DMA_FP8=0` and `B12X_PCIE_DMA_FP8=0` in production.
2. Preserve the rank fix in any future lossy-wire experiment; rank divergence
   is still an invalid collective contract.
3. Do not test `ring`: `ag` already failed the required quality gate and ring
   introduces additional reduce-scatter requantization.
4. A future compressed-wire candidate needs a higher-fidelity codec or
   selective-precision routing and must repeat the entire server ladder.
