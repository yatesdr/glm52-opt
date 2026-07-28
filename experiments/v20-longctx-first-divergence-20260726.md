# v20 long-context first-divergence investigation

Date: 2026-07-26  
Host: CN4  
Status: selector semantic change causally confirmed; explicit compatibility fix
passes frozen gate; full promotion qualification pending

## Frozen contract

The causal request is `fail-350k-r1` from the immutable
`causal-gate-freeze-20260726` bundle:

- rendered prompt: 343,727 tokens;
- prompt SHA-256:
  `f0d1c16d816b777f27a3882d9e6b5ef056852684ea155fb11dd845f9e1654ab5`;
- rendered input-ID SHA-256:
  `05adb5ce38c8d621bce4b6b1ddb165d4b62a69e22553cb1cb4a82d5389861a65`;
- needle: `738216`, once, at approximately token 137,496;
- required cache state: `cached_tokens=0`;
- required result: exact ticket retrieval and normal finalization.

The trace is gated on its unique 2,735-token final prefill chunk. It records
the final query row at four stage boundaries for 78 target layers and four TP
ranks: 78 x 4 x 4 = 1,248 records.

## Reference traces

The known-good reference is v19 `7ea567a` with B12X `4cfa...`, MNBT=3,072,
NVFP4 MLA KV, FP8 RoPE, and `i8_ring`. The failing v20 reference is
`5517197` with SparkInfer `be0edca`.

The initial v19/v20 comparison was not input-aligned. The v20 layer-0 input is
bit-identical to the checkpoint BF16 embedding row. The v19 input is the
large-batch block-INT8 all-reduce result:

- changed BF16 elements: 4,797 / 6,144;
- max absolute delta: `3.814697265625e-05`;
- relative L2: `0.006619501209041023`;
- cosine: `0.9999790787696838`.

This excludes tokenizer, token ID, checkpoint-row, and weight-loading drift.
It does not prove that the transport perturbation controls retrieval.

## Failed discriminator attempt: actually wire=0

The first v20 retry attempted to set:

```text
VLLM_PCIE_DMA_FP8=i8_ring
B12X_PCIE_DMA_FP8=i8_ring
```

That was not a valid discriminator. The release launcher derives all
low-level wire variables from `F8_DMA`, whose default is zero, and overwrote
the attempted values. Runtime `/proc/1/environ` proves:

```text
B12X_PCIE_DMA_FP8=0
SPARKINFER_PCIE_DMA_FP8=0
VLLM_PCIE_DMA_FP8=0
```

The compiled SparkInfer cache record independently reports
`SPARKINFER_PCIE_DMA_FP8=0`.

The run is retained as a repeated wire=0 control:

- boot: RC 0, zero restarts, healthy;
- diagnostic max length: 460,000;
- KV pool: 500,445 tokens;
- trace: 1,248 / 1,248 records, zero temporary files;
- frozen result: `ABSENT`, `finish_reason=length`, 2,000 output tokens,
  `cached_tokens=0`, 434 seconds;
- morphology: degenerate repeated `how` output.

Its comparison reproduces the unaligned result:

- first hidden/residual difference: layer 0 input;
- first top-k difference: layer 0 attention;
- first v19-present/v20-absent needle selection: layer 34 attention;
- v19 needle positions at that layer:
  `137491, 137493, 137503, 137512, 137529`;
- v20 needle positions: none;
- top-k Jaccard at that layer: `0.10792534487422234`.

Hidden and residual records are rank-consistent in both images. The only
rank-varying trace field is the expected rank-local top-k index vector.

Evidence:

- comparator:
  `/home/derek/proof-results/20260726/longctx-first-divergence/comparison-v19-v20-i8.json`
  (`f5c606fc3bec00a60339a31eea3b586a076e48e88a6f401125a391f33b38fd34`);
- container inspect:
  `v20-i8-ring/container-inspect-wire0.json`
  (`3b8e89894c2d334435b54ad1cb6f6ca4c533d2f1cbc8ff614cb9393959231c74`);
- log:
  `v20-i8-ring/docker-wire0.log`
  (`002c4e679fcd45ea2fc84af7e3d0eb393305e0bbc478093071620ae98092ac53`);
- runtime environment:
  `v20-i8-ring/runtime-env-wire0.txt`
  (`e3337b3e2158f245f2c978bc508474fc9af8e6f61930f6af22a6911fa9ecc0f2`).

## Corrected discriminator

The corrected override uses the launcher's authoritative public control:

```text
F8_DMA=i8_ring
```

Pinned override SHA-256:
`d4c2c30156cca5c881827a834236cb23178a30b367b80166477382a7e9b6e279`.

Resolved Compose SHA-256:
`2596c4a9e8dc515a36f50f6b779ff5b4d642faf647f22a297054307b0ac03566`.

The live process started at `2026-07-26T12:52:05Z` and proves:

```text
F8_DMA=i8_ring
B12X_PCIE_DMA_FP8=i8_ring
SPARKINFER_PCIE_DMA_FP8=i8_ring
VLLM_PCIE_DMA_FP8=i8_ring
```

All other causal variables remain fixed: exact v20 image/weights,
TP4/DCP4/MTP3, MNBT=3,072, prefetch depth zero, NVFP4 MLA KV, FP8 RoPE, and
the same frozen request. The 460k maximum is a diagnostic admission-only
change; the request remains 343,727 tokens.

The corrected run remained a retrieval failure:

- boot healthy, zero restarts;
- KV pool: 502,493 tokens;
- complete trace: 1,248 / 1,248 records;
- frozen result: `ABSENT`, `finish_reason=length`, 2,000 output tokens,
  `cached_tokens=0`, 360 seconds;
- morphology: degenerate repeated `how` output.

The corrected wire aligns the layer-0 input exactly. The first non-exact stage
is layer-0 attention:

- changed hidden elements: 5,915 / 6,144;
- maximum absolute delta: `0.00146484375`;
- relative L2: `0.039866`;
- top-k Jaccard: `0.5312`.

The first needle-selection loss is layer-34 attention. The v19 trace selects
needle-adjacent positions
`137491, 137493, 137503, 137512, 137529`; the v20 trace selects none.

Evidence:

- result directory:
  `/home/derek/proof-results/20260726/longctx-first-divergence/v20-i8-ring-actual`;
- comparator:
  `comparison-v19-v20-i8-actual.json`
  (`c177f7524a042038145b9642c0c3050c9cf5fb4f12662fac9b242a94fa6bb55d`).

## Root cause: default-output lifetime violation

The layer-0 attention record exposed a stronger invariant than a numeric
drift. In v19, the residual at the attention boundary is bit-identical to the
layer input on all four ranks. In v20:

- 6,141 / 6,144 residual elements differ from the input;
- the residual is bit-identical to the attention output;
- `norm(residual - attention_output) == 0`;
- `norm(residual - input - attention_output) == norm(input)`.

The original residual has therefore been replaced by the attention result.
This is storage corruption, not accumulated low-precision error.

Source history identifies the exact semantic change. v19's
`PCIeDmaAllReduce.all_reduce()` allocates a fresh default result with
`torch.empty_like(inp)`. SparkInfer PR #76 changed v20 to return every
default-output DMA all-reduce through one persistent byte workspace. That
violates the out-of-place collective contract whenever one result remains
live across a later call.

The model has precisely that lifetime:

1. the embedding all-reduce result becomes the layer-0 residual;
2. the attention output projection performs another large all-reduce;
3. both calls return a view beginning at the same persistent address;
4. the attention collective overwrites the live embedding residual.

PR #76's tests asserted that successive default outputs had the same pointer,
but never retained and rechecked an earlier result. They therefore encoded
the bug as an expected implementation detail.

## Off-model production-geometry proof

`harness/v20_pcie_dma_output_lifetime_proof.py` runs two four-rank
`i8_ring` collectives at the frozen tail geometry (`2735 x 6144`, 33,607,680
bytes per result). It retains call 1 while call 2 executes and checks storage,
content, and rank consistency.

Stock v20 (`be0edca`, installed `pcie_dma.py`
`8cd43a0a9f7b8e1558f617e2a8eca7c7755d2f800a479d00cdb6ad04ed12f82f`):

```text
pointer_alias=true
retained_mismatches=16,803,840 / 16,803,840
rank_consistent=true
status=FAIL
```

Minimal fix `12c0bc19691fb09c10b3f4ac786420c83758f854` restores a fresh
default output while preserving explicit caller-supplied output buffers:

```text
pointer_alias=false
retained_mismatches=0 / 16,803,840
rank_consistent=true
status=PASS
```

The collective output hashes are identical between stock and fixed images.
The patch changes lifetime only, not wire numerics.

Evidence:

- stock:
  `/home/derek/proof-results/20260726/pcie-output-lifetime/current-image.jsonl`;
- fixed:
  `/home/derek/proof-results/20260726/pcie-output-lifetime/fixed-image.jsonl`;
- fixed image:
  `glm52-serve:v20-longctx-trace-output-lifetime-fix-12c0bc19-20260726`
  (`sha256:5aabe80d6df0105c271717630c895146c0a59602d3ca55e2335405eb0c919ad2`).

## Causal model qualification

The fixed image boot uses the same resolved compose, weights, trace,
TP4/DCP4/MTP3 posture, MNBT=3,072, NVFP4 MLA KV, FP8 RoPE, and runtime-proven
`i8_ring`. Only the image, container name, and trace output path differ.

Resolved Compose SHA-256:
`b661412453d4d31c279484fbb2cb519b94d8618c281f6819c2d5a986e01c4162`.

The qualification order is deliberately short:

1. boot and KV admission;
2. frozen cold 350k causal request;
3. layer-0 residual must remain bit-identical to layer input;
4. only after those pass, run the remaining deep ladder.

The output-lifetime fix passed the storage invariant: the layer-0 residual
remained bit-identical to the layer input. It did **not** restore retrieval.
The frozen 350k request remained `ABSENT`, cold, and ended by length after
2,000 output tokens. The lifetime defect was therefore real and mandatory to
fix, but it was not the only causal defect.

With the lifetime defect removed, the learned indexer boundary became
input-aligned:

- quantized query bytes were exact on all four ranks;
- indexer weights were exact on all four ranks;
- the final BF16 current-token key was exact on all four ranks;
- sequence length, page IDs, and geometry were exact;
- historical v19 and current v20 first differed in local top-k selection.

## Learned-input selector replay

`harness/v19_v20_learned_paged_indexer_replay.py` replays the captured
production tensors through either implementation while preserving the real
2,735-row dispatch shape, shared page-table route, 85,932 rank-local key
length, and top-k 2,048.

The four-way replay establishes that the result follows the selector
implementation rather than the model inputs:

- current v20 returns the exact dense-reference top-k, 2,048 / 2,048, on
  either v19 or v20 learned tensors;
- historical v19 returns approximately 1,872--1,896 of the exact-reference
  set and omits approximately 165--193 candidates whose proxy scores are
  strictly higher;
- the explicit `bounded_compat` v20 policy reproduces the historical v19 set
  at approximately 0.97--0.99 Jaccard;
- the widened exact policy differs from the bounded set by approximately
  15%--16%.

This corrects an earlier overclaim. The widened selector is exact with respect
to the **quantized indexer proxy score**. That does not establish that the
proxy's exact ordering is the model-optimal sparse-attention set. The proxy
uses E4M3 query values and an FP8 index-key cache; the attention computation
that consumes the selected pages is a different, higher-information
calculation.

Source history separates the semantic and performance changes:

- `1012199e` added the exact overflow rescan;
- `83a58444` widened the coarse radix from 8 to 10 bits and the shared
  candidate buffer from 4,096 to 8,192.

The first commit changes the selected set whenever the threshold bucket
overflows. The second makes the exact path faster and changes where overflow
occurs. Treating only the later width change as the regression was therefore
incomplete.

## End-to-end selector discriminator

The discriminator image changes only the selector policy on top of the
output-lifetime-fixed v20 image:

```text
image:
  glm52-serve:v20-longctx-indexer-bounded-discriminator-20260726
image ID:
  sha256:0469df9293ecb129f60abbce0f38e0d86edf3996600d7acb9f657a7e4ac529e2
source SHA-256:
  a136f11140da3b582bb136eb50b9977ba13adf6d111cd2ffa1420a86d81361eb
policy:
  SPARKINFER_NSA_TOPK_SELECTION_POLICY=bounded_compat
```

`bounded_compat` is explicit and bounds-checks every shared-memory write. It
uses the historical 8-bit coarse bucket and 4,096-entry refinement budget and
does not invoke the exact overflow rescan. Default `exact` behavior remains
unchanged.

Boot evidence:

- healthy, restart count zero;
- production and speculator decode graph capture completed 16 through 1;
- KV pool 507,612 tokens at max length 460,000;
- zero illegal-access, cuBLAS, EngineDead, OOM, traceback, assertion, or
  worker-died signatures.

Frozen cold causal results:

| Cell | Stock v20 | Discriminator | Finish | Cached | Content |
|---|---|---|---|---:|---|
| 250k control | `EXACT` | `EXACT` | `stop` | 0 | `738216` |
| 350k r1 | `ABSENT` | `EXACT` | `stop` | 0 | `738216` |
| 350k r2 | `ABSENT` | `EXACT` | `stop` | 0 | `738216` |
| 350k r3 | `ABSENT` | `EXACT` | `stop` | 0 | `738216` |

All three independent frozen failure prompts recovered exactly. This is an
end-to-end causal result: with weights, runtime posture, transport, cache
format, RoPE format, batching, and prompts held fixed, changing the selector
semantics changes the model result from failure to exact retrieval.

Primary evidence:

- stage 1 summary:
  `/home/derek/proof-results/20260726/longctx-indexer-bounded-discriminator/causal-stage1/summary.json`
  (`5881aee96f98bdcbacef3fc9f13b8b90aa24b5074f51da8174c1587dafe97916`);
- stage 2 summary:
  `/home/derek/proof-results/20260726/longctx-indexer-bounded-discriminator/causal-stage2/summary.json`
  (`1d91c2445bf01e0b7bdbc9167085c4e1d4db393d7050359218f265e7ede9699`);
- final container log:
  `/home/derek/proof-results/20260726/longctx-indexer-bounded-discriminator/final-container.log`
  (`81e03f3cdf9153ae5a8c38d23655ac719385ceb79d550ba48376b84fd365eb22`);
- final container inspect:
  `/home/derek/proof-results/20260726/longctx-indexer-bounded-discriminator/final-container-inspect.json`
  (`08376198a06520db91b3a1971c067bb9f12b21e7cb0a2745c90bc38f9c2ddafd`).

The same live process then passed an independent cold generalization ladder
using newly randomized natural-language headers:

| Target | Rendered tokens | Cached | Completion | Result |
|---:|---:|---:|---:|---|
| 50k | 49,100 | 0 | 91 | `EXACT` |
| 150k | 147,275 | 0 | 107 | `EXACT` |
| 300k | 294,619 | 0 | 119 | `EXACT` |
| 450k | 441,964 | 0 | 80 | `EXACT` |

Every response finalized normally with `content == "738216"`. The 441,964
token cell is the highest safe generalization point under this discriminator
boot's 460,000-token admission limit. A clean 480,000-token promotion boot is
still required for the established 475k cell.

Generalization evidence:

- summary:
  `/home/derek/proof-results/20260726/longctx-indexer-bounded-discriminator/generalization-ladder/summary.json`
  (`321f68d7fd44b916c80b570af90d60a2dde65eaa14aeacc5a53d729893d70b54`);
- run log:
  `/home/derek/proof-results/20260726/longctx-indexer-bounded-discriminator/generalization-ladder/run.log`
  (`e789dea27361d650b60870f0a1bae8f4e63312932894c4a0a0951dc807bf1437`).

## Clean 480k production-candidate proof

The final derived candidate removes the selector trace and every diagnostic
overlay. It is built from the clean `5517197/be0edca` release image and changes
only the two reviewed SparkInfer files:

- PCIe DMA output-lifetime fix from PR #80;
- explicit `bounded_compat` selector policy from PR #82.

Candidate identity:

```text
image: glm52-serve:v20-longctx-production-candidate-20260726
image ID: sha256:259098b0f5a83c775ff09f8979f3fec8982b88d8d58434dc4c9284f2b4e68905
max model length: 480000
KV pool: 500992 tokens
```

At TP4/DCP4/MTP3, MNBT 3,072, NVFP4 MLA KV, FP8 RoPE, `i8_ring`,
prefetch depth zero, and cold cache, the established deepest gate passed:

| Target | Rendered tokens | Cached | Completion | Finish | Content |
|---:|---:|---:|---:|---|---|
| 475k | 466,493 | 0 | 78 | `stop` | `738216` |

The arithmetic and coherence side checks also passed. The container remained
healthy with restart count zero, and the final log contained zero
illegal-access, cuBLAS-error, EngineDead, OOM, Xid, assertion, or worker-died
signatures.

Evidence:

- cell:
  `/home/derek/proof-results/20260726/longctx-production-candidate/needle-475k/cell-475k.json`
  (`f1ca6da58dd7f6528be04b6fd200315a972735ec83afa5ad851898b5310c6e23`);
- summary:
  `/home/derek/proof-results/20260726/longctx-production-candidate/needle-475k/summary.json`
  (`4b9bba777e311943f82b668f18146765f04ef5829c11e4d34f82a4f9c32ba987`);
- run log:
  `/home/derek/proof-results/20260726/longctx-production-candidate/needle-475k/run.log`
  (`f924d418a18d233a901617fe082c40504f07292ed002ace9c9fe72716bf18127`).

This closes the clean-image quality proof. Performance and memory
qualification remain separate promotion gates.

## Why the proxy-exact set hurts this model

`harness/v20_indexer_selection_bias_analysis.py` compares the exact and
bounded sets from the same captured learned tensors. Across all four ranks,
the changed portion of the exact set is strongly position-biased:

- exact-only candidates: 151--173 per rank, and every one is in the final
  quarter of the 85,932-token rank-local context;
- bounded-only candidates: 151--173 per rank, with 130--152 in the first
  half and only four per rank in the final quarter;
- the exact-only candidates have strictly higher quantized-proxy scores, so
  this is not an implementation error in the exact selector.

Thus the exact overflow rescan systematically reallocates approximately
7.4%--8.4% of the sparse-attention budget from older positions to the newest
quarter. The historical bounded selector preserves materially more old-token
coverage. That is the relevant semantic difference for a deep needle.

This model does not configure an independent sliding/local window for sparse
layers. `B12X_MLA_SPARSE` rejects non-null `sliding_window`, and the
SparkInfer block-sparse paged-attention path rejects `window_left`. Sparse
layers consume only the selected 2,048 indices. The checkpoint interleaves
full and sparse layers (`full, full, full, shared, shared, shared, ...`), but
that does not restore an omitted old token inside a sparse layer.

Evidence:

- selection-bias report:
  `/home/derek/proof-results/20260726/learned-indexer-replay/selection-bias.json`
  (`b09c41acc3690ee11284e87b8c6a2e5e4e0f721c22c938177ed93a19a0755357`);
- analyzer:
  `harness/v20_indexer_selection_bias_analysis.py`
  (`b687227fbd3fc6cef91027b2f703d0d3de48e14bd50542107bb637e9c44cfb68`).

## Production-fix posture

The causal result does **not** justify silently restoring an accidental
overflow as the final upstream design. It proves that selector semantics are
the controlling variable.

The minimal production-compatible fix is an explicit, fail-closed
`bounded_compat` policy. It preserves the validated v19 selection semantics,
keeps every shared-memory write bounds-checked, leaves `exact` as the default,
and makes the quality/performance choice visible in configuration. It is
appropriate for the GLM-5.2 NF3/NVFP4-KV/FP8-RoPE deployment while a new
universal default is designed.

The production path is:

1. retain the output-lifetime fix;
2. use `bounded_compat` explicitly and qualify it against the v19 quality
   baseline;
3. require the frozen 250k control, all three 350k failures, the complete deep
   ladder, and performance/KV gates before making that policy the default.
4. separately design and validate a deterministic age-aware candidate budget
   or a higher-information rerank before proposing a new universal default.

## Decision rule

1. Selector compatibility is causally established, not merely correlated.
2. The compatibility image is not promoted until the full deep ladder passes.
3. A new universal default must be deterministic and justified against
   end-to-end model quality, not only exactness for the quantized proxy.
4. No parser workaround, lower-precision cache disablement, or v19 fallback is
   accepted as a fix.
