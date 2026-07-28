# Block-INT8 PCIe-DMA all-gather candidate

Status: field quality confirmed at 300k and 350k total context, three runs
each with `finish_reason=stop` and zero restarts. Throughput, KLD, standalone
four-rank equality, and remaining acceptance gates are pending.

## 1. Corrected root-cause statement

The v1.3 and v19 `b12x/distributed/pcie_dma.py` files are byte-identical
(`96e07e55c3843766999b88e184ce06dd`), but their dispatch integrations differ:

- the v1.3 image lineage includes dispatcher commit `9ee73133d`, which
  constructs `PCIeDmaAllReduce`, lets it read `B12X_PCIE_DMA_FP8`, and sets an
  autotuned DMA crossover;
- v19 passes the FP8 environment value explicitly and selects eligible TP
  tensors at a fixed 6 MiB floor;
- A full 3,072 x 6,144 BF16 hidden-state reduction is 36 MiB.  The active v19
  path therefore wire-rounds the large attention-output and MoE/MLP-output
  reductions, approximately twice per transformer layer on every full
  prefill chunk.

The preserved v1.3 quality record reaches 128k total context publicly and
200k total context in this project's full-context acceptance; it has no
matched 300k or 350k E4M3 cell.  Short arithmetic requests do not exercise the
large DMA route.  Current BF16 DMA passes the 350k target-total-context gate;
rank-identical E4M3 DMA misses at 300k and degenerates at 350k.  A dispatch
ledger is still required to distinguish a codec limit from changed exposure.

The owner-roundtrip patch remains required for any lossy replicated
all-reduce, but it is not sufficient for retrieval quality.

## 2. Objective

Retain a one-byte activation payload and the existing BF16 reduce-scatter / 
compressed all-gather topology while reducing codec error enough to pass the
deep-context ladder.

The candidate adds one opt-in wire mode, `i8`, with:

1. BF16 reduce-scatter, unchanged from `ag`;
2. one symmetric signed-INT8 quantization of each completed owner shard;
3. one FP32 scale per 128 consecutive values;
4. byte-identical payload forwarding around the all-gather ring; and
5. local owner dequantization from that exact payload before return.

No INT8 reduce-scatter or a2a mode is proposed.  That keeps the candidate to
one rounding and avoids partial-sum requantization.

## 3. Why INT8 is the next codec, not another E4M3 routing tweak

The existing block-E4M3 codec transmits one byte per value plus one four-byte
scale per 128 values: 132 bytes per block.  Symmetric block-INT8 has the exact
same wire size and can reuse the same stage/scratch capacities.

For a block with absolute maximum `a`, INT8 uses `scale = a / 127` and
round-to-nearest with saturation to `[-127, 127]`.  Its absolute error is
bounded by `a / 254` before the final BF16 store.  E4M3 preserves exponent
range but has only three explicit mantissa bits, producing materially coarser
relative spacing for normalized activation distributions.  This is a codec
motivation, not a quality proof; outlier-heavy blocks can still make uniform
INT8 lose small values.

Using the same 128-value block is deliberate for the first gate:

- identical wire bytes and workspace sizes;
- no routing, event, flag-slot, or CE-copy changes;
- a direct E4M3-versus-INT8 quality A/B with precision as the only intended
  variable.

Smaller INT8 blocks are a later option only if activation-distribution
evidence shows outlier domination.  They change wire size and are outside
this candidate.

## 4. Source changes

`b12x/distributed/pcie_dma.cu`:

- add a one-warp-per-128 `quant_i8_kernel`;
- add `dequant_store_i8_kernel`;
- expose `dma_quant_i8` and `dma_dequant_store_i8`;
- keep all existing E4M3 entry points byte-for-byte behaviorally unchanged.

`b12x/distributed/pcie_dma.py`:

- normalize `i8`, `int8`, `ag_i8`, and `int8_ag` to the canonical `i8` mode;
- admit `i8` only to the all-gather compression branch;
- choose the INT8 quantize/dequantize entry points for that mode;
- preserve owner round-trip materialization and rank-identical output;
- report `wire_mode=int8-ag`.

The existing v19 integration already forwards the string value supplied in
`VLLM_PCIE_DMA_FP8`; no dispatcher change is required for a direct-compose
test.  The historical environment name is awkward for INT8 but avoiding a
third integration-file change keeps the experiment narrow.  A production
interface would rename this to a codec-neutral wire-mode setting only after
quality acceptance.

## 5. Collective and graph contracts

- Eligibility remains a function only of mode, shape, dtype, and configured
  capacity.  It is rank-invariant.
- No rank-local allocation or data-dependent routing is introduced.
- The owner publishes the immutable INT8 payload-ready event before locally
  dequantizing it.  The CE broadcast and local materialization may overlap as
  read-only consumers of the same payload.
- Peers forward received payload bytes verbatim.  There is no second INT8
  rounding.
- Existing flag-slot, neighbor-handshake, stream, and graph-replay ordering is
  unchanged.
- BF16, E4M3 `ag`, E4M3 `ring`, and E4M3 `a2a` behavior is unchanged.

## 6. Failure boundaries

The 6/6 field result demonstrates that one-byte block-INT8 precision is
sufficient for the previously failing 300k/350k retrieval cells. Reject the
overall production candidate if any remaining condition occurs:

- CUDA compilation or in-image import failure;
- eager or graph-replay output differs between ranks;
- error against FP32/BF16 reference exceeds the predeclared INT8 band;
- 50k, 200k, 300k, or 350k needle failure;
- abnormal `finish_reason`, repetition, engine restart, or CUDA error;
- throughput loses enough that BF16 is the simpler operating point.

The 300k and 350k failure boundaries have passed. They must not be weakened or
removed from future regression batteries.

## 7. Acceptance order

1. Static byte pins and source-contract checks. **PASS**
2. In-image CUDA extension compile/import. **PASS through serving execution**
3. Four-GPU eager and CUDA-graph replay at 512 and 3,072 rows:
   bit-identical across ranks plus comparison to an FP32 NCCL reference.
4. Patched image with wire mode `0`: existing BF16 acceptance ladder.
5. Separate `i8` boot, changing only both wire-mode aliases:
   50k, 200k, 300k, and 350k target-total-context needle tests with full
   choice JSON and prefix-cache deltas. **300k/350k PASS 3/3 each; other cells
   not reported in this verdict.**
6. Cold 55k throughput and phase timings only after every quality cell passes.
   **Pending.**

Production remains:

```text
VLLM_PCIE_DMA_FP8=0
B12X_PCIE_DMA_FP8=0
```
