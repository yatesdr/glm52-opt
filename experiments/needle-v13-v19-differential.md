# Needle failure differential: v1.3 versus v19

Status: root cause isolated. The same-bandwidth block-INT8 candidate passed
three runs each at 300k and 350k total context; performance and margin gates
remain.

## Bottom line

The owner-roundtrip patch fixed a real collective bug, but it did not fix the
needle failure. The four-GPU output became bit-identical and the model still
missed at 300k and 350k total context.

The historical v1.3 evidence does not show a contradictory pass at those
sizes. The v1.3 image likely did run E4M3 DMA, but its preserved quality cells
end at 128k total context publicly and 200k total context in the phase-2
acceptance. No preserved v1.3 result uses 300k or 350k total context.

Therefore the best-supported current diagnosis is:

1. E4M3 DMA is the causal trigger in the matched v19 A/B: BF16 passes 350k;
   both lossy modes fail, and rank-consistent E4M3 still fails.
2. The rank-divergence defect is real but not sufficient to explain retrieval.
3. The repository does not establish a v1.3-to-v19 regression at matched
   total context. If an external v1.3 350k log exists, it is new evidence and
   must be compared byte-for-byte with the current prompt and response.
4. Block-INT8 provides that higher-fidelity one-byte wire mode and restores
   the failed retrieval cells. Production remains BF16 only until INT8 clears
   throughput, KLD, and the remaining acceptance gates.

## Source differential

### Codec implementation

The pristine v1.3 and pinned v19 `b12x/distributed/pcie_dma.py` files are
byte-identical:

```text
96e07e55c3843766999b88e184ce06dd
```

Both implement the same block-E4M3 codec and the same owner/peer asymmetry in
the unpatched source. There is no codec-source regression between these files.

### Dispatcher

The checked-out v1.3 branch does not contain the large-DMA integration at its
HEAD, but the image lineage contains commit `9ee73133d` ("Dispatch B12X DMA
allreduce at prefill sizes with autotuned crossovers"). It constructs
`PCIeDmaAllReduce`, which reads `B12X_PCIE_DMA_FP8`, and dispatches eligible
large tensors. The v1.3 release documentation also names `FP8 DMA wire (ag)`,
and the project's measured `ag`/`ring` speed difference supports activation.

v1.3 selected `dma.min_bytes` with a startup sweep. The window-2 record places
the measured floor at approximately 6.29 MiB. v19 uses a fixed 6 MiB floor and
passes the configured mode explicitly. Those floors are too close to explain
the current result without a tensor-shape dispatch ledger; a full
3,072-by-6,144 BF16 reduction is 36 MiB and routes through both.

### Exposure

For GLM-5.2, the large generic TP reductions are principally:

- attention `o_proj`, through `RowParallelLinear`; and
- the MoE/shared-output final reduction in `MoERunner`.

This is roughly two lossy replicated residual updates per transformer layer
for a full prefill chunk. Short arithmetic/decode requests are below the DMA
floor, explaining why the arithmetic control can remain clean while long
prefill retrieval fails.

## Test-semantic differential

`harness/quality_gate.py --depth-tokens N` does not put the needle at token
`N`. It creates approximately `N` tokens of total context and inserts the
needle at 40%. Thus the current 350k cell has roughly 350k total distractor
context with the needle around 140k.

`harness/quality_gate_fp8_ext.py --deep` creates approximately 200k total
context and puts its needle at 95%, around 190k. That was the historical
phase-2 ship gate. It is "deeper" by relative position but is still a much
shorter total-context retrieval problem than the current 350k cell.

Preserved evidence:

| Lineage/configuration | Total context | Relative needle position | Result |
|---|---:|---:|---|
| public v1.2/v1.3 record | 128k | 10/35/65/90% | 4/4 pass |
| project phase-2 E4M3 gate | 200k | 95% | pass |
| patched v19 E4M3 `ag` | 200k | 40% | pass |
| patched v19 E4M3 `ag` | 300k | 40% | miss, `stop` |
| patched v19 E4M3 `ag` | 350k | 40% | miss, `length` |
| patched v19 block-INT8 `ag` | 300k | 40% | 3/3 pass, `stop` |
| patched v19 block-INT8 `ag` | 350k | 40% | 3/3 pass, `stop` |
| v19 BF16 | 350k | 40% | pass |

The missing matched historical cell is the central fact. "Full depth" in the
old acceptance record meant 95% of a 200k document, not a 300k/350k total
document.

## Mechanism

`ag` performs one E4M3 round trip on each completed all-reduce shard. `ring`
also requantizes partial sums during reduce-scatter, so it has a strictly
larger numerical exposure. Each compressed residual update affects the states
used to write later KV records and the final query used to retrieve them.
Longer total context adds more sparse-attention competitors and a longer
distance between the planted fact and the final question. A short arithmetic
prompt does not exercise this route.

This explains why `ag` or `ring` can cause deterioration even though they are
"only communication" settings: they alter activations, not merely transport.

## Patch disposition

### Rank-consistency overlay

`patches/fp8-rank-consistent/` is a valid collective-correctness fix. It must
remain part of any future lossy replicated all-reduce experiment. It is not a
needle fix and E4M3 remains rejected.

### Block-INT8 candidate

`patches/int8-ag-rank-consistent/` keeps BF16 reduce-scatter and replaces only
the E4M3 all-gather round trip with signed block-INT8. It has the same wire
size: 128 payload bytes plus one FP32 scale per block. Local layout, source,
and error-bound proofs pass. The real CUDA serving path passed retrieval 3/3
at 300k and 3/3 at 350k with `finish_reason=stop` and zero restarts.

This is a controlled causal result: topology, block size, scale overhead, wire
bytes, and routing are held constant while the numerical representation
changes. E4M3 precision—not generic compression or the DMA schedule—caused
the observed deep-retrieval failures.

## Decisive next tests

1. Measure cold 55k throughput on the currently live INT8 server, with
   prefix-cache deltas and route evidence.
2. Capture the standalone four-GPU eager/graph equality gate if it was not
   already run.
3. Measure a clean local BF16/E4M3/INT8 KLD matrix; do not combine a local
   INT8 number with an unmatched community E4M3 number.
4. Complete the historical 200k@95% and standard operational/quality gates.
5. If an actual v1.3 300k/350k raw response exists, preserve the image digest,
   effective environment, crossover log, exact prompt hash, completion
   fields, and cache deltas. That would justify a call-site/activation ledger
   rather than treating the present record as a matched regression.
