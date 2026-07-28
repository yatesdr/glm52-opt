# FP8 PCIe all-reduce rank-consistency fix

Status: four-GPU equality passed; server quality failed at 300k and 350k total
context. This is a collective-correctness fix, not a needle fix, and does not
authorize a production FP8 flip.

## 1. Problem statement

The v19 PCIe-DMA all-reduce offers three lossy E4M3 wire modes: `ag`,
`ring`, and `a2a`.  Matched server runs show that `ag` and `ring` fail the
350k-context needle gate while the BF16 wire passes under otherwise identical
serving configuration.

The immediate production response is `VLLM_PCIE_DMA_FP8=0` and
`B12X_PCIE_DMA_FP8=0`.  This design addresses a narrower implementation defect
so the FP8 modes can be tested again without weakening that rollback.

## 2. Evidence classification

### Measured

- BF16 wire passes the 350k-context needle gate.
- `ag` and `ring` return empty `message.content` at the same context target.
- The only serving flag changed in the matched run was the FP8 PCIe-DMA mode.
- The throughput benefit is material: 1,327 tok/s BF16, 1,458 tok/s `ag`, and
  1,639 tok/s `ring` at the recorded 55k profile.

### Code-proven

- vLLM dispatches eligible TP all-reduces of at least 6 MiB to
  `PCIeDmaAllReduce`; this is not limited to the DCP transport.
- A full 3,072-row, 6,144-hidden BF16 activation is 36 MiB.  The attention
  output projection is row-parallel and reduces its output in every layer;
  the MoE path can add another full hidden-state reduction.  Consequently,
  `ag`'s "single rounding" means one rounding at each eligible collective,
  not one rounding across the request.
- `ag` computes a reduced shard in BF16 on one owner, quantizes that shard for
  the all-gather, and never materializes the owner output from the payload.
  Peers do materialize it with `dma_dequant_store`.
- `ring` emits an FP8 final partial while separately storing the pre-wire BF16
  value on the owner; peers again materialize from the FP8 payload.
- `a2a` accumulates a BF16 owner shard, quantizes it for broadcast, and leaves
  the owner BF16 value in place while peers materialize from FP8.
- Therefore every FP8 mode returns rank-dependent replicated output: each
  rank retains one different pre-wire owner shard and receives the other
  shards through FP8.

### Inferred, not yet GPU-proven

- Rank-dependent TP inputs are more damaging than a uniformly quantized
  activation because the next tensor-parallel layer evaluates different
  inputs on different weight shards.
- The fixed 2,048-token sparse-index selection amplifies small activation
  changes as context grows: the selected fraction falls from about 4.1% at
  50k to 0.59% at 350k.
- The arithmetic and prose controls are separate short requests below the
  6 MiB dispatch floor, so their success does not exercise the failing FP8
  route.
- Removing rank divergence was necessary but did not restore the deep-needle
  gate. The field result isolates the remaining problem to E4M3-path numerical
  loss or a version-dependent interaction exposed by that lossy path.

## 3. Required invariant

For every logical reduced shard `s`, all ranks must materialize the same BF16
value:

```text
payload_s = quantize(reduced_bf16_s)
output_on_every_rank_s = dequantize(payload_s)
```

The legacy FP8 behavior instead implements:

```text
output_on_owner_s = reduced_bf16_s
output_on_peer_s  = dequantize(quantize(reduced_bf16_s))
```

An approximate all-reduce may differ from an FP32 reference, but its completed
output must not depend on the consumer rank.

## 4. Proposed change

After publishing each final owner payload, enqueue one local
`dma_dequant_store` from that exact payload back into the owner output.

- `ag`: quantize the final BF16 owner shard, record `_ag_ready`, then locally
  dequantize it.
- `ring`: reuse the FP8 payload emitted by the final reduce-scatter hop, record
  `_ag_ready`, then locally dequantize it.
- `a2a`: after quantizing the accumulated owner shard and recording
  `_a2a_ownq`, locally dequantize it.

The readiness event deliberately precedes the local dequantization.  The copy
engine can therefore broadcast the read-only payload concurrently with the
owner's read-only dequantization.

## 5. Unchanged contracts

- BF16 mode executes no new kernel.
- No collective, route, shape, stride, dtype, allocation, event, or stream is
  added.
- The FP8 payload and scale bytes are unchanged.
- Every routing decision remains rank-invariant.
- CUDA-graph capture uses existing kernels, storage, streams, and persistent
  events; there are no host values or module-global Triton reads.
- The patch changes only `b12x/distributed/pcie_dma.py` and requires no CUDA
  extension rebuild.

## 6. Expected cost

Each rank adds one local FP8-to-BF16 materialization for its one-quarter owner
shard.  At the 3,072 x 6,144 BF16 profile this is a 9 MiB BF16 output shard
read from roughly 4.6 MiB of payload plus scales.  It adds no PCIe bytes and is
scheduled to overlap the existing all-gather copy.  Actual throughput impact
must be measured; no numerical bound is claimed before a GPU run.

## 7. Gates

### Local gate

1. Dependency-free schedule proof demonstrates the legacy rank divergence and
   the rank-identical owner-roundtrip result for TP2/TP4/TP8 owner mappings.
2. AST/source-contract check permits only the owner materialization additions.
3. Python compilation and input/output MD5 pins pass.

### GPU collective gate

For `ag`, `ring`, and `a2a`, on four GPUs:

1. Run deterministic BF16 input through the patched collective.
2. Broadcast rank 0's completed result and require `torch.equal` on every
   rank, not a tolerance comparison.
3. Separately compare against the FP32 NCCL reference with the existing FP8
   error band.
4. Repeat under CUDA-graph replay with changed input.

### Server gate

Use the stable BF16 production configuration as control.  Change one variable
per boot:

1. `ag` plus this overlay: 50k, 200k, 300k, and 350k context-target needle
   tests; capture full choice JSON (`content`, `reasoning_content`,
   `finish_reason`, and usage), not only `message.content`.
2. If `ag` passes, run the same ladder for `ring`.
3. Record cold 55k throughput with prefix-cache metric deltas.
4. Reject a mode on any retrieval, arithmetic, JSON, coherence, restart,
   NaN/Inf, cross-rank, or unexplained throughput failure.

The collective fix succeeded at rank equality but failed as a retrieval fix.
Cross-rank equality is necessary but not sufficient evidence of model quality.
