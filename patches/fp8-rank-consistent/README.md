# FP8 PCIe-DMA rank-consistency patch

This bundle fixes a rank-dependent output defect in the v19
`PCIeDmaAllReduce` FP8 paths.  The four-GPU equality gate passed, but the
deep-context server gate subsequently failed at 300k and 350k target total
context.  This is a collective-correctness fix, not a needle-retrieval fix or
a production recommendation.  Keep production on `F8_DMA=0`.

## Scope

One Python source file changes:

```text
b12x/distributed/pcie_dma.py
```

No CUDA source, collective routing, allocation, or serving configuration is
changed.  For `ag`, `ring`, and `a2a`, the owner now materializes its reduced
shard from the same FP8 payload forwarded to peers.  BF16 mode is byte-for-byte
behaviorally unchanged.

See `../../design/fp8-rank-consistency-design.md` for the mechanism analysis
and evidence labels.

## Byte pins

Apply only when the installed input file has this MD5:

```text
96e07e55c3843766999b88e184ce06dd  b12x/distributed/pcie_dma.py
```

The reviewed overlay MD5 is recorded in `md5-manifest.txt`.  Refuse deployment
on input drift; do not force the patch or copy the overlay over a different
base.

## Local gates

From the repository root:

```bash
python3 harness/fp8-rank-consistency-proofs/test_owner_roundtrip.py
python3 patches/fp8-rank-consistent/checks/check_source_contract.py
python3 -m py_compile \
  patches/fp8-rank-consistent/overlays/b12x/distributed/pcie_dma.py \
  patches/fp8-rank-consistent/checks/check_source_contract.py \
  patches/fp8-rank-consistent/checks/test_pcie_dma_rank_consistency_gpu.py
```

The dependency-free proof demonstrates the legacy divergence and the fixed
rank-identical result.  The static gate verifies the pinned source and exact
owner materialization sites.

## Deployment staging

Do not change the production BF16 flags during installation.  Verify the image
file first, then either apply `fp8-rank-consistent.patch` at the Python source
root or mount/copy the overlay to the exact path.  Recheck the output MD5 before
booting.

The first boot remains:

```text
VLLM_PCIE_DMA_FP8=0
B12X_PCIE_DMA_FP8=0
```

This proves the overlay is inert in BF16 mode.  Only after that control passes
should a separate test boot change both values to `ag`.

## Required GPU collective gate

Inside the target image with four visible GPUs and the overlay installed:

```bash
python /path/to/test_pcie_dma_rank_consistency_gpu.py --mode ag
python /path/to/test_pcie_dma_rank_consistency_gpu.py --mode ring
python /path/to/test_pcie_dma_rank_consistency_gpu.py --mode a2a
```

Each mode must report that all four outputs are bit-identical.  A tolerance
comparison alone is insufficient; that was the gap in the prior GPU test.

## Required server gate

One variable per boot:

1. Patched overlay with BF16 wire: existing full acceptance ladder.
2. Change only the two FP8 aliases to `ag`: 50k, 200k, 300k, and 350k
   context-target needle gates, full response diagnostics, and cold 55k
   throughput with prefix-cache deltas.
3. Test `ring` only if `ag` passes every quality gate.

Capture `content`, `reasoning_content`, `finish_reason`, completion-token
usage, errors/restarts, and the effective environment.  The existing harness's
empty `message.content` alone cannot distinguish a retrieval miss from a
reasoning-budget exhaustion.

Reject the FP8 mode on any deep-needle failure even if the collective equality
test passes.  This patch removes rank divergence; it does not prove that E4M3
rounding itself is acceptable.

## Final field verdict

Patched `ag` passed at 50k and 200k, missed at 300k with `finish_reason=stop`,
and exhausted 3,000 completion tokens at 350k with `finish_reason=length`.
The mode is rejected for the long-context endpoint.

Historical v1.3 results are not a 300k+ counterexample in the preserved test
record.  That image lineage did have a large-DMA dispatcher and likely used
the configured E4M3 mode, but its recorded quality batteries stop at 128k and
200k total context.  No preserved v1.3 cell matches the current 300k or 350k
total-context test.
