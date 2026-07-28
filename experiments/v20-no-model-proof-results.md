# v20 recovery no-model proof results

Date: 2026-07-24/25  
Host: CN4, four RTX PRO 6000 GPUs, TP4/DCP4 topology  
Operator: Fable  
Technical owner: Sol

## Status

Two platform/transport questions are closed. The exact fused-query and
long-context paged-top-k discriminator remains in progress.

| Proof | Result | Decision |
|---|---|---|
| Directed/concurrent CUDA peer matrix | PASS, 16/16 records | Cross-root `NODE` fabric is a real platform bottleneck; rank reordering alone cannot remove it |
| Explicit INT8 DMA versus NCCL | PASS | Keep the DMA crossover; DMA is 12–15x faster at 1,024/3,072 rows |
| Persistent-output INT8 DMA | PASS | Keep SparkInfer PR #76; no measurable persistent-output penalty |
| Persistent-output CUDA-graph replay | PASS | Keep SparkInfer PR #76; changed input produces finite, rank-identical replay output |
| Fused MLA query versus staged query | pending | Select fused or staged BF16-weight/FP8-output route from exact byte/overlap/timing result |
| DCP4 logical-index paged top-k | pending | Test the exact 32,767/32,768 two-level-fold transition before selecting/reverting the widened selector |

## Matched model control

The byte-identical v19 production image passed retrieval on the same CN4
hardware and configuration:

```text
depth=150000 pos=40% ctx=147369 cached=0 completion=85 finish=stop secs=609
retrieval=PASS (where=content) finalization=PASS
content='738216'
```

It also decoded at 57–66 tok/s. The current v20 candidate failed retrieval at
150k and decoded at 11.7 tok/s. Therefore the shared approximately 300 tok/s
prefill limit is a CN4 platform/collective problem, while the severe decode and
retrieval regressions are v20 software-path problems.

## Peer matrix

Artifact:

```text
harness/sol-proof-results/pcie-peer-matrix.jsonl
sha256 21ccc7e6619ab1210fadf043bea72cfcfbcb588cd705c356c33465ef1755fadf
```

| Pattern | Throughput |
|---|---:|
| GPU 0↔1 and 2↔3, one edge | 14.1–14.3 GB/s |
| GPU 0/1 → 2/3, one edge | approximately 10.1 GB/s |
| GPU 2/3 → 0/1, one edge | 4.6–6.2 GB/s |
| Four simultaneous within-pair edges | 6.88 GB/s/edge |
| Four simultaneous cross-pair edges | 1.82 GB/s/edge |
| Ring 0→1→2→3 | 1.81 GB/s/edge |
| Ring 0→1→3→2 | 1.57 GB/s/edge |

The two ring orders are both cross-root and both collapse. A rank permutation
cannot turn a four-GPU collective into an entirely within-switch operation.

Root-level read-only PCIe configuration inspection falsified the earlier ACS
hypothesis. Both Intel GPU root ports and all four PLX downstream ports report:

```text
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd-
```

Both root-to-PEX uplinks are 8 GT/s x16. ACS is not redirecting peer traffic.
The remaining platform mechanism is the inherent two-root-port/two-PEX `NODE`
path, shared Gen3 uplinks, and root/uncore contention.

## Collective matrix

First-pass explicit-output artifact:

```text
harness/sol-proof-results/pcie-collective-matrix.jsonl
sha256 207a0b74e1fd4ca3cf800d29797b422aa612e7d3e0e3abff646ad816670101c8
```

| Rows | Payload | INT8 DMA median | BF16 NCCL median | DMA/NCCL |
|---:|---:|---:|---:|---:|
| 64 | 0.92 MB | 366 us | 2,008 us | 0.182 |
| 256 | 3.67 MB | 539 us | 8,160 us | 0.066 |
| 1,024 | 14.68 MB | 2,175 us | 33,165 us | 0.066 |
| 3,072 | 44.04 MB | 8,894 us | 104,934 us | 0.085 |

All rows were finite and bit-identical across ranks. INT8-versus-BF16 mean
absolute error was approximately 0.00103.

The superseding persistent-output run also passed all explicit/default/graph
forms. At 1,024/3,072 rows, `dma_default_over_nccl` was 0.0659/0.0844 and
`dma_default_over_explicit` was approximately 1.0. CUDA-graph replay after
changing input bytes was finite and rank-identical. Preserve Fable's final v2
artifact:

```text
pcie-collective-matrix-v2.jsonl
sha256 8f51e462ce953cdd84653d848e406a389704a5354e6159b47ef9c5c45bbeedf0
```

Conclusion: do not move prefill-sized collectives to NCCL and do not change the
6 MiB crossover. PR #76's persistent output is correct and effectively free.
The remaining prefill work is phase synchronization/ownership and a
topology-aware collective design, not transport dispatch.

## Pending exact functional proof

The final script pin is:

```text
harness/v20_decode_retrieval_microprobes.py
sha256 9785aae7c9d78c1df8b9c1ea1d88c9876b72e61b14a7855cb032c7497386eaa4
```

It imports `vllm._custom_ops` to reproduce serving's stable-extension
registration. Its query leg separates safe-BMM/staged quantization, fused-BF16
plus staged quantization, and direct fused-FP8 output, reporting byte,
retrieval-overlap, and graph-timing deltas for all three. Its top-k leg mirrors
the DCP4 logical-index route:

1. fixed 32,768-token paged scorer scratch;
2. 16,384-token pseudo-row slices via `extent_splits`;
3. exact slice top-k into a shared candidate table;
4. final `run_row_topk` remap through `output_gather_table`.

It tests 32,767, 32,768, 32,769 and the observed 150k/250k DCP4 local widths
at query row counts 1, 9, 16 and 32 under monotonic, close-clustered, random,
fold-dominant, and quantized-tie scores. This is the direct discriminator
because the two-level route begins at exactly 32,768 local tokens; the passing
50k global request remains below it, while the failing 150k request crosses
it. The row matrix includes first-token decode and the uneven MTP/DCP shape,
not only one full selector tile.
