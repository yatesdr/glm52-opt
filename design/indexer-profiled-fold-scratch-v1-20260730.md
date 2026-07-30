# Profile-accounted exact-fold scratch v1

Date: 2026-07-30  
Status: CPU/CUDA operator gates and full-model validation passed

## Problem

SparkInfer's adaptive exact sparse-indexer fold selects a fast two-level
candidate reduction when the request geometry is within
`SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB` (256 MiB by default). The cap
bounds the allocation size but the implementation allocated `fold_values`,
`fold_indices`, and `fold_lengths` with request-time `torch.empty`/`full`.
Those bytes were therefore absent from vLLM's memory profile and KV-pool
sizing.

CN4 reproduced the resulting engine-killing failure twice on the stock r9
implementation:

| candidate | GPU KV pool | failing row | request | physically free |
|---|---:|---:|---:|---:|
| fused candidate 3 | 551,680 | 250k | 72 MiB | 56.69 MiB |
| fused candidate 5 | 535,040 | 350k | 120 MiB | 116.69 MiB |

The exact streaming-carry control passed 250k, causally isolating the late
parallel-fold allocation. It was too slow to be the optimized production
answer.

## Contract

The configured candidate budget is now one rank-invariant contract shared by
planning and execution:

1. `fold_policy.py` parses the mode and byte budget once.
2. The paged-indexer scratch planner appends value, index, and length regions
   to its existing single scratch allocation.
3. vLLM sees that larger scratch during profile work and sizes the KV pool
   around it.
4. Runtime planning selects the two-level fold only when its exact candidate
   byte count fits the same budget.
5. The fold borrows views from the reserved scratch; it performs no
   request-time allocation.
6. Forced mode fails closed if a shape exceeds the reservation. It cannot
   silently restore the unsafe late allocation.

The default 256 MiB budget reserves 16,384 candidate rows at top-k 2,048. The
observed 350k geometry requires 15,360 rows and therefore stays on the fast
parallel path. Wider geometries that exceed the budget use the existing exact
streaming carry.

No scorer arithmetic, selected indices, KV bytes, record ABI, collective
route, or cache identity changes.

## Gates

Image:

`ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-r9-exl3-tr3-325-mixk-v6-fused-m8-foldscratch-candidate4-r2-20260730`

Manifest:

`sha256:a2b233f60329f22c7e541406b8972b3d8dc46c5c7d104d0aa3b4fee26374870e`

Source pins:

| file | SHA-256 |
|---|---|
| `fold_policy.py` | `44138ebd939b56fa732322f84a89cec47d7b1190ae25331500c091db6f5bee88` |
| `paged.py` | `c1f01f3fbc7731e82f69af7d9ec9cf100bf9650a7728ad3b1383e49a85b34697` |
| `scratch.py` | `236d287580a07b618ae2385184a9cb970b78bc12751104339b2d8f401e6aba3a` |

Completed:

- fail-closed base and overlay byte checks;
- `py_compile`;
- 11 focused CPU planner/reservation tests;
- CUDA exactness for the reserved two-level fold;
- CUDA exactness for the zero-budget streaming fallback.

Full-model acceptance order:

1. model boot and reported KV pool;
2. MTP3 C32;
3. immediate cold 55k;
4. immediate cold 250k;
5. immediate cold 350k;
6. cold 475k;
7. remaining quality and performance matrix.

The order deliberately loads the allocator before deep prefill. A clean-boot
deep request alone is not a fit proof.

All full-model gates passed with the profiled implementation present. The
final balanced profile selects the exact streaming path
(`SPARKINFER_INDEXER_TWO_LEVEL_FOLD=0`) to reclaim the 256 MiB fold
reservation; it then passed the ordered C32 -> cold-350k fragmentation gate
and the complete 50k/250k/350k/475k retrieval ladder. The profiled parallel
fold remains available for deployments that prefer its prefill behavior and
budget the reservation explicitly.
