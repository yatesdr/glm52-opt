## 2026-07-27 update: explicit bounds-safe boundary policy recovers all frozen 350k failures

The `oldest_boundary` causal image completed the frozen end-to-end gate. It
preserves the useful old-history exploration measured in the working trace
without restoring the historical out-of-bounds/capacity accident.

### Implemented policy

For the 8-bit FP16 coarse score histogram:

1. retain every candidate in a bucket strictly above the Kth bucket;
2. enumerate members of the Kth bucket in logical-history order;
3. retain the oldest 4,096 boundary members through tile-ordered warp-prefix
   compaction;
4. refine that bounded pool by the full FP32 score key;
5. emit exactly K entries.

Every candidate write is bounds checked. The complete boundary population is
still counted for diagnostics. `exact` remains the default mathematical
control and `bounded_compat` remains diagnostic-only.

### Off-model evidence

The CPU reference reconstructed historical captured top-k sets at:

- four v19 production rows: `2017..2035 / 2048` (97.0%--98.7%);
- independent v20 bounded rows: `2018..2041 / 2048` (97.1%--99.3%).

Production-shaped CUDA replay at M=3,072 completed on all four ranks and
stayed within 0--3 entries per 2,048-entry row of the independent CPU
reference. Exact-policy replay remained `2048/2048`.

Pins:

```text
d07442767bf0cdd7f891204f717f7abd7db3d0b6f9ed7402010b3b040627a349
  harness/v20_indexer_boundary_policy_cpu_proof.py

4d89638440a1bab62d632a4ebbba3de2cdbda8bc5bcd5c8d559b940d8c45e42e
  tested sparkinfer/attention/nsa_indexer/tiled_topk.py

7e47e9acd6b6698a97ea217802ec65bb5cee3292
  SparkInfer implementation commit
```

### Causal image and live contract

```text
image:
  glm52-serve:v20-20260727-indexer-oldest-boundary-causal-r2
manifest:
  sha256:2463080ecbdd0109244b10bd1266fb7acc74e803c0d1a1a1252dfb3d6837b6fc
```

The live installed selector file matched the source hash above. Relevant
runtime settings were:

```text
SPARKINFER_NSA_TOPK_SELECTION_POLICY=oldest_boundary
KV_FP8_ROPE=1
F8_DMA=i8_ring
MAX_BATCHED_TOKENS=3072
MAX_MODEL_LEN=480000
TP=4 DCP=4 MTP=3
```

Boot gates:

- healthy, zero restarts;
- 500,992 KV tokens;
- maximum concurrency at 480,000 tokens: 1.04x;
- no memory-utilization increase beyond the matched candidate (`0.974`).

### Frozen causal result

All prompts were replayed byte-for-byte from the immutable freeze bundle.
Acceptance required `cached_tokens=0`, `finish_reason=stop`, and finalized
`content.strip() == "738216"` exactly.

| Cell | Stock | New verdict | Prompt tokens | Cached | Finish | Output | Time |
|---|---|---|---:|---:|---|---:|---:|
| 250k control | EXACT | EXACT / PASS | 245,497 | 0 | stop | 4 | 195 s |
| 350k-r1 | ABSENT | EXACT / PASS | 343,727 | 0 | stop | 4 | 290 s |
| 350k-r2 | ABSENT | EXACT / PASS | 343,727 | 0 | stop | 4 | 290 s |
| 350k-r3 | ABSENT | EXACT / PASS | 343,727 | 0 | stop | 4 | 290 s |

Verdict:

```text
CONFIRMED — every frozen stock-FAIL prompt recovered
```

Primary summary:

```text
fa835422f8708c7a294eb358bf2372bf9ad1f7f01bebc23d9ccb391434153b5e
  summary.json
```

### Evidence boundary and remaining work

This establishes end-to-end causality for the frozen failure set. It does not
yet promote the image.

Two items remain before the implementation is ready as a production/upstream
fix:

1. The stable boundary candidate pool is deterministic, but the inherited
   downstream full-key radix passes still use atomic candidate compaction.
   Two repeated operator replays differed by 0--2 entries per row. Replace
   those per-pass atomics and the final equal-key atomic decrement with the
   same stable tile/warp-prefix compaction, then require exact repeatability.
2. Run the final kernel through the randomized cold
   50k/150k/250k/300k/350k/475k ladder, KLD/quality, prefill/decode, and KV
   capacity gates.

The production proposal is therefore an explicit, GLM-specific,
server-static coarse-boundary selection contract. It is not the v19 overflow
implementation and should not silently replace exact selection for unrelated
checkpoints.
