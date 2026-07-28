### Fix candidate and mechanism update

The causal selector result is now packaged as draft SparkInfer PR
[local-inference-lab/sparkinfer#82](https://github.com/local-inference-lab/sparkinfer/pull/82).

The PR adds an explicit, fail-closed
`SPARKINFER_NSA_TOPK_SELECTION_POLICY=bounded_compat` mode. It preserves the
historical 8-bit/4,096-candidate selection semantics while keeping every
shared-memory write bounds-checked. Default `exact` behavior is unchanged.

#### Why this changes deep retrieval

A new position-distribution analysis compared exact and bounded selections on
the same captured production tensors. Across all four ranks:

- exact-only candidates: 151--173 per rank; **100% are in the final quarter**
  of the 85,932-token rank-local context;
- bounded-only candidates: 151--173 per rank; 130--152 are in the first half;
- exact-only candidates have strictly higher quantized-proxy scores.

So the v20 exact overflow rescan is working as implemented, but it
systematically moves roughly 7.4%--8.4% of the 2,048-entry sparse-attention
budget from older positions into the newest quarter.

There is no independent local/sliding window to make that recency allocation
free: `B12X_MLA_SPARSE` rejects `sliding_window`, the SparkInfer MSA path
rejects `window_left`, and sparse layers consume only the selected 2,048
indices. The model interleaves full and sparse layers, but an omitted old
token is still unavailable inside each sparse layer.

This resolves the apparent paradox: exact v20 is mathematically better for
the E4M3-query/FP8-key **proxy**, but that proxy ordering is not the same
objective as end-to-end model retrieval. The checkpoint's known-good quality
baseline used the historical bounded selection distribution.

#### End-to-end status

The explicit compatibility policy has already passed the frozen causal gate:

| Cell | Stock exact v20 | `bounded_compat` |
|---|---|---|
| 250k control | EXACT | EXACT |
| 350k seed 1 | ABSENT | EXACT |
| 350k seed 2 | ABSENT | EXACT |
| 350k seed 3 | ABSENT | EXACT |

All four compatibility responses finalized normally with content `738216`
and `cached_tokens=0`. The boot stayed healthy with a 507,612-token KV pool
at 460k, zero restarts, and no fatal signatures.

Evidence:

- selector-bias report SHA-256:
  `b09c41acc3690ee11284e87b8c6a2e5e4e0f721c22c938177ed93a19a0755357`;
- causal stage-1 summary:
  `5881aee96f98bdcbacef3fc9f13b8b90aa24b5074f51da8174c1587dafe97916`;
- causal stage-2 summary:
  `1d91c2445bf01e0b7bdbc9167085c4e1d4db393d7050359218f265e7ede9699`;
- final log:
  `81e03f3cdf9153ae5a8c38d23655ac719385ceb79d550ba48376b84fd365eb22`.

The remaining work is promotion qualification, not root-cause discovery:
run the complete 50k--475k cold ladder, then KV-memory and throughput gates.
The new policy should remain explicit until that suite passes.
