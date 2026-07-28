### Clean upstream draft PR opened; general quality and baseline performance

The minimal SparkInfer change is now reviewable as:

https://github.com/local-inference-lab/sparkinfer/pull/84

It is a one-commit draft against current `master` and changes only:

- `sparkinfer/attention/nsa_indexer/tiled_topk.py`;
- `tests/attention/test_nsa_topk_selection_policy.py`.

`exact` remains the default. The new `oldest_boundary` policy is explicit,
server-static, bounds checked, and contains none of the historical overflow
behavior.

After the 6/6 current-v20 cold ladder, the same process also passed:

- a separate 200k needle placed at 95% depth (`592847`);
- exact nested numeric JSON round-trip.

Cold prefill baseline on CN4:

```text
8k:   1,460 tok/s server-side, cached=0
55k:  1,501 tok/s server-side, cached=0
```

Decode baseline (`256` requested tokens each):

| Concurrency | Aggregate tok/s | MTP acceptance |
|---:|---:|---:|
| 1 | 55.06 | 0.5033 |
| 4 | 109.95 | 0.5547 |
| 8 | 144.58 | 0.5725 |
| 16 | 180.50 | 0.5765 |

All requests completed without errors and produced finalized output. The
55k prefill result is essentially equal to the previously shipped v19
full-context figure (~1,509 tok/s), so no prefill penalty is visible from the
selector fix. C1 decode remains only MTP0-class because roughly 50% MTP3
acceptance does not repay its overhead; that is a separate performance-tuning
item, not part of the retrieval patch.

Remaining promotion work is the established KLD/reference suite and a matched
stock-current-image throughput A/B. The retrieval, current-base integration,
480k capacity, and general API-quality gates are now passing.
