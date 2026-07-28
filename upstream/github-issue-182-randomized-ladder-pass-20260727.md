## Causal selector passes the full randomized cold ladder through 475k

The explicit bounds-safe `oldest_boundary` selector has now passed the
randomized long-context ladder on a clean-cache boot, in addition to the
previous frozen 250k + 3x350k causal gate.

Configuration:

- causal image manifest:
  `sha256:2463080ecbdd0109244b10bd1266fb7acc74e803c0d1a1a1252dfb3d6837b6fc`
- selector source:
  `4d89638440a1bab62d632a4ebbba3de2cdbda8bc5bcd5c8d559b940d8c45e42e`
- `SPARKINFER_NSA_TOPK_SELECTION_POLICY=oldest_boundary`
- TP4 / DCP4 / MTP3, `MAX_BATCHED_TOKENS=3072`
- NVFP4 MLA KV, FP8 RoPE (`KV_FP8_ROPE=1`)
- rank-consistent, numerically lossy block-INT8 `i8_ring` DCP wire
- `MAX_MODEL_LEN=480000`

The harness constructs a fresh natural-language prefix for every request and
fails unless `cached_tokens=0`, the needle appears in non-empty finalized
`content`, `finish_reason=stop`, and arithmetic/coherence/degeneration side
checks all pass.

| Target | Actual prompt tokens | Cached | Completion | Final content | Verdict |
|---:|---:|---:|---:|---|---|
| 50k | 49,101 | 0 | 91 | `738216` | PASS |
| 150k | 147,276 | 0 | 81 | `738216` | PASS |
| 250k | 245,505 | 0 | 66 | `738216` | PASS |
| 300k | 294,620 | 0 | 67 | `738216` | PASS |
| 350k | 343,734 | 0 | 113 | `738216` | PASS |
| 475k | 466,495 | 0 | 75 | `738216` | PASS |

Pinned summary:

```text
073a90ac63617f8ffd795203211f51f2a103b43e50462e7d4687382abd00ee6d
  harness/cn4-evidence-archive/20260727/
  oldest-boundary-clean-ladder-v1/summary.json
```

The container remained at zero restarts. This fresh-cache boot exposed
498,432 KV tokens at a 480,000-token maximum. That is enough for the deepest
cell but sits 1,568 tokens below our separate 500,000-token promotion floor;
capacity is therefore still an integration gate, not part of the selector
quality claim.

I have also ported the same accepted implementation to current SparkInfer
master as a two-file draft (`tiled_topk.py` plus the policy parser unit test)
without the historical `bounded_compat` mode. A fail-closed derived image on
the current topology-calibrated v20 base (`0c79e41` / SparkInfer `e603f74`)
is built and byte-verified:

```text
sha256:43e5a48781ee5cf40a92cc494749b21306b72280bd1a875721a45422323f2599
```

That current-v20 integration image still needs its clean boot and promotion
gates. The result above establishes the causal mechanism; it is not yet a
claim that the newer integration is production-ready.
