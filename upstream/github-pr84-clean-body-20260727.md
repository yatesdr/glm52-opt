## Summary

Adds an opt-in `oldest_boundary` selection policy to the NSA tiled top-k
indexer. The policy provides deterministic, bounds-safe handling of candidates
that share the coarse Kth-score bucket.

`exact` remains the default policy.

Closes the SparkInfer portion of
[`local-inference-lab/vllm#182`](https://github.com/local-inference-lab/vllm/issues/182).

## Background

GLM-5.2 uses a learned sparse indexer to choose 2,048 historical token
positions for attention. A token that is not selected cannot contribute to
that sparse-attention layer, so selector membership is directly observable as
long-context retrieval quality.

The reproduction uses a deterministic long document of financial-record
prose. It inserts one sentence stating that the Facility 27 maintenance ticket
number is `738216` at 40% of the document, then asks for that number at the end.
Each prompt and its chat-rendered token IDs are SHA-256 pinned. Requests use
temperature 0, thinking disabled, and a 2,000-token output limit. A request
passes only when:

- `cached_tokens == 0`;
- `finish_reason == "stop"`;
- finalized `content.strip() == "738216"`.

On the GLM-5.2 NF3/NVFP4-MLA-KV/FP8-RoPE configuration, the current v20
`exact` policy passes the pinned 245,497-token control and fails all three
pinned 343,727-token prompts. The failed requests stop normally and produce
coherent finalized output, but the target is absent.

Layer-level captures of the same 350k inputs show that the default selector
does not contain the complete three-token target value together until layer
74. With `oldest_boundary`, the complete target was selected by layer 38 and
the final answer was restored in all three cases.

## Why the policies differ

The indexer first reduces scores to a coarse radix bucket and then refines the
candidates around the Kth bucket. At long context, many positions can share
that boundary bucket. The choice of which boundary candidates enter full-score
refinement changes the 2,048-token sparse-attention set.

`exact` resolves the full quantized score field. `oldest_boundary` limits a
large tied coarse bucket by logical position before applying full-score
refinement. This produces a different, deterministic candidate set for the
learned indexer. This PR makes that choice explicit and checkpoint-selectable
without changing the default selector.

## Selection contract

With `SPARKINFER_NSA_TOPK_SELECTION_POLICY=oldest_boundary`, the selector:

1. computes an 8-bit coarse score histogram;
2. retains every candidate in a bucket strictly above the Kth bucket;
3. enumerates candidates in the Kth bucket in ascending logical-position
   order;
4. retains the first 4,096 candidates from that boundary bucket;
5. applies the existing four-pass full 32-bit radix refinement;
6. emits exactly `topk` indices.

The boundary scan uses tile-ordered warp-prefix compaction. It records the full
boundary population while writing only ordinals in `[0, 4096)`, so candidate
writes cannot exceed the allocated buffer.

## Configuration and scope

```text
SPARKINFER_NSA_TOPK_SELECTION_POLICY=exact            # default
SPARKINFER_NSA_TOPK_SELECTION_POLICY=oldest_boundary  # opt-in
```

The value is normalized and validated at module initialization. Unsupported
values fail immediately with `ValueError`.

The policy is included in the compilation-cache key. It is server-static and
must not be changed inside a running process.

This PR does not change `exact`, does not select `oldest_boundary`
automatically for other checkpoints, and does not add an overflow or
out-of-bounds compatibility path.

## Validation

### Frozen end-to-end retrieval gate

All requests were cold (`cached_tokens=0`) and required finalized content
`738216` with `finish_reason=stop`.

| Cell | Default `exact` | `oldest_boundary` |
|---|---|---|
| 250k control | pass | pass |
| 350k seed 1 | target absent | pass |
| 350k seed 2 | target absent | pass |
| 350k seed 3 | target absent | pass |

The current-v20 result summary is pinned by SHA-256:

```text
dda7bddd33919d0947bcf45e0731c7fe07e1d4918944781fca9928cafe1d18f6
```

### Randomized cold ladder

The same current-v20 process passed 50k, 150k, 250k, 300k, 350k, and 475k.
The deepest prompt contained 466,493 tokens. Every cell:

- reported `cached_tokens=0`;
- returned finalized content `738216`;
- stopped normally;
- passed arithmetic, coherence, and degeneration checks.

The ladder summary is pinned by SHA-256:

```text
b855f1febae880a6ae146797fbf37707e3ea02bccd213578d41ec5ba19ae6268
```

### Selector and capacity checks

- Captured selector rows reconstruct at 97.0%--99.3% set overlap.
- A model-free 3,072-row production-shape test
  (`lengths=1..3072`, `topk=2048`, `block_q=32`, `block_k=256`) reports the
  correct valid count for every row and no out-of-range index.
- The current-v20 integration image serves a 480,000-token maximum with a
  545,280-token KV pool and zero restarts.
- Unit tests cover the default, normalization, both supported values, and
  rejection of unsupported values.

### Matched KLD comparison

Three runs per policy were compared with the same pinned BF16 reference logits
(2,047 token positions):

| Policy | Runs | Mean KLD ± SD | Min | Max |
|---|---:|---:|---:|---:|
| `exact` | 3 | 0.15823696 ± 0.00468419 | 0.15539664 | 0.16364348 |
| `oldest_boundary` | 3 | 0.16044075 ± 0.00297924 | 0.15700885 | 0.16236257 |

The paired `oldest_boundary - exact` deltas were `-0.00663463`,
`+0.00655420`, and `+0.00669180`. Their mean was `+0.00220379` (+1.39%
of the exact-policy mean), with sample SD `0.00765461`. The mixed signs and
variance larger than the mean delta do not show a large shallow
distribution-level regression at n=3.

This is a 2,048-token no-regression gate, not a selector-sensitive comparison:
the selector budget is also 2,048, so both policies select every eligible
token. Deep-context efficacy is covered by the frozen 350k gate and randomized
50k–475k ladder.

## Qualification status

The long-context retrieval, selector-safety, and KV-capacity gates pass on the
current v20 base (`vLLM 0c79e41db4`, SparkInfer `e603f74bb6`). The matched
three-run shallow KLD comparison passes without a large regression.
Prefill/decode performance comparison remains integration-qualification work
and is not claimed as complete by this PR.

The pinned image, A/B configuration, test procedure, and evidence links are
available in the
[community validation specification](https://gist.github.com/yatesdr/a2e84aa3171ee0b355649704f04f96a8).
