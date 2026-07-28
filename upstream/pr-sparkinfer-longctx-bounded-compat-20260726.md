## Summary

Add an explicit `SPARKINFER_NSA_TOPK_SELECTION_POLICY=bounded_compat`
policy for sparse-MLA deployments whose checkpoints were validated against
the historical bounded radix selector.

The default remains `exact`. Unknown values fail closed at import time.

Fixes the SparkInfer component of
[local-inference-lab/vllm#182](https://github.com/local-inference-lab/vllm/issues/182).

## Root cause

The v20 exact-overflow changes (`1012199e`, later optimized by `83a58444`)
changed the selected sparse-attention set at long context. On captured
production tensors:

- exact v20 matches the quantized-proxy top-k 2,048/2,048;
- historical bounded selection matches 1,872--1,896/2,048;
- the changed exact-only candidates are all in the final quarter of context;
- the bounded-only candidates are predominantly in the first half.

This model has no separate sliding/local window in sparse layers. The exact
proxy selector therefore reallocates part of the only 2,048-entry sparse
attention budget away from older positions. Exactness for the
E4M3-query/FP8-key proxy did not preserve end-to-end retrieval quality.

## Change

`bounded_compat` explicitly selects the historical:

- 8-bit coarse radix;
- 4,096-entry candidate buffer;
- bounded refinement without the exact overflow rescan.

All shared-memory writes remain bounds-checked. The selection policy is part
of the compile-cache key. No behavior changes unless the new policy is
explicitly enabled.

## Reproduction and causal evidence

Frozen production posture:

- GLM-5.2 NF3 hybrid weights;
- TP4/DCP4/MTP3, MNBT 3,072;
- NVFP4 MLA KV, FP8 RoPE;
- `i8_ring`;
- cold cache (`cached_tokens=0`);
- identical rendered prompts, token IDs, and seeds.

Results:

| Cell | Stock exact v20 | `bounded_compat` |
|---|---|---|
| 250k control | EXACT | EXACT |
| 350k seed 1 | ABSENT | EXACT |
| 350k seed 2 | ABSENT | EXACT |
| 350k seed 3 | ABSENT | EXACT |

All recovered responses finalized normally with content `738216`. The
discriminator boot remained healthy with a 507,612-token KV pool at 460k,
zero restarts, and no illegal-access, cuBLAS, EngineDead, OOM, traceback,
assertion, or worker-died signatures.

An independent cold generalization ladder on the same process also passed:

| Rendered context | Result | Cached tokens |
|---:|---|---:|
| 49,100 | EXACT | 0 |
| 147,275 | EXACT | 0 |
| 294,619 | EXACT | 0 |
| 441,964 | EXACT | 0 |

The trace-free production candidate then passed the established deepest cold
gate at a 480,000-token admission limit:

| Rendered context | Result | Cached tokens | Finish | Content |
|---:|---|---:|---|---|
| 466,493 | EXACT | 0 | `stop` | `738216` |

That candidate was derived from the clean `5517197/be0edca` release image and
contained only PR #80 plus this PR. It exposed a 500,992-token KV pool, remained
healthy with zero restarts, and had zero fatal signatures.

Full reproduction, frozen prompt hashes, boundary traces, learned-tensor
replay, and evidence hashes are in
[vLLM issue #182](https://github.com/local-inference-lab/vllm/issues/182#issuecomment-5084645265).

## Validation

- `python -m py_compile` passes for the changed module and policy test.
- A derived v20 image compiled the new policy, completed production and
  speculator graph capture, and passed the frozen causal gate above.
- Existing exact behavior remains the default and its kernel branches are
  unchanged.

The focused pytest was not run on the authoring Mac because that environment
does not have pytest installed; CI should run
`tests/attention/test_nsa_topk_selection_policy.py`.

## Scope

This PR provides a visible compatibility policy, not a claim that bounded
selection should become the universal default. A future default should be
deterministic and justified by end-to-end model quality rather than only
quantized-proxy exactness.
