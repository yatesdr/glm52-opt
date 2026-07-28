# Draft PR: preserve oldest coarse-boundary candidates for calibrated indexers

Closes the SparkInfer portion of
`local-inference-lab/vllm#182`.

## Problem

The bounds-safe exact long-context selector is exact for its quantized score
field, but GLM-5.2's learned sparse indexer was calibrated with a different
coarse-boundary exploration pattern. On frozen 350k prompts, the current exact
selector does not select the complete needle-local token cluster until layer
74. The known-working behavior selects it by layer 38.

The useful behavior was not the historical out-of-bounds write. It was:

1. an 8-bit FP16 coarse score threshold;
2. all candidates strictly above the Kth bucket;
3. a 4,096-candidate preference for older positions in the boundary bucket;
4. full-key refinement of that bounded candidate set.

Changing the candidate distribution changed the model trajectory even though
the newer selector was mathematically cleaner in isolation.

## Change

Add an explicit, server-static
`SPARKINFER_NSA_TOPK_SELECTION_POLICY=oldest_boundary` policy:

- retain every candidate above the Kth 8-bit coarse bucket;
- enumerate the boundary bucket in logical-history order;
- materialize at most its oldest 4,096 candidates with tile/warp prefix
  compaction;
- refine those candidates with the existing full-score radix selector;
- keep all writes bounds checked and still emit exactly `topk` entries.

`exact` remains the default. This PR deliberately does **not** include the
historical `bounded_compat` diagnostic or any unsafe overflow behavior.

The policy is server static because changing candidate/cache semantics within
a process would mix incompatible trajectories.

## Evidence

CPU reconstruction of captured historical selections:

- v19 production rows: 97.0%--98.7% set overlap;
- independent v20 compatibility rows: 97.1%--99.3%.

Frozen end-to-end causal gate on the target NF3/NVFP4-KV/FP8-RoPE stack:

| Cell | Stock | New | Cache | Finish | Final content |
|---|---|---|---:|---|---|
| 250k control | exact | pass | 0 | stop | `738216` |
| 350k seed 1 | absent | pass | 0 | stop | `738216` |
| 350k seed 2 | absent | pass | 0 | stop | `738216` |
| 350k seed 3 | absent | pass | 0 | stop | `738216` |

The causal image served a 480,000-token maximum with a 500,992-token KV pool
and zero restarts.

The subsequent fresh-cache randomized ladder passed at all six target depths:
50k, 150k, 250k, 300k, 350k, and 475k. The actual deepest prompt contained
466,495 tokens. Every cell reported `cached_tokens=0`, returned finalized
`738216` with `finish_reason=stop`, and passed arithmetic, coherence, and
degeneration side checks.

Safety coverage includes a model-free 3,072-row profile geometry
(`lengths=1..3072`, `topk=2048`, `block_q=32`, `block_k=256`). The proposed
implementation passes with every selected index in bounds and the correct
valid count for every row.

An experimental follow-up that tried to stabilize every later radix pass was
rejected: the same warmup-shape probe reproduced invalid global reads and a
server warmup crash. Those commits and that code are not in this PR.

The clean PR source was then forward-applied, without any other code overlay,
to the newer topology-calibrated v20 base
`0c79e41db4` / SparkInfer `e603f74bb6`. It passed the same 3,072-row safety
probe, booted healthy at a 480,000-token maximum with a 545,280-token KV pool
and zero restarts, and passed the complete frozen causal gate:

| Cell | Prompt tokens | Cache | Finish | Final content |
|---|---:|---:|---|---|
| 250k control | 245,497 | 0 | stop | `738216` |
| 350k seed 1 | 343,727 | 0 | stop | `738216` |
| 350k seed 2 | 343,727 | 0 | stop | `738216` |
| 350k seed 3 | 343,727 | 0 | stop | `738216` |

The pinned summary SHA-256 is
`dda7bddd33919d0947bcf45e0731c7fe07e1d4918944781fca9928cafe1d18f6`.

The same live current-v20 process then passed the randomized cold ladder at
50k, 150k, 250k, 300k, 350k, and 475k. The deepest actual prompt was 466,493
tokens. Every cell had `cached_tokens=0`, finalized `738216` with
`finish_reason=stop`, and passed its arithmetic, coherence, and degeneration
checks. The pinned current-v20 ladder summary is
`b855f1febae880a6ae146797fbf37707e3ea02bccd213578d41ec5ba19ae6268`.

Primary pins and artifacts are recorded in
`design/v20-indexer-oldest-boundary-permanent-fix.md` in the companion
reproduction repository.

## Validation

- policy parsing unit test;
- Python compilation and diff checks;
- frozen single-row production-score selector replay;
- 3,072-row model-free warmup-shape safety gate;
- frozen cold 250k control plus 3x350k causal gate;
- randomized cold 50k/150k/250k/300k/350k/475k ladder.

Before promotion, the current-v20 integration image must additionally pass the
KLD/quality suite and throughput comparison. The randomized cold ladder and
the 500,000-token KV-capacity floor at a 480k maximum now pass.
