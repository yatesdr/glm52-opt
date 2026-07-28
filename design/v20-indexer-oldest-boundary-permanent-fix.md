# v20 long-context selector: explicit oldest-boundary policy

Date: 2026-07-27

Status: off-model proof, frozen causal gate, and randomized cold 50k--475k
ladder passed; an experimental stable-radix cleanup was rejected; integration
qualification on the current v20 base is pending

## Problem statement

The v20 `exact` sparse-indexer selector is memory safe and exact for its
quantized score field, but it changes the model's long-context trajectory.
On the frozen 350k failure, exact selection does not grow the needle-local
cluster early enough: all three ticket-value tokens appear together only at
layer 74. The known-working bounded selector has already selected the full
value by layer 38 and returns `738216`.

The relevant historical behavior was not the out-of-bounds write itself. An
8-bit coarse score histogram selected all entries above the Kth coarse bucket,
then a capacity-limited 4,096-entry buffer preferentially retained older
entries from the boundary bucket. That supplied score-aware old-history
exploration which this checkpoint uses.

## Replacement contract

`oldest_boundary` makes that behavior explicit and bounds safe:

1. compute the same 8-bit FP16 coarse score bucket as the historical selector;
2. retain every candidate in a bucket strictly above the Kth bucket;
3. enumerate members of the Kth bucket in logical-history order;
4. retain the oldest 4,096 boundary members;
5. refine those candidates by their full FP32 score key and emit exactly K
   entries.

The buffer population is performed by tile-ordered warp-prefix compaction.
No write can exceed the candidate allocation, and the full boundary
population remains available for diagnostics.

`exact` remains the general-purpose mathematical control.
`bounded_compat` remains diagnostic-only and is not the production proposal.

## Evidence before model boot

The CPU reference reconstructs captured historical output sets at:

- v19 production rows: 97.0%--98.7%;
- independent v20 bounded rows: 97.1%--99.3%.

The initial production-shaped CUDA replay stayed within 0--3 entries of the
CPU reference per 2,048-entry row.

A selector-only discriminator now reconstructs the captured production score
row once on CPU and feeds it directly into the production tiled-top-k geometry.
It passes 4/4 repetitions at 2,048/2,048 set identity both against the first
GPU run and the CPU `oldest_boundary` oracle. A full learned-indexer replay
still moves 1--2 entries between processes, while an unchanged `exact`-policy
control is repeatable. This localizes that residual sensitivity outside the
tiled-top-k policy itself, in the learned-score/integration path. It is tracked
separately and is not represented as selector nondeterminism.

Relevant pins:

```text
d07442767bf0cdd7f891204f717f7abd7db3d0b6f9ed7402010b3b040627a349
  harness/v20_indexer_boundary_policy_cpu_proof.py

4d89638440a1bab62d632a4ebbba3de2cdbda8bc5bcd5c8d559b940d8c45e42e
  tested tiled_topk.py

7e47e9acd6b6698a97ea217802ec65bb5cee3292
  SparkInfer implementation commit

2463080ecbdd0109244b10bd1266fb7acc74e803c0d1a1a1252dfb3d6837b6fc
  causal image manifest

e3e668459b3657d8c34fb46e61aa04ce7ee333553749142f2e1798fdcabbe202
  selector-only determinism result

42769970adad6ee05c9ea86794464cb7011932122d8d4d9d46e0afc4d23420ec
  harness/v20_oldest_boundary_warmup_shape_probe.py

6d32434593a932026ad16fdde2aded4f5e1b45c584cadc52452edf4397e6b23d
  causal-image 3072-row warmup result

7c015a03d0573846a037b7bb7fcea4a6810b74e7c50f6fe6dc7d443327a90ddb
  rejected-cleanup 3072-row warmup failure
```

## Frozen causal result

The byte-pinned cold 250k control and all three byte-pinned cold 350k stock
failures returned exact finalized content:

| Cell | Stock | New | Cached | Finish | Content |
|---|---|---|---:|---|---|
| 250k control | EXACT | PASS | 0 | stop | `738216` |
| 350k-r1 | ABSENT | PASS | 0 | stop | `738216` |
| 350k-r2 | ABSENT | PASS | 0 | stop | `738216` |
| 350k-r3 | ABSENT | PASS | 0 | stop | `738216` |

The image booted with 500,992 KV tokens at a 480,000-token maximum and zero
restarts. Primary result:

```text
fa835422f8708c7a294eb358bf2372bf9ad1f7f01bebc23d9ccb391434153b5e
  harness/cn4-evidence-archive/20260727/
  indexer-oldest-boundary-causal-r2/summary.json
```

## Clean-boot randomized result

A fresh-cache boot of the same causal image completed the full randomized
cold ladder. Every cell returned finalized content `738216`, stopped normally,
reported `cached_tokens=0`, and passed the arithmetic, coherence, and
degeneration checks:

| Target | Actual prompt tokens | Completion tokens | Elapsed | Verdict |
|---:|---:|---:|---:|---|
| 50k | 49,101 | 91 | 40.65 s | PASS |
| 150k | 147,276 | 81 | 112.07 s | PASS |
| 250k | 245,505 | 66 | 192.84 s | PASS |
| 300k | 294,620 | 67 | 240.44 s | PASS |
| 350k | 343,734 | 113 | 292.33 s | PASS |
| 475k | 466,495 | 75 | 431.83 s | PASS |

Primary result:

```text
073a90ac63617f8ffd795203211f51f2a103b43e50462e7d4687382abd00ee6d
  harness/cn4-evidence-archive/20260727/
  oldest-boundary-clean-ladder-v1/summary.json
```

This clean-cache boot exposed 498,432 KV tokens at a 480,000-token maximum.
That is sufficient for the full ladder and proves the retrieval mechanism, but
it is 1,568 tokens below the separate 500,000-token promotion floor. An earlier
boot of the same image exposed 500,992. Capacity therefore remains a
current-v20 integration gate rather than being folded into the selector's
quality claim.

## Rejected deterministic-cleanup experiment

Commits `c1ffc029` and `87b61b04` attempted to apply stable tile/warp-prefix
compaction to every full-key radix pass and serialize the final output
allocator. That experiment is **not** part of the production proposal.

It passed the single-row 85,932-token selector proof, but failed the
server-profile geometry:

- known-good causal image, 3,072 ramped rows through length 3,072:
  PASS, valid counts and bounds;
- stable-radix image, identical input:
  `CUDA_ERROR_INVALID_ADDRESS_SPACE`;
- the full server boot surfaced the same asynchronous illegal access during
  profile warmup on all four ranks;
- NVIDIA memcheck observed invalid global reads inside the tiled selector
  kernel near rows 2,988--2,997.

This is exactly the shape gap that the earlier single-row proof did not cover.
The cleanup images are rejected and the production branch is pinned at
`d4385494` / `7e47e9ac`, whose selector source is
`4d89638440a1bab62d632a4ebbba3de2cdbda8bc5bcd5c8d559b940d8c45e42e`.

The direct selector proof establishes deterministic membership for fixed
scores. The full learned-indexer sensitivity described above is deliberately
kept outside this PR's claim boundary; folding it into the selector change
would conceal a second mechanism rather than fix it.

## Promotion sequence

1. Frozen cold 250k exact-answer control.
2. All three frozen cold 350k stock failures must return exact finalized
   content `738216`.
3. Randomized cold 50k/150k/250k/300k/350k/475k ladder, including both
   `content` and reasoning-field audit.
4. Cross-run selector repeatability, with the 3,072-row warmup-shape safety
   probe as a mandatory pre-boot gate.
5. KLD/quality suite.
6. Prefill and decode benchmarks versus the same v20 base.
7. KV capacity at 480k, with a hard floor of 500,000 tokens.

Only after all seven gates pass should `oldest_boundary` be selected by a
GLM-specific server-static configuration. It must not silently replace
`exact` for unrelated checkpoints.

## Latest-v20 integration status

The clean PR source was forward-applied to the 2026-07-26 topology-calibrated
base without carrying any other overlay:

```text
base:
  voipmonitor/vllm@sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0
  vLLM 0c79e41db4 / SparkInfer e603f74bb6

derived image:
  glm52-serve:v20-20260726-oldest-boundary-pr-candidate
  sha256:43e5a48781ee5cf40a92cc494749b21306b72280bd1a875721a45422323f2599

installed selector:
  b15bab73f1fcd6434f712f6fc99ec5369104969cb9157ae473926bf40d72e23b
```

The model-free 3,072-row production-shape GPU probe passes on this image and
is byte-identical to the causal-image result:

```text
6d32434593a932026ad16fdde2aded4f5e1b45c584cadc52452edf4397e6b23d
```

The first model boot used GMU `0.974`, MTP3, FP8 RoPE, `i8_ring`, MNBT 3,072,
and max length 480,000. It completed model load, warmup, graph profiling, and
graph capture, then correctly failed the KV-capacity check:

```text
available KV memory: 3.08 GiB
required for 480,000: 3.57 GiB
estimated maximum:    412,928
```

This was not a selector allocation or quality failure. The new base changes
MRV2 graph accounting from a disposable pool with only the incremental cost
charged to a reusable global pool with the complete 1.03-GiB high-water mark
charged. Approximately 0.87 GiB was retained in that reusable pool. The boot
log calculated GMU `0.9848` as the equivalent memory posture to the former
`0.974` behavior. The second integration boot used that exact calculated
value; all quality-sensitive settings and image bytes remained unchanged.

That boot passed:

```text
health:             healthy
restart count:       0
KV pool:             545,280 tokens
max model length:    480,000
max concurrency:     1.14x
```

The complete frozen causal gate then passed on this newest base:

| Cell | Prompt tokens | Cached | Finish | Completion | Final content |
|---|---:|---:|---|---:|---|
| 250k control | 245,497 | 0 | stop | 4 | `738216` |
| 350k-r1 | 343,727 | 0 | stop | 4 | `738216` |
| 350k-r2 | 343,727 | 0 | stop | 4 | `738216` |
| 350k-r3 | 343,727 | 0 | stop | 4 | `738216` |

Primary result:

```text
dda7bddd33919d0947bcf45e0731c7fe07e1d4918944781fca9928cafe1d18f6
  harness/cn4-evidence-archive/20260727/
  current-v20-oldest-boundary-causal-v1/summary.json
```

The same live process then passed the complete randomized cold ladder:

| Target | Actual prompt tokens | Cached | Completion | Elapsed | Verdict |
|---:|---:|---:|---:|---:|---|
| 50k | 49,097 | 0 | 93 | 33.60 s | PASS |
| 150k | 147,275 | 0 | 99 | 107.33 s | PASS |
| 250k | 245,506 | 0 | 66 | 193.73 s | PASS |
| 300k | 294,620 | 0 | 66 | 241.03 s | PASS |
| 350k | 343,735 | 0 | 64 | 292.17 s | PASS |
| 475k | 466,493 | 0 | 71 | 432.71 s | PASS |

Every response finalized `738216` with `finish_reason=stop`; arithmetic,
coherence, and degeneration checks also passed at every depth. Primary result:

```text
b855f1febae880a6ae146797fbf37707e3ea02bccd213578d41ec5ba19ae6268
  harness/cn4-evidence-archive/20260727/
  current-v20-oldest-boundary-clean-ladder-v1/summary.json
```

The actual production graph capture consumed 0.19 GiB versus the conservative
1.03-GiB pre-allocation estimate. That 0.83-GiB estimator gap is a separate
capacity-reclaim opportunity; it is not part of the selector patch.

## Final live acceptance audit

The post-qualification audit was performed after the frozen causal gate, full
randomized ladder, extended quality gate, and throughput baselines. It
confirmed the same live process throughout:

```text
container:
  0c5c132661ca189068e0adf5ebc0d55d5bada7bb9504bd0752a9c109d472cd47
image:
  sha256:43e5a48781ee5cf40a92cc494749b21306b72280bd1a875721a45422323f2599
state:
  running / healthy / restart count 0
installed selector:
  b15bab73f1fcd6434f712f6fc99ec5369104969cb9157ae473926bf40d72e23b
active policy:
  oldest_boundary / coarse bits 8 / boundary cap 4096
fatal signatures:
  0
```

The API continued to report `max_model_len=480000`. Runtime configuration was
TP4/DCP4/MTP3, MNBT 3,072, NVFP4 MLA KV, FP8 RoPE, and `i8_ring`.

Additional quality and performance results from that same process:

```text
200k needle at 95% depth: PASS, exact 592847
nested numeric JSON echo: PASS

cold prefill:
  8k   1,460 tok/s, cached=0
  55k  1,501 tok/s, cached=0

decode aggregate:
  C1    55.06 tok/s
  C4   109.95 tok/s
  C8   144.58 tok/s
  C16  180.50 tok/s
```

C1 MTP acceptance was 0.5033, so C1 decode remains a separate optimization
target. It does not weaken the retrieval result or belong in the selector PR.
The formal reference-KLD and matched stock-image performance A/B are promotion
and optimization follow-ups rather than unresolved root-cause questions.
