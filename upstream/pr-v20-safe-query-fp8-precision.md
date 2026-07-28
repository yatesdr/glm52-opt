# Proposed title

[GG] fix(mla): keep FP8 query reductions in FP32

> **Retrieval claim withdrawn — 2026-07-25.** An exact PEDANTIC binary
> rewind preserved the old operator fingerprints but still produced the same
> cold 100k empty-finalization MISS. This proposal may still be useful as a
> numeric-correctness improvement, but it is not the v20 long-context
> retrieval fix and must not be bundled into that production qualification.

Local preparation:

```text
branch: fix/v20-safe-query-accurate-reduction-clean-20260725
head:   0eb51f992
base:   83a1f7f7d (build/gilded-gnosis-v20-dcp-final-20260725)
files:  5
status: local only; static proof passed, GPU/model gates pending
```

This clean branch is based on Festr's current v20 DCP-final integration and
contains no production overlays or unrelated #171 changes.

## Summary

Keep tensor-core-eligible `CUBLAS_COMPUTE_32F` for the safe MLA query BMM.
When its BF16 output will immediately be requantized to FP8, set
`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` for that call so split-K
and intermediate reductions cannot accumulate in the BF16 output type.

## Why

Commit `992b874cf` changed the safe query BMM from PEDANTIC to regular FP32
compute to recover tensor-core prefill performance. Its test accepts
`rtol=0.05, atol=0.05` at the BF16 output. In the compact quantized-KV route,
that output immediately crosses a static FP8 quantizer, where small
accumulation differences become different query bytes and may alter sparse
retrieval decisions.

NVIDIA documents that mixed-precision GEMM heuristics can select an
output-type reduction: intermediate results are then accumulated in the
lower-precision output type even though the requested compute type is FP32.
The documented precision remedy is
`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`. This patch uses that
narrower control instead of reverting production prefill to PEDANTIC mode.

The later staged BF16-to-FP8 query patch does not restore the old compute
mode. Its actual route remains:

```text
regular-FP32 safe BMM -> query assembly -> static FP8 quantization
```

## Change

- Add a backward-compatible `precise=False` argument to
  `torch.ops._C.safe_mla_query_bmm`.
- When `precise=True`, preserve the handle's existing math mode, add only
  `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`, launch regular
  `CUBLAS_COMPUTE_32F`, and restore the exact incoming handle mode.
- Request the precise reduction contract only when `fp8_attention` and the
  backend's quantized-query-input contract are both active.
- Preserve regular FP32/tensor-core eligibility for all existing
  three-argument callers and BF16-output paths.
- Fail closed if the precise path is requested on CUDA but the stable op is
  unavailable.

## Validation

Completed locally:

- Python compile: PASS
- diff whitespace check: PASS
- static route/schema proof: PASS

```text
harness/v20_safe_query_accum_static_proof.py
sha256 bddde4847ee16058aceb16582f6c4a145e55a55f9250fc19cd2cdbcfec16e41c
```

The proof takes an explicit source-tree path, so it validates the clean PR
branch rather than silently reading the older production-integration
worktree. It can separately require #171 when checking a production image,
but #171 is intentionally absent from this PR diff.

Prepared GPU coverage:

- regular and precise numerical cases at M=1/6/11/8192;
- fixed production-boundary cases requiring precise error to be no worse than
  regular and proving at least one BF16 and post-FP8 byte difference;
- regular and precise CUDA-graph replay;
- a regular → precise → regular sequence requiring exact restoration of the
  shared cuBLAS handle's incoming mode;
- legacy three-argument schema compatibility.

Dedicated model-free proof and timing harness:

```text
harness/v20_safe_query_accum_gpu_proof.py
sha256 22f1c412b0f548b33c9448c047af77d99e38a9139bff784bbf14929acd6f8ea9
```

Model-free GPU discriminator, completed on the same CN4 process:

```text
cases:                         54
reference stable:              yes
BF16 output digests changed:   45/54
post-FP8 digests changed:      16/54
synthetic selected IDs changed: 0/54
verdict:                       CANDIDATE_SUPPORTED
```

The post-FP8 differences are concentrated where the proposed fix matters:

```text
M=1:       0/9
M=4:       0/9
M=9:       0/9
M=16:      4/9
M=32:      3/9
M=3072:    9/9
```

At M=3072 the regular kernel's maximum reference error reached 0.0078125
while the old precise kernel was exact in all nine cases. This prevents
scoping the correction only to short decode/MTP batches: long prefill is the
strongest observed numerical discriminator.

Evidence:

```text
old JSONL sha256 08ae9da7501debee3dfb4144371f9f9c7929828047d57e737da17b610ca60084
new JSONL sha256 6ad60efbc92f1922baaeb6f9f555c8079b556130974dc375926e2f519654a2ba
```

The selected-ID control is distribution-specific and does not supersede the
positive post-FP8 byte result. A separate production-geometry fused-indexer
discriminator matched serial, cooperative, and auto paths in 42/42 cases,
removing the suspected selector crossover from the leading cause.

Still pending before publication as an independent numeric-correctness PR:

- compiled execution of the new four-argument stable op;
- confirmation that the reduction guard matches or improves the old PEDANTIC
  output without its performance penalty;
- a maintainer decision that the stricter numeric contract is desired despite
  having no demonstrated link to the retrieval regression;
- matched prefill/decode measurement to quantify the accurate-reduction cost.

## Performance scope

This is not a PEDANTIC revert. The optimized regular compute type remains in
use; only reduced-precision intermediate reductions are forbidden at the FP8
query boundary. The cuBLAS heuristic may still select a different algorithm,
so model-free BMM timing and end-to-end prefill/decode measurements remain
required.

Alternatives considered:

| Option | Accuracy posture | Cost / disposition |
|---|---|---|
| Restore `CUBLAS_COMPUTE_32F_PEDANTIC` | Known precise discriminator | Causal fallback only; loses tensor-core prefill performance |
| FP32 output scratch, then BF16 cast | Explicit full-precision accumulation boundary | Adds a context-scaled scratch allocation and extra conversion; unacceptable on the tight KV budget |
| Chunk the token dimension | Can avoid the large-M heuristic | Multiple launches and a heuristic-dependent guarantee; not robust |
| Upstream #174 fused query | 44/45 exact in the small-query probe | M<=32 only and ~2x slower than staged on CN4; cannot fix long prefill |
| Disallow reduced-precision reductions | Preserves FP32 compute contract at the FP8 boundary | Selected; no persistent allocation, GPU/graph/perf proof pending |

Upstream PR #174's fused M<=32 kernel implements the desired numerical order:

```text
FP32 accumulation -> BF16 round -> static E4M3 scale
```

However, the model-free CN4 A/B found it slower than the current staged route
in all 90 timed cases across both current and PEDANTIC-rewind images:

```text
current image:  fused 0.0763 ms vs staged 0.0393 ms (1.94x)
PEDANTIC image: fused 0.0755 ms vs staged 0.0357 ms (2.11x)
```

The timing uses CUDA events around warmed current-stream operations, so it is
a valid end-to-end route measurement; graph-replay timing remains useful to
separate kernel cost from Python dispatch at these sub-0.1 ms durations.
Accordingly, #174 is not bundled into this accuracy proposal. The production
retrieval candidate must not include this change or #171 without separate
causal evidence.
