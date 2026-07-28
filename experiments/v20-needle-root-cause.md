# v20 deep-needle regression — corrected source diagnosis

Date: 2026-07-23

> **Diagnosis status updated 2026-07-24:** PR #171 remains a valid,
> independently scoped correction—compact `nvfp4_ds_mla` was never qualified
> for the new decode verifier—but it is not yet proven to be the sole
> retrieval root cause. The later failing image also added the fused
> BF16-weight/FP8-query epilogue and SparkInfer's widened exact top-k selector;
> the previously passing `af9d01cf/6d32a0c3` image predates both.
>
> A cold 50k/250k/350k/475k control on that prior image and the exact no-model
> query/top-k microproof now select among these direct differentials. The
> production candidate includes #171, while the query guard and any top-k
> change remain proof-selected. Do not cite the older CPU fold proof alone as
> proof that #171 completely resolves the model-level regression.

## Working verdict

The evidence does **not** support KV-cache byte corruption.

The compact-NVFP4 cache writer, NVFP4 dequantization primitives, paged-indexer
metadata preparation, long-context page traversal, and established extend
kernel are AST-identical between the passing v19 stack and v20 after normalizing
the B12X → SparkInfer namespace move.

The relevant v20-only execution change is vLLM commit `3e731bc0`
(`fix(attention): auto-route B12X MTP verification to decode`). It changed the
default MTP verifier from the established extend kernel to the flattened
split-K decode kernel. The qualification added by that commit tests only
`fp8_ds_mla`; production uses the distinct `nvfp4_ds_mla` compact record and its
BF16-QK kernel arm.

The minimal fix makes `auto` mean “use decode only for the format actually
qualified.” Compact NVFP4 stays on extend, matching v19. Explicit
`VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1` remains available for focused decode
bring-up.

Implementation:

```text
workspace/vllm-v20-nvfp4-mtp-extend/
```

## Why the earlier split-K conclusion was incomplete

The first CPU counterexample correctly proved that BF16 split partials are not
single-pass equivalent. It did **not** prove that split rounding was the whole
field failure.

The controlled Boot 12 result falsified that stronger claim:

| Candidate | Deep-needle result | KV pool |
|---|---|---:|
| v20 auto decode, normal split policy | pass 50k/150k; genuine miss by 300k | 557,824 |
| v20 decode, forced one split | pass through 300k; genuine miss 350k/475k | 525,568 |

One split improved the margin but did not restore the v19 result, and it cost
32,256 KV tokens in that boot. Therefore the fix must remove the unqualified
compact-NVFP4 verifier route, not tune split count.

## Source and byte proof

Run:

```bash
python3 harness/v20-nvfp4-mtp-route-proof.py
```

The proof compares the exact source trees used by v19 and v20 and reports:

```text
PASS: compact-NVFP4 KV writer/reader primitives are unchanged
  ConcatAndCacheNvfp4MlaFp8RopeKernel: e2ffb69fe100
  concat_and_cache_nvfp4_mla_fp8_rope: f4e0d24b087b
  _nvfp4_pair_bfloat2: 21993f38977c
  s1_qk_nope_nvfp4_bf16: 70888c54e05e
  s6_xv_nope_nvfp4_bf16: e504373d1713
  run_unified_prefill: b0fdd5f7b143
  prepare_paged_indexer_metadata: 96824f31e4e8
  index_topk_fp8: d1324c8fa5b2
PASS: patched route preserves fp8 auto and restores NVFP4 extend
PASS: DCP4 mapping is bijective through 475,000 positions
PASS: >4,096-candidate radix overflow fallback returns exact top-k
PASS: four-chunk 118,750-row top-k fold equals direct top-k
scratch: rows 64 -> 16 minimum recovered=99.38 MiB/GPU
```

The AST equality is stronger than a textual diff: comments, formatting, and the
package rename are removed from the comparison while executable structure and
constants remain.

### DCP mapping

For DCP4 and interleave 1, the runtime mapping is:

```text
owner = global_position mod 4
local = floor(global_position / 4)
global = 4 * local + owner
```

The proof exhaustively checks the inverse over all positions `[0, 475000)`.

### Long-context top-k fold

At 475k global context, one DCP rank owns at most 118,750 positions. The paged
indexer processes this in four 32,768-position supertiles. Its fold is valid
because:

```text
TopK(TopK(A, k) union B, k) = TopK(A union B, k)
```

The proof evaluates the exact production geometry (`k=2048`, 118,750 rows,
32,768-row chunks), including a forced winner in the oldest quarter.

The other v20 indexer delta is the exact fallback used when a coarse score
bucket contains more than the 4,096-entry fast candidate buffer. The proof
constructs a 12,000-value adversarial same-bucket case and verifies the
four-round radix result equals direct FP32-key top-k. This change corrects the
old silent overflow; it does not explain the regression.

## Exact route delta

v19 default:

```python
VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE = "0"
```

The source comment explicitly kept multi-token MTP verification on extend
because decode was the independent-one-token path.

v20 commit `3e731bc0` changed the default to:

```python
VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE = "auto"
```

and routed genuine verifier batches to decode. Its new causality test invokes:

```python
kv_cache_dtype="fp8_ds_mla"
```

There is no corresponding compact-NVFP4 MTP verifier qualification. Compact
NVFP4 is not a cache alias: it selects a 368/432-byte E2M1+E4M3 record, a
different scale format, native dequantization, and the BF16-QK math arm.

## Minimal fix

The route resolver now implements:

| Mode | `fp8_ds_mla` | `nvfp4_ds_mla` |
|---|---|---|
| `0` | extend | extend |
| `auto` | decode | extend |
| `1` | decode | decode |

The same decision controls decode scratch capacity. At MNS16/MTP3:

```text
old verifier reservation = 16 * (1 + 3) = 64 rows
new ordinary-decode reservation = 16 rows
```

For 64 gathered heads, 32 split slots, and value width 512, the removed
over-reservation is at least:

```text
tmp_output: 48 * 64 * 32 * 512 * 2 B = 96.00 MiB
tmp_lse:    48 * 64 * 32 * 4 B       =  0.38 MiB
output:     48 * 64 * 512 * 2 B      =  3.00 MiB
total:                                      99.38 MiB/GPU
```

This does not reduce the ordinary decode route, MNS16, max model length, or KV
format. It removes verifier-only scratch and should increase—not decrease—the
KV pool.

## Internal field evidence

- v19 `i8_ring`, compact NVFP4: 5/5 needle pass at
  50k/200k/300k/350k/475k.
- v19 decode baseline: 165.1 aggregate tok/s at concurrency 16.
- v20 `auto`, `i8_ring` confirmed engaged with no fallback:
  genuine retrieval miss by 300k.
- v20 one-split experiment: moved the boundary but still genuinely missed at
  350k/475k and reduced the pool to 525,568.
- Harness artifacts were separated from real failures: 250k/300k values found
  in `reasoning_content` count as passes; 350k/475k emitted coherent
  `finish_reason=stop` answers saying the ticket was absent and are true misses.

## Alternatives ruled out

- **INT8 wire disabled:** runtime introspection reported
  `enabled=True`, `backend=b12x`, `wire=i8_ring`; fallback audit was empty.
- **KV writer/reader drift:** executable AST hashes match, as shown above.
- **DCP position overflow/remap:** int32 positions are far below the limit and
  the production mapping is exhaustively bijective.
- **Paged multi-supertiling:** paged orchestration is unchanged and the fold
  identity is proven at production geometry.
- **Needle budget/parser:** diagnostic runs captured content,
  `reasoning_content`, completion count, and `finish_reason`; genuine misses end
  normally.
- **Split-K alone:** falsified by Boot 12.

## Validation status

Completed without a server boot:

- exact v19/v20 source-history trace;
- executable AST equality checks for the compact-KV path;
- exhaustive DCP mapping proof;
- adversarial exact-radix overflow proof;
- production-geometry long-context top-k fold proof;
- pure route-policy execution;
- syntax compilation;
- `git diff --check`;
- unit tests added for mode/format policy and scratch sizing.

The full pytest module was not collected on this Mac because the local Python
environment does not include pytest. The dependency-free proof executes the
same route helper directly and passed; CUDA tests remain part of the later GPU
gate.

Still required when a GPU window is available:

1. verify the INFO route line says compact NVFP4 remains on extend;
2. record the clean KV pool with diagnostics disabled;
3. run cold 50k/150k/250k/300k/350k/475k needles while accepting either
   `content` or `reasoning_content`;
4. run decode C1/C4/C8/C16 and confirm ordinary decode remains within band.

The patch is intentionally fail-safe: until that runtime gate is available, it
restores the exact verifier operator family used by the passing v19 production
stack rather than introducing new kernel math.
