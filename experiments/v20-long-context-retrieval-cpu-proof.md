# v20 long-context retrieval: CPU causal proof

Date: 2026-07-25

## Verdict

**Superseded by end-to-end evidence.** This document proves a real numeric
mechanism and a valid NVFP4 attention counterexample, but the exact PEDANTIC
binary rewind still missed the cold 100k needle. The proof is therefore not
causal for the observed model regression and the safe-query change is
withdrawn from retrieval qualification.

Run:

```bash
python3 harness/v20_long_context_retrieval_cpu_proof.py
```

The proof joins three independently checkable links:

1. archived SM120 measurements show the regular and PEDANTIC safe-query BMM
   differ at full prefill width (`M=3072`) in 9/9 cases;
2. source inspection proves the target `nvfp4_ds_mla` path consumes the
   absorbed query as BF16 in SparkInfer's NVFP4 QK arm, even though
   `supports_quant_query_input` is false;
3. a pure-NumPy production-shape witness proves that reduced-precision
   intermediate accumulation can reverse the attention result inside a valid
   2,048-token NVFP4 sparse window, while one final BF16 round after FP32
   accumulation favors the needle.

The completed model discriminator overruled the synthetic mechanism for this
symptom. Preserve the proof as numeric-correctness evidence only.

## Why the old selector was wrong

`supports_quant_query_input` means that the outer attention layer may pass an
already-quantized query. B12X intentionally sets it to false: B12X accepts BF16
and owns its quantized-KV attention math.

The prior candidate used:

```python
fp8_attention and self.impl.supports_quant_query_input
```

That expression is false for the exact production route
(`B12X_MLA_SPARSE` + `nvfp4_ds_mla` + `KV_FP8_ROPE=1`), so it would not have
enabled the fix.

The corrected contract is a separate backend capability:

```python
requires_precise_query_projection = True
```

and the call-site decision becomes:

```python
fp8_attention and (
    impl.supports_quant_query_input
    or impl.requires_precise_query_projection
)
```

The distinction is important for upstream correctness: input type and required
accumulation semantics are different backend properties.

## Measured erosion exposure

The archived 8-head `M=3072` fixture has 12,582,912 BF16 output values. From
the recorded mean and maximum absolute errors, without needing raw tensors:

```text
changed_count >= ceil(sum(abs(error)) / max(abs(error)))
```

This yields a conservative lower bound of 63–133 changed BF16 values per
full-width call. The model has 78 attention layers and a long prefill repeatedly
executes full 3,072-token chunks:

| Context | Full chunks | Full-width layer calls | Repeated-fixture lower bound |
|---:|---:|---:|---:|
| 50k | 16 | 1,248 | 78,624 |
| 100k | 32 | 2,496 | 157,248 |
| 150k | 48 | 3,744 | 235,872 |
| 250k | 81 | 6,318 | 398,034 |
| 350k | 113 | 8,814 | 555,282 |
| 475k | 154 | 12,012 | 756,756 |

The last column is an exposure calculation using the measured synthetic
fixture, not a claim that unknown model activations change at that exact rate.
It establishes why the defect is context-sensitive: the decode/small-M path can
look healthy while the number of full-width numerical divergence opportunities
grows roughly linearly with context.

## Exact NVFP4 witness

The CPU proof implements:

- BF16 ties-to-even rounding;
- E4M3FN scale encoding;
- SparkInfer's exact E2M1 thresholds;
- 512-wide NVFP4 group-16 quantize/dequantize;
- a 192×512 absorbed query projection;
- an already-selected production-width 2,048-token sparse-attention window.

The deterministic witness found by the proof reports:

```text
seed=16
changed BF16 projection values=320
max projection delta=0.0078125

precise needle score     +0.0107717812
precise distractor score -0.0107717812
reduced needle score     -0.0033762679
reduced distractor score +0.0033762679

precise attention output component  > 0 (favors needle)
reduced attention output component  < 0 (favors distractor)
```

This is a valid input and cache representation for the target numeric
contract. It is a counterexample to treating reduced-precision intermediate
reduction as semantics-preserving inside NVFP4 sparse attention.

The safe BMM runs after index selection. The proof therefore does **not** claim
that this BMM directly ejects an indexer candidate. It proves the direct
production operation instead: its changed BF16 query can reverse the attention
result over the selected set. That output enters the residual stream and can
alter later-layer indexer queries; confirming the model-level propagation is
the purpose of the one fixed-seed ladder.

## Fix contract

The scoped CUDA fix retains tensor-core-eligible `CUBLAS_COMPUTE_32F` and sets
`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` only when the backend marks
the quantized-KV query projection as precise. It restores the original cuBLAS
math mode before returning.

Acceptance order:

1. CPU proof and static source proof;
2. no-model GPU equivalence: accurate mode must reproduce the PEDANTIC
   `M=3072` fingerprint without global PEDANTIC;
3. one fixed-seed CN4 ladder with NVFP4 KV and FP8 RoPE;
4. only after quality passes, measure memory and throughput.
