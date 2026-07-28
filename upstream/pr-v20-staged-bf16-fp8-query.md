# Proposed title

[GG] fix(mla): preserve staged BF16 projection for FP8 query output

Current-head draft:

```text
branch: fix/bf16-fp8-query-staged-current-20260724
base:   local-inference-lab/vllm dev/gilded-gnosis 89b4a98d1
commit: 3be735af305f4e4261d6335bd6716829d9c0acdd
files:  2
status: prepared locally; publish as draft after consolidated quality evidence
```

## Summary

Keep the fused B12X BF16 query-projection path for BF16 output, but use the
established staged path when BF16 absorbed weights must produce an FP8 query:

```text
safe query BMM -> query assembly -> static FP8 quantization
```

The MXFP8-weight fused path remains eligible for FP8 output. This patch narrows
only the BF16-weight/FP8-output combination.

## Why

The fused projection changes floating-point accumulation order relative to
the safe cuBLAS BMM. Quantizing the result to FP8 makes small BF16 differences
observable as different wire bytes.

In a four-GPU production-geometry microprobe:

- selector/top-k cases: 160/160 exact;
- query cases: 60;
- fused-versus-staged FP8 byte differences: 17/60;
- direct fused FP8 and fused BF16 followed by static quantization were
  byte-identical;
- both differed from the safe staged BMM result in the same cases;
- minimum downstream retrieval overlap remained 0.999786.

This localizes the difference to projection accumulation order, not the
quantizer itself. The stage boundary must therefore be before the projection,
not merely before static quantization.

Evidence:

```text
/home/derek/sol-proof-results/v20-decode-retrieval-microprobes-v3.jsonl
records: 237
sha256: eb8b4e495ee7dedf06c172274a614481e9fc4b5dd22f2ecf79826b1ed811b11b
```

## Change

`_module_fused_mla_query_spec()` now rejects a BF16 absorbed weight when the
requested fused output is not BF16. The existing fallback then performs the
safe staged BMM and assembly/quantization path.

The guard is format-specific:

| Weight | Requested output | Result |
|---|---|---|
| BF16 | BF16 | fused, unchanged |
| BF16 | FP8 | staged, corrected |
| MXFP8 packed | BF16 | fused, unchanged |
| MXFP8 packed | FP8 | fused, unchanged |

No new environment variable is introduced.

## Performance scope

SparkInfer's BF16 fused query specialization is bounded to `M <= 32`. The
production prefill scheduler uses 3,072-token chunks. For the observed cold
8k and 55k prompts, every scheduled chunk was larger than 32, so this guard
cannot explain or improve their prefill throughput.

Executable source proof:

```text
harness/v20_prefill_query_route_static_proof.py
sha256: 81b89904dbb1a9b8bdbe81f65bcddba0a6e7f8f79584cd4eae31ad212a10bcf8
```

Expected:

```text
PASS staged-query prefill-route proof: scheduler_chunk=3072 fused_max_m=32
```

The affected region is tiny-M decode/speculation. The microprobe measured
roughly 4.1 microseconds for fused query projection versus roughly
7–8 microseconds for the staged operation, so the maximum expected end-to-end
cost is small but must be measured in the consolidated C1/C4/C8/C16 decode
gate.

## Validation

Completed:

- exact current-head base;
- one functional commit;
- Python syntax compilation;
- `git diff --check`;
- unit coverage for BF16/FP8 eligibility and retained BF16/MXFP8 routes;
- 60-case GPU byte comparison;
- 160-case exact selector control;
- executable proof that the guard is outside production-sized prefill.

Required before leaving draft:

1. cold salted needles at 50k/150k/250k/350k/475k;
2. finalized non-empty content containing the expected needle at every depth;
3. `cached_tokens=0` and `finish_reason=stop`;
4. matched decode C1/C4/C8/C16 with MTP acceptance;
5. confirmation that cold 8k/55k prefill is unchanged within run variance.

## Scope

This PR does not include:

- block-INT8 PCIe wire modes;
- compact-NVFP4 verifier routing from #171;
- native MTP3 flattening policy;
- top-k/indexer changes;
- MRV2 memory accounting;
- filesystem-tier capacity;
- Docker or deployment configuration.

AI assistance was used for investigation, the proof harness, and draft text.
The human submitter will review every changed line before requesting merge.
