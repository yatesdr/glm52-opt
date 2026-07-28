# Proposed title

[GG] fix(mla): keep B12X MTP3 indexer on the native SM120 path

Current-head draft:

```text
branch: fix/b12x-native-mtp3-current-20260724
base:   voipmonitor/vllm 89b4a98d1
runtime commit: 83782e474 (Brandon Music)
test/refactor commit: 146199ef9 (Derek Yates)
status: draft, consolidated v20 qualification pending
```

# Summary

Keep speculative verification with `next_n > 2` on the native
`(batch, next_n)` sparse-indexer path when the canonical B12X backend is active
on SM120.

The generic non-SM100 fallback flattens these rows because DeepGEMM's paged
MQA-logits kernel supports only `next_n` 1 or 2. SparkInfer's B12X indexer
supports native row counts 1, 2 and 4 on SM120, so applying the DeepGEMM
fallback to B12X MTP3 is unnecessary and can alter speculative-verification
metadata.

This is a focused extraction of Brandon Music's implementation from #139. The
runtime commit retains Brandon's authorship; this PR adds only a small pure
policy helper and CPU matrix test around it.

# Change

Use the canonical `use_b12x_sparse_indexer()` backend predicate when selecting
the MTP flattening fallback:

| Platform/backend | `next_n` | Result |
|---|---:|---|
| SM100 | 4 | native, unchanged |
| generic non-SM100 | 4 | flattened, unchanged |
| SM120 + B12X | 4 | native, corrected |
| generic non-SM100 | 1 or 2 | native, unchanged |

This requires no new environment variable. Selection works whether B12X is
enabled through its canonical environment setting or through
`--attention-backend B12X_MLA_SPARSE`.

# Why separate from #139

#139 adds the complete EXL3/Trellis backend. The MTP-indexer correction is
independently useful to every SM120+B12X MTP3 deployment, including non-EXL3
GLM-5.2 checkpoints. Keeping it focused lets the base image consume the
correctness fix without coupling it to a quantization backend.

# Validation

## CPU/source gates

- Python syntax compilation: pass.
- `git diff --check`: pass.
- Pure routing-policy matrix:
  - SM100, `next_n=4`: native;
  - generic SM120, `next_n=4`: flatten;
  - B12X SM120, `next_n=4`: native;
  - generic SM120, `next_n=2`: native.
- Runtime patch-id matches the implementation carried in #139.

## Independent field evidence

An independent four-GPU RTX PRO 6000 Blackwell deployment applied this exact
backend-predicate correction to a v20 GLM-5.2 EXL3 image:

- TP4 / DCP4 / MTP3;
- `use_flattening=False` through the native B12X path;
- tool-calling round trip: pass;
- nine cold, salted long-context needles: 9/9 exact;
- prompt lengths: 299,556, 397,057 and 490,658 tokens;
- needle depths at each length: 0.25, 0.60 and 0.93;
- all nine responses ended with `finish_reason=stop`.

The focused patch is also present in a production-candidate image derived from
the current v20 final candidate. Its local acceptance run is pending; those
results can be added without changing the patch.

# Scope

This PR does not include:

- the EXL3 loader or kernels from #139;
- filesystem/NVMe tier capacity work from #165;
- format-qualified verifier routing from #171;
- configuration or Docker-image changes.

It changes only the backend-aware flattening policy and its unit coverage.
