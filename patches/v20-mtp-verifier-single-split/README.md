# v20 B12X MTP verifier single-split precision fix

This is the runtime-only patch for the v20 deep-needle regression diagnosed in
`../../v20-needle-root-cause.md`.

Apply from the vLLM source root:

```bash
git apply --check /path/to/b12x_mla_sparse.patch
git apply /path/to/b12x_mla_sparse.patch
python -m py_compile vllm/v1/attention/backends/mla/b12x_mla_sparse.py
```

The patch was generated against v20 base commit `3e731bc0` and separately
passes `git apply --check` on:

- the MLA query-BMM source at `7562bb27`; and
- the CKV profile-reset source at `ce1746b7`.

Do not replace the combined candidate's whole source file. Apply this narrow
patch so both independent fixes remain present.

Expected boot evidence:

```text
B12X MTP verifier decode uses one split for BF16-partial precision; ordinary decode retains up to 32 splits
```

Source changes only affect genuine `is_spec_decode` batches. Ordinary
one-token decode still requests the existing maximum split count.

Validation is defined in:

```text
../../v20-mtp-verifier-split-precision-fable-test-spec.md
```

Artifact SHA-256:

```text
6c091f75941777670d435e7d1cfe70e520b5de2ddd1f392f4b5133bec1c245aa  b12x_mla_sparse.patch
```
