# Safe-query post-FP8 build fingerprint gate

Status: implemented; GPU acceptance pending  
Scope: SM120 / CUDA 13.2 v20 images that expose the precise
`safe_mla_query_bmm` call

## Decision

Every completed image must pass a GPU gate **after image construction and
before push or promotion**. It cannot run inside a Dockerfile: the Docker build
environment has no CUDA driver. Calling this a build-time gate means it is a
mandatory stage of the build pipeline, not a `RUN` instruction.

The gate executes the production non-contiguous MLA query layout at
`M={1,4,9,16,32,3072}`, three seeds, and three FP8 scales. It compares all 54
post-quantization byte digests with the checked-in known-good reference.
`M=3072` is mandatory because the 2026-07-25 regression changed every tested
prefill-width FP8 result while many small decode-width differences vanished at
the FP8 boundary.

BF16 changes are reported diagnostically. Post-FP8 changes fail the build.
Tolerance tests remain useful for ordinary accuracy coverage but are not a
substitute: a value inside `rtol=atol=0.05` can cross an FP8 bin and alter every
downstream chunk.

## Files and contract

- Probe: `harness/v20_safe_query_build_fingerprint_probe.py`
- Comparator/waiver validator:
  `harness/v20_safe_query_build_fingerprint_gate.py`
- Pipeline wrapper: `harness/run_v20_safe_query_build_fingerprint_gate.sh`
- Reference JSONL:
  `harness/references/safe_mla_query_bmm_sm120_cu132_pedantic_v1.jsonl`
- Reference metadata:
  `harness/references/safe_mla_query_bmm_sm120_cu132_pedantic_v1.meta.json`
- Waiver schema example:
  `harness/references/safe_mla_query_bmm_waiver.example.json`

The reference is byte-pinned by its metadata. The observed file must contain
exactly the Cartesian 54-case grid and one `PASS/54` summary. Missing,
duplicate, extra, malformed, wrong-platform, wrong-CUDA, wrong-call-mode, or
wrong-extension records fail closed.

The probe requires the four-argument precise operator schema. An image that
silently removes the precision control cannot fall back to the regular path.

## Legitimate numeric changes

There is no wildcard or “accept current output” option. A reviewed waiver must:

1. name the exact source commit that intentionally changes numerics;
2. pin the current reference file;
3. pin the complete observed 54-case fingerprint-set digest;
4. enumerate every changed case with its old and new FP8 digest;
5. name the author and a distinct reviewer;
6. carry a UTC review timestamp, a substantive reason, and evidence links.

The gate validates all fields and only accepts the waiver for that exact
commit and exact observed bytes. A malformed, stale, broader, or partial
waiver fails. The normal landing process is:

1. CI fails and emits a candidate report;
2. the author supplies model-quality and performance evidence;
3. an independent reviewer approves the exact waiver;
4. the intended commit runs once as `PASS_REVIEWED_WAIVER`;
5. the reviewed candidate fingerprints become the next reference, preserving
   the waiver in history as the explanation for the baseline change.

## Invocation

```bash
harness/run_v20_safe_query_build_fingerprint_gate.sh \
  IMAGE \
  STABLE_LIBTORCH_SHA256 \
  EXACT_SOURCE_COMMIT \
  FRESH_OUTPUT_DIR \
  [REVIEWED_WAIVER_JSON]
```

The output directory preserves image inspection, raw 54-case JSONL, probe log,
gate report, and gate log. A nonzero result blocks push and promotion.

## Other quantization boundaries that need the same posture

The general rule applies where a floating-point kernel immediately feeds a
low-bit representation and the low-bit bytes affect routing, communication,
cache state, or expert selection. In this stack the next audit targets are:

- BF16 query assembly into static FP8 MLA queries;
- FP8/INT8 PCIe-DMA packers before collective transport;
- blockwise NVFP4 KV-cache quantization, including scale bytes;
- router logits before top-k expert selection;
- MTP verifier/draft logits before acceptance decisions.

Not every target needs a 54-case gate. Each needs a small production-geometry
set that fingerprints the actual quantized or discrete boundary, not merely a
float tensor under a loose tolerance.
