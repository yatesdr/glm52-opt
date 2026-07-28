> **Current-head rebase prepared — 2026-07-24**
>
> ```text
> branch: fix/gg-nvfp4-mtp-current-rebase-20260724
> base:   local-inference-lab/vllm dev/gilded-gnosis 89b4a98d1
> commit: ead9fd5c0
> ```
>
> Its stable patch-id
> `924c431e3bbb4fa0b52d1842c9d94906065fbf8e` is identical to existing
> PR #171 commit `dc770590`. Use this branch to update the existing draft;
> do not open a duplicate PR.
>
> The matched pre-change-image control has now separated two regressions:
> `6d32a0c3` passes cold retrieval at 250k but genuinely fails at 350k, while
> the later `992/a93` image already fails by 150k. This patch is therefore
> selected for the residual compact-NVFP4 verifier regression. It is not
> presented as the cause of the separate post-`6d32a0c3` 150k regression,
> which the fused-query/top-k microproof covers.

> **RETRIEVAL CLAIM WITHDRAWN — 2026-07-25.** PR #171 is absent from stock
> vLLM `5517197`, and an exact fa71 no-#171 discriminator still missed cold
> 100k (`cached_tokens=0`, two completion tokens, acceptance 0, empty final
> content). This route change is not the cause of the post-6d32 retrieval
> onset and must not be promoted as its fix. Retain the draft only if its
> independent verifier-routing rationale is still wanted and can pass its own
> quality/performance gates.

## Summary

Keep compact `nvfp4_ds_mla` MTP verification on SparkInfer's established
extend path in `auto` mode. Continue auto-routing the already-qualified
`fp8_ds_mla` verifier through decode, and preserve explicit
`VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1` as an opt-in for further qualification.

This corrects the scope of `3e731bc0`. Its new decode-verifier correctness test
uses `fp8_ds_mla`, while GLM-5.2 production uses the distinct compact
`nvfp4_ds_mla` record and BF16-QK kernel arm.

## Root cause

The v20 regression is not consistent with persistent KV-byte corruption:

- the compact-NVFP4 cache writer, dequantization primitives, established extend
  front door, and paged-indexer orchestration are executable-AST identical
  between the passing v19 stack and v20 after normalizing the B12X → SparkInfer
  namespace move;
- DCP4 global/local position mapping is bijective through the tested 475k
  range; and
- the four-supertile paged top-k fold at 118,750 local rows is mathematically
  equivalent to direct top-k.

The relevant execution delta is that `3e731bc0` changed the default from
extend to decode for genuine MTP verification without qualifying the compact
NVFP4 kernel arm.

## Change

`auto` is now format-qualified:

| Mode | `fp8_ds_mla` | `nvfp4_ds_mla` |
|---|---|---|
| `0` | extend | extend |
| `auto` | decode | extend |
| `1` | decode | decode |

The policy also controls scratch sizing. At MNS16/MTP3, compact NVFP4 no longer
reserves 64 decode rows for verifier batches; it reserves the 16 rows needed by
ordinary decode. For 64 gathered heads, 32 split slots, and value width 512,
this removes at least 99.38 MiB/GPU of verifier-only scratch:

```text
tmp_output: 96.00 MiB
tmp_lse:     0.38 MiB
output:      3.00 MiB
```

Ordinary one-token decode, MNS16, max model length, and KV format are unchanged.

## Internal field evidence

The production comparison that motivated this patch:

- v19 compact NVFP4 + `i8_ring`: 5/5 deep-needle pass at
  50k/200k/300k/350k/475k; decode baseline 165.1 aggregate tok/s at C16.
- pre-`992/a93` v20 (`6d32a0c3`), on one unchanged healthy process:
  - 250k cold: prompt 245,605, cached 0, completion 66,
    `content=738216`, `finish_reason=stop` — **PASS**;
  - 350k cold, retested with an 8,000-token budget: completion 84,
    `finish_reason=stop`, finalized content selected facility number `27`
    rather than ticket `738216` — **retrieval FAIL**;
  - 475k cold, retested with an 8,000-token budget: completion 213,
    `finish_reason=stop`, no needle in any scored field, and empty final
    content — **retrieval/finalization FAIL**.
- v20 compact NVFP4 + confirmed `i8_ring` + default auto route:
  pass at 50k/150k, genuine miss by 300k.
- v20 forced-one-split decode experiment: moved the pass boundary through 300k
  but still genuinely missed at 350k/475k and reduced the measured KV pool
  from 557,824 to 525,568.

The last result is important: it falsifies the earlier hypothesis that BF16
split partials were the complete cause. Reducing splits changes the margin but
does not qualify the compact-NVFP4 decode verifier.

Needle classification captured `content`, `reasoning`, `reasoning_content`,
the serialized message, `finish_reason`, and token usage. The higher-budget
350k/475k reruns ended normally without `738216`, so they supersede the
earlier short-budget ambiguity and prove genuine retrieval failures rather
than parser or budget artifacts.

## Validation

Completed without a GPU boot:

- Python syntax compilation;
- `git diff --check`;
- unit tests added for every mode/format combination, invalid input, and
  FP8/compact-NVFP4 scratch sizing;
- executable source proof:
  - matching compact-KV writer/reader AST hashes;
  - exhaustive DCP4 inverse mapping for `[0, 475000)`;
  - adversarial 12,000-candidate exact-radix overflow equality;
  - production-geometry long-context top-k fold equality;
  - isolated execution of the patched route policy.

The source proof reports:

```text
PASS: compact-NVFP4 KV writer/reader primitives are unchanged
PASS: patched route preserves fp8 auto and restores NVFP4 extend
PASS: DCP4 mapping is bijective through 475,000 positions
PASS: >4,096-candidate radix overflow fallback returns exact top-k
PASS: four-chunk 118,750-row top-k fold equals direct top-k
scratch: rows 64 -> 16 minimum recovered=99.38 MiB/GPU
```

The unchanged `6d32a0c3` container remained healthy through the 475k capture:
restart count 0, stable container/start identity, and no cache reuse
(`cached_tokens=0`). The non-monotonic 250k/350k/475k result does not define a
simple depth cutoff, but it does prove that the pre-query-fusion image still
has an independent long-context reliability/finalization defect. It is not a
restart, cache, parser, or token-budget artifact.

The independent post-`6d32a0c3` query/top-k discriminator completed in one
no-model execution:

```text
harness/v20_decode_retrieval_microprobes.py
sha256 9785aae7c9d78c1df8b9c1ea1d88c9876b72e61b14a7855cb032c7497386eaa4

v20-no-model-proof-fable-handoff.md
sha256 22b223d1ea2241ecc5587e1294ca7239cecff606745c2a7c11eb0e9b795e0116
```

Its selector leg passed 160/160 production-width and boundary cases, excluding
the widened top-k implementation as the residual old-v20 failure. Its three
query routes found small but real fused-versus-staged differences in 17/60
cases; the fused BF16+static-quant and direct-FP8 outputs were byte-identical,
localizing that separate newer regression to fused BF16 projection order.
Neither result removes the old image's independent failure beyond 250k, so
this compact-NVFP4 route correction remains selected.

```text
/home/derek/sol-proof-results/v20-decode-retrieval-microprobes-v3.jsonl
records: 237
sha256: eb8b4e495ee7dedf06c172274a614481e9fc4b5dd22f2ecf79826b1ed811b11b
```

The local Mac Python environment does not include pytest, so the full vLLM test
module was not collected here. The dependency-free proof executes the same
route helper directly and passed; CUDA test collection remains part of the GPU
gate.

Consolidated model validation is still required before this should leave
draft:

1. cold needles at 50k/150k/250k/350k/475k, requiring the needle in finalized
   content while also recording every response field;
2. decode C1/C4/C8/C16;
3. clean KV-pool measurement with diagnostics disabled; and
4. confirmation that the route INFO line selects extend for compact NVFP4.
