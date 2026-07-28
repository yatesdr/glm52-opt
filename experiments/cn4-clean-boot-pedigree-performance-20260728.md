# GLM-5.2 v20 dynamic-KV clean-boot record

Date: 2026-07-28  
Host: CN4 only  
Status: healthy production/review boot

## Published image

```text
ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-dynamickv-prod-clean-r2-20260728
sha256:b4ef498c6b3494961ba381ccc85215590d5e3b3cd17ec04146152a101b653790
```

The digest above is the corrected release. Do not use the superseded
`sha256:beecd32f...` image: its first clean boot exposed a missing environment
registration and stopped before weight loading.

## Pedigree

| Layer | Pinned identity | Purpose |
|---|---|---|
| Upstream v20 runtime-stride RC | image digest `sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0` | Canonical GLM-5.2 v20 base |
| vLLM | `0c79e41db41f250ccdfc4be92d171960a5787f73` | Runtime-stride v20 vLLM base |
| SparkInfer | `c3828fd7f807ce237a9ac36ef033659e6f6b6dd3` | Runtime-stride SparkInfer base, including the merged runtime-stride fixes |
| FlashInfer | `801d57a08958c13d375ddbb6be3be4808f48a708` | Pinned CUDA 13.2 FlashInfer |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` | Pinned buffered loader |
| Runtime | PyTorch `2.12.0+cu132`, CUDA `13.2.1`, NCCL `2.30.4` | Base compute/runtime stack |
| SparkInfer PR #86 | final head `0ddd13b4fdbb6a287581aec55fcf9dbbb7e52fd3` | Dynamic per-token NVFP4 MLA-KV scale, readers, tests, and record/cache contract |
| vLLM PR #189 | final head `b57062274c3f53bec69b431bfae7230977f5f10c` | Server-static mode, writer and DCP re-quantization wiring, fail-closed validation, and external cache ABI identity |
| Bounded filesystem offload | vLLM `95488c3885d46145d49fb07ecb911b76c7b80a44` | Bounded DRAM plus NVMe prefix-cache tier |
| Launcher/calibration | `05626808ebdf9e0be89657d49bebbaae03ef0933` | Canonical launcher plus topology/wire probes |
| Clean-boot integration repair | `envs.py` output SHA-256 `4020fbc4a778f2a4d09da96ea03069ba4eb7a8e6a9b338e1ed22cd1d98b2722f` | Preserves three base DCP/PCIe environment registrations omitted by the PR #189 full-file overlay |

The integration repair changes only environment registration. It does not
change a model path, kernel, selector, record layout, collective, or launcher.

## Behavior unique to this image

- The exact v20 top-k selector remains active. No `bounded_compat`,
  `oldest_boundary`, or other historical selector policy is included.
- Each 368-byte NVFP4 MLA-KV record carries its token-local FP32 outer scale
  at bytes `[292,296)`. Record size and GPU KV capacity are unchanged by the
  scale.
- Dynamic/static record semantics are server-static and included in the
  external-cache namespace/ABI identity.
- `KV_CACHE_MEMORY_BYTES` is accepted as an explicit, fail-closed launcher
  input so a previously measured graph estimate can be replaced by the
  guarded exact KV allocation.
- `NCCL_P2P_LEVEL=auto` and `F8_DMA=auto` are the only added host-dependent
  automatic choices. CN4 selected `PXB` and `i8_ring`.
- The optional Destroyed quality-first MXFP8 membership remains available
  behind `DESTROYED_MXFP8=1`, but it is off in the default production posture
  because it consumes additional VRAM.

## Clean-boot result

The test used fresh `/cache` and NVMe namespaces. The prior production
compose, image identity, and NVMe cache were retained separately for
rollback.

The first published image (`sha256:beecd32f...`) failed before weight loading:

```text
AttributeError: module 'vllm.envs' has no attribute 'VLLM_PCIE_DMA_MIN_BYTES'
```

Source comparison found that the PR #189 `envs.py` overlay retained its new
dynamic-scale controls but omitted three registrations already consumed by
the unchanged base runtime:

```text
VLLM_PCIE_DMA_MIN_BYTES
VLLM_DCP_INDEXER_SHARDS
VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS
```

The corrected `r2` image merged those registrations with fuzz disabled,
verified input/patch/output hashes, compiled the module, and exercised the
environment parser before publication.

| Gate | Result |
|---|---|
| Corrected image | `sha256:b4ef498c...` |
| Fresh boot | PASS |
| Container restarts | `0` |
| Fatal log signatures | `0` |
| PCIe topology | `PXB` |
| Wire mode | `i8_ring` |
| Dynamic record | `nvfp4_ds_mla`, FP8 RoPE, 368 bytes |
| GPU KV pool | **617,728 tokens / 4.60 GiB** |
| Maximum request length | 480,000 tokens |
| Prefix caching | enabled |
| DRAM offload | **64,000,000,000 bytes** shared mmap |
| NVMe offload | fresh namespace, **1,000,000,000,000-byte** cap |
| API health from Mac | HTTP 200 |
| Deterministic smoke | HTTP 200, content `ok`, `finish_reason=stop` |

CN4 was left running this healthy container:

```text
/home/derek/glm52-v20-prod-clean-r2-20260728/compose.yaml
```

## Matched v20 control versus dynamic fix

“v20 control” below means the exact-selector v20 stack using the static
PR #145 scale file. It is the same-image, same-checkpoint, one-variable
control for the dynamic-record mechanism. It should not be confused with an
unmatched benchmark from another release image or serving posture.

| Metric | v20 static-scale control | Dynamic per-token scale | Change |
|---|---:|---:|---:|
| Decode, MTP0, n=10 | 46.076 tok/s, SD 0.0268 | 45.901 tok/s, SD 0.0218 | **-0.380%** |
| Cold prefill wall time, 49k–466k | see rows below | see rows below | **+0.74% to +1.87%** |
| KLD, n=3, lower is better | 0.14622770, SD 0.00468791 | 0.13903565, SD 0.00201006 | **-4.92%** |
| Frozen 343,727-token retrieval failures | not recovered by the uncalibrated stock posture | 3/3 recovered | PASS |
| Randomized deep-context ladder | — | 6/6 exact, 49,098–466,493 | PASS |

Matched cold-prefill rows:

| Actual context | v20 static control | Dynamic | Dynamic delta |
|---:|---:|---:|---:|
| 49,098 | 36.88 s | 37.57 s | +1.87% |
| 147,273 | 118.46 s | 119.80 s | +1.13% |
| 245,503 | 213.11 s | 215.08 s | +0.92% |
| 294,619 | 265.15 s | 267.10 s | +0.74% |
| 343,735 | 319.31 s | 322.44 s | +0.98% |
| 466,493 | 468.62 s | 472.62 s | +0.85% |

Every dynamic prefill row was cold (`cached_tokens=0`), returned the expected
needle, and finalized normally. The decode comparison used MTP0 deliberately
to isolate reader/kernel cost from speculative-acceptance variation.

For operational context, the same dynamic performance code previously
measured on CN4 at the production MTP3 posture:

| Operational metric | Result |
|---|---:|
| C1 decode | 75.3 tok/s |
| C4 aggregate decode | 133.3 tok/s |
| 8k cold prefill | 1,382 tok/s |
| 64k cold prefill | 1,315 tok/s |

Those operational numbers are not a single-variable base comparison. The
`r2` release adds only the environment-registry integration repair, so it
does not change their kernel-level interpretation.

## Evidence

- `design/pr86-pr189-bot-review-closure-20260728.md`
- `design/v20-nvfp4-scaling-kld-n3-comparison-20260728.md`
- `fable-dynamic-scale-packaging-handoff-20260728.md`
- `harness/cn4-evidence-archive/20260728/pr86-pr189-cleanup/`
- `harness/cn4-evidence-archive/20260728/nvfp4-dynamic-token-scale-kld-n3-v1/`
- `harness/cn4-evidence-archive/20260728/nvfp4-dynamic-token-scale-randomized-ladder-v1/`

