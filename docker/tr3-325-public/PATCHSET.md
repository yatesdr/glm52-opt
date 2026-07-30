# TR3 3.25-bpw fused-runtime patchset

This directory is the self-contained source and build package for the fused
GLM-5.2 EXL3/TR3 3.25-bpw runtime validated on CN4.

The published image is:

```text
ghcr.io/yatesdr/glm52-serve@sha256:a2b233f60329f22c7e541406b8972b3d8dc46c5c7d104d0aa3b4fee26374870e
```

Model:

```text
willfalco/GLM-5.2-EXL3-TR3-3.25bpw
revision e2b03576cd103e6ad322a1e091e5d0e2d0529073
```

## Components

### Mixed-K EXL3 integration

`patch_exl3_mixk.py` modifies v20-r9's vLLM EXL3 layer to:

- accept per-expert mixed 3/4-bpw metadata;
- partition experts into K-homogeneous Trellis tiers;
- plan all persistent outputs and scratch before KV-cache sizing;
- use the fused heterogeneous SparkInfer route for the exact proven
  `K3=192, K4=64` signature at `M<=8`;
- retain the existing exact serial tier path for larger decode, the homogeneous
  MTP draft layer, and all prefill shapes;
- reuse the existing tier-local rotation tables instead of allocating a
  duplicate global rotation table.

Patch SHA-256:

```text
f337457c29063f0467516a69ded808196e029c65f3359493226dddd4e3f422a1
```

Patched vLLM `exl3.py` SHA-256:

```text
ddcb1ae7f21c72119a0c24c376ed3f9d2f09e49a49f272af25e6cfcf4fc98ec5
```

### SparkInfer heterogeneous fused route

The exact deployed source is vendored under `overlays/sparkinfer/`. The
corresponding review diff is
`review-patches/sparkinfer-mixed-k-fused-v1.patch`, based on SparkInfer commit:

```text
669a12ddc7cf3021e91a25f398b1a883b703fd12
```

The implementation:

- packs the global route once;
- maps each routed expert to its K3/K4 tier and tier-local expert id;
- runs one cooperative persistent Trellis launch;
- preserves separate FP32 K3 and K4 accumulators and the production
  `sum(K3) + sum(K4)` reduction association;
- exposes a planned/bound public API that allocates no request-time scratch;
- is graph-replay safe and fail-closed on unsupported tier signatures.

Pinned deployed source:

```text
8fb83a73be4a3ea7ad0b2093cad674130dee84183b01c8d8187249b87be0feae  sparkinfer/moe/_shared/kernels/w4a16/kernel.py
dc977562db1dd394cef3df8163daf0a0eabe8057452f7659d39e1e03b9155427  sparkinfer/moe/trellis_moe/__init__.py
17a58dc97f29ba4792a63af26e0dee30aac5e39d32ee562c4ee45c3b303c0ca9  sparkinfer/moe/trellis_moe/_impl.py
33a910eb86648ef333d374bfd283cbd06fbcb4e1777b910346a949a4d331c929  sparkinfer/moe/trellis_moe/api.py
```

### Profile-accounted indexer fold scratch

The final image also contains the separately reviewed exact-fold workspace
fix. Its deployed sources are vendored under the same overlay tree and its
review diff is `review-patches/sparkinfer-profiled-fold-scratch-v1.patch`,
based on SparkInfer commit:

```text
36cade0bd8d87a26b7eb2fc8cc46188496465a06
```

This component is independent of the fused MoE arithmetic. It makes the
parallel exact-fold workspace visible during vLLM memory profiling and
provides rank-invariant streaming fallback. The final balanced compose
selects exact streaming carry to reclaim its reservation.

### Other image overlays

- `patch_pcie_dma_inplace_fusion.py`: removes a production-size late temporary
  from the fused PCIe all-reduce plus RMSNorm fallback without changing the
  generic collective API.
- `bounded_fs_manager.py`: vLLM #165 bounded filesystem/NVMe eviction. It is
  present in the image but not enabled by the GPU-only validation compose.
- Dynamic per-token NVFP4 MLA KV and its record ABI are inherited from the
  paired vLLM #189 / SparkInfer #86 implementation in the r9 base.

## Proof and validation

Source/operator proofs:

- `harness/test_tr3_mixk_fused_reduction_contract.py`
- `harness/tr3_mixk_fused_gpu_equivalence.py`
- `harness/tr3_mixk_serial_window_probe.py`
- `harness/test_exl3_mixk_persistent_output.py`

The operator matrix covered M=1/2/4/8/16/24/32 with K3-only, K4-only, and
mixed routes: 21/21 bit-exact, including CUDA graph replay. Direct timing
placed the fused crossover at M=8, so the production selector is capped there.

The complete live results, rejected candidates, memory failures, and final
promotion evidence are documented in:

```text
experiments/tr3-325-fused-route-results-20260730.md
```

Final promoted highlights:

| Gate | Result |
|---|---:|
| GPU KV pool | 500,224 tokens |
| MTP0 C1 | 34.84 tok/s |
| MTP3 C1 / C8 / C16 | 65.47 / 154.44 / 190.64 tok/s |
| Cold 55k prefill | 1,325 tok/s |
| Cold needles 50k/250k/350k/475k | all exact |
| Ordered C32 then cold 350k | pass |
| KLD, 2,047 positions | 0.0959706378925062 |

## Build

The Dockerfile verifies both the expected r9 input bytes and every installed
output byte. No local worktree is required:

```bash
docker/tr3-325-public/build-and-push.sh
```

Override `GLM52_PUBLIC_TAG` to choose the destination tag.
