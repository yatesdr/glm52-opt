# GLM-5.2 EXL3/TR3 3.25-bpw fused runtime

This is the self-contained source, build, and serving package for the fused
mixed-K runtime validated on CN4.

Model:
[willfalco/GLM-5.2-EXL3-TR3-3.25bpw](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw),
revision `e2b03576cd103e6ad322a1e091e5d0e2d0529073`.

Published image:

```text
ghcr.io/yatesdr/glm52-serve@sha256:a2b233f60329f22c7e541406b8972b3d8dc46c5c7d104d0aa3b4fee26374870e
```

The image derives from immutable Gilded Gnosis v20-r9 and adds:

- mixed 3/4-bpw EXL3 loading and planned persistent outputs;
- an exact single-pack/single-launch heterogeneous K3/K4 Trellis MoE route
  for M<=8;
- the profile-accounted exact-fold workspace implementation;
- allocation-free production PCIe DMA plus RMSNorm fallback;
- bounded filesystem/NVMe eviction from vLLM #165.

See [PATCHSET.md](PATCHSET.md) for source commits, byte pins, mechanism
details, review diffs, and validation results.

## Download the model

```bash
huggingface-cli download willfalco/GLM-5.2-EXL3-TR3-3.25bpw \
  --revision e2b03576cd103e6ad322a1e091e5d0e2d0529073 \
  --local-dir /models/GLM-5.2-EXL3-TR3-3.25bpw
```

## Start the validated profile

```bash
MODEL_PATH=/models/GLM-5.2-EXL3-TR3-3.25bpw \
CACHE_PATH=/var/cache/glm52-tr3-325 \
docker compose -f docker/tr3-325-public/compose.yaml up -d
```

The companion compose is the validated TP4/DCP4/MTP3 C1-C16 profile:

- exact image digest and model revision;
- dynamic per-token NVFP4 MLA KV with FP8 RoPE;
- maximum model length 480,000;
- 500,224-token GPU KV pool;
- fused mixed-K route through M=8;
- exact serial decode route and CUDA-graph coverage through M=64;
- cold deep-retrieval validation through 475k;
- `PXB` plus `i8_ring`, as measured on CN4.

`NCCL_P2P_LEVEL=PXB` and `F8_DMA=i8_ring` are topology-specific. Measure them
on a different host instead of copying them blindly. `F8_DMA=0` is the
conservative uncompressed transport control.

The compose uses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid
the reproduced post-decode deep-prefill fragmentation failure. It is a
GPU-cache profile and must not be combined with vLLM's `OffloadingConnector`.
The bounded offload implementation is included in the image for separate
profiles that do not use expandable segments.

## Build from source

The Dockerfile consumes only files checked into this package. It no longer
depends on local worktrees:

```bash
docker/tr3-325-public/build-and-push.sh
```

Set `GLM52_PUBLIC_TAG` to override the default rebuild tag. The build fails
closed on every expected base and output SHA-256.

## Review

For bot or human review:

1. Read `PATCHSET.md`.
2. Review `review-patches/sparkinfer-mixed-k-fused-v1.patch`.
3. Review the vLLM wiring in `patch_exl3_mixk.py`.
4. Run the CPU/source gates listed in `PATCHSET.md`.
5. Use `experiments/tr3-325-fused-route-results-20260730.md` for the complete
   live evidence and rejected-candidate history.
