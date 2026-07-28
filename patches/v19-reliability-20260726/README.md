# v19 reliability backport — Tier 1 series

Base: vLLM fork `7ea567a2` (the commit inside
`ghcr.io/yatesdr/glm52-serve@sha256:ca8481687f71`, tag `gilded-gnosis-v19-int8-block-patched`).

Full analysis: `../../v19-reliability-backport-20260726.md`.

## Apply

The recommended set is 0001, 0002, 0003 (vLLM) plus 0006 (b12x).

```sh
# vLLM (structural memory-safety). --exclude MUST precede --include: git apply is first-match-wins.
git apply --exclude='*b12x_moe.py' --include='vllm/*' 0001-b3ea2e8f.patch 0002-a8b59fbe.patch 0003-ef7cae43.patch

# b12x (keeps the shared-experts overlap ON and makes it safe)
patch -p0 b12x/moe/fused/w4a16/kernel.py < 0006-b12x-w4a16-cooperative-grid.diff
```

**0004 and 0005 are retained for reference only and are NOT part of the recommended set.**
They disable the shared-experts aux-stream overlap (~11% decode). For a single-quant-method
deployment they are equivalent to `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`, which needs no image
change — so they buy nothing an env var doesn't, while making the loss the default.

## Contents

| File | Upstream | Effect |
|---|---|---|
| `0001-b3ea2e8f.patch` | vLLM #136 (Martin Vit) | Honors `force_contiguous_mla_bmm_{input,weight,output}`, which v19's B12X_MLA_SPARSE backend sets but v19's `mla_attention.py` ignores. Fixes the OOB strided BMM at `mla_attention.py:1476` — the exact frame in `a2a-cublas-crash-spec.md`. |
| `0002-a8b59fbe.patch` | PR #132 family | Workspace lanes; stops the MTP draft from aliasing/reallocating the target's captured workspace. |
| `0003-ef7cae43.patch` | vLLM #130 (Martin Vit) | Allocates the retained DCP A2A staging pair before CUDA capture instead of from the shared graph pool. |
| `0006-b12x-w4a16-cooperative-grid.diff` | new | `cooperative=True` on `W4A16FusedMoeKernel` + `W4A16FusedMoeHybridKernel` — the two launches whose kernels use the spin-wait `_grid_barrier`. Mirrors the pre-existing fix in `b12x/moe/fused/dynamic.py`. **Keeps the overlap enabled.** |
| `0004-93735960.patch` | PR #132 family | *Reference only.* Capability hook for aux-stream overlap. |
| `0005-e5b6cabb.patch` | PR #132 family | *Reference only.* Makes `nvfp4_nf3_hybrid` decline overlap — the blunt workaround this backport replaces. |
| `tier2-0001-d6b49f4cd.patch` | — | Optional. Trims padded block tables during MTP expansion (480k / block-64 / DCP4 nonuniform decode). |

## Result

7 `.py` files (6 vLLM + 1 b12x). No CUDA rebuild, no wheel rebuild, no change to the baked
INT8 extension — ships as a file-overlay layer on the existing v19 image.

Verified: the vLLM patches apply clean to `7ea567a2`; all 7 files pass `py_compile` under the
image's py3.12; no residual reference in `vllm/` to any b12x symbol missing from `b12x 0.30.2`;
the built image differs from the base in exactly these 7 files.

**Not measured yet.** No throughput loss is expected — the overlap is retained. Boot and
measure before promoting; see the acceptance criteria doc.
