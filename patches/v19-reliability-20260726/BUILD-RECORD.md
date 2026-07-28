# Build record — v19 Tier-1 reliability candidate

Built on **cn3**, 2026-07-26. Prod (`glm52-prod`) was never stopped, restarted, or modified:
`RestartCount=0`, `StartedAt=2026-07-24T16:39:00Z`, health `healthy`, image digest unchanged
before and after the build.

## Result

**r2 is the candidate to test.** r1 is retained only as the superseded variant that gated the
shared-experts overlap off; see the revision note at the end.

```text
image   glm52-serve:v19-reliability-r2-20260726
        ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v19-reliability-r2-20260726
id      sha256:d5dd0e1375b9766e83e2bbb78b8ee4cb880d43d9eae46d08a0431738c388319e
context /home/claude/v19-candidate-r2-20260726/   (on cn3)

superseded (r1, gate variant, do not test)
        glm52-serve:v19-reliability-20260726
id      sha256:c585a23f0f1c9e5aaa4adee50ad360dff4d0b34de0fb2d74d244367fb2a38a63
base    ghcr.io/yatesdr/glm52-serve@sha256:ca8481687f7169adf177f02df83c06259af0941e9a1de8695b5a4e60d745463a
        (tag gilded-gnosis-v19-int8-block-patched — the exact image serving prod)
context /home/claude/v19-candidate-20260726/   (on cn3)
```

Not pushed to any registry. To move it to cn4:

```sh
cn3$ docker save glm52-serve:v19-reliability-r2-20260726 | zstd -T0 > /tmp/v19rel.tar.zst
cn4$ zstd -d -c /tmp/v19rel.tar.zst | docker load
```

## Provenance chain

**1. The base image is the git base plus exactly one commit.**
sha256 manifest of all 2,253 `vllm/**/*.py` in the base image vs. `git archive 7ea567a2`:

```text
files only in image : vllm/_version.py, vllm/third_party/deep_gemm/**   (build-time vendoring)
files only in git   : none
content differences : 1
                      vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
```

That one file is PR #133 (`2705bb2b`, offload store-ready prefix), already shipped in v19.
So the deployed base is `7ea567a2 + #133`, and **all vLLM files this backport touches are
byte-identical to `7ea567a2`** — the patch series applies to the real deployed bytes, not just
to a git tree that resembles them.

**2. The patches were applied to bytes extracted from the image**, not to a git checkout.

```text
git apply --exclude='*b12x_moe.py' --include='vllm/*' <each patch>
```

`--exclude` must precede `--include`: git apply uses first-match-wins, so the reverse order
lets `b12x_moe.py` through and the patch aborts on the missing file.

**3. Build-time gates** (all passed — see the Dockerfile):

| Gate | Checks |
|---|---|
| 1 | base vLLM version contains `gilded.gnosis.v19.vllm7ea567a.b12x4cfa530`; b12x is exactly `0.30.2` |
| 2 | stale `__pycache__` for each overlaid module removed; each recompiled under the image's py3.12 |
| 3 (r1) | all patch markers present; **zero** references to `tp_moe_plan_supports_aux_stream_overlap`, `checkpoint_channels`, `rollback_channels`, `_capture_channel_stack`; `VLLM_PCIE_ONESHOT_SINGLE_CHANNEL` default still `1` |
| 3 (r2) | as above, **plus**: exactly 2 `cooperative=True` in w4a16; `dynamic.py` keeps its pre-existing one; **build fails if `supports_shared_experts_aux_stream` appears anywhere in `vllm/`** (the overlap must stay enabled); `VLLM_DISABLE_SHARED_EXPERTS_STREAM` still present in `envs.py` as a runtime fallback |

**4. Post-build verification.**

- Full-tree sha256 diff, candidate vs base: **r1 exactly 10 files, r2 exactly 7 files** differ; none added, none removed.
- CPU-only import smoke test:
  - `init_workspace_manager(device, num_ubatches, num_lanes=1)` — lane parameter present
  - (r1 only) `NvFp4Nf3HybridMoEMethod.supports_shared_experts_aux_stream(None, 8)` → **False**
  - `mla_attention`, `dcp_alltoall`, fused-MoE modules all import
  - (the trailing `vllm._C` error is the container having no CUDA runtime, not a patch fault)
- Identity gate, r1: **31/31** on the candidate, **9/31** on unpatched prod.
- Identity gate, r2: **28/28** on the candidate, **12/28** on unpatched prod.

## Overlay file hashes — r2 (the candidate to test)

Identical on the host, in the build context, and inside the built image:

```text
6cbcd84b51ebbce11d08b5520c9283d2e83b290b3a4ab81d119b17fed388be99  vllm/model_executor/layers/attention/mla_attention.py
f1f14905b50d5aab2adca1559d8c6a2253bd25b301c1b02e1035064d222ee15f  vllm/v1/attention/ops/dcp_alltoall.py
b0c6ca44c5688c340631b85ca823baf57636f3b76346e74e2e15cc241314da0d  vllm/v1/worker/gpu/model_runner.py
df4696f295af496373172c4f1d47f6d8409f34c03152d930adebea41a9b05fb3  vllm/v1/worker/gpu/warmup.py
2b40f5f959a848b7f2c99839a1d33b4265223a3194f5a223160f98b39098f92b  vllm/v1/worker/gpu_worker.py
730a633339937ac95662a9a8d484b3184eff921c61834cec7810fbf724114c28  vllm/v1/worker/workspace.py
89533575f22082189fe98d748a4241f5c49165ba1d74b8bdf688fc3d8984b1f8  b12x/moe/fused/w4a16/kernel.py
```

## Not yet done

Nothing has been booted on GPUs. No performance, retrieval, or wedge-reproduction data exists
for this image yet — that is the acceptance run
(`v19-reliability-acceptance-criteria-20260726.md`).


---

## Revision 2 (same day)

r1 included the v20 capability gate (`93735960` + `e5b6cabb`), which disables the shared-experts
aux-stream overlap and costs ~11% decode. That was the wrong trade, for two reasons:

1. For a single-quant-method deployment the gate is **equivalent to
   `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`** — `self._stream = None` produces the identical
   `NO_OVERLAP` outcome. Hardcoding it into the image buys nothing an env var doesn't, while
   making the loss the non-toggleable default.
2. The hazard has a **proper fix**. `W4A16FusedMoeKernel._grid_barrier` is a spin-wait all-CTA
   barrier, and `_fused_grid_x` already sizes the grid `<= sms*blocks_per_sm` "so the cooperative
   barrier never deadlocks" — the code assumed co-residency but never asked CUDA to guarantee it.
   `cooperative=True` makes it explicit. `b12x/moe/fused/dynamic.py` already does exactly this
   for the Grid188/unified path, with the same reasoning in its comment; the w4a16 path (prod's,
   under `B12X_MOE_FORCE_A16=1`) had been missed.

r2 therefore drops 0004+0005 and adds `cooperative=True` to both w4a16 launches whose kernels use
the barrier — `W4A16FusedMoeKernel` and `W4A16FusedMoeHybridKernel`. The overlap stays on.

Build gates for r2 additionally **refuse the build** if `supports_shared_experts_aux_stream`
appears anywhere in `vllm/`, if w4a16 does not have exactly 2 cooperative launches, or if
`dynamic.py` loses its pre-existing one.

Identity gate: **28/28** on r2, **12/28** on the unpatched base.

Verified full-tree: r2 differs from the base in exactly 7 `.py` files, none added, none removed.
