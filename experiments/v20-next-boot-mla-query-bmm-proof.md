# v20 next boot — restore the safe MLA query-BMM layout contract

Date: 2026-07-23  
Operator: Fable  
Objective: prove the source fix on the existing v20/Boot-8 stack, then use
the same image for the already-planned CN3 production qualification.

This specification supersedes the MoE/shared-expert Boot-10 proposal. Do not
include commit `bf1b32cf` in this build. Boot 10 with
`CUDA_LAUNCH_BLOCKING=1` identified a different first failing operation.

## Root cause

Boot 10 made the asynchronous error synchronous at:

```text
MLAAttention.forward_impl
  mqa_q_nope = mqa_q_nope.transpose(0, 1)
  torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)
                                    ^ first failing CUDA launch
```

The first operand is a split-and-transpose view and is not contiguous. The
failure is the MTP draft model's MLA query-absorption BMM during production
decode-speculator warmup. There were no MoE frames in the synchronous stack.

The source history contains the regression:

1. `b3ea2e8f` (`[GG] Fix MLA BMM layout contract for cuBLAS read-ahead
   (#136)`) introduced backend-selected contiguous BMM operands.
2. `6a2edcf1` later kept DCP attention outputs head-major and removed all
   three B12X contiguity flags. Removing the V-up output copy was correct for
   the new head-major DCP output.
3. Query absorption is independent of that DCP output layout. It still
   creates the strided split-and-transpose view, but v20 no longer
   materializes it before `torch.bmm`.
4. The v20 image commit `3e731bc0` contains `6a2edcf1`; the unprotected BMM is
   exactly the line named by Boot 10.

Profiling and production capture use different allocation neighborhoods.
That explains why the same descriptor can survive profiling and cross an
unmapped tile boundary in production.

## Fix under test

Source commit:

```text
7562bb27 fix(mla): restore safe query BMM layouts
```

Patch:

```text
patches/v20-mla-query-bmm-contiguity/
  0001-fix-mla-restore-safe-query-BMM-layouts.patch
SHA-256 ea628dc7d06f8766b6a8d02813f67b6223b497320f3392076350af7d8e4e411a
```

The production change is 29 added lines and 2 replaced lines across two
vLLM files:

- B12X opts the query input and absorbed weights into the safe BMM layout;
- `MLAAttention` propagates that backend contract and materializes
  `mqa_q_nope` immediately before the affected BMM; and
- absorbed-weight preallocation refuses compatible-but-strided storage and
  replaces it rather than copying back into the unsafe old allocation.

The patch deliberately does **not** restore the old V-up output temporary.
The newer head-major DCP output remains in place. It also does not change MTP,
CUDA-graph sizes, A2A/AG-RS routing, i8-ring, CKV prefetch, MoE streams, GMU,
DRAM offload, NVMe offload, or model length.

## Local proof already completed

On the exact v20 image source `3e731bc0`:

```text
pytest -q tests/v1/attention/test_mla_backends.py -m cpu_test
11 passed, 2314 deselected
python3 -m py_compile ...                         PASS
git diff --check                                 PASS
```

The formatted patch was then applied to the Boot-8 CKV-reset source
`ce1746b7`, not merely to stock v20. It applied without offsets or conflicts,
and the same CPU suite again passed 11/11. The tests cover:

- backend contract propagation;
- the exact non-contiguous split-and-transpose query operand arriving at
  `torch.bmm` contiguous;
- compatible contiguous absorbed-weight reuse; and
- replacement, rather than pointer-preserving reuse, of strided absorbed
  weights.

## Gate 0 — build from Boot 8, changing only this defect

Use the exact Boot-8 image/source stack as input:

```text
…-int8-nvme-bt1876-w4a16coop-ckvreset
image ID prefix 5f0c7b43daaa
```

Retain its W4A16 cooperative-grid, CKV-generation reset, aligned block table,
MRV2 accounting, INT8 wire, DRAM offload and bounded NVMe patches. Add only
the patch above. Do not add the held MoE patch.

Required Boot-8 input and expected output SHA-256 pins:

```text
vllm/model_executor/layers/attention/mla_attention.py
  input   04f09c58c00b4b2282e6cee9575ccca7f39e1bbb7d6f4718dc34f254136e8954
  output  54181f5fe83030c4fb6e8cb8d2315a0b55c83974ae5693d99035d21db878449f

vllm/v1/attention/backends/mla/b12x_mla_sparse.py
  input   ee08b603a266b752791ac3a811b23eb0680d9834d84d97323fcc11280f4e927d
  output  3ada9852c37b56cf1b0092ca86282119e6cf95be932ae6aac782c938ec74835a
```

Fail closed before building on any input mismatch. Verify both output pins in
the completed image. If repository tests are present in the build context,
run:

```bash
python -m pytest -q tests/v1/attention/test_mla_backends.py \
  -k 'query_bmm_contiguity or query_absorb_materializes or absorbed_weight_preallocation or post_load_replaces_strided or post_load_preserves_runtime_weight_addresses'
```

Expected: six selected tests pass. Runtime images lacking tests are not a
blocker because the exact patched Boot-8 tree already passed locally.

## Gate 1 — one discriminating boot

Use the exact Boot-8 Compose and production geometry without a tuning change:

```text
TP4 / DCP4 / MTP3
max_model_len=480000
max_num_seqs=16
max_cudagraph_capture_size=64
gpu_memory_utilization=0.980
VLLM_DCP_A2A_MAX_TOKENS=64
SPARKINFER_PCIE_DMA_FP8=i8_ring
VLLM_PCIE_DMA_FP8=i8_ring
DRAM offload=64,000,000,000 bytes
NVMe acceptance cap=8,589,934,592 bytes
```

For this first boot only:

```text
VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS=1
CUDA_LAUNCH_BLOCKING must be unset
```

The diagnostic already failed at the old M=9 boundary in Boot 7, so it does
not mask the reproduced defect. It gives descriptor-level proof without
another localization boot.

PASS requires all of the following on the first process:

1. profiling decode-speculator capture completes all 16 descriptors;
2. production decode-speculator capture completes sizes 16 through 1;
3. M=9 reports clean input preparation, eager warmup and capture stages;
4. the API reaches serving state and deterministic liveness returns `4`;
5. one normal request uses MTP and finishes with `finish_reason=stop`;
6. `RestartCount=0`, with unchanged container ID and `StartedAt`; and
7. no illegal access, cuBLAS error, worker exit, EngineDead, OOM, Xid,
   assertion or 5xx appears.

On failure, stop and seal the first-run log and inspect JSON. Do not tune GMU,
MNS, graph cap, A2A threshold or model length, and do not start a patch
ladder. Report the diagnostic's first failing descriptor and operation.

## Gate 2 — qualify this same process

If Gate 1 passes, do not restart or rebuild. Continue the existing combined
v20 qualification on the live process:

1. decode C1/C4/C8/C16 and the required ctx-50k cells;
2. unique-prefix needles at 300k, 350k and 475k;
3. bounded NVMe fill, turnover and promotion at 8 GiB;
4. 16 overlapping unique-prefix 50k requests to force GPU-KV overflow;
5. cold 8k and 50k prefill with prefix-cache miss evidence; and
6. final liveness, process identity, restart count, fatal-log and cache
   inventory audit.

This preserves the maintenance window: the proof boot becomes the first
qualification boot rather than being discarded.

## Gate 3 — persistence restart and production promotion

Use the already-required NVMe persistence restart as the clean production
boot. Keep the same image and serving configuration and the same NVMe
namespace. Remove `VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS`; keep
`CUDA_LAUNCH_BLOCKING` unset.

Require clean production decode capture again, persisted external-cache
promotion, liveness, one MTP request, one deep needle, `RestartCount=0` for
the new container, and an empty fatal-signature audit. If green, leave this
second process online. No third boot is part of this plan.

