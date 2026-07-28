# v20 MTP verifier split-precision fix — CN3 proof

Date: 2026-07-23  
Operator: Fable  
Target: the current v20 Gate-1-passing image with the MLA query-BMM fix

## Purpose

Prove or falsify one source-level diagnosis in one boot:

- v20 commit `3e731bc0` changed genuine MTP verification from the single-pass
  extend kernel to the split-K decode kernel;
- vLLM forces `ceil(2048 / 64) = 32` split slots;
- under DCP4, roughly one quarter of the 2,048 global winners are local, so
  about eight 64-candidate partitions carry data on each rank;
- every active partition writes its normalized 512-value vector to BF16
  `tmp_output` before the merge converts it back to FP32; and
- that BF16 boundary is not numerically equivalent to one FP32 accumulation.

The patch keeps the v20 decode-verifier route but forces genuine MTP verifier
batches to one split. Ordinary one-token decode retains the existing split-K
count and throughput path.

The CPU byte/numeric proof is:

```text
harness/v20-mtp-split-precision-proof.py
```

Expected output begins:

```text
PASS: BF16 split merge is not single-pass equivalent
numeric: single_pass=0.000499725 split_merge=0.000000000
```

## Important integration rule

Do **not** copy a complete `b12x_mla_sparse.py` from an older worktree into the
image. The live candidate already combines the CKV profile-reset and MLA
query-BMM fixes in this file (reported input SHA-256
`3ada9852c37b56cf1b0092ca86282119e6cf95be932ae6aac782c938ec74835a`).

Apply only the narrow diff from:

```text
workspace/vllm-v20-mtp-dcp-guard
```

relative to base `3e731bc0`, for:

```text
vllm/v1/attention/backends/mla/b12x_mla_sparse.py
```

The diff applies cleanly on top of the independent MLA query-BMM commit
`7562bb27`; this was checked locally with `git apply --check`. Record the
resulting full-file SHA-256 from the built image as the output pin.

## Gate 0 — source and image proof

Require all of the following before boot:

1. The live file contains `_decode_forced_num_splits`.
2. The helper returns `1` for `is_spec_decode=True`.
3. Both `run_decode` calls pass the local `forced_num_splits` variable rather
   than `self._num_splits_cap` directly.
4. The MLA query-BMM contiguity fix is still present.
5. The CKV profile-reset fix is still present.
6. `python -m py_compile` passes for the live file.
7. Record image digest, complete file SHA-256, Compose SHA-256 and all other
   existing patch pins.

Leave:

```text
VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=auto
```

Do not disable the route with `0`; that would test the old fallback rather than
this fix. Remove CUDA launch blocking and CUDA-graph diagnostics unless they
are independently required by the current acceptance image.

## Gate 1 — boot and route engagement

Use the same known boot geometry:

```text
TP4 / DCP4 / MTP3
max_model_len=480000
max_num_seqs=16
max_cudagraph_capture_size=64
gpu_memory_utilization=0.980
B12X_MLA_SPARSE
nvfp4_ds_mla + KV_FP8_ROPE=1
i8_ring
DRAM offload + bounded NVMe tier unchanged
```

Require this INFO line:

```text
B12X MTP verifier decode uses one split for BF16-partial precision; ordinary decode retains up to 32 splits
```

Then require:

- profiling and production decode capture complete;
- API health returns a valid answer;
- MTP is live with nonzero accepted draft tokens;
- `RestartCount=0`, stable container ID and `StartedAt`;
- no illegal access, cuBLAS failure, OOM, Xid, assertion, EngineDead or worker
  death; and
- KV pool remains within normal allocator variance. The patch does not change
  scratch allocation, so it should not intentionally reduce the pool.

## Gate 2 — minimal quality proof

Run `needle_diag.py` with the established unique-prefix construction, needle
`738216` at 40%, deterministic decoding and a 3,000-token completion budget:

```text
50k
150k
300k
350k
475k
```

The critical discriminator is 300k:

- PASS means content includes `738216` with `finish_reason=stop`;
- a short coherent answer saying the ticket is absent is a real FAIL;
- `finish_reason=length` is a budget failure and must be rerun;
- never score empty `content` without recording reasoning content, finish
  reason and usage.

Stop and report if 300k fails. Do not spend the window on NVMe or throughput
qualification for a quality-failing process.

## Gate 3 — performance and stability

Only after all needles pass, run on the same process:

1. decode ctx0 at C1/C4/C8/C16;
2. decode ctx16k or ctx50k at C16;
3. cold 8k and 55k prefill with prefix-cache miss evidence;
4. the existing bounded NVMe fill/turnover/promotion gate; and
5. 16 x 50k overlapping unique-prefix stress.

Expected tradeoff:

- ordinary one-token decode is byte-for-byte on the old 32-split policy;
- MTP verifier attention has less split parallelism, so some speculative
  verification throughput may be traded for restored quality, especially at
  C1;
- high-concurrency verifier batches have enough row/head CTAs that the penalty
  should be smaller; and
- KV capacity should be unchanged because the 32-split scratch remains
  allocated for ordinary decode.

Record MTP acceptance rate and accepted-token throughput, not just aggregate
decode tok/s, so a change in draft acceptance is not mistaken for kernel speed.

## Verdict

PASS only if:

- all five needles return `738216` with `finish_reason=stop`;
- the process completes the stability gates with no restart or fatal signal;
- `i8_ring` remains positively identified with no fallback; and
- the performance report clearly separates ordinary decode, MTP acceptance
  and end-to-end throughput.

If this passes, the source root cause is confirmed and the patch is suitable
for a draft PR. If it fails at 300k, the BF16 split boundary is a real defect
but not sufficient to explain the field regression; preserve the process log
and do not promote the patch.
