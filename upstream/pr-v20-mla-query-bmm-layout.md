## Summary

Restore the B12X MLA query-absorption layout contract that was removed when
DCP attention outputs moved to a head-major layout.

The head-major DCP change correctly made the old V-up output copy unnecessary,
but query absorption is a separate BMM:

```python
mqa_q_nope = mqa_q_nope.transpose(0, 1)
torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)
```

`mqa_q_nope` is a non-contiguous split-and-transpose view. On the v20
TP4/DCP4/MTP3 stack, production decode-graph warmup reproducibly raised a CUDA
illegal-address error at this BMM, while the profiling capture of the same
descriptors succeeded.

This patch:

- lets B12X request contiguous query-BMM input and absorbed weights;
- materializes the query operand immediately before the affected BMM;
- prevents compatible-but-strided absorbed-weight storage from being reused;
  and
- preserves weight addresses when the existing storage already satisfies the
  contract.

It deliberately does not restore the old V-up output temporary. The newer
head-major DCP output path remains unchanged.

## Root cause

`b3ea2e8f` / #136 originally added backend-selected contiguous MLA BMM
operands after a cuBLAS read-ahead failure. `6a2edcf1` subsequently kept DCP
attention outputs head-major and removed all three B12X contiguity flags.

That removal was valid for the superseded V-up output layout, but it also
removed protection from the independent query-absorption BMM. The v20 source
still constructs its first operand with `split(...).transpose(0, 1)` and
passed that view directly to `torch.bmm`.

With `CUDA_LAUNCH_BLOCKING=1`, the previously asynchronous boot failure
localized to that exact launch:

```text
speculator capture
  -> deepseek_mtp.py: forward
  -> mla_attention.py: forward_impl
  -> torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)
  -> CUDA error: an illegal memory access was encountered
```

## Validation

### Unit tests

```bash
python -m pytest -q tests/v1/attention/test_mla_backends.py -m cpu_test
# 11 passed
```

The same suite passed twice:

1. on the v20 image source; and
2. after applying the formatted patch to the later CKV-reset candidate source.

New coverage verifies backend flag propagation, materialization of the exact
split-and-transpose query operand, compatible contiguous-weight reuse, and
replacement of strided absorbed-weight storage.

### TP4/DCP4/MTP3 runtime proof

Configuration:

```text
GLM-5.2, TP4 / DCP4 / MTP3
max_model_len=480000
max_num_seqs=16
max_cudagraph_capture_size=64
gpu_memory_utilization=0.980
B12X_MLA_SPARSE
nvfp4_ds_mla + KV_FP8_ROPE=1
i8_ring
DRAM + bounded NVMe KV offload enabled
```

Before the patch, the production decode-speculator capture failed
reproducibly across the configuration and memory-control runs. Descriptor
M=9 was the first diagnostic failure; a launch-blocking run named the BMM
above as the first failing CUDA operation.

Patched result:

```text
profiling decode capture:   sizes 16 -> 1 PASS
production decode capture:  sizes 16 -> 1 PASS
CG_DIAG boundaries:         624 PASS / 0 FAIL
M=9:                        all five stages PASS in both rounds
API/liveness:               PASS
MTP acceptance:             46/48 draft tokens (95.8%)
GPU KV pool:                557,824 tokens (1.16x at 480k)
RestartCount:               0
illegal access/cuBLAS/OOM/
Xid/EngineDead/assertion:    0
```

The image byte-verified both patched output files. The separate experimental
MoE overlap patch was not present, isolating this fix as the change that
cleared the boot failure.

Extended throughput, long-context needle and offload qualification is running
on the same live process and will be added when complete.

## Scope and tradeoff

The added copy is limited to the query-absorption operand selected by the
B12X backend. At the reproduced M=9 descriptor it is a small
head-major query tensor; it does not copy KV state or DCP attention output.

This patch does not change MTP, graph sizes, A2A/AG-RS routing, INT8 wire mode,
CKV prefetch, MoE scheduling, GPU-memory accounting, or KV offloading.
