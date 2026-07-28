# CN3 v20 go-live runbook

Status: **execute now**  
Operator: Fable  
Objective: boot and qualify the v20 production candidate, then leave the
qualified v20 engine online.

This is a production-forward run, not another differential experiment. Build
one candidate containing both narrow source fixes that are independently
required after the profiling-to-production transition:

1. cooperative admission for both barrier-bearing W4A16 fused grids; and
2. reset of profiling-generation CKV prefetch bindings before production KV
   cache installation.

The fixes touch different repositories and state machines. Neither disables
MTP, shared-expert overlap, CUDA graphs, DCP4, i8 ring, DRAM offload, NVMe
offload, MNS16, or the 64-token graph cap.

## 1. Build immediately

Use the exact Boot 7 combined v20 source/image as the build input. Retain its
vLLM MRV2 pool-reuse fix, aligned MTP block table, INT8 wire patch, bounded
filesystem tier, and all existing v20 code. Do not add the superseded
shared-expert-order patch.

Apply these two patches:

```text
patches/v20-w4a16-cooperative-grid/
  0001-fix-moe-cooperatively-launch-fused-W4A16-grids.patch
  SHA-256 1a76f60f8e8c4fd491412fabd66825b70cbca2151827c033088e99e974cc527e

patches/v20-ckv-prefetch-profile-reset/
  0001-fix-mrv2-reset-CKV-prefetch-cache-bindings.patch
  SHA-256 cdd456974545baf5381c9bb1cdd104bf86e63390ff5c26ddffa98027cf16d3b8
```

Required runtime byte checks:

```text
SparkInfer W4A16 input  7b15236dfd73c8eea6d692b661aa22f8e526c16f60f14551b6c43abcc6322e00
SparkInfer W4A16 output 7bae99dfff0ab8f61f1d2a0f36a401543f32a39e2e3982668fcc89a44e882f05

vLLM model_runner input  2eab8362e2ce3e1004941988347b9921072053d52198b6f88be2d98d03cdd779
vLLM model_runner output 526ac1643d9cbbf6e03fe505fdc64e4ab0c78bb898eb0b7013926e18a38ccf17
vLLM B12X MLA input      e06b35c88db6691c11cdef0e1a134746060682f596478365661847f681d4e0bb
vLLM B12X MLA output     ee08b603a266b752791ac3a811b23eb0680d9834d84d97323fcc11280f4e927d
```

Fail before building if an input byte does not match. Verify all three runtime
output bytes in the completed image. Tag the image uniquely and record its ID,
digest, and build-context commits.

If the SparkInfer test environment is available without delaying the image
boot materially, run this one targeted GPU regression before launch:

```bash
pytest -q tests/moe/test_fused_moe.py::test_run_w4a16_m9_graph_replay_with_prequeued_aux_work
```

It is useful preflight, but absence of repository tests in the runtime image is
not a blocker. The full engine boot is the authoritative proof.

## 2. Launch the production geometry

Start from the last v20 candidate Compose and replace only the image with the
new combined image. Restore the intended production A2A threshold; do not
carry Boot 5's diagnostic cap of 16 into production.

Required settings:

```text
TP=4
DCP=4
MTP=3
max_model_len=480000
max_num_seqs=16
max_num_batched_tokens=3072
max_cudagraph_capture_size=64
gpu_memory_utilization=0.980
VLLM_DCP_A2A_MAX_TOKENS=64
SPARKINFER_PCIE_DMA_FP8=i8_ring
VLLM_PCIE_DMA_FP8=i8_ring
DRAM tier=64,000,000,000 bytes
NVMe acceptance cap=8,589,934,592 bytes
```

Keep the known v20 production requirements unchanged: compact
`nvfp4_ds_mla`, `KV_FP8_ROPE=1`, CKV gather, A2A, prefix caching,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, `ipc: host`, and the
existing stale `/dev/shm/vllm_offload_*.mmap` cleanup.

Use the fresh CN3 NVMe namespace already staged for acceptance. Confirm it is
empty and on the NVMe-backed ext4 filesystem before launch. Do not delete or
reuse another namespace.

Remove these diagnostic settings if present:

```text
CUDA_LAUNCH_BLOCKING
VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS
```

Keep compile-progress logging. Allow active compilation to finish; abort only
for a real process exit, traceback, OOM, Xid, fatal assertion, or 15 minutes
with neither log progress nor compiler/GPU activity.

## 3. Boot gate

Watch continuously through production decode-speculator capture. Record the
KV pool and container identity.

PASS requires:

1. all target, prefill-speculator, and decode-speculator captures finish;
2. the API reaches serving state;
3. a deterministic liveness request returns `4`;
4. one normal MTP request returns `finish_reason=stop` with nonzero MTP
   acceptance;
5. `RestartCount=0` and unchanged container ID/`StartedAt`; and
6. no illegal access, EngineDead, worker exit, assertion, OOM, Xid, or 5xx.

The former M=9 failure is proven fixed when production decode capture passes
size 9 and continues through sizes 8 to 1. Do not change GMU, MNS, graph cap,
A2A threshold, or route selection during this boot.

On failure, preserve the complete first-run log and inspect JSON, stop the
container, and report the first failing operation. Do not start an automatic
tuning ladder.

## 4. Qualify on the same live process

As soon as the boot gate passes, keep the process running and execute in this
order:

1. one decode cell at concurrency 1, then 4, 8, and 16;
2. needles at 300k, 350k, and 475k;
3. bounded NVMe fill and eviction proof at the 8 GiB acceptance cap;
4. 16 x 50k unique-prefix overlapping stress;
5. cold 8k and 50k prefill with prefix-cache deltas; and
6. final liveness, restart/container identity, and fatal-signature audit.

Minimum production acceptance:

- every needle returns the expected value;
- all concurrency requests complete without 5xx or restart;
- NVMe bytes increase, stay within the configured bound, and an evicted block
  can be promoted;
- no illegal access, OOM, Xid, EngineDead, or offload assertion; and
- decode/prefill remain within the previously accepted v19/int8 bands.

## 5. Promote and leave v20 online

After first-process gates pass, perform the already-required controlled
restart for NVMe persistence. During that restart only:

1. retain the same image and all serving settings;
2. retain the same NVMe namespace;
3. raise the filesystem cap from the 8 GiB acceptance fixture to the chosen
   production limit (128 GiB if that remains the approved CN3 value);
4. do not clear the NVMe namespace; and
5. verify a retained prefix is restored/promoted after restart.

Then repeat liveness, one MTP request, one deep needle, and the
restart/container/error audit. If green, leave this v20 container online as
production and preserve the exact final Compose, image digest, byte pins, KV
pool, and acceptance report.
