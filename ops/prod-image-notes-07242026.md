# Production image notes — 2026-07-24

> **Superseded candidate notice — 2026-07-24 23:30 EDT**
>
> The memory-reclaim-only candidate below is retained as history but must not
> be built. The current production integration is
> `workspace/vllm-v20-staged-bf16-fp8-query` at
> `e16288f5d006726b0492981d3db5627fc5d9f70e`, plus
> `workspace/sparkinfer-v20-current-recovery` at
> `d4969d993cdd16cc417056d471af42d10ac3fada`.
>
> It stages the query guard for Proof 3 selection, includes the now-selected
> PR #171 and PR #168, and overlays the #76 storage-validation follow-up.
> SparkInfer PR #76 itself is already present in the base as `cd089a4`, with a
> stable patch-id identical to its upstream head. The production posture is
> GMU 0.978, CKV cap 480,000, graph sizes
> `1,2,4,8,16,32,64`, MNS16, 64 GB DRAM, and bounded 64 GB NVMe.
>
> The active artifacts are:
>
> ```text
> docker/Dockerfile.v20-prod-ready-20260724
> deploy/glm52-v20-prod-ready-20260724.yaml
> v20-prod-ready-20260724-fable-handoff.md
> v20-pr-ledger-20260724.md
> ```
>
> Do not build until `fable-sol-comms.md` contains an explicit
> `FINAL BUILD PIN`; the final no-model top-k/query proof is still selecting
> the last functional delta.

## Next v20 test candidate

The next production-shaped test image is prepared but has not yet been built or
booted. Fable will build and validate it on CN4.

### Published base

```text
image:       voipmonitor/vllm:gilded-gnosis-v20-vllm992b874-sia93df67-fi801d57a-cu132-20260724
digest:      sha256:adddafd2b1749729fdf2d2ca23818c7c39f2a95e6fb05edd98657251913b83f2
vLLM:        992b874cf7ae504616bbb1d2d4f7a7355be6972b
SparkInfer:  a93df671cc7b33734f499b57228e542c3d3c3697
```

This is the newest published Festr v20 base as of preparation time. Its
SparkInfer revision includes the widened, exact long-context top-k path, so the
derived image must retain the base's compiled SparkInfer artifacts unchanged.
The top-k commit has the same stable patch-id as current SparkInfer `83a5844`;
it is not a missing upstream fix. The base also includes the functional PR
#172 startup-profile change as `cb27f671d`. PR #168 is complementary
graph-pool lifetime accounting, not a duplicate of #172.

### Integration source

```text
repository:  https://github.com/yatesdr/vllm-opt
branch:      integration/v20-memory-reclaim-20260724
revision:    7373bb24c881fa05af57d7eaf8aa7b4e9f2d2ddb
```

The branch contains:

1. PR #165 — bounded filesystem/NVMe-tier capacity and eviction.
2. The field-proven SM120/B12X native-MTP flattening gate.
3. A conflict-resolved forward port of PR #154 — release dead MXFP8
   `kv_b_proj` source storage after the absorbed MLA weights have been
   materialized.

The PR #154 port preserves Martin Vit's implementation and authorship. The only
merge conflict was adjacent unit-test coverage introduced by the newer fused
query source. Both test sets were retained and adapted to the new wrapper.

## Memory policy changes

The test configuration incorporates the current memory-recovery recommendations:

```text
VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=480000
cudagraph capture sizes=1,2,4,8,16,32,64
gpu_memory_utilization=0.976
max_model_len=480000
max_num_seqs=16
max_cudagraph_capture_size=64
```

Expected mechanisms:

- PR #154 previously recovered approximately 290 MiB/GPU without a measured
  throughput, MTP-acceptance, or KLD regression.
- Bounding CKV gather storage to the real 480k endpoint saves approximately
  38.9 MiB/GPU.
- Removing intermediate CUDA-graph capture sizes frees persistent graph
  allocations while retaining power-of-two service points through 64.
- GMU 0.976 restores 0.4 percentage points of device memory as runtime safety
  headroom compared with 0.980.

The expected KV improvement must be verified at runtime; it is not recorded as
a result until CN4 supplies the measured resident-memory and KV-allocation
lines.

## Preserved production behavior

The candidate retains:

- TP4 / DCP4 / MTP3
- maximum model length 480,000
- maximum sequences 16
- block-INT8 `i8_ring` PCIe wire mode
- B12X MLA sparse attention and MoE
- honest CUDA-graph memory profiling
- 64 GB DRAM KV-offload tier
- bounded 64 GB NVMe filesystem tier
- the NVFP4/NF3 hybrid checkpoint and compact FP8-RoPE KV format

It deliberately does **not**:

- enable `VLLM_B12X_ABSORB_BMM=1`;
- disable CUDA-graph memory profiling;
- switch DCP routing merely to inflate the displayed KV pool;
- carry CUDA-graph diagnostics; or
- enable autoheal/restart during the first acceptance boot.

The acceptance container uses `restart: "no"` so a failed first process remains
visible and cannot silently consume another boot cycle.

## Prepared artifacts

```text
docker/Dockerfile.v20-memory-reclaim-test
deploy/glm52-v20-memory-reclaim-test.yaml
harness/v20_memory_reclaim_unit_proof.py
v20-memory-reclaim-test-fable-handoff.md
```

Artifact hashes at preparation time:

```text
Dockerfile:  cbda6c69ff62d6d2ee7bb8f2f8eec1cedc0ba34ef54529dc3bad7db34c49037f
Compose:     b98a3f74905a595065dd32404db6158e6b4fd39804a07e8ed4ac3faa9dd382c1
CPU proof:   fc77ccfea9b799395ca4ee2746f8676da7821a43dbec49f5783ca2a450ff83cd
Handoff:     96e508cd3fb0409f94fb0cf5fc43905019eb8e61a3dc8d5c5a2a127e460ed39b
```

The Dockerfile is fail-closed. It verifies:

- all five expected base Python-file hashes;
- the untouched MRV2 model-runner hash;
- presence of the compiled `safe_mla_query_bmm` operation;
- Python compilation of every overlaid runtime file; and
- all five expected output hashes.

Any input or output hash mismatch is a hard build failure and must be
investigated rather than bypassed.

## Validation completed locally

Completed:

- exact Festr base and source revisions resolved;
- PR #154 forward-ported over the new fused-query source;
- Python syntax compilation passed for all changed production files and tests;
- `git diff --check` passed;
- source worktree was clean at the pinned revision;
- Compose resolution passed with all required paths supplied;
- integration branch was pushed to the `yatesdr/vllm-opt` fork.

Not yet completed:

- CN4 image build;
- in-image focused CPU proof;
- GPU/model boot;
- measured KV-pool result;
- throughput, needle, MTP, offload, and concurrency qualification;
- image push and final registry digest pin.

## CN4 acceptance summary

Use the full procedure in `v20-memory-reclaim-test-fable-handoff.md`. All
qualification should run on one live process without rebuilding or rebooting
between cells.

Hard boot requirements:

- restart count remains zero and container identity is unchanged;
- no OOM, illegal access, cuBLAS failure, assertion, EngineDead, Xid, or dead
  worker;
- profiling and production MTP graph capture both complete;
- finalized API response is non-empty;
- GPU KV pool is at least 480,000 tokens;
- target KV pool is at least 500,000 tokens;
- i8-ring, CKV limit 480,000, graph-size list, GMU 0.976, MNS16, and 480k
  maximum context are confirmed from the live configuration.

Qualification on that same process:

1. Cold 8k and 55k prefill with prefix-cache miss evidence.
2. Decode C1/C4/C8/C16, including long-context cells.
3. Cold needles at 50k, 250k, 350k, and 475k. Score all reasoning fields but
   require finalized, non-empty `content`.
4. Ordered-event NVMe capacity monitoring, counting completed and temporary
   files against the 64,000,000,000-byte bound.
5. Overlapping 16×50k unique-prefix stress.
6. Final health check on the original container identity.

If the hard boot gate fails, stop after that first attempt and preserve the
complete evidence. Do not begin a GMU or configuration ladder during the same
window.
