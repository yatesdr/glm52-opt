# GLM-5.2 v20 on CN3 — test results, 2026-07-22/23 window

Operator: Fable · Host: CN3 (4× RTX PRO 6000, 96 GB, PCIe, no NVLink)
Window: 2026-07-22 21:09Z → ongoing · All times UTC

Facts only. Measured values, exact identifiers, and observed outcomes. Items not yet run are
marked NOT RUN rather than estimated.

---

## 1. Image under test

```
ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-vllm3e731bc-si1a88b38-int8-nvme-bt1876-mlaqbmm-7562bb27
digest sha256:16a4c78494586e6bfa14b8dc3fc32c2039ab5a48528d5026971153d1e3806479
local image ID sha256:16a4c78494586e6bfa14b8dc3fc32c2039ab5a48528d5026971153d1e3806479
size 37.5 GB · pushed 2026-07-23 03:03:30Z · public (anonymous pull verified HTTP 200)
```

Base: `voipmonitor/vllm:gilded-gnosis-v20-vllm3e731bc-si1a88b38-fi801d57a-cu132-20260722`
Engine version string: `v0.11.2.dev280+gilded.gnosis.v20.vllm3e731bc.si1a88b38.fi801d57a.cu132.20260722`

### 1.1 Overlays and byte pins (all verified in-image)

| Change | Ref | File | SHA-256 |
|---|---|---|---|
| Block-INT8 `i8_ring` PCIe wire | PR #69 | `sparkinfer/comm/pcie/pcie_dma.py` | `5a6e6a0ef72fd2e46d5b8a42106763817998d411cf4d55d2ecb127c63d9630d5` |
| " | PR #69 | `sparkinfer/comm/pcie/pcie_dma.cu` | `70f4be323350353bfe2df8c41c6129907a786f0ef25831a0b5604ef5e9161048` |
| Bounded NVMe fs-tier eviction | PR #165 | `vllm/v1/kv_offload/tiering/fs/manager.py` | `653edbf4b393e2acd6204bf4664c300eaee9e959656040864491c94548b4cc60` |
| Aligned MTP block table (1876) | PR #166 @ `c9a2e28d` | `vllm/v1/attention/backends/mla/indexer.py` | `d419af9e1e84a1fa316246d0cb930f6559dcb5859a8e71bb54240dbf41fb0cde` |
| MRV2 graph-pool reuse + CKV profile reset | PR #168 / `982cda45` + local | `vllm/v1/worker/gpu/model_runner.py` | `526ac1643d9cbbf6e03fe505fdc64e4ab0c78bb898eb0b7013926e18a38ccf17` |
| CKV profile reset + MLA query-BMM | local + `7562bb27` | `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` | `3ada9852c37b56cf1b0092ca86282119e6cf95be932ae6aac782c938ec74835a` |
| W4A16 cooperative grid | local | `sparkinfer/moe/_shared/kernels/w4a16/kernel.py` | `7bae99dfff0ab8f61f1d2a0f36a401543f32a39e2e3982668fcc89a44e882f05` |
| **MLA query-BMM contiguity** | `7562bb27` | `vllm/model_executor/layers/attention/mla_attention.py` | `54181f5fe83030c4fb6e8cb8d2315a0b55c83974ae5693d99035d21db878449f` |
| CUDA-graph descriptor diagnostic | `f0021cc3` | `vllm/v1/worker/gpu/cudagraph_utils.py` | `001365a061da3e5aeaa8bd3e7f04a41eac7b58de9020afbf4ae2d29be5b6a3f3` |

**Excluded (verified absent):** `bf1b32cf` MoE shared-expert aux-stream safety
(`shared_experts.py` = `5d1cc3158e2eddc5b3bfe88ecd50a390e18e9fe0c58fde0060c873783ab813b6`, unpatched);
`1b8e7f8f` superseded MTP routing guard.

The diagnostic (`f0021cc3`) is a no-op unless `VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS=1`.
It **was enabled** for the Gate 1 / Gate 2 process documented here.

---

## 2. Serving configuration

```
TP=4  DCP=4  MTP=3 (num_speculative_tokens, method=mtp, draft_sample_method=probabilistic)
max_model_len            480000
max_num_seqs             16
max_num_batched_tokens   3072
max_cudagraph_capture_size 64
cudagraph_capture_sizes  [1,2,4,8,16,24,32,40,48,56,64]
cudagraph_mode           FULL_AND_PIECEWISE
gpu_memory_utilization   0.980
kv_cache_dtype           nvfp4_ds_mla     KV_FP8_ROPE=1
quantization             nvfp4_nf3_hybrid
attention_backend        B12X_MLA_SPARSE
dcp_comm_backend         a2a       dcp_kv_cache_interleave_size 1
VLLM_DCP_A2A_MAX_TOKENS  64        VLLM_DCP_A2A_LARGE_BACKEND ag_rs
enable_prefix_caching    True      async_scheduling True
PYTORCH_CUDA_ALLOC_CONF  expandable_segments:False     ipc: host
default_chat_template_kwargs {'reasoning_effort': 'high'}
```

Offload tiers (`OffloadingConnector` / `TieringOffloadingSpec`):
```
cpu_bytes_to_use      64,000,000,000        (DRAM)
secondary fs tier     /nvme-kv/glm52-v20-acceptance
max_cache_size_bytes  8,589,934,592 (8 GiB) n_read_threads 16  n_write_threads 16
```

### 2.1 INT8 wire activation — verified at runtime

```
$ docker exec glm52-prod /opt/venv/bin/python -c "from vllm import envs; import os; ..."
enabled: True          (VLLM_ENABLE_PCIE_ALLREDUCE)
backend: b12x          (VLLM_PCIE_ALLREDUCE_BACKEND)
wire:    i8_ring       (VLLM_PCIE_DMA_FP8)
```

Boot log:
```
custom_all_reduce.py:564  Configured b12x PCIe crossovers: oneshot max=65536, fused max=86016, DMA min=6291456
cuda_communicator.py:272  Using ['B12X_PCIE_ONESHOT_DMA','PYNCCL'] all-reduce backends for group 'tp:0'
allreduce_rms_fusion.py   Using B12X PCIe fused all-reduce + residual-add RMSNorm
dcp_alltoall.py:157       Using B12X PCIe DCP collectives (world_size=4, max_batch_size=64, heads=64,
                          query_head_dim=576, output_head_dim=512)
```
Zero occurrences of `PCIe DMA allreduce unavailable`, `initialization failed`, or `Falling back to PyNCCL`.

In-container check: `_normalize_fp8_mode('i8_ring') -> 'i8_ring'` → `wire_mode = "int8-ring"`.
Note: `_normalize_fp8_mode` maps unrecognized input to `'ag'` (E4M3). `wire_mode` is logged at
DEBUG only, so no boot log in this window contains an engine statement of the wire actually used.

`VLLM_USE_B12X_PCIE_DMA=1` is present in the Compose and is obsolete/unused in v20 (flagged for
removal). It generates an "Unknown vLLM environment variable" warning that is cosmetic.

---

## 3. Boot ledger — 11 boots

| # | Time | Distinguishing change | Avail KV | KV pool | Outcome |
|---|---|---|---|---|---|
| 1 | 21:5xZ | v20 base+#69+#165+#166@`858d8225`, GMU 0.970 | 3.25 GiB | ~435,968 est | `ValueError` 480k needs 3.57 GiB. Graceful exit |
| 2 | — | GMU 0.980 | 4.14 GiB | 544,000 | illegal memory access @ spec capture |
| 3 | — | MNS 8 / cap 32 / GMU 0.978 | 4.41 GiB | 592,640 | illegal memory access, same site |
| 4 | 22:48Z | + MRV2 pool reuse `982cda45` | 4.14 GiB | 555,520 | illegal memory access, same site |
| 5 | 23:21Z | + A2A cap 16 → AG/RS | 4.19 GiB | 562,432 | illegal memory access, same site |
| 6 | 23:58Z | + PR#166 @ `c9a2e28d` (1876 aligned) | 4.13 GiB | 554,496 | illegal memory access, same site |
| 7 | 00:25Z | + CG_DIAG `f0021cc3` | 4.13 GiB | 554,496 | **fault localized** (§3.1) |
| 8 | 01:10Z | + W4A16 coop grid + CKV reset | 4.14 GiB | 555,520 | illegal memory access; decode bar 6/16 |
| 9 | 01:41Z | `CUDA_LAUNCH_BLOCKING=1`, GMU 0.970 | 3.19 GiB | — | `ValueError`; never reached capture |
| 10 | 01:53Z | `CUDA_LAUNCH_BLOCKING=1`, GMU 0.980 | 4.14 GiB | 544,000 | **failing op named** (§3.2); decode bar 7/16 |
| 11 | 02:31Z | **+ MLA query-BMM `7562bb27`** | **4.15 GiB** | **557,824** | ✅ **PASS — serving** |

Boots 2–8, 10 all terminated at the identical stack ending `input_batch.py:151 make_dummy`.

### 3.1 Boot 7 — descriptor localization

1,162 CG_DIAG records, 3 FAIL, one distinct failing tuple:
```
label=capturing_decode_cuda_graphs  stage=warmup_forward  cg_mode=FULL
num_tokens=9  num_reqs=9  uniform_token_count=1  max_req_tokens=None  num_active_loras=0
```
Production decode-speculator sizes 16→10 passed all five stages; **M=9 failed at `warmup_forward`**
after `warmup_inputs` passed. The identical descriptor passed all five stages during profiling.
Production target manager and prefill-speculator manager: 0 FAIL.

### 3.2 Boot 10 — failing operation

With `CUDA_LAUNCH_BLOCKING=1` the report moved from `make_dummy` to the true launch site:

```
gpu_worker.py:804 → model_runner.py:1013 capture_model → speculator.py:125 capture
 → autoregressive/cudagraph_utils.py:74 → cudagraph_utils.py:531 forward_fn(CUDAGraphMode.NONE)
 → speculator.py:567 _generate_draft → speculator.py:438 _run_model
 → deepseek_mtp.py:676 forward → mla_attention.py:2017 unified_mla_attention_with_output
 → torch/_ops.py:1275 → kv_transfer_utils.py:48 → mla_attention.py:1214 forward_impl
 → torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)
```
Zero MoE frames (`sparkinfer|fused_moe|shared_experts|moe_runner|nvfp4|grid188|w4a16`) in the log.

---

## 4. Gate 1 — boot + integrity (Boot 11) — PASS

Container `9ee8a2f1e1622ad8c1ebe64ddeb982028cdb96343605f1a1c8b4109f93109b21`
StartedAt `2026-07-23T02:31:22.47360076Z` · **RestartCount 0** · Health `healthy`

| Criterion | Result |
|---|---|
| Profiling decode capture, all 16 | PASS (16→1) |
| Production decode capture 16→1 | PASS — `16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1` |
| **M=9 all stages** | PASS ×5 in **both** profiling and production rounds |
| CG_DIAG totals | **624 BEGIN / 624 PASS / 0 FAIL** |
| API serving | `Application startup complete` |
| Liveness (`2+2`, temp 0) | content `'4'`, `finish_reason=stop`, 25 prompt / 62 completion tok |
| MTP acceptance | 46/48 draft tokens accepted (**95.8%**), 16 drafts |
| Fatal-signature audit | illegal-access 0 · cuBLAS 0 · EngineDead 0 · OOM 0 · Xid 0 · assertion 0 · worker-died 0 |

Memory at Gate 1: MRV2 `0.39 GiB additional`, **Available KV 4.15 GiB**, **GPU KV pool 557,824
tokens**, max concurrency **1.16× @ 480,000**.

---

## 5. Gate 2 — qualification on the same live process (no restart)

### 5.1 Throughput — COMPLETE, 0 errors in 10/10 cells

Harness: `llm_decode_bench.py`, identical invocation to the v19 `i8_ring` baseline
(`--concurrency 1,2,4,8,16 --contexts 0,16k --max-tokens 512 --duration 45 --standalone-prefill
--prefill-contexts 8k,55k --prefill-metric auto`).

**Prefill (cold, unique prefix, `cached_tokens=0`):**

| Context | Client tok/s | Server (Prometheus) tok/s | TTFT |
|---|---|---|---|
| 8,192 | **1,680** | 1,702 | 4.88 s |
| 56,320 | **1,432** | 1,442 | 38.725 s |

**Decode aggregate tok/s:**

| ctx \ conc | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| **0** | 61.9 | 71.7 | 99.2 | 124.2 | **156.8** |
| **16,384** | 63.6 | 71.3 | 95.5 | 118.3 | **168.6** |

Per-cell: `num_errors=0` in all 10. Effective concurrency 0.9/1, 2.0/2, 3.9–4.0/4, 7.7–7.8/8,
15.7–15.8/16. Server spec-accept rate 0.385–0.963 (rises with context; 0.75 at ctx16k C16).

**Comparison to v19 `i8_ring` production baseline:**

| Metric | v19 | v20 | Δ |
|---|---|---|---|
| KV pool | 644,864 | 557,824 | −13.5% |
| Prefill 8k | 1,607 | 1,680 | **+4.5%** |
| Prefill 55k | 1,641 | 1,432 | **−12.7%** |
| Decode ctx0 C1 | 63.2 | 61.9 | −2.1% |
| Decode ctx0 C8 | 127.2 | 124.2 | −2.4% |
| Decode ctx0 C16 | 165.1 | 156.8 | −5.0% |

### 5.2 Needle retrieval — FAIL

Ladder (`quality_gate.py`, `reasoning_effort: high`), 03:26:19Z–03:39:20Z, RC 0 throughout:

| Depth | Needle | Arithmetic | Coherence | Gate |
|---|---|---|---|---|
| 300,000 | **FAIL** (`''`) | PASS | PASS (190 words, trigram-repeat 0.00) | FAIL |
| 350,000 | **FAIL** (`''`) | PASS | PASS (193 words, trigram-repeat 0.00) | FAIL |
| 475,000 | **FAIL** (`''`) | PASS | FAIL (0 words) | FAIL |

Diagnostic re-run (`needle_diag.py`, `--max-tokens 3000 --effort low`) — depth sweep:

| Depth (actual ctx) | needle_in_content | needle_in_reasoning | finish | completion_tok | Result |
|---|---|---|---|---|---|
| 50,000 (~49,952) | True | False | stop | 90 | **PASS** — `738216` |
| 150,000 (~149,968) | True | False | stop | 96 | **PASS** — `738216` |
| 200,000 (~199,976) | True | False | stop | 92 | **PASS** — `738216` |
| 250,000 (~249,984) | True | False | stop | 107 | **PASS** — `738216` |
| **300,000** (~299,992) | **False** | False | stop | 219 | **MISS** |

Content at 300k: *"The maintenance ticket number for the Facility 27 compressor overhaul is not
men[tioned]…"* — `finish_reason=stop`, not truncated. A first 300k diagnostic run returned
`completion_tok=18` with empty content and empty reasoning; the second returned the coherent
"not mentioned" answer above. Both are MISS.

**Measured breakpoint: PASS ≤ 250,000 · MISS ≥ 300,000.**

v19 `i8_ring` baseline on the same needle construction: **5/5 PASS at 50k / 200k / 300k / 350k / 475k.**

### 5.3 KV offload tiers — eviction works; capacity bound BREACHED transiently (harness FAIL)

#### 5.3.1 At-rest measurement after throughput + needle workloads (organic fill from empty)

| Metric | Value |
|---|---|
| NVMe bytes on disk (idle) | **8,589,116,068** |
| Configured cap | 8,589,934,592 (8 GiB) |
| Ratio at rest | 0.9999 (818,524 B under cap) |
| Files | 1,050 |
| mtime span | 02:50:20Z → 03:47:53Z (active turnover) |
| DRAM written GPU→CPU | **48,452,081,664 B (48.45 GB)** |
| DRAM read CPU→GPU | **5,656,426,496 B (5.66 GB)** |
| Offload time GPU→CPU / CPU→GPU | 12.138 s / 1.048 s |

Filesystem: `/dev/nvme0n1p2`, ext4, 828 G free.

**The at-rest ratio above does NOT establish bound compliance.** It was sampled while the engine was
idle. The formal harness samples during active fill and records breaches (§5.3.2).

#### 5.3.2 Formal harness — `nvme_kv_eviction_acceptance.py fill` — **FAIL**

Run 03:49:52Z–03:58:27Z, `--allow-existing`, state dir
`~/glm52-test-artifacts/v20-combined-20260723-01`. RC 0 throughout.

```
Calibrated 4162 filler repetitions to 50009 chat prompt tokens
Sentinel completed in 31.41s
Fill  1/32 .. 10/32   bytes=8589111296   nominal 3.19 GB -> 17.56 GB
Replay anchor completed in 31.102s
FAIL: _model_a1f720d91ee6_r0 exceeded capacity:
      8605487104 > 8589934592
      8597299200 > 8589934592
```

Capacity sample series (`capacity-samples.csv`, 306 samples):

| Quantity | Value |
|---|---|
| `completed_bytes` max | **8,605,487,104** |
| `completed_bytes` min | 8,523,608,064 |
| `limit_bytes` | 8,589,934,592 (constant, correct) |
| **Samples over cap** | **5 / 306** |
| **Max overshoot** | **15,552,512 B (0.181%)** |
| Second distinct overshoot | 7,364,608 B (0.086%) |

**Eviction itself works.** `sentinel_missing` progressed 0 → 0 → 0 → 0 → **121** → **195** across
fills 1–6 and held at 195 through fill 10, i.e. planted sentinel blocks were evicted by turnover as
intended, which is the required precondition for the `replay` promotion test.

#### 5.3.2a WITHDRAWN: the capacity "FAIL" is a monitor artifact, not a real breach

Filesystem block size **8,187,904 bytes**. Every observed value is an exact whole-block multiple:

| Blocks | Bytes | Observed as |
|---|---|---|
| 1,049 | **8,589,111,296** | stable cache size |
| 1,050 | **8,597,299,200** | "breach 1" (+1 block) |
| 1,051 | **8,605,487,104** | "breach 2" (+2 blocks) |

`8,589,934,592 / 8,187,904 = 1049.10` → 1,049 blocks is the largest whole-block count under the cap.

The acceptance monitor scans the directory recursively while eviction and replacement are in
progress, so it can count an old file early in a scan and its replacement later even though the two
never coexisted on disk. The tested `manager.py` (`653edbf4…`) reserves pending bytes under its
capacity lock before performing I/O, and every quiescent checkpoint was below the cap
(at-rest measurement §5.3.1: 8,589,116,068, i.e. 818,524 B under).

Sol corrected `harness/nvme_kv_eviction_acceptance.py:427` — live non-atomic observations are still
recorded for diagnostics but no longer raise failures; stable temp-free quiescent snapshots still
enforce the limit. Corrected harness SHA-256
`f059e25c894c8269941dbaf7549e665c0686fe6566e57317825efa6b8aa92a79`.

**Corrected NVMe verdict:**
- Eviction / turnover: **PROVEN**
- Stable bounded capacity: **strongly supported**; formal PASS requires a rerun on the corrected harness
- Restart persistence and NVMe→DRAM→GPU promotion: **NOT YET PROVEN**
- No manager/source patch is justified by these excursions. PR #165 remains draft pending the
  promotion/persistence result.

**Operator note:** the earlier text in this section called this a transient capacity breach and
attributed it to bound-after-write ordering. That interpretation was not supported by the data and
is withdrawn. The measured numbers were correct; the reading of them was not.

Because the namespace was already at cap when this run started, it exercises hot-cache turnover, not
fill-from-empty; fill-from-empty is covered by §5.3.1.

### 5.3.3 16 × 50k unique-prefix stress — criteria PASS; overflow NOT achieved

Run 04:01Z–04:09:20Z. 16 concurrent requests, unique prefix each (`cached_tokens=0` on all 16),
49,119–49,123 prompt tokens each (~786k total vs 557,824-token pool = 1.41× nominal), `max_tokens=128`,
temperature 0, each carrying a verifiable expected answer.

| Criterion | Result |
|---|---|
| Completed | **16/16**, `num_errors=0` |
| `finish_reason` | `stop` ×16 |
| **Answer match** | **16/16 exact** |
| RestartCount | **0** (container unchanged since 02:31:22Z) |
| Fatal audit | illegal-access 0 · EngineDead 0 · OOM 0 · Xid 0 · AssertionError 0 · offload-assert 0 · 5xx 0 |
| Wall time | 477.29 s |
| Latency spread | 36.28 s → 477.25 s (near-linear staircase) |
| DRAM offload GPU→CPU | 67.61 GB → **92.62 GB (+25.0 GB)** |
| DRAM offload CPU→GPU | 10.96 GB → 10.96 GB (unchanged) |
| NVMe bytes | 8,589,116,068 → 8,589,116,068 (unchanged, under cap) |

**Measured scheduler occupancy.** Sampler ran every 10 s for the full run; **48 samples** collected.

`num_requests_running` — full distribution across all 48 samples:

| Value | Samples |
|---|---|
| 0.0 | 1 |
| 1.0 | 29 |
| 2.0 | 18 |
| >2.0 | **0** |

`num_requests_waiting` — drained monotonically **15 → 0** (≈3 samples at each integer depth),
confirming all 16 requests were submitted and queued at t₀ rather than released gradually by the
client. First sample: `running=1.0, waiting=15.0`. Last sample: `running=0.0, waiting=0.0`.

`kv_cache_usage_perc` — max **0.13820018365472908**; top observed values 0.0941, 0.1102, 0.1107,
0.1162, 0.1217, 0.1382.

`num_preemptions_total` — **0.0 in every sample**, unchanged from the pre-run baseline.

Configuration context: `max_num_batched_tokens=3072` with chunked prefill; each ~49k prompt requires
≈16 scheduler steps of prefill alone.

*Operator interpretation (not a measurement):* concurrency capped at 2 with KV at 13.8% and zero
preemptions indicates the requests were largely serialized, so this run did not place the GPU KV
cache under overflow pressure. Whether that meets the step-4 objective ("force GPU-KV overflow")
is a judgement for the spec owner; the measurements above are the record.

### 5.4 Gate 2 outcome

| Step | Result |
|---|---|
| Throughput (10 cells) | **PASS** — 0 errors, within band |
| Needle retrieval | **FAIL** — MISS ≥300k (v19 baseline: 5/5 through 475k) |
| NVMe bounded eviction | **FAIL** — 5/306 samples over cap, max +0.181% |
| NVMe eviction mechanism | works (`sentinel_missing` 0→195) |
| 16×50k stress (correctness/stability) | **PASS** — 16/16 ok, 16/16 correct, RC 0, clean audit |
| 16×50k stress (measured occupancy) | peak concurrency **2/16**, peak KV **13.8%**, preemptions **0** (48/48 samples) |

### 5.5 NOT RUN

- 16 × 50k overlapping unique-prefix stress
- Cold 8k/50k prefill with prefix-cache-miss deltas (beyond §5.1)
- `replay` phase / persisted NVMe promotion after restart
- Final liveness + cache-inventory audit
- Gate 3 persistence restart at 128 GiB NVMe cap

---

## 6. Environment notes observed during the window

- v19's 64 GB offload mmap survives `docker compose down` as a root-owned file in `/dev/shm` and
  must be removed with elevated privileges before a subsequent 64 GB tier can allocate.
- Docker log driver is `json-file`, `max-size 50m`, `max-file 3`. Boot logs ran 0.6–0.8 MB.
- `CUDA_LAUNCH_BLOCKING=1` changed the reported KV pool from 555,520 (Boot 8) to 544,000 (Boot 10)
  at identical 4.14 GiB available.
- GMU 0.970 yielded 3.19 GiB available at Boot 9 vs 3.25 GiB at Boot 1 (same GMU, later patches).

## 7. Evidence files (CN3 `~/glm52-test-artifacts/v20-window-evidence/`)

```
v20-mrv2fix-FAILED.log / -inspect.json          (Boot 4)
v20-boot5-a2a16-FAILED.log / -inspect.json      (Boot 5)
v20-boot6-bt1876-FAILED.log / -inspect.json     (Boot 6)
v20-boot7-cgdiag.log / -inspect.json            (Boot 7, 1,162 CG_DIAG records)
v20-golive-FAILED.log / -inspect.json           (Boot 8)
v20-boot9-lb-gmu970.log / -inspect.json         (Boot 9)
v20-boot10-lb-gmu980.log / -STREAM.log / -inspect.json  (Boot 10)
v20-boot11-mlaqbmm-PASS.log / -STREAM.log       (Boot 11)
~/bench/v20-gate2/throughput.txt, v20-thru.json, needles.txt, nvme-fill.txt
```

## 8. Current state

- v20 Boot 11 process **live and serving**, RC 0, health `healthy`, container unchanged since
  02:31:22Z.
- v19 production **down by choice** since 21:09Z. Rollback armed and unused:
  `glm52-prod-ring.yaml`, image `ca8481…`, warm cache, ~12–15 min to serve.
- No user traffic observed since 19:57:50Z on 2026-07-22.
