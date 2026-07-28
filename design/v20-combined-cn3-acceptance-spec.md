# CN3 combined v20 + INT8 wire + concurrency + NVMe acceptance

Status: **RETIRED — DO NOT EXECUTE**  
Owner/operator: Fable  
Target: GLM-5.2 v20, TP4/DCP4, 480k, `i8_ring`, shared-expert concurrency
fixes, 64 GB DRAM offload, and bounded NVMe filesystem tier  
Date prepared: 2026-07-22

> This specification predates the 2026-07-25 safe-query post-FP8 regression,
> the current DCP-final base, the 500,000-token production pool floor, and the
> corrected exact-finalization harness. Its image, source pins, expected
> 644,864-token pool, and `quality_gate.py` acceptance are obsolete. It is
> retained as historical NVMe/INT8 procedure evidence only.
>
> Do not use it to promote CN3. The replacement contract is staged in
> `design/cn3-promotion-post-causal-delta.md` and can be finalized only after
> the authorized numeric and causal window resolves the precision candidate.

## 1. Decision and objective

Use one long-lived v20 engine process to collect as much independent evidence
as possible. A second boot is unavoidable because persisted filesystem-cache
promotion can only be proven after GPU and DRAM state are discarded.

The run must prove, in this order:

1. the exact v20 image and patched runtime bytes boot at 480k;
2. the fresh filesystem tier fills, remains bounded at 8 GiB, and evicts old
   unpinned blocks;
3. the `i8_ring` implementation is selected for eligible prefill all-reduces;
4. block INT8 retains the needle at the E4M3 failure boundary and full depth;
5. v20 remains stable under the known 16 x 50k shared-expert/offload pressure;
6. prefill, decode, and GPU KV-pool capacity do not materially regress; and
7. after one controlled restart, a retained NVMe block is promoted and counted
   as an external prefix-cache hit.

KLD is explicitly out of scope for this window. It is useful follow-up evidence
but is not an acceptance dependency for the INT8 wire change.

This procedure intentionally tests several already-reviewed changes together.
That is the operator's choice to maximize evidence per expensive v20 boot. If a
gate fails, preserve all artifacts before using the isolated v19 NVMe procedure
in `nvme-capacity-eviction-cn3-fable-test-spec.md` to separate variables.

## 2. Fixed artifacts and byte pins

### Candidate image

```text
ghcr.io/yatesdr/glm52-serve@sha256:bc4b0185bb72a2722a57e615587f67bf35cd16554720514d47ee0184051a4cd7
```

Expected tag/comment: `gilded-gnosis-v20-int8-block-nvme`.

The candidate Compose source reviewed on the Mac was:

```text
/Users/derek/Downloads/glm52-v20-prod-candidate.yaml
SHA-256 4b886894d16e1ff879ef971ef5bf0fc7f66671c8b1fae6b4b3ceb024d04951c9
```

Preserve the final CN3 Compose and hash it. Its hash will differ after the
required CN3/NVMe edits below.

### INT8 wire patch

- SparkInfer branch: `feat/sparkinfer-v20-int8-wire`
- commit: `d6f0baa3107b3e774c047945234188f75636da9a`
- PR: local-inference-lab/b12x #69
- expected `pcie_dma.py` SHA-256:
  `5a6e6a0ef72fd2e46d5b8a42106763817998d411cf4d55d2ecb127c63d9630d5`
- expected `pcie_dma.cu` SHA-256:
  `70f4be323350353bfe2df8c41c6129907a786f0ef25831a0b5604ef5e9161048`

### vLLM PCIe integration

Expected `custom_all_reduce.py` SHA-256:

```text
1a15c6266e2f7eb1a64b74d2db1504663e1535ff411e63a9eeae1f0abe6349c6
```

### Bounded filesystem tier

- commit: `d74c0a6c397a76dd1e0aede8b1c8927c9a7c74ac`
- expected `manager.py` MD5:
  `a72eeb81c735036b281ff97f5d759122`
- expected `manager.py` SHA-256:
  `653edbf4b393e2acd6204bf4664c300eaee9e959656040864491c94548b4cc60`

### Acceptance harnesses

```text
7df4b95926816fac96139531c1bc25be4b1e9a4a569360073ef2aea80b9b595f  nvme_kv_eviction_acceptance.py
97cb98fad95efe94ff71582ab163fa09d19c386df8029e182a2b39beba7cb0c3  prefill_bench.py
56373a368c88853d2db0bb46d5eaab1614ff64873321dc51e6641065515640da  quality_gate.py
```

Copy them from `/Users/derek/glm52-opt/harness/` and verify on CN3 before the
window. Run Python through `uv` or an existing project virtual environment, not
the system interpreter.

## 3. Required Compose edits

Start from the reviewed v20 candidate and make only these operational edits:

1. remove the duplicate `pull_policy: missing` line;
2. map the CN3 NVMe host directory:

   ```yaml
   - /home/claude/nvme-kv:/nvme-kv:rw
   ```

3. use a fresh acceptance namespace and the 8 GiB test limit:

   ```json
   {
     "type": "fs",
     "root_dir": "/nvme-kv/glm52-v20-acceptance",
     "n_read_threads": 16,
     "n_write_threads": 16,
     "max_cache_size_bytes": 8589934592,
     "locality": "LOCAL"
   }
   ```

4. add this diagnostic-only environment setting so a quiet first compile is
   visibly making progress:

   ```yaml
   - SPARKINFER_PRINT_COMPILE_PROGRESS=1
   ```

Keep all of the following exactly as reviewed:

- image by digest;
- `ipc: host` and stale `/dev/shm/vllm_offload_*.mmap` cleanup;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`;
- `PYTHONHASHSEED=0`;
- `SPARKINFER_PCIE_DMA_FP8=i8_ring` and `VLLM_PCIE_DMA_FP8=i8_ring`;
- `TORCH_EXTENSIONS_DIR=/cache/v20_sparkinfer_ext`;
- `cpu_bytes_to_use=64000000000`;
- TP4/DCP4, `--dcp-comm-backend=a2a`, MTP3;
- GMU `0.970`, max model length `480000`, MNS `16`, MNBT `3072`;
- compact `nvfp4_ds_mla` KV cache and prefix caching; and
- the v20 shared-expert stream behavior. Do **not** add
  `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`; exercising the v20 concurrency fixes
  is one of this run's objectives.

The 8 GiB filesystem cap is an acceptance fixture, not a production sizing
recommendation. Keep the known-good 64 GB DRAM primary tier: every new primary
store is cascaded to the filesystem tier, so DRAM does not need to fill before
NVMe receives blocks.

## 4. Window budget and priorities

Expected core duration:

| Work | Expected time |
|---|---:|
| Preflight before stopping production | 5-10 min |
| First v20 boot with cold compile cache | 35-45 min |
| Bounded NVMe fill/eviction | 10-12 min |
| Minimal prefill/decode cells | 4-8 min |
| 300k, 350k, and 475k needles | 12-18 min |
| 16 x 50k overlapping concurrency | 10-12 min |
| Controlled restart with warm compile cache | 7-15 min |
| Persisted replay and final smoke | 2-4 min |

Plan on **80-110 minutes** for the core proof. The full optional matrix can
extend the run to roughly two hours. The first boot dominates; do all first-boot
gates before restarting.

Priority when time is short:

1. boot/integrity and GPU-pool fit;
2. NVMe fill/eviction;
3. i8 route proof plus 300k/350k/475k needles;
4. 16 x 50k concurrency;
5. controlled restart and persisted replay;
6. minimal throughput cells;
7. optional expanded performance and general-quality cells.

The restart/replay gate must not be omitted after a passing fill; otherwise
persistence and promotion remain unproven.

Do not treat active compilation as a hang. Abort only for a worker/EngineCore
exit, traceback, OOM, Xid, fatal assertion, unexpected process replacement, or
at least 15 minutes with no log progress **and** no active compiler/linker/GPU
work. If compilation is visibly active, allow the full 45-minute cold-boot
budget.

## 5. Gate 0 — preflight before the GPU window

Create a fresh host path without deleting or reusing a production namespace:

```bash
mkdir -p /home/claude/nvme-kv/glm52-v20-acceptance
find /home/claude/nvme-kv/glm52-v20-acceptance -mindepth 1 -print -quit
findmnt -T /home/claude/nvme-kv/glm52-v20-acceptance
df -h /home/claude/nvme-kv/glm52-v20-acceptance
df -h /dev/shm
free -h
```

The `find` command must print nothing. `findmnt` must resolve to CN3's ext4
filesystem on `/dev/nvme0n1p2`. At least 8 GiB plus normal filesystem headroom
must be free. Only one live engine may own this generated namespace.

Before boot, also record:

- exact Compose contents and SHA-256;
- acceptance-script SHA-256 values;
- current production image/container fingerprint and rollback command;
- `/cache` and `/root/.cache` mappings; and
- host GPU state and any unexpected live CUDA processes.

Static-check the main harness:

```bash
uv run --no-project --python 3.12 python -m py_compile \
  ./nvme_kv_eviction_acceptance.py
```

The NVMe test state directory must be on the host and must not already exist.
Use a unique explicit path such as:

```text
/home/claude/glm52-test-artifacts/v20-combined-20260722-01
```

## 6. Gate 1 — boot, bytes, and capacity

Start v20 and follow timestamped logs. Expect the first build to take about 40
minutes. The compile-progress environment setting should show forward motion.

Before sending any inference request, capture:

```bash
docker inspect glm52-prod
docker exec glm52-prod md5sum \
  /opt/venv/lib/python3.12/site-packages/vllm/v1/kv_offload/tiering/fs/manager.py
docker exec glm52-prod sha256sum \
  /opt/venv/lib/python3.12/site-packages/vllm/v1/kv_offload/tiering/fs/manager.py
docker exec glm52-prod sha256sum \
  /opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/pcie_dma.py \
  /opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/pcie_dma.cu
docker exec glm52-prod sha256sum \
  /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/custom_all_reduce.py
docker exec glm52-prod env
```

Require all byte pins from section 2. A mismatch is a **hold**, not permission
to bless the observed byte dynamically. Compare it to the candidate checkout
and explain the difference first.

Required startup evidence:

- image ID resolves to the pinned digest;
- all four workers initialize;
- `Configured b12x PCIe crossovers` reports DMA min `6291456` bytes;
- no `PCIe DMA allreduce initialization failed` or large-allreduce fallback;
- both wire environment values are exactly `i8_ring`;
- `Filesystem KV cache capacity enabled: 0/8589934592 bytes` appears for the
  fresh generated namespace;
- DRAM offload mmap and secondary FS tier are created;
- `/health` returns 200;
- GPU KV pool is recorded and is at least 480,000 tokens;
- expected target is 644,864 tokens for the reviewed configuration; any change
  requires explanation even if the 480k hard floor still fits; and
- no restart, traceback, OOM, Xid, assertion, watchdog, or I/O failure.

Save the container ID, image ID, `StartedAt`, and `RestartCount` as the
first-boot fingerprint.

### Why this proves the i8 route is eligible

The patched SparkInfer bytes map `i8_ring` to wire mode `int8-ring`. The vLLM
integration constructs that object, applies a fixed DMA threshold of 6 MiB,
and routes a tensor to it when `should_allreduce()` passes. With MNBT 3072,
GLM-5.2 hidden size 6144, and BF16, each full prefill chunk is:

```text
3072 * 6144 * 2 = 37,748,736 bytes
```

That is far above 6 MiB and divisible by TP4's codec constraints. Therefore
full contiguous prefill chunks deterministically take the DMA object once the
startup evidence confirms it initialized. The 50k throughput result later is
the behavioral corroboration. The current integration has no per-dispatch
counter, so do not claim a counter that does not exist.

## 7. Gate 2 — run bounded NVMe fill first

This must be the first inference workload. Do not run a needle, throughput, or
decode request before it: the harness deliberately requires a fresh filesystem
namespace. Health checks and byte inspection are safe.

Run from the CN3 host:

```bash
uv run --no-project --python 3.12 \
  ./nvme_kv_eviction_acceptance.py fill \
  --cache-root /home/claude/nvme-kv/glm52-v20-acceptance \
  --state-dir /home/claude/glm52-test-artifacts/v20-combined-20260722-01
```

Adjust only explicit environment paths, API URL, model name, or container name
when they genuinely differ. If authentication is enabled, export
`VLLM_API_KEY`; do not put the key in saved commands.

The harness verifies health and runtime bytes, calibrates approximately 50k
fresh prompts, inserts needle `738216` at 40%, forces nominal writes greater
than 2x the 8 GiB cap, samples capacity every second, requires older sentinel
paths to disappear, and preserves a newest replay anchor.

Required result:

- `fill-report.json` says `PASS`;
- all responses are HTTP 200, `finish_reason=stop`, and contain `738216`;
- nominal completed writes exceed 16 GiB per discovered namespace;
- completed `*.bin` files never exceed 8,589,934,592 bytes;
- at least one old sentinel path disappears;
- replay-anchor paths remain present;
- container ID, `StartedAt`, and `RestartCount` are unchanged; and
- no fatal log pattern appears.

This gate also supplies repeated, fresh 50k needle passes; do not spend more
window time on a separate 50k needle cell.

## 8. Gate 3 — first-boot v20 validation

Run these against the same process after Gate 2.

### 8.1 Minimal throughput

Run one fresh-prefix 8k cell and one fresh-prefix 50k cell with the pinned
`prefill_bench.py`:

```bash
uv run --no-project --python 3.12 ./prefill_bench.py \
  --tokens 8000 --label v20-i8-ring-8k
uv run --no-project --python 3.12 ./prefill_bench.py \
  --tokens 50000 --label v20-i8-ring-50k
```

Save `/metrics` before and after. Require a prefix-cache miss for each random
first block. Historical v19 `i8_ring` references are 1,607 tok/s at 8k and
1,641 tok/s at 50k. Treat more than 10% regression as an alert and more than
15% as a production hold unless a measured v20-specific cause explains it.
Thus the provisional hard floors are about 1,365 tok/s at 8k and 1,395 tok/s
at 50k.

Using the existing pinned decode harness, run only these mandatory cells:

- ctx 0: concurrency 1, 8, and 16;
- ctx 50k: concurrency 8.

Capture aggregate output tok/s, per-user tok/s, errors, and finish reasons.
Historical v19 `i8_ring` aggregate decode at ctx 0 was about 63.2, 127.2, and
165.1 tok/s at concurrency 1, 8, and 16. Use the same 10% alert / 15% hold
rule. Do not substitute client wall throughput for the established harness
metric.

### 8.2 Deep needle boundary

Use the established unique-prefix needle harness with deterministic decoding,
needle `738216` at 40% depth, and sufficient completion budget. Run:

1. 300k — directly retests the reproducible E4M3 failure boundary;
2. 350k — validates the originally reported hunt target; and
3. 475k — proves the 480k endpoint margin.

For every cell save the request, raw JSON response, prompt token count,
content, `finish_reason`, completion tokens, timing, prefix-cache metric deltas,
and process fingerprint. Required result is HTTP 200,
`finish_reason=stop`, content containing `738216`, and no cache hit on the
unique first block. A short `stop` response without the needle is a real miss;
a `length` response is also a fail, not an ambiguous pass.

The 50k point came from Gate 2. A separate 200k point is optional because it
does not distinguish the known E4M3 failure regime when time is tight.

### 8.3 Concurrency and offload pressure

Run the already-proven PR #133 workload unchanged:

- 16 overlapping requests;
- approximately 49,800 prompt tokens each;
- unique random first-block prefixes;
- fresh total demand approximately 797k tokens;
- normal completions, not tiny health probes.

This must exceed the observed GPU KV pool and force real eviction while also
exercising v20 shared-expert overlap. Capture a one-second process watcher,
prefix-cache query/hit deltas, all raw responses, and logs.

Require:

- 16/16 HTTP 200 and `finish_reason=stop`;
- approximately 797k prefix-cache queries and zero hits;
- demand demonstrably above the recorded GPU pool;
- container ID and `StartedAt` unchanged;
- `RestartCount` unchanged;
- post-load liveness returns exactly `4`; and
- no `_build_store_jobs` assertion, EngineDead, cuBLAS error, OOM, Xid, 5xx,
  worker death, or watchdog action.

Run this stress cell last on the first boot so it cannot contaminate throughput
measurements. After it passes, save the final first-boot fingerprint, full logs,
metrics, FS namespace inventory/bytes, and `nvidia-smi` state.

## 9. Gate 4 — the one controlled restart and persisted promotion

Restart only the vLLM service. Preserve all of these:

- `/home/claude/nvme-kv/glm52-v20-acceptance`;
- the harness state directory;
- the v20 JIT/AOT/extension cache; and
- the exact Compose configuration.

Wait for health without sending inference. Require the same image and source
bytes, a new `StartedAt`, and exactly the expected controlled process
replacement. The startup capacity marker must report a **nonzero** value no
greater than 8,589,934,592 in the same generated namespace.

Then run:

```bash
uv run --no-project --python 3.12 \
  ./nvme_kv_eviction_acceptance.py replay \
  --cache-root /home/claude/nvme-kv/glm52-v20-acceptance \
  --state-dir /home/claude/glm52-test-artifacts/v20-combined-20260722-01
```

Required result:

- `replay-report.json` says `PASS`;
- saved anchor paths existed before replay;
- the request returns HTTP 200, `finish_reason=stop`, and `738216`;
- `vllm:external_prefix_cache_queries` increases;
- `vllm:external_prefix_cache_hits` increases by at least one;
- filesystem bytes remain under the cap;
- no unexpected restart or fatal log; and
- final liveness returns exactly `4`.

No full performance or deep-needle rerun is needed after restart. The replay is
itself a 50k quality/liveness request and the code/image did not change.

## 10. Optional expansion if the core finishes early

Only after every core gate passes:

1. repeat 8k and 50k prefill three times and report median/range;
2. fill the decode matrix at ctx 0 and ctx 50k for concurrency 1/2/4/8/16,
   except any cell whose active-token demand exceeds a deliberately chosen
   safe test boundary;
3. add the 200k needle point and repeat 300k/350k/475k for reproducibility;
4. run arithmetic and long-coherence checks from `quality_gate.py`; and
5. collect detailed NVMe latency/IOPS and promotion timing.

Do not begin KLD in this window.

## 11. Final verdict and evidence bundle

The combined verdict is `PASS` only if all core gates pass. Report:

```text
verdict:
image digest / container ID:
Compose SHA-256:
INT8 Python/CUDA SHA-256:
vLLM custom_all_reduce SHA-256:
filesystem manager MD5/SHA-256:
config summary:
boot GPU KV pool:
first boot StartedAt / RestartCount:
NVMe fill report / high-water / turnover / evicted paths:
prefill 8k / 50k:
decode mandatory cells:
needle 50k(fill) / 300k / 350k / 475k:
16x50k results / query-hit deltas / active demand:
post-restart capacity marker:
replay query-hit deltas:
final liveness:
fatal-log grep:
artifact directory:
```

Preserve at minimum:

- exact Compose and hashes;
- both container fingerprints and image inspection;
- complete timestamped logs from both boots;
- all NVMe harness files, reports, exchanges, capacity CSVs, and metrics;
- raw needle and concurrency requests/responses;
- throughput/decode outputs and before/after metrics;
- NVMe inventories and `findmnt` output; and
- one-second process watcher output during concurrency.

If a core gate fails, verdict is `HOLD`, not partial pass. Preserve the failed
state before rollback. The isolated v19 NVMe spec remains the diagnostic
fallback, but do not consume the window on it unless the combined run stops
early enough to leave a realistic second test window.

## 12. Known unrelated historical issue

Raja's earlier boot cancellation occurred on another host while allocating a
DRAM offload region and compiling. That host/run is not the CN3 acceptance
center point. The useful lesson retained here is only operational: do not kill
an active cold compile, and distinguish the secondary `RuntimeError:
cancelled` shutdown traceback from the first worker-side error. CN3's known-good
64 GB DRAM configuration and `/dev/shm` setup are the authority for this run.
