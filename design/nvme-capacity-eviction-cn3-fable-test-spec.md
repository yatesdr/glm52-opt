# CN3 maintenance-window test: bounded NVMe KV-cache eviction

Status: **implementation complete; CN3 runtime acceptance pending**  
Owner/operator: Fable  
Target: GLM-5.2 v19, TP4/DCP4, 480k, `OffloadingConnector` with DRAM +
bounded filesystem/NVMe secondary tier  
Date prepared: 2026-07-22

> **Tonight's execution plan:** this remains the isolated v19 fallback and
> reference procedure. The operator-approved combined v20 procedure is
> `v20-combined-cn3-acceptance-spec.md`; use that document for the maintenance
> window unless the v20 image fails its boot/integrity gate.

## 1. Objective

Prove on CN3 that the filesystem secondary tier:

1. receives real KV blocks from the DRAM primary tier;
2. remains at or below its configured byte limit under sustained unique-prefix
   stores;
3. evicts the least-recently-used unpinned blocks when full;
4. survives a clean engine restart with the NVMe namespace preserved; and
5. promotes retained blocks from NVMe, demonstrated by a nonzero
   `vllm:external_prefix_cache_hits` delta.

This is the missing runtime gate for the local bounded-NVMe patch. The test is
independent of the INT8 wire codec and the scheduler assertion fix, although the
acceptance image includes those production patches so the complete CN3 stack is
exercised.

## 2. Artifacts and byte pins

### vLLM patch

- Exact v19 base: `7ea567a2458a4800a6a0e3e0a6ba41fcbd00d146`
- Branch: `feat/fs-tier-capacity-v19`
- Commit: `d74c0a6c397a76dd1e0aede8b1c8927c9a7c74ac`
- Commit URL:
  <https://github.com/yatesdr/vllm-opt/commit/d74c0a6c397a76dd1e0aede8b1c8927c9a7c74ac>
- Runtime file:
  `vllm/v1/kv_offload/tiering/fs/manager.py`
- Unpatched input MD5: `5e341cdfef3456ae72f00063756d4dc9`
- Required patched MD5: `a72eeb81c735036b281ff97f5d759122`

### Acceptance image

```text
ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v19-int8-block-nvme-test@sha256:dcf9779a3a3fbd78e2f84a8c2ad42cdaec7c2558d9f78f99a7b52a8eaef1d974
```

This image should contain:

- bounded filesystem-tier commit `d74c0a6c`;
- the already-validated OffloadingConnector scheduler fix corresponding to
  PR #133; and
- the already-validated block-INT8 wire implementation.

The scheduler fix is important because pressure testing can evict GPU KV
blocks. It is separate from the filesystem capacity patch and must not be
reported as part of this patch's behavior.

### End-to-end acceptance script

- Workspace path:
  `/Users/derek/glm52-opt/harness/nvme_kv_eviction_acceptance.py`
- SHA-256:
  `7df4b95926816fac96139531c1bc25be4b1e9a4a569360073ef2aea80b9b595f`
- Standard-library only; run with Python 3.12 through `uv` or a project virtual
  environment.

Verify the copy placed on CN3 before running:

```bash
sha256sum nvme_kv_eviction_acceptance.py
```

## 3. What the patch does

The filesystem tier gains one opt-in setting:

```json
"max_cache_size_bytes": 8589934592
```

When configured, the manager:

- indexes completed `*.bin` files in the active namespace at startup;
- uses modification time as the restart-safe initial LRU order;
- trims an oversized existing namespace before serving;
- reserves bytes before concurrent asynchronous stores, so completed files plus
  pending reservations cannot oversubscribe the limit;
- updates LRU order on hits, loads, and touches;
- pins lookup hits and queued load sources so eviction cannot delete a block
  being promoted;
- evicts the oldest unpinned block before admitting a replacement; and
- fails a store with `ENOSPC` only if every resident block is pinned.

The patch does **not** change the default behavior. If
`max_cache_size_bytes` is omitted, the filesystem tier remains unbounded.
The O_DIRECT read/write implementation is unchanged.

The limit covers completed block files in the current generated
`<model>_<digest>_r<rank>` namespace. It excludes `config.json`, temporary
files, other configuration digests, and unrelated disk usage. GLM-5.2's
parallelism-agnostic mapper is expected to use `_r0`; the acceptance script
discovers and independently checks every `_rN` namespace if more appear.

Bounded mode requires exclusive ownership: only one live vLLM instance may use
the generated namespace during the test.

## 4. CN3 test configuration

Use a fresh, test-specific directory on the real CN3 NVMe filesystem. Do not
reuse or delete a production cache namespace.

Filesystem tier:

```json
{
  "type": "fs",
  "root_dir": "/nvme-kv/glm52-eviction-acceptance",
  "n_read_threads": 16,
  "n_write_threads": 16,
  "max_cache_size_bytes": 8589934592,
  "locality": "LOCAL"
}
```

Connector shape:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 64000000000,
    "secondary_tiers": [
      {
        "type": "fs",
        "root_dir": "/nvme-kv/glm52-eviction-acceptance",
        "n_read_threads": 16,
        "n_write_threads": 16,
        "max_cache_size_bytes": 8589934592,
        "locality": "LOCAL"
      }
    ]
  }
}
```

Keep CN3's known-good 64 GB DRAM primary tier. `TieringOffloadingManager`
cascades every successful new primary store to every secondary tier, so the
DRAM tier does not need to fill or evict before NVMe receives data. Reducing
the DRAM allocation is therefore unnecessary for this proof. The 8 GiB NVMe
limit is deliberately small and is not a production recommendation.

Keep the known-good CN3 settings:

- `ipc: host`;
- entrypoint cleanup of `/dev/shm/vllm_offload_*.mmap`;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`;
- `PYTHONHASHSEED=0` so hashes and filenames survive restart;
- `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` for this v19 isolation test;
- GMU `0.970`, max model length `480000`, MNS `16`, TP4/DCP4; and
- the current validated `i8_ring` wire configuration.

Map a host NVMe directory to `/nvme-kv`. Record the exact host/container
mapping. If the Compose mapping is:

```yaml
- /mnt/cn3-nvme/vllm:/nvme-kv:rw
```

then the acceptance script's host path is:

```text
/mnt/cn3-nvme/vllm/glm52-eviction-acceptance
```

Verify the underlying device rather than trusting the directory name:

```bash
findmnt -T /mnt/cn3-nvme/vllm/glm52-eviction-acceptance
```

## 5. Time budget and wait discipline

Reserve a **30-minute minimum** maintenance window and preferably 45 minutes
including setup and rollback.

Expected durations on CN3:

- cold v19 boot: approximately 7-10 minutes;
- fill/eviction phase: approximately 10-12 minutes;
- controlled restart: approximately 7-10 minutes; and
- persisted replay: approximately 1 minute.

Expected fill + restart + replay: **18-25 minutes after the first engine is
healthy**. Including the initial cold boot, expect approximately 25-35 minutes
from deployment start; retain the full 45-minute window for slow compilation,
setup, evidence capture, and rollback.

Do not treat quiet compilation as a hang. Do not stop a boot merely because a
few minutes pass without a new log line. During a cold boot, model loading can
take about five minutes and compilation/warmup another two or more.

Abort a boot only for a concrete terminal signal such as:

- worker or EngineCore exits;
- an explicit traceback, OOM, Xid, or fatal assertion;
- container ID/`StartedAt` changes unexpectedly; or
- at least 15 minutes without forward progress **and** inspection shows no
  active load, compile, link, or GPU work.

If work is still visibly active, allow up to 25 minutes before declaring a
startup hang and collect diagnostics before stopping it.

## 6. Gate 0 — preflight and code proof

Before consuming the GPU window:

1. Confirm the image digest and runtime manager MD5.
2. Confirm the script SHA-256.
3. Confirm the test namespace is fresh and exclusively owned.
4. Record `findmnt`, free NVMe bytes, host `/dev/shm`, host memory, container
   ID, image ID, `StartedAt`, and `RestartCount`.
5. Preserve the exact Compose file used for the run.

Runtime MD5 check:

```bash
docker exec glm52-prod md5sum \
  /opt/venv/lib/python3.12/site-packages/vllm/v1/kv_offload/tiering/fs/manager.py
```

Required result:

```text
a72eeb81c735036b281ff97f5d759122
```

Static-check the acceptance script:

```bash
uv run --no-project --python 3.12 python -m py_compile \
  ./nvme_kv_eviction_acceptance.py
```

If the exact patched checkout and its test dependencies are already available,
run the focused unit tests before boot:

```bash
.venv/bin/python -m pytest -q \
  tests/v1/kv_offload/tiering/test_fs_tier.py \
  -k 'fs_capacity or bounded_fs or lookup_hit_is_pinned or lookup_pin_transfers or queued_load_source or concurrent_stores_reserve_capacity or capacity_index or capacity_eviction'
```

Then run the entire file:

```bash
.venv/bin/python -m pytest -q \
  tests/v1/kv_offload/tiering/test_fs_tier.py
```

Do not spend the maintenance window building a new development environment if
the exact test checkout is not already ready. Record that the in-image unit
gate was unavailable and continue with the end-to-end gate; CI/unit results
remain required before upstream submission.

## 7. Gate 1 — boot and initialization

Start the acceptance deployment and follow timestamped logs. Required startup
evidence:

```text
Created mmap file ... (16.00 GB)
Allocating 101 CPU tensors...
Filesystem KV cache capacity enabled: <used>/8589934592 bytes in <namespace>
Created secondary tier #0 (fs)
Created TieringOffloadingManager ... 1 secondary tier(s)
```

For a fresh namespace, `<used>` should normally be zero. Require:

- engine healthy;
- no worker death or container restart;
- the generated namespace is beneath the intended NVMe test root;
- boot KV-pool size recorded; and
- no assertion, OOM, Xid, watchdog, or filesystem I/O error.

Do not run pressure until `/health` succeeds.

## 8. Gate 2 — bounded fill and real eviction

Run the script on the **CN3 host**, not inside the container. It needs Docker,
`findmnt`, the HTTP endpoint, and direct access to the host NVMe path.

The state directory must not already exist. Use a host directory that survives
the container restart:

```bash
uv run --no-project --python 3.12 \
  ./nvme_kv_eviction_acceptance.py fill \
  --cache-root /mnt/cn3-nvme/vllm/glm52-eviction-acceptance \
  --state-dir /var/tmp/glm52-nvme-acceptance-20260722
```

Adjust only the host NVMe path and state-directory suffix as needed. If CN3's
container name, API URL, or served model name differs from the defaults, use
`--container`, `--base-url`, or `--model`. If authentication is enabled, export
`VLLM_API_KEY`; do not place the key in saved commands or reports.

The script will:

1. verify health, image process identity, manager MD5, mount, free space,
   startup capacity marker, and metrics;
2. calibrate a chat prompt to approximately 50k tokens;
3. put a unique nonce in the first block of every prompt, preventing local
   prefix-cache reuse;
4. place needle `738216` at 40% depth and require a clean answer;
5. submit sequential unique prompts until conservative nominal completed FS
   writes exceed 2x the 8 GiB limit;
6. sample completed `*.bin` bytes every second;
7. require earlier sentinel paths to disappear, directly proving eviction;
8. require the same container ID, `StartedAt`, and `RestartCount`; and
9. submit and preserve one final newest replay anchor.

Expected fill runtime is 10-12 minutes. The script prints progress after each
request. Do not interrupt it while progress continues. Its default maximum is
32 fill requests; a slower/smaller payload can extend the phase toward 20-25
minutes.

Required fill verdict:

- every request HTTP 200;
- every response `finish_reason=stop` and contains `738216`;
- prompt size approximately 50k tokens;
- nominal unique writes greater than `2 * 8589934592`;
- at least one earlier sentinel block path disappears;
- completed `.bin` bytes never exceed `8589934592` in any discovered
  namespace;
- no 5xx, assertion, `EngineDead`, OOM, Xid, watchdog, I/O failure, or restart;
  and
- `fill-report.json` says `PASS`.

## 9. Gate 3 — persisted NVMe promotion

After Gate 2 passes, restart only the vLLM container while preserving the test
NVMe directory and script state directory:

```bash
docker restart glm52-prod
```

The entrypoint may remove the stale `/dev/shm/vllm_offload_*.mmap`; it must not
remove the NVMe namespace. Wait through the full cold boot again and require:

```text
Filesystem KV cache capacity enabled: <used>/8589934592 bytes in <same namespace>
```

`<used>` must be nonzero and at or below the limit. Once `/health` succeeds,
run:

```bash
uv run --no-project --python 3.12 \
  ./nvme_kv_eviction_acceptance.py replay \
  --cache-root /mnt/cn3-nvme/vllm/glm52-eviction-acceptance \
  --state-dir /var/tmp/glm52-nvme-acceptance-20260722
```

The replay phase refuses to run if `StartedAt` did not change, which prevents a
GPU/DRAM-resident request from masquerading as an NVMe hit. It resubmits the
exact saved prompt and requires:

- at least one saved replay-anchor path survived restart;
- a positive `vllm:external_prefix_cache_queries` delta;
- a positive `vllm:external_prefix_cache_hits` delta;
- `finish_reason=stop` and needle `738216` in the answer;
- completed files still within the configured limit;
- no process identity/restart change during replay; and
- `replay-report.json` says `PASS`.

A correct model answer with zero external hits is a **failure**. It proves only
recomputation, not NVMe promotion.

## 10. Required artifacts

Preserve the complete script state directory. At minimum return:

```text
fill-report.json
replay-report.json
capacity-samples.csv
capacity-samples-replay.csv
container-before.json
container-after-fill.json
container-before-replay.json
container-after-replay.json
metrics-before.txt
metrics-after-fill.txt
metrics-before-replay.txt
metrics-after-replay.txt
runtime-log-findings-fill.txt
runtime-log-findings-replay.txt
sentinel-paths.json
replay-anchor-paths.json
exchanges/
```

Also save:

- exact Compose file;
- complete timestamped container logs for both boots and both test phases;
- image digest and manager MD5;
- `findmnt`, `df -B1` for NVMe and `/dev/shm`, `free -b`;
- boot KV-pool allocation; and
- final container `RestartCount`, ID, and `StartedAt`.

Report in this compact form:

```text
verdict:                         PASS/FAIL
image digest / manager MD5:      ... / ...
config:                          TP4 DCP4 480k MNS16, DRAM 64GB, FS cap 8GiB
host NVMe path / filesystem:     ... / ...
boot capacity marker:            used/8589934592
fill requests / prompt tokens:   ... / ...
nominal attempted FS bytes:      ...
max observed completed bytes:    ...
sentinel paths removed:          ...
fill RestartCount:               before -> after
restart startup indexed bytes:   ...
replay external queries delta:   ...
replay external hits delta:      ...
replay answer / finish reason:   738216 / stop
fatal log matches:               NONE or exact lines
artifacts:                       path
```

## 11. Failure handling

Do not immediately restart or delete evidence after a failure. First capture:

```bash
docker inspect glm52-prod
docker logs --timestamps glm52-prod
df -B1 /dev/shm
free -b
findmnt -T /mnt/cn3-nvme/vllm/glm52-eviction-acceptance
df -B1 /mnt/cn3-nvme/vllm/glm52-eviction-acceptance
sudo journalctl -k --since "30 minutes ago"
```

Record the last successful script progress line and the current namespace byte
count. Fail the patch for any of the following:

- completed block bytes exceed the configured limit;
- old paths never disappear after more than 2x nominal unique writes;
- retained replay yields zero external hits;
- an active/queued load loses its source;
- a filesystem job deadlocks or never settles;
- assertion, `EngineDead`, OOM, Xid, watchdog, 5xx, or restart;
- startup indexing deletes files outside the active namespace; or
- default unbounded behavior changes when the option is omitted.

An `ENOSPC` store failure is expected only in the focused unit test that pins
every resident entry. It is not expected in this integration run and should be
treated as a failure here.

## 12. Optional extended proof

Run only after the minimal two-phase proof passes:

1. Repeat the full unit file three times.
2. Run three fill/restart/replay cycles and record external query/hit deltas.
3. Replay a retained prompt while seven unique 50k prefills run concurrently;
   require all eight requests to finish and no source eviction race.
4. Exercise startup trimming: populate beyond 8 GiB without a limit, stop
   cleanly, enable the 8 GiB limit on the same namespace, and prove startup
   trimming plus a successful replay.
5. Run the 50k/200k/300k/350k needle ladder.
6. Run the capped production throughput matrix and confirm prefill, decode, and
   GPU KV-pool allocation are unchanged relative to the identical stack without
   the FS capacity option.

## 13. Production follow-up after acceptance

The 8 GiB value is a test limit. For production:

- use a new exclusive namespace;
- size the cap below usable NVMe capacity;
- leave at least 10-20% of the filesystem free for metadata, temporary files,
  logs, and unrelated data;
- remember the cap covers only the active model/configuration namespace; and
- manage old digest namespaces separately.

Do not submit or enable this feature in production based only on a successful
boot. Both bounded eviction and persisted external-hit replay are required.

## Appendix A — status before the CN3 window

- Patch implementation and documentation: complete.
- Local commit and byte pins: complete.
- CN3 end-to-end bounded eviction: not yet run.
- CN3 persisted NVMe promotion: not yet run.
- Raja's attempts were manually aborted during loading/compilation and are
  inconclusive. They are neither positive nor negative evidence for this
  patch and do not determine CN3 configuration.

## Appendix B — unrelated host-memory reference

An earlier non-CN3 64 GB attempt reached shared-region creation and
`Allocating 101 CPU tensors...`, then a worker disappeared while the full
region was being registered/pinned. That host had 128 GB RAM and only about
67.1 GB total `/dev/shm`; the 64 GB mapping left little headroom.

CN3 has not shown that behavior. Do not use the other host's result as a CN3
failure signature or reduce CN3's known-good 64 GB tier because of it. If CN3
nevertheless dies during offload initialization, capture the full worker log
and kernel journal before changing configuration.
