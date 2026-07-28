# NVMe (fs) secondary-tier offload failure — ROOT-CAUSED

Status: **root cause confirmed by Sol (high confidence).** Original symptom below, then the actual
cause (which supersedes the earlier O_DIRECT-alignment hypothesis), the fix, a residual concern,
and the confirmation proof.

## Original symptom (facts)

- Enabling a filesystem/NVMe **secondary tier** on `OffloadingConnector` / `TieringOffloadingSpec`
  failed at **startup** with `OSError: [Errno 14] Bad address` (EFAULT), in/around the tier worker
  setup, before serving. We dropped to `secondary_tiers: []` (GPU + 64 GB DRAM) and moved on;
  the full traceback was not captured at the time.
- Environment: v19 image `gilded-gnosis-v19-…`, GLM-5.2 hybrid TP4/DCP4, 480k, offload on,
  `cpu_bytes_to_use=64000000000`.

## Root cause (confirmed — Sol)

**`/dev/shm` exhaustion during `MADV_POPULATE_WRITE` on the `SharedOffloadRegion` mmap — not an
NVMe / O_DIRECT problem.**

- Linux returns `EFAULT` from `madvise(MADV_POPULATE_WRITE)` when prefaulting *would* raise SIGBUS,
  which happens when a **sparse tmpfs-backed file lacks physical backing** (i.e. `/dev/shm` doesn't
  have the free capacity to back the mapping). man7 madvise(2).
- Local repro: a 32 MiB region succeeds on a 64 MiB `/dev/shm`; a 96 MiB region returns exactly
  `bad address`. Reproducer: `workspace/nvme-investigation/madvise_repro.go`.
- Upstream issue **vllm-project/vllm#46949** documents the identical traceback + errno at the same
  `SharedOffloadRegion.madvise()` call. Upstream PR **#47073** already fixes it (checks `/dev/shm`
  capacity, replaces the opaque error with an actionable one).
- The O_DIRECT path (`vllm/v1/kv_offload/tiering/fs/io.py`) only runs *after* store/load jobs begin,
  not at tier construction; v19 page-aligns the mmap base, block stride, transfer size, and offsets,
  and a misaligned O_DIRECT op would report `EINVAL`, not this `EFAULT`. So the earlier
  O_DIRECT-alignment hypothesis is **ruled out.**

**Why it presented as NVMe-specific:** enabling the tiering config triggered the 64 GB shared mmap
allocation. `cpu_bytes_to_use=64e9` needs ≈ **59.6 GiB** of available `/dev/shm`. With `shm_size: 32g`
and no effective `ipc: host` (or a host `/dev/shm` crowded by stale `vllm_offload_*.mmap` files),
the prefault had nowhere to land → EFAULT. It looked like the fs tier, but it was the primary
region's backing.

## Same mechanism as our in-session `/dev/shm` crashes

The crowded-`/dev/shm` offload-init crash we debugged this session (stale 64 GB
`vllm_offload_*.mmap` files → new region couldn't allocate → WorkerProc exception) is the **same
root**: the `SharedOffloadRegion` failing to back its 64 GB in `/dev/shm`. One cause, two symptoms.

## Fix

- **Functional fix = sufficient `/dev/shm`.** Our current launch already has `ipc: host` (uses the
  host's 100 GB `/dev/shm`) **and** the entrypoint stale-mmap cleanup (`rm -f
  /dev/shm/vllm_offload_*.mmap`). That very likely resolves the original failure operationally.
- **Optional:** backport upstream **#47073** to v19 for a capacity pre-check + actionable error +
  cleanup. Do **not** author a competing patch — #47073 is the exact duplicate.

## Residual concern (separate issue)

The fs tier has **no native disk eviction** — it assumes storage never fills. NVMe capacity
management must be supplied externally before the fs tier is used in production.

## Confirmation proof (Sol's recommended next steps — run in a disposable boot)

1. Record host and in-container `df -B1 /dev/shm`.
2. Confirm `ipc: host` is active and there are no live/stale offload mappings.
3. Re-enable one filesystem tier.
4. Verify boot, do a unique-prefix store, confirm NVMe block files appear, restart while preserving
   the fs root, then repeat the prompt to prove promotion/load.
