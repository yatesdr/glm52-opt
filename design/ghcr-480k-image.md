# GHCR 480k image design and acceptance contract

Date: 2026-07-17. Status: source complete; image build and GPU boot acceptance
pending.

## Scope

The public image has one profile only: the shipped 480k configuration. It is
strictly TP4/DCP4 on four approximately 96 GB SM120 RTX PRO 6000 Blackwell
GPUs, `nvfp4_ds_mla`, 16 local heads, MNBT 3072, BLOCKS 2340, MTP-3, and a
56,000,000,000-byte DRAM warm offload tier. There is no 64k image profile and
no NVMe tier.

The base is pinned to the linux/amd64 v1.3 manifest digest
`sha256:99ae7b28bb7069b9f7a96f75ea815be56266d2cccf7808d4c497340bb8658bd5`.
The Dockerfile installs exactly the 12 files observed in the production
container: seven v1.4 overlays plus five stage-3/phase-2 overlays. A build-time
`md5sum -c` rejects any byte drift.

## Packaging decisions

1. Exactly the seven live v1.4 files are vendored under
   `docker/overlays/v14/`. They are Apache-2.0 derivative work from
   `davidsyoung/vllm-glm52`/vLLM, identified in the adjacent notice. Vendoring
   is selected so the requested GitHub Actions workflow is independently
   reproducible; the seven unused source-bundle files are not included.
2. The production environment and argv are baked as overridable image
   defaults. `PYTORCH_CUDA_ALLOC_CONF` is intentionally not set: expandable
   segments are outside the validated phase-2 memory posture.
3. `VLLM_DISABLE_COMPILE_CACHE=1` stays enabled. It costs approximately 5–8
   minutes of compilation during each boot, but changing it is a separate,
   unvalidated workstream. The named `/cache` volume remains so JIT artifacts
   that can persist do persist.
4. The entrypoint preserves the production cleanup exactly:
   `rm -f /dev/shm/vllm_offload_*.mmap` before `exec vllm serve`. Stale mmap
   regions can otherwise fill shared memory and hang the next boot.
5. The public compose uses an isolated 64 GB `/dev/shm` and publishes port
   5001. Production currently uses host IPC and therefore the host's 100 GB
   `/dev/shm`; its nominal 32 GB compose setting is not binding. Dropping host
   IPC is safer for external machines but changes the tested IPC posture, so
   the 64 GB compose remains gate-pending until the server acceptance below.

## Release gates

Fable performs the image build and GPU checks. Publication as `latest` requires:

1. The digest-pinned base resolves as linux/amd64 and the embedded 12-file MD5
   gate passes.
2. In-image imports of all 12 destinations succeed; source/constexpr checks for
   phase 2 remain green.
3. The root compose has only the 480k service and requires only `MODEL_DIR`.
4. On the target four-GPU host, the isolated-64-GB-shm compose reaches READY,
   creates one approximately 56 GB offload mmap, and the cleanup removes a
   seeded stale `vllm_offload_*.mmap` before boot.
5. All four ranks arm CKV phase 2 and the 192 MiB escrow. First prefill releases
   it cleanly with probes disabled; packed-CKV routes, while query gather and
   output reduce-scatter remain absent.
6. A cold request plus the standard/deep quality gates match the shipped
   acceptance signature. Any shared-memory failure, geometry mismatch, missing
   escrow log, or sub-150-MiB first-prefill headroom is fail-closed.
