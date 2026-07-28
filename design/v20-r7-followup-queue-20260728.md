# GLM-5.2 v20 r7 follow-up queue

Queued: 2026-07-28  
State: deferred until the active CN4 EXL3-TR3 qualification completes

## Immutable release identity

- Image:
  `voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r7`
- Digest:
  `sha256:fdc107c917f5ce7c7f78a51a2b76b171a0eb25569be58c1284809e7e6ba33482`
- vLLM tree: `936ed48`
- SparkInfer tree: `f532ec9`
- FlashInfer tree: `801d57a`
- CUDA: 13.2
- LMCache source: merged `local-inference-lab/LMCache@9cebd405`
- Release record:
  `local-inference-lab/rtx6kpro#33`, comment `5107420067`
- Runtime guide:
  `models/glm5.2_v20.md` in `local-inference-lab/rtx6kpro`

## Claimed release delta

- SparkInfer #86 and vLLM #189 are integrated in the release trees.
- DCP-aware LMCache 0.5.2 is built from merged source, without an LMCache
  patch overlay.
- The helper uses HTTP readiness, supervises LMCache for the full server
  lifetime, and leaves ordinary serving unchanged when `LMCACHE_MODE=off`.
- Physical-cache geometry supports replicated and partially replicated DCP
  layouts.

## Relationship to the EXL3-TR3 candidate

r7 natively contains both the dynamic-NVFP4 cache work from SparkInfer #86 /
vLLM #189 and the EXL3/Trellis runtime. No duplicate dynamic-cache or EXL3
overlay is expected to be required for TR3.

The release build recipe proves that the generic image is EXL3-capable:

- it builds pinned `brandonmmusic-max/exllamav3` source and validates the
  `exl3_gemm`, fused-MoE, retile, and concurrency extension symbols;
- the immutable vLLM composition contains PR #190, the rank-sliced EXL3
  Trellis backend;
- the immutable SparkInfer composition contains PR #49, the matching
  `trellis_moe` integration;
- its release checks dry-run `MODEL_FAMILY=glm52-exl3` and require
  `--quantization exl3`.

The current derived TR3 image remains useful as an independently qualified
pre-r7 control. The follow-up should first exercise r7's native
`MODEL_FAMILY=glm52-exl3` path with the same pinned checkpoint and compose
posture. Build an overlay only if that direct path fails an identity, import,
or behavioral gate. This avoids carrying superseded copies of overlapping
files such as `vllm/envs.py`.

## Follow-up acceptance

Do not begin until the TR3 record is complete and CN4 is available.

1. Pull and verify the exact manifest digest and source/provenance labels,
   including vLLM PR #190, SparkInfer PR #49, and the native
   `MODEL_FAMILY=glm52-exl3` dry-run contract.
2. Confirm #86/#189 behavior with the dynamic 368-byte record compile/ABI
   gates already used for TR3.
3. Boot the documented TP4/DCP4 CN4 production posture with
   `LMCACHE_MODE=off`; verify it remains behaviorally identical to ordinary
   serving.
4. RAM mode: require `LMCache ready`, healthy helper, cold request with zero
   cached tokens, identical repeat with non-zero cached tokens and identical
   generated-output hash.
5. Disk mode: use a fresh persistent `/cache` namespace; require the same
   cold/hit and output-hash contract, plus bounded host-storage accounting and
   restart persistence.
6. Run only the minimum quality/performance cells needed to establish that r7
   did not regress the already-proven NF3 production record; do not repeat the
   full NF3 qualification without a discrepant result.

### Additional lifecycle regression check

The TR3 qualification exposed a common-vLLM offload lifecycle failure relevant
to r7 testing. After a normal model-container replacement, a root-owned
`/dev/shm/vllm_offload_<engine-id>.mmap` remained allocated. The next process
created a second 64 GB mapping, filled the 63 GB host tmpfs, and all ranks
failed `mmap.madvise()` with `OSError: [Errno 14] Bad address`. Neither file
had an owner after engine shutdown.

For every r7 offload-mode restart:

1. record `/dev/shm` usage and exact `vllm_offload_*.mmap` ownership before
   and after normal container shutdown;
2. require the prior engine's mapping to be unlinked automatically;
3. fail the release gate if a manual root cleanup is required.

This check is distinct from LMCache's own RAM/disk lifecycle and must not be
masked by a pre-test cleanup.
