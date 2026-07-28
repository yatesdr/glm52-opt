# v20 Boot 8 — W4A16 cooperative-grid proof

Status: **ready for one discriminating CN3 boot**  
SparkInfer base: `1a88b389a8d14f26dbe4c157965938cfd8f1bf51`  
Patch commit: `0c29627b22d53604293ed2e3fbc87a884e0990c4`

## Root cause

Boot 7 localized the first CUDA fault to the decode speculator's eager
`warmup_forward` at exactly `num_tokens=9`, after sizes 16 through 10 passed.
That boundary has a direct source explanation:

- W4A16 fused-micro decode covers `M <= 8` and already launches
  cooperatively after SparkInfer commit `695c011`.
- `M=9` is the first descriptor routed to the packed W4A16 fused fallback.
- `W4A16FusedMoeKernel` crosses software all-CTA barriers between FC1,
  activation, output initialization, and FC2.
- `W4A16FusedMoeHybridKernel` uses the same barrier machinery.
- Both launch sites set a resident-grid cap but omit `cooperative=True`.

The host planner already documents the contract: it caps `grid_x` at
`sms * blocks_per_sm` so the cooperative grid can be resident at once. A size
cap is necessary but not sufficient. Without cooperative launch admission,
unrelated work can occupy one or more SMs after only part of the W4A16 grid is
resident. Those CTAs spin at the software barrier while unscheduled peers can
never become resident. The production pass has shared-expert and other stream
activity; the profiling pass does not reproduce the same admission pressure,
which explains why the identical M=9 descriptor passes profiling and fails
production.

This is the same CUDA scheduling contract SparkInfer fixed for the fused-micro
kernel in `695c011`, but that change covered only `M <= 8`. It did not update
the W4A16 fallback entered at M=9.

## Minimal fix

Set `cooperative=True` on the two barrier-bearing W4A16 fused launch sites:

1. `W4A16FusedMoeKernel.__call__`; and
2. `W4A16FusedMoeHybridKernel.__call__`.

No launch geometry, tile selection, quantization math, graph size, MTP setting,
shared-expert overlap, DCP route, i8 ring transport, KV allocation, DRAM
offload, or NVMe policy changes.

The only behavioral change is CUDA admission: the driver admits the complete
bounded grid atomically instead of allowing partial residency. That is the
required execution contract for a software whole-grid barrier. The existing
`grid_x <= sms * blocks_per_sm` planner cap makes the cooperative launch
admissible.

## Patch and byte pins

Patch artifact:

`patches/v20-w4a16-cooperative-grid/0001-fix-moe-cooperatively-launch-fused-W4A16-grids.patch`

Patch SHA-256:

`1a76f60f8e8c4fd491412fabd66825b70cbca2151827c033088e99e974cc527e`

| File | image/base input SHA-256 | patched output SHA-256 |
|---|---|---|
| `sparkinfer/moe/_shared/kernels/w4a16/kernel.py` | `7b15236dfd73c8eea6d692b661aa22f8e526c16f60f14551b6c43abcc6322e00` | `7bae99dfff0ab8f61f1d2a0f36a401543f32a39e2e3982668fcc89a44e882f05` |
| `tests/moe/test_fused_moe.py` | `24cb5605ee2e5820d8be14a097659033c3c1c45dbc10fe2d6458c69ce8970296` | `8081dfd1b61540f50d8885a61d115c0d61ceda12c11fcb8e1c73b53b90313691` |

The test change is GPU regression coverage and does not need to be copied into
a runtime-only image overlay.

## Gate 0 — build and static verification

1. Start from SparkInfer `1a88b389...`, the exact commit in the v20 image.
2. Verify the runtime input hash above.
3. Apply the patch cleanly.
4. Verify the runtime output hash in the built image.
5. Confirm exactly two `cooperative=True` additions in W4A16, one in each fused
   entry point; do not add it to the standalone `W4A16GemmKernel`.
6. Keep the vLLM, CUDA, model, compose, graph, GMU, and offload inputs from
   Boot 7 unchanged.
7. Do not apply the superseded shared-expert-order patch. Do not apply CKV
   draft PR #169 in this proof boot; it is a separate lifecycle fix.

If the image can execute the repository GPU test directly, run:

```bash
pytest -q tests/moe/test_fused_moe.py::test_run_w4a16_m9_graph_replay_with_prequeued_aux_work
```

The regression uses the production GLM shard geometry (`M=9`, 16 experts,
hidden 6144, intermediate 512, top-k 8), captures the route-packed W4A16 path,
prequeues 16 large BF16 GEMMs on an auxiliary stream, replays the graph, and
requires finite, nonzero, numerically stable output.

## Gate 1 — one boot, fail closed

Use the unchanged Boot 7 production candidate:

- TP4 / DCP4 / MTP3;
- max model length 480,000;
- max sequences 16;
- graph cap 64;
- the same GMU and MRV2 accounting patch;
- i8 ring, A2A policy, DRAM offload, and staged NVMe tier unchanged; and
- `VLLM_CUDAGRAPH_CAPTURE_DIAGNOSTICS=1` retained for descriptor evidence.

Remove `CUDA_LAUNCH_BLOCKING=1` if it was used for the separate localization
boot. It is no longer needed for this proof and changes timing/admission.

PASS requires all of the following:

1. Profiling and production target captures finish.
2. Profiling and production prefill-speculator captures finish.
3. Production decode-speculator descriptors 16 through 1 finish.
4. The former failure tuple
   `tokens=9,reqs=9,uniform=1,warmup_forward` is explicitly `PASS`.
5. The API reaches serving state.
6. A liveness request returns `4` and one ordinary MTP request ends with
   `finish_reason=stop`.
7. `RestartCount=0`; container ID and `StartedAt` remain unchanged.
8. No illegal access, EngineDead, worker exit, assertion, OOM, Xid, or 5xx.

FAIL closed on the first failed diagnostic boundary. Preserve the complete
first-run log and inspect JSON, then stop. Do not tune MNS, graph cap, GMU,
route thresholds, or memory settings in the same boot.

## Gate 2 — continue qualification without another boot

Only after Gate 1 passes, use the same live server for the planned combined
qualification:

1. decode cells at concurrency 1, 4, 8, and 16, recording aggregate and
   per-user throughput plus MTP acceptance;
2. 300k, 350k, and 475k needles;
3. bounded NVMe fill, eviction, restore, and persistence evidence;
4. 16 x 50k unique-prefix overflow stress;
5. cold 8k and 50k prefill measurements with prefix-cache deltas; and
6. final liveness, restart/container identity, and error-signature audit.

Cooperative admission can slightly change scheduling latency at the affected
M>=9 W4A16 fallback, so decode throughput must be measured. It does not remove
shared-expert overlap or reduce max sequences; MNS16 and graph cap 64 remain
the intended production settings.

## Evidence status

Static/source proof is complete:

- exact image commit and runtime input byte verified;
- all whole-grid barrier entry points identified;
- both omitted cooperative launch flags fixed;
- no standalone GEMM launch changed;
- a targeted M=9 graph-plus-aux-stream regression added; and
- `git diff --check` passes.

GPU proof is pending the single CN3 Gate 1 boot above. Do not present this as a
runtime-proven fix until that gate passes.
