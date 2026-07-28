# v20 MTP verifier CUDA illegal-access fix — CN3 proof

> **SUPERSEDED — DO NOT RUN.** Code review proved the complete size-64
> target and speculative graph set already captured and synchronized during
> MRV2 memory profiling. That falsifies this document's intrinsic MTP-route
> hypothesis. The replacement diagnosis and one-boot acceptance plan are in
> `v20-mrv2-graph-pool-reuse-fix.md`; commit `1b8e7f8f` must not be deployed or
> proposed upstream.

Date: 2026-07-22  
Operator: Fable  
Target: CN3, TP4/DCP4, GLM-5.2, MTP3, `nvfp4_ds_mla`

## Purpose

Prove and fix the v20 boot crash that surfaced at
`spec_decode/autoregressive/speculator.py -> decode_cudagraph_manager.capture()
-> make_dummy()` as `CUDA error: an illegal memory access was encountered`.

The reported `make_dummy()` host-to-device copy is a CUDA synchronization
point. The likely faulting work is the preceding asynchronous target-model
warmup. Commit `3e731bc0` changed the default from the established multi-token
extend route to SparkInfer's flattened MTP decode route. Its tests qualified
DCP1, `fp8_ds_mla`, 16 heads and at most 48 verifier rows; CN3 exercises DCP4,
`nvfp4_ds_mla`, 64 gathered heads and 64 rows.

A second user's unchanged `3e731bc/1a88b38` image successfully boots and
benchmarks TP4/DCP4 NVFP4 MTP3 with MNS 8 and graph cap 32. That proves the
topology and KV format are not generically broken. It isolates the unproven
region to the failed profile's 64-row capture and/or its tighter graph-memory
headroom. The successful profile also uses GMU `0.978` and an A2A limit of 16;
the failed profile uses GMU `0.980` and an A2A limit of 64.

The patch therefore keeps the optimized verifier decode route through the
field-proven 32-row DCP/NVFP4 envelope and sends only larger verifier batches
through the established extend path. Ordinary one-token decode is unchanged.
Explicit mode `1` still overrides the guard for future qualification runs.

## Exact artifacts

- Candidate image:
  `gilded-gnosis-v20-vllm3e731bc-si1a88b38-int8-nvme-mtpfix`
- Fix branch:
  `fix/b12x-mtp-decode-dcp-guard-20260722`
- Fix commit: `1b8e7f8f`
- Base vLLM commit: `3e731bc043d23ec21277fb76d3e15fe6da91b23b`
- Input `b12x_mla_sparse.py` SHA-256:
  `e06b35c88db6691c11cdef0e1a134746060682f596478365661847f681d4e0bb`
- Patched `b12x_mla_sparse.py` SHA-256:
  `ad05c0a441ca615e81f93da94b5dee8c26c417cf36c3db9ed93c44a3a89ec438`

The runtime destination is:

```text
/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py
```

Before overlaying, require the input hash to match. After overlaying, require
the patched hash to match. Stop if either pin differs.

## Minimal proof

Use the exact compose that reached a 544,000-token pool at GMU `0.980`. Change
only the item named by each gate. Keep the image, SparkInfer bytes, PR #69,
PR #165, PR #166, TP/DCP topology, MTP3, capture sizes and model configuration
unchanged.

### Gate A — causal toggle, no new code

Add:

```yaml
environment:
  VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE: "0"
```

Boot once. This recreates the pre-`3e731bc0` routing behavior while leaving all
three of our PRs present.

PASS requires all of the following:

1. KV allocation completes and reports the pool.
2. CUDA-graph capture completes on all four workers.
3. The API becomes healthy.
4. `RestartCount` remains `0`; container ID and `StartedAt` do not change.
5. No `illegal memory access`, `EngineDeadError`, worker death or 5xx appears.

If Gate A fails with the same illegal access, stop: the route hypothesis is
falsified and the guarded patch must not be promoted.

For diagnosis after that stop, use one-variable boots in this order:

1. Restore `auto`, change only `VLLM_DCP_A2A_MAX_TOKENS=16`. A PASS implicates
   the 64-row B12X DCP A2A path.
2. If that still fails, restore the A2A limit and use GMU `0.978` while keeping
   MNS 16 and graph cap 64. A PASS supports graph-capture headroom exhaustion.
3. If that still fails, keep GMU `0.978` and set graph cap 32. A PASS isolates
   the problem to the 64-row graph; a FAIL contradicts the successful user's
   reported envelope and requires a byte/config audit.

### Gate B — patched `auto`

Overlay or bake the patched `b12x_mla_sparse.py`. Remove the Gate A environment
override so the variable is unset/default `auto`.

Require this log line (once is sufficient):

```text
B12X MTP verifier decode auto-route is capped at 32 rows for dcp_world_size=4, kv_cache_dtype=nvfp4_ds_mla; larger batches use the extend path
```

Apply the same five PASS requirements as Gate A. Also record the KV pool; the
guard may slightly reduce decode scratch accounting, but pool growth is not a
correctness requirement.

### Gate C — minimal live MTP check

After Gate B boots:

1. Send one short deterministic request with MTP3 enabled and require HTTP 200,
   `finish_reason=stop`, non-empty correct content.
2. Send eight concurrent short requests and require 8/8 HTTP 200. This exercises
   the retained 32-row verifier decode route.
3. Send sixteen concurrent short requests and require 16/16 HTTP 200. This
   exercises the 64-row extend fallback that replaces the crashing route.
4. Require no worker or container restart across either concurrency cell.
5. Capture generated/drafted/accepted-token counter deltas if available; the
   counters must show speculative decoding is still active. The patch changes
   only target-model attention verification routing, not whether MTP runs.

Minimal verdict is PASS only if Gates A, B and C all pass.

## Optional full proof

After the minimal proof:

1. Repeat boot/capture three times with patched `auto`.
2. Run ctx0 decode at concurrency 1, 8 and 16; compare aggregate and per-user
   throughput with v19. C1/C8 retain the verifier decode route, while C16 uses
   the extend fallback, so report the C16 delta separately.
3. Run ctx50k at concurrency 8.
4. Run needles at 300k, 350k and 475k.
5. Run the 16 x 50k unique-prefix overflow load and confirm NVMe/offload gates,
   zero restarts and post-load liveness.
6. Record `nvidia-smi` memory, vLLM available-KV memory and final KV-token pool.

## Interpretation

- Gate A PASS + prior `auto` FAIL is the causal proof that the new MTP verifier
  decode route, not PR #69/#165/#166, triggers this boot crash.
- Gate B PASS proves the code guard reproduces the safe routing without a
  deployment-specific environment override.
- This patch fixes the CUDA boot blocker. It does not by itself recover the
  separate v20 KV-pool deficit (544,000 versus the 600,000 target).
