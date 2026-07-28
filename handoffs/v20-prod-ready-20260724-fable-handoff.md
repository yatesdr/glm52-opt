# v20 production-candidate single-boot handoff

Date: 2026-07-24/25  
Technical owner: Sol  
Operator/change authority: Fable  
Target: CN4 first; CN3 only after the complete verdict

## Objective

Build one fail-closed image from the published
`vLLM 992b874cf / SparkInfer a93df671` base, boot it once, and run the full
functional, performance, long-context, and offload qualification on that same
process. Do not create a configuration ladder. A hard failure stops the suite
and preserves evidence.

The pre-change deep ladder and no-model Proof 3 are complete. The final build
pin in `fable-sol-comms.md` authorizes Gate 0. Use only the exact hashes in
that pin; do not add another patch or configuration cell.

## Candidate contents

vLLM integration, in independent commits:

1. PR #165 bounded filesystem-tier capacity;
2. native SM120/B12X MTP3 flattening gate;
3. PR #154 absorbed `kv_b_proj` source reclaim;
4. full staged BF16-weight/FP8-output query qualification guard, selected
   because the fused route was not byte-identical to the safe reference;
5. PR #171 compact-NVFP4 verifier qualification, selected after the
   pre-change image passed 250k but genuinely failed at 350k;
6. PR #168 MRV2 profiling/production graph-pool reuse.

SparkInfer integration:

1. the base already carries PR #76's persistent PCIe-DMA default output as
   `cd089a4` (stable patch-id identical to PR head `8670c57`);
2. overlay only the output-capacity, typed-view, explicit-output, and close
   validation follow-up.

The base `a93df671` top-k file remains untouched. Proof 3 passed all 160 exact
selector cases. Its `a93df671` top-k commit has the same stable patch-id as
current SparkInfer `83a5844`, so the upstream exact-overflow fix is already
present rather than missing from this image.

Proof 3 also compared the safe staged query, fused-BF16 plus static
quantization, and direct-FP8 fused query. The fused paths were mutually
byte-identical but differed from the staged reference in 17/60 cases. Under
the predeclared fail-closed threshold, this selects the full staged guard.
Evidence:

```text
/home/derek/sol-proof-results/v20-decode-retrieval-microprobes-v3.jsonl
records: 237
sha256: eb8b4e495ee7dedf06c172274a614481e9fc4b5dd22f2ecf79826b1ed811b11b
```

The base also already contains PR #172's functional startup-profile work as
`cb27f671d`. PR #168 remains included because it fixes the separate lifetime
of the profiled global CUDA-graph pool. Do not apply PR #175: at TP4/DCP4 its
transposed query-split groups have world size 1 and therefore cannot divide
indexer work.

## Gate 0 — build and byte proof

Preconditions:

- the previous model process has stopped normally;
- no GPU compute processes remain;
- no container is auto-restarting;
- on CN4, set and record `-pl 300` and `-lgc 0,2600` before launch; the
  unlocked v20 control reset the host during 50k prefill, while the clock
  ceiling is the established stability posture and does not measurably change
  this box's PCIe/DCP-limited throughput;
- preserve all prior logs and JSON responses;
- use a new NVMe namespace; do not delete prior evidence.

From the repository root:

```bash
docker build \
  -f docker/Dockerfile.v20-prod-ready-20260724 \
  -t ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-prod-ready-20260724 \
  .
```

The build must fail closed on any input or output hash mismatch. Do not bypass
a hash check. Record:

- final Dockerfile SHA-256;
- both integration HEADs;
- image ID and all OCI labels;
- runtime hashes for all eight overlaid files;
- unchanged top-k hash;
- presence of `safe_mla_query_bmm` in the stable extension;
- `python -m py_compile` result.

Run the focused Linux CPU tests from the two integration worktrees before
launch. Required coverage:

```text
BF16 fused-query eligibility/warmup guards
PR #171 mode resolution and scratch capacity
PR #154 release/reload behavior
PR #165 capacity bookkeeping
PR #168 profiling/production graph-pool identity and cleanup
PR #76 persistent output view/explicit-output/close
```

**PASS:** every byte and focused test is green.  
**FAIL:** stop; no model boot.

## Gate 1 — one clean boot

Launch only:

```text
deploy/glm52-v20-prod-ready-20260724.yaml
```

Resolve and archive the Compose before launch. Keep `restart: "no"` and do not
attach autoheal.

Required boot evidence:

- immutable image ID, container ID, `StartedAt`, and `RestartCount=0`;
- recorded CN4 power and clock ceilings remain 300 W and 2600 MHz;
- TP4/DCP4/MTP3, MNS16, max length 480,000, graph sizes
  `1,2,4,8,16,32,64`;
- `nvfp4_ds_mla`, FP8 RoPE, B12X sparse MLA/MoE;
- `i8_ring` normalized and enabled, with no fallback warning;
- CKV gather cap exactly 480,000;
- `auto + nvfp4_ds_mla` must resolve to the #171 extend verifier;
- MRV2 profiling and production capture use the same global graph pool;
- BF16-weight/FP8-output fused-query eligibility is false;
- profiling and production CUDA-graph capture complete;
- API health and a finalized arithmetic response pass;
- no OOM, illegal access, cuBLAS failure, assertion, `EngineDead`, Xid,
  dead worker, or process restart.

Memory:

- absolute minimum GPU KV pool: 480,000 tokens;
- production floor: 500,000 tokens;
- record per-rank available KV, weight, peak activation, non-torch, retained
  graph, and persistent-kernel estimates;
- do not raise GMU if the pool misses the floor. Stop and return the evidence.

## Gate 2 — cold long-context correctness

Use one unique random first block per request and prove `cached_tokens=0`.
Score `content`, `reasoning`, `reasoning_content`, and the serialized message,
but require all of:

- expected needle `738216` in finalized `content`;
- non-empty finalized content;
- `finish_reason=stop`;
- no repetition/degeneration;
- correct arithmetic/coherence side checks.

Order:

```text
50k -> 150k -> 250k -> 350k -> 475k
```

The 150k cell is the early regression discriminator. Stop immediately on a
real MISS or finalization failure. Archive request/response JSON, context
tokens, cache tokens, completion tokens, timing, and all response fields.

## Gate 3 — throughput and decode characterization

Use unique cold prefixes and server-side metric deltas.

Prefill:

```text
8k cold
55k cold
```

Matched CN4 v19 controls are 415 and 392 effective tok/s. Hard floor is 90% of
those matched controls:

```text
8k  >= 374 tok/s
55k >= 353 tok/s
```

Record both effective request throughput and engine chunk rate. A throughput
number without cache-miss evidence is invalid.

Decode:

```text
ctx0:   C1, C4, C8, C16
ctx16k: C1, C4, C8, C16
```

Requirements:

- C1 aggregate at ctx0 at least 55 tok/s (matched v19 parity floor);
- aggregate throughput is nondecreasing with concurrency;
- requested and effective concurrency agree;
- MTP acceptance is nonzero and reported for every cell;
- zero request errors, restarts, or fatal signatures.

Record route counters and DCP communication timings if available. Do not
change the 6 MiB DMA crossover: the four-rank no-model matrix already proved
DMA is 12–15x faster than NCCL at prefill-sized rows.

## Gate 4 — DRAM/NVMe eviction and concurrency

Start the ordered event monitor before load. Its total must include completed
files and in-flight temporary files.

Run 16 overlapping, unique-prefix requests at approximately 50k context:

```text
active demand ~= 800k tokens > measured GPU pool
```

Requirements:

- 16/16 HTTP and model completions succeed with `finish_reason=stop`;
- prefix-cache queries show fresh work, not cache hits;
- GPU eviction occurs;
- DRAM tier activity occurs;
- filesystem-tier creates and reads entries in the new namespace;
- a re-request after eviction demonstrates tier promotion/reuse;
- physical filesystem usage never exceeds 64,000,000,000 bytes;
- no `_build_store_jobs` assertion, OOM, `EngineDead`, 5xx, restart, or
  container-identity change.

The earlier ordered 8 GiB saturation proof remains the capacity-bound proof.
This 64 GB production cell proves integration and turnover; it is not expected
to fill the entire tier.

## Gate 5 — final audit and verdict

On the original process:

- clean liveness response;
- `RestartCount=0`;
- unchanged container ID and `StartedAt`;
- unchanged runtime file hashes;
- no fatal signatures over the complete log;
- record final GPU/DRAM/NVMe utilization and high-water marks.

Verdict is `PASS` only if every prior gate passes. Push the image by immutable
digest after the verdict, then prepare the CN3 production Compose with the
same flags and only host-path/port/restart-policy changes.

Do not mark any draft PR ready solely because the image boots:

- the query-route draft needs Proof 3 plus Gate 2 and Gate 3;
- PR #171 needs Gate 2, decode, and memory evidence;
- the PR #76 follow-up needs its final JSONL hash attached;
- PR #154 needs the complete long-context ladder and measured KV delta.
- PR #168 already has an isolated +11,520-token field result, but the
  consolidated image must preserve that gain without a capture fault.
