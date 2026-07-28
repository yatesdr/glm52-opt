# v20 Boot 6 — aligned MTP block-table proof (Fable handoff)

## Decision

Run **one boot**, with no parameter ladder. Boot 6 keeps the complete Boot 5
profile and changes only the PR #166 source revision.

The old revision trimmed the runner's aligned 1,876-column block table into an
unaligned 1,875-column indexer buffer. It removed the Python shape error but
did not survive production CUDA-graph capture. The replacement makes the
indexer's buffer 1,876 columns as well and preserves that row layout through
MTP expansion.

## Exact source

- Draft PR: <https://github.com/local-inference-lab/vllm/pull/166>
- Required PR head: `c9a2e28db9942e7028ea8ec978b07dcbe8cd420f`
- Updated production file:
  `vllm/v1/attention/backends/mla/indexer.py`
- Boot 5 input SHA-256:
  `4057267018b987e355fb7e56802b1c36219bbd1adc8a7642e356a572674c61b0`
- Boot 6 output SHA-256:
  `d419af9e1e84a1fa316246d0cb930f6559dcb5859a8e71bb54240dbf41fb0cde`

The corrective patch for an image/source tree that **already contains old PR
#166 commit `858d8225`** is:

`patches/v20-mtp-aligned-block-table/0001-fix-mla-preserve-aligned-block-table-width-for-MTP.patch`

Patch SHA-256:
`c274ebe26dbc4d2e1fe924424eca8f00d64b92fa51a3a2306e8f97e183ccda58`

Do not apply that one-commit corrective patch directly to bare v20
`3e731bc0`; use the full PR head when building from bare v20.

## Gate 0 — build and byte proof

Rebuild the exact Boot 5 candidate, changing only PR #166 from `858d8225` to
`c9a2e28d`. Retain:

- PR #69 INT8 wire files;
- PR #165 bounded filesystem tier;
- MRV2 pool-reuse commit `982cda45`;
- the same launcher, model, quantization, cache volume, DRAM tier, and NVMe
  tier.

Before launch, verify the new image:

```bash
docker run --rm --entrypoint sh IMAGE -lc \
  'sha256sum /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/indexer.py'
```

Expected: `d419af9e...` exactly. Also rerun the four unchanged source pins from
Boot 5 (`pcie_dma.py`, `pcie_dma.cu`, filesystem `manager.py`, and
`model_runner.py`).

Optional no-GPU semantic check inside the image:

```bash
docker run --rm --entrypoint python IMAGE -c \
  'from vllm.v1.attention.backends.mla.indexer import _align_block_table_width as a; assert a(1875,64)==1876 and a(1874,64)==1874; print("alignment: PASS")'
```

## Gate 1 — the single proof boot

Use the exact Boot 5 runtime settings:

- `MAX_MODEL_LEN=480000` — do **not** reduce it to 479744;
- TP4 / DCP4 / MTP3;
- `MAX_NUM_SEQS=16`;
- CUDA graph cap 64;
- GMU 0.980;
- `VLLM_DCP_A2A_MAX_TOKENS=16`;
- `VLLM_DCP_A2A_LARGE_BACKEND=ag_rs`;
- `i8_ring` wire mode;
- unchanged 64 GB DRAM offload and bounded 8 GiB NVMe acceptance namespace.

Preserve Boot 5's compile/cache state. A fresh cache is not required: changing
the block-table width changes the SparkInfer tensor compile key from 1,875 to
1,876, so the relevant kernel variant cannot reuse the failed odd-width key.

Watch continuously for:

```text
illegal memory access
Target sizes: [7, 1875]
EngineDeadError
CUDA error
NCCL watchdog
Traceback
```

### Gate 1 PASS

All must hold:

1. profiling speculator capture succeeds;
2. KV allocation reports at least 500,000 tokens;
3. target graph capture completes;
4. production speculator capture completes;
5. API reaches healthy/serving;
6. container ID and `StartedAt` remain unchanged and `RestartCount` remains 0;
7. none of the signatures above appears.

### Gate 1 FAIL

If the same illegal access occurs, stop. Do not vary GMU, graph cap, MNS, A2A,
or max length. Preserve the full log and timestamps. The next artifact will be
a source-instrumented build that synchronizes immediately after each target
graph descriptor, which can localize the pending fault in one diagnostic boot.

## Gate 2 — minimal serving proof

Only after Gate 1 passes:

1. one deterministic short request, `max_tokens >= 64`;
2. four concurrent short requests;
3. sixteen concurrent unique-prefix requests, `max_tokens >= 128`;
4. final deterministic liveness request.

For every cell require HTTP 200, a normal finish reason, no worker error, and
`RestartCount=0`. The C16 cell is important: MTP3 gives the graph-cap-64 path
that production needs.

If Gate 2 passes, continue directly into the existing combined v20 acceptance
ladder (needle, throughput, then DRAM/NVMe eviction). Do not reboot between
Gate 2 and that ladder; the purpose is to qualify the exact week-production
process and cache state.

## Why this is the highest-value next boot

- Four post-trim failures were invariant across memory, graph size, MRV2 pool
  reuse, and A2A routing.
- The separately reported successful v20 configuration uses 479,744, whose
  1,874 local pages are already aligned; it does not cross this seam.
- Boot 6 keeps 480,000 and changes only the source contract, so a pass or fail
  is attributable.
- Local regression suite: `16 passed`; no file-format or diff errors.
