# v20 depth-zero CKV workspace reclaim

Date: 2026-07-26  
Status: CPU-proven; consolidated CN4 runtime gate in progress  
vLLM base: `0c79e41db41f250ccdfc4be92d171960a5787f73`  
Patch commit: `ff9b8fc50`

## Problem

The production posture explicitly disables CKV lookahead:

```text
VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=0
```

Even at depth zero, v20 allocates a persistent CKV workspace pool sized for
two speculative execution lanes. On the qualified TP4/DCP4/MTP3 boot this
logged:

```text
Preallocated 460.2 MiB for 2 persistent CKV execution lane(s)
```

Depth-zero gather uses one local staging region and one gathered-cache slot
only for the duration of the current attention forward. It does not carry CKV
data across layers. Reserving two cross-layer persistent lanes therefore
consumes real device memory without providing overlap.

## Fix

Use the persistent pool only when the effective prefetch depth is greater than
zero.

At depth zero, append the CKV byte workspace to the existing
`WorkspaceManager.get_simultaneous()` request beside the query, optional dense
output, and attention scratch. This matters because every independent
workspace-manager borrow begins at offset zero. All simultaneously-live views
must be requested together or they can alias.

The runtime split is:

| Effective depth | Workspace lifetime | Behavior |
|---:|---|---|
| 0 | forward-local borrowed view | synchronous gather |
| >0 | persistent per-lane pool | cross-layer lookahead |

The prefetch-depth budget may cap a requested positive depth to zero; that case
also takes the borrowed synchronous path.

## Expected memory effect

The current persistent pool is approximately 460.2 MiB per GPU. Borrowing the
single synchronous workspace grows the already-profiled transient workspace
by approximately one 230 MiB lane. The expected net reclaim is therefore about
230 MiB per GPU, or roughly 28k--31k NVFP4 MLA KV tokens at the measured token
density.

This is real allocation removal, unlike raising GPU memory utilization or
disabling memory accounting. It does not change attention math, selector
semantics, KV format, or transport.

## Off-GPU proof

Focused vLLM suites:

```text
tests/v1/attention/test_b12x_ckv_prefetch_policy.py
tests/v1/attention/test_b12x_mla_dcp_workspace.py
```

Result:

```text
66 passed
ruff check: PASS
ruff format --check: PASS
git diff --check: PASS
```

The standalone in-image proof additionally checks that:

- depth zero does not request persistent storage;
- positive depth does;
- the synchronous CKV view is returned from the same borrow as query/scratch;
- all live ranges are disjoint.

Artifact:

```text
/home/derek/proof-results/20260726/sync-ckv-borrow-cpu/result.json
sha256:8ee4c7d5bdc138ace0817390d35688a3f7cb3511c0bd72c58e719c63df77f91b
```

## Forward-base compatibility

The relevant source and tests are byte-identical between the qualified
`5517197` image and the topology-calibrated `0c79e41` image. The patch rebased
onto `0c79e41` without conflict.

The derived candidate is:

```text
glm52-serve:v20-20260726-prod-optimized-candidate
image ID: sha256:29f474acd4eee61517dec76cfcf36afd47097c3cf9ea7e598784d09abe516856
```

It also carries SparkInfer PR #80 and draft PR #82, with no diagnostic code.

## Runtime acceptance

One consolidated CN4 boot must show:

1. no `Preallocated ... persistent CKV execution lane(s)` line at depth zero;
2. an explicit synchronous borrowed-workspace line;
3. KV pool above the 500,992-token qualified baseline, with the delta
   consistent with the predicted reclaim;
4. max model length 480,000 and graph capture complete;
5. cold 475k retrieval exact;
6. no fatal signatures or restart;
7. cold 8k/55k prefill and C1/C4/C8/C16 decode recorded before promotion.

Positive-depth behavior is unchanged and remains covered by the existing
prefetch state/ring tests.
