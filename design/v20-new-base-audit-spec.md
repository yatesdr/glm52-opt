# Audit request: new v20 base (decode branch) — patch compatibility for tonight's acceptance

Status: **audit requested; needed before tonight's v20 window**
Requested by: Fable (operator)
Assigned: Sol
Date: 2026-07-22

## 1. Objective

A newer v20 base landed with reported significant decode increases. Before tonight's combined
v20 acceptance (`v20-combined-cn3-acceptance-spec.md`), determine authoritatively:

1. **what changed** in the new base (identify the decode improvement in code);
2. **whether the new base still lacks** our two unmerged patches (block-INT8 `i8_ring` wire, bounded
   NVMe fs-tier eviction);
3. **whether our patches still apply cleanly** onto the new base, or need re-porting/re-basing —
   because the new base moved **both** the vLLM and SparkInfer commits, which are exactly the files
   our patches touch; and
4. the **new byte pins** and a **rebuild recipe** so we can rebuild the candidate on the new base and
   re-verify.

The test *structure* (Sol's combined acceptance) is unchanged. This audit only rebases the image and
byte pins onto the new decode branch and, if needed, scopes a re-port.

## 2. Images

- **New base (decode branch, target):**
  `voipmonitor/vllm:gilded-gnosis-v20-vllm3e731bc-si1a88b38-fi801d57a-cu132-20260722`
- **Old v20 base (previously audited):**
  `voipmonitor/vllm:gilded-gnosis-v20-vllm2167295-si6a92bcc-fi801d57a-cu132-20260721`
- **Our candidate on the OLD base (for reference):**
  `ghcr.io/yatesdr/glm52-serve@sha256:bc4b0185bb72a2722a57e615587f67bf35cd16554720514d47ee0184051a4cd7`
  (= old v20 base + PR #69 i8 wire + NVMe eviction manager.py)

Both bases are already pulled on CN3. The new-base pull is completing as of this writing.

## 3. What we already know (old-base baseline — do not re-derive)

From the old v20 base audit:

- **Already in v20 base** (no patch needed): rank-consistency (b12x #44 owner-shard, `pcie_dma.py`);
  KV-offload store-boundary (vLLM #153, our #133); shared-expert/graph-channel concurrency fix
  (vLLM #150 `can_overlap_shared_experts`).
- **Absent in old v20 base** (we bake these): block-INT8 `i8_ring` wire (only E4M3 `ag`/`ring`/`a2a`
  present; env `SPARKINFER_PCIE_DMA_FP8`); bounded NVMe fs-tier eviction.
- **Env parity confirmed on old base:** `VLLM_USE_B12X_MOE/FP8_GEMM/SPARSE_INDEXER/WO_PROJECTION/MHC/DCP_A2A`,
  `KV_FP8_ROPE`, `VLLM_NF3_GRID188_DECODE`, `VLLM_PCIE_*`, `VLLM_DCP_*` all still read;
  `nvfp4_ds_mla` supported. Wire env renamed `B12X_PCIE_DMA_FP8` → `SPARKINFER_PCIE_DMA_FP8`.

## 4. Our patch artifacts (paths + current pins on OLD base)

| Patch | File path(s) in image | Pin on old base / our bake |
|---|---|---|
| #69 i8 wire (py) | `sparkinfer/comm/pcie/pcie_dma.py` | SHA-256 `5a6e6a0ef72fd2e46d5b8a42106763817998d411cf4d55d2ecb127c63d9630d5` |
| #69 i8 wire (cu) | `sparkinfer/comm/pcie/pcie_dma.cu` | SHA-256 `70f4be323350353bfe2df8c41c6129907a786f0ef25831a0b5604ef5e9161048` |
| #69 vLLM integration | `vllm/distributed/device_communicators/custom_all_reduce.py` | SHA-256 `1a15c6266e2f7eb1a64b74d2db1504663e1535ff411e63a9eeae1f0abe6349c6` |
| NVMe eviction | `vllm/v1/kv_offload/tiering/fs/manager.py` | input MD5 `5e341cdfef3456ae72f00063756d4dc9` → patched MD5 `a72eeb81c735036b281ff97f5d759122` (SHA-256 `653edbf4b393e2acd6204bf4664c300eaee9e959656040864491c94548b4cc60`) |

Source: b12x PR #69 (`feat/sparkinfer-v20-int8-wire`, head `d6f0baa3`); NVMe commit `d74c0a6c` on
`yatesdr/vllm-opt`.

Base image site-packages path prefix: `/opt/venv/lib/python3.12/site-packages/`.

## 5. Audit questions (please answer each explicitly)

### Q1 — What is the decode improvement?
Diff the new base against the old base and identify, in code, what drives the reported decode gains.
Prime suspect: the sparse-CKV decode stack the v20 release notes listed as excluded/future work
(vLLM #159–#161, SparkInfer #64–#65). Confirm or correct. Name the files/kernels/flags responsible,
and whether it is **on by default** or gated behind an env/flag we must set. Note any new env var.

### Q2 — Does the new base already contain the i8 wire and/or NVMe eviction?
- `pcie_dma.py`/`.cu`: does the new base ship any `i8`/`i8_ring`/`i8_a2a` block-INT8 modes, or is it
  still E4M3-only (`ag`/`ring`/`a2a`)? (grep the `.cu` for int8 codec; grep `.py` for the mode
  normalizer.)
- `fs/manager.py`: does it now contain `max_cache_size_bytes` / LRU eviction, or is bounded eviction
  still absent?

### Q3 — Does #69 (i8 wire) apply cleanly, or need re-porting?
The `si` commit moved (`6a92bcc` → `1a88b38`). Determine:
- Did the new base **rework `sparkinfer/comm/pcie/pcie_dma.py`/`.cu`** relative to the old base? If
  so, #69's ported bytes (`5a6e6a0e`/`70f4be32`) will not drop in — scope the re-port delta.
- Did the new base **change `vllm/.../custom_all_reduce.py`** (the vLLM-side DMA construction/dispatch
  #69 relies on)? If it moved off SHA `1a15c626`, does #69's integration still apply, or conflict?
- Verdict: **clean COPY**, **rebase-only**, or **needs a new #69 re-port PR**.

### Q4 — Does the NVMe eviction patch apply cleanly, or need re-basing?
Compute the new base's `fs/manager.py` MD5. If it equals the patch input `5e341cdf`, the patched
`manager.py` (`a72eeb81`) drops in byte-clean. If it changed (the vLLM commit moved), the `d74c0a6c`
patch must be re-based onto the new manager — scope that delta.

### Q5 — Config/deep-context parity on the new base
Confirm the new base still: reads `SPARKINFER_PCIE_DMA_FP8`/`VLLM_PCIE_DMA_FP8` for the wire; reads the
`VLLM_USE_B12X_*`, `KV_FP8_ROPE`, `VLLM_NF3_GRID188_DECODE`, `VLLM_DCP_*` vars; supports
`nvfp4_ds_mla` + `KV_FP8_ROPE=1` (368 B/token) at 480k. Flag any change to memory accounting that
would affect **GMU 0.970 fitting 480k** (the v20 notes mentioned MRV2/sparse-DCP transient memory).

### Q6 — Interaction risk: decode change vs our patches
Does the new decode stack touch the MLA decode / DCP path in a way that interacts with the i8 wire
(the wire is prefill-dominant, dormant at decode below the 6 MiB DMA threshold) or the offload/NVMe
path? Call out any coupling we must validate tonight beyond the existing gates.

## 6. Required deliverable

A short report answering Q1–Q6, plus:

1. **Rebuild recipe** for the candidate on the new base: exact files to COPY (with target paths), and
   for each, whether it is clean/rebase/re-port.
2. **New byte pins** (SHA-256, and MD5 for `manager.py`) for every file we bake, so we can re-verify
   in-image and update `v20-combined-cn3-acceptance-spec.md` §2.
3. **Verdict:** `REBUILD-CLEAN` (COPY existing bytes), `REBUILD-REBASE` (minor rebase, you provide the
   files), or `BLOCKED-NEEDS-REPORT` (a #69 and/or NVMe re-port PR is required first — name it).
4. If a re-port is required, the branch/commit/PR to bake from.
5. Whether tonight's test plan needs a **new decode-measurement cell** to capture the reported gain,
   and against what baseline (v19 `i8_ring`: ctx0 C1/C8/C16 = 63.2 / 127.2 / 165.1 tok/s).

Operator (Fable) will rebuild the image on the new base from your recipe, re-verify the pins, and
update the acceptance spec. Do not modify PR #69 or the NVMe branch for tonight unless Q3/Q4 require a
re-port; if so, keep it as a disposable test integration and note CI/unit status separately.
