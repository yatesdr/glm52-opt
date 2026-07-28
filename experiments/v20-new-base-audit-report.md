# New v20 decode-base compatibility audit

Status: **complete — rebuild is unblocked**  
Date: 2026-07-22  
Verdict: **`REBUILD-CLEAN`**

The new base does not contain block-INT8 wire support or bounded filesystem-tier
eviction. Both patched seams are nevertheless byte-identical to the old v20
base, so the existing PR outputs can be copied onto the new image without a
rebase or source adaptation.

## Source identity and method

The image labels resolve to these exact source commits:

| Component | Old v20 | New v20 target |
|---|---|---|
| vLLM | `2167295cd3e133d38ab22a67a42b0004db65d0a6` | `3e731bc043d23ec21277fb76d3e15fe6da91b23b` |
| SparkInfer | `6a92bcc0f2bf03b13dd03dbc7ce97e26133c580e` | `1a88b389a8d14f26dbe4c157965938cfd8f1bf51` |

The histories diverge, so this audit used direct old-tree-to-new-tree diffs,
not only first-parent commit logs. For all three patch seams, the old and new
Git blob IDs are identical:

| Seam | Old/new Git blob |
|---|---|
| `sparkinfer/comm/pcie/pcie_dma.py` | `b8179c8159c8c5758da2170456ba8de02036107a` |
| `sparkinfer/comm/pcie/pcie_dma.cu` | `06642f961bb4d0f701cf2ace5e711def5237f069` |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | `1ecad2f9c2a3248e14680536bcc81f2e76ae8231` |
| `vllm/v1/kv_offload/tiering/fs/manager.py` | `f12ef2c6d4d3daddefbc6ced42309e6ff83b7fa4` |

## Q1 — What is the decode improvement?

The prime suspect is **not** the selected-record sparse-CKV decode stack.
[vLLM #159](https://github.com/local-inference-lab/vllm/pull/159),
[#160](https://github.com/local-inference-lab/vllm/pull/160), and
[#161](https://github.com/local-inference-lab/vllm/pull/161), plus SparkInfer
[#64](https://github.com/local-inference-lab/sparkinfer/pull/64) and
[#65](https://github.com/local-inference-lab/sparkinfer/pull/65), are all still
open and are absent from the target commits.

The relevant new vLLM change is `3e731bc0`, **auto-route B12X MTP verification
to decode**, in
`vllm/v1/attention/backends/mla/b12x_mla_sparse.py`:

- metadata now distinguishes a genuine speculative-verification batch from a
  short chunked prefill by requiring all requests to have completed prefill;
- those MTP verification rows use SparkInfer `run_decode` instead of the much
  heavier `run_extend` route;
- the production MTP3 shape has at most four query rows per request and is
  admitted by the default eight-row limit; and
- short prefills continue to use extend, avoiding the old unsafe blanket
  interpretation of every short extend as decode.

This is **enabled by default**. The existing
`VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE` variable changed from default `0` to
default `auto`:

- `auto` (default): route only verified MTP batches;
- `0`/`false`/`off`/`no`: disable the route; and
- `1`/`true`/`on`/`yes`: force eligible short extends through decode.

`VLLM_B12X_MLA_SPEC_DECODE_MAX_Q` remains default `8`. No new Compose variable
is required, and production should leave the route in `auto` rather than force
it with `1`.

Other new-source changes are not the main TP4 decode mechanism:

- SparkInfer `93fc919` qualifies MXFP8 BMM for TP8 MLA shards; current TP4 does
  not use that new geometry.
- SparkInfer `d4f82a6` removes live page-table width from a compile-cache key;
  it reduces unnecessary variants, not steady-state TP4 decode work.
- vLLM `f06173f9` fixes memory-resource profiling.
- SparkInfer `695c011` makes the fused micro-MoE all-CTA-barrier launch
  cooperative. That is a concurrency safety correction, not itself evidence
  of the reported throughput increase.

### Correction to the old-base concurrency note

The previous audit said the vLLM #150 / SparkInfer #60 shared-expert workaround
was part of v20. It was present in the old image's trees, but both PRs were
closed without merge and their source changes are absent from the new tree.
The new tree instead contains the lower-level cooperative-launch fix
`695c011`, which requires whole-grid admission for the fused barrier kernel.
This is intentional replacement, not an omitted overlay. Keep the 16 x 50k
concurrency gate; do not restore the closed #150/#60 pair and do not add
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`.

## Q2 — Does the new base contain either optional patch?

No.

The target SparkInfer mode normalizer still accepts only E4M3 `ag`, `ring`, and
`a2a` modes. Its CUDA source uses `__nv_fp8_e4m3` and has no signed-INT8 codec,
INT8 quant/dequant kernels, or `i8` dispatch.

The target vLLM filesystem manager has no `max_cache_size_bytes`, capacity
index, reservations, pinning, LRU eviction, or bounded-store failure path.

## Q3 — Does INT8 PR #69 apply cleanly?

**Verdict: clean COPY.**

The target `pcie_dma.py` and `.cu` are byte-identical to the files against
which [SparkInfer PR #69](https://github.com/local-inference-lab/sparkinfer/pull/69)
was built. PR head `d6f0baa3107b3e774c047945234188f75636da9a` is based on
`d4f82a6`; the later target changes do not touch either PCIe-DMA file.

The new vLLM `custom_all_reduce.py` is also byte-identical to the old base and
already has the integration PR #69 needs. Do **not** copy a replacement vLLM
file. Verify its base-native hash only.

Wire-variable precedence is worth pinning precisely:

- SparkInfer directly reads `SPARKINFER_PCIE_DMA_FP8` when constructed without
  an explicit mode.
- The integrated vLLM path reads `VLLM_PCIE_DMA_FP8`, falls back to legacy
  `B12X_PCIE_DMA_FP8`, and passes that value explicitly into SparkInfer.
- Consequently, `SPARKINFER_PCIE_DMA_FP8` alone does not select the mode on the
  integrated vLLM path. Tonight's Compose correctly keeps both
  `SPARKINFER_PCIE_DMA_FP8=i8_ring` and `VLLM_PCIE_DMA_FP8=i8_ring`; retain both.

## Q4 — Does bounded NVMe eviction apply cleanly?

**Verdict: clean COPY.**

The target `fs/manager.py` has MD5
`5e341cdfef3456ae72f00063756d4dc9`, exactly the patch input. The patched file
is available from draft [vLLM PR #165](https://github.com/local-inference-lab/vllm/pull/165),
head `a1f5cc6cd0bcbcedfc607f98afbd8883ea3a3d5a`. This forward port is an exact
replay of the v19 implementation because the input manager bytes did not
change.

Copy only the patched runtime manager into the image. The PR's documentation
and tests belong in source review and are not runtime overlays.

## Q5 — Config and deep-context parity

The new target retains the required production interfaces:

- `VLLM_USE_B12X_FP8_GEMM`, `VLLM_USE_B12X_MOE`,
  `VLLM_USE_B12X_SPARSE_INDEXER`, `VLLM_USE_B12X_WO_PROJECTION`,
  `VLLM_USE_B12X_MHC`, and `VLLM_USE_B12X_DCP_A2A` are defined and consumed.
- `VLLM_DCP_GLOBAL_TOPK`, `VLLM_DCP_SHARD_DRAFT`,
  `VLLM_DCP_QUERY_SPLIT`, `VLLM_DCP_A2A_MAX_TOKENS`,
  `VLLM_DCP_A2A_LARGE_BACKEND`, `VLLM_DCP_PROJECT_BEFORE_MERGE`, and its
  threshold remain consumed.
- `VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE` and the existing full-record
  `VLLM_B12X_MLA_CKV_GATHER` route remain present. This is distinct from the
  unmerged selected-record sparse-decode PR stack.
- `VLLM_NF3_GRID188_DECODE`, `KV_FP8_ROPE`, `nvfp4_ds_mla`, and the GLM-only
  368-byte record accounting remain present. The cache shape and page-size
  accounting both select 368 B/token with `KV_FP8_ROPE=1`.
- TP4/DCP4, MTP3, GMU 0.970, max length 480,000, MNS 16, and MNBT 3072 remain
  valid configuration surfaces.

`VLLM_USE_B12X_PCIE_DMA` is not read in this target and is vestigial; this is
not a new regression. The actual PCIe activation gates are
`VLLM_ENABLE_PCIE_ALLREDUCE=1` and
`VLLM_PCIE_ALLREDUCE_BACKEND=b12x`, which the candidate already sets.

### Memory accounting

`f06173f9` changes sparse DCP transient profiling so projected AG/RS fallback
rows below the workspace start are not omitted, and ensures a lazily-created
breakable CUDA-graph runner inherits the disposable profiling pool.

It should not change this production profile's KV capacity:

- projection begins above 1,024 rows;
- the workspace begins at 1,025 rows, so there is no newly-accounted projected
  fallback interval before workspace takeover;
- if workspace eligibility fails, both old and new code already profile the
  projected fallback; and
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` makes the graph-pool correction inactive.

The MTP decode change raises the decode plan's nominal maximum from 16 to 64
rows for MNS16/MTP3, but the already-allocated 3,072-row extend plan remains
the larger shared scratch layout. No material static-pool reduction is expected.

Therefore 644,864 tokens remains the expected boot pool and 480,000 remains the
hard acceptance floor. Record the actual pool; a changed value still requires
investigation rather than silently repinning the expectation.

## Q6 — Interaction risk

There is no direct code conflict with either overlay.

The MTP change is in B12X sparse-MLA attention dispatch. PR #69 changes the TP
all-reduce transport, while bounded NVMe eviction changes filesystem-tier block
retention. Neither patched file is called by the new branch-selection logic.

INT8 wire should remain dormant during decode. At the largest production
verification batch, 16 requests x 4 MTP rows x 6,144 hidden values x 2 BF16
bytes is 786,432 bytes, one eighth of vLLM's fixed 6,291,456-byte DMA threshold.
Large prefill tensors still exceed the threshold and exercise `i8_ring`.

The remaining runtime couplings are covered by tonight's existing gates:

- MTP3 deep needles and general quality cover correctness of the new
  `run_decode` verifier route.
- The 16 x 50k load covers the cooperative micro-MoE launch, shared-expert
  scheduling, DRAM overflow, and filesystem-store concurrency together.
- Restarted promotion covers offload lookup/load pinning and persisted NVMe
  reuse under the new attention base.

Do not set `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1`; default `auto` is the
specific route reviewed here.

## Rebuild recipe

Base image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm3e731bc-si1a88b38-fi801d57a-cu132-20260722
```

Copy exactly these three runtime files:

| Source | Image target | Action |
|---|---|---|
| SparkInfer PR #69 `sparkinfer/comm/pcie/pcie_dma.py` at `d6f0baa3` | `/opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/pcie_dma.py` | clean COPY |
| SparkInfer PR #69 `sparkinfer/comm/pcie/pcie_dma.cu` at `d6f0baa3` | `/opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/pcie_dma.cu` | clean COPY |
| vLLM PR #165 `vllm/v1/kv_offload/tiering/fs/manager.py` at `a1f5cc6c` | `/opt/venv/lib/python3.12/site-packages/vllm/v1/kv_offload/tiering/fs/manager.py` | clean COPY |

Do not copy `custom_all_reduce.py`; the new base already has the required
integration byte. Use a fresh extension directory such as
`TORCH_EXTENSIONS_DIR=/cache/v20_sparkinfer_ext` so the replaced CUDA source is
compiled and no prior E4M3-only extension can be mistaken for the candidate.

Retain both wire variables:

```text
SPARKINFER_PCIE_DMA_FP8=i8_ring
VLLM_PCIE_DMA_FP8=i8_ring
```

## Byte pins

### New base inputs / verify-only file

| File | MD5 | SHA-256 |
|---|---|---|
| base `pcie_dma.py` | — | `438adc8fcf63929e1e930c421c091ed6a121fc9b6d3e4c34295c7daacfbccaae` |
| base `pcie_dma.cu` | — | `c1cc60258b287be1a50a5f1d73a3b00ebbedd53f8774c2b8b40c1a98777bce23` |
| base `custom_all_reduce.py` (verify only) | — | `1a15c6266e2f7eb1a64b74d2db1504663e1535ff411e63a9eeae1f0abe6349c6` |
| base `fs/manager.py` | `5e341cdfef3456ae72f00063756d4dc9` | `ffb5904cba86f5703e2acd6204bf4664c300eaee9e959656040864491c94548b4cc60` |

### Required candidate outputs

| File | MD5 | SHA-256 |
|---|---|---|
| patched `pcie_dma.py` | — | `5a6e6a0ef72fd2e46d5b8a42106763817998d411cf4d55d2ecb127c63d9630d5` |
| patched `pcie_dma.cu` | — | `70f4be323350353bfe2df8c41c6129907a786f0ef25831a0b5604ef5e9161048` |
| base-native `custom_all_reduce.py` | — | `1a15c6266e2f7eb1a64b74d2db1504663e1535ff411e63a9eeae1f0abe6349c6` |
| patched `fs/manager.py` | `a72eeb81c735036b281ff97f5d759122` | `653edbf4b393e2acd6204bf4664c300eaee9e959656040864491c94548b4cc60` |

These patched output pins are intentionally unchanged from the old-base
candidate; source identity proves that this is reuse, not a blind repin.

After building, pin the new candidate by image digest and verify all four
runtime hashes inside the container before inference.

## Decode measurement recommendation

No new workload shape is required for acceptance. The existing ctx0 C1/C8/C16
and ctx50k C8 cells all run MTP3 and will measure the delivered new route.

For attribution, wrap at least ctx0 C1 and C16 with the existing
`workspace/sol-decode-route-mtp/harness/decode_mtp_cell.py` metric wrapper so
generated, drafted, and accepted-token deltas are captured. Use a matched old
v20 candidate result as the branch-performance baseline if one exists. The v19
`i8_ring` values 63.2 / 127.2 / 165.1 tok/s remain the production acceptance
reference, but they cannot by themselves attribute a delta specifically to
`3e731bc0`.

An environment-toggle A/B (`auto` versus `0`) on the same new source would be
the causal proof, but it requires another boot and is not necessary for
tonight's go/no-go window.

## Final disposition

**`REBUILD-CLEAN`** — build from the new image, copy the existing PR #69 two-file
SparkInfer output and PR #165 one-file manager output, verify the unchanged
vLLM integration byte, and run the combined acceptance ladder. No new re-port
branch or PR is required.
