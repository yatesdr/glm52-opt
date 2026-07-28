# CN4 GLM-5.2 EXL3-TR3 qualification record

Date: 2026-07-28  
Host under test: CN4 only, 4x RTX PRO 6000 Blackwell 96 GB, TP4/DCP4  
Status: corrected candidate passed the frozen deep-retrieval gate; canonical
CN4 topology/scheduling performance posture is now being qualified; NF3
rollback remains preserved

## Objective

Build a reproducible EXL3-TR3 v20 candidate with the applicable correctness,
dynamic-NVFP4, cache-ABI, offload, and CN4 topology fixes, then compare it with
the latest NF3 candidate using the same hardware and measurement contracts.

Required measured outputs:

- MTP3 sustained decode at concurrency 1, 2, 4, and 16
- MTP0 sustained decode
- cold standalone prefill
- reported GPU KV capacity
- frozen and randomized deep-context retrieval
- BF16-reference KLD, n=3
- explicit NF3-vs-TR3 comparison

## Immutable inputs

| Component | Pin |
|---|---|
| TR3 package repository | `brandonmmusic-max/glm52-exl3-sparkinfer@0f9b71d5049228589534614c6a15c139c8238959` |
| Published TR3 v31 image | `sha256:0433ae94665b769b78dd301f952d907508a3ba80bce47a1630ec20ade8812dff` |
| Published TR3 base | `voipmonitor/vllm:gilded-gnosis-v20-vllm0c79e41-sic3828fd-fi801d57a-cu132-20260727@sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0` |
| TR3 model repository | `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` |
| TR3 model revision | `9297b9f1d53af5c67cffa01e30cc071a1ff7144b` |
| Model repository storage | `336,543,556,886` bytes |
| Checkpoint manifest SHA-256 | `bfb6dc39f28da08c1cfc5b89603414046adf7003152d69e9ee350e11f7a1fa63` |
| Checkpoint config SHA-256 | `fcde001350291a0048318d4a1136e0732e31f829f804a57cfbb558903e54171a` |
| EXL3 vLLM integration | PR #139 head `00787eeabebc11cee12cff12a823011b4e1a5ebc` |
| EXL3 SparkInfer integration | PR #49 head `669a12ddc7cf3021e91a25f398b1a883b703fd12` |
| Dynamic vLLM integration | PR #189 head `b57062274c3f53bec69b431bfae7230977f5f10c` |
| Dynamic SparkInfer integration | PR #86 head `0ddd13b4fdbb6a287581aec55fcf9dbbb7e52fd3` |
| Bounded filesystem offload | `a1f5cc6cd0bcbcedfc607f98afbd8883ea3a3d5a` |

The checkpoint's full runtime payload was verified at the pinned revision:
all 79 layer shards, embedding/head shards, indexes, tokenizer, tier map,
config, and calibration assets matched `MANIFEST.sha256`. The repository's
manifest contains stale entries for only `.gitattributes` and `README.md`;
independent downloads of those two files from the exact Hugging Face revision
matched the CN4 copies byte-for-byte. Neither file is used at runtime.

## Source compatibility result

The dynamic-NVFP4 work is applicable to TR3. EXL3 changes routed expert
weights; both TR3 and NF3 still use the same `nvfp4_ds_mla` KV-cache writer,
368-byte record, sparse-attention readers, and offload machinery.

The two integrations were replayed as real Git histories, not copied by
inspection:

- vLLM #189 cherry-picked cleanly onto vLLM #139. Runtime validation then
  found four base registrations absent from the resulting `envs.py`;
  restoring those registrations produced `42779a70`. The existing bounded
  filesystem-tier commit was then applied as `a247c7ac`.
- SparkInfer #86 cherry-picked cleanly onto SparkInfer #49, producing
  `7ef6d26e`. The EXL3 production files and dynamic MLA production files are
  disjoint. The runtime-stride base delta from #86's base to `c3828fd` changes
  none of the overlaid MLA files.
- Python compilation passed for all 18 runtime files in the combined trees.

The derived image also includes PR #189's immutable record-mode capture and
cache/offload namespace separation. That matters because a legacy/static
368-byte record and a dynamic record have the same byte length but different
meaning at bytes `[292,296)`.

## Built candidate

| Artifact | Pin |
|---|---|
| Local image tag | `glm52-exl3-tr3-dynamic:20260728` |
| Local image ID | `sha256:a5608e0b4a2fcdaec476de79fbe5cf2f6e9ce2ecf30bf2dfe0c1314d97c6666e` |
| Dockerfile SHA-256 | `814e015d18829ab6986b39a430260843f9e3aee21a5ed2b83f93585a45d7b2ce` |
| Qualification Compose SHA-256 | `d1d9a58d103683ba12338c507a36cb2517b27a3b5063d72ab86cdc91964b2d66` |
| KLD wrapper SHA-256 | `d8e3d0dd3492e44787a312d6f2f8f37407af3f4e3ca0c87cec2453886f397191` |
| Combined vLLM tree | `a247c7ac20cc2041954cd75d19844f5e600d154b` |
| Combined SparkInfer tree | `7ef6d26ee36b13b8e5473090f2f79ab8a371201c` |

The build completed on CN4 with exit code zero. Its labels pin the TR3
repository/model, parent image, both EXL3 integration heads, and both merged
dynamic-scale trees. The in-image Python compilation gate passed.

The no-model cache-format gate passed:

```text
cache_format_gate=PASS
abi=nvfp4_ds_mla:fp8-rope-368:dynamic-token-v1
tr3_fs_gate=PASS
```

That gate asserted:

- vLLM recognizes the dynamic-mode environment variable;
- the cache ABI is the dynamic 368-byte FP8-RoPE variant;
- static and dynamic scale modes reject simultaneous activation;
- dynamic mode rejects a non-FP8-RoPE record layout;
- every overlaid runtime file has the expected source-tree SHA-256.
- the filesystem tier accepts and enforces `max_cache_size_bytes`.

## Integration failures closed before qualification

Two failures were found before any request was served:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is incompatible with
   `OffloadingConnector`; the inherited compose setting was removed.
2. The merged EXL3/dynamic `envs.py` omitted four registrations used by the
   parent runtime (`VLLM_PCIE_DMA_MIN_BYTES`,
   `VLLM_DCP_REPLICATE_INDEXER_CACHE`, `VLLM_DCP_INDEXER_SHARDS`, and
   `VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS`). They are restored and asserted
   at image build time.
3. The qualification compose enabled bounded NVMe capacity, but the initial
   derived image did not overlay the matching filesystem-tier implementation.
   It reached model load, reported 1,101,312 GPU KV tokens, and then rejected
   `max_cache_size_bytes` at scheduler construction. The existing v20 bounded
   tier commit is now included and asserted in the corrected image.
4. Replacing the first healthy model container left its root-owned 64 GB
   `/dev/shm/vllm_offload_<engine-id>.mmap` allocated. The next boot created a
   second mapping, filled the 63 GB host tmpfs, and all workers failed
   `mmap.madvise()` with `OSError: [Errno 14] Bad address`. Both mappings were
   confirmed orphaned (`fuser` empty, no GPU compute PIDs) before the two exact
   paths were removed. `/dev/shm` returned from 100% to 1% use and the
   unchanged canonical posture was relaunched. This is an offload lifecycle
   defect, not evidence against the SYS/query-split posture.

No request result or performance number was collected from a failed boot.

## CN4 matched serving posture

The current NF3 control resolved the following on this same host:

- `NCCL_P2P_LEVEL=PXB`
- `F8_DMA=i8_ring`
- `MAX_NUM_BATCHED_TOKENS=3072`
- `DCP_QUERY_SPLIT=0`
- `DCP_TOPK_OWNER_MERGE=0`
- `DCP_INDEXER_SHARDS=0`
- `DCP_CKV_PREFETCH_DEPTH=0`
- exact v20 selector

The TR3 qualification Compose pins those values rather than importing the
published TR3 author's host-specific `CUDA_VISIBLE_DEVICES=3,1,2,0`,
`NCCL_P2P_LEVEL=SYS`, FP8 `ag` wire, query split, owner merge, and CKV
prefetch settings.

That matched PXB posture is retained as the topology control. A second pinned
overlay changes only `NCCL_P2P_LEVEL=SYS`, query split, owner merge, and CKV
prefetch depth to the upstream TR3 TP4/DCP4 posture. `i8_ring`, exact top-k,
dynamic NVFP4, FP8 RoPE, MTP3, context, offload, image, and checkpoint remain
unchanged. The overlay SHA-256 is
`20fb2b6202a01d924a079515b7cd84353c053fb75912c548b4304d84ba759142`.

TR3-specific invariants retained:

- `--quantization exl3`
- serving aliases `GLM-5.2` (for byte-identical frozen harness replay) and
  `GLM-5.2-EXL3-TR3-3.0bpw`
- Trellis routed experts including the layer-78 MTP draft
- asynchronous scheduling disabled
- `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=0`
- exact v20 top-k
- `MAX_MODEL_LEN=480000`
- dynamic NVFP4 + FP8-RoPE 368-byte cache records

The checkpoint declares `GlmMoeDsaForCausalLM`, 78 target layers plus the
layer-78 draft shard in its 79-layer manifest, exact `index_topk=2048`, and
`max_position_embeddings=1,048,576`. Attention/indexer and shared/dense
weights are excluded from the routed-expert 3-bpw quantization contract.

Prefix caching is enabled. The production-like serving arm uses the same
64 GB DRAM and bounded 1 TB Intel-NVMe offload tiers as the NF3 control, under
a distinct dynamic-TR3 cache namespace.

## Preserved NF3 rollback

Before staging TR3, CN4 was healthy with:

- container: `glm52-prod`
- image: `sha256:b4ef498c6b3494961ba381ccc85215590d5e3b3cd17ec04146152a101b653790`
- Compose:
  `/home/derek/glm52-v20-prod-clean-r2-20260728/compose.yaml`
- GPU KV capacity: 617,728 tokens
- dynamic NVFP4 records, prefix caching, 64 GB DRAM offload, bounded 1 TB NVMe
- restart count: 0

This Compose and its cache/offload namespaces are not modified by TR3 staging.

## Measurement contracts

### Serving smoke and identity

Record image ID, labels, model revision/manifest, resolved process environment,
GPU ordering/topology, driver/clocks/power limits, boot log, restart count,
health endpoint, KV capacity, and prefix-cache enablement.

### Decode

Use the same `llm_decode_bench.py` harness and zero-context sustained-decode
method as the NF3 runs. Record aggregate and per-request throughput, request
fill, MTP acceptance, errors, power, temperature, and throttling.

- MTP3: concurrency `1,2,4,16`
- MTP0: at least concurrency 1; additional matched cells may be recorded
- duration: 20 seconds per cell after warmup
- temperature: 0
- max output: 8192

`MAX_NUM_SEQS=16` is required so C16 is measured rather than reported as
capacity-capped.

### Prefill

Use cold standalone prefill with randomized first blocks so prefix caching
cannot turn a nominally cold row into a warm result. Headline metric is
`prompt_tokens / TTFT`; record server counter deltas when uncontaminated.
Initial contexts: 8k, 64k, and 128k. A deep row is added only after retrieval
passes.

### Deep retrieval

Run the existing frozen NF3 causal rows first, including the 250k control and
three 350k failures that exposed the v20 precision problem. Then run the
seeded depth/context ladder through the configured 480k ceiling. Record exact
prompt hashes, real prompt token counts, cache deltas, completions, and
verdicts.

### KLD

Use the same local BF16 reference-logit artifact and fallback runner as the
latest NF3 dynamic-scale study:

- reference logits SHA-256:
  `87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063`
- 2,047 scored positions from the pinned 2,048-token WikiText-2 sample
- MTP disabled, eager execution, exact selector
- TP4, DCP1
- NVFP4 MLA KV with FP8 RoPE and dynamic per-token outer scaling
- three independent cold runs
- report mean, sample standard deviation, min/max, and all raw values

The primary matched comparison is TR3 dynamic versus the archived NF3 dynamic
n=3 result. Published TR3 KLD values are contextual only unless their
reference artifact, prompt tokens, posture, and scoring direction are proven
identical.

The staged KLD wrapper was preflighted while NF3 was still live. It verified
the candidate image ID/merge labels, runner, BF16 reference, manifest,
validator, compile-prover, and model path, then exited with the required code
3 because the four NF3 worker PIDs still owned the GPUs. This is the intended
fail-closed overlap guard; no KLD work ran against occupied devices.

## Fresh matched NF3 control

While the TR3 checkpoint transferred, the preserved NF3 service was measured
with the exact harness and commands planned for TR3. No restart or
configuration change occurred. The NF3 image remained healthy with zero
container restarts and reported 617,728 GPU KV tokens.

MTP3 zero-context sustained decode, 20 seconds per cell:

| Concurrency | Aggregate tok/s | Per-request tok/s | Errors |
|---:|---:|---:|---:|
| 1 | 79.6476 | 79.6476 | 0 |
| 2 | 95.9780 | 47.9890 | 0 |
| 4 | 144.0618 | 36.0154 | 0 |
| 16 | 218.6431 | 13.6652 | 0 |

Cold standalone prefill:

| Requested context | Actual prompt tokens | TTFT | Client tok/s | Cached tokens |
|---:|---:|---:|---:|---:|
| 8k | 8,200 | 6.076 s | 1,351 | 0 |
| 64k | 64,511 | 54.371 s | 1,187 | 0 |
| 128k | 128,880 | 109.871 s | 1,173 | 0 |

The randomized calibration block was newly generated and the server-side
counter independently reported `cached_tokens=0` for every row.

Evidence:

- `harness/cn4-evidence-archive/20260728/tr3-qualification-nf3-matched-control-v1/`
- decode JSON SHA-256:
  `121a214506680551c2e1ce107a76656a8f5f8ae4d3193427c4b8f20ebd91d7fa`
- prefill JSON SHA-256:
  `50129c7a95b13ba25e7c37bc824cabe215001852c09a3ef59ac0d87da216e4f3`

## Corrected TR3 frozen retrieval result

The corrected image reported 275,328 local KV tokens per DCP rank, or
**1,101,312 physical GPU KV tokens** at DCP4. All four production writer
compile specs recorded `per_token_scale=true`.

The frozen causal gate produced the exact ticket `738216` in the 250k control
and in all three formerly failing 350k prompts:

| Row | Prompt tokens | Result | Output tokens | Wall time |
|---|---:|---|---:|---:|
| 250k control | 245,497 | EXACT | 4 | 418 s |
| 350k r1 | 343,727 | EXACT | 4 | 575 s |
| 350k r2 | 343,727 | EXACT | 4 | 575 s |
| 350k r3 | 343,727 | EXACT | 4 | 575 s |

The legacy harness labeled the rows non-cold because this API returned
`prompt_tokens_details=null`. That label is a response-schema mismatch, not
cache evidence. Authoritative server counters were archived after every row:
`prefix_cache_hits_total=0` and `external_prefix_cache_hits_total=0`
throughout, while `prompt_tokens_total` advanced cumulatively to 1,276,697,
exactly the four prompt lengths plus request framing. The cold-run verdict is
therefore **PASS: control exact and 3/3 failures recovered exactly**. The raw
harness output is retained unchanged.

Evidence root:
`/home/derek/glm52-tr3-qualification-20260728/results/tr3-mtp3/frozen-causal-a560/`

- final metrics SHA-256:
  `b0cf96ea8feeb898cc438be2512be254cc32c9ec960c6dd3f63d69b836cf8f6f`
- raw rows SHA-256:
  `09b39dd93308af6e97674f60d317a5ec70e2ef172336201fb0c380be69d992e4`
- raw summary SHA-256:
  `7cad33e79a76919871b3889d747a3daa406ff0266b220cf6850d7e1154fa53c9`

## TR3 PXB topology control

MTP3 zero-context sustained decode, 20 seconds per cell:

| Concurrency | Aggregate tok/s |
|---:|---:|
| 1 | 72.2 |
| 2 | 110.6 |
| 4 | 149.5 |
| 16 | 92.4 |

Cold standalone prefill:

| Requested context | Actual prompt tokens | TTFT | Client tok/s |
|---:|---:|---:|---:|
| 8k | 8,200 | 6.01 s | 1,364 |
| 64k | 64,494 | 94.17 s | 685 |
| 128k | 128,854 | 203.06 s | 635 |

This long-context prefill collapse is the reason for the single-variable
canonical SYS/query-split posture qualification. Decode JSON SHA-256 is
`dd44f3528d0ca4e39793a1ef76be0c39dcf84e785267935c52d490cdd02de3a7`;
prefill JSON SHA-256 is
`4cd50e202e308b27170ffeea74e2e67986594d3683fe15f46e1a6d02e06f9a5b`.

## Canonical SYS/query-split/owner-merge rejection

The single topology overlay was booted unchanged apart from:

- `NCCL_P2P_LEVEL=SYS`
- `VLLM_DCP_QUERY_SPLIT=1`
- `VLLM_DCP_TOPK_OWNER_MERGE=1`
- `VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=1`

The image, checkpoint, `i8_ring`, exact selector, dynamic NVFP4 record,
FP8-RoPE, MTP3, and offload configuration remained fixed. It reached health
with restart count zero and a 1,099,264-token GPU KV pool.

Matched MTP3 decode fell to 63.3 / 67.2 / 76.3 / 49.0 aggregate tok/s at
C1 / C2 / C4 / C16, versus 72.2 / 110.6 / 149.5 / 92.4 under PXB. The
direction and magnitude reject this overlay for CN4. The long prefill matrix
was stopped during its 128k cell at the user's direction because the decode
result had already made the production decision and further characterization
would delay KLD. Completed prefill cells showed no new internal or external
prefix-cache hits. The original PXB posture remains the best qualified TR3
configuration.

## Production decision table

This is the final user-facing comparison table. The NF3 column is populated
from already-proven production records; CN4 will not be rebooted into NF3
again. TR3 cells are filled only from the corrected candidate and exact
qualification protocol.

| Metric | Production NF3 | Best qualified TR3 |
|---|---:|---:|
| GPU KV pool | **617,728 tokens** | **1,101,312 tokens** |
| Configured max request | 480,000 | 480,000 |
| Needle 50k | PASS, exact, cold | pending |
| Needle 250k | PASS, exact, cold | PASS, exact, cold |
| Needle 450k | PASS exact at 450k; separate cold ladder passed 475k | pending |
| Frozen 3×350k recovery | 3/3 exact, cold | **3/3 exact, cold** |
| MTP0 C1 decode | **45.901 ± 0.0218 tok/s**, n=10 | pending |
| MTP3 C1 decode | **79.648 tok/s** fresh matched; 75.3 prior operational | **72.226 tok/s** |
| MTP3 C2 aggregate | **95.978 tok/s** | **110.589 tok/s** |
| MTP3 C4 aggregate | **144.062 tok/s** fresh matched; 133.3 prior operational | **149.456 tok/s** |
| MTP3 C16 aggregate | **218.643 tok/s** | **92.357 tok/s** |
| Best proven cold prefill | 1,382 tok/s @8k; 1,315 tok/s @64k | **1,364 tok/s @8k** |
| Fresh matched cold prefill | 1,351 @8k; 1,187 @64k; 1,173 @128k | **1,364 @8k; 685 @64k; 635 @128k** |
| DRAM offload | PASS, 64 GB | initialized; functional gate pending |
| NVMe offload | PASS, bounded 1 TB | initialized at 0/1 TB; functional gate pending |
| BF16-reference KLD | **0.13903565 ± 0.00201006**, n=3 | pending |

Needle evidence uses finalized `content`, not reasoning-only retrieval. The
NF3 450k record itself was not cold, but the same production dynamic-record
stack independently passed a cold randomized 466,493-token row; both facts
are retained rather than collapsing them into a stronger claim than measured.

## Artifacts

- Dockerfile:
  `docker/Dockerfile.glm52-exl3-tr3-dynamic-20260728`
- CN4 qualification Compose:
  `compose/glm52-exl3-tr3-dynamic-cn4-qualification-20260728.yaml`
- matched KLD runner:
  `harness/run_glm52_tr3_dynamic_kld.sh`
- matched KLD validator:
  `harness/summarize_glm52_tr3_dynamic_kld.py`
- combined vLLM tree:
  `workspace/vllm-tr3-dynamic-integration`
- combined SparkInfer tree:
  `workspace/b12x-tr3-dynamic-integration`
- CN4 evidence root:
  `/home/derek/glm52-tr3-qualification-20260728`

Results and verdicts will be appended only after the corresponding artifact
has passed its fail-closed checks.
