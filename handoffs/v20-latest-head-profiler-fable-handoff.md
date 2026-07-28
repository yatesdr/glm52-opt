# v20 latest-head production integration + one-boot profiler handoff

Date: 2026-07-25  
Code owner: Sol  
Runtime owner/operator: Fable  
Hosts: CN4 first; CN3 remains read-only/protected  
Status: prepared and CPU/static-proven; **do not boot without Derek's go**

## Outcome this handoff is designed to produce

Use one diagnostic boot to answer which measured phase still limits 55k cold
prefill on the exact latest-head production stack. Do not use this boot for a
configuration ladder and do not promote its diagnostic image.

This work deliberately separates:

1. a clean production image containing only intended production changes; and
2. a non-promotable child containing the fail-closed phase profiler.

The current candidate's cold-needle qualification takes precedence. Preserve
that process and all evidence. Start this procedure only after that gate has
finished and Derek explicitly authorizes the next boot.

## Exact source graph

Published binary base:

```text
voipmonitor/vllm@sha256:adddafd2b1749729fdf2d2ca23818c7c39f2a95e6fb05edd98657251913b83f2
vLLM in base:       992b874cf7ae504616bbb1d2d4f7a7355be6972b
SparkInfer in base: a93df671cc7b33734f499b57228e542c3d3c3697
```

Clean production source:

```text
worktree: workspace/vllm-v20-prod-integration-head
branch:   integration/v20-prod-head-20260725
base:     89b4a98d1ffebb2dda1e1ac5e55238e3a9cfbd58
head:     625ac3b75bf26741b4d8de06a46ec803a8a80f23
```

The nine independent commits, in order:

```text
2dc0c67e  #165 bounded filesystem tier
33b4b59e  native B12X MTP3 flattening gate (Brandon Music)
6c8aa17b  native-gate tests
85b1ffd8  #154 absorbed kv_b_proj source reclaim (Martin Vit)
07ec1206  #154 documentation (Martin Vit)
031aa09a  #154 fused-query-era test adaptation
4a637301  staged BF16-weight -> FP8-output safety guard
f3df903f  #171 compact NVFP4 verifier qualification
625ac3b7  #168 MRV2 global graph-pool reuse
```

SparkInfer production source:

```text
worktree: workspace/sparkinfer-v20-current-recovery
upstream: c39b8062ba450c030e669d898a026d10980c9470
head:     d4969d993cdd16cc417056d471af42d10ac3fada
```

This is current master plus the persistent PCIe-DMA output allocation and
validation follow-up. The source delta from the published base is entirely
Python/Triton; no extension or CUDA/C++ binary changed.

Diagnostic child:

```text
worktree: workspace/vllm-v20-prod-profiler-head
branch:   diagnostic/v20-prod-profiler-head-20260725
parent:   625ac3b75bf26741b4d8de06a46ec803a8a80f23
head:     71053e516c13279d5735a54431cb44a8111d4af3
```

## Why the profiler is the next useful discriminator

The staged BF16-to-FP8 query guard cannot explain the current cold-prefill
shortfall. SparkInfer's fused BF16 query is eligible only through `M <= 32`.
The production scheduler uses 3,072-token chunks; every chunk of the observed
8k and 55k cold requests was larger than 32. The guard therefore affects
tiny-M decode/speculation, not those prefill measurements.

Executable proof:

```bash
python3 harness/v20_prefill_query_route_static_proof.py
```

Expected:

```text
PASS staged-query prefill-route proof: scheduler_chunk=3072 fused_max_m=32
```

The profiler measures the remaining layer wall time rather than selecting
another optimization by inference.

## Gate 0 — local/source proof and image build

Run from the repository root:

```bash
python3 harness/v20_prod_head_integration_static_proof.py
python3 harness/v20_prod_head_packaging_proof.py
python3 harness/v20_compute_profiler_static_proof.py \
  --tree workspace/vllm-v20-prod-profiler-head \
  --expected-parent 625ac3b75bf26741b4d8de06a46ec803a8a80f23
python3 harness/v20_compute_profiler_state_proof.py \
  --profiler workspace/vllm-v20-prod-profiler-head/vllm/model_executor/layers/compute_phase_profiler.py
python3 harness/v20_prefill_query_route_static_proof.py
```

Every command must print `PASS`. A failure stops the build.

Build the clean production layer first, but do not boot it:

```bash
docker build \
  -f docker/Dockerfile.v20-prod-head-20260725 \
  -t glm52-v20-prod-head:20260725 \
  .
```

Then build the non-promotable profiler child:

```bash
docker build \
  --build-arg PRODUCTION_IMAGE=glm52-v20-prod-head:20260725 \
  -f docker/Dockerfile.v20-prod-profiler-head-20260725 \
  -t glm52-v20-prod-profiler-head:20260725 \
  .
```

Required build evidence:

- both builds exit zero;
- production label `io.yatesdr.diagnostics=none`;
- profiler label `io.yatesdr.promotable=false`;
- exact source revisions from the labels;
- image IDs and resolved parent IDs;
- no bypass of a base-byte or manifest check;
- the production image has no
  `vllm/model_executor/layers/compute_phase_profiler.py`;
- the profiler child does have that file.

The package manifests cover 2,650 production vLLM files, 2,651 profiler vLLM
files, and 167 SparkInfer files. They are stronger than selected-file spot
checks.

## Gate 1 — one profiler boot

Derive the runtime Compose from
`deploy/glm52-v20-prod-ready-20260724.yaml`. Make only these functional
changes:

1. image: `glm52-v20-prod-profiler-head:20260725`;
2. a unique container name and unique NVMe namespace;
3. add exactly:

```text
B12X_COMPUTE_PROF=1
B12X_COMPUTE_PROF_MODE=prefill
B12X_COMPUTE_PROF_CALLS=78
B12X_COMPUTE_PROF_FIRST_LAYER=0
B12X_COMPUTE_PROF_NUM_LAYERS=78
B12X_COMPUTE_PROF_ROWS=3072
B12X_COMPUTE_PROF_POLL_TOKENS=1
B12X_COMPUTE_PROF_EXPECT_TP_ATTN=1
B12X_COMPUTE_PROF_EXPECT_TP_MOE=1
B12X_COMPUTE_PROF_BASELINE_MS=0
B12X_COMPUTE_PROF_TOP_N=12
```

Preserve the established CN4 posture:

- TP4/DCP4/MTP3;
- MNS16 and graph sizes `1,2,4,8,16,32,64`;
- max model length 480,000;
- max batched tokens 3,072;
- GMU 0.978;
- `NCCL_P2P_LEVEL=PXB`;
- i8-ring;
- CKV gather cap 480,000;
- 300 W / 2,600 MHz host ceiling;
- restart disabled;
- no autoheal;
- no CUDA-graph diagnostics;
- no additional tuning variable.

Boot failure ends the attempt. Do not change GMU, graph sizes, DCP route, or
wire mode and retry.

## Gate 2 — one cold 55k measurement

After the API is healthy, submit harmless liveness requests if needed. The
`ROWS=3072` selector prevents them from arming the profiler.

Then submit exactly one salted cold 55k request:

- unique random first block;
- `cached_tokens=0`;
- normal production response budget;
- archive complete request and response;
- require HTTP success, `finish_reason=stop`, and non-empty finalized content.

Do not run another throughput cell before collecting the profiler record.

Required log result:

- exactly one summary per rank, ranks 0 through 3;
- `mode=prefill`;
- `calls=78`;
- `rows=3072`;
- `route=split_tiers`;
- `ledger_valid=1`;
- `ordinal_valid=1`;
- `phase_count_valid=1`;
- `baseline_valid=1`;
- `negative_buckets=0`;
- `tp_attention_bad_layers=0`;
- `tp_moe_bad_layers=0`;
- absolute `unaccounted_pct <= 2.0`;
- no `state=disarmed`, `state=init_failed`, fatal signature, restart, or
  container-identity change.

Save the full container log, then validate and render it:

```bash
python3 harness/v20_compute_profiler_report.py \
  /path/to/full-container.log \
  --world-size 4 \
  --mode prefill \
  --calls 78 \
  --rows 3072 \
  --json-output /path/to/v20-prefill-profiler-summary.json
```

The parser must print JSON with `verdict: PASS`. Preserve the JSON and its
SHA-256.

## Stop point and handback

Stop after the single valid profiler sample. Do not run a decode profile,
configuration ladder, or full qualification suite on this diagnostic image.
Return:

- image IDs and labels;
- Compose SHA-256;
- container ID, `StartedAt`, and restart count;
- cold-request JSON and cache-miss evidence;
- full log SHA-256;
- parsed profiler JSON and SHA-256;
- per-rank `mean_layer_ms`;
- ordered `phases` table from the parser;
- fatal-signature audit.

Sol will use the largest valid exclusive phase to select the next code change.
The subsequent clean production image will be rebuilt from
`integration/v20-prod-head-20260725` plus only that proven fix. The profiler
child itself is never a promotion candidate.

## Prepared artifact hashes

```text
.dockerignore
  80c421d3eb3e69b12f7b5f9898a98df34198d292e0188c1102fb0861a78dfc87
docker/Dockerfile.v20-prod-head-20260725
  04de5483311e0dc9720ba19239e49a2962f6f92546de2c72605ef53ae347134b
docker/Dockerfile.v20-prod-profiler-head-20260725
  bc8c61e4199c59862b16ed2c4fe57866d95cd1328fc23ccf3aa4322936014fb5
harness/v20_prod_head_integration_static_proof.py
  a7bcc390af4b9c67d496dcf6ceb83f719a8982bc1b4b2dfcabe5173529ad8c27
harness/v20_prod_head_packaging_proof.py
  b30e6279e6c0fdebfff084abab1616c35bb549476057d8aaacfb8bf4ea2b7d09
harness/v20_compute_profiler_static_proof.py
  3ae2c32e307666cc7b587d5fd2e4eb218e26264ff00e97b9d75326ea2e5c6fc5
harness/v20_compute_profiler_state_proof.py
  55e960b55e8e9b8e996813ba2d263afe0dab0778ba4c84697681cac6c5720947
harness/v20_compute_profiler_report.py
  28d8eeb3dfa0100163d7004158ea6a79fdf66bab9c3a52d5df943076abebbc7e
harness/v20_prefill_query_route_static_proof.py
  81b89904dbb1a9b8bdbe81f65bcddba0a6e7f8f79584cd4eae31ad212a10bcf8
```
