# GLM-5.2 v20 oldest-boundary selector: community validation specification

Status: **draft implementation, validated on one 4x RTX PRO 6000 Blackwell
system; local matched n=3 shallow KLD complete, awaiting independent
reproduction**

Upstream implementation:
[local-inference-lab/sparkinfer#84](https://github.com/local-inference-lab/sparkinfer/pull/84)

Tracking issue:
[local-inference-lab/vllm#182](https://github.com/local-inference-lab/vllm/issues/182)

## 1. What changed

GLM-5.2 uses a learned sparse-attention indexer. For each query, the indexer
must reduce a very large history to exactly 2,048 positions that the sparse
attention layer is allowed to read.

The current v20 SparkInfer selector is exact for its quantized score field.
However, when many positions fall in the same coarse score bucket at the
selection boundary, it does not preserve the older-candidate preference seen
by the GLM-5.2 checkpoint during calibration. On the frozen failing prompts,
the relevant old token cluster is discarded before final high-resolution
selection. The rest of the attention implementation cannot recover a token it
was never allowed to read.

PR #84 adds an explicit server-static policy:

```text
SPARKINFER_NSA_TOPK_SELECTION_POLICY=oldest_boundary
```

It:

1. finds the Kth threshold using an 8-bit FP16 coarse score;
2. keeps every position strictly above that coarse threshold;
3. keeps at most the oldest 4,096 positions tied in the threshold bucket;
4. refines the bounded candidate set with the existing full-score selector;
5. emits exactly the requested top-k with every write bounds checked.

This does **not** restore the historical v19 out-of-bounds behavior. It
restores only the observed coarse-boundary candidate preference, under an
explicit allocation and bounds contract. `exact` remains the default.

No model weights, KV-cache writer, RoPE kernel, PCIe transport, MLA attention
kernel, MoE kernel, or vLLM source file is changed by the candidate image.

## 2. Scope

The affected implementation is selected when a model with `index_topk` uses
the SparkInfer/B12X sparse indexer. Either of these requests it:

```text
VLLM_USE_B12X_SPARSE_INDEXER=1
--attention-backend B12X_MLA_SPARSE
```

The Gilded Gnosis GLM launchers set both. Consequently, the selector is active
for the `glm52` and `glm52-hybrid` presets without an additional user flag.

This is not specifically an NVFP4 KV-cache defect. BF16 weights or a BF16 KV
configuration can still encounter it if they use the same B12X sparse
indexer. Conversely, a path that bypasses this selector is outside the scope
of PR #84.

End-to-end evidence currently covers:

```text
model family:       GLM-5.2 hybrid
weight format:      NVFP4/NF3 hybrid with online NF3-MXFP8
KV cache:           nvfp4_ds_mla
RoPE cache field:   FP8
attention/indexer:  B12X_MLA_SPARSE
hardware:           4x RTX PRO 6000 Blackwell, PCIe Gen3
parallelism:        TP4 / DCP4 / MTP3
```

Do not generalize the result to unrelated checkpoints without running the
same A/B.

## 3. Pinned review image

Friendly tag:

```text
ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-oldest-boundary-pr84-20260727
```

Immutable pull:

```text
ghcr.io/yatesdr/glm52-serve@sha256:43e5a48781ee5cf40a92cc494749b21306b72280bd1a875721a45422323f2599
```

Public package:

https://github.com/users/yatesdr/packages/container/package/glm52-serve

The immutable digest above is the public OCI index. Its `linux/amd64`
platform manifest is:

```text
sha256:c3062f4fc4b28032b6f85ebdc429d6d3adfc1d7b9206bba88414379f09077106
```

Image identity:

```text
local image ID:
  sha256:43e5a48781ee5cf40a92cc494749b21306b72280bd1a875721a45422323f2599

base image:
  voipmonitor/vllm@sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0

base vLLM:
  0c79e41db41f250ccdfc4be92d171960a5787f73

base SparkInfer:
  e603f74bb67d0fce547336f1fb73c3c23e8f1887

PR #84 commit:
  1a71afe07aa88d1161ed7e3fc68aa5daf58fa80d

installed tiled_topk.py SHA-256:
  b15bab73f1fcd6434f712f6fc99ec5369104969cb9157ae473926bf40d72e23b
```

The image contains both policies. This is important: a valid causal A/B uses
the **same image** for both arms and changes only the policy environment
variable.

## 4. Required A/B design

Run two fresh containers:

| Arm | Environment |
|---|---|
| control | `SPARKINFER_NSA_TOPK_SELECTION_POLICY=exact` |
| candidate | `SPARKINFER_NSA_TOPK_SELECTION_POLICY=oldest_boundary` |

Rules:

1. Use the immutable image digest in both arms.
2. Keep checkpoint, tokenizer, template, driver, GPUs, topology, TP/DCP/MTP,
   MNBT, max length, KV format, FP8 RoPE, transport, graph sizes, and all
   sampling arguments identical.
3. Restart the container between arms. The policy is resolved at module import
   and is intentionally server static.
4. Use separate compile/cache directories for the two arms. This prevents a
   compiled kernel from one policy being reused by the other arm.
5. Run every quality request cold and require `cached_tokens=0`.
6. Pin the rendered token IDs or their SHA-256, not only the source text.
7. Use temperature zero and fixed prompt/sampling seeds.
8. Score `content`, `reasoning`, `reasoning_content`, and the serialized
   message for diagnostic retrieval, but require non-empty finalized
   `content` and `finish_reason=stop` for a production pass.

The companion `docker-compose.oldest-boundary-ab.yaml` is parameterized by
`SELECTION_POLICY` and `CACHE_DIR`, so the same file can launch both arms.

Example:

```bash
export MODEL_DIR=/absolute/path/to/GLM-5.2-hybrid
export IMAGE_REF=ghcr.io/yatesdr/glm52-serve@sha256:43e5a48781ee5cf40a92cc494749b21306b72280bd1a875721a45422323f2599

SELECTION_POLICY=exact \
CACHE_DIR=/absolute/path/to/cache/exact \
docker compose -f docker-compose.oldest-boundary-ab.yaml up -d

# Run and archive the control suite, then stop it.
docker compose -f docker-compose.oldest-boundary-ab.yaml down

SELECTION_POLICY=oldest_boundary \
CACHE_DIR=/absolute/path/to/cache/oldest-boundary \
docker compose -f docker-compose.oldest-boundary-ab.yaml up -d
```

The supplied `GPU_MEMORY_UTILIZATION=0.9848` is the measured 480k fit posture
for the tested 4x96-GB system and this exact image. It is not a universal
recommendation. On different hardware, record any necessary max-length or
memory change and apply it identically to both arms.

## 5. Long-context causal gate

The public reproduction package is:

https://gist.github.com/yatesdr/a2e84aa3171ee0b355649704f04f96a8

Freeze the prompt package once:

```bash
python3 v20_freeze_causal_gate_prompts.py \
  --base http://127.0.0.1:5001 \
  --out ./causal-gate-freeze
```

Run the 250k control and the three frozen 350k rows:

```bash
python3 v20_run_causal_gate.py \
  --base http://127.0.0.1:5001 \
  --freeze-dir ./causal-gate-freeze \
  --out ./causal-result \
  --image <immutable-image-digest> \
  --labels pass-250k-ctl,fail-350k-r1,fail-350k-r2,fail-350k-r3
```

The decisive frozen prompt uses:

```text
needle:                 738216
prompt tokens:          343,721
rendered tokens:        343,727
needle position:        approximately 40%
temperature:            0
max output tokens:      2,000
enable_thinking:        false
required cached tokens: 0
required final content: 738216
required finish reason: stop
```

After the four-row gate, run the randomized cold ladder:

```text
50k, 150k, 250k, 300k, 350k, 475k
```

Every row must:

- contain exactly one needle in the rendered prompt;
- record `cached_tokens=0`;
- retrieve the exact value;
- finalize it in `content`;
- finish with `stop`;
- pass the arithmetic, coherence, and degeneration side checks.

Expected candidate result from the original system:

```text
frozen gate:       4/4 exact
randomized ladder: 6/6 pass through 475k target / 466,493 actual tokens
```

The control arm is expected to preserve the known 250k control and miss the
frozen 350k cases. Independent testers should report the actual result rather
than treating that expectation as an acceptance shortcut.

## 6. KLD protocol

### 6.1 The established BF16 safety gate

Use the published GLM-5.2 BF16 reference:

```text
dataset:
  https://huggingface.co/datasets/festr2/GLM-5.2-BF16-KLD-Reference-Logits-20260708

dataset revision:
  a8fbe8a277394e838c75190a0ab376625dfb1393

source BF16 checkpoint:
  zai-org/GLM-5.2@4d67f66cc64d3219133b767c253b2ad1425c6c88

context / stride / windows:
  2048 / 512 / 1

tensor:
  logits, [2047, 154880], float32

logits_0.safetensors SHA-256:
  87f992a689c054a0548a4b3863da6c809f9239beacd5786d0401e45904fec063

manifest.json SHA-256:
  985120136741037918bcd4dc8da9813c1f6268b35a730302f99cf6b3eebb7606
```

The authoritative candidate-side runner is:

```text
https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.2/gguf-bf16-kld-2026-07-08/scripts/prefill_kld_fallback.py

SHA-256:
  e3958eb8b2f603a8a33e42b851fbaaa0f059e16c69881610c0e6d8a7a7776341
```

The runner captures full-vocabulary prompt logits and computes:

```text
KL(P_BF16_reference || P_candidate)
```

The reference-token identity gate is:

```text
[284, 8396, 425, 10960, 465, 284, 14721, 8396,
 425, 10960, 465, 374, 458, 6364, 4531, 1154]
```

Run one independent container for each repeat. Use three repeats per arm:

```text
exact:            run1, run2, run3
oldest_boundary:  run1, run2, run3
```

Every run must contain `fallback_prefill_kld_done`, report exactly 2,047
positions, match the token-identity gate, exit zero under
`set -euo pipefail`, and retain its raw log and configuration. A shell
pipeline returning zero after the model process was killed is a failure.

The repository wrapper `run_v20_pr84_kld.sh` provides:

```bash
# Fast red-flag check: one fresh exact run and one fresh candidate run.
RUNS=3 ./run_v20_pr84_kld.sh smoke

# Reuses valid smoke rows and completes three independent runs per arm.
RUNS=3 ./run_v20_pr84_kld.sh full

./run_v20_pr84_kld.sh summarize
```

### 6.2 What the 2,048-token result does and does not prove

The selector budget is also 2,048 positions. In the published KLD window,
every causal history fits inside that budget. There is no over-budget history
for either policy to discard.

Therefore this established KLD cell is a valuable **shallow no-regression
gate**, but it is not a selector-sensitive proof of the long-context fix.
Equal KLD between `exact` and `oldest_boundary` is expected and means the
patch did not disturb the in-budget path. It must not be described as evidence
that the boundary policy improves 350k retrieval.

Previous local attempts did not complete:

| Attempt | Result |
|---|---|
| MNBT 1,024, GMU 0.85 | no KV-cache blocks available |
| MNBT 512, eager, GMU 0.90 | process killed with RC 137 before the completion marker |

One old wrapper omitted `pipefail` and consequently printed exit zero for a
failed model process. Those rows are invalid and are not prior KLD results.

The matched three-run comparison on the review image completed on 2026-07-27:

| Policy | Runs | Mean KLD ± SD | Min | Max |
|---|---:|---:|---:|---:|
| `exact` | 3 | 0.15823696 ± 0.00468419 | 0.15539664 | 0.16364348 |
| `oldest_boundary` | 3 | 0.16044075 ± 0.00297924 | 0.15700885 | 0.16236257 |

Paired `oldest_boundary - exact` deltas were `-0.00663463`, `+0.00655420`,
and `+0.00669180`. Their mean was `+0.00220379` (+1.39% of the exact-policy
mean), with sample SD `0.00765461`. The mixed signs and variance larger than
the mean delta do not show a large shallow distribution-level regression at
n=3. This remains a non-selector-sensitive cell for the reason above.

### 6.3 Selector-sensitive quality evidence

Until a full-precision long-context reference-logit corpus is published, the
selector-sensitive adoption evidence is:

1. the frozen 250k control plus three 350k failures, using identical rendered
   token IDs in both arms;
2. the randomized cold ladder through the 475k target;
3. full-precision selector-oracle comparisons on captured long-context rows.

Do not compare independently generated continuations and call the result KLD;
their conditioning histories diverge. A future long-context KLD corpus must
use teacher forcing:

1. choose one fixed continuation token sequence;
2. feed identical prompt and continuation IDs to the BF16 reference, `exact`,
   and `oldest_boundary`;
3. capture full-vocabulary pre-softmax logits at the same output positions;
4. compute `log_softmax` and KL in FP64;
5. report mean, median, p95, p99, max, and prompt-level paired confidence
   intervals.

The primary comparison would be:

```text
delta_KL =
  KL(reference || oldest_boundary)
  - KL(reference || exact)
```

Negative is better for the candidate.

Do **not** calculate “KLD” from:

- generated text alone;
- top-20 API logprobs with missing probability mass discarded;
- different prompt token IDs;
- different generated histories;
- warm/cached rows mixed with cold rows.

If only truncated logprobs are available, label the result
`truncated-support divergence`, preserve an explicit “other” probability
bucket, and do not present it as full-vocabulary KLD.

### 6.4 Selector-level companion metrics

For captured indexer rows, compare both policies to the full-precision
selection oracle and report:

```text
recall@2048
Jaccard@2048
score-weighted false-negative mass
needle-neighborhood inclusion by layer
first layer selecting the needle neighborhood
repeatability across identical runs
```

This connects an output-distribution change to the operator that was changed.

### 6.5 Interpretation

There is not yet a justified universal numeric KLD cutoff for this checkpoint.
Adoption evidence should therefore include all of:

1. the frozen causal gate;
2. the randomized ladder;
3. the established 2,048-token BF16 KLD no-regression gate;
4. selector-level long-context oracle metrics;
5. a teacher-forced long-context KLD only if a valid reference corpus exists;
6. performance and capacity measurements.

A strong result is:

- no statistically distinguishable 2,048-token KLD regression;
- improved oracle recall/weighted false-negative mass;
- exact repeated deep retrieval;
- no safety, performance, or capacity gate failure.

## 7. Performance and capacity

After quality:

1. record KV-pool tokens and maximum admitted context;
2. run cold 8k and 55k prefill with prefix-cache deltas;
3. run decode C1/C4/C8/C16 with MTP acceptance;
4. report all boot/restart/fatal signatures;
5. compare policies on the same image and same process configuration.

Candidate baseline from the original system:

| Metric | Result |
|---|---:|
| max model length | 480,000 |
| KV pool | 545,280 tokens |
| cold prefill 8k | 1,460 tok/s |
| cold prefill 55k | 1,501 tok/s |
| decode C1 | 55.06 tok/s |
| decode C4 | 109.95 tok/s |
| decode C8 | 144.58 tok/s |
| decode C16 | 180.50 tok/s |
| restarts / fatal signatures | 0 / 0 |

These are orientation numbers, not cross-machine acceptance thresholds.

## 8. Evidence to publish

Please include:

```text
host GPU model/count and nvidia-smi topo -m
driver version
image tag and immutable digest
vLLM and SparkInfer commits
installed tiled_topk.py SHA-256
checkpoint and revision/digest
tokenizer and chat-template identity
complete quality-sensitive environment
active selector policy
max model length and KV pool
prompt text SHA-256 and rendered-token-ID SHA-256
cached_tokens, finish_reason, raw response fields
per-row retrieval verdicts
KLD reference identity and exact computation method
raw per-prompt KLD aggregates
prefill/decode numbers with cache deltas and MTP acceptance
restart count and fatal-log audit
```

Suggested result heading:

```text
GLM-5.2 v20 PR84 selector A/B
image: <digest>
checkpoint: <identity>
hardware: <identity>
exact: <summary>
oldest_boundary: <summary>
KLD reference: <identity>
quality verdict: <pass/fail/inconclusive>
```

## 9. Existing evidence pins

Current-v20 frozen causal result:

```text
dda7bddd33919d0947bcf45e0731c7fe07e1d4918944781fca9928cafe1d18f6
```

Current-v20 randomized ladder result:

```text
b855f1febae880a6ae146797fbf37707e3ea02bccd213578d41ec5ba19ae6268
```

Production-shape, model-free 3,072-row GPU safety result:

```text
6d32434593a932026ad16fdde2aded4f5e1b45c584cadc52452edf4397e6b23d
```

The review image should remain a candidate until independent KLD and matched
A/B results are available.
