# PR #86 / #189 bot-review closure — 2026-07-28

## Scope

This report closes the four pre-merge requests raised against:

- SparkInfer [#86](https://github.com/local-inference-lab/sparkinfer/pull/86)
- vLLM [#189](https://github.com/local-inference-lab/vllm/pull/189)

The validation window used CN4 only. Its production compose was checksummed
and backed up before testing, stopped once, and restored from the identical
file after testing. CN3 was not touched.

## Reviewed source identities

| Repository | PR head |
|---|---|
| SparkInfer | `0ddd13b4fdbb6a287581aec55fcf9dbbb7e52fd3` |
| vLLM | `b57062274c3f53bec69b431bfae7230977f5f10c` |

The final local integration image is
`glm52-v20-pr86-pr189-cleanup-final:20260728`, image ID
`sha256:beecd32f951c8f4479da8c920fffbc0a1d249655057202c80c3fc05b78c22c55`.
Its labels pin both full commit hashes.

## Review request closure

### 1. Record-mode identity and persistent caches

The record itself remains an inline-scale record, not a self-identifying
record. Safety is enforced externally:

- one frozen `Nvfp4MlaCacheFormat` is captured per process;
- every writer and reader imports the same instance;
- incompatible writer/reader signatures and layouts fail during
  initialization;
- dynamic records use a versioned external-cache ABI;
- static calibrated records include the scale-file SHA-256 in their ABI;
- resolved bytes per KV block also participate in persistent file identity.

Non-NVFP4 and unconfigured/implicit NVFP4 modes retain the literal
`vllm-default-v1` identity, so their existing persistent cache namespaces do
not change. This was a valid follow-up issue caught by CodeRabbit and fixed in
vLLM commit `b5706227`.

### 2. Subnormal wording and test

The implementation guarantees that the largest group scale in each token is
positioned near the top of E4M3. It does not guarantee that every smaller
group in the token is normal. Documentation and tests now state and verify
the narrower invariant.

### 3. Focused tests

The final vLLM source passed:

- `67 passed`, 17 warnings, 37.08 seconds;
- pinned Ruff 0.14.0 check: pass;
- pinned Ruff 0.14.0 format check: 11 files unchanged.

Coverage includes:

- server-static mode capture;
- dynamic + static rejection;
- dynamic + non-368-byte rejection;
- incompatible SparkInfer writer and reader rejection;
- both normal and DCP gathered-chunk writer sites;
- static-file content hash;
- dynamic/static/geometry namespace separation;
- legacy namespace preservation through the real offloading-config flow for
  float16, bfloat16, auto, and implicit NVFP4;
- vLLM environment registration.

SparkInfer production-path GPU tests passed `28/28` on CN4, covering static
and dynamic records, zero-token handling, multisplit decode, and multitile MG
prefill.

### 4. Matched MTP0 decode

The measurement used the same image ID, compose, model, TP4/DCP4 layout,
fixed prompts, two uncounted warmups per arm, and 1,024 output tokens per
counted request. MTP was disabled and server metrics confirmed zero draft and
accepted tokens. Offload was disabled for both arms to prevent cross-ABI
records.

| Arm | n | Mean tok/s | Sample SD | Range |
|---|---:|---:|---:|---:|
| Static calibrated | 10 | 46.076 | 0.0268 | 46.04–46.12 |
| Dynamic per-token | 10 | 45.901 | 0.0218 | 45.87–45.93 |

Paired observed delta: `-0.175 tok/s`, or `-0.380%`. This passes the
pre-committed 1% decode-reader overhead limit.

Raw records and post-run GPU state:

- `harness/cn4-evidence-archive/20260728/pr86-pr189-cleanup/static-mtp0.jsonl`
- `harness/cn4-evidence-archive/20260728/pr86-pr189-cleanup/dynamic-mtp0.jsonl`
- `harness/cn4-evidence-archive/20260728/pr86-pr189-cleanup/static-post-gpu.csv`
- `harness/cn4-evidence-archive/20260728/pr86-pr189-cleanup/dynamic-post-gpu.csv`

## Previously completed quality evidence retained in the PRs

| Gate | Result |
|---|---|
| Frozen 245,497-token control | pass |
| Frozen 343,727-token failures | 3/3 recovered |
| Randomized 49,098–466,493-token ladder | 6/6 pass |
| KLD static calibrated | 0.146228 mean, 0.004688 SD, n=3 |
| KLD dynamic per-token | 0.139036 mean, 0.002010 SD, n=3 |
| Matched KLD change | -4.92% |
| Prefill wall-time change | +0.74% to +1.87%, within 2% gate |
| Dynamic KV pool at 480k max length | 550,144 tokens |

## Final image smoke

The final integration image:

- imports the frozen format configuration;
- registers `VLLM_NVFP4_MLA_DYNAMIC_SCALE` and
  `VLLM_NVFP4_MLA_SCALES_FILE`;
- emits `nvfp4_ds_mla:fp8-rope-368:dynamic-token-v1` for dynamic records;
- emits `vllm-default-v1` for non-NVFP4/default records;
- carries the exact SparkInfer and vLLM cleanup commit labels.

## CI note

The vLLM pre-run workflow accepts `verified`, `ready`, or
`ready-run-all-tests`, but the repository currently defines none of those
labels. Local and CN4 gates are complete; a repository maintainer must create
or apply one of the workflow-accepted labels before full hosted CI can start.

## CN4 production restoration

Final read-only audit after the validation window:

| Check | Restored value |
|---|---|
| Compose SHA-256 | `767ab28a118b47fa12ae4a6e9aeb494ad1ee1ab6afe0ee67cac7fc6047b9e570` |
| Image ID | `sha256:cacf3304e586906aa504aab966f2eed6e82e34f6700bad06ac65e69313f37cdc` |
| Container | `glm52-prod`, healthy, restart count 0 |
| Runtime posture | TP4 / DCP4 / MTP3, 480k max length, MNBT 3072 |
| KV pool | 617,728 tokens |
| DRAM offload | 64 GB shared mmap |
| NVMe offload | 1 TB limit at `/nvme-kv/glm52-v20-dynamic-prod` |
| Dynamic NVFP4 | enabled, 368-byte FP8-RoPE record |
| Health endpoint | HTTP 200 |
| Power service | active; 300 W and `-lgc 0,2600` on all GPUs |

The original model, cache, and NVMe mounts were restored. The temporary MTP0
test container was removed before production startup, and no further restart
was performed after the healthy audit.
