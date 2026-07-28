# v20 NF3 long-context causal status

Date: 2026-07-25

## Decision status

No production fix has been selected.  The `legacy` MLA projection mode is a
causal discriminator only.  It must not become the default merely because it
resembles v19.

The current evidence separates two questions that were previously conflated:

1. Why does the chat API return empty `content` after retrieving the needle?
2. Why does the v20-generated stream fail to leave reasoning when the
   byte-identical v19 stack normally does?

The first question has a source-level answer.  The second remains the actual
v19-to-v20 regression question.

## What is already pinned

The following installed files are byte-identical between the current CN4 v20
process and the known-good CN3 v19 process:

| File | SHA-256 |
|---|---|
| `/model/chat_template.jinja` | `172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679` |
| `/model/tokenizer_config.json` | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` |
| `/model/generation_config.json` | `ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55` |
| `/model/config.json` | `254974797e9f455716a30ab5505ba68272181b20b58a3693e54f94fb8056f3ef` |
| `chat_completion/protocol.py` | `d2fe93e01f7483638d4f01c6bb99668619ee0a76a3a8599f819eda3f105c5aa1` |
| `vllm/parser/glm47_moe.py` | `ce3629319e56e882d25cb75d62e3e7088a4eec1518885fc69fc696eafb4a97b2` |
| `vllm/parser/abstract_parser.py` | `54c8ad805cfc8370804e88e2cc18052d8531c2100becb2be6a167dafea2dfab5` |
| OpenAI chat `serving.py` | `6c3e80dd7d2671049eed9fc8ede5038074b8dddac905652ff17a404965a02066` |
| `vllm/renderers/params.py` | `3a551bcbda5d18c3258ab4a4c17db953e97c733badab9bd2157ff62726c73442` |
| `vllm/renderers/hf.py` | `06f0fad8ed924ccb24434effb28b76c9e55303ca37dad331f39b62a4339edefc` |

Consequently, parser or template byte drift cannot explain the v19-to-v20
behavioral change.

The exact v19 and v20 `serve-glm52.sh` invocations also select the same GLM
reasoning parser.  v19 does not carry v20's Compose-level
`reasoning_effort=high` default, but that does not distinguish the needle
cells: renderer `merge_kwargs` gives the request's explicit nested
`reasoning_effort=low` value precedence on v20.  The rendered-request source
and merge semantics are therefore pinned, not merely inferred.

## The empty-content mechanism

The template opens an assistant `<think>` block.  The GLM parser initializes
in `REASONING` and transitions to `CONTENT` only after a generated
`</think>`.  If the model emits EOS first, a correct answer remains in the
reasoning field and `content` is empty.

There are also two real template/API defects:

- parser thinking-state selection understands `thinking` and
  `enable_thinking`, not `reasoning_effort`;
- the template maps only literal `reasoning_effort=high` to `high`; other
  values fall through to `max`.

Those defects explain misleading “low effort” tests and the blank API
response.  They do not yet explain why v20 emits EOS before `</think>` while
v19, with the same template and parser, passes the deep ladder.

A generic “copy reasoning into content at EOS” change is not acceptable.  It
would expose chain-of-thought, conceal changed model numerics, and turn a
quality regression into an API-shape pass.

## What the current NF3 image can retrieve

On one live `5517197` process, the untemplated completion endpoint returned the
exact ticket at 250k, 350k, and 450k:

| Target | Prompt tokens | Untemplated completion | Chat completion |
|---:|---:|---|---|
| 250k | 245,491 | exact, 5 tokens, stop | exact ticket in reasoning; no content |
| 350k | 343,721 | exact, 5 tokens, stop | 2,000-token reasoning run; length |
| 450k | 441,951 | exact, 4 tokens, stop | exact ticket in reasoning; no content |

The 350k and 450k cells reused 97,792 and 137,216 prefix tokens respectively,
so they are not cold acceptance rows.  In both, the needle itself was beyond
the reused prefix.

This proves the NF3/KV/DCP stack can retrieve at those depths for the
untemplated prompt.  It does **not** yet prove that response parsing alone
causes the chat failure because the chat template changes the model's input
tokens.  The token-identical discriminator below closes that remaining gap.

## v20-only numerical seam

For generated-token batches (`M <= 32`), v20 intentionally changed both MLA
absorbed projections:

| Stage | v19 | v20 |
|---|---|---|
| Query weight | Materialized BF16 `W_UK_T` | Native ModelOpt MXFP8 pack |
| Query BMM | `torch.bmm` | SparkInfer native MXFP8 BMM |
| Query assembly | Staged concat + quantize | Fused assembly + static E4M3 |
| Value weight | Materialized BF16 `W_UV` | Native ModelOpt MXFP8 pack |
| Value BMM | `torch.bmm` | SparkInfer native MXFP8 BMM |

The native BMM dequantizes the checkpoint representation exactly and uses
FP32 accumulation, but it is not required to reproduce cuBLAS's reduction
order.  SparkInfer tests bound its error against a float64 reference; they do
not require v19 byte identity.  Fused MXFP8 query assembly is separately
required to be byte-identical to native BMM plus staged assembly.

This makes the native query/value BMM results the important numerical
question, not concatenation of the RoPE suffix.

The distinction between an intended v20 change and a defect is therefore:

- keeping the checkpoint's MXFP8 weights native, avoiding persistent BF16
  copies, and fusing query assembly are intended improvements;
- silently changing post-E4M3 query bytes without a quality-sensitive
  contract is an uncovered numerical behavior change;
- that behavior becomes the retrieval defect only if a token-identical or
  fixed-seed model A/B shows it controls the generated stream.

A bounded error against float64 is insufficient by itself at a subsequent FP8
boundary.  Conversely, a byte difference from cuBLAS is not automatically a
bug if the native result is at least as accurate and the model-quality gates
remain intact.  The pending decomposition and model A/B decide which case this
is.

## Pending discriminators

1. `harness/v20_chat_token_identity_probe.py`
   (`d3f5f90d1104a1c83626dce5a8c57b3fedab4522c9a887329abc25f3941b0a0e`)
   renders the exact chat token IDs and feeds those IDs through the raw
   completion endpoint.  It separates chat-template input differences from
   response parsing.
2. `harness/v20_nf3_absorbed_projection_probe.py`
   (`ecd99b8890c8d8d044d90a9d8301f6ecacac2be0a419cd69db190e13f845e390`)
   separates native query BMM, native fused assembly, BF16 fused assembly,
   post-FP8 bytes, and native value BMM without loading a model.
3. Only if those results still implicate the projection seam should the
   bundled staged-projection model A/B run.  A passing A/B localizes the seam;
   it does not select the final implementation.

## Forward-fix options if projection is causal

GLM-5.2 has 64 global MLA heads, 78 target layers, and one MTP layer.  At TP4,
each rank owns 16 heads.  Approximate persistent BF16 costs per GPU are:

| Option | Persistent BF16 cost | What remains optimized |
|---|---:|---|
| Correct native kernel numerics | 0 MiB | Everything |
| Materialize query only | 3 MiB/layer, about 237 MiB | Native value BMM; fused BF16 query assembly can remain |
| Materialize value only | 4 MiB/layer, about 316 MiB | Native/fused query path |
| Full v19-style materialization | 7 MiB/layer, about 553 MiB | Neither native projection |

At the observed roughly 8 KiB of KV allocation per token, query-only
materialization costs roughly 31k KV tokens, value-only roughly 40k, and full
materialization roughly 71k.  These are planning estimates; the accepted pool
must come from a same-phase boot measurement.

Preferred order:

1. correct the native kernel contract if a specific implementation defect is
   found;
2. otherwise narrow native eligibility only for the causal projection and
   retain the other v20 optimizations;
3. use full staged materialization only as a diagnostic or last resort.

No option should be applied until the two pending discriminators report and
the result is discussed with Derek.

## 2026-07-26 update: projection/MoE levers closed; CKV prefetch next

The frozen cold causal gate has now rejected the broad legacy-projection
route: the 250k control remained exact, but all three 350k prompts missed.
The route also reduced the KV pool from 559,616 to 491,520, below the 500k
production floor.  The apparent A16 lever was inert for the NF3 hybrid
checkpoint because that checkpoint already routes every expert tier through
its W4A16 execution method.  Neither result selects a source fix.

Raising `MAX_BATCHED_TOKENS` from 2048 to 3072 did not reach the causal gate.
At otherwise identical 480k settings it left 2.98 GiB available for KV versus
3.57 GiB required (estimated maximum length 399,616).  It is therefore not a
valid 480k discriminator without introducing a second memory variable.

Source inspection identifies a narrower v20-only correctness seam:
`VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=1`.  The CKV gather itself remains the
desired packed-record ownership inversion.  The depth-one optimization
gathers a future layer's cache on a side stream before that layer writes its
current chunk, then reconstructs the missing records when the layer consumes
the prefetched buffer.  In the exact production geometry it preallocates
828.4 MiB for two execution lanes.  v19's known-good path did not use this
cross-layer prefetch.

The policy/lifecycle tests do not compare a synchronous gathered CKV buffer
byte-for-byte with the prefetched-plus-appended buffer.  A new CPU proof,
`harness/v20_ckv_prefetch_append_slot_proof.py`
(`b43cb24cfcd485485ffcfd0607d03629a102c6ff4eed205295f8220e42302585`),
exhausted 20,120 deterministic/random geometries and proved the append's
integer owner/slot mapping.  That narrows any prefetch defect to quantized
record equivalence or CUDA stream/collective ordering; it does not exonerate
the path.

The next model gate is staged in
`deploy/glm52-v20-5517197-nf3-prefetch0-20260726.yaml`
(`ba752d1f0a019a0a457d64d6f2fe9c76699a364d39f7a8aed9e57c729c6e444b`).
It keeps the exact stock image, TP4/DCP4/MTP3, 480k, NVFP4 KV, FP8 RoPE,
PXB routing, and MNBT=2048; the only behavioral delta is prefetch depth
`1 -> 0`.  The fail-closed runner now accepts an explicit label subset, so
the first pass runs only the 250k control and one frozen 350k failure.  A
350k miss rejects the lead immediately; an exact recovery triggers the other
two frozen repetitions and then byte-level localization.

Before that model boot, three no-model GPU gates are ready:

| Gate | Exact question | SHA-256 |
|---|---|---|
| CKV record equivalence | Does side-stream gather-history-then-append produce the same 368-byte NVFP4+FP8-RoPE records as owner-write-then-synchronous-gather? | `867baaea013f380923a110f2dba44fa1b04d4b70f544c4c48c92edc15c6e4ab0` |
| DCP owner merge | Does the packaged owner-sharded all-to-all/PyNccl/Triton merge return the same global ID set as the established replicated merge at 2,048 rows × 2,048 candidates? | `0f380cd25970d3230974f91eaef9634f51514f2d72d528ebe7680a070ccaf28d` |
| Fused indexer crossover | Does the packaged cooperative/serial scorer and page-table/top-k path agree with an independent PyTorch oracle at the 16,384 crossover and the observed 87.5k/117k local lengths? | `5805f4a217b0e55bb2a198e019993044dd03b7b2eeff039f37a5d11b245fc181` |

The owner-merge input uses a bijection over the 24-bit FP32 significand, so
all 16,777,216 candidates in the full four-rank geometry have distinct,
exactly representable scores.  A collective/layout defect therefore cannot
be hidden by, or confused with, top-k tie ordering.

Two CPU gates already pass:

- CKV append owner/slot mapping: 20,120 cases;
- PR #177/#178 source/math audit: 720 exact owner/oracle cases including
  tied scores, with the packaged CKV source pinned to
  `a2002892614587a737475ef58834b9445a65de764bcbcd646c586a9162a2f2bf`.

The execution order is fail-fast: owner merge, CKV records, fused indexer,
then the one-variable prefetch-depth-zero boot only if the byte-level gates
do not already identify a defect.
