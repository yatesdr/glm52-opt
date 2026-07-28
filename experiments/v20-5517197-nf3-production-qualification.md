# v20 `5517197` NF3 production qualification

Date: 2026-07-25

> **BLOCKED — do not boot this candidate yet.** The no-`#171` fa71
> discriminator missed cold 100k, so the conditional path that selected this
> stock image did not pass. In addition, every current failure used
> `NCCL_P2P_LEVEL=PXB`, whereas the known 6d32 cold-250k pass predates the PXB
> routing change. Complete the same-image SYS/PXB causal control first. The
> launcher work below remains valid packaging work, but PXB is not
> quality-qualified.

## Selected semantic base

Use the published image as the exact vLLM/SparkInfer/FlashInfer/model-runtime
base:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm5517197-sibe0edca-fi801d57a-cu132-20260725
sha256:e7a8a8549c10b5d16899e0fb45ff7eeca09dd7c1d1a83eee13fb03930d8eb80a
```

Pins:

```text
vLLM       551719766029e78824a30d97ae6ac63917405b5f
SparkInfer be0edcaae6f5d284bb29a82325aba7a0ead6960f
FlashInfer 801d57a08958c13d375ddbb6be3be4808f48a708
```

Do not apply the safe-query precision candidate. A causal PEDANTIC rewind
reproduced the identical cold 100k failure, so that operator-level numerical
difference does not control the model symptom.

Do not apply PR #171. Exact ancestry and source inspection show that `5517197`
does not contain `dc770590`; its `auto` mode retains the pre-#171 behavior.
The independent fa71 no-#171 causal boot still missed cold 100k, proving that
this route change does not control the observed onset.

### CN4 launcher correction

The image's pinned launcher commit `48c8add` contains:

```bash
export NCCL_P2P_LEVEL=SYS
```

That unconditionally overwrites an explicit Compose value. The controlled CN4
fabric matrix measured `PXB` 10.1–11.3x faster for packed-CKV gather than
`SYS`, so preserving an explicit deployment value is still the correct
launcher behavior. However, PXB has not passed the long-context quality gate
and is now under direct causal test. Do not select it for production merely
from the throughput result.

The clean upstream launcher fix is:

```text
repo:   local-inference-lab/blackwell-llm-docker
branch: fix/launcher-preserve-nccl-p2p-level
commit: 9590e93
```

It preserves `SYS` as the default and honors an explicit deployment value.
Build the zero-compile derived layer from
`workspace/blackwell-v20-p2p-launcher/Dockerfile.v20-5517197-pxb-launcher`
(`sha256:aa0489dae6901a7ebca1099e05e4dacb7729952316998e78e86ab47185e67227`).
The Dockerfile is pinned `FROM` the exact published manifest and replaces
only `/usr/local/bin/serve-glm52-v16.sh`. Record the resulting image manifest
before launch. If Festr publishes a replacement image containing the same
launcher fix first, use that pinned image instead.

## Runtime configuration

Use the release entrypoint and its measured policy:

```yaml
image: voipmonitor/vllm@sha256:e7a8a8549c10b5d16899e0fb45ff7eeca09dd7c1d1a83eee13fb03930d8eb80a
entrypoint: ["/usr/local/bin/serve-gilded-gnosis.sh"]
restart: "no"
environment:
  MODEL_FAMILY: glm52-hybrid
  SERVED_MODEL_NAME: GLM-5.2
  TP: "4"
  DCP: "4"
  MTP: "3"
  DCP_PREFILL_WORKSPACE: auto
  MAX_MODEL_LEN: "480000"
  MAX_NUM_SEQS: "16"
  GRAPH: "64"
  KV_FP8_ROPE: "1"
```

Do not override `MAX_BATCHED_TOKENS`; the NF3 preset selects `2048`. Retain
CN4's proven `NCCL_P2P_LEVEL=PXB`, cache mounts, port, and other host-only
settings. Do not copy the old low-level attention, CKV-gather, query-split, or
MoE routing variables. The release preset and measured DCP auto-policy own
those choices.

`KV_FP8_ROPE=1` is mandatory and must be verified in the boot log. The
published hybrid launcher defaults to `nvfp4_ds_mla` but does not set this
variable; vLLM defaults it to `0`. Omitting the line tests a 432-byte
BF16-RoPE record, not the requested 368-byte FP8-RoPE configuration.

## Gate 0 — before the only boot

Run the fail-closed image dry-run gate:

```bash
harness/v20_5517197_nf3_gate0.sh <derived-image> [SYS|PXB]
```

Pinned script SHA-256:
`79094a5ba19662aa4b2ec7150c08d70b16915837a453cf7e5b1ae83b81677598`.

The route argument defaults to `SYS`. Use `PXB` only after the causal transport
control passes. The script deliberately verifies the effective dry-run value
selected by that argument.

Fail closed unless:

1. The derived image records the exact base manifest and all three inherited
   integration pins.
2. There are no source mounts or Python/CUDA overlays.
3. Installed
   `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` is
   `a2002892614587a737475ef58834b9445a65de764bcbcd646c586a9162a2f2bf`.
4. `_resolve_spec_decode_mode` is absent and the pre-#171 auto route is
   present:

   ```python
   self.spec_extend_as_decode = spec_decode_mode not in disabled_modes
   ```

5. `/usr/local/bin/serve-glm52-v16.sh` is
   `fee02f8cd61a4c7edfc9d2b31b62f35ea18424ecde2968064eb212bd441fd883`,
   contains `NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"`, and its dry-run output
   reports `NCCL_P2P_LEVEL=PXB`.
6. The rendered container environment contains exactly
   `KV_FP8_ROPE=1`, `MAX_MODEL_LEN=480000`, `TP=4`, `DCP=4`, and `MTP=3`.
7. Restart is disabled and the previous model's orphaned
   `vllm_offload_*.mmap` files have been accounted for before launch.

## Gate 1 — boot and identity

Require:

- stable container/process identity and health;
- all profiling and production CUDA-graph capture sizes complete;
- zero illegal access, OOM, Xid, assertion, engine-dead, or worker-death
  signatures;
- log evidence:

  ```text
  B12X GLM MLA KV format: KV_FP8_ROPE=1 kv_gmem_stride=368 kv_cache_dtype=nvfp4_ds_mla
  ```

- the launched process reports `NCCL_P2P_LEVEL=PXB`;
- max batched tokens 2,048 from the NF3 preset;
- max model length 480,000;
- KV pool at least 500,000 tokens;
- MTP3 active with nonzero draft acceptance on a short smoke request.

The published 934,912-token pool is an MTP0 result. Record the MTP3 pool from
this process; do not carry the MTP0 number into acceptance.

## Gate 2 — quality before benchmarks

Use `harness/v20_needle_duplication_onset_probe.py` with natural cold heads.
For every row, preserve the raw response and record:

- `prompt_sha256`;
- `cached_tokens`;
- `finish_reason`;
- completion tokens;
- `content`, `reasoning`, `reasoning_content`, and serialized message;
- MTP draft/accepted deltas.

Sequence:

1. 100k x3 and 150k x3.
2. Continue only if all six are exact.
3. 50k, 250k, 350k, and 475k x3.

Every row must have:

- `cached_tokens=0`;
- exact ticket `738216`;
- `finish_reason=stop`;
- non-empty finalized `content`;
- no digit or adjacent-word duplication.

Any missing row, error, empty content, `MISS`, `DUP`, `REASON_OK`, or
truncation stops the run. Do not tune or change routes in the same boot.

## Gate 3 — capacity and performance

Only after Gate 2 passes:

1. Record the clean MTP3 KV pool.
2. Run cold 8k and 55k prefill with cache-delta evidence.
3. Run decode C1/C4/C8/C16 and the established long-context decode cell.
4. Compare MTP3 only with MTP3. Do not compare the published 57.3 tok/s MTP0
   control with the older approximately 104 tok/s MTP3 result.
5. Require decode within the established 10% band and prefill above the
   existing cold floors.

NVMe and stress qualification can then continue on this same process. No
restart or rebuild is needed.

## PR disposition

- PR #171: hold/withdraw pending its causal discriminator. Do not forward-port
  it to `5517197`.
- Safe-query precise reduction: retain as a separate numeric-correctness
  draft; it is not the long-context retrieval fix.
- The four new `5517197` DCP topology commits are upstream base behavior and
  need no local overlay.

The no-#171 fa71 discriminator failed cold 100k, so the prior conditional
promotion path is closed. Keep `#171` out of `5517197`, record it as noncausal
for this regression, and resume this qualification only after the transport
control identifies a quality-safe routing posture.
