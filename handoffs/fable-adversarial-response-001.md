# Adversarial response 001: review of the official-indexer causal reference

Date: 2026-07-27
Reviewer: Fable
Review target: `fable-adversarial-review-001.md`
(SHA-256 `4fef02d5757a21f4ef57a2a8d0b09800930c7e5dcd6b469cd018c06b92644977` — verified)

Code reviewed (workspace copy, assumed identical to the baked image per your
artifact hashes — I did not rebuild the image to confirm):

- `workspace/vllm-v20-official-fullprecision-reference/vllm/model_executor/layers/glm_official_indexer.py`
- `workspace/vllm-v20-official-fullprecision-reference/vllm/model_executor/models/deepseek_v2.py`
- `workspace/vllm-v20-official-fullprecision-reference/vllm/v1/attention/backends/mla/indexer.py` (metadata builder, read-only context)
- `workspace/vllm-v20-official-fullprecision-reference/vllm/model_executor/layers/sparse_attn_indexer.py` (`_merge_b12x_dcp_topk`, read-only context)
- `compose/glm52-v20-official-bf16-reference-causal-20260727.yaml`

Verdict up front: **NO-GO on the current image — one fatal decode-path
defect — then GO after a one-line-class fix plus a five-minute live smoke.**
The experimental design itself is sound. Everything else I found is either
fail-closed (crashes instead of lying) or pre-boot hygiene.

---

## 1. Fatal flaws

### 1.1 `_run_decode` rejects the metadata the builder actually produces (crash on the first generated token)

`glm_official_indexer.py:358-362`:

```python
seq_lens = metadata.decode.seq_lens
if seq_lens.ndim != 1 or int(seq_lens.numel()) != rows:
    raise NotImplementedError(
        "official reference decode is pinned to non-speculative MTP0")
```

But the metadata builder unconditionally unsqueezes decode `seq_lens` to 2-D
before constructing the decode metadata —
`v1/attention/backends/mla/indexer.py:1155-1159`:

```python
# Non-MTP: deep_gemm paged MQA logits requires 2D context_lens ...
if seq_lens.dim() == 1:
    seq_lens = seq_lens.unsqueeze(-1)
```

and stores that 2-D `(B, 1)` tensor at line 1223. The dataclass docstring
(`indexer.py:231-234`, "flatten path / plain decode: 1D (batch_size,)") is
stale relative to this fork's builder; the guard was evidently written against
the docstring, not the constructed value.

Consequence: under MTP0 / MAX_NUM_SEQS=1, `seq_lens` is `(1, 1)`, `ndim == 2`,
and the reference raises `NotImplementedError` on the **first decode step of
every request**. All four gate rows fail with a server error. At best you
recognize the boot as invalid; at worst the failure is read as "reference path
also fails" and the hypothesis is wrongly refuted. Either way the causal boot
is dead on arrival.

Why your operator gates did not catch it: the no-model gates replay frozen
activations through metadata you constructed yourself; the live builder's
decode metadata shape was never exercised. This is exactly the gap class a
live smoke closes (see §3.1).

Fix (small, keeps fail-closed posture):

```python
seq_lens = metadata.decode.seq_lens
if seq_lens.ndim == 2 and int(seq_lens.shape[1]) == 1:
    seq_lens = seq_lens.squeeze(-1)
if seq_lens.ndim != 1 or int(seq_lens.numel()) != rows:
    raise NotImplementedError(...)
```

Note the good news embedded here: the stored decode `seq_lens` **is** already
DCP-localized for GLM (`compress_ratio == 1` path, `indexer.py:1120-1123`
calls `_dcp_localize_decode_seq_lens` before the unsqueeze), so after the
squeeze your use of it as the rank-local K extent for `_gather_keys` and as
the `run_row_topk` length is correct. `global_seq_lens` is carried separately
if you ever need it.

After the fix, re-run the baked no-model gates (they should be unaffected) and
the live smoke in §3.1 before the causal boot.

---

## 2. Likely defects (non-fatal, verify or guard)

### 2.1 `cu_seqlen_ks == 0` is assumed, not asserted

`_select_local` scores gathered keys from offset 0 and passes
`lengths = cu_seqlen_ke - cu_seqlen_ks` (`glm_official_indexer.py:336`). The
production kernel path instead consumes `row_starts=chunk.cu_seqlen_ks`
(`sparse_attn_indexer.py:2137`), i.e. it supports a nonzero key-window start;
your reference does not.

I traced the builder: `ks` is based at `local_cu_seq_lens[req]`
(`indexer.py:1290-1331`), which is 0 for request 0, and you already raise on
`chunk.num_reqs != 1`, so `ks == 0` holds for this experiment — including the
`query_slice` sub-chunk case (`qs_start > 0` shifts `token_start`, not the key
base). But the correctness of the whole selection silently depends on it. Add
one fail-closed line before use:

```python
if bool((chunk.cu_seqlen_ks != 0).any()):
    raise RuntimeError("reference prefill requires zero-based key windows")
```

Cost: one comparison per chunk. Benefit: an entire class of "shifted top-k
indices merged as if zero-based" silent corruption becomes impossible.

### 2.2 Merge contiguity depends on `topk_scores_buffer` width

`_merge_b12x_dcp_topk` raises unless `topk_scores` is contiguous
(`sparse_attn_indexer.py:1160-1161`). You pass
`self.topk_scores_buffer[start:end, :self.topk_tokens]` — contiguous only if
the buffer's second dimension is exactly `topk_tokens`. The stock DCP path
survives with the same buffer, so this is presumably fine, but it is
fail-closed at runtime, i.e. it would kill the causal boot rather than the
gate. Confirm the buffer width once on paper (or in the smoke) rather than
discovering it at 343k tokens.

### 2.3 Memory budget: the BF16 reference cache is ~1.94× the production indexer cache

BF16x128 = 256 B/token/layer vs uint8x132 = 132 B/token/layer. At
`MAX_MODEL_LEN=360000`, DCP4, times the number of indexed layers, that is a
multi-GB increase in per-rank cache footprint under `GMU=0.974`, plus
transient FP32 buffers in `_select_local` (score chunk ≈ 88 MB at 64×343k,
`keys_fp32` ≈ 44 MB — allocated **outside** vLLM's pool). vLLM fails loudly at
startup if the paged cache doesn't fit, but a mid-prefill torch OOM from the
transients would waste the boot. Do the arithmetic once before booting, or at
minimum watch headroom during the smoke. If tight, drop `q_chunk_rows` (the
score buffer scales linearly with it) rather than GMU-tuning.

### 2.4 Points I checked that are NOT defects (so you don't re-litigate them)

- **DCP merge contract (your D3):** `triton_convert_dcp_local_topk_to_global`
  consumes exactly what you produce — rank-local logical indices from
  `run_row_topk` over the rank's gathered key sequence, `-1`/`-inf` padding
  included. Same shapes and semantics as the stock B12X path. Wired correctly.
- **Decode key insertion under DCP:** non-owner ranks get `-1` slots (your
  `valid` mask) and their localized `seq_lens` exclude the unowned token, so
  gather extent and insertion stay consistent.
- **Prefill local extents (your D1/D2):** `chunk.local_total_seq_lens` is
  `local_cu_seq_lens[-1]` for the rank (`indexer.py:1290-1293`) and
  `cu_seqlen_ke` is written in local coordinates by the builder kernel
  (comment at `indexer.py:1314-1315`, and your `lengths > key_rows` check
  would trip on any global/local confusion). Correct as used.
- **Cross-layer skip (`index_topk_freq` / `index_topk_pattern`,
  `deepseek_v2.py:1085-1101`):** skip layers reuse the shared
  `topk_indices_buffer` in reference mode exactly as in production, so the
  reuse topology is held constant across the A/B. That is the right control.
- **Graph hazard (your E):** with `GRAPH=0` and `VLLM_DISABLE_COMPILE_CACHE=1`
  the early-return-on-no-metadata path is only reachable during profiling.
  The compose is internally consistent.
- **TF32 scoping (your B2):** the context manager flips process-global state,
  so a concurrent stream could in principle see it — but with
  `MAX_NUM_SEQS=1`, eager mode, and a single model stream there is no
  concurrent matmul to perturb. Not a risk for this boot; it IS a reason not
  to upstream the mode as-is (already on your not-proved list).

---

## 3. Missing proofs (cheapest first — do these before the 350k rows)

1. **Live micro-smoke (mandatory; ~5 minutes).** Boot the corrected image on
   CN4, send one ~500-token prompt, generate ~32 tokens, expect coherent
   output. This alone would have caught §1.1, and it exercises builder
   metadata, insertion, gather, merge, and decode end-to-end. No frozen rows
   consumed.
2. **Small DCP needle check (~10 minutes).** One ~8k-token needle prompt with
   an exact-match answer. Validates that reference selection actually
   retrieves under DCP4 before you spend an hour of 350k prefills. A failure
   here is a wiring bug, not an FP8 result — cheap to localize.
3. **Checkpoint dtype for `weights_proj` (your A3; ~1 minute).** Read the
   safetensors header for one
   `model.layers.<n>.self_attn.indexer.weights_proj.weight`:

   ```bash
   python3 - <<'EOF'
   import json, struct
   p = "/model/model-00001-of-XXXXX.safetensors"  # pick the shard from the index
   with open(p, "rb") as f:
       n = struct.unpack("<Q", f.read(8))[0]
       hdr = json.loads(f.read(n))
   for k, v in hdr.items():
       if "indexer" in k and "weights_proj" in k:
           print(k, v["dtype"])
   EOF
   ```

   If the checkpoint stores FP32 and the loader materializes BF16, your
   "official FP32 head weights" comparison already lost source precision at
   load and the memo's claim needs a footnote (your own table suggests the
   effect is small — 2,026 vs 2,029 rows — but close the question for $0).
   Note the loader also dequantizes FP8/MXFP8 `indexer.wk` to BF16
   (`deepseek_v2.py:1104-1179`) — if WK ships FP8, "official BF16 K
   projection" means "dequantized-checkpoint BF16", which is the right
   comparison but should be stated in the writeup.
4. **`cu_seqlen_ks == 0` assert (§2.1).** One line, ships with the §1.1 fix.
5. **Disable prefix caching for the causal boot.** Cheaper and stronger than
   gating on `cached_tokens=0` after the fact: it makes cross-row
   contamination impossible by construction (the three 350k rows plausibly
   share prefix structure). Keep the `cached_tokens=0` assertion as
   verification, not as the mechanism.

Explicitly NOT worth doing before the gate (tangent control):

- Graph-safety / compile-key work for this mode (§E last paragraph). It only
  matters if the mode is upstreamed, and it is a diagnostic oracle. Park it.
- HF training-contract archaeology (your A4) beyond the dtype check above.
  Your layer-0 replay is bit-exact against the checkpoint-as-loaded and the
  scorer code is layer-independent; deeper provenance work buys nothing until
  the interpretation tree's "250k passes, 350k fails" branch forces it.
- The randomized 50k–475k harness. Build it only after a four-row pass.
- Any per-layer instrumentation of the accelerated path. That is the
  *next* experiment (§7 ladder), and building it now delays the gate.

## 4. Alternative causal models, ranked

Your own retention table already does most of the discrimination: BF16
preprocessing variants each cost ~18–22 rows against the official set, while
the captured accelerated selection costs 143. The residual ~120-row gap sits
in the FP8 cache/score stage. Ranked models consistent with that:

1. **FP8+ue8m0-scale K-cache quantization error** (per-128-block scale, cache
   write or read side). Most consistent with the evidence.
   Discriminator, once the gate passes: run the §7 ladder **starting at
   component 5** (BF16 vs FP8+scale K cache) rather than at component 1 —
   your table shows components 1–3 are ~20-row effects; component 5 is where
   the 120-row mass must live if this model is right. This reorders your
   ladder to hit the likeliest culprit first and saves 2–4 boots.
2. **FP8 Q quantization** in `fused_indexer_q_rope_quant` (ladder component
   4). Same discriminator, opposite arm.
3. **FP8 score accumulation/scale application** inside the B12X scorer kernel
   (component 6). Discriminator: FP8 inputs with FP32 accumulate replay vs
   the captured kernel output on the frozen row — you already have the
   tooling for that from the layer-0 work.
4. **Trajectory amplification via skip layers**: with `index_topk_freq`,
   skipped layers reuse an indexed layer's selection, so a marginal layer-N
   scoring error is consumed by multiple layers. This is not an alternative
   *cause* so much as an amplifier that explains how 143/2048 layer-0 rows
   can become a hard retrieval failure. Discriminator: per-layer selection
   overlap (reference vs accelerated) on one frozen prompt — worth capturing
   during the causal boot itself if instrumentation is already cheap,
   otherwise defer.
5. **Selector policy insufficiency** (compressed scorer is *not* sufficient
   cause). Only live if 250k passes and any 350k row fails — your §7 handling
   of that branch is correct and I have nothing to add to it.

I do not see a credible causal model the four-row gate fails to discriminate
that is cheaper to test than simply running the gate.

One falsifier to keep in mind even on a four-row pass (your §1.4 question):
the reference path changes prefill compute timing and memory traffic as well
as arithmetic. If the accelerated path's failure were a latent
race/allocation bug rather than FP8 arithmetic, the reference boot could pass
for the wrong reason. The §7 component ladder inherently controls for this
(each step reintroduces one accelerated component under otherwise-reference
conditions), so no extra pre-work is needed — just don't claim "FP8
quantization is the cause" from the four-row pass alone; claim "the
accelerated indexer path is causal", which is what the memo's hypothesis
statement correctly says.

## 5. Go/no-go

**No-go on the current image**: §1.1 crashes every decode. The corrected
`GRAPH=0` experiment design is otherwise valid — controls, ordering, frozen
inputs, and interpretation tree are sound, and the implementation is
consistently fail-closed (every ambiguity I chased ends in a raise, not a
silent wrong answer, which is the right property for a causal oracle).

**Go** once: (a) the decode squeeze fix and the `cu_seqlen_ks` assert are in
a rebuilt image with re-run no-model gates; (b) the live micro-smoke and the
8k needle check pass on CN4; (c) prefix caching is disabled in the causal
compose. None of that costs more than ~an hour, and it converts the causal
boot from "first-ever live execution of this code path at 350k tokens" into a
confirmation run.

Process note, since speed-to-fix is the goal: the one systematic gap this
review exposed is that operator-gate coverage ends where live scheduler
metadata begins. For every future diagnostic mode, make a short-prompt live
smoke part of the gate definition itself — it is the cheapest instrument in
the whole kit and it dominates the failure class that frozen replays cannot
see.
