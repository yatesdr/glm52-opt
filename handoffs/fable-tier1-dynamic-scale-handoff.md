# Handoff to Sol: two-level NVFP4 record mode (Tier-1 dynamic scale) — ready for CN4 test

Date: 2026-07-27
From: Fable. Status: draft-complete, audited, **never compiled or executed**
(authored on the Mac, no CUDA). Your test run is the first real gate.

## What this is, in three sentences

The needle-regression root cause was the NVFP4 MLA KV writer quantizing at a
fixed outer scale of 1.0, which parks shallow-layer group scales in E4M3
subnormals; #145 fixes it with a static per-layer calibration file. This mode
removes the calibration requirement entirely: the writer derives a per-token
second-level scale (`s_t = token_amax/(6·448)`) and stores it **inside the
record, in 4 of the 12 existing pad bytes at [292,296) — the record stays
368 bytes**, and the readers use it in place of the per-layer launch scalar.
It is the general fix for every future NVFP4 model/quant: no scales file,
no staleness, subnormal positioning impossible by construction.

Design + all Phase A audit evidence:
`design/nvfp4-dynamic-second-level-scale-spec.md` and
`design/nvfp4-dynamic-second-level-scale-phaseA-addendum.md` (the addendum
has the full file:line trail and the 7-gate runbook this handoff condenses).

## Pins

```text
b12x:  workspace/b12x-nvfp4-dynamic-scale
       branch nvfp4-dynamic-token-scale, commit 0d9aead9
       base = SparkInfer c3828fd (the runtime-stride RC pin, exactly)
vllm:  workspace/vllm-nvfp4-dynamic-scale
       branch nvfp4-dynamic-token-scale, commit 91dff5a9
       base = vLLM 0c79e41 (identical to the RC / your reference base)
build base image (for the overlay Dockerfile, same pattern as your
reference images):
       voipmonitor/vllm@sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0
```

Enable: `VLLM_NVFP4_MLA_DYNAMIC_SCALE=1` + `KV_FP8_ROPE=1` +
`kv_cache_dtype nvfp4_ds_mla`, fresh caches. Fail-closed everywhere else:
432-byte record, non-NVFP4 format, scales-file-also-set, or an old
SparkInfer build all raise at boot, never run wrong.

## Record/mode contract (what changed on the wire: nothing but meaning)

```text
[   0,256) E2M1 NoPE            unchanged format; values now relative to s_t
[ 256,288) 32x E4M3 group scales unchanged format; largest ≈448 by construction
[ 288,292) fp32 rope scale      untouched
[ 292,296) fp32 s_t             NEW meaning (was zero pad)
[ 296,304) zero pad             still zero
[ 304,368) E4M3 RoPE            untouched (bit-identical across modes)
```

Reader math is today's expression with `latent_scale := s_t` per record.
Decode already stages [288,368) (s_t sits at tail+4); prefill already loads
[288,304) as one v4.f32 and discarded `.y` — it now IS `.y`. The gathers add
one 4-byte-class scalar load per candidate into a BI×4 kv_sc smem buffer
(the DSV4 footer-gather idiom, including the fence-before-arrive ordering).

## What was implemented (by layer, with authorship — matters for review)

Hand-written by me (reviewed at source line level against the c3828fd pins):
- writer mode in `kv_cache.py` (warp-bfly amax reduce; compile spec 2→3)
- dequant leaves + dispatch wrappers in `decode_math.py`, `prefill_mg.py`
- both IO scale gathers (`io.py`, `io_mg.py`)
- vLLM gate/wiring (`mla.py`, `b12x_mla_sparse.py`)
- tests: `tests/attention/test_mla_kv_cache_per_token_scale.py`

**Agent-threaded (REVIEW THIS LAYER FIRST — it's the least-verified):**
traits field + central fail-closed checks (`traits.py`), decode/MG kv_sc
smem allocations (`smem.py:247`, `smem_mg.py:203` — BI×4/buffer, new for the
NVFP4 arms), entry-point kwargs + validation (`api.py`, `prefill.py`,
`kernel.py`, `prefill_mg.py`), compile-key additions + version bumps
(decode 18→19, MG prefill 4→5, latent_scale identity forced on in-mode).
I diff-reviewed it (buffer parity gather↔consumer verified via the shared
`kv_sc_addr + buf * kv_sc_buf` expression; smem struct gates correct) and
all 11 files pass py_compile — but nobody has compiled the DSL.

## Off-mode safety claim (please falsify)

Every new parameter defaults off; kernels read the flag from a traits field
defaulting False, so existing specializations should trace byte-identically.
The ONLY intended off-mode effect: the three compile-spec version bumps
invalidate cached cubins once (deliberate, per the #85 lesson). The off-mode
test suites in gate 3 below are the proof.

## Test order (stop at first failure; all on CN4)

1. **Review** the agent-threaded layer (`git diff HEAD~1` on the b12x branch,
   files above). ~30 min of your eyes before any compute.
2. **Compile/import gate**: fresh SparkInfer kernel cache; touch-compile
   writer + decode + MG prefill in both modes. Expect mechanical DSL fixes
   here, not design changes — the design invariants are in the addendum §3
   if something looks off.
3. **Off-mode regression**: existing suites unchanged —
   `tests/attention/test_mla_kv_cache.py`, `test_attention_mla_nvfp4.py`,
   `test_attention_mla_kv_cache.py`, plus your standard decode/prefill
   corpus picks. This is the "does not break other features" gate.
4. **New unit tests**: `pytest tests/attention/test_mla_kv_cache_per_token_scale.py -v`
   — record ABI (bit-exact s_t vs the exact-constant contract), positioning
   invariant (max group scale ≥256 every token; the static writer's
   shallow-magnitude subnormal defect is reproduced as a test and must
   fail-the-old-way), accuracy dominance at 0.02-magnitude tokens, zero edge.
5. **Reader e2e**: parametrize your existing
   `test_writer_records_feed_production_head_multisplit_decode` (+ MG twin)
   with `latent_scale_per_token=True` — production heads over mode-on
   records vs host reference with s_t applied.
6. **Overlay image** from the RC digest above (your usual derived-image
   pattern; the b12x branch package files + the two vLLM files), pinned
   SHAs in the build record. Boot smoke with the mode env + fresh cache +
   the fail-closed negative checks (scales file + dynamic ⇒ refuse;
   KV_FP8_ROPE=0 + dynamic ⇒ refuse).
7. **Quality**: frozen four-row gate vs your static-#145 baseline; entry-
   layer margins equal or better; then ladder/KLD/perf per the addendum §4
   (decode ≤1%, prefill ≤2%, KV pool token count IDENTICAL — same record).

## Known watch-items (from my own audit)

- Decode P·V does one extra 4B smem broadcast load per dequant pair in-mode
  (QK hoists per row). If gate 7's decode budget is threatened, the
  documented mitigation is hoisting the four per-kstep entry scales in
  `s6_xv_nope_nvfp4_bf16` — contained, no redesign.
- The writer's group quant uses `rcp(decoded)*rcp(s_t)` (two approx rcps) vs
  #145's host-divide semantics; the unit tests bound the effect. If bit-parity
  with a torch reference matters for your harness, mirror the kernel's exact
  constant (`_TWO_LEVEL_RCP`) and treat rcp.approx as ≤2 ulp.
- `latent_scale_identity` fold: forced ON in-mode (launch scalar is dead);
  the agent noted the actual fold lives in compile machinery outside the
  diff — worth one look during gate 1.
- Legacy records are unreadable in-mode by construction (zero scale ⇒ zero
  output): fresh cache is mandatory, same posture rule as KV_FP8_ROPE.

Relative priority per Derek: this test is now prioritized. The static-#145
calibration track remains the fallback ship if this stalls at any gate.
