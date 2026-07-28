# Phase A addendum: NVFP4 dynamic second-level scale — audit answers + implementation record

Date: 2026-07-27
Author: Fable
Parent spec: `design/nvfp4-dynamic-second-level-scale-spec.md`
Branches (both at the exact RC pins):

```text
workspace/b12x-nvfp4-dynamic-scale    branch nvfp4-dynamic-token-scale @ SparkInfer c3828fd
workspace/vllm-nvfp4-dynamic-scale    branch nvfp4-dynamic-token-scale @ vLLM 0c79e41
```

Status: implementation drafted on both branches (writer, readers, IO gathers,
vLLM wiring, tests). NOT compiled — this machine has no CUDA; the CN4 runbook
below is the required next step. PARKED per Derek until the static-calibration
promotion track completes.

## 1. Phase A questions — answered from source

### Q1. Record layout / pad space — ANSWERED: the scale fits in existing pad; ZERO growth

The 368-byte `nvfp4_ds_mla` KV_FP8_ROPE=1 record
(`sparkinfer/attention/_shared/mla/kv_cache.py:10-18`, constants `:70-86`):

```text
[   0, 256)  packed E2M1 NoPE (512 x 4-bit, 32 group-16 blocks)
[ 256, 288)  32 x E4M3 group scale bytes (group amax / 6.0, implicit outer 1.0)
[ 288, 292)  fp32 RoPE scale (rope amax / 448.0)   <- per-token dynamic ALREADY
[ 292, 304)  zero pad (12 bytes)                   <- our fp32 s_t goes at [292,296)
[ 304, 368)  64 x E4M3 RoPE
```

Decisive facts:
- **12 bytes of zero pad at [292,304)** — the 4-byte fp32 per-token scale
  fits with 8 bytes to spare. **The record stays 368 bytes. No wire, cache,
  page-table, or capacity change of any kind.** (Derek's constraint met
  exactly; the spec's "Option R with zero growth".)
- The RoPE lane already implements the identical per-token dynamic-scale
  pattern (amax → fp32 scale in the record → E4M3 payload), so the design
  extends an in-record precedent, not a novel mechanism.
- Readers ALREADY move the chosen bytes: prefill reads [288,304) as one
  `ld.global.v4.f32` and discards `.y/.z/.w` (`prefill_mg.py:489-497`);
  decode stages the whole [288,368) tail into smem (`decode_math.py:996-999`,
  scale at +0, our word at +4). **Zero additional memory traffic on the
  existing tail paths;** the gathers add one 4-byte-class scalar load per
  candidate (see Q2).
- Alignment: record base/strides are 8/16-byte aligned; 292 is 4-aligned but
  not 8-aligned, so scalar gathers load the 8-aligned pair at [288,296)
  (rope scale, latent scale) and keep the second word — the same
  `ld_global_nc_v2_u32` idiom as the DSV4 footer gather.

### Q2. Consumer inventory — ANSWERED: two dequant leaves; all movers are raw-byte

Complete inventory (agent-audited with file:line, both trees):
- **Dequantizers (the only code that interprets NoPE bytes):**
  decode `_nvfp4_pair_bfloat2` (decode_math.py, single decode dequant point,
  used by `s1_qk_nope_nvfp4_bf16` QK and `s6_xv_nope_nvfp4_bf16` P·V) and
  prefill-MG `_nvfp4_pair_bfloat2_mg` (used by `s1_qk_nope_nvfp4_bf16_mg2`).
  MG prefill's P·V **reuses the decode `s6_xv_nope`** (prefill_mg.py:3204,
  :3231), so the decode leaf covers it.
- **Dispatch:** both stage wrappers (`s1_qk_nope_block_scaled`,
  `s6_xv_nope`) already receive `kv_sc_base_addr` (the DSV4 footer buffer) —
  the per-token scale reuses that exact channel.
- **Raw-byte movers (carry the scale for free, no changes):** TMA/bulk smem
  staging (io.py/io_mg.py), DCP CKV gather = `ops.cp_gather_cache` verbatim
  copy + NCCL all-gather of raw uint8 (b12x_mla_sparse.py:2116-2193),
  prefetch machinery (record-size-driven), `swap_blocks`/`copy_blocks_mla`
  (byte-blind), offload tiers (byte-length-driven). The ONE transform point,
  `_append_current_chunk_to_gathered` (:2224), re-runs the writer — wired
  with the same flag.
- **Prefix-cache hashing:** token-content only (`kv_cache_utils.py:598-625`);
  cached-block reuse is pointer-level, no copy. Scales are content-derived
  (same tokens+prefix ⇒ same activations ⇒ same scales), so hashing is
  untouched (Q5 answered: **no**, scales do not join the hash — verified
  against `generate_block_hash_extra_keys`, which carries only LoRA/MM/salt
  keys).
- **Record-parsing tests:** b12x `tests/attention/test_mla_kv_cache.py`
  (offsets literal) and vLLM `tests/kernels/attention/test_cache.py` (432B
  stock writer only — unaffected). New mode gets its own test file (below).

### Q3. Scale dtype — DECIDED: fp32

fp32 costs zero extra bytes (pad is free), matches the rope-scale precedent
in the same record, and removes any need for a BF16 error analysis. BF16
would only matter under a side-table design, which is moot per Q1.

### Q4. Granularity — DECIDED: per-token; the fallback is unnecessary

The writer's 32 group threads are exactly warp 0; the token amax is a
5-shuffle butterfly reduction (`cute.arch.shuffle_sync_bfly`) — no shared
memory, no extra passes. There is no cost pressure that would justify
per-page granularity, so the fallback is retired.

### Q5. Prefix-cache key — ANSWERED above: no change (verified, not assumed).

### Q6. MXFP8 sharing — ANSWERED: not shared; out of scope

MXFP8 in this stack is weight/GEMM quantization only; there is no MXFP8 KV
writer and `b12x/attention/mla/` has no mxfp8 references. The only MLA KV
writer is `concat_and_cache_nvfp4_mla_fp8_rope`. The 432-byte KV_FP8_ROPE=0
variant (stock CUDA writer in vLLM csrc, 16-byte pad at [288,304)) could take
the same field later; not needed for the GLM posture and not implemented.

### Q7. Mainline layout compatibility — no upstream NVFP4 MLA KV record exists
to be compatible with; the fork IS the reference implementation. Revisit at
upstreaming time.

## 2. What is implemented on the branches

### b12x (`nvfp4-dynamic-token-scale` @ c3828fd)

1. **Writer** (`kv_cache.py`): `per_token_scale` mode on
   `ConcatAndCacheNvfp4MlaFp8RopeKernel` — warp-0 butterfly max over the 32
   group amaxes; `s_t = token_amax * f32(1/2688)` (exact-constant contract,
   mirrored by tests) stored fp32 at [292,296) by lane 0; every group scale
   encoded relative to `s_t` (`group_amax * rcp(s_t) * rcp(6)` → satfinite
   E4M3, values packed with `rcp(decoded) * rcp(s_t)`); pad zeroing narrowed
   to [296,304); `s_t == 0` writes a fully zero NoPE lane. Threaded through
   the lru builder, flat launch, torch custom op (+fake), and the public
   wrapper (kwarg defaults False everywhere). **Compile-spec version 2 → 3.**
2. **Decode readers** (`decode_math.py`): `_nvfp4_pair_bfloat2` /
   `_nvfp4_scalar_bf16_u16` take `latent_scale_per_token` +
   `kv_sc_base_addr`; per-entry `ld_shared_f32(kv_sc + entry*4)` replaces the
   launch scalar. `s1_qk_nope_nvfp4_bf16` hoists its lane's row scale once;
   `s6_xv_nope_nvfp4_bf16` passes through per entry. Both dispatch wrappers
   forward the flag (defaults keep every existing call site byte-identical).
3. **Prefill-MG readers** (`prefill_mg.py`): same pattern on
   `_nvfp4_pair_bfloat2_mg` + `s1_qk_nope_nvfp4_bf16_mg2` (row scale hoisted).
4. **IO gathers**: decode `io_issue_gather` (io.py) — in the NVFP4/fp8-rope
   arm, scalar-gathers the second word of the 8-aligned [288,296) pair into
   `kv_sc + entry*4` during the existing validity pass (before the
   load-bearing CTA fence, same ordering as the DSV4 footer). MG
   `io_issue_gather_glm_mg` (io_mg.py) — new scale pass + CTA fence before
   the leader's arrive/expect_tx, mirroring `io_issue_gather_dsv4_nope`.
5. **Threading** (kernel.py / api.py / prefill entries / compile keys /
   fail-closed validation): delegated to a subagent with an explicit spec —
   see its report + `git diff` review before commit. Includes: kv_sc smem
   allocation for the NVFP4 arms (BI×4, following the DSV4 idiom) if not
   already present, cache-key + version bumps on decode/prefill specs, and
   entry-point validation (mode ⇒ scale_format 2 + 368-byte records).
6. **Tests** (`tests/attention/test_mla_kv_cache_per_token_scale.py`):
   record ABI (bit-exact `s_t`, pad, mode-independent RoPE lane), the
   positioning invariant (max group scale ≥ 256 for every token; static
   writer shown subnormal at shallow magnitude — the #145 defect reproduced
   in a unit test), accuracy dominance at shallow magnitudes with a no-loss
   bound at deep ones, and the zero-token edge.

### vLLM (`nvfp4-dynamic-token-scale` @ 0c79e41)

- `VLLM_NVFP4_MLA_DYNAMIC_SCALE=1` gate (`b12x_mla_sparse.py`): requires the
  368-byte KV_FP8_ROPE=1 record (RuntimeError otherwise); requires a
  SparkInfer build whose writer supports `per_token_scale` and whose
  decode/extend forwards accept `latent_scale_per_token` (inspect-based,
  same fail-closed idiom as the existing NVFP4 API check).
- Writer calls (`do_kv_cache_update` and the DCP gather re-quantization)
  pass `per_token_scale=True` in mode, and omit the kwarg entirely when off
  so pre-two-level SparkInfer builds keep working.
- `_b12x_kernel_format_kwargs` adds `latent_scale_per_token: True` and pins
  `latent_scale` to 1.0, raising if a per-layer outer scale is also set.
- `mla.py`: `VLLM_NVFP4_MLA_SCALES_FILE` + dynamic mode ⇒ ValueError
  (mutually exclusive by design); the host-side divide stays identity in
  dynamic mode.

## 3. Design invariants (for review and for the upstream PR text)

1. Mode OFF ⇒ every kernel's PTX and every record byte is unchanged (all new
   params default off; compile keys extended but versions bumped so stale
   cubins can never run).
2. Mode ON ⇒ record width, offsets, RoPE lane, transports, hashing, capacity
   all unchanged; only [292,296) gains meaning and NoPE group scales change
   value (not format).
3. Legacy records are unreadable in mode ON by construction (zero scale ⇒
   zero output); the mode is server-static, joins compile identity, and must
   boot on a fresh cache — enforced by the same posture rules as
   KV_FP8_ROPE.
4. Positioning failures (E4M3-subnormal group scales) are impossible in mode
   ON: the largest group scale per token encodes ≈448 by construction. The
   saturation counters become a test-fixture invariant instead of a runtime
   worry.

## 4. CN4 validation runbook (Phase B gate — nothing here ran yet)

Order matters; stop at the first failure.

1. `git diff` review of the threading agent's kernel.py/api.py changes
   (Sol + Fable) — the one layer written by an agent, not by hand.
2. Build/import gate on CN4: `uv` venv per repo AGENTS.md; compile-touch the
   writer + decode + prefill kernels (fresh SparkInfer kernel cache;
   versions bumped so stale cubins are structurally excluded).
3. Unit: `pytest tests/attention/test_mla_kv_cache_per_token_scale.py -v`
   (new), then the full existing suites with the mode OFF —
   `test_mla_kv_cache.py`, `test_attention_mla_nvfp4.py`,
   `test_attention_mla_kv_cache.py` — proving byte-identical off-mode
   behavior.
4. Reader e2e (mode ON): extend/parametrize
   `test_writer_records_feed_production_head_multisplit_decode` and the MG
   prefill twin with `latent_scale_per_token=True` — writer records feed the
   production heads and match the host reference with `s_t` applied.
5. vLLM smoke on CN4 (one GPU, short ctx): boot with
   `VLLM_NVFP4_MLA_DYNAMIC_SCALE=1`, `KV_FP8_ROPE=1`, fresh caches; the §9
   micro-smoke + 8k needle from the calibration track, then error-injection
   checks (scales file + dynamic ⇒ refuses; KV_FP8_ROPE=0 + dynamic ⇒
   refuses).
6. Quality gates: frozen four-row suite vs the static-calibrated baseline;
   ticket-entry-layer margins equal or better; randomized ladder; KLD vs
   static-calibrated NVFP4 and fp8_ds_mla.
7. Perf gates (pre-committed in the spec): decode ≤1%, prefill ≤2% vs the
   static baseline, capacity unchanged (same record bytes — assert KV-pool
   token count identical).

## 5. Honest limitations of this draft

- **Nothing has compiled.** CuTeDSL edits were written against the pinned
  source with line-level verification, but this host has no CUDA; step 2 of
  the runbook is the first real gate. Expect mechanical fixes (imports,
  DSL idiosyncrasies), not design changes.
- The kernel.py/api.py threading layer was implemented by a directed
  subagent; it is the least-verified layer and is called out as review item
  #1 on purpose.
- The MG kv_sc smem allocation for the NVFP4 arm depends on what the smem
  plan already reserves; the agent was instructed to follow the DSV4 idiom
  and report exactly what it found — verify in its report + diff.
- Decode P·V adds one 4-byte smem broadcast load per dequant pair in mode
  ON (the QK path hoists per row). If the ≤1% decode budget is threatened,
  the documented mitigation is hoisting the four per-kstep entry scales in
  `s6_xv_nope_nvfp4_bf16` — a contained follow-up, not a redesign.
