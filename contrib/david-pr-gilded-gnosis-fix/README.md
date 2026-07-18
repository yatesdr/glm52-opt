# Stopgap for running fp8-rope on the *deployed* gilded-gnosis-v18 image — image/branch version skew

**For:** `lukealonso/b12x#37` (compact NVFP4 FP8-RoPE writer) +
`local-inference-lab/vllm#129` (use the packaged writer).

> **RESOLVED — not a PR defect (David Young, 2026-07-18).** David's merged
> `dev/gilded-gnosis` branch **already binds both writer call sites** to the
> packaged writer, so **no patch is needed there** — verified below. This
> directory is only a **stopgap for the specific already-published image**
> (`…-20260718`), which was built *before* that merge and therefore still calls
> the old torch op at the second site. Rebuild the image from `dev/gilded-gnosis`
> and this patch is unnecessary. **Do not reboot or repatch a working machine on
> account of this.**

**TL;DR:** Both PRs are correct and their writer is byte-identical to the one we
independently ported (same layout scale@288/pad@292/RoPE@304, same recipe). The
issue is purely a **version skew**: the *deployed 20260718 image* predates the
`dev/gilded-gnosis` merge, so on that image one of the two writer call sites —
the DCP CKV-gather prefill path — still calls `torch.ops._C_fp8_rope_ops`, which
`#129`'s loader no longer registers, and it crashes at first prefill under
`KV_FP8_ROPE=1` + `DCP_CKV_GATHER=1`. The one-line binding below is what let us
run `KV_FP8_ROPE=1` on that pre-merge image for the comparison numbers.

## Verified against `dev/gilded-gnosis` — no patch needed there

Checked `local-inference-lab/vllm@dev/gilded-gnosis`
`vllm/v1/attention/backends/mla/b12x_mla_sparse.py` directly:

- `do_kv_cache_update` → `self._concat_and_cache_nvfp4_mla_fp8_rope(...)` ✅
- `_append_current_chunk_to_gathered` (the CKV-gather site) →
  `self._concat_and_cache_nvfp4_mla_fp8_rope(...)` ✅ **already bound**
- no `_FP8_ROPE_WRITER_LOADED` / `_load_fp8_rope_writer` ✅

So both sites already use the packaged writer on the merged branch. The mismatch
is only with the **pre-merge published image**, whose
`b12x_mla_sparse.py` (md5 `14c14eabc937cddf481532fb19e1dcb5`,
`voipmonitor/vllm:gilded-gnosis-v18-…-20260718`) still has:

```python
        k_scale = getattr(layer, "_k_scale", None)
        if self._kv_fp8_rope:
            torch.ops._C_fp8_rope_ops.concat_and_cache_nvfp4_mla_fp8_rope(   # pre-merge image only
                kv_c, k_pe_flat, gathered_buffer, slots, k_scale,
            )
```

**Action item is a rebuild, not a code change:** publish a gilded-gnosis image
from `dev/gilded-gnosis` and the stopgap retires itself.

## The fix (`ckv-gather-callsite.patch`)

One-line binding change, identical in spirit to #129's `do_kv_cache_update`
change — call the packaged API you already bound in `__init__`:

```diff
-            torch.ops._C_fp8_rope_ops.concat_and_cache_nvfp4_mla_fp8_rope(
+            self._concat_and_cache_nvfp4_mla_fp8_rope(
                 kv_c, k_pe_flat, gathered_buffer, slots, k_scale,
             )
```

The patch also drops the now-dead `_FP8_ROPE_WRITER_LOADED` module global (its
removal hunk in #129 didn't apply on the image due to surrounding-context drift).

## Validation on CN3 (4× RTX PRO 6000 Blackwell, SM120, PCIe Gen3)

- David's `b12x#37` `kv_cache.py` **compiles and passes the writer smoke** in the
  gilded-gnosis image against `b12x bc85ef3` (layout/canary: pad `[292,304)` zero,
  zero-token→zero record, negative-slot skip, allocation canary preserved).
- With this second-call-site fix applied, the full **480k @ 368B** boot
  (`KV_FP8_ROPE=1`, `VLLM_DISABLE_COMPILE_CACHE=1`, `DCP_CKV_GATHER=1`) is under
  test; results table appended below when the battery completes.

### Results (480k @ 368B, KV_FP8_ROPE=1, DCP_CKV_GATHER=1, VLLM_DISABLE_COMPILE_CACHE=1)

With `b12x#37` + `vllm#129` + this fix, the full battery **passes** and lands
byte-for-byte on our independent port (expected — the writer records are identical):

| Metric | David's PR (#37+#129+fix) | Our independent port | Base v18 (432B) |
|---|---|---|---|
| 480k fit | ✅ **536,064 tok** (1.12×) | ✅ 536,064 tok | ❌ won't fit |
| Prefill @8k | 1,178 tok/s | 1,184 | — |
| Prefill @55k (cold→warm) | 1,208 → **1,379** | 1,220 → 1,372 | 1,358 warm |
| Decode C1/ctx0 | **67.4** tok/s | 68.0 | 78 |
| Needle @56k | ✅ PASS (`738216`) | ✅ PASS | — |
| Coherence | ✅ 189 words, 0 repeat | ✅ 191 words | — |
| Arithmetic | ✅ 13,444 | ✅ | — |

The **CKV-gather prefill path ran without crashing** (log: *"Using transient
full-CKV gather for B12X sparse MLA prefill"*), confirming the fix. Decode 67.4 on
David's writer independently matches our 68.0 — the ~13% decode delta vs base v18's
432B is **inherent to the 368B FP8-rope compact record** (dequant on read), seen by
both implementations, not an artifact of either port.

*Shared for incorporation; attribution not a concern. Collaboration with
David Young + Festr.*
