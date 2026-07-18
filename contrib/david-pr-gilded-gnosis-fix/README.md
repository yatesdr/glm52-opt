# Fix for David's fp8-rope PRs on the gilded-gnosis-v18 image — 2nd writer call site

**For:** `lukealonso/b12x#37` (compact NVFP4 FP8-RoPE writer) +
`local-inference-lab/vllm#129` (use the packaged writer).

**TL;DR:** Both PRs are correct and their writer is byte-identical to the one we
independently ported (same layout scale@288/pad@292/RoPE@304, same recipe). But
`vllm#129` converts **only one** of the two writer call sites present in the
**deployed gilded-gnosis-v18 image**. After #129 removes the
`torch.ops._C_fp8_rope_ops` registration path, the second call site — the
**DCP CKV-gather prefill path** — still calls that now-unregistered op and
**crashes at first prefill** under `KV_FP8_ROPE=1` + `DCP_CKV_GATHER=1`. This
patch binds that site to the same packaged writer #129 already imports.

## Why this appears on the image but maybe not in your base

The image's `vllm/v1/attention/backends/mla/b12x_mla_sparse.py`
(md5 `14c14eabc937cddf481532fb19e1dcb5`,
`voipmonitor/vllm:gilded-gnosis-v18-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718`)
is ~400 lines larger than the `local-inference-lab/vllm` base #129 was written
against — the extra code is the CKV-gather machinery (`gathered_buffer`,
`dcp_padded_total_tokens`, owner/local-pos slot mapping). That block contains a
**second** writer invocation:

```python
        k_scale = getattr(layer, "_k_scale", None)
        if self._kv_fp8_rope:
            torch.ops._C_fp8_rope_ops.concat_and_cache_nvfp4_mla_fp8_rope(   # <-- still the old op
                kv_c, k_pe_flat, gathered_buffer, slots, k_scale,
            )
```

`#129` deletes `_load_fp8_rope_writer()` and the op registration, then rebinds
`do_kv_cache_update`'s call to `self._concat_and_cache_nvfp4_mla_fp8_rope`.
This CKV-gather site is left untouched, so under `DCP_CKV_GATHER=1` (which the
gilded-gnosis NF3 profile runs) the first prefill hits an unregistered
`torch.ops._C_fp8_rope_ops` and dies. If your `local-inference-lab/vllm` base has
this same CKV-gather call site, #129 needs this second conversion there too;
if not, it is specific to the gilded-gnosis build and Festr should carry it.

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
