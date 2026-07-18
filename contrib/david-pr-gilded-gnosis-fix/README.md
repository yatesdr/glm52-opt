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

<!-- RESULTS-PLACEHOLDER -->

*Shared for incorporation; attribution not a concern. Collaboration with
David Young + Festr.*
