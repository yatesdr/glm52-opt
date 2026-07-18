# v18 source-available 368-byte FP8-RoPE writer (for review)

**What this is:** a pure-source drop-in for the FP8-RoPE compact-KV *writer* that
the `gilded-gnosis-v18-final` image expects but does not ship. v18's vLLM calls
`torch.ops._C_fp8_rope_ops.concat_and_cache_nvfp4_mla_fp8_rope(...)` and hard-fails
unless a standalone `/opt/fp8rope/_C_fp8_rope.so` is present — that library is
absent from every published image, so `KV_FP8_ROPE=1` (the 368-byte compact
record) cannot boot. The *reader* is already compiled into B12X
`bc85ef36192cb6e444d42ba7be86e1e125cca98a`; only the writer is missing.

This bundle supplies that writer by adapting our proven v1.3 B12X CuTeDSL NVFP4
writer to v18's exact compact-record byte layout, registered as the exact op
v18 already calls — **no compiled artifact, no `.so`**, reviewable Python/CuTeDSL.

**Why it matters:** the 368-byte record (vs v18's native 432-byte) is what makes
the full **480k** context fit on our 4×RTX PRO 6000 (96 GB) cell, and it reduces
CKV gather bytes on the prefill path. It is the missing half of v18's own
`KV_FP8_ROPE` feature.

## Status (live, CN3 SM120 — 4×RTX PRO 6000 Blackwell, PCIe Gen3)

| Gate | State |
|---|---|
| Source / pins / contract suite (9/9), py_compile | PASS (host) |
| **`checks/gpu_writer_smoke.py`** — first-call CuTeDSL compile + layout/canary on SM120 | **PASS** (compiles under `bc85ef3`, registers the exact op, pad `[292,304)` zero, zero-token→zero record, negative-slot skip, canary preserved) |
| **`checks/gpu_writer_reader_roundtrip.py`** — writer → v18's actual `run_unified_decode` NVFP4/FP8-RoPE reader vs a byte-decoded torch oracle | provided; running |
| 480k @ 368B end-to-end boot (`KV_FP8_ROPE=1`) + 55k/463k quality & MTP gates | in progress |

The GPU compile gate passing is the key de-risk: our v1.3 writer builds cleanly
inside v18's *different* B12X build (`bc85ef3` vs our `7bfc945`).

## The finding and the adaptation

The writer and reader agree on the first 288 bytes and disagree only on the
`[288,368)` tail — same encodings, same 368-byte total, different field order:

| Region | v1.3 source writer | v18 `bc85ef3` reader ABI |
|---|---:|---:|
| packed E2M1 latent | `[0,256)` | `[0,256)` |
| E4M3 group scales | `[256,288)` | `[256,288)` |
| FP32 RoPE scale | `[352,356)` | **`[288,292)`** |
| zero pad | `[356,368)` | **`[292,304)`** |
| E4M3 RoPE data | `[288,352)` | **`[304,368)`** |

`overlays/b12x/attention/mla/fp8_rope_writer.py` preserves the v1.3 latent/RoPE
quantization recipe and changes **only** the scale/pad/RoPE store destinations to
v18's layout. v18's reader loads the scale from the first 4 bytes of its 80-byte
tail and the E4M3 data 16 bytes later, so the adapted ranges are byte-aligned to it.

The op accepts v18's `scale` argument for schema parity but does not consume it:
v18 divides the 512-D compressed latent by its per-layer outer scale *before* the
writer and the reader restores it, so the 32 inline group scales keep the v1.3
implicit global scale of 1.0. The 64-D post-RoPE tensor is never outer-scaled.

## Loader change (`vllm-loader.patch`)

Touches only `_load_fp8_rope_writer()` in
`vllm/v1/attention/backends/mla/b12x_mla_sparse.py`:

1. Import `b12x.attention.mla.fp8_rope_writer` — the pure-source path, authoritative.
2. Only if that exact module is absent, fall back to `KV_FP8_ROPE_WRITER_LIB`
   (so a future compiled `.so` still supersedes cleanly). A dependency/import
   failure *inside* the bundled module fails loudly rather than being masked.
3. Require the expected namespace + op or fail closed.

`KV_FP8_ROPE=0` never calls the loader and retains the shipped 432-byte path.

## How to apply / reproduce

Two files go into the image (both under `KV_FP8_ROPE=1` + `KV_CACHE_DTYPE=nvfp4_ds_mla`):

| Host artifact | Image destination |
|---|---|
| `overlays/b12x/attention/mla/fp8_rope_writer.py` | `/opt/venv/lib/python3.12/site-packages/b12x/attention/mla/fp8_rope_writer.py` |
| stock `b12x_mla_sparse.py` + `vllm-loader.patch` | `/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py` |

GPU gates, in order, on SM120 with both files mounted:

```bash
python3 checks/gpu_writer_smoke.py            # compile + layout/canary
python3 checks/gpu_writer_reader_roundtrip.py # writer -> v18 real reader vs oracle
# then boot KV_FP8_ROPE=1; require every rank to log kv_gmem_stride=368
```

Host-side source gates (no GPU):

```bash
python3 checks/check_pins.py
python3 checks/test_writer_contract.py
```

`md5-manifest.txt` / `MANIFEST.md` pin the exact inputs, the `bc85ef3` reader
blobs validated against, and the produced artifacts.

## Suggested upstream homes

Per David: the kernel/writer belongs in the B12X repo (`lukealonso/b12x`) and the
loader preference change in the vLLM tree (`local-inference-lab/vllm`). This
bundle keeps them separable — `overlays/b12x/...` is the kernel side,
`vllm-loader.patch` is the loader side. A follow-on can enable the NF3 sibling
records (`nf3_ds_mla` 304B, `nf3bf16` 368B) in the same writer module.

*Provenance: adapted from our v1.3 B12X NVFP4 writer; shared for incorporation —
attribution not a concern. Generated in collaboration with David Young + Festr.*
