# i8_a2a decode crash — CUBLAS_STATUS_INTERNAL_ERROR in MLA `_v_up_proj` (Fable → Sol)

Facts only, same format as the other specs. The exact crash, the stack, the single most
important analytical finding (the failing op is **wire-mode-independent**), the contrast that
isolates the variable, the labeled hypothesis, and the reproduction test (in progress).

## TL;DR

During the `i8_a2a` perf bench, one TP worker died with `CUBLAS_STATUS_INTERNAL_ERROR` on a
**bf16 batched GEMM inside MLA attention** (`_v_up_proj`), during the **decode ctx=50k** phase.
The failing op is in the attention V-up-projection — **not** the b12x PCIe fp8-DMA wire path, which
is dormant at decode message sizes. `i8_ring` ran the byte-identical bench and did **not** crash.
So the crash is **very unlikely to be caused by the `i8_a2a` wire mode itself.** Leading suspect is
allocator/workspace pressure under `expandable_segments:False` after a run of deep-context prefills.

## Environment (exact)

- Image `voipmonitor/vllm:gilded-gnosis-v19-vllm7ea567a-b12x4cfa530-fi801d57a-cu132-20260718`.
- GLM-5.2 hybrid, TP4 / DCP4, `--dcp-comm-backend=a2a`, `--dcp-kv-cache-interleave-size=1`.
- KV: `--kv-cache-dtype=nvfp4_ds_mla`, `KV_FP8_ROPE=1` (368 B/token), `--attention-backend=B12X_MLA_SPARSE`.
- Quant: `nvfp4_nf3_hybrid` + mxfp8 linear/shared-experts. MTP `num_speculative_tokens=3`.
- `--gpu-memory-utilization=0.970`, `--max-model-len=480000`, `--max-num-seqs=16`,
  `--max-num-batched-tokens=3072`, `--async-scheduling`.
- Offload **on**: `OffloadingConnector` / `TieringOffloadingSpec`, `cpu_bytes_to_use=64e9`,
  `secondary_tiers:[]`. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`** (required for offload).
- Wire mode this run: `B12X_PCIE_DMA_FP8=i8_a2a` / `VLLM_PCIE_DMA_FP8=i8_a2a` (block-INT8, a2a topology).
- KV pool at boot: `GPU KV cache size: 644,864 tokens` (1.34x @ 480k). Clean boot, RestartCount 0.

## The crash (exact)

- **Faulting op** (deepest frame):
  ```
  vllm/model_executor/models/deepseek_v2.py:1555  forward  (AOT-compiled / cudagraph region)
    → mla_attention.py:1620  unified_mla_attention_with_output
    → mla_attention.py:1237  forward_impl
    → mla_attention.py:1476  _v_up_proj
        torch.bmm(x, self.W_UV, out=out.transpose(0, 1))
  RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling
    cublasGemmStridedBatchedEx(handle, opa, opb, m, n, k, &falpha,
      a, CUDA_R_16BF, lda, stridea, b, CUDA_R_16BF, ldb, strideb, &fbeta,
      c, CUDA_R_16BF, ldc, stridec, num_batches, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
  ```
- **Rank:** `Worker_TP2_DCP2 pid=351` (single rank raised; the other 3 did not log the GEMM error).
- **Proximate engine death:** the raise hung the collective; `EngineCore` then died 5 min later on
  `TimeoutError: RPC call to sample_tokens timed out` (23:59:03). Root fault is the 23:54:04 GEMM.
- **Recovery:** `restart: unless-stopped` auto-restarted the container; RestartCount 0→1; it
  re-initialized (weights + offload region) and returned to `health=200`. No data corruption on restart.

## Timeline (bench relative)

| Time | Event |
|---|---|
| 23:46:44 | `i8a2a_prescribed.sh` starts |
| ~23:46–23:51 | **DECODE ctx=0** phase (conc 1,2,4,8,16) — **completed OK**, full matrix captured (C1 62.7 … C16 171.0) |
| ~23:51–23:54 | **DECODE ctx=50k** phase (conc 1,2,4,8) — crash landed here (~conc 4–8 cell) |
| 23:54:04 | `Worker_TP2` `cublasGemmStridedBatchedEx` → `CUBLAS_STATUS_INTERNAL_ERROR` in `_v_up_proj` |
| 23:54:06 | offload metric snapshot: `kv_offload_cpu_cache_usage_perc=0.0` (offload **not** actively spilling) |
| 23:59:03 | `sample_tokens` RPC timeout → `EngineDeadError` → auto-restart |

## The single most important finding: the failing op is wire-mode-independent

`_v_up_proj`'s `torch.bmm(x, W_UV)` is the MLA attention V-up-projection. It is a **bf16 batched
GEMM that runs identically for every value of `B12X_PCIE_DMA_FP8`.** The `i8_a2a` code only changes
the **TP all-reduce transport** (`custom_all_reduce` / `b12x/distributed/pcie_dma`), and that path
**only engages above the 6.29 MiB DMA threshold** — decode all-reduces (`[batch, 6144]`) are far
below it, so the fp8-DMA wire path is **dormant during the decode phase where the crash occurred.**
The `i8_a2a` wire mode is therefore **not in the crash stack** and is unlikely to be the cause.

## What isolates the variable

- **`i8_ring` ran the byte-identical perf bench** (verified: `diff` of the two scripts is names only)
  and **completed with RestartCount 0**, no cuBLAS error. Same offload, same GMU, same concurrency caps.
- `--dcp-comm-backend=a2a` is **constant** across the ring and a2a runs (both set it), so the DCP
  all-to-all is **not** the differentiator either.
- The only config delta between the crashing run and the clean `i8_ring` run is
  `B12X_PCIE_DMA_FP8=i8_a2a` vs `=i8_ring` — and per the finding above, that path is inactive at decode.
- **This points to run *state/sequence*, not topology.**

## Hypothesis (labeled — not yet proven)

**Allocator/workspace pressure under `expandable_segments:False`.** Immediately before the perf bench,
the same engine served, back-to-back with no restart: the full a2a needle sweep (5 depths to 475k)
**plus 3× 350k recheck prefills** — i.e. **four near-max-context prefills in a row**. With
`expandable_segments:False` (mandatory for the offload tier) the CUDA caching allocator cannot
compact, so that history can leave the arena fragmented. The subsequent decode-ctx50k phase (8×50k
concurrent) then needs a fresh cuBLAS workspace for the strided-batched `_v_up_proj` GEMM;
`CUBLAS_STATUS_INTERNAL_ERROR` is a known sticky-downstream symptom of a failed workspace allocation
(or of an earlier async fault poisoning the context). `i8_ring`'s perf bench did **not** have a
3×-recheck prefill barrage in front of it — cleaner arena. The crash op sits inside an
AOT-compiled/cudagraph region (`caching.py:optimized_call` → `execution_fn`), so a graph-replay ↔
workspace interaction is also in scope.

Alternative not excluded: a genuine transient CUDA/cuBLAS fault unrelated to any of the above.

## Ruled out

- **Not** the offload `_build_store_jobs` scheduler assertion (different bug; PR #133). Offload
  `cpu_cache_usage=0.0` at crash time — it wasn't spilling.
- **Not** an INT8 kernel — the failing GEMM is bf16 (`CUDA_R_16BF`).
- **Not** a boot/KV-fit issue — pool allocated cleanly at 644,864, ran a full ctx=0 decode matrix first.

## Reproduction test (in progress)

Re-running the **identical** perf bench from a **clean post-restart state** (no preceding deep-prefill
barrage; GPUs idle 0%, health 200, RestartCount baseline 1). Discriminator:
- **Completes clean** → supports the fragmentation/sequence hypothesis; `i8_a2a` exonerated; it gets a
  full prefill/decode row.
- **Crashes again at the same decode-ctx50k point** → the fragmentation hypothesis is wrong and this
  is load-intrinsic (and worth Sol treating as a real MLA `_v_up_proj` / workspace bug independent of
  wire mode).

**RESULT: did NOT reproduce.** The clean-state re-run **completed the full bench** — ctx=0 decode,
**ctx=50k decode (the crash stage from run 1)**, and standalone prefill — with **RestartCount
unchanged (no new crash)**. Full numbers captured: standalone prefill 8k **1,463** / 50k **1,432**;
decode ctx0 C1 62.1 → C16 162.7; decode ctx50k C1 56.0 → C8 121.4.

**Interpretation:** one crash, not reproducible from a clean arena → **supports the
fragmentation/sequence hypothesis and exonerates `i8_a2a` as intrinsically unstable.** The consistent
story: the run-1 crash followed a 4× back-to-back deep-prefill barrage that run 2 did not have. The
failing op remains a **wire-mode-independent MLA `_v_up_proj` bf16 GEMM**, so if it recurs it should
be treated as a general MLA/workspace-under-pressure edge case (could bite `ring`/`i8` too), not an
a2a defect. Still worth Sol's eyes on the 4 questions below — a non-reproducible cuBLAS INTERNAL_ERROR
under memory pressure is a latent risk regardless of which wire mode surfaced it.

## Open questions for Sol

1. Does `_v_up_proj`'s `cublasGemmStridedBatchedEx` allocate a per-call cuBLAS workspace, and is that
   allocation on the vLLM caching-allocator arena (so it's exposed to fragmentation under
   `expandable_segments:False`)? Is the `out=out.transpose(0,1)` non-contiguous output a factor?
2. Is this GEMM inside a captured CUDA graph at decode? If so, can a replay hit
   `CUBLAS_STATUS_INTERNAL_ERROR` from a reclaimed/moved workspace pointer under memory pressure?
3. Is there a known interaction between the offload connector's DRAM staging and cuBLAS workspace
   headroom at high decode concurrency (8×50k) that would starve this GEMM specifically on one rank?
4. Given the op is wire-mode-independent, do you agree `i8_a2a` is not implicated, and the correct
   framing is "MLA `_v_up_proj` GEMM fails under workspace pressure" — i.e. it could bite `ring`/`i8`
   too under the same preceding-load sequence?

## Artifacts

- Full stack + error: container logs (frames captured in this doc). Crashed-run bench data preserved
  at `~/bench/i8a2a-prescribed-crashed-run1/` on CN3; nohup log `~/bench/i8a2a_prescribed.log`.
- Clean-state re-run: `~/bench/i8a2a_prescribed_run2.log` (+ `~/bench/i8a2a-prescribed/`).
- Byte-identical `i8_ring` control run (no crash): `~/bench/i8ring-prescribed/`, `~/bench/i8ring_prescribed.log`.
