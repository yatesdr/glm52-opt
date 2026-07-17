# Packed-CKV Stage 3 integration bundle

This is the code-gate delivery authorized by
`../glm-5/sol-packed-ckv-gate2-verdict.md`. It contains exactly the four
permitted runtime overlays, one unified diff against the pinned v14eq
composite bases, and an MD5 manifest.

The default remains `B12X_DCP_PREFILL_TRANSPORT=query`. An unset, empty, or
unknown value takes the query route; an unknown value emits one startup
warning. `ckv` is a process-level mode and is fail-closed: it never falls back
to query transport after a packed-CKV communicator or capacity failure.

## Contents

```text
overlays/b12x/distributed/pcie_dma.py
overlays/vllm/model_executor/layers/attention/mla_attention.py
overlays/vllm/v1/attention/backends/mla/b12x_mla_sparse.py
overlays/vllm/v1/attention/ops/common.py
packed-ckv.patch
md5-manifest.txt
```

No CUDA/C++ source file changes are required. The byte all-gather uses the
existing `pcie_dma.cu` copy and flag entry points.

## Byte verification and deployment

Do not deploy over a different composite. From the directory containing the
four currently deployed source files, verify these exact input hashes first:

```bash
md5 vllm/v1/attention/backends/mla/b12x_mla_sparse.py
md5 vllm/v1/attention/ops/common.py
md5 b12x/distributed/pcie_dma.py
md5 vllm/model_executor/layers/attention/mla_attention.py
```

Expected values, in that order:

```text
f4462905759d332b4536059e5f2341c1
255bde14c794b6653b102df741634ac9
0cb86590849643ac15e17aa3ebb8ec5e
998654b58cd7ef60c77d979da235406c
```

Either apply `packed-ckv.patch` at that source root or copy the four files
from `overlays/` to their identical relative paths. Re-run MD5 and compare the
overlay section of `md5-manifest.txt` before booting. Preserve the existing
v14eq overlay set and collective-safe `common.py`; this bundle replaces only
the four paths listed above.

Before a GPU boot, run the in-image checks against the installed paths:

```bash
python -m py_compile \
  b12x/distributed/pcie_dma.py \
  vllm/model_executor/layers/attention/mla_attention.py \
  vllm/v1/attention/backends/mla/b12x_mla_sparse.py \
  vllm/v1/attention/ops/common.py
pyflakes \
  b12x/distributed/pcie_dma.py \
  vllm/model_executor/layers/attention/mla_attention.py \
  vllm/v1/attention/backends/mla/b12x_mla_sparse.py \
  vllm/v1/attention/ops/common.py
python -c 'import b12x.distributed.pcie_dma; import vllm.v1.attention.backends.mla.b12x_mla_sparse; import vllm.v1.attention.ops.common'
```

Local `pyflakes` and `ast.parse` checks pass for all four overlays. The three
Stage 2 CPU proofs also pass again under the isolated checker environment;
their captured output is in `../sol-packed-ckv-stage2/` and was approved in
Gate 2. Fable's Stage 3 code gate also recorded a clean in-image import:
`PCIeDmaAllGather` resolves and the unset transport remains `query`.

The overlay includes Fable's first-field-boot Triton fix from
`sol-packed-ckv-fix1.md`: `_remap_ckv_topk_kernel` receives virtual-block,
world-size, and page geometry as explicit `tl.constexpr` launch parameters.
The field-fixed sparse overlay MD5 is
`20a2cf60ce2e99d8c90249d458f330f8`.

## Required 64k boot geometry

Use the established v14eq test compose with:

```text
--kv-cache-dtype nvfp4_ds_mla
--attention-backend B12X_MLA_SPARSE
--tensor-parallel-size 4
--decode-context-parallel-size 4
--dcp-kv-cache-interleave-size 1
--max-model-len 64000
--max-num-seqs 8
--max-num-batched-tokens 3072
--num-gpu-blocks-override 400
```

Keep the existing non-DBO persistent-W_UV configuration and the existing
small-batch A2A threshold of 16. `B12X_MLA_DCP_GATHER_IN_WORKSPACE=1` is the
common eligibility gate for both transports.

At this geometry, startup must report a maximum of 2,000 packed blocks per
rank, 188,416,000 gathered-record bytes, a 47,106,000-byte local payload, and
a 141,350,912-byte non-PyTorch communicator slab. A different result means
the boot is not exercising the reviewed capacity contract.

## Paired acceptance sequence

Every throughput run is cold. Keep `prefill_bench.py`'s random first block
and record prefix-cache metric deltas.

### 1. Query parity

Remove `B12X_DCP_PREFILL_TRANSPORT` from the container environment. Keep the
v14eq baseline settings:

```text
B12X_MLA_DCP_GATHER_IN_WORKSPACE=1
B12X_DCP_GATHER_FP8=1
B12X_DCP_RS_RING=1
B12X_PCIE_DMA_FP8=ring
B12X_DCP_PROF=1
B12X_DCP_PROF_CALLS=1200
```

After READY:

```bash
python3 ~/bench/prefill_bench.py --tokens 8000 --label packed-ckv-query
python3 ~/bench/prefill_bench.py --tokens 55000 --label packed-ckv-query
python3 ~/bench/quality_gate.py
python3 ~/bench/quality_gate_fp8_ext.py
```

The 55k result must be within three percent of 964 tok/s (935–993 tok/s).
There must be no packed-CKV activation or communicator log. Existing FP8
query gather and DCP-RS ring activation signatures should remain present.

### 2. CKV mechanism run

Change only the process-level transport choice:

```text
B12X_DCP_PREFILL_TRANSPORT=ckv
```

Deliberately leave `B12X_DCP_GATHER_FP8=1`, `B12X_DCP_RS_RING=1`, and
`B12X_PCIE_DMA_FP8=ring` set. In CKV mode the code must hard-disarm the FP8
query staging and RS-ring slab despite those settings. Boot logs must contain
`Packed-CKV transport armed`; the byte communicator initializes lazily on the
first eligible layer and exactly once per process.

Run:

```bash
python3 ~/bench/prefill_bench.py --tokens 8000 --label packed-ckv
python3 ~/bench/prefill_bench.py --tokens 55000 --label packed-ckv
python3 ~/bench/quality_gate.py
python3 ~/bench/quality_gate_fp8_ext.py
```

Interpret the cold 55k result against the single 964 tok/s baseline:

| Band | Result | Verdict |
|---|---:|---|
| Confirm | at least 1,253 tok/s | Mechanism proven; proceed to phase 2 |
| Inconclusive | 1,109–1,252 tok/s | Review the phase profile |
| Kill | below 1,109 tok/s | Reject v1 mechanism |

The profiler summary is route-neutral and must fire after the configured
eligible-call count. For a pure CKV run it must show query route count zero,
CKV route count nonzero, `gather` and `rs` counts zero, `ckv_pack`, `ckv_ag`,
`ckv_remap`, and `ckv_stage` counts nonzero, `local_heads=16`, and
`missing_blocks=0`. The `ckv_stage` tag is the accepted 48 MiB head-major
copy at a full 3,072-row layer. In the inconclusive band, the transport
mechanism passes only if late-55k `ckv_ag` is at most 7 ms per layer and the
query gather/RS phases are absent.

### 3. Decode and mixed-route regression

With the 64k CKV boot still active, run Fable's C1 decode smoke using the
same config-specific baseline. Also exercise one mixed prefill/decode batch
if the harness exposes it. These calls are deterministically CKV-ineligible:
they must use the current query/NCCL path, must not enter the CKV collective,
and must not regress decode throughput or quality.

### 4. Expected full-context refusal

As a separate fail-closed check, configure `ckv` with the supplied
`--max-model-len 480000 --max-num-seqs 8` profile. Startup must reject the
15,000-block/rank layout with `Packed-CKV startup layout does not fit` and an
instruction to restart in query mode. It must fail before any CKV
communicator is adopted; runtime fallback is a test failure.

## Kill signatures

Reject the patch if any rank diverges in route choice, the CKV communicator
initializes more than once, FP8 query staging or the RS ring initializes in a
CKV process, `missing_blocks` is nonzero, gathered traffic is query-sized,
an output collective remains on pure CKV chunks, quality fails, or the
required full-context startup refusal turns into a runtime fallback.
