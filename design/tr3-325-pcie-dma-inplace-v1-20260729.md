# TR3 3.25-bpw PCIe DMA late-allocation fix

Date: 2026-07-29

## Problem

After mixed-K EXL3 v4 removed its terminal 36 MiB conversion allocation, the
first cold 350k request reached a second late allocation:

```text
sparkinfer/comm/pcie/pcie_dma.py:346
out = torch.empty_like(inp)
torch.OutOfMemoryError: Tried to allocate 36.00 MiB
```

The tensor is the production scheduler geometry:
`3072 rows * 6144 hidden * 2 BF16 bytes = 37,748,736 bytes`.
At GMU 0.9700 the boot exposed 565,504 KV tokens, but the request still failed
with 10.69 MiB physically free and 507.02 MiB reserved/unallocated. Lowering
GMU therefore did not solve the late contiguous-allocation failure.

CN4 evidence:

```text
/home/derek/glm52-tr3-325-public-20260729/evidence/
  v4-gmu09725/failure/
  v4-gmu0970-final/failure/
```

## Lifetime analysis

`PCIeDmaAllReduce.all_reduce()` must remain out-of-place by default. Its API
explicitly permits a caller to retain one result while a later collective is
in flight, and `harness/v20_pcie_dma_output_lifetime_proof.py` protects that
contract.

The B12X fused all-reduce + residual-add RMSNorm custom op has a narrower
lifetime:

1. obtain an out-of-place all-reduce result;
2. copy that result back over `allreduce_in`;
3. immediately consume `allreduce_in` in fused add/RMSNorm;
4. discard the temporary result.

The ring algorithm can therefore target `allreduce_in` only at this seam,
eliminating both the allocation and the redundant copy without changing the
generic collective API.

## GPU proof

Harness:
`harness/v20_pcie_dma_inplace_equivalence_proof.py`

Image under proof:
`ghcr.io/yatesdr/glm52-serve@sha256:f4e4d87a605c6fe15a0d1881bd5b0206711e2fa6f0cbbd3a3ae407fa5a6dba1c`

Command:

```bash
torchrun --standalone --nproc-per-node=4 \
  v20_pcie_dma_inplace_equivalence_proof.py \
  --rows 3072 --hidden 6144 --wire-mode i8_ring --generations 2
```

Result:

| Generation | In-place vs out-of-place mismatches | Retained-result mismatches | Rank-consistent | Pointer preserved | SHA-256 |
|---:|---:|---:|:---:|:---:|---|
| 0 | 0 | 0 | yes | yes | `883031e2cc7b33e9f65a5c07a4c89e76277234191c34db2678a6bf7c31829fb4` |
| 1 | 0 | 0 | yes | yes | `8f4e719968f72a833ee66e0913857cf6c21f12b132acc7fd0b7116f0c5cd6cc1` |

Verdict: `PASS`.

## Patch

`docker/tr3-325-public/patch_pcie_dma_inplace_fusion.py` changes only
`CustomAllreduce.try_fused_add_rms_norm()`:

- the existing small-tensor one-shot fused path is unchanged;
- a production-size tensor accepted by the configured DMA channel is reduced
  with `self._pcie_dma.all_reduce(inp, out=inp)`;
- the same `ops.fused_add_rms_norm()` runs immediately on that stream;
- generic `PCIeDmaAllReduce.all_reduce(inp)` remains out-of-place.

Fail-closed pins:

| Artifact | SHA-256 |
|---|---|
| r9 input `custom_all_reduce.py` | `ac3d5dcd4bf1f933e98576ca2c0c0ff64a9436d27c1d790c144012346f6d43b5` |
| patcher | `330db3e8d26213ce6a09328e69e9fd217b92e9e6d2fd0d78d1c49e513f71d107` |
| patched `custom_all_reduce.py` | `e68dfe9805ec9df2ca1acf7e6bc0835e2cd475993a0f7ccbdeca5686db3012bd` |
| unchanged generic `pcie_dma.py` | `245654168cecf9cc0263e6f7d9577bb098a59a941dece6880a73077a91269975` |

CPU/source tests:
`harness/test_v20_pcie_dma_inplace_fusion_patch.py` — 3/3 pass.

Exact-r9 disposable-container gates:
hashes, patch application, `py_compile`, import, marker, and unchanged generic
DMA source — `PASS`.

Published validation image:

```text
ghcr.io/yatesdr/glm52-serve@sha256:b53d5d551937a0580848101dfc5df9b7fb2638419cfa6da0fa35d0a2d339fe2e
```

## End-to-end acceptance

GMU 0.9690 exposed 552,448 KV tokens, but the first production-shaped prefill
failed at a separate normal 36 MiB shared-expert output when the tightest GPU
had only 30.69 MiB physically free. This was a capacity-only pass, not a
serving pass. The exact failure is archived at:

```text
/home/derek/glm52-tr3-325-public-20260729/evidence/final-b53d5d55/failure-350k
```

The final no-restart acceptance posture used GMU 0.9688. Its measured results:

| Gate | Result |
|---|---:|
| GPU KV pool | 532,992 tokens |
| Frozen cold 350k row | EXACT (`738216`), 343,727 prompt tokens, cached=0 |
| C1 decode | 52.77 tok/s |
| C32 decode | 120.04 aggregate tok/s |
| Cold 55k-class prefill | 1,320 server tok/s; 1,314 wall tok/s |

Evidence:

```text
/home/derek/glm52-tr3-325-public-20260729/evidence/final-b53d5d55-gmu09688
```
