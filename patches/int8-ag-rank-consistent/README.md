# Rank-consistent block-INT8 PCIe-DMA candidate

This is a direct two-file overlay for the pinned v19 B12X source. It includes
the owner-roundtrip rank-consistency fix and adds the opt-in `i8` all-gather
wire mode. Server testing passed retrieval 3/3 at 300k and 3/3 at 350k total
context with zero restarts. Throughput, KLD, and remaining acceptance gates
are pending before a production recommendation.

The candidate keeps the BF16 reduce-scatter and compresses only the completed
all-gather shard. Each 128-value block uses 128 signed payload bytes plus one
FP32 scale, exactly matching the E4M3 mode's 132-byte wire footprint. The
owner and every peer materialize the same BF16 values from the same payload.

## Input and output pins

Refuse installation unless both installed inputs match:

```text
96e07e55c3843766999b88e184ce06dd  b12x/distributed/pcie_dma.py
356cff4d16db2364916325d369ea5fde  b12x/distributed/pcie_dma.cu
```

After mounting or copying both files from `overlays/`, verify the output MD5s
in `md5-manifest.txt`. The Python and CUDA overlay must be installed as a
pair; mixing either file with stock or another patch will fail at runtime.

## Local gates

From the repository root:

```bash
python3 harness/int8-dma-proofs/test_int8_codec.py
python3 patches/int8-ag-rank-consistent/checks/check_source_contract.py
```

The local machine has no CUDA compiler/runtime gate. Inside the target image,
force a fresh extension build/import and then run:

```bash
python /path/to/checks/test_pcie_dma_int8_gpu.py
```

That gate exercises 512- and 3,072-row eager calls plus CUDA-graph replay on
four GPUs. Cross-rank output must be bit-identical and the declared numerical
band must pass.

## Server order

1. Install both overlays with both wire aliases still `0`; repeat the BF16
   control gate.
2. In a separate test boot, set both aliases to `i8`.
3. Run the exact historical `quality_gate_fp8_ext.py --deep` comparison first,
   then the 50k/200k/300k/350k total-context ladder with full choice JSON.
4. Measure 55k speed only if every quality cell passes.

## Field quality result

The candidate returned `738216` with `finish_reason=stop` in all six reported
deep-context runs:

```text
300k: 3/3 PASS
350k: 3/3 PASS
RestartCount: 0
```

The corresponding E4M3 failures were also reproduced three times. With wire
bytes and layout held constant, this identifies E4M3 precision as the cause of
the retrieval loss and demonstrates that a one-byte activation wire is viable
with the block-INT8 codec.

Do not interpret a 200k pass as a 350k result. The preserved v1.3 record has
no matched 300k/350k total-context E4M3 cell.

Until throughput, KLD, and the remaining acceptance gates complete, production
remains:

```text
VLLM_PCIE_DMA_FP8=0
B12X_PCIE_DMA_FP8=0
```
