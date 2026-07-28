# v20 recovery — CN4 no-model proof handoff

Date: 2026-07-24  
Owner/operator: Fable  
Technical owner: Sol

## Purpose

Complete the two outstanding GPU-only discriminators after the already-qualified
v19 control and first-pass CN4 proofs. These tests require no model weights and
select the source changes for one consolidated current-head v20 recovery image:

1. persistent-output and CUDA-graph block-INT8 DMA-ring correctness;
2. exact old-versus-new MLA query assembly plus production-width long-context
   top-k correctness.

Do not stop CN3 production and do not build or boot another model between these
tests.

The raw peer matrix is complete and accepted; do not repeat it. The first
collective run is also accepted for explicit-output timing and the crossover
decision. Rerun Proof 2 only because that first script did not exercise the
persistent default output or graph replay added by SparkInfer PR #76.

## Control already complete

The exact v19 150k boundary is already preserved at
`/home/derek/probe150.log`:

```text
depth=150000 pos=40% ctx=147369 cached=0 completion=85 finish=stop secs=609
retrieval=PASS (where=content) finalization=PASS
content='738216'
```

Do not repeat it.

## Inputs and byte pins

Copy the two outstanding proof files from the shared repository to a temporary
proof directory on CN4. Verify before execution:

```text
efc26165c0e48db02b392da2a281b47bf9f882286e4114ad241071d521569919  harness/pcie_collective_matrix.py
9785aae7c9d78c1df8b9c1ea1d88c9876b72e61b14a7855cb032c7497386eaa4  harness/v20_decode_retrieval_microprobes.py
```

Use the already-local exact failing image:

```text
ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-memory-reclaim-test-20260724
sha256:0567498e6d790e6fcd294be431381b71c03409049b0fd635462c1b1623ec2b91
```

The parent Festr image `sha256:adddafd2…` contains the same SparkInfer/query/top-k
bytes, but using the failing derived image removes any ambiguity.

## Gate 0 — quiescence

Fable is the only operator authorized to stop the current CN4 control
container. Let its 475k ladder request and requested C1 decode control finish
first; then stop that same old-v20 process normally. Do not stop CN3
production. After the CN4 control stops, verify:

```bash
docker ps --format '{{.Names}}'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
sha256sum /path/to/proofs/*.py
docker image inspect \
  ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-memory-reclaim-test-20260724 \
  --format '{{.Id}}'
```

PASS requires no model process on any GPU and both hashes exact.

Set local shell variables to explicit paths; do not use a broad home-directory
mount:

```bash
PROOF_DIR=/absolute/path/to/the/two/proof/files
IMAGE=ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-memory-reclaim-test-20260724
RESULT_DIR=/absolute/path/to/a/new/result-directory
mkdir -p "$RESULT_DIR"
```

## Proof 1 — raw peer matrix (complete; do not repeat)

Accepted artifact:

```text
harness/sol-proof-results/pcie-peer-matrix.jsonl
sha256 21ccc7e6619ab1210fadf043bea72cfcfbcb588cd705c356c33465ef1755fadf
```

All 12 directed edges and four concurrent patterns completed. Direct within-pair
copies reached 14.1–14.3 GB/s. Cross-pair traffic was directional
(approximately 10.1 GB/s toward GPUs 2/3 versus 4.6–6.2 GB/s toward GPUs 0/1),
and concurrent cross/ring patterns collapsed to 1.57–1.82 GB/s per edge. This
proves a topology/fabric limitation. ACS/root-complex routing remains a
hypothesis pending PCIe configuration evidence; do not present it as proven by
throughput alone.

Historical command, retained only for reproducibility:

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --entrypoint /opt/venv/bin/python \
  -v "$PROOF_DIR:/proof:ro" \
  "$IMAGE" \
  /proof/pcie_peer_matrix.py \
  | tee "$RESULT_DIR/pcie-peer-matrix.jsonl"
```

No rerun is requested.

## Proof 2 — persistent i8-ring DMA and graph replay

The accepted first pass proved that explicit-output DMA beats NCCL by 12–15x at
the prefill-relevant 1,024/3,072 rows, so the crossover is not being changed.
This rerun validates only the newly added persistent-output and graph-replay
contracts while retaining the timing rows as a consistency check.

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --entrypoint /opt/venv/bin/torchrun \
  -v "$PROOF_DIR:/proof:ro" \
  "$IMAGE" \
  --standalone --nproc-per-node=4 \
  /proof/pcie_collective_matrix.py \
  | tee "$RESULT_DIR/pcie-collective-matrix.jsonl"
```

PASS correctness:

- process exit 0;
- `explicit_finite=true`, `default_finite=true`, and `graph_finite=true` in
  all four records;
- `explicit_rank_max_abs_divergence=0`,
  `default_rank_max_abs_divergence=0`, and
  `graph_rank_max_abs_divergence=0` in all four records.

The three DMA forms are deliberate:

- `explicit` verifies a caller-provided output tensor;
- `default` verifies and times the persistent-output path used by vLLM;
- `graph` captures and replays that persistent output after changing the
  input bytes, proving that the pointer and result remain valid under CUDA
  graphs.

Selection:

- if `dma_default_over_nccl > 1.0` at the prefill-relevant 1,024/3,072 rows,
  the universal 6 MiB DMA crossover is wrong for CN4; select a group-uniform
  measured crossover/fail-closed NCCL patch;
- if DMA wins those rows, do not change the crossover. The prefill target moves
  above transport dispatch to DCP synchronization/ownership profiling.
- record `dma_default_over_explicit`; a large regression isolates the
  persistent-output implementation even if the raw explicit-output ring wins.

The INT8 result is expected to differ numerically from BF16 NCCL. Record its
max/mean error; rank identity and finiteness are the hard correctness gates.

## Proof 3 — fused query and long-context top-k

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --entrypoint /opt/venv/bin/python \
  -v "$PROOF_DIR:/proof:ro" \
  "$IMAGE" \
  /proof/v20_decode_retrieval_microprobes.py \
  | tee "$RESULT_DIR/v20-decode-retrieval-microprobes.jsonl"
```

PASS top-k requires exit 0 and summary `topk_failures=0`.

The top-k cases use the production DCP4 logical-index paged specialization
(`block_q=32`, `block_k=512`, 32,768-token supertiles). They test 32,767,
32,768, and 32,769 explicitly: at exactly 32,768 local tokens the serving path
switches to the two-level 16k slice fold, and at 32,769 it adds a second
supertile. The probe reproduces both `run_tiled_topk(... extent_splits=...)`
and the final `run_row_topk(... output_gather_table=...)`, plus the observed
150k/250k DCP4 local widths. `fold_boundary` and `quantized_ties` cover
slice/cross-chunk winner preservation and FP8-derived repeated score levels
rather than random logits alone. It runs each width at `rows=1/9/16/32`:
single-token decode, the first uneven MTP/DCP shape, the production capture
ceiling, and one full selector tile.

Query-path selection:

- the query proof reports three routes: established safe BMM + assembly +
  static quantization; fused BF16 projection/assembly + the established static
  quantizer; and the fully fused direct-FP8 output;
- if fused-BF16 + static quantization preserves staged retrieval materially
  better than direct FP8 while retaining most of the fused timing gain, select
  that narrower epilogue correction;
- select the fully staged BF16-weight + FP8-output correction if the
  fused-BF16 intermediate already reduces production-scale top-2,048
  retrieval overlap, or if neither fused route improves tiny-M timing;
- keep fusion if it wins timing and retrieval overlap remains effectively
  exact. Raw BF16/FP8 byte inequality alone is diagnostic, not an automatic
  failure.

Top-k selection:

- any `out_of_range`, duplicate row, below-threshold result, value mismatch, or
  nonzero `topk_failures` is a real selector defect and selects a top-k patch;
- if every production width passes, do not revert the widened selector.

If both query and top-k prove clean, the next and only model-level corrective
A/B is the format-qualified compact NVFP4 MTP verifier route from PR #171.

## Evidence return

Append to `fable-sol-comms.md`:

- exact Proof 2/3 commands and image ID;
- both exit codes;
- the complete summary records;
- artifact paths and SHA-256 hashes;
- GPU Xid/illegal-access/OOM check over the proof interval.

Leave CN4 idle after the proofs. Do not start a new v20 model until Sol has
selected and pinned the consolidated recovery branch.
