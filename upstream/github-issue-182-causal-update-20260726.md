## Causal update: local sparse-indexer selection semantics control the failure

We now have an end-to-end causal result, not only a source bisect or
off-model numeric difference.

### What was held fixed

- same v20 `5517197`/`be0edca` model and SparkInfer lineage;
- same NF3 hybrid weights;
- TP4/DCP4/MTP3;
- MNBT 3072;
- NVFP4 MLA KV and FP8 RoPE (368-byte record);
- `i8_ring`;
- prefetch depth zero;
- identical frozen prompts, token IDs, seeds, and cold-cache requirement;
- same PCIe default-output lifetime fix from SparkInfer PR #80.

The only semantic discriminator was the local radix selector policy.

### Boundary localization

After fixing the independent PCIe DMA output-lifetime bug:

- `q_fp8` is byte-identical between v19 and v20 on all four ranks;
- indexer weights are byte-identical;
- the current-token BF16 index key is byte-identical;
- sequence length, page table, and dispatch geometry match;
- the first material difference is the rank-local top-k selected set.

A four-way learned-input replay shows that the output follows the
implementation, not the captured inputs:

- current v20 returns the exact dense-reference top-k (2048/2048) on either
  v19 or v20 learned tensors;
- historical v19 returns 1872--1896/2048 of that set and omits 165--193
  candidates with strictly higher *quantized proxy* scores;
- a v20 `bounded_compat` selector reproduces the historical v19 selected set
  at about 0.97--0.99 Jaccard.

Source history separates two changes:

- `1012199e` added an exact overflow rescan;
- `83a58444` widened the coarse radix from 8 to 10 bits and the shared
  candidate buffer from 4096 to 8192.

The exact rescan is mathematically exact for the E4M3-query/FP8-key indexer
proxy. That does not prove the proxy's exact ordering is the best sparse
attention set for the model's downstream higher-information attention
calculation.

### End-to-end discriminator

The discriminator keeps all shared-memory writes bounds-checked, but uses the
historical 8-bit coarse bucket and 4096-entry candidate budget and does not run
the exact overflow rescan. Default exact behavior is unchanged unless the
explicit compatibility policy is selected.

Image:

```text
glm52-serve:v20-longctx-indexer-bounded-discriminator-20260726
sha256:0469df9293ecb129f60abbce0f38e0d86edf3996600d7acb9f657a7e4ac529e2
```

Live selector source SHA-256:

```text
a136f11140da3b582bb136eb50b9977ba13adf6d111cd2ffa1420a86d81361eb
```

Frozen cold results:

| Cell | Stock v20 | Discriminator | Finish | Cached | Content |
|---|---|---|---|---:|---|
| 250k control | EXACT | EXACT | stop | 0 | `738216` |
| 350k r1 | ABSENT | EXACT | stop | 0 | `738216` |
| 350k r2 | ABSENT | EXACT | stop | 0 | `738216` |
| 350k r3 | ABSENT | EXACT | stop | 0 | `738216` |

Boot stayed healthy with restart count zero, a 507,612-token KV pool at
460k max length, complete graph capture, and zero illegal-access, cuBLAS,
EngineDead, OOM, traceback, assertion, or worker-died signatures.

### Interpretation and proposed upstream direction

This confirms selector semantics as a controlling cause of the deep-context
regression. It does **not** justify making accidental bounded overflow the new
default.

Our proposed direction is:

1. keep the output-lifetime fix;
2. retain an explicit bounded compatibility policy as a fail-safe and
   reproducible discriminator;
3. compare the exact-vs-bounded candidate difference against a
   higher-precision relevance reference rather than the quantized proxy;
4. replace overflow-dependent omission with a deterministic,
   precision-aware candidate policy or rerank;
5. require the frozen controls, all 350k replicas, the full 50k--475k ladder,
   and performance/KV gates before changing the default.

The complete internal result record is
`design/v20-longctx-first-divergence-20260726.md`; I can publish the
compatibility patch and learned-tensor replay harness as a reviewable draft
once maintainers confirm whether they prefer an explicit policy first or a
precision-aware default in one PR.
