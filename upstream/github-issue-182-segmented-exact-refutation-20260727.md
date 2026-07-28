## 2026-07-27 update: two-stage segmented-exact candidate repair is necessary but not sufficient

We tested a new v20-native, deterministic, bounds-safe candidate policy. This
does **not** use the v19 threshold-bucket overflow behavior.

### Why this experiment was run

The frozen failing 350k trace localized the first observed irreversible
candidate loss to sparse-indexer layer 34. At that layer:

- the complete existing four-rank DCP candidate union retained only one token
  in the needle region;
- therefore no global-only merge policy could reconstruct the missing local
  candidates;
- an offline calibration found the smallest tested policy that retained the
  needed candidates:
  - local: exact top 64 per chronological quartile on every DCP rank;
  - global: exact top 256 per chronological quartile;
  - fill the remaining top-2048 budget by exact FP32 score.

The reserved winners only receive selection priority. The emitted IDs and
scores are restored from the original exact candidates. There is no
out-of-bounds access and no variable-size output.

### Operator-level proof

The actual patched SparkInfer local selector and actual patched vLLM global
selector were run on GPU against the frozen layer-34 activation:

- base image:
  `voipmonitor/vllm@sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0`
- geometry: 340,992 eligible tokens, DCP4, 16,384-token local slices,
  top-k 2048;
- both production selectors matched independent Torch oracles exactly;
- CUDA graph capture and replay passed;
- the final bounded global set retained four needle-region tokens:
  `137483`, `137485`, `137493`, `137503`.

Artifact:
`/home/derek/proof-results/20260726/segmented-exact-runtime/layer34-runtime-v1.json`

### End-to-end causal image

The hash-gated overlay used:

- vLLM commit `9466cf51c7d08533a4d3bb5f9d7b299ac1838d8c`;
- SparkInfer commit `1ed69d6f5dc81b25a43c7b2c782d5a4db483108d`;
- prior PCIe output-lifetime fix `84aa4ac3`;
- image ID
  `sha256:c892037645aa4a183bb751cdb34d45adf515c88bee1701430ff9aaa8759b71d4`;
- `SPARKINFER_NSA_TOPK_SELECTION_POLICY=segmented_exact`;
- `VLLM_DCP_INDEXER_SELECTION_POLICY=segmented_exact`.

The live environment and installed file hashes were verified after startup.
Boot was clean:

- 981,236 KV tokens at max length 360k;
- 7.24 GiB KV cache;
- 0.05 GiB actual CUDA-graph memory;
- zero restarts and no fatal signatures.

Frozen, byte-pinned prompts were replayed cold:

| Cell | Prompt tokens | Cached | Finish | Content | Verdict |
|---|---:|---:|---|---|---|
| 250k control | 245,497 | 0 | stop | `738216` | EXACT / PASS |
| 350k-r1 | 343,727 | 0 | stop | `RRYQ-056281` | ABSENT / FAIL |

Artifact:
`/home/derek/proof-results/20260726/segmented-exact-causal/frozen-gate1/summary.json`

### Conclusion and next discriminator

This result refutes the claim that repairing the measured layer-34 candidate
loss alone restores end-to-end retrieval. It does **not** refute the measured
candidate loss or the selector implementation: those passed independently.
It establishes that at least one additional lossy boundary remains downstream
or at another sparse-indexer layer.

The next efficient experiment is not another blind selector calibration. It
is one trace boot using this exact causal image and the same frozen 350k-r1
prompt, capturing the final selected IDs at the previously implicated sparse
layers (34, 38, 42, 46, 50, 58, 66) and, where the needle first disappears,
the complete bounded local/global candidate sets. That will identify the first
remaining irreversible boundary under the repaired policy and decide whether
the next fix belongs in local selection, global selection, or downstream
sparse attention.
