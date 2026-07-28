# v20 runtime-stride RC long-context causal gate

Date: 2026-07-27

Status: complete; SparkInfer PR #85 is ruled out as a sufficient fix.

## Why this gate precedes the full-precision reference experiment

The prior accelerated-path evidence used:

```text
vLLM       0c79e41db41f250ccdfc4be92d171960a5787f73
SparkInfer e603f74bb67d0fce547336f1fb73c3c23e8f1887
image      sha256:10261c7d65101c8aba2ce1fb59eabe73aff9d35eca5043b330cc0ce76d3c98d0
```

SparkInfer PR #85 established that this SparkInfer lineage contains a silent
runtime page-table-row-stride defect in tiled top-k. A cubin compiled for one
page-table width could be reused at another width while retaining the old
two-dimensional CuTe row stride. Row zero remained correct, but later rows
could read adjacent columns. The symptom began beyond the 16-row tiled-path
threshold and depended on compile/cache population order.

That failure mode overlaps the observed long-context symptom closely enough
that the old accelerated selection capture cannot be used to attribute the
350k misses to FP8 arithmetic. The already-built full-precision oracle remains
valid as an instrument, but running it against the known-bad SparkInfer base
would not discriminate arithmetic from page-table corruption.

## Immutable test stack

The refreshed stock RC is:

```text
image:
  voipmonitor/vllm@sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0
CN4 Docker-inspect manifest ID:
  sha256:131481b0f12c455a8fbad72c5909eb3a2c3accd96815743fdcfa134396e548c0
vLLM:
  0c79e41db41f250ccdfc4be92d171960a5787f73
SparkInfer:
  c3828fd7f807ce237a9ac36ef033659e6f6b6dd3
installed tiled_topk.py:
  284bd167a971cc6c992c8b2b3ce120000185ef6ffe93be845036e098bfc834f2
```

The causal image has no source overlay. It adds only a pinned PATH wrapper
that appends `--no-enable-prefix-caching` to the stock helper's vLLM command:

```text
wrapper:
  docker/vllm-no-prefix-caching
  3708581ff5115d0546c632e87c25e6c47dc565fb673dc3b38c2f30b5f4e46271
Dockerfile:
  docker/Dockerfile.v20-runtime-stride-stock-causal-20260727
  cf58f66c87bb345fe3f26b6014dd5ba8ffbbf9a0bb7c05e7335da21cae392981
CN4 image ID:
  sha256:e288bc87717df765769052decfe716c63af87ef54e6b09e7d93e6b85ff8f9dae
```

## Frozen gate

Configuration remains matched to the current-base exact/oldest-boundary
causal tests:

```text
TP4 / DCP4 / MTP3
max model length 480000
max batched tokens 3072
NVFP4 MLA KV
FP8 RoPE
i8_ring
GRAPH=64
exact selector
DCP query split off
DCP owner merge off
CKV prefetch depth 0
prefix caching disabled
fresh compile/cache namespace
```

Frozen package pins:

```text
manifest:
  a2ad521f83750b696add479cd91f1b82bb49582761a34a91f85bdf562e15f79f
250k control:
  fde493ea5b921594d239e2a743229d61c9977557057aa49bd1389700d5a56b54
350k r1:
  f0d1c16d816b777f27a3882d9e6b5ef056852684ea155fb11dd845f9e1654ab5
350k r2:
  d5b6755331b634bbabc24486f74925832179eac7842b7a8a7ee225b52b1cdec6
350k r3:
  a50329d3866ba97ead8ae10291cfb8903b8e542f79ea4e985d365f9db7447b46
```

Each request must be cold, finish with `stop`, and finalize exact content
`738216`.

## Boot result

The stock RC reached healthy state with:

```text
restart count:       0
prefix caching:      false (verified in parsed server arguments)
selection policy:    exact (verified in live container environment)
DCP query split:     0
DCP owner merge:     0
wire:                i8_ring
KV pool:             550,144 tokens
maximum model len:   480,000 tokens
maximum concurrency: 1.15x
```

The 250k frozen control completed before any 350k request:

| Cell | Prompt tokens | Cached | Finish | Output tokens | Content | Verdict |
|---|---:|---:|---|---:|---|---|
| `pass-250k-ctl` | 245,497 | 0 | `stop` | 4 | `738216` | EXACT |
| `fail-350k-r1` | 343,727 | 0 | `stop` | 16 | `The maintenance ticket number ... is 27.` | ABSENT |
| `fail-350k-r2` | 343,727 | 0 | `stop` | 25 | `... is MAINT-2024-0917` | ABSENT |
| `fail-350k-r3` | 343,727 | 0 | `stop` | 17 | `The maintenance ticket number ... is 27.` | ABSENT |

All three frozen 350k rows failed on the #85-fixed stock RC. The 250k control
passed first, every request was cold, every response finalized normally, and
the server remained healthy with zero restarts. The result rules out PR #85 as
a sufficient restoration of deep retrieval.

This was a clean single-variable test of the stride fix. The DCP query-split
and owner-merge policies were pinned off, calibration did not override those
explicit choices, and the remaining serving configuration matched the prior
accelerated-path gate. The correct conclusion is not that the defect is
irrelevant: PR #85 fixes a real independent corruption bug. It simply is not
the cause of these frozen long-context misses.

The wrong-answer morphology was repeatable within this run: r1 and r3 returned
the near-context number `27`, while r2 fabricated a ticket-shaped value. No
byte-for-byte archive of the old-build r2 response was found, so cross-build
per-prompt determinism is not claimed.

Local evidence archive:

```text
harness/cn4-evidence-archive/20260727/
  runtime-stride-stock-causal/control-first/

gate.log:
  b88aca66a54ee161411ccf7385e22819c6abfe6eda672eae4ae42fa38aa673e9
boot-before-gate.log:
  e50d25ab34f3f4e2c6da0910bc56f5ab14e5a02f9b92da30f93a82a64c4fab91
boot-after-gate.log:
  c5d592d00e960bc9b9b6281f6ee1a6b8093f4eba67842320a7126fac04672e70
results/summary.json:
  32336eebfc0216a2dd479a37706813af616ee0bac1237475feaa08243c016c55
results/rows.json:
  336a0b70c9430ffe4f2646af3e98558eedaa56db31372d52e43247b6ddf8fe84
```

## Decision rule

The second branch fired: the FP8/scorer hypothesis survives its strongest
alternative explanation. The corrected full-precision oracle is rebased onto
this RC and proceeds through its no-model proofs, smoke, 8k, and frozen causal
gates.

## Invalid reference boot retained for audit

The earlier reference boot ran no requests. It failed during startup because
the diagnostic BF16 indexer cache was not included in vLLM's final generic KV
allocation budget, causing the last 100 MiB KV allocation to OOM. Fable also
found that the baked source would reject the live builder's `(B, 1)` MTP0
decode lengths. Both are fail-closed and neither is quality evidence.

Remote archive:

```text
/home/derek/proof-results/20260727/official-reference-mode-v1/causal-boot-eager-invalid
VERDICT.txt:
  150733a1198731ebd35d2b7b22728aff6530be0e6ad6003a99f8dc54c39de4ad
container.log:
  97be728d9a6c1424e4a2f54581032cd0cf22b167c4146c760b71c4eee4d3b789
```

The corrected reference source is commit
`a38fc99af111c2074eb222a19b8f1f24362fab48`; it normalizes `(B, 1)` decode
lengths and rejects nonzero prefill key-window bases. It is held in reserve
until the stock RC gate decides whether it is needed.
