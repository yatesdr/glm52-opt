## 2026-07-27 update: working trajectory isolated; overflow can be replaced by an explicit boundary policy

The matched 21-layer trace requested in the previous update is complete. It
compares the same frozen cold 350k-r1 request under:

- failing v20 `segmented_exact`;
- working v20 `bounded_compat`, which returns exact final content `738216`.

The request bytes and tokenization are identical. The value is three tokens:

```text
137499 = "7"
137500 = "38"
137501 = "216"
```

### Cross-layer causal difference

The working selector develops the needle sentence progressively:

| Layer | Working `bounded_compat` selection |
|---:|---|
| 30 | token `137485` (the comma before the sentence) |
| 34 | 10 sentence-local tokens |
| 38 | 19 sentence-local tokens, including all three value tokens |
| 42 | 29 sentence-local tokens |

The failing `segmented_exact` trajectory has no broad sentence coverage
through layer 38, sees only the comma at layer 42, and does not select all
three value tokens until layer 74. That is too late to recover the final
answer.

This establishes that the relevant effect is early sparse-layer
exploration/cluster growth. It is not a one-shot layer-34 merge defect.

Comparison artifact:

```text
e87deb50d43fde4cdb154e11c9dbc262964dca00846b1d9d650fc9b443e1b79a
  bounded-vs-segmented-ticket-value-comparison.json
```

### Correction to the earlier rank language

An earlier offline note described the "needle" as having a full-precision rank
near 4k. That token was the comma at `137485`, not the ticket value. On the
preserved stock layer-34 activation, the exact full-precision ranks of the
three value tokens are:

```text
137499 ("7")   rank 92,886
137500 ("38")  rank 75,542
137501 ("216") rank 82,581
```

Raw FP8 ranks are 93,815 / 85,058 / 91,678. Therefore neither exact FP32
top-2048 nor a more accurate rerank can directly recover the value at that
already-diverged layer.

Two principled but simple alternatives were also rejected offline:

- exact quotas over equal chronological segments or fixed 4k--32k tiles need
  8,928--18,304 total slots to cover all three value tokens;
- selecting max-score contiguous blocks of width 4--256 cannot cover the
  value within a 2,048-token expanded budget.

Artifacts:

```text
b9618f40ac0f74cee2ad9f8a9bf3674392d676a0c7344ee9f08771b03f0310d7
  exact-ticket-rank-layer34-v1/report.json
830a409760a74fd8f152c6145be820ecb240532f6abc589aa860f1d2b45019d6
  exact-ticket-tile-ranks-layer34-v1/report.json
f7faa653946f358caa6cc3ea514be4e504b784da64bcc9a9749583d0cb104340
  exact-ticket-block-expansion-layer34-v1/report.json
```

### What the historical selector was actually doing

The historical 8-bit selector:

1. keeps every candidate in a coarse FP16 score bucket strictly above the
   bucket containing the Kth score;
2. refines only a capacity-limited subset of that threshold bucket.

The old implementation populated that subset through shared atomics into a
4,096-entry buffer. That made a capacity/scan-order accident behave like an
old-history exploration policy.

A new CPU-only proof replaces that accident with an explicit semantic:

1. keep all strictly higher coarse buckets;
2. order threshold-bucket members by logical history position;
3. refine the oldest 4,096 members exactly;
4. emit exactly K entries, with no out-of-bounds write or variable output.

On four preserved v19 production rows, this deterministic policy reconstructs
the captured historical set at:

```text
2035/2048, 2035/2048, 2017/2048, 2022/2048  (97.0%--98.7%)
```

On an independent v20 `bounded_compat` trace, it reconstructs:

```text
2038/2048, 2039/2048, 2018/2048, 2041/2048  (97.1%--99.3%)
```

Evenly stratified and half-oldest/half-stratified boundary pools do not
reproduce the working selections. A 2,048-member pool is far too narrow, and
an 8,192-member pool collapses to exact behavior. The measured 4,096 oldest
boundary members are the specific historical semantic.

Proof and artifacts:

```text
d07442767bf0cdd7f891204f717f7abd7db3d0b6f9ed7402010b3b040627a349
  harness/v20_indexer_boundary_policy_cpu_proof.py

0108d5a21c5e5bb32c302d03d433c74eb843ffbf4a039ed308a1f6a45d1b4bfd
  v19-captured.json

7972dc29209cb8bfedb7413391a943a2f4f35d6611fb65b18a531b37ecef1a45
  v20-bounded-captured.json
```

### Permanent-fix direction

The next causal image will implement this as a distinct
`oldest_boundary` policy, not as `bounded_compat`:

- stable, deterministic threshold-bucket compaction;
- every write bounds checked;
- full bucket population retained for diagnostics;
- exactly 4,096 oldest boundary candidates refined;
- `exact` remains available as the universal mathematical control;
- `bounded_compat` remains diagnostic-only and can be retired after the new
  policy passes.

The decisive gate remains the frozen cold 250k control plus all three frozen
350k failures. If that passes, the same image receives the randomized
50k--475k ladder, repeatability, KLD, prefill/decode, and KV-capacity gates.

This is not a claim that lower proxy-score candidates are mathematically
better. It is a claim, supported by matched trajectories, that the checkpoint
depends on early old-history exploration inside a very coarse score
equivalence class, and that this behavior can be specified safely instead of
being supplied by an overflow bug.
