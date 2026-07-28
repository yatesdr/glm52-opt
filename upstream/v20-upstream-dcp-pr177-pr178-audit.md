# v20 upstream DCP PR #177/#178 audit

Date: 2026-07-25  
Scope: source/CPU-math proof only; no CN3/CN4 changes  
Base: `local-inference-lab/vllm` `dev/gilded-gnosis` at
`89b4a98d1ffebb2dda1e1ac5e55238e3a9cfbd58`

## Outcome

PR #178 is a strong, isolated prefill candidate. Its opt-in row-owner merge
preserves the existing rank-major FP32 top-k operation while reducing, for the
production 3072-row TP4/DCP4 chunk:

- per-rank merge rows from 3072 to 768 (4x less top-k work);
- estimated received collective payload from 144 MiB to 54 MiB (62.5% less);
- temporary merge workspace from 432 MiB to 162 MiB (62.5% less).

PR #177 is a real correctness/lifecycle improvement, but its current default
has an unacceptable memory cost for this deployment. With the production
480k/DCP4/MNS16/FP8-RoPE shape and MTP enabled:

- current singleton ping-pong CKV workspace: 421.37 MiB/GPU;
- PR #177 at prefetch depth 0: 421.37 MiB/GPU;
- PR #177 at its default prefetch depth 1: 758.46 MiB/GPU;
- default delta: **+337.09 MiB/GPU**.

The live candidate has only 1,504 KV tokens above the 500k floor. PR #177 must
not be silently added at its default settings; it would consume far more than
that margin. Depth 0 preserves the old allocation size but disables layer
lookahead, defeating the performance purpose. This needs a memory-aware
integration decision or an upstream refinement, not a configuration guess.

## Exact source pins

### PR #177

```text
commit: affff57c0dd482c356e96bc6c774fbd3a3e1e69d
subject: fix(dcp): harden CKV prefetch workspace lifecycle
b12x_mla_sparse.py sha256:
  a2002892614587a737475ef58834b9445a65de764bcbcd646c586a9162a2f2bf
```

This head includes its parent `cb5d9eeb`, which introduces the preallocated,
lane-partitioned CKV ring. The head then hardens storage identity, retirement,
pending-write ordering, and MRV2 cleanup.

### PR #178

```text
commit: b6fe79ded5878269c2e488dd51e2ce074e43cd26
subject: perf(dcp): merge sparse top-k by row owner
sparse_attn_indexer.py sha256:
  b7db1a78b90afc3516e52a9f354bf3b4aafd14e53c546417aafee19479f879ec
```

The route is opt-in through `VLLM_DCP_TOPK_OWNER_MERGE=1`. Non-divisible row
tails fall back to the established replicated merge.

## PR #177 memory proof

Production inputs:

```text
max CKV gather tokens: 480,000
DCP:                   4
max sequences:         16
KV interleave:         1
block size:            64
record bytes:          368 (compact FP8-RoPE NVFP4 record)
DBO ubatches:           1
speculative decoding:  enabled
```

The local capacity is:

```text
ceil((ceil(480000 / 4) + 16 * 1) / 64) * 64 = 120,064 records
```

The existing implementation allocates two halves, each containing one local
staging region and one four-rank gathered region:

```text
2 * (4 + 1) * 120064 * 368 = 441,835,520 B = 421.37 MiB
```

PR #177 allocates one local region plus `depth + 1` four-rank gathered slots,
per execution lane. It reserves two lanes whenever speculative decoding is
configured:

```text
depth 0:
  2 * (1 + 1*4) * 120064 * 368
  = 441,835,520 B = 421.37 MiB

depth 1:
  2 * (1 + 2*4) * 120064 * 368
  = 795,303,936 B = 758.46 MiB
```

This is not an accounting ambiguity: `_CKVPrefetchWorkspacePool.__init__`
eagerly allocates `slot_nbytes * max_slots`.

## PR #178 equivalence and economics proof

The dependency-free proof generated 720 deterministic cases over:

- DCP2 and DCP4;
- 1, 2, and 7 rows per owner;
- top-k widths 2, 7, and 32;
- random and tie-heavy FP32 scores;
- 20 seeds.

For every case it compared:

1. the existing rank-major oracle, which merges all ranks for every row; and
2. the row-owner route, which all-to-alls contiguous row shards, performs the
   same rank-major merge once on the owner, then gathers final indices.

All selected index rows were exactly equal. This is a CPU-math proof of the
partitioning, not a GPU kernel proof; the upstream PR's own tests cover the
real tensor layouts and tail fallback.

For `rows=3072`, `topk=2048`, `DCP=4`, int/FP32 elements:

```text
replicated workspace: 432.01 MiB/rank
owner workspace:      162.00 MiB/rank

replicated received collective bytes: 144.00 MiB/rank
owner received collective bytes:       54.00 MiB/rank

replicated top-k rows: 3072/rank
owner top-k rows:       768/rank
```

These figures cover the merge stage, not the paged-indexer logits kernel or
the CKV gather/attention stage. The compute profiler remains the authority on
how much of end-to-end 55k time the merge consumes.

## Recommendation

1. Keep the current candidate's Gate 2 long-context ladder first. It is the
   necessary correctness baseline and must not be contaminated by another
   variable.
2. Run the prepared one-boot compute profiler on the clean latest-head
   production child.
3. If `indexer_merge` is material, test upstream PR #178 as the single
   functional delta, with its opt-in flag. Require exact top-k comparison and
   cold 8k/55k throughput before considering it for production.
4. Do not add PR #177 at its default depth to the current 500k-floor config.
   Review whether the speculative second lane is genuinely CKV-eligible or
   whether the upstream pool can reserve only the proven active prefill lanes.
   Until that is proven, retaining two lanes is the safe behavior.
5. Keep PR #179 separate. It is a 20-file, ~2k-line mixed-KV/partial-indexer
   topology series, not a minimal optimization to fold into tonight's image.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 harness/v20_upstream_dcp_pr177_pr178_audit.py
```

Proof script:

```text
harness/v20_upstream_dcp_pr177_pr178_audit.py
sha256: 482a5d5574bbac9653582d7a502eec32fac38d390d6664f1826b042a167fc88e
```

Expected first line:

```text
PASS: v20 upstream DCP PR #177/#178 source and math audit
```
