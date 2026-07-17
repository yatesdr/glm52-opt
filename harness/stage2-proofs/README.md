# Packed-CKV Stage 2 CPU proofs

These three dependency-free scripts are the correctness gate authorized by
`../glm-5/sol-packed-ckv-gate1-verdict.md`. They model only the approved v1
mechanism; they do not modify or import the vLLM integration.

Run them with any Python 3 interpreter:

```bash
python3 sol-packed-ckv-stage2/test_ring_schedule.py
python3 sol-packed-ckv-stage2/test_ownership_inversion.py
python3 sol-packed-ckv-stage2/test_remap_reads.py
```

Coverage:

1. `test_ring_schedule.py` proves byte-exact four-rank concatenation for
   arbitrary uneven payloads. It repeats the schedule, then reuses the same
   fixed-capacity slots for a smaller call and proves stale tails are excluded.
2. `test_ownership_inversion.py` compares query-AG/local-KV partial attention
   plus stable LSE merge against local-Q/gathered-KV attention. It uses two
   requests, uneven striped tails, noncontiguous physical blocks, and invalid
   top-k IDs.
3. `test_remap_reads.py` packs the exact `[4,B,64,368]` request-major layout,
   remaps boundary-heavy global IDs, and compares every valid byte record with
   its direct owner-shard read. Owner-specific holes, partial tails, invalid
   IDs, and noncontiguous block tables are included.

`captured-output.txt` records a clean local run. Stage 3 remains gated.

Gate 1 rider A is covered by the repeated schedule and the
smaller-after-larger fixed-capacity call. Riders B and C remain explicit
Stage 3 work: startup assertions must validate the temporary gathered-cache
kernel geometry and profiling must expose the accepted 48 MiB staging copy.
