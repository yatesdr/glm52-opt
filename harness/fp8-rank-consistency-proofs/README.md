# FP8 all-reduce rank-consistency proof

Run:

```bash
python3 harness/fp8-rank-consistency-proofs/test_owner_roundtrip.py
```

The proof models the final dissemination semantics shared by the three FP8
PCIe-DMA modes.  It demonstrates that retaining the pre-wire owner shard gives
every rank a different completed result, while locally materializing the exact
forwarded payload gives every rank the same result.

It intentionally does not claim that E4M3 is sufficiently accurate for the
model.  That remains a four-GPU collective gate followed by the deep-context
server gate.

