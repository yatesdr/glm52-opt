# Packed-CKV phase-2 Gate B CPU/source proofs

These five dependency-free scripts implement the Gate B authorized by
`../glm-5/sol-phase2-gateA-verdict.md`. They model the approved contracts;
they do not modify or import the vLLM/B12X integration and perform no server
work.

Run with any isolated Python 3 interpreter:

```bash
python test_pool_remap.py
python test_active_layout.py
python test_route_determinism.py
python test_escrow_state.py
python test_profiler_state.py
```

Coverage:

1. `test_pool_remap.py` creates 2,530 request-major logical pages, above
   `B_active=2403`, backed by only 17 owner-specific physical pages/rank.
   It proves stable remapping through heavy prefix-style aliasing and holes,
   comparing every accepted 368-byte record with a direct owner-cache read.
2. `test_active_layout.py` independently constructs the Stage-3
   `[owner,B,64,368]` reference and proves that two NCCL-style rank
   concatenations (records and validity) produce the exact same bytes for
   uneven tails and owner holes.
3. `test_route_determinism.py` generates 1,008 randomized/boundary metadata
   cases. Four rank-local selectors with deliberately different owner-table
   payloads must choose identical routes and collective byte lengths. It also
   proves the Gate A rider-B warning fires exactly once per process on first
   pool-route activation.
4. `test_escrow_state.py` proves the 192 MiB direct-CUDA escrow lifecycle:
   default-off zero activity, arm-time before/after driver-free logging,
   all-rank allocation vote, held-window behavior, exactly-one free, A/B
   group-min probes, cleanup, and fatal allocation/gate/order failures.
5. `test_profiler_state.py` uses fixed fake-event pools for nested F/S layers
   spanning query and CKV routes. Exclusive time exactly reproduces the
   layer total; the 5% perturbation bar, one-TP-AR-per-region rule, overflow,
   nesting failure, one-shot sync, and default-off zero allocation are all
   self-tested.

Gate A riders A and B are therefore pinned in the Gate B proofs. Rider C is
an operational acceptance condition: the measured DCP1 prefill result remains
non-shipping until its own quality gates and decode-C1 record pass.

`captured-output.txt` records the clean AST, pyflakes, and harness run. Gate C
integration remains gated on Fable's Gate B verdict.
