## Clean 480k production candidate passes the deepest cold gate

The selector fix now has a trace-free, diagnostic-free production-candidate
proof.

Candidate:

- base: clean `5517197/be0edca` release image;
- overlays: only SparkInfer PR #80 (PCIe DMA output lifetime) and draft PR #82
  (`bounded_compat`);
- image ID:
  `sha256:259098b0f5a83c775ff09f8979f3fec8982b88d8d58434dc4c9284f2b4e68905`;
- TP4/DCP4/MTP3, MNBT 3072, max model length 480000;
- NVFP4 MLA KV, FP8 RoPE, `i8_ring`, CKV prefetch depth zero;
- KV pool: 500992 tokens.

Deepest cold result:

| Target | Rendered tokens | Cached | Completion | Finish | Content |
|---:|---:|---:|---:|---|---|
| 475k | 466493 | 0 | 78 | `stop` | `738216` |

Arithmetic and coherence side checks passed. The container remained healthy,
restart count stayed zero, and the log had zero illegal-access, cuBLAS-error,
EngineDead, OOM, Xid, assertion, or worker-died signatures.

Evidence SHA-256:

- `cell-475k.json`:
  `f1ca6da58dd7f6528be04b6fd200315a972735ec83afa5ad851898b5310c6e23`
- `summary.json`:
  `4b9bba777e311943f82b668f18146765f04ef5829c11e4d34f82a4f9c32ba987`
- `run.log`:
  `f924d418a18d233a901617fe082c40504f07292ed002ace9c9fe72716bf18127`

Together with the frozen 250k control, 3/3 recovered 350k failures, and the
independent cold 49k/147k/294k/441k generalization ladder, this closes the
clean-image quality proof for the explicit compatibility policy. Performance
and KV-memory qualification remain separate promotion gates.
