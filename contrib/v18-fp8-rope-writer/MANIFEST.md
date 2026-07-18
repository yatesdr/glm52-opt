# Bundle manifest

Status: source/CPU gated and SM120 compile/canary accepted; direct reader
round-trip and full server acceptance pending.

| File | Purpose |
|---|---|
| `overlays/b12x/attention/mla/fp8_rope_writer.py` | Adapted v1.3 CuTeDSL writer and exact v18 torch-op registration |
| `vllm-loader.patch` | Prefer the bundled source writer, retain `.so` only as a missing-module fallback, then validate the op |
| `checks/test_writer_contract.py` | Nine CPU/source contract checks, including the mirrored-reader byte pin |
| `checks/gpu_writer_smoke.py` | In-image first-call compile, layout, skip, and canary gate |
| `checks/gpu_writer_reader_roundtrip.py` | Source-writer to actual v18 B12X decode-reader numerical gate |
| `checks/check_pins.py` | md5 verifier for immutable inputs and bundle artifacts |
| `md5-manifest.txt` | Input and output byte pins |
| `README.md` | Adaptation evidence, installation, and acceptance sequence |

The bundle intentionally does not include a compose file. It stacks onto the
current 480k v18 contributions overlay; no 64k or other short-context compose
is part of the requested serving contract.
