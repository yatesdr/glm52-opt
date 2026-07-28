# Block-INT8 DMA CPU proof

`test_int8_codec.py` checks the parts of the proposed wire codec that do not
require a GPU:

- one signed payload byte per value and one four-byte scale per 128 values;
- saturation to `[-127, 127]`;
- the pre-BF16 absolute error bound `amax / 254`; and
- identical owner/peer materialization from the same payload.

Run from the repository root:

```bash
python3 harness/int8-dma-proofs/test_int8_codec.py
```

This is a layout/math proof, not a CUDA, collective, model-quality, or
throughput result. Those remain explicit server gates.
