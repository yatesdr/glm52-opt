### Status: causal fix still passes; follow-up stable-radix cleanup rejected

The smaller `oldest_boundary` implementation remains the known-good causal
candidate:

- image manifest:
  `sha256:2463080ecbdd0109244b10bd1266fb7acc74e803c0d1a1a1252dfb3d6837b6fc`
- selector source:
  `4d89638440a1bab62d632a4ebbba3de2cdbda8bc5bcd5c8d559b940d8c45e42e`
- frozen cold gate: 250k control plus 3/3 previously failing 350k prompts
  returned exact finalized `738216`;
- 480k service boot: 500,992 KV tokens, zero restarts.

I subsequently tried to make every full-key radix pass use the same stable
tile/warp compaction. That cleanup (`c1ffc029`, followed by output-allocation
change `87b61b04`) is **rejected** and is not part of the proposed fix.

The server boot failed asynchronously during profile warmup. A new model-free
probe reproduced the failure at the exact server-profile geometry:

```text
rows:       3072
lengths:    1..3072
topk:       2048
block_q/k:  32 / 256
```

Results:

| Image | Result |
|---|---|
| causal `2463080e…` | PASS; valid counts and all indices in bounds |
| stable-radix `f0a5ab36…` | `CUDA_ERROR_INVALID_ADDRESS_SPACE` |
| stable-radix + allocator `6f81f1c3…` | same failure |

NVIDIA memcheck observed invalid global reads in the tiled selector kernel,
concentrated near ramp rows 2,988--2,997. The intermediate `c1ffc029` image
already fails, so the later allocator commit is not the cause.

Evidence pins:

```text
42769970adad6ee05c9ea86794464cb7011932122d8d4d9d46e0afc4d23420ec
  v20_oldest_boundary_warmup_shape_probe.py

6d32434593a932026ad16fdde2aded4f5e1b45c584cadc52452edf4397e6b23d
  causal-image warmup result

7c015a03d0573846a037b7bb7fcea4a6810b74e7c50f6fe6dc7d443327a90ddb
  stable-radix warmup failure log
```

The clean production branch is therefore pinned at `d4385494` /
implementation `7e47e9ac`; neither rejected cleanup commit will be included.
The 3,072-row probe is now a mandatory pre-boot gate. Full randomized
long-context, KLD, throughput, and capacity qualification of the smaller
causal implementation remains pending.
