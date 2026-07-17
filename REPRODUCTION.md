# Reproduction guide

Target: reproduce the 1,696 tok/s cold 55k prefill (packed-CKV) and the
964 tok/s stage-1 result on a 4x RTX PRO 6000 (SM120) box, TP4/DCP4.

## 0. Prerequisites

- 4x 96 GB SM120 GPUs (RTX PRO 6000 Blackwell class). PCIe Gen3 works
  (that's what these numbers were measured on); Gen4/Gen5 should do
  better. GPU P2P functional (`torch.cuda.can_device_access_peer` true
  across pairs). No NVLink needed.
- Docker + NVIDIA container toolkit.
- The serving image: `davidyoung/vllm-glm52-nvfp4-nf3-hybrid-lowbit-kv:v1.3`
  (Docker Hub). Everything here overlays that image via bind mounts —
  no image rebuild required.
- The checkpoint: `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` (~341 GB).
  Use the pristine `config.json` as shipped.
- A sibling clone of `davidsyoung/vllm-glm52` (the v1.4 decode overlays
  are mounted straight from that repo):

```bash
git clone https://github.com/yatesdr/glm52-opt.git
git clone https://github.com/davidsyoung/vllm-glm52.git   # sibling dir
```

## 1. Verify patch bases (do not skip)

The overlays in `patches/` were built against exact bytes of the v1.3
image + v1.4 overlay set. Before mounting, extract and md5 the in-image
files and compare with `patches/stage3-packed-ckv/md5-manifest.txt`
("Pinned input bytes" section). If your image tag/digest differs, diff
before trusting anything.

## 2. Boot — stage 1 only (fp8 gather + RS ring, +51%)

```bash
cd glm52-opt/compose
MODEL_DIR=/path/to/GLM-5.2-hybrid \
MAXLEN=64000 BLOCKS=400 MNBT=3072 \
FP8_MODE=ring GATHER_FP8=1 RS_RING=1 PROF=1 \
docker compose -f docker-compose.v14eq.yml up -d
```

Expected activation lines in `docker logs`:
- `Configured b12x PCIe crossovers` (DMA alive)
- `B12X_DCP_GATHER_FP8=1: fp8 DCP query all_gather armed` (per rank)
- `B12X_DCP_RS_RING=1: DCP output reduce-scatter on CE DMA ring` at first
  large prefill (lazy; look after the first request)
- `GPU KV cache size: 102,400 tokens`

## 3. Boot — packed CKV (the 1,696 config)

```bash
MODEL_DIR=/path/to/GLM-5.2-hybrid \
CKV_TRANSPORT=ckv \
MAXLEN=64000 BLOCKS=400 MNBT=3072 FP8_MODE=ring PROF=1 \
docker compose -f docker-compose.ckv.yml up -d
```

Expected: `Packed-CKV transport armed: max_model_len=64000 max_seqs=8
packed_blocks/rank=2000 ...` on every rank at startup. The fp8 query
staging and RS ring must NOT arm (hard-disarmed in ckv mode — their
absence is correct).

Stage-3 capacity rule: `max_num_seqs x ceil(max_model_len/256) <= 2403`.
64k x 8 fits. 480k does NOT (by design — startup refuses; the phase-2
work in `design/` addresses full context).

## 4. Bench + acceptance

```bash
# cold prefill (server-side tok/s; random first block defeats prefix cache)
python3 harness/prefill_bench.py --tokens 8000  --label my_8k
python3 harness/prefill_bench.py --tokens 55000 --label my_55k

# quality gates
python3 harness/quality_gate.py
python3 harness/quality_gate_fp8_ext.py --depth-tokens 60000
```

Acceptance signature (in `docker logs`, after ~15 chunks with PROF=1) for
the CKV boot — one line per rank:

```
routes: query=0 ckv=1200 | ... missing_blocks=0 ... local_heads=16
gather: n=0 ... rs: n=0 ... ckv_ag: n=1200 mean~0.6ms
```

Parity check (recommended): boot `docker-compose.ckv.yml` with
`CKV_TRANSPORT` unset — throughput must match your stage-1 numbers ±3%
and the summary must show `routes: query=N ckv=0`. That proves the patch
is inert when disabled.

Reference numbers to compare against: see `RESULTS.md` §1–3. Report
prefix-cache metric deltas with any number you publish
(`vllm:prefix_cache_queries/hits`, `vllm:prompt_tokens_cached`).

## 5. Known limits of this drop

- The boot recipes above are the **64k test profile** — that is the
  configuration these steps reproduce. Full 480k context (phase 2) is
  **confirmed, gated, and shipped**: 1,509 tok/s @ 55k and 1,126 tok/s on
  a cold 463k request, with a 599,040-token pool. Its overlays and gate
  checks are in `patches/phase2-fullcontext/` and the measurements are in
  `RESULTS.md` §8. What this guide does **not** yet give you is a
  turnkey 480k boot: the escrow/probe memory contract, `BLOCKS=2340`, and
  the tiered-KV connector have no compose profile here. A prebuilt image
  and one-line compose for the shipped configuration are in progress.
- Geometry is strict v1: TP4/DCP4, interleave 1, `nvfp4_ds_mla`,
  MNBT 3072, topk 2048, 16 local heads. The CKV startup asserts name any
  mismatch and tell you to fall back to `query`.
- Decode paths are intentionally untouched; MTP/decode numbers ride the
  base image + v1.4 overlays.

## 6. Troubleshooting

- Boot hangs at first prefill with mixed collective log lines → you are
  running a pre-fix `common.py`; the group-vote init in this repo's copy
  is mandatory (see `RESULTS.md` §6).
- `CompilationError ... Cannot access global variable` from a Triton
  kernel → stale pre-fix `b12x_mla_sparse.py`; use this repo's copy
  (`design/field-fix1-triton-constexpr.md`).
- OOM at first prefill at large max-len: read `RESULTS.md` §5 before
  reaching for block cuts — they will not buy you transient float.
