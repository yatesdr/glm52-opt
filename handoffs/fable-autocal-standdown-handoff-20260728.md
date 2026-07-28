# Stand-down handoff: v20 release-autocal state (Fable -> Sol)

Date: 2026-07-28. Derek has transferred execution to Sol; Fable reviews when
done. This file is the exact pickup state — nothing below has been touched
after the stand-down order.

## Where the smoke test stopped

The wire-mode race is ONE known one-line fix away from its first clean run.
Last CN3 result: all three i8 trials ran the real DMA all_reduce
successfully and failed only in MY verification step:
`TypeError: Got unsupported ScalarType BFloat16` — numpy cannot ingest BF16.

**Fix NOT yet applied anywhere** (my patch command was interrupted). In
`docker/wire_mode_probe.py`, worker section, replace:

```python
        digest = hashlib.sha256(res.cpu().numpy().tobytes()).hexdigest()
```
with:
```python
        digest = hashlib.sha256(
            res.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
```

Then: rerun the race standalone on CN3 (command in §4), expect three OK
trials and a selection; then rebake the release image with the fixed probe.

## Artifact inventory (all in glm52-opt unless noted)

| Artifact | State |
|---|---|
| `docker/derive_nccl_p2p_level.py` | DONE. 7/7 fixtures; live-validated CN4 (NODE→PXB) + CN3 (PHB→PXB); explicit-respect guard field-proven (caught v19 image's baked SYS) |
| `docker/nccl_p2p_probe.py` | DONE. First fully-verified measured run on CN3: PXB 4.77 GB/s > PHB 4.71, selected+cached. Two bugs fixed en route: elastic launcher replaced with direct rank spawn; empty `CUDA_VISIBLE_DEVICES` export hid all GPUs |
| `docker/wire_mode_probe.py` | ONE FIX PENDING (above). DMA path + timing + spawn machinery proven; 600s pre-build step included |
| `docker/serve-with-autocal.sh` | v5: wraps direct `vllm serve "$@"`; posture from real args (--kv-cache-dtype) + KV_FP8_ROPE; wire winner exported as all three env spellings; dynamic-KV default-on in posture; shm mmap cleanup; explicit-always-wins throughout |
| `compose/glm52-v20-release-autocal-20260728.yaml` (+ copy in ~/Downloads) | Full PROMOTION-grade flag set with hand-tuned comm knobs replaced by auto; destroyed's quality-first quant membership (capacity alternate in comment); DRAM offload tier; autoheal service; digest-pinned to the release image |
| Image `ghcr.io/yatesdr/glm52-serve@sha256:4ea6bcf6…` (tag …-release-autocal-20260728) | Pushed. Contains v5 wrapper + probes with the wire BF16 bug — REBAKE after the fix. Entrypoint/CMD verified clean |
| Earlier images v1 9bc5fcc3 / v2 56a0c7c0 / v3 6968b69b / v4 bd4d1896 | Superseded iterations; keep or prune at will |

## CN3 state at stand-down

- `glm52-prod-candidate` (v19) STOPPED (Derek-authorized). Restore =
  `docker start glm52-prod-candidate`. Its unexplained 04:20Z start remains
  unattributed (no cron/timer; two live derek sessions; possibly autoheal
  from a prior stack or an operator).
- No serving container running; GPUs idle. Release image + old compose in
  /tmp; probe scratch in /tmp (probes, worker debris r*.out/err, freshcache,
  wirecache).
- Probe caches: /tmp/freshcache has a cached NCCL decision (PXB) for the
  container fingerprint.

## Remaining sequence (as Derek directed it)

1. Apply the BF16 hash fix; standalone wire race on CN3 must pass 3/3
   verified (command: run `/usr/local/bin/wire_mode_probe.py --family i8
   --refresh` in the release image with `--gpus all --ipc=host`).
2. Rebake release image with fixed probe; push; repin compose digest.
3. Deploy `compose/glm52-v20-release-autocal-20260728.yaml` on CN3.
4. Detection report from the boot: `[autocal]` lines (posture echo, measured
   NCCL level, wire race winner), `per_token_scale=true, version=3` JIT
   metadata, quantization membership layer list.
5. Bench: short prefill + decode (harness/v20_release_smoke.py staged at
   cn3:/tmp — uses /tokenize sizing + raw /v1/completions), then needle
   @350k (same script, target 350k, expects 738216).
6. Known open judgment calls for the release compose: NCCL_PROTO/LD_PRELOAD
   interplay with measured P2P level (probe measures under the same env, so
   the result is honest, but SYS-vs-PXB under the custom NCCL deserves one
   glance at the probe numbers); v19-lineage flags on the v20 image (boot
   will fail loudly on any unsupported arg — none expected, not yet proven);
   quality-first vs capacity quant profile (Derek chose quality-first as
   default; alternate documented in the compose comment).

## Watch-outs learned tonight (so they are not relearned)

- sshpass from this Mac needs `-o PubkeyAuthentication=no` (stray keyfile
  stalls sessions) and an absolute path in backgrounded shells.
- Rejected/interrupted ssh command chains can still complete remotely —
  check remote effects after any interrupt.
- torchrun swallows worker tracebacks; the probes now direct-spawn ranks
  and preserve per-rank stderr for exactly this reason.
- Probe trials must never run beside a resident model (context alloc OOMs
  at ~850 MiB free); boot-time (pre-load) is their designed home.
