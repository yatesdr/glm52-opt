#!/usr/bin/env python3
"""Measure-and-select the SparkInfer PCIe DMA wire transport within a codec
family (F8_DMA=auto/i8_auto -> race i8 / i8_ring / i8_a2a). Same contract
as nccl_p2p_probe.py: explicit values always win,
per-trial killable subprocesses with hard timeouts, verification before
speed, and fingerprint caching. Probe failure is fail-safe: the launcher
falls back to the uncompressed F8_DMA=0 path, never an unverified codec.

Quality line: this probe only compares explicitly requested INT8-family
modes. These modes can have different requantization counts, so every
candidate must independently pass the same reference-error and rank-
consistency gates before performance is considered.

Verification per trial (all must hold or the mode is FAILED):
  - the DMA all_reduce result agrees with an NCCL BF16 all_reduce of the
    same input within the family's error bound (relative to the reference
    absolute maximum);
  - results are rank-consistent: every rank's output hash is identical;
  - no hang: the parent kills a trial at the timeout.

Selection: lowest MAX-reduced median step time at the largest payload;
ties within 2% resolve to the family's tested-default mode.

Exit codes: 0 = selected mode on stdout; 2 = not applicable; 3+ = probe
machinery failure (caller falls back to F8_DMA=0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

PROBE_REVISION = "wire-mode-probe-v2"
_FAMILIES = {
    "i8": ["i8", "i8_ring", "i8_a2a"],
    "mx": ["mx", "mx_ring", "mx_a2a"],
}
# Tested-default used only as the performance-tie preference. A failed probe
# falls back to uncompressed mode in the launcher.
_FAMILY_DEFAULT = {"i8": "i8_ring", "mx": "mx_ring"}
# Payloads: prefill-allreduce-shaped (rows x hidden, bf16).
_SHAPES = [(1024, 4096), (4096, 4096)]  # 8 MB, 32 MB
_WARMUP = 5
_ITERS = 20
_TRIAL_TIMEOUT_S = 120  # first trial pays the one-time nvcc extension build
_TOTAL_BUDGET_S = 420
_TIE_BAND = 0.02
# Family error bounds vs BF16 reference, relative to max|reference|:
# symmetric INT8 is amax/254 per quant stage; ring re-quantizes per hop.
# 4-rank worst case ~3 stages -> 3/254 ~ 1.2%; allow 2% headroom. MXFP8
# E4M3 payload ~ 2^-3 relative per stage bound is coarser; allow 6%.
_FAMILY_TOL = {"i8": 0.02, "mx": 0.06}

_WORKER = r"""
import hashlib, json, os, sys, time
import torch, torch.distributed as dist

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world)
dev = torch.device("cuda", rank)
mode = os.environ["WIRE_TRIAL_MODE"]
shapes = json.loads(os.environ["WIRE_TRIAL_SHAPES"])
warmup = int(os.environ["WIRE_TRIAL_WARMUP"]); iters = int(os.environ["WIRE_TRIAL_ITERS"])
tol = float(os.environ["WIRE_TRIAL_TOL"])

from sparkinfer.comm.pcie.pcie_dma import PCIeDmaAllReduce

max_bytes = max(r * h for r, h in shapes) * 2
dma = PCIeDmaAllReduce(
    exchange_group=dist.group.WORLD,
    device=dev,
    max_bytes=max_bytes,
    fp8=mode,
)
out = {"wire_mode": dma.wire_mode}
try:
    for rows, hidden in shapes:
        torch.manual_seed(1234)  # identical inputs on every run of a shape
        inp = (torch.randn(rows, hidden, dtype=torch.bfloat16, device=dev)
               * 0.05 * (1 + rank))
        ref = inp.clone(); dist.all_reduce(ref)  # BF16 NCCL reference
        res = dma.all_reduce(inp)
        torch.cuda.synchronize()
        # --- verification ---
        err = (res.float() - ref.float()).abs().max().item()
        bound = tol * ref.float().abs().max().item() + 1e-6
        digest = hashlib.sha256(
            res.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        digest_tensor = torch.tensor(
            list(bytes.fromhex(digest)), device=dev, dtype=torch.uint8)
        digest_tensors = [torch.empty_like(digest_tensor) for _ in range(world)]
        dist.all_gather(digest_tensors, digest_tensor)
        digest_values = {
            bytes(item.cpu().tolist()) for item in digest_tensors
        }
        verified = bool(err <= bound and len(digest_values) == 1)
        # --- timing ---
        dist.barrier(); torch.cuda.synchronize()
        for _ in range(warmup):
            dma.all_reduce(inp)
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            dma.all_reduce(inp)
        torch.cuda.synchronize()
        dt_local = (time.perf_counter() - t0) / iters
        t = torch.tensor([dt_local], device=dev, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        key = f"{rows}x{hidden}"
        out[key] = {"verified": verified, "step_ms": round(float(t) * 1e3, 3),
                    "max_err": round(err, 6), "bound": round(bound, 6)}
finally:
    dma.close()
dist.destroy_process_group()
if rank == 0:
    print("WIRE_RESULT " + json.dumps(out))
sys.stdout.flush()
"""


def _spawn_ranks(worker: str, env: dict, world: int, timeout_s: float) -> dict:
    """Direct multi-process rank spawn (no elastic launcher): deterministic,
    hang-guarded, and preserves each rank's stderr for diagnostics."""
    import signal
    procs = []
    errs = []
    port = 29400 + (os.getpid() % 400)
    for r in range(world):
        e = dict(env)
        e.update({"RANK": str(r), "LOCAL_RANK": str(r),
                  "WORLD_SIZE": str(world),
                  "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
        ef = tempfile.NamedTemporaryFile("w+", suffix=f".r{r}.err", delete=False)
        errs.append(ef)
        procs.append(subprocess.Popen(
            [sys.executable, worker],
            env=e,
            stdout=subprocess.PIPE if r == 0 else subprocess.DEVNULL,
            stderr=ef, text=True))
    deadline = time.monotonic() + timeout_s
    out0 = ""
    try:
        for r, p in enumerate(procs):
            left = max(1.0, deadline - time.monotonic())
            try:
                o, _ = p.communicate(timeout=left)
            except subprocess.TimeoutExpired:
                for q in procs:
                    with __import__("contextlib").suppress(Exception):
                        q.send_signal(signal.SIGKILL)
                for q in procs:
                    with __import__("contextlib").suppress(Exception):
                        q.wait(timeout=5)
                return {"status": "FAILED",
                        "reason": f"timeout>{timeout_s}s (hang)"}
            if r == 0:
                out0 = o or ""
        rcs = [p.returncode for p in procs]
        if any(rc != 0 for rc in rcs):
            tails = []
            for r, ef in enumerate(errs):
                if procs[r].returncode != 0:
                    ef.flush(); ef.seek(0)
                    lines = [ln.strip() for ln in ef.read().splitlines()
                             if ln.strip()][-3:]
                    tails.append({f"rank{r}": lines})
            return {"status": "FAILED", "reason": f"exit codes {rcs}",
                    "stderr": tails[:2]}
        return {"status": "SPAWN_OK", "stdout0": out0}
    finally:
        for ef in errs:
            with __import__("contextlib").suppress(Exception):
                ef.close(); os.unlink(ef.name)


def _fingerprint(family: str, world: int) -> str:
    q = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.width.max",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout
    topo = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout
    blob = "|".join([
        PROBE_REVISION, family, str(world), q.strip(),
        os.environ.get("NCCL_P2P_LEVEL", ""),
        hashlib.sha256(topo.encode()).hexdigest(), json.dumps(_SHAPES),
    ])
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _run_trial(mode: str, family: str, world: int) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_WORKER)
        worker = f.name
    env = dict(os.environ)
    env.update({
        "WIRE_TRIAL_MODE": mode,
        "WIRE_TRIAL_SHAPES": json.dumps(_SHAPES),
        "WIRE_TRIAL_WARMUP": str(_WARMUP),
        "WIRE_TRIAL_ITERS": str(_ITERS),
        "WIRE_TRIAL_TOL": str(_FAMILY_TOL[family]),
    })
    env.pop("SPARKINFER_PCIE_DMA_FP8", None)  # constructor arg is the source
    t0 = time.monotonic()
    try:
        spawned = _spawn_ranks(worker, env, world, _TRIAL_TIMEOUT_S)
    finally:
        os.unlink(worker)
    if spawned["status"] != "SPAWN_OK":
        return spawned
    try:
        line = [ln for ln in spawned["stdout0"].splitlines()
                if ln.startswith("WIRE_RESULT ")][-1]
        res = json.loads(line[len("WIRE_RESULT "):])
    except Exception:
        return {"status": "FAILED", "reason": "unparseable worker output"}
    shape_keys = [f"{r}x{h}" for r, h in _SHAPES]
    if not all(res.get(k, {}).get("verified") for k in shape_keys):
        res["status"] = "FAILED"
        res["reason"] = "verification mismatch"
        return res
    res["status"] = "OK"
    res["elapsed_s"] = round(time.monotonic() - t0, 1)
    return res


def _select(results: dict, family: str) -> str | None:
    big = f"{_SHAPES[-1][0]}x{_SHAPES[-1][1]}"
    ok = {m: r for m, r in results.items() if r.get("status") == "OK"}
    if not ok:
        return None
    best = min(r[big]["step_ms"] for r in ok.values())
    within = [m for m, r in ok.items()
              if r[big]["step_ms"] <= best * (1 + _TIE_BAND)]
    default = _FAMILY_DEFAULT[family]
    if default in within:
        return default
    order = _FAMILIES[family]
    within.sort(key=order.index)
    return within[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=sorted(_FAMILIES), required=True)
    ap.add_argument("--world", type=int,
                    default=int(os.environ.get("WIRE_PROBE_WORLD", "0")) or None)
    ap.add_argument("--cache-dir", default="/root/.cache/wire-mode-probe")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    # Explicit concrete mode ALWAYS wins (even against a mis-wired caller).
    explicit = os.environ.get("F8_DMA", "").strip()
    if explicit and not explicit.endswith("_auto") and explicit.lower() != "auto":
        print(f"wire_mode_probe: F8_DMA={explicit} set explicitly; respecting it",
              file=sys.stderr)
        print(explicit)
        return 0

    world = args.world
    if not world:
        try:
            import re as _re
            topo = subprocess.run(["nvidia-smi", "topo", "-m"],
                                  capture_output=True, text=True, timeout=30,
                                  check=True).stdout
            world = len(_re.findall(r"^GPU\d+", topo, _re.M))
        except Exception as exc:  # noqa: BLE001
            print(f"wire_mode_probe: GPU discovery failed: {exc}", file=sys.stderr)
            return 3
    if world < 2:
        print("wire_mode_probe: fewer than two GPUs; not applicable",
              file=sys.stderr)
        return 2

    cands = _FAMILIES[args.family]
    if args.plan:
        print(f"plan: family={args.family} candidates={cands} world={world} "
              f"shapes={_SHAPES} timeout={_TRIAL_TIMEOUT_S}s "
              f"budget={_TOTAL_BUDGET_S}s tol={_FAMILY_TOL[args.family]} "
              "probe_failure_fallback=0", file=sys.stderr)
        print(_FAMILY_DEFAULT[args.family])
        return 0

    try:
        fp = _fingerprint(args.family, world)
    except Exception as exc:  # noqa: BLE001
        print(f"wire_mode_probe: fingerprint failed: {exc}", file=sys.stderr)
        return 3
    cache = os.path.join(args.cache_dir, f"{fp}.json")
    if not args.refresh and os.path.exists(cache):
        try:
            with open(cache) as f:
                saved = json.load(f)
            selected = saved["selected"]
            if (saved.get("revision") == PROBE_REVISION
                    and selected in _FAMILIES[args.family]):
                print(f"wire_mode_probe: cache hit {fp} -> {selected}",
                      file=sys.stderr)
                print(selected)
                return 0
            print("wire_mode_probe: stale/invalid cache; remeasuring",
                  file=sys.stderr)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            print("wire_mode_probe: unreadable cache; remeasuring",
                  file=sys.stderr)

    # One-time CUDA extension build, OUTSIDE the per-trial timing/timeout:
    # a fresh install pays nvcc once here (bounded), so trial 1 is never
    # unfairly failed by compilation time. Cached thereafter.
    try:
        build = subprocess.run(
            [sys.executable, "-c",
             "from sparkinfer.comm.pcie.pcie_dma import _load_extension; "
             "_load_extension()"],
            capture_output=True, text=True, timeout=600,
        )
        if build.returncode != 0:
            tail = (build.stderr or "").strip().splitlines()[-3:]
            print(f"wire_mode_probe: extension build failed: {tail}",
                  file=sys.stderr)
            return 5
        print("wire_mode_probe: DMA extension ready", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("wire_mode_probe: extension build timeout (600s)", file=sys.stderr)
        return 5

    results: dict[str, dict] = {}
    t0 = time.monotonic()
    for mode in cands:
        if time.monotonic() - t0 > _TOTAL_BUDGET_S:
            print(f"wire_mode_probe: total budget exceeded; stopping",
                  file=sys.stderr)
            break
        print(f"wire_mode_probe: trial {mode} ...", file=sys.stderr)
        results[mode] = _run_trial(mode, args.family, world)
        print(f"wire_mode_probe:   {mode}: {json.dumps(results[mode])[:220]}",
              file=sys.stderr)

    selected = _select(results, args.family)
    if selected is None:
        print("wire_mode_probe: no mode verified; caller must use "
              "uncompressed F8_DMA=0", file=sys.stderr)
        return 4

    os.makedirs(args.cache_dir, exist_ok=True)
    with open(cache, "w") as f:
        json.dump({"fingerprint": fp, "selected": selected, "results": results,
                   "family": args.family, "revision": PROBE_REVISION}, f, indent=1)
    print(f"wire_mode_probe: selected {selected} (cached {fp})", file=sys.stderr)
    print(selected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
