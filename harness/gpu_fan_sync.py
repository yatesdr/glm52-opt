#!/usr/bin/env python3
"""Sync all GPU fans as ONE heat exchanger (CN4: 4x desktop RTX PRO 6000, physically stacked).

PHYSICAL MODEL (Derek, 2026-07-26). These are desktop cards stacked directly on top of each other.
They are not four independent coolers -- the fans are series-coupled into a single exchanger that
drags air along the card chain, and the only exhaust is out the end of the stack. Air entering at
the intake end is progressively preheated by every card it passes.

Two things follow, and both matter for the control policy:

  1. A slow fan ANYWHERE is a flow restriction for the WHOLE chain, not a local underperformance.
     An idling upstream fan starves every card downstream of it.
  2. The card at the exhaust end has an IRREDUCIBLE penalty. It breathes air already heated by the
     others, so no fan curve can bring it down to the intake card's temperature. More mass flow
     raises its ceiling; it cannot remove the cumulative preheat.

Measured on CN4 under a 350k prefill, all four at 100% utilization, stock per-card curves:

    gpu0  45C  fan 30%  167W     <-- intake end, fresh air, fan coasting
    gpu1  59C  fan 35%  176W
    gpu2  73C  fan 43%  189W
    gpu3  89C  fan 54%  234W     <-- exhaust end, near the Blackwell slowdown point

The monotonic 45/59/73/89 gradient IS the airflow order: each card adds roughly 12-18C to the air
before passing it on. Use that gradient to identify intake vs exhaust end after any reseating --
it is more reliable than guessing from slot numbers, which need not match CUDA indices under
CUDA_DEVICE_ORDER=PCI_BUS_ID.

Note gpu3 draws 67W more than gpu0 at identical utilization. Most of that gap is thermal leakage,
so it is a self-reinforcing loop: hotter -> leakier -> more power -> hotter. Cooling the exhaust
card should therefore also reduce its power draw; if the gap does not close, the workload is
genuinely uneven rather than thermally penalised.

POLICY. Because the fans form one exchanger, they get ONE speed, computed from the hottest card in
the stack and applied uniformly. Per-card curves are deliberately not used: letting a cool intake
card choose its own low speed is the exact failure being corrected. The floor is high for the same
reason -- under load this stack never has a card that can afford to coast.

WHY ctypes AND NOT pynvml/nvidia-settings.
  * `nvidia-smi` 595.71.05 exposes no fan setter at all.
  * `nvidia-settings` needs a running X server with Coolbits; CN4 is headless.
  * `pynvml` is not installed and this box runs experiments -- no pip install as a side effect.
libnvidia-ml.so.1 already exports nvmlDeviceSetFanSpeed_v2 and nvmlDeviceSetFanControlPolicy, so
bind those directly. No dependencies, no driver or Xorg change.

SAFETY.
  * Read-only by default. It will not touch a fan unless --apply is passed.
  * Manual fan policy is restored to the driver's automatic curve on exit, including SIGINT/SIGTERM
    and unhandled exceptions -- never leave fans pinned by a dead process.
  * PANIC_C forces 100% on every fan and overrides the curve.
  * Fan control is independent of CUDA compute; running this does not disturb a job in flight.

Requires root (NVML rejects fan writes otherwise).

    sudo python3 gpu_fan_sync.py                 # dry run, prints what it would do, changes nothing
    sudo python3 gpu_fan_sync.py --apply         # sync once and exit
    sudo python3 gpu_fan_sync.py --apply --daemon --interval 5
    sudo python3 gpu_fan_sync.py --restore       # hand fans back to the driver and exit
"""
import argparse, ctypes, os, signal, sys, time

# --- NVML constants -----------------------------------------------------------------------------
NVML_TEMPERATURE_GPU = 0
NVML_FAN_POLICY_TEMPERATURE_CONTINUOUS_SW = 0     # driver's own curve
NVML_FAN_POLICY_MANUAL = 1
NVML_SUCCESS = 0

# temp(C) -> fan(%). Applied to EVERY card, keyed on the HOTTEST card in the stack.
#
# Deliberately aggressive, and tuned from measurement rather than from a generic desktop curve.
# With all fans at 100% under a sustained 350k prefill this stack equilibrates at 39/46/53/63C.
# A softer curve (65% at 68C) would let it drift back into the low 80s, because a series chain has
# no card that can afford to coast. Reaching 100% by 75C means any real load runs the fans hard.
#
# The cost of running hard is close to nil here: syncing to 100% dropped stack draw from 766W to
# 685W at identical work, because ~48W of the intake-to-exhaust power gap was thermal leakage.
CURVE = [(0, 50), (45, 60), (55, 75), (65, 90), (75, 100)]
PANIC_C = 85            # any card at/above this -> all fans 100%, curve ignored (belt and braces:
                        # the curve already reaches 100% by 75C)
HYSTERESIS = 3          # only change a fan if the new target differs by more than this


def curve_for(temp):
    pct = CURVE[0][1]
    for t, p in CURVE:
        if temp >= t:
            pct = p
    return pct


class NVML:
    def __init__(self):
        self.lib = None
        for cand in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
            try:
                self.lib = ctypes.CDLL(cand)
                break
            except OSError:
                continue
        if self.lib is None:
            sys.exit("[fan-sync] cannot load libnvidia-ml.so.1 — is the NVIDIA driver installed?")
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")

    def _check(self, rc, what):
        if rc != NVML_SUCCESS:
            raise RuntimeError(f"{what} failed with NVML code {rc}")
        return rc

    def count(self):
        n = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(n)), "nvmlDeviceGetCount_v2")
        return n.value

    def handle(self, i):
        h = ctypes.c_void_p()
        self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i), ctypes.byref(h)),
                    f"nvmlDeviceGetHandleByIndex_v2({i})")
        return h

    def temp(self, h):
        t = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetTemperature(h, ctypes.c_uint(NVML_TEMPERATURE_GPU),
                                                      ctypes.byref(t)), "nvmlDeviceGetTemperature")
        return t.value

    def num_fans(self, h):
        n = ctypes.c_uint()
        rc = self.lib.nvmlDeviceGetNumFans(h, ctypes.byref(n))
        return n.value if rc == NVML_SUCCESS else 0

    def fan_speed(self, h, fan):
        s = ctypes.c_uint()
        rc = self.lib.nvmlDeviceGetFanSpeed_v2(h, ctypes.c_uint(fan), ctypes.byref(s))
        return s.value if rc == NVML_SUCCESS else None

    def set_speed(self, h, fan, pct):
        return self.lib.nvmlDeviceSetFanSpeed_v2(h, ctypes.c_uint(fan), ctypes.c_uint(pct))

    def set_policy(self, h, fan, policy):
        return self.lib.nvmlDeviceSetFanControlPolicy(h, ctypes.c_uint(fan), ctypes.c_uint(policy))

    def shutdown(self):
        try:
            self.lib.nvmlShutdown()
        except Exception:
            pass


class FanSync:
    def __init__(self, nvml, apply_changes):
        self.n = nvml
        self.apply = apply_changes
        self.touched = []          # (handle, fan) we switched to MANUAL — must be restored
        self.last = {}

    def inventory(self):
        gpus = []
        for i in range(self.n.count()):
            h = self.n.handle(i)
            fans = self.n.num_fans(h)
            gpus.append({"idx": i, "h": h, "fans": fans, "temp": self.n.temp(h),
                         "speeds": [self.n.fan_speed(h, f) for f in range(fans)]})
        return gpus

    def restore(self):
        """Hand every fan we touched back to the driver's automatic curve."""
        if not self.touched:
            return
        ok = 0
        for h, fan in self.touched:
            if self.n.set_policy(h, fan, NVML_FAN_POLICY_TEMPERATURE_CONTINUOUS_SW) == NVML_SUCCESS:
                ok += 1
        print(f"[fan-sync] restored automatic fan policy on {ok}/{len(self.touched)} fan(s)",
              flush=True)
        self.touched = []

    def step(self, quiet=False):
        gpus = self.inventory()
        if not gpus:
            print("[fan-sync] no GPUs visible", flush=True)
            return
        hottest = max(g["temp"] for g in gpus)
        panic = hottest >= PANIC_C
        # ONE speed for the whole stack: the fans are a single series-coupled exchanger, so a cool
        # intake card is not entitled to a lower speed -- its fan is part of the exhaust card's
        # cooling path. See the physical model in the module docstring.
        target = 100 if panic else curve_for(hottest)

        if not quiet:
            hot = max(gpus, key=lambda g: g["temp"])["idx"]
            coolest = min(gpus, key=lambda g: g["temp"])
            print(f"[fan-sync] stack {coolest['temp']}..{hottest}C "
                  f"(hottest=gpu{hot}, span {hottest - coolest['temp']}C) -> all fans {target}%"
                  + ("  *** PANIC ***" if panic else ""), flush=True)

        for g in gpus:
            want = target
            for fan in range(g["fans"]):
                cur = g["speeds"][fan]
                curtxt = "?" if cur is None else f"{cur}%"
                if not self.apply:
                    print(f"           gpu{g['idx']} fan{fan} {g['temp']}C {curtxt} "
                          f"-> would set {want}%  (dry run)", flush=True)
                    continue
                prev = self.last.get((g["idx"], fan))
                if prev is not None and abs(prev - want) <= HYSTERESIS:
                    continue
                rc = self.n.set_policy(g["h"], fan, NVML_FAN_POLICY_MANUAL)
                if rc != NVML_SUCCESS:
                    print(f"           gpu{g['idx']} fan{fan} SET POLICY FAILED rc={rc} "
                          f"(need root? card may not allow manual control)", flush=True)
                    continue
                if (g["h"], fan) not in self.touched:
                    self.touched.append((g["h"], fan))
                rc = self.n.set_speed(g["h"], fan, want)
                if rc != NVML_SUCCESS:
                    print(f"           gpu{g['idx']} fan{fan} SET SPEED FAILED rc={rc}", flush=True)
                    continue
                self.last[(g["idx"], fan)] = want
                if not quiet:
                    print(f"           gpu{g['idx']} fan{fan} {g['temp']}C {curtxt} -> {want}%",
                          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually change fan speeds (default: dry run)")
    ap.add_argument("--daemon", action="store_true", help="loop until signalled")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--restore", action="store_true",
                    help="return all fans to the driver's automatic curve and exit")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if (a.apply or a.restore) and os.geteuid() != 0:
        sys.exit("[fan-sync] fan control requires root: re-run under sudo")

    nvml = NVML()
    fs = FanSync(nvml, apply_changes=a.apply)

    if a.restore:
        gpus = fs.inventory()
        for g in gpus:
            for fan in range(g["fans"]):
                fs.touched.append((g["h"], fan))
        fs.restore()
        nvml.shutdown()
        return

    stop = {"flag": False}

    def _sig(_s, _f):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        if not a.daemon:
            fs.step()
            if a.apply:
                print("[fan-sync] one-shot applied. NOTE: fans stay MANUAL until --restore "
                      "or a driver reload.", flush=True)
            return
        print(f"[fan-sync] daemon: every {a.interval}s, curve={CURVE}, panic>={PANIC_C}C",
              flush=True)
        i = 0
        while not stop["flag"]:
            fs.step(quiet=(i % 12 != 0))    # verbose roughly once a minute at 5s
            i += 1
            for _ in range(int(max(1, a.interval * 10))):
                if stop["flag"]:
                    break
                time.sleep(0.1)
    finally:
        # Daemon mode owns the fans only while alive; hand them back on any exit path.
        if a.daemon:
            fs.restore()
        nvml.shutdown()


if __name__ == "__main__":
    main()
