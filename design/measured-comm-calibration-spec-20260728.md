# Spec: measured communication calibration (`auto` contract for topology and wire)

Date: 2026-07-28
Author: Fable (per Derek)
Executor: Sol — queued behind the promotion gates (decode confirmation,
restart/repeatability) and KLD arms C/D.
Status: P2P-level tooling built and dry-validated; wire-protocol extension
designed, to be implemented inside the SparkInfer #81 probe framework.

## 1. The `auto` contract (uniform across knobs)

For every calibrated knob:

- **Explicit value** (anything other than `auto`/unset) → respected verbatim,
  no measurement, logged as "explicit". Enforced INSIDE each tool, not only
  in the launcher, so a mis-wired wrapper cannot override an operator.
- **`auto` or unset** → measured selection with fingerprint caching; on any
  probe failure, fall back one rung (measured → static derivation → stack
  defaults), each step logged.
- **No hangs by construction**: every trial runs in a killable subprocess
  with a hard per-trial timeout; a global budget bounds total probe time; a
  hung trial is a recorded FAILED result, never a stalled boot.
- **No silent quality changes**: `auto` may only choose among options that
  are quality-identical. Lossy codecs are never entered automatically.

## 2. Knob 1 — `NCCL_P2P_LEVEL=auto` (topology)  [BUILT]

- `docker/derive_nccl_p2p_level.py`: static derivation from
  `nvidia-smi topo -m` (ANSI-safe parser). Policy: worst pairwise link
  class, capped at PXB (P2P through switches yes, through/above host
  bridges no — the wedge paths); `--permissive` lifts the cap. 7/7 fixture
  self-tests; live-validated on CN4 (NODE-worst → PXB) and CN3 (PHB-worst →
  PXB), both matching the hand-validated production value.
- `docker/nccl_p2p_probe.py`: measured selection. Candidates = static cap up
  to the literal worst class (+ one level below the cap at live-test time);
  per candidate: fresh subprocess rank group, `NCCL_P2P_LEVEL` exported
  before communicator init, 5 warmup + 20 timed all_gather/all_reduce at
  DCP-shaped sizes (4/64 MB), bit-verification every iteration (rank-tagged
  patterns element-exact; ramp all_reduce vs closed form), 60 s per-trial
  timeout, 180 s total budget, fastest verified wins, ties within 2% go
  conservative, fingerprint-cached (topo hash + GPU/driver + PCIe gen/width
  + revision). Explicit-respect guard verified on CN4 (explicit/unset/auto
  all behave per contract).
- REMAINING: one live probe run on idle CN4 (not under Sol's experiments),
  then launcher wiring (`auto` → probe → static → default).

## 3. Knob 2 — wire protocol `F8_DMA` auto tokens  [DESIGN]

Token semantics (the quality line):

| Token | Meaning |
|---|---|
| `off`, `i8_ring`, `mx_a2a`, … | explicit; respected verbatim as today |
| `auto` | race the LOSSLESS options only: raw-BF16 SparkInfer DMA vs NCCL path (the crossover #81 already measures), under the selected P2P level |
| `i8_auto` | operator has opted into the INT8 codec; race `i8` / `i8_ring` / `i8_a2a` transports — same encoding, same amax/254 bound, quality-identical by construction — pick the fastest verified |
| `mx_auto` | same for the MX family |

Implementation home: the SparkInfer #81 calibration probe (it already loads
the wire paths and owns policy caching); this spec's contribution is the
trial contract from §2 (killable subprocess, timeout, verify-or-fail,
conservative tiebreak, explicit-wins) plus the token grammar.

Verification per trial: all ranks materialize byte-identical decoded
payloads (the rank-consistency contract), and decoded-vs-source error is
within the codec's documented bound; any mismatch ⇒ FAILED regardless of
speed.

Ordering and caching: P2P level resolves FIRST (transport perf depends on
it), then the wire race runs under the winner; one fingerprint covers both
decisions plus codec family and probe revision.

## 4. Knob 3 — `DCP_TOPK_OWNER_MERGE` (kcramp's finding)  [QUEUED]

Unverified claim, real results: owner-merge=1 regresses prefill on his
system while upstream measured +10% on TP8/DCP8. Topology-dependent; today
it is static-eligibility (unprobed). Plan: reproduce on CN4 (1-vs-0 prefill
A/B, n=3, production posture), understand the mechanism (#79/#178 owner
exchange vs fabric), then either add it to the measured set with the same
trial contract or publish a topology-conditional recommendation upstream.

## 5. Validation gates before any of this ships in an image

1. Live probe run on idle CN4: selected level must match the validated PXB
   (or beat it with verified evidence); archive the decision JSON.
2. Negative tests on hardware: explicit value respected; probe killed at
   per-trial timeout leaves a clean GPU state (no stuck processes, verified
   via nvidia-smi); cache hit performs zero GPU work; corrupted-cache file
   falls back to measurement.
3. Fixture corpus expansion: solicit real `nvidia-smi topo -m` outputs from
   the Discord community (destroyed, kcramp, timricese, ufear — varied
   fabrics: single/dual switch, switchless, dual-root, PCIe 3/4/5) and add
   each as a static-derivation fixture. The measured stage is the safety
   net; the static stage should still be right on real matrices.
4. Boot-time budget: probe adds ≤ ~3 min worst case on first boot per
   machine, ~0 thereafter (cache). If that is unacceptable for some
   deployments, `NCCL_P2P_LEVEL=PXB` in the compose opts out entirely — the
   explicit-wins contract is the escape hatch.

## 6. Bake plan

Next image iteration (gated on arms C/D + promotion gates), one build:
1. destroyed's quality-first MXFP8 membership defaults (pending arm C/D
   confirmation on our rig);
2. `derive_nccl_p2p_level.py` + `nccl_p2p_probe.py` at /usr/local/bin with
   the launcher `auto` hook;
3. wire-protocol auto tokens if the #81 extension lands in time; otherwise
   ship the token grammar documented and explicit-only, extension follows.
Evidence and pins recorded per house convention; the calibration decision
JSONs join the boot logs as first-class evidence artifacts.
