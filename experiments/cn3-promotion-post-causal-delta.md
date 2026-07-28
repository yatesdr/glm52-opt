# CN3 promotion contract after the safe-query causal window

Status: staged requirements; candidate-specific pins pending the authorized
CN4 window

This document prevents the obsolete 2026-07-22 combined spec from being reused
while the final candidate is still under proof. It contains no authorization
to touch CN3.

## Outcome split

1. If accurate-precise reproduces the PEDANTIC post-FP8 reference at 54/54,
   preserves regular-mode bytes at 54/54, passes handle restoration, and the
   PEDANTIC discriminator returns 6/6 exact fixed-seed answers, build the CN4
   production candidate from the scoped accurate-reduction source.
2. If the numeric equivalence gate fails, revise the accurate implementation.
   Do not boot or promote it on the strength of tolerance tests.
3. If the PEDANTIC discriminator fails any decisive cell, the operator-level
   change is not sufficient end to end. Preserve evidence and reopen root
   cause; neither the discriminator nor accurate branch may advance.

The global-PEDANTIC binary-rewind image is proof-only in every outcome.

## Gates required before CN3

### Source and build

- current upstream DCP-final base, exact commits recorded;
- scoped accurate-reduction commit and retained `#171` recorded separately;
- no debug descriptor instrumentation or source mounts;
- mandatory post-image/pre-push fingerprint gate:
  `design/safe-query-build-fingerprint-gate.md`;
- unexplained post-FP8 change is a build failure; any intentional change needs
  an exact-commit, exact-fingerprint, independently reviewed waiver;
- image digest copied to the controlled registry before any promotion boot.

### CN4 production qualification

- GPU KV pool **at least 500,000 tokens** at max model length 480,000;
- no GMU increase to rescue a pool miss;
- production decode CUDA graphs complete at every configured size;
- cold final-content needle answers are exactly `738216`, not substring
  matches and not reasoning-only retrieval;
- fixed-seed 100k ×3 and 150k ×3 are 6/6 exact;
- cold ladder includes 50k, 250k, 350k, and 475k; `finish_reason=length` is
  inconclusive and must be repeated with adequate budget;
- every row records content, reasoning fields, finish reason, prompt and
  completion tokens, cached tokens, request hash, and image/container identity;
- prefill and decode stay within the approved floors against the matched CN4
  baseline; precision cost is reported independently from model-quality gain;
- 16-request stress, MTP acceptance, NVMe bounded eviction, persistence, and
  promotion pass on the same release candidate;
- zero restart, OOM, illegal access, cuBLAS failure, assertion, Xid, or worker
  death.

### CN3 promotion

- CN4 image digest and Compose are immutable inputs;
- CN3 preflight is read-only until the maintenance window is explicitly
  authorized;
- first boot uses the same GMU, graph cap, MNS, MTP, DCP, wire, DRAM, and NVMe
  settings qualified on CN4;
- pool floor remains 500,000;
- run a compact exact-finalization smoke, cold prefill/decode cells, and
  offload health checks before routing traffic;
- rollback identity and command are recorded before stopping the incumbent;
- promotion requires stable container identity and a final fatal-signature
  audit.

## Pins to fill after the causal window

- upstream vLLM/SparkInfer/FlashInfer commits;
- accurate source and integration commits;
- built image tag, manifest digest, and stable-libtorch SHA-256;
- fingerprint probe, reference, comparator, and report SHA-256 values;
- final Compose SHA-256;
- CN4 evidence directory and measured pool/performance/quality verdicts.
