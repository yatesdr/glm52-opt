### Independent cold ladder: PASS through 441,964 tokens

The same `bounded_compat` discriminator process passed a second,
newly-randomized cold ladder:

| Target | Rendered tokens | Cached | Completion | Result |
|---:|---:|---:|---:|---|
| 50k | 49,100 | 0 | 91 | EXACT |
| 150k | 147,275 | 0 | 107 | EXACT |
| 300k | 294,619 | 0 | 119 | EXACT |
| 450k | 441,964 | 0 | 80 | EXACT |

Every response finalized with `finish_reason=stop` and exact final content
`738216`. The container remained healthy with restart count zero and zero
illegal-access, cuBLAS, EngineDead, OOM, traceback, assertion, or worker-died
signatures.

Evidence SHA-256:

- summary:
  `321f68d7fd44b916c80b570af90d60a2dde65eaa14aeacc5a53d729893d70b54`;
- run log:
  `e789dea27361d650b60870f0a1bae8f4e63312932894c4a0a0951dc807bf1437`.

The 441,964-token row is the highest safe point under this diagnostic boot's
460,000-token admission limit. The remaining promotion item is a clean
480,000-token boot and the established 475k cell, followed by performance and
KV-memory gates. Root cause and the quality-restoring selector policy are no
longer speculative.
