# GLM-5.2 v20 PR #84 shallow BF16 KLD

| Policy | Runs | KLD mean ± sd | Min | Max |
|---|---:|---:|---:|---:|
| `exact` | 3 | 0.15823696 ± 0.00468419 | 0.15539664 | 0.16364348 |
| `oldest_boundary` | 3 | 0.16044075 ± 0.00297924 | 0.15700885 | 0.16236257 |

Paired `oldest_boundary - exact` KLD deltas: -0.00663463, +0.00655420, +0.00669180

Mean paired delta: +0.00220379 (+1.39% of the exact-policy mean).

> This is a 2,048-token no-regression gate. The selector budget is also 2,048, so this cell is not selector-sensitive and does not prove deep-context efficacy.
