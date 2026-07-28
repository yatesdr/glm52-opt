The digest-pinned review image and community A/B package are now public.

Image:

```text
ghcr.io/yatesdr/glm52-serve:gilded-gnosis-v20-oldest-boundary-pr84-20260727
ghcr.io/yatesdr/glm52-serve@sha256:43e5a48781ee5cf40a92cc494749b21306b72280bd1a875721a45422323f2599
```

Public package:

https://github.com/users/yatesdr/packages/container/package/glm52-serve

Community validation spec, same-image `exact`/`oldest_boundary` Compose,
frozen long-context reproducers, and fail-closed KLD wrapper:

https://gist.github.com/yatesdr/a2e84aa3171ee0b355649704f04f96a8

The matched three-run BF16-reference KLD comparison completed:

| Policy | Runs | Mean `KL(BF16 reference || candidate)` ± SD | Min | Max |
|---|---:|---:|---:|---:|
| `exact` | 3 | 0.15823696 ± 0.00468419 | 0.15539664 | 0.16364348 |
| `oldest_boundary` | 3 | 0.16044075 ± 0.00297924 | 0.15700885 | 0.16236257 |

All six runs used the same image, checkpoint, 368-byte NVFP4+FP8-RoPE KV
format, TP4/DCP1 eager posture, and the published 2,048-token BF16 logits.
Every run matched the reference token IDs and emitted the fail-closed
completion marker without OOM/fatal errors.

Paired `oldest_boundary - exact` deltas were `-0.00663463`, `+0.00655420`,
and `+0.00669180`. The mean delta was `+0.00220379` (+1.39% of the
exact-policy mean), with sample SD `0.00765461`. The mixed signs and variance
larger than the mean do not show a large shallow distribution-level regression.
This is intentionally classified as a shallow no-regression result: the
reference context and selector budget are both 2,048, so this cell is not
selector-sensitive and does not prove the 350k fix. The frozen 350k gate and
randomized 475k ladder carry that causal evidence.

Published artifact SHA-256:

```text
spec           af70f2043bea50d3d50aadbf48ad875601a04a30eb97304fc9e8e8b1b2d839df
compose        5dce77c527791069b793c835a62fe0699f6a60af37ebabbc96dd71644dd22ceb
matrix wrapper 1ba0c392953baf21c11b1a36190680c767f249fe5f4ad826df47b5c638d178ad
summarizer     fc81b049872bf19aebd9d0e449c64d52f58cee446210fea58c5716370b217dbc
aggregate JSON e9bc81d775e6830b8b3101943a6bba714b50ffd65aab8255c4a5d07acb663ff3
summary        1d18c949f4ccf7f053fe4188dd2aecbf05fefef0dcdab9a2ca2c9d3321647ce1d
full report    a5e0faabf73064ec6435bad02b03bd54fe1cf45c73e7c2c44ae9627c1e6821bc
```
