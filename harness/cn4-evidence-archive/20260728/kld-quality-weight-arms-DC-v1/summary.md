# GLM-5.2 v20 dynamic NVFP4 scale shallow BF16 KLD

| Scale mode | Runs | KLD mean ± sample SD | Min | Max |
|---|---:|---:|---:|---:|
| `static_calibrated` | 3 | 0.13378422 ± 0.00126327 | 0.13232554 | 0.13452094 |
| `dynamic_per_token` | 3 | 0.13326164 ± 0.00212500 | 0.13117311 | 0.13542133 |

Paired `dynamic_per_token - static_calibrated` KLD deltas: -0.00133044, +0.00091515, -0.00115243

Mean paired delta: -0.00052258 (-0.39% of the static mean).

> This is a 2,048-token no-regression gate. The selector budget is also 2,048, so this cell is not selector-sensitive. The frozen 350k gate and randomized 475k ladder provide deep-context evidence.
