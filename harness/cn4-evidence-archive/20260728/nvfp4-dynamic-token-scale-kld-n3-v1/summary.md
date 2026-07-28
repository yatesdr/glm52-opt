# GLM-5.2 v20 dynamic NVFP4 scale shallow BF16 KLD

| Scale mode | Runs | KLD mean ± sample SD | Min | Max |
|---|---:|---:|---:|---:|
| `static_calibrated` | 3 | 0.14622770 ± 0.00468791 | 0.14180147 | 0.15113949 |
| `dynamic_per_token` | 3 | 0.13903565 ± 0.00201006 | 0.13672545 | 0.14038452 |

Paired `dynamic_per_token - static_calibrated` KLD deltas: -0.00574516, -0.01075497, -0.00507602

Mean paired delta: -0.00719205 (-4.92% of the static mean).

> This is a 2,048-token no-regression gate. The selector budget is also 2,048, so this cell is not selector-sensitive. The frozen 350k gate and randomized 475k ladder provide deep-context evidence.
