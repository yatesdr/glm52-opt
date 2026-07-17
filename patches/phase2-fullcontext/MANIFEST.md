# Packed-CKV phase-2 Gate-C manifest

All digests are MD5. The transport patch is a delta on top of the field-fixed
Stage-3 package; it is not applicable to the pre-fix `5092bf94...` sparse
overlay.

| Deployment path | Exact base | Gate-C result |
|---|---|---|
| `b12x/distributed/pcie_dma.py` | `fc796fa9af58d5b63bce85f8d5c195e8` | `34a011ed647b52fe7fb6645e77205ae5` |
| `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` | `20a2cf60ce2e99d8c90249d458f330f8` | `d56bd035296c87e4fe293a6020e11a73` |
| `vllm/v1/worker/gpu_worker.py` | `0829a65484d4dd14c385366291e7a25c` | `6afef2e7c20b0aac870d9a48b59cabaf` |

`packed-ckv-phase2.patch`:
`b3c21fae55eabf0c3f98a0be6f253e65`

The `pcie_dma.py` result includes field fix #2 from the server gates: B12X's
logger hierarchy has no serving-container handler, so escrow/probe evidence is
registered explicitly under `vllm.b12x.pcie_dma`. The patch and overlay-copy
deployment routes now both reproduce the deployed `34a011ed...` bytes; the
earlier Gate-C pin `6d7028e1...` predated this observability fix.

The first two bases are the Stage-3 overlays after commit `67cd084` folded in
the field fix from `../glm-5/sol-packed-ckv-fix1.md`. The worker base is the
mounted v14-equivalent overlay byte-confirmed in
`../glm-5/sol-phase2-gateA-verdict.md`.

No phase-2 delta is needed in the Stage-3 `mla_attention.py` or `common.py`:
the former already dispatches the packed-CKV backend with its block table,
and the latter already hard-disarms the DCP output ring when CKV transport is
selected. Those Stage-3 files remain prerequisites.

## Check files

| File | MD5 |
|---|---|
| `checks/check_phase2_source_contract.py` | `00df29c8cb721f70c9f02481849805a4` |
| `checks/check_triton_constexpr.py` | `cd671b5e47907055c26cec8e6cbfac05` |
| `checks/test_active_layout.py` | `cce6200443c747fffd9756d1d96d266d` |
| `checks/test_escrow_state.py` | `2ed02ab0498d2743f6ee01eac3667248` |
| `checks/test_pool_remap.py` | `c3150949f9eb150778ca736279b03be2` |
| `checks/test_profiler_state.py` | `2bcef61de4aa04b4469c380325da2c7a` |
| `checks/test_route_determinism.py` | `f16368f6384edb721b32ed303b0c9bff` |
