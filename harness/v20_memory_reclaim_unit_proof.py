#!/usr/bin/env python3
"""CPU proof for the v20 absorbed kv_b_proj memory-reclaim overlay."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.attention import mla_attention as mla_module
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    _materialize_kv_b_proj_weight,
    _release_b12x_mxfp8_kv_b_proj,
)


class PackedLinearMethod:
    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert layer.b12x_mxfp8_packed_weight is not None
        assert bias is None
        return x @ self.weight.T


def make_layer() -> tuple[MLAAttention, torch.Tensor]:
    weight = torch.arange(28.0, dtype=torch.float32).reshape(14, 2)
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_lora_rank = 2
    layer.num_heads = 2
    layer.qk_nope_head_dim = 3
    layer.v_head_dim = 4
    layer.kv_b_proj = torch.nn.Module()
    layer.kv_b_proj.register_parameter(
        "weight", torch.nn.Parameter(weight, requires_grad=False)
    )
    layer.kv_b_proj.register_parameter(
        "weight_scale", torch.nn.Parameter(torch.ones((14, 1)), requires_grad=False)
    )
    layer.kv_b_proj.input_size_per_partition = 2
    layer.kv_b_proj.quant_method = PackedLinearMethod(weight)
    layer.kv_b_proj.b12x_mxfp8_packed_weight = object()
    layer.is_aiter_triton_fp4_bmm_enabled = False
    layer.is_aiter_triton_fp8_bmm_enabled = False
    layer.quant_config = None
    layer.layer_name = "proof"
    layer.kv_cache_dtype = "auto"
    layer.impl = SimpleNamespace(
        can_release_kv_b_proj_after_loading=True,
        supports_quant_query_input=False,
    )
    layer.prefill_backend = None
    return layer, weight


def main() -> None:
    layer, expected = make_layer()

    original_set_scales = mla_module.set_default_quant_scales
    original_absorb_enabled = mla_module._b12x_absorb_bmm_enabled
    mla_module.set_default_quant_scales = lambda *_, **__: None
    mla_module._b12x_absorb_bmm_enabled = lambda: False
    try:
        with torch.no_grad():
            layer.process_weights_after_loading(torch.float32)
    finally:
        mla_module.set_default_quant_scales = original_set_scales
        mla_module._b12x_absorb_bmm_enabled = original_absorb_enabled

    assert tuple(layer.W_UK_T.shape) == (2, 3, 2)
    assert tuple(layer.W_UV.shape) == (2, 2, 4)
    assert not hasattr(layer.kv_b_proj, "weight")
    assert not hasattr(layer.kv_b_proj, "weight_scale")
    assert layer.kv_b_proj.b12x_mxfp8_packed_weight is None

    # A reload recreates the B12X pack before post-load processing. Prove that
    # the helper can rematerialize from that owner after source release.
    layer.kv_b_proj.b12x_mxfp8_packed_weight = object()
    rematerialized = _materialize_kv_b_proj_weight(
        layer.kv_b_proj,
        out_dtype=torch.float32,
        fallback_device=torch.device("cpu"),
    )
    torch.testing.assert_close(rematerialized, expected)

    inert = torch.nn.Module()
    inert_weight = torch.nn.Parameter(torch.ones((4, 3)), requires_grad=False)
    inert.register_parameter("weight", inert_weight)
    assert _release_b12x_mxfp8_kv_b_proj(inert) is False
    assert inert.weight is inert_weight

    print(
        json.dumps(
            {
                "verdict": "PASS",
                "absorbed_shapes": {
                    "W_UK_T": list(layer.W_UK_T.shape),
                    "W_UV": list(layer.W_UV.shape),
                },
                "source_parameters_released": True,
                "reload_rematerialization": True,
                "non_b12x_owner_inert": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
