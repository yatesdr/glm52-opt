#!/usr/bin/env python3
"""Write compact records, consume them with v18's real B12X decode reader."""

from __future__ import annotations

import math
from importlib import import_module

import torch


OP_NAME = "concat_and_cache_nvfp4_mla_fp8_rope"
NUM_HEADS = 16
HEAD_DIM = 576
V_DIM = 512
PAGE_SIZE = 64
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _decode_writer_records(cache: torch.Tensor, slots: list[int]) -> torch.Tensor:
    """Torch oracle for the exact operations performed by v18's reader."""
    e2m1 = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=cache.device,
        dtype=torch.float32,
    )
    rows: list[torch.Tensor] = []
    for slot in slots:
        row = cache[slot // PAGE_SIZE, slot % PAGE_SIZE]
        packed = row[:256].to(torch.int64)
        codes = torch.empty((512,), device=cache.device, dtype=torch.int64)
        codes[0::2] = packed & 0xF
        codes[1::2] = packed >> 4
        group_scales = row[256:288].view(torch.float8_e4m3fn).float()
        latent = e2m1[codes] * group_scales.repeat_interleave(16)

        rope_scale = row[288:292].contiguous().view(torch.float32)
        rope = row[304:368].view(torch.float8_e4m3fn).float() * rope_scale

        # Both exact decode paths convert the reconstructed operands to BF16
        # before their BF16 MMAs.
        rows.append(torch.cat((latent, rope)).to(torch.bfloat16).float())
    return torch.stack(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this gate requires an SM120 CUDA image/GPU")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) < (12, 0):
        raise RuntimeError("v18 B12X compact-reader gate requires SM120+")

    import_module("b12x.attention.mla.fp8_rope_writer")
    from b12x.attention.mla.kernel import run_unified_decode
    from b12x.attention.mla.traits import ScaleFormat
    from b12x.attention.workspace import B12XAttentionWorkspace

    namespace = getattr(torch.ops, "_C_fp8_rope_ops", None)
    if namespace is None or not hasattr(namespace, OP_NAME):
        raise RuntimeError("writer module did not register the v18 operator")

    generator = torch.Generator(device=device).manual_seed(0xB12_0368)
    kv_c = torch.randn(
        (2, V_DIM), device=device, dtype=torch.bfloat16, generator=generator
    )
    k_pe = torch.randn(
        (2, HEAD_DIM - V_DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    cache = torch.zeros(
        (1, PAGE_SIZE, 368), device=device, dtype=torch.uint8
    )
    slots = torch.tensor([0, 1], device=device, dtype=torch.int64)
    scale = torch.ones((), device=device, dtype=torch.float32)
    getattr(namespace, OP_NAME)(kv_c, k_pe, cache, slots, scale)
    torch.cuda.synchronize()

    keys = _decode_writer_records(cache, [0, 1])
    values = keys[:, :V_DIM]
    q = torch.zeros(
        (1, NUM_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16
    )
    direction = (keys[0, V_DIM:] - keys[1, V_DIM:]).to(torch.bfloat16)
    multipliers = torch.linspace(
        -1.5, 1.5, NUM_HEADS, device=device, dtype=torch.float32
    )
    q[0, :, V_DIM:] = (
        multipliers[:, None] * direction.float()[None, :]
    ).to(torch.bfloat16)

    # The kernel's candidate window is 64. Only the first two entries are
    # active, but padding the table to one full window mirrors its live shape.
    selected = torch.zeros((1, 64), device=device, dtype=torch.int32)
    selected[0, 1] = 1
    lengths = torch.tensor([2], device=device, dtype=torch.int32)
    workspace = B12XAttentionWorkspace.for_contract(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        v_head_dim=V_DIM,
        topk=64,
        max_total_q=1,
        max_batch=1,
        max_kv_rows=64,
        page_size=PAGE_SIZE,
        use_cuda_graph=False,
        max_chunks_per_row=1,
    )
    actual = run_unified_decode(
        q_all=q,
        swa_k_cache=cache,
        swa_indices=selected,
        swa_topk_lengths=lengths,
        workspace=workspace,
        sm_scale=SM_SCALE,
        latent_scale=1.0,
        swa_page_size=PAGE_SIZE,
        forced_num_splits=1,
        scale_format_override=ScaleFormat.NVFP4_E4M3,
        fp8_rope_override=True,
    )
    torch.cuda.synchronize()

    logits = torch.einsum("hd,td->ht", q[0].float(), keys) * SM_SCALE
    probabilities = torch.softmax(logits, dim=-1)
    expected = torch.einsum("ht,tv->hv", probabilities, values)
    if not torch.isfinite(actual).all():
        raise AssertionError("v18 reader returned a non-finite result")
    torch.testing.assert_close(
        actual[0].float(),
        expected,
        rtol=0.08,
        atol=0.08,
    )
    max_abs = (actual[0].float() - expected).abs().max().item()
    print(
        "PASS: source writer -> v18 B12X compact decode reader round trip; "
        f"max_abs={max_abs:.6g}"
    )


if __name__ == "__main__":
    main()
