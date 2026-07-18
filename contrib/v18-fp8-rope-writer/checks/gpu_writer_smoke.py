#!/usr/bin/env python3
"""In-image dry-compile and canary smoke test for the compact writer."""

from __future__ import annotations

from importlib import import_module

import torch


OP_NAME = "concat_and_cache_nvfp4_mla_fp8_rope"
CANARY = 0xA5


def record(cache: torch.Tensor, slot: int) -> torch.Tensor:
    block_size = int(cache.shape[1])
    return cache[slot // block_size, slot % block_size]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this gate requires an SM120 CUDA image/GPU")

    import_module("b12x.attention.mla.fp8_rope_writer")
    namespace = getattr(torch.ops, "_C_fp8_rope_ops", None)
    if namespace is None or not hasattr(namespace, OP_NAME):
        raise RuntimeError("writer module did not register the v18 operator")

    generator = torch.Generator(device="cuda").manual_seed(0x368)
    kv_c = torch.randn(
        (4, 512), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    k_pe = torch.randn(
        (4, 64), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    # Token 2 proves the zero-scale branch; token 3 proves negative-slot skip.
    kv_c[2].zero_()
    k_pe[2].zero_()
    slots = torch.tensor([0, 65, 2, -1], device="cuda", dtype=torch.int64)
    cache = torch.full(
        (2, 64, 368), CANARY, device="cuda", dtype=torch.uint8
    )
    scale = torch.ones((), device="cuda", dtype=torch.float32)

    getattr(namespace, OP_NAME)(kv_c, k_pe, cache, slots, scale)
    torch.cuda.synchronize()

    for slot in (0, 65, 2):
        row = record(cache, slot)
        if not torch.equal(row[292:304], torch.zeros_like(row[292:304])):
            raise AssertionError(f"slot {slot}: v18 pad [292,304) is not zero")

    # The zero token must encode to an all-zero record, including both scales.
    if torch.count_nonzero(record(cache, 2)).item() != 0:
        raise AssertionError("zero token did not produce an all-zero record")

    # Records never named by slot_mapping must retain the allocation canary.
    untouched = record(cache, 1)
    if not torch.equal(untouched, torch.full_like(untouched, CANARY)):
        raise AssertionError("writer modified an unselected cache record")

    for token, slot in ((0, 0), (1, 65)):
        row = record(cache, slot)
        rope_scale = row[288:292].contiguous().view(torch.float32).item()
        if not (rope_scale > 0.0):
            raise AssertionError(f"slot {slot}: invalid RoPE scale {rope_scale}")
        rope_quant = row[304:368].view(torch.float8_e4m3fn).float()
        rope_reconstructed = rope_quant * rope_scale
        torch.testing.assert_close(
            rope_reconstructed,
            k_pe[token].float(),
            rtol=0.15,
            atol=0.03,
            msg=lambda message: f"slot {slot}: E4M3 RoPE mismatch: {message}",
        )

    print("PASS: v18 FP8-RoPE writer compiled and passed layout/canary checks")


if __name__ == "__main__":
    main()
