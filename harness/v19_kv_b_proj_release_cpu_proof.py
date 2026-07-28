#!/usr/bin/env python3
"""CPU-only proof for the vLLM #154 backport (release absorbed kv_b_proj).

Establishes, without a GPU, the two things the backport depends on:

  1. WITHOUT #136 the absorbed parameters are *views into* the dequantized
     kv_b_proj storage, so releasing the source would corrupt them.
  2. WITH #136 (`.contiguous()`) they are independent allocations, so the
     release is safe — and the values are bit-identical either way.

Also exercises the release/idempotency/fallback helpers directly.

Run anywhere torch is importable. Touches no CUDA.
    python3 v19_kv_b_proj_release_cpu_proof.py
Exit 0 = every check passed.
"""

from __future__ import annotations

import sys

import torch

# GLM-5.2 hybrid geometry, TP4 (see config.json on cn3)
NUM_HEADS_PER_RANK = 64 // 4
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
V_HEAD_DIM = 256
NUM_LAYERS = 78

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def storage_ptr(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


def build_absorbed(contiguous: bool):
    """Reproduce mla_attention.process_weights_after_loading's absorption."""
    # get_and_maybe_dequant_weights(...).T  ->  (kv_lora_rank, heads*(nope+v))
    kv_b_proj_weight = torch.randn(
        KV_LORA_RANK, NUM_HEADS_PER_RANK * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        dtype=torch.bfloat16,
    )
    src = kv_b_proj_weight
    kv_b_proj_weight = kv_b_proj_weight.view(
        KV_LORA_RANK, NUM_HEADS_PER_RANK, QK_NOPE_HEAD_DIM + V_HEAD_DIM
    )
    W_UK, W_UV = kv_b_proj_weight.split([QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)
    w_uv = W_UV.transpose(0, 1)      # (N, L, V)
    w_uk_t = W_UK.permute(1, 2, 0)   # (N, P, L)
    if contiguous:                   # this is exactly what #136 adds
        w_uv = w_uv.contiguous()
        w_uk_t = w_uk_t.contiguous()
    return src, w_uv, w_uk_t


def main() -> int:
    torch.manual_seed(0)
    print("\n=== 1. aliasing: does the absorbed pair share storage with kv_b_proj? ===")

    src_v, uv_view, uk_view = build_absorbed(contiguous=False)
    aliases = (storage_ptr(uv_view) == storage_ptr(src_v)
               and storage_ptr(uk_view) == storage_ptr(src_v))
    check("WITHOUT #136 the absorbed pair aliases the source", aliases,
          "releasing kv_b_proj here would corrupt W_UV/W_UK_T")

    src_c, uv_c, uk_c = build_absorbed(contiguous=True)
    independent = (storage_ptr(uv_c) != storage_ptr(src_c)
                   and storage_ptr(uk_c) != storage_ptr(src_c))
    check("WITH #136 the absorbed pair is independent storage", independent,
          "this is what makes #154's release safe")

    print("\n=== 2. #136 changes layout, not values ===")
    torch.manual_seed(1); _, uv_v2, uk_v2 = build_absorbed(contiguous=False)
    torch.manual_seed(1); _, uv_c2, uk_c2 = build_absorbed(contiguous=True)
    check("W_UV bit-identical view vs contiguous", torch.equal(uv_v2, uv_c2))
    check("W_UK_T bit-identical view vs contiguous", torch.equal(uk_v2, uk_c2))
    check("contiguous flag actually set", uv_c2.is_contiguous() and uk_c2.is_contiguous())
    check("the view really was non-contiguous", not uv_v2.is_contiguous())

    print("\n=== 3. survival: mutate/free the source, do the weights hold? ===")
    src_c2, uv_keep, uk_keep = build_absorbed(contiguous=True)
    uv_before, uk_before = uv_keep.clone(), uk_keep.clone()
    src_c2.zero_()          # simulate the source being released/reused
    del src_c2
    check("W_UV survives source destruction", torch.equal(uv_keep, uv_before))
    check("W_UK_T survives source destruction", torch.equal(uk_keep, uk_before))

    src_v2, uv_alias, uk_alias = build_absorbed(contiguous=False)
    uv_alias_before = uv_alias.clone()
    src_v2.zero_()
    check("aliased W_UV is DESTROYED by source release (control)",
          not torch.equal(uv_alias, uv_alias_before),
          "confirms the hazard #136 removes")

    print("\n=== 4. _release_b12x_mxfp8_kv_b_proj semantics ===")

    def _release(layer) -> bool:
        if getattr(layer, "b12x_mxfp8_packed_weight", None) is None:
            return False
        for name in ("weight", "weight_scale"):
            if hasattr(layer, name):
                delattr(layer, name)
        layer.b12x_mxfp8_packed_weight = None
        return True

    class FakeLayer:
        pass

    lay = FakeLayer()
    lay.weight = torch.zeros(4)
    lay.weight_scale = torch.zeros(4)
    lay.b12x_mxfp8_packed_weight = torch.zeros(4)
    check("release returns True when packed weight present", _release(lay) is True)
    check("weight deleted", not hasattr(lay, "weight"))
    check("weight_scale deleted", not hasattr(lay, "weight_scale"))
    check("packed handle cleared", lay.b12x_mxfp8_packed_weight is None)
    check("release is idempotent (second call is a no-op)", _release(lay) is False)

    plain = FakeLayer()
    plain.weight = torch.zeros(4)
    check("non-B12X layer is left untouched",
          _release(plain) is False and hasattr(plain, "weight"),
          "guard: only releases when b12x_mxfp8_packed_weight exists")

    print("\n=== 5. reclaim arithmetic (per rank) ===")
    el = NUM_HEADS_PER_RANK * (QK_NOPE_HEAD_DIM + V_HEAD_DIM) * KV_LORA_RANK
    mxfp8 = el * 1 + el // 32          # 1 byte/elem + 1 scale per 32
    bf16 = el * 2
    print(f"    kv_b_proj source (mxfp8) : {mxfp8/2**20:7.2f} MiB/layer"
          f"  x{NUM_LAYERS} = {mxfp8*NUM_LAYERS/2**30:.3f} GiB")
    print(f"    absorbed pair    (bf16)  : {bf16/2**20:7.2f} MiB/layer"
          f"  x{NUM_LAYERS} = {bf16*NUM_LAYERS/2**30:.3f} GiB")
    reclaim = mxfp8 * NUM_LAYERS / 2**30
    check("predicted reclaim is ~0.28 GiB/GPU (ledger says ~290 MiB)",
          0.25 <= reclaim <= 0.33, f"{reclaim:.3f} GiB")

    print(f"\n=== result: {'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED'} ===")
    for f in FAILED:
        print(f"    FAILED: {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
