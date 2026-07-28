#!/usr/bin/env python3
"""Focused no-model gate for the GLM official-reference indexer.

This gate exercises the implementation that will be used in the causal boot:

* official interleaved RoPE against an independent scalar construction;
* BF16 paged-cache insert/gather semantics, including padded DCP slots;
* streamed FP32 GLM scoring followed by production ``run_row_topk``;
* IEEE/TF32 state restoration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.glm_official_indexer import (
    OfficialGLMReferenceIndexer,
    _ieee_fp32_matmul,
    _normalize_decode_seq_lens,
    _require_zero_based_key_windows,
    apply_official_glm_indexer_rope,
)


def _sha(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(payload).hexdigest()


def _independent_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_dim: int,
    theta: float,
) -> torch.Tensor:
    # Do not replace this with ``positions[:, None] * inv[None, :]``.  GLM's
    # reference uses a rank-3 matrix multiply, and the scalar shortcut is not
    # bit-identical at large positions even though it is algebraically equal.
    inv = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                rope_dim,
                2,
                dtype=torch.int64,
                device=tensor.device,
            ).float()
            / rope_dim
        )
    )
    with _ieee_fp32_matmul():
        angles = (
            inv[None, :, None].float() @ positions[None, None, :].float()
        ).transpose(1, 2).squeeze(0)
    cos = torch.cos(angles).to(torch.bfloat16).unsqueeze(1)
    sin = torch.sin(angles).to(torch.bfloat16).unsqueeze(1)
    even = tensor[..., :rope_dim][..., 0::2]
    odd = tensor[..., :rope_dim][..., 1::2]
    rotated = torch.cat(
        (even * cos - odd * sin, odd * cos + even * sin),
        dim=-1,
    )
    return torch.cat((rotated, tensor[..., rope_dim:]), dim=-1)


def _bare_indexer(*, topk: int, q_chunk_rows: int):
    target = OfficialGLMReferenceIndexer.__new__(OfficialGLMReferenceIndexer)
    torch.nn.Module.__init__(target)
    target.topk_tokens = topk
    target.head_dim = 128
    target.num_q_heads = 32
    target.q_chunk_rows = q_chunk_rows
    target.dcp_world_size = 1
    target.dcp_rank = 0
    target.cp_kv_cache_interleave_size = 1
    return target


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(20260727)

    live_decode_shape = torch.tensor([[9], [17]], device=device)
    normalized_decode_shape = _normalize_decode_seq_lens(
        live_decode_shape,
        rows=2,
    )
    if normalized_decode_shape.shape != (2,) or not torch.equal(
        normalized_decode_shape,
        live_decode_shape[:, 0],
    ):
        raise RuntimeError("live (B, 1) decode metadata normalization failed")
    try:
        _normalize_decode_seq_lens(torch.zeros((2, 2), device=device), rows=2)
    except NotImplementedError:
        pass
    else:
        raise RuntimeError("unsupported speculative decode metadata was accepted")

    _require_zero_based_key_windows(torch.zeros(3, device=device))
    try:
        _require_zero_based_key_windows(torch.tensor([0, 1], device=device))
    except RuntimeError:
        pass
    else:
        raise RuntimeError("nonzero prefill key-window base was accepted")

    positions = torch.tensor([0, 1, 17, 349_999], device=device)
    q = torch.randn(
        (4, 32, 128),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        (4, 128),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    q_actual, k_actual = apply_official_glm_indexer_rope(
        q,
        k,
        positions,
        rope_dim=64,
        rope_theta=8_000_000.0,
    )
    q_expected = _independent_rope(
        q,
        positions,
        rope_dim=64,
        theta=8_000_000.0,
    )
    k_expected = _independent_rope(
        k.unsqueeze(1),
        positions,
        rope_dim=64,
        theta=8_000_000.0,
    ).squeeze(1)
    if not torch.equal(q_actual, q_expected) or not torch.equal(
        k_actual, k_expected
    ):
        raise RuntimeError("official interleaved RoPE mismatch")

    target = _bare_indexer(topk=512, q_chunk_rows=2)
    cache = torch.full(
        (5, 8, 128),
        -99,
        dtype=torch.bfloat16,
        device=device,
    )
    target.k_cache = SimpleNamespace(kv_cache=cache)
    inserted = torch.randn(
        (6, 128),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    slots = torch.tensor([0, -1, 7, 8, 17, -1], device=device)
    target._insert_keys(inserted, slots)
    if not torch.equal(cache.view(-1, 128)[0], inserted[0]):
        raise RuntimeError("cache insertion failed at slot 0")
    if not torch.equal(cache.view(-1, 128)[7], inserted[2]):
        raise RuntimeError("cache insertion failed at slot 7")
    if not torch.equal(cache.view(-1, 128)[8], inserted[3]):
        raise RuntimeError("cache insertion failed at slot 8")
    if not torch.equal(cache.view(-1, 128)[17], inserted[4]):
        raise RuntimeError("cache insertion failed at slot 17")
    gathered = target._gather_keys(
        torch.tensor([0, 1, 2, 3, 4], device=device),
        18,
    )
    if not torch.equal(gathered, cache.view(-1, 128)[:18]):
        raise RuntimeError("paged-cache gather changed physical row order")

    rows, key_rows = 3, 2111
    q_score = torch.randn(
        (rows, 32, 128),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    keys = torch.randn(
        (key_rows, 128),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    weights = torch.randn(
        (rows, 32),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    lengths = torch.tensor([2111, 1800, 769], dtype=torch.int32, device=device)
    actual_indices = torch.empty((rows, 512), dtype=torch.int32, device=device)
    actual_values = torch.empty((rows, 512), dtype=torch.float32, device=device)
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    target._select_local(
        q=q_score,
        keys=keys,
        weights=weights,
        lengths=lengths,
        output_indices=actual_indices,
        output_scores=actual_values,
    )
    if torch.backends.cuda.matmul.allow_tf32 != original_tf32:
        raise RuntimeError("reference scorer leaked global TF32 state")

    with _ieee_fp32_matmul():
        per_head = torch.matmul(
            q_score.float().unsqueeze(0),
            keys.float().T.unsqueeze(0).unsqueeze(1),
        )
        per_head.mul_(128**-0.5)
        per_head.relu_()
        reference_scores = torch.matmul(
            weights.float().unsqueeze(0).unsqueeze(-2),
            per_head,
        ).squeeze(0).squeeze(-2)
    for row, length in enumerate(lengths.tolist()):
        reference_scores[row, length:] = -float("inf")
    reference_values, reference_indices = torch.topk(
        reference_scores,
        512,
        dim=-1,
        sorted=False,
    )
    for row in range(rows):
        actual_set = set(actual_indices[row].tolist())
        reference_set = set(reference_indices[row].tolist())
        if actual_set != reference_set:
            raise RuntimeError(
                f"production top-k set mismatch on scorer row {row}: "
                f"{len(actual_set ^ reference_set)} indices differ"
            )
        source = reference_scores[row].gather(
            0, actual_indices[row].to(torch.int64)
        )
        if not torch.allclose(actual_values[row], source, atol=1e-5, rtol=1e-6):
            raise RuntimeError(f"production top-k values mismatch on row {row}")

    # A DCP rank can own no local keys for a short or unevenly-sharded row.
    # It must contribute only -inf scores and -1 indices to the global union.
    empty_indices = torch.empty((1, 512), dtype=torch.int32, device=device)
    empty_values = torch.empty((1, 512), dtype=torch.float32, device=device)
    target._select_local(
        q=q_score[:1],
        keys=keys[:0],
        weights=weights[:1],
        lengths=torch.zeros(1, dtype=torch.int32, device=device),
        output_indices=empty_indices,
        output_scores=empty_values,
    )
    if not bool(torch.isneginf(empty_values).all()):
        raise RuntimeError("zero-local-length row emitted a finite score")
    target._merge_global(empty_indices, empty_values)
    if not bool((empty_indices == -1).all()):
        raise RuntimeError("zero-local-length row emitted a candidate index")

    report = {
        "schema": "v20-glm-official-reference-indexer-gate-v1",
        "device": str(device),
        "rope": {
            "exact": True,
            "q_sha256": _sha(q_actual),
            "k_sha256": _sha(k_actual),
        },
        "cache": {
            "insert_exact": True,
            "gather_exact": True,
            "cache_sha256": _sha(cache),
        },
        "scorer_topk": {
            "rows": rows,
            "keys": key_rows,
            "topk": 512,
            "set_exact": True,
            "values_match_source": True,
            "indices_sha256": _sha(actual_indices),
            "values_sha256": _sha(actual_values),
        },
        "tf32_state_restored": True,
        "metadata_contract": {
            "live_decode_b1_normalized": True,
            "speculative_decode_rejected": True,
            "zero_based_prefill_required": True,
            "zero_local_length_contributes_no_candidates": True,
        },
        "status": "PASS",
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
