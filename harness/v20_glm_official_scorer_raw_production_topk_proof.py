#!/usr/bin/env python3
"""Replay the complete official GLM indexer scorer on frozen real activations.

The trace consumed here begins before the indexer's Q/K preprocessing.  This
proof reproduces the reference implementation in the pinned Transformers
source:

* separate BF16 Q and K projections;
* LayerNorm on K;
* GLM interleaved RoPE;
* FP32 Q.K, scale, ReLU;
* FP32 learned per-head projection and head reduction;
* exact top-k.

It reports the first tensor boundary where the optimized vLLM path differs,
measures how each boundary changes the selected set, and feeds the complete
reference score row through both production SparkInfer top-k entrypoints.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ACTIVATION_SCHEMA = "v20-indexer-official-reference-activation-v1"
SELECTION_SCHEMA = "v20-indexer-official-reference-runtime-selection-v1"
HEAD_DIM = 128
ROPE_DIM = 64
DEFAULT_TOPK = 2048
DEFAULT_ROPE_THETA = 8_000_000.0
TILED_BLOCK_Q = 32
TILED_BLOCK_K = 256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(payload).hexdigest()


def _load_trace(trace_dir: Path) -> dict[str, Any]:
    chunk_paths = sorted(trace_dir.glob("chunk-*.pt"))
    if not chunk_paths:
        raise RuntimeError(f"{trace_dir}: no chunk-*.pt files")

    concatenated: dict[str, list[torch.Tensor]] = {
        "positions": [],
        "k_raw_fused": [],
        "k_raw_separate": [],
        "k_post_norm_runtime": [],
        "k_post_runtime": [],
    }
    tail: dict[str, torch.Tensor] | None = None
    layer: int | None = None
    rank: int | None = None
    manifest: list[dict[str, Any]] = []

    for expected_chunk, path in enumerate(chunk_paths):
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record.get("schema") != ACTIVATION_SCHEMA:
            raise RuntimeError(f"{path}: wrong schema {record.get('schema')!r}")
        if int(record["chunk"]) != expected_chunk:
            raise RuntimeError(
                f"{path}: expected chunk {expected_chunk}, got {record['chunk']}"
            )
        record_layer = int(record["layer"])
        record_rank = int(record["tp_rank"])
        if layer is None:
            layer = record_layer
            rank = record_rank
        elif record_layer != layer or record_rank != rank:
            raise RuntimeError(f"{path}: layer/rank changed within trace")

        positions = record["positions"].to(torch.int64).contiguous()
        tensors = {
            "k_raw_fused": record["k_raw_fused_bf16"].contiguous(),
            "k_raw_separate": record["k_raw_separate_bf16"].contiguous(),
            "k_post_norm_runtime": (
                record["k_post_norm_runtime_bf16"].contiguous()
            ),
            "k_post_runtime": record["k_post_runtime_bf16"].contiguous(),
        }
        rows = int(positions.numel())
        if positions.ndim != 1:
            raise RuntimeError(f"{path}: positions must be rank 1")
        for name, tensor in tensors.items():
            if tensor.dtype != torch.bfloat16:
                raise RuntimeError(f"{path}: {name} is {tensor.dtype}, not BF16")
            if tensor.shape != (rows, HEAD_DIM):
                raise RuntimeError(
                    f"{path}: {name} shape {tuple(tensor.shape)} != "
                    f"({rows}, {HEAD_DIM})"
                )
            concatenated[name].append(tensor)
        concatenated["positions"].append(positions)

        if "q_pre_rope_final_bf16" in record:
            if tail is not None:
                raise RuntimeError("trace contains more than one tail record")
            tail = {
                key: value.contiguous()
                for key, value in record.items()
                if isinstance(value, torch.Tensor)
                and key
                not in {
                    "positions",
                    "k_raw_fused_bf16",
                    "k_raw_separate_bf16",
                    "k_post_norm_runtime_bf16",
                    "k_post_runtime_bf16",
                }
            }
        manifest.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )

    if tail is None:
        raise RuntimeError("trace has no tail record")
    positions_all = torch.cat(concatenated["positions"], dim=0)
    if int(torch.unique(positions_all).numel()) != int(positions_all.numel()):
        raise RuntimeError("trace positions contain duplicates")
    if int(positions_all.numel()) > 1 and not bool(
        torch.all(positions_all[1:] == positions_all[:-1] + 1)
    ):
        raise RuntimeError("trace positions are not contiguous")

    required_tail = {
        "q_pre_rope_final_bf16": (torch.bfloat16, (32, HEAD_DIM)),
        "q_pre_separate_final_bf16": (torch.bfloat16, (32, HEAD_DIM)),
        "q_post_rope_final_bf16": (torch.bfloat16, (32, HEAD_DIM)),
        "weights_fused_final_bf16": (torch.bfloat16, (32,)),
        "hidden_final_bf16": (torch.bfloat16, (6144,)),
        "q_resid_final_bf16": (torch.bfloat16, (2048,)),
        "projection_weight_bf16": (torch.bfloat16, (160, 6144)),
        "q_projection_weight_bf16": (torch.bfloat16, (4096, 2048)),
        "k_norm_weight_fp32": (torch.float32, (HEAD_DIM,)),
        "k_norm_bias_fp32": (torch.float32, (HEAD_DIM,)),
    }
    for key, (dtype, shape) in required_tail.items():
        if key not in tail:
            raise RuntimeError(f"tail record is missing {key}")
        tensor = tail[key]
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise RuntimeError(
                f"tail {key}: got {tensor.dtype} {tuple(tensor.shape)}, "
                f"expected {dtype} {shape}"
            )

    selection_path = trace_dir / "runtime-selection.pt"
    if not selection_path.is_file():
        raise RuntimeError(f"{selection_path}: missing runtime selection")
    selection = torch.load(selection_path, map_location="cpu", weights_only=True)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise RuntimeError(
            f"{selection_path}: wrong schema {selection.get('schema')!r}"
        )
    runtime_indices = selection["topk_indices"].to(torch.int64).contiguous()
    if runtime_indices.ndim != 1:
        raise RuntimeError("runtime top-k indices must be rank 1")
    if int(selection["layer"]) != layer or int(selection["tp_rank"]) != rank:
        raise RuntimeError("runtime selection layer/rank does not match trace")
    if int(selection["absolute_position"]) != int(positions_all[-1].item()):
        raise RuntimeError("runtime selection position does not match trace tail")

    manifest.append(
        {
            "name": selection_path.name,
            "bytes": selection_path.stat().st_size,
            "sha256": _sha256_file(selection_path),
        }
    )
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "chunk_paths": chunk_paths,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "rank": rank,
        "positions": positions_all,
        "k_raw_fused": torch.cat(concatenated["k_raw_fused"], dim=0),
        "k_raw_separate": torch.cat(concatenated["k_raw_separate"], dim=0),
        "k_post_norm_runtime": torch.cat(
            concatenated["k_post_norm_runtime"], dim=0
        ),
        "k_post_runtime": torch.cat(
            concatenated["k_post_runtime"], dim=0
        ),
        "tail": tail,
        "runtime_indices": runtime_indices,
        "runtime_selection": selection,
    }


def _tensor_delta(
    label: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, Any]:
    if actual.shape != reference.shape:
        raise RuntimeError(
            f"{label}: shape mismatch {tuple(actual.shape)} vs "
            f"{tuple(reference.shape)}"
        )
    actual_cpu = actual.detach().contiguous().cpu()
    reference_cpu = reference.detach().contiguous().cpu()
    different = actual_cpu != reference_cpu
    delta = actual_cpu.float() - reference_cpu.float()
    return {
        "label": label,
        "shape": list(actual.shape),
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(reference.dtype),
        "bit_exact": bool(torch.equal(actual_cpu, reference_cpu)),
        "different_elements": int(different.sum().item()),
        "elements": int(actual.numel()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(delta.square())).item()),
        "actual_sha256": _sha256_tensor(actual_cpu),
        "reference_sha256": _sha256_tensor(reference_cpu),
    }


def _official_cos_sin(
    positions: torch.Tensor,
    *,
    device: torch.device,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, ROPE_DIM, 2, dtype=torch.int64)
            .to(device=device, dtype=torch.float32)
            / ROPE_DIM
        )
    )
    # Match GlmMoeDsaRotaryEmbedding.forward literally: [1,D/2,1] @
    # [1,1,T], transpose, then cos/sin with autocast disabled.
    inv_expanded = inv_freq[None, :, None].float()
    pos_expanded = positions.to(device=device)[None, None, :].float()
    freqs = (inv_expanded @ pos_expanded).transpose(1, 2).squeeze(0)
    return freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)


def _official_interleaved_rope(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Match Transformers apply_rotary_pos_emb_interleave exactly."""

    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    first = tensor[..., 0::2]
    second = tensor[..., 1::2]
    return torch.cat(
        [
            first * cos - second * sin,
            second * cos + first * sin,
        ],
        dim=-1,
    )


def _official_to_vllm_interleaved(tensor: torch.Tensor) -> torch.Tensor:
    """Map HF's half-split output ordering to vLLM's GPT-J ordering.

    Q and K receive the same permutation, so this ordering difference preserves
    their dot products.  Mapping it here permits a direct tensor comparison to
    the optimized runtime RoPE output.
    """

    first, second = torch.chunk(tensor, 2, dim=-1)
    return torch.stack((first, second), dim=-1).flatten(-2)


@torch.inference_mode()
def _build_official_qk(
    trace: dict[str, Any],
    *,
    device: torch.device,
    rope_theta: float,
    chunk_rows: int,
) -> dict[str, torch.Tensor]:
    tail = trace["tail"]
    positions = trace["positions"]
    q_pre = tail["q_pre_separate_final_bf16"].to(device)
    q_position = positions[-1:].contiguous()
    q_cos, q_sin = _official_cos_sin(
        q_position,
        device=device,
        rope_theta=rope_theta,
    )
    q_rot = _official_interleaved_rope(
        q_pre[:, :ROPE_DIM].unsqueeze(0),
        q_cos,
        q_sin,
    ).squeeze(0)
    q_official = torch.cat([q_rot, q_pre[:, ROPE_DIM:]], dim=-1)

    k_raw = trace["k_raw_separate"]
    k_norm_weight = tail["k_norm_weight_fp32"].to(
        device=device, dtype=torch.bfloat16
    )
    k_norm_bias = tail["k_norm_bias_fp32"].to(
        device=device, dtype=torch.bfloat16
    )
    k_norm_parts: list[torch.Tensor] = []
    k_official_parts: list[torch.Tensor] = []
    for start in range(0, int(k_raw.shape[0]), chunk_rows):
        end = min(start + chunk_rows, int(k_raw.shape[0]))
        raw = k_raw[start:end].to(device)
        normalized = F.layer_norm(
            raw,
            (HEAD_DIM,),
            k_norm_weight,
            k_norm_bias,
            1e-6,
        )
        cos, sin = _official_cos_sin(
            positions[start:end],
            device=device,
            rope_theta=rope_theta,
        )
        rotated = _official_interleaved_rope(
            normalized[:, :ROPE_DIM].unsqueeze(1),
            cos,
            sin,
        ).squeeze(1)
        k_norm_parts.append(normalized.cpu())
        k_official_parts.append(
            torch.cat([rotated, normalized[:, ROPE_DIM:]], dim=-1).cpu()
        )
    return {
        "q_official": q_official,
        "k_norm_official": torch.cat(k_norm_parts, dim=0),
        "k_official": torch.cat(k_official_parts, dim=0),
    }


@torch.inference_mode()
def _score(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    chunk_rows: int,
) -> torch.Tensor:
    # Preserve the tensor ranks used by GlmMoeDsaIndexer.forward:
    # q [B,S,H,D], k [B,T,D], scores [B,S,H,T], weights [B,S,H].
    # A rank-2 simplification for a single query row is algebraically
    # equivalent but can choose a different cuBLAS reduction algorithm.
    q = q.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    weights = (
        weights.to(device=device, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    scores = torch.empty(
        (int(k.shape[0]),),
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, int(k.shape[0]), chunk_rows):
        end = min(start + chunk_rows, int(k.shape[0]))
        keys = k[start:end].to(device=device, dtype=torch.float32).unsqueeze(0)
        per_head = torch.matmul(
            q,
            keys.transpose(-1, -2).unsqueeze(1),
        )
        per_head.mul_(HEAD_DIM**-0.5)
        per_head.relu_()
        scores[start:end] = (
            torch.matmul(weights.unsqueeze(-2), per_head)
            .squeeze(0)
            .squeeze(0)
            .squeeze(0)
        )
    if bool(torch.isnan(scores).any()):
        raise RuntimeError("scorer produced NaN")
    return scores


def _canonical_indices(indices: torch.Tensor) -> torch.Tensor:
    return torch.sort(indices.reshape(-1).to(torch.int64)).values


def _selection_metrics(
    label: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, Any]:
    actual = _canonical_indices(actual)
    reference = _canonical_indices(reference)
    actual_unique = torch.unique(actual)
    reference_unique = torch.unique(reference)
    if actual_unique.numel() != actual.numel():
        raise RuntimeError(f"{label}: actual selection contains duplicates")
    if reference_unique.numel() != reference.numel():
        raise RuntimeError(f"{label}: reference selection contains duplicates")
    intersection = torch.isin(actual, reference).sum()
    union = int(actual.numel() + reference.numel() - intersection.item())
    return {
        "label": label,
        "actual_count": int(actual.numel()),
        "reference_count": int(reference.numel()),
        "intersection": int(intersection.item()),
        "jaccard": float(intersection.item() / union),
        "recall_of_reference": float(intersection.item() / reference.numel()),
        "exact": bool(torch.equal(actual, reference)),
        "actual_sha256": _sha256_tensor(actual),
        "reference_sha256": _sha256_tensor(reference),
    }


def _score_delta(
    label: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, Any]:
    actual = actual.float()
    reference = reference.float()
    delta = actual - reference
    actual_centered = actual.double() - actual.double().mean()
    reference_centered = reference.double() - reference.double().mean()
    denominator = torch.sqrt(
        torch.sum(actual_centered.square())
        * torch.sum(reference_centered.square())
    )
    correlation = (
        float(torch.sum(actual_centered * reference_centered).item())
        / float(denominator.item())
        if float(denominator.item()) != 0.0
        else float("nan")
    )
    return {
        "label": label,
        "elements": int(delta.numel()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(delta.square())).item()),
        "pearson": correlation,
        "actual_sha256": _sha256_tensor(actual),
        "reference_sha256": _sha256_tensor(reference),
    }


def _canonical_values(
    values: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    canonical_indices, order = torch.sort(indices.reshape(-1).to(torch.int64))
    canonical_values = torch.gather(values.reshape(-1).float(), -1, order)
    return canonical_indices, canonical_values


def _compare_production_topk(
    *,
    label: str,
    scores: torch.Tensor,
    actual_values: torch.Tensor,
    actual_indices: torch.Tensor,
    reference_values: torch.Tensor,
    reference_indices: torch.Tensor,
) -> dict[str, Any]:
    actual_i, actual_v = _canonical_values(actual_values, actual_indices)
    reference_i, reference_v = _canonical_values(
        reference_values, reference_indices
    )
    source_values = scores[actual_indices.reshape(-1).to(torch.int64)]
    return {
        "label": label,
        "exact_index_set": bool(torch.equal(actual_i, reference_i)),
        "exact_values_by_index": bool(torch.equal(actual_v, reference_v)),
        "actual_values_match_source_bitwise": bool(
            torch.equal(actual_values.reshape(-1), source_values)
        ),
        "actual_indices_sha256": _sha256_tensor(actual_i),
        "reference_indices_sha256": _sha256_tensor(reference_i),
        "actual_values_sha256": _sha256_tensor(actual_v),
        "reference_values_sha256": _sha256_tensor(reference_v),
        "pass": bool(
            torch.equal(actual_i, reference_i)
            and torch.equal(actual_v, reference_v)
            and torch.equal(actual_values.reshape(-1), source_values)
        ),
    }


def _characterize_cutoff(
    scores: torch.Tensor,
    selected_values: torch.Tensor,
    *,
    topk: int,
) -> dict[str, Any]:
    cutoff = selected_values.reshape(-1).min()
    strict_winners = int((scores > cutoff).sum().item())
    rows_at_cutoff = int((scores == cutoff).sum().item())
    selected_at_cutoff = int(
        (selected_values.reshape(-1) == cutoff).sum().item()
    )
    needed_at_cutoff = topk - strict_winners
    return {
        "cutoff_value": float(cutoff.item()),
        "strict_winners": strict_winners,
        "rows_at_cutoff": rows_at_cutoff,
        "selected_at_cutoff": selected_at_cutoff,
        "needed_at_cutoff": needed_at_cutoff,
        "tie_at_selection_boundary": rows_at_cutoff > needed_at_cutoff,
        "unique_cutoff_row": rows_at_cutoff == 1 and needed_at_cutoff == 1,
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    from sparkinfer.attention.nsa_indexer import tiled_topk
    from sparkinfer.attention.nsa_indexer.tiled_topk import (
        run_row_topk,
        run_tiled_topk,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    trace = _load_trace(args.trace_dir)
    topk = int(args.topk)
    width = int(trace["positions"].numel())
    if topk <= 0 or topk > width:
        raise ValueError(f"invalid topk={topk}")
    if bool(
        (trace["runtime_indices"] < 0).any()
        or (trace["runtime_indices"] >= width).any()
    ):
        raise RuntimeError("runtime selection contains out-of-range indices")

    official = _build_official_qk(
        trace,
        device=device,
        rope_theta=float(args.rope_theta),
        chunk_rows=int(args.chunk_rows),
    )
    tail = trace["tail"]
    head_weight = tail["projection_weight_bf16"][HEAD_DIM:].to(
        device=device, dtype=torch.float32
    )
    hidden = tail["hidden_final_bf16"].to(device=device, dtype=torch.float32)
    weights_official = F.linear(hidden, head_weight)
    weights_official.mul_(32**-0.5)
    weights_runtime = tail["weights_fused_final_bf16"].float().to(device)
    weights_runtime.mul_(32**-0.5)

    q_runtime = tail["q_post_rope_final_bf16"]
    k_runtime = trace["k_post_runtime"]
    q_official = official["q_official"]
    k_official = official["k_official"]
    score_variants = {
        "runtime_bf16_preprocessing": _score(
            q=q_runtime,
            k=k_runtime,
            weights=weights_runtime,
            device=device,
            chunk_rows=int(args.chunk_rows),
        ),
        "official_qk_runtime_weights": _score(
            q=q_official,
            k=k_official,
            weights=weights_runtime,
            device=device,
            chunk_rows=int(args.chunk_rows),
        ),
        "runtime_qk_official_weights": _score(
            q=q_runtime,
            k=k_runtime,
            weights=weights_official,
            device=device,
            chunk_rows=int(args.chunk_rows),
        ),
        "full_official": _score(
            q=q_official,
            k=k_official,
            weights=weights_official,
            device=device,
            chunk_rows=int(args.chunk_rows),
        ),
    }
    selected = {
        name: torch.topk(
            scores,
            topk,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        for name, scores in score_variants.items()
    }
    full_scores = score_variants["full_official"]
    torch_values, torch_indices = torch.topk(
        full_scores.view(1, -1),
        topk,
        dim=-1,
        largest=True,
        sorted=False,
    )

    lengths = torch.tensor([width], dtype=torch.int32, device=device)
    row_values, row_indices = run_row_topk(
        row_logits=full_scores.view(1, -1).contiguous(),
        lengths=lengths,
        topk=topk,
    )
    num_k_tiles = math.ceil(width / TILED_BLOCK_K)
    padded = torch.full(
        (num_k_tiles * TILED_BLOCK_K,),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    padded[:width].copy_(full_scores)
    tile_logits = torch.full(
        (num_k_tiles, TILED_BLOCK_Q, TILED_BLOCK_K),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    tile_logits[:, 0, :].copy_(padded.view(num_k_tiles, TILED_BLOCK_K))
    tiled_values, tiled_indices = run_tiled_topk(
        tile_logits=tile_logits.reshape(-1).contiguous(),
        k_start=torch.zeros((1,), dtype=torch.int32, device=device),
        k_end=lengths.clone(),
        topk=topk,
        block_q=TILED_BLOCK_Q,
        block_k=TILED_BLOCK_K,
        num_k_tiles=num_k_tiles,
    )
    torch.cuda.synchronize(device)

    q_official_runtime_order = torch.cat(
        [
            _official_to_vllm_interleaved(
                q_official[:, :ROPE_DIM]
            ),
            q_official[:, ROPE_DIM:],
        ],
        dim=-1,
    )
    k_official_runtime_order = torch.cat(
        [
            _official_to_vllm_interleaved(
                k_official[:, :ROPE_DIM]
            ),
            k_official[:, ROPE_DIM:],
        ],
        dim=-1,
    )
    stage_deltas = [
        _tensor_delta(
            "q_projection_separate_vs_runtime",
            tail["q_pre_separate_final_bf16"],
            tail["q_pre_rope_final_bf16"],
        ),
        _tensor_delta(
            "k_projection_separate_vs_fused",
            trace["k_raw_separate"],
            trace["k_raw_fused"],
        ),
        _tensor_delta(
            "k_layernorm_official_vs_runtime",
            official["k_norm_official"],
            trace["k_post_norm_runtime"],
        ),
        _tensor_delta(
            "q_rope_official_vs_runtime_after_order_mapping",
            q_official_runtime_order.cpu(),
            q_runtime,
        ),
        _tensor_delta(
            "k_rope_official_vs_runtime_after_order_mapping",
            k_official_runtime_order.cpu(),
            k_runtime,
        ),
        _tensor_delta(
            "learned_head_weights_official_fp32_vs_runtime_bf16",
            weights_official.cpu(),
            weights_runtime.cpu(),
        ),
    ]
    first_divergence = next(
        (item["label"] for item in stage_deltas if not item["bit_exact"]),
        None,
    )

    runtime_indices = trace["runtime_indices"].to(device)
    selection_deltas = [
        _selection_metrics(
            f"{name}_vs_full_official",
            indices,
            selected["full_official"],
        )
        for name, indices in selected.items()
        if name != "full_official"
    ]
    selection_deltas.append(
        _selection_metrics(
            "captured_runtime_fp8_selection_vs_full_official",
            runtime_indices,
            selected["full_official"],
        )
    )
    score_deltas = [
        _score_delta(
            f"{name}_vs_full_official",
            scores,
            full_scores,
        )
        for name, scores in score_variants.items()
        if name != "full_official"
    ]
    production_comparisons = [
        _compare_production_topk(
            label="production_row_topk_vs_torch",
            scores=full_scores,
            actual_values=row_values,
            actual_indices=row_indices,
            reference_values=torch_values,
            reference_indices=torch_indices,
        ),
        _compare_production_topk(
            label="production_tiled_topk_vs_torch",
            scores=full_scores,
            actual_values=tiled_values,
            actual_indices=tiled_indices,
            reference_values=torch_values,
            reference_indices=torch_indices,
        ),
    ]
    module_path = Path(inspect.getsourcefile(tiled_topk) or tiled_topk.__file__)
    report = {
        "schema": "v20-glm-official-scorer-raw-production-topk-proof-v1",
        "claim_boundary": (
            "frozen real pre-normalization/pre-RoPE layer activation; complete "
            "official GLM indexer scorer and production top-k replay"
        ),
        "reference": {
            "transformers_source_sha256": args.transformers_source_sha256,
            "model_config_sha256": args.model_config_sha256,
            "formula": (
                "separate BF16 q/k projections -> BF16 LayerNorm(k) -> GLM "
                "interleaved RoPE -> q.float@k.float.T * 128**-0.5 -> ReLU "
                "-> FP32 learned head projection * 32**-0.5 -> head sum -> "
                "exact top-2048"
            ),
            "tensor_contract": (
                "literal Transformers ranks q=[B,S,H,D], k=[B,T,D], "
                "scores=[B,S,H,T], weights=[B,S,H]"
            ),
            "rope_theta": float(args.rope_theta),
            "tf32": False,
        },
        "trace": {
            "directory": str(args.trace_dir),
            "schema": ACTIVATION_SCHEMA,
            "selection_schema": SELECTION_SCHEMA,
            "layer": trace["layer"],
            "rank": trace["rank"],
            "chunks": len(trace["chunk_paths"]),
            "positions": width,
            "first_position": int(trace["positions"][0].item()),
            "query_position": int(trace["positions"][-1].item()),
            "manifest_sha256": trace["manifest_sha256"],
        },
        "stage_deltas": stage_deltas,
        "first_bitwise_divergence": first_divergence,
        "scores": {
            name: {
                "sha256": _sha256_tensor(scores),
                "finite": int(torch.isfinite(scores).sum().item()),
            }
            for name, scores in score_variants.items()
        },
        "score_deltas": score_deltas,
        "selection_deltas": selection_deltas,
        "production_kernel": {
            "module": str(module_path),
            "module_sha256": _sha256_file(module_path),
            "proof_harness_sha256": _sha256_file(Path(__file__)),
            "row_entrypoint": "run_row_topk",
            "tiled_entrypoint": "run_tiled_topk",
            "tiled_block_q": TILED_BLOCK_Q,
            "tiled_block_k": TILED_BLOCK_K,
            "cutoff": _characterize_cutoff(
                full_scores,
                torch_values,
                topk=topk,
            ),
            "comparisons": production_comparisons,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "pass": all(item["pass"] for item in production_comparisons),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--chunk-rows", type=int, default=16384)
    parser.add_argument("--rope-theta", type=float, default=DEFAULT_ROPE_THETA)
    parser.add_argument(
        "--transformers-source-sha256",
        default="adb8317a21716b01273046e46c807f14f0dbaf035af59b60d52bd6bc3007cf72",
    )
    parser.add_argument(
        "--model-config-sha256",
        default="254974797e9f455716a30ab5505ba68272181b20b58a3693e54f94fb8056f3ef",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
