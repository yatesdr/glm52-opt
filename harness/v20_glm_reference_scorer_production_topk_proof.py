#!/usr/bin/env python3
"""Feed frozen real GLM indexer scores through production exact top-k kernels.

This proof consumes the diagnostic trace written immediately after indexer K
normalization, GLM interleaved RoPE, and 128-D Q/K concatenation. It reconstructs
the remaining official GLM scorer in FP32:

    relu((Q.float @ K.float.T) * 128**-0.5)
    -> learned per-head weight
    -> sum over heads
    -> causal mask
    -> top-2048

The identical score row is selected by:

* ``torch.topk``;
* SparkInfer's production row-major exact top-k kernel; and
* SparkInfer's production tiled exact top-k kernel.

This closes the scorer-suffix/top-k question on immutable real activations. It
does not claim to re-execute K normalization or RoPE from pre-transform inputs;
the report marks that boundary explicitly. A separate raw-activation capture is
required for the complete official-scorer proof.
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


ACTIVATION_SCHEMA = "v20-indexer-prequant-activation-v1"
HEAD_DIM = 128
DEFAULT_TOPK = 2048
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

    keys: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    q_final: torch.Tensor | None = None
    weights_final: torch.Tensor | None = None
    tail_count = 0
    layer: int | None = None
    rank: int | None = None
    expected_chunk = 0
    for path in chunk_paths:
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record.get("schema") != ACTIVATION_SCHEMA:
            raise RuntimeError(f"{path}: wrong schema {record.get('schema')!r}")
        if int(record["chunk"]) != expected_chunk:
            raise RuntimeError(
                f"{path}: expected chunk {expected_chunk}, got {record['chunk']}"
            )
        expected_chunk += 1
        record_layer = int(record["layer"])
        record_rank = int(record["tp_rank"])
        if layer is None:
            layer = record_layer
            rank = record_rank
        elif record_layer != layer or record_rank != rank:
            raise RuntimeError(f"{path}: layer/rank changed within trace")

        key = record["k_bf16"]
        pos = record["positions"].to(torch.int64)
        if key.dtype != torch.bfloat16:
            raise RuntimeError(f"{path}: expected BF16 K, got {key.dtype}")
        if key.ndim != 2 or int(key.shape[1]) != HEAD_DIM:
            raise RuntimeError(f"{path}: invalid K shape {tuple(key.shape)}")
        if pos.ndim != 1 or int(pos.numel()) != int(key.shape[0]):
            raise RuntimeError(f"{path}: position/K mismatch")
        keys.append(key.contiguous())
        positions.append(pos.contiguous())

        if "q_final_bf16" in record:
            tail_count += 1
            q_final = record["q_final_bf16"].contiguous()
            weights_final = record["weights_final_bf16"].contiguous()

    if tail_count != 1 or q_final is None or weights_final is None:
        raise RuntimeError(f"expected exactly one final Q/weight record, got {tail_count}")
    if q_final.dtype != torch.bfloat16 or q_final.ndim != 2:
        raise RuntimeError(f"invalid final Q: {q_final.dtype} {tuple(q_final.shape)}")
    if int(q_final.shape[1]) != HEAD_DIM:
        raise RuntimeError(f"invalid final Q width {int(q_final.shape[1])}")
    if weights_final.ndim != 1 or int(weights_final.numel()) != int(q_final.shape[0]):
        raise RuntimeError("final learned-head weights do not match Q heads")

    key_all = torch.cat(keys, dim=0).contiguous()
    pos_all = torch.cat(positions, dim=0).contiguous()
    if int(torch.unique(pos_all).numel()) != int(pos_all.numel()):
        raise RuntimeError("trace positions contain duplicates")
    if int(pos_all.numel()) > 1 and not bool(torch.all(pos_all[1:] == pos_all[:-1] + 1)):
        raise RuntimeError("trace positions are not strictly contiguous")
    query_position = int(pos_all[-1].item())
    causal = pos_all <= query_position
    if not bool(causal.all()):
        raise RuntimeError("trace contains post-query K positions")

    manifest = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in chunk_paths
    ]
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "chunk_paths": chunk_paths,
        "manifest": manifest,
        "manifest_sha256": manifest_digest,
        "layer": layer,
        "rank": rank,
        "positions": pos_all,
        "k": key_all,
        "q": q_final,
        "weights": weights_final,
        "query_position": query_position,
    }


@torch.inference_mode()
def _official_scorer_suffix(
    *,
    q_post_rope: torch.Tensor,
    k_post_rope: torch.Tensor,
    learned_weights: torch.Tensor,
    positions: torch.Tensor,
    query_position: int,
    device: torch.device,
    chunk_rows: int,
) -> torch.Tensor:
    """Official GLM score order from post-normalization/post-RoPE activations."""

    q = q_post_rope.to(device=device, dtype=torch.float32)
    head_count = int(q.shape[0])
    weights = learned_weights.to(device=device, dtype=torch.float32)
    weights = weights * (head_count**-0.5)
    scale = HEAD_DIM**-0.5
    scores = torch.empty(
        (int(k_post_rope.shape[0]),),
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, int(k_post_rope.shape[0]), chunk_rows):
        end = min(start + chunk_rows, int(k_post_rope.shape[0]))
        keys = k_post_rope[start:end].to(device=device, dtype=torch.float32)
        # Match the official order literally: FP32 Q.K, scale, ReLU, learned
        # head weighting, sum across heads.
        per_head = torch.matmul(q, keys.T)
        per_head.mul_(scale)
        per_head.relu_()
        scores[start:end] = torch.matmul(weights, per_head)
    positions_gpu = positions.to(device=device)
    scores.masked_fill_(positions_gpu > query_position, float("-inf"))
    if bool(torch.isnan(scores).any()):
        raise RuntimeError("official scorer produced NaN")
    return scores


def _canonical_by_index(
    values: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    canonical_indices, order = torch.sort(indices.to(torch.int64))
    canonical_values = torch.gather(values.float(), -1, order)
    return canonical_indices, canonical_values


def _compare_selection(
    *,
    label: str,
    scores: torch.Tensor,
    actual_values: torch.Tensor,
    actual_indices: torch.Tensor,
    reference_values: torch.Tensor,
    reference_indices: torch.Tensor,
) -> dict[str, Any]:
    width = int(scores.numel())
    actual_values = actual_values.reshape(-1)
    actual_indices = actual_indices.reshape(-1).to(torch.int64)
    reference_values = reference_values.reshape(-1)
    reference_indices = reference_indices.reshape(-1).to(torch.int64)
    if bool(((actual_indices < 0) | (actual_indices >= width)).any()):
        raise RuntimeError(f"{label}: out-of-range production index")

    actual_i, actual_v = _canonical_by_index(actual_values, actual_indices)
    reference_i, reference_v = _canonical_by_index(
        reference_values, reference_indices
    )
    exact_set = bool(torch.equal(actual_i, reference_i))
    exact_values_by_index = exact_set and bool(torch.equal(actual_v, reference_v))
    source_values = scores[actual_indices]
    actual_values_match_source = bool(torch.equal(actual_values, source_values))

    cutoff = reference_values.min()
    strict_indices = torch.nonzero(scores > cutoff, as_tuple=False).reshape(-1)
    tie_indices = torch.nonzero(scores == cutoff, as_tuple=False).reshape(-1)
    strict_in_actual = bool(torch.isin(strict_indices, actual_indices).all())
    strict_in_reference = bool(torch.isin(strict_indices, reference_indices).all())
    actual_only = actual_indices[~torch.isin(actual_indices, reference_indices)]
    reference_only = reference_indices[~torch.isin(reference_indices, actual_indices)]
    actual_only_are_cutoff_ties = bool(
        actual_only.numel() == 0 or torch.all(scores[actual_only] == cutoff)
    )
    reference_only_are_cutoff_ties = bool(
        reference_only.numel() == 0 or torch.all(scores[reference_only] == cutoff)
    )
    outside_ties_exact = (
        strict_in_actual
        and strict_in_reference
        and actual_only_are_cutoff_ties
        and reference_only_are_cutoff_ties
    )
    pass_result = (
        outside_ties_exact
        and actual_values_match_source
        and (exact_values_by_index or not exact_set)
    )
    return {
        "label": label,
        "pass": pass_result,
        "exact_index_set": exact_set,
        "exact_values_by_index": exact_values_by_index,
        "actual_values_match_source_bitwise": actual_values_match_source,
        "outside_cutoff_ties_exact": outside_ties_exact,
        "cutoff_value": float(cutoff.item()),
        "strict_winner_count": int(strict_indices.numel()),
        "cutoff_tie_count": int(tie_indices.numel()),
        "cutoff_slots": int(reference_indices.numel() - strict_indices.numel()),
        "actual_only_count": int(actual_only.numel()),
        "reference_only_count": int(reference_only.numel()),
        "actual_indices_sha256": _sha256_tensor(actual_i),
        "actual_values_sha256": _sha256_tensor(actual_v),
        "reference_indices_sha256": _sha256_tensor(reference_i),
        "reference_values_sha256": _sha256_tensor(reference_v),
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
    if topk <= 0 or topk > int(trace["k"].shape[0]):
        raise ValueError(f"invalid topk={topk}")
    scores = _official_scorer_suffix(
        q_post_rope=trace["q"],
        k_post_rope=trace["k"],
        learned_weights=trace["weights"],
        positions=trace["positions"],
        query_position=int(trace["query_position"]),
        device=device,
        chunk_rows=int(args.chunk_rows),
    )
    score_row = scores.view(1, -1).contiguous()
    torch_values, torch_indices = torch.topk(
        score_row, topk, dim=-1, largest=True, sorted=False
    )

    lengths = torch.tensor(
        [int(scores.numel())], dtype=torch.int32, device=device
    )
    row_values, row_indices = run_row_topk(
        row_logits=score_row,
        lengths=lengths,
        topk=topk,
    )
    torch.cuda.synchronize(device)

    width = int(scores.numel())
    num_k_tiles = math.ceil(width / TILED_BLOCK_K)
    padded = torch.full(
        (num_k_tiles * TILED_BLOCK_K,),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    padded[:width].copy_(scores)
    # Production tiled layout:
    # (q_tile, k_tile, q_local, k_local). Only q_local=0 is active here.
    tile_logits = torch.full(
        (num_k_tiles, TILED_BLOCK_Q, TILED_BLOCK_K),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    tile_logits[:, 0, :].copy_(padded.view(num_k_tiles, TILED_BLOCK_K))
    k_start = torch.zeros((1,), dtype=torch.int32, device=device)
    k_end = lengths.clone()
    tiled_values, tiled_indices = run_tiled_topk(
        tile_logits=tile_logits.reshape(-1).contiguous(),
        k_start=k_start,
        k_end=k_end,
        topk=topk,
        block_q=TILED_BLOCK_Q,
        block_k=TILED_BLOCK_K,
        num_k_tiles=num_k_tiles,
    )
    torch.cuda.synchronize(device)

    comparisons = [
        _compare_selection(
            label="production_row_topk_vs_torch",
            scores=scores,
            actual_values=row_values,
            actual_indices=row_indices,
            reference_values=torch_values,
            reference_indices=torch_indices,
        ),
        _compare_selection(
            label="production_tiled_topk_vs_torch",
            scores=scores,
            actual_values=tiled_values,
            actual_indices=tiled_indices,
            reference_values=torch_values,
            reference_indices=torch_indices,
        ),
    ]
    module_path = Path(inspect.getsourcefile(tiled_topk) or tiled_topk.__file__)
    report = {
        "schema": "v20-glm-reference-scorer-production-topk-proof-v1",
        "claim_boundary": (
            "real frozen post-K-normalization/post-interleaved-RoPE activations; "
            "official FP32 scorer suffix and production top-k proven; raw K "
            "normalization/RoPE reconstruction remains separate"
        ),
        "trace": {
            "directory": str(args.trace_dir),
            "schema": ACTIVATION_SCHEMA,
            "layer": trace["layer"],
            "rank": trace["rank"],
            "chunks": len(trace["chunk_paths"]),
            "manifest_sha256": trace["manifest_sha256"],
            "positions": int(trace["positions"].numel()),
            "first_position": int(trace["positions"][0].item()),
            "query_position": int(trace["query_position"]),
            "heads": int(trace["q"].shape[0]),
            "head_dim": int(trace["q"].shape[1]),
            "q_sha256": _sha256_tensor(trace["q"]),
            "k_sha256": _sha256_tensor(trace["k"]),
            "learned_weights_sha256": _sha256_tensor(trace["weights"]),
            "positions_sha256": _sha256_tensor(trace["positions"]),
        },
        "scorer": {
            "formula": (
                "matmul(q.float,k.float.T) * 128**-0.5 -> relu -> "
                "weights.float * heads**-0.5 -> head sum -> causal mask"
            ),
            "tf32": False,
            "score_count": width,
            "scores_sha256": _sha256_tensor(scores),
            "finite_scores": int(torch.isfinite(scores).sum().item()),
            "topk": topk,
        },
        "production_kernel": {
            "module": str(module_path),
            "module_sha256": _sha256_file(module_path),
            "row_entrypoint": "run_row_topk",
            "tiled_entrypoint": "run_tiled_topk",
            "tiled_block_q": TILED_BLOCK_Q,
            "tiled_block_k": TILED_BLOCK_K,
            "tiled_k_tiles": num_k_tiles,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "comparisons": comparisons,
        "pass": all(item["pass"] for item in comparisons),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--chunk-rows", type=int, default=16384)
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
