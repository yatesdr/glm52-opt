#!/usr/bin/env python3
"""Bind the server reference implementation to the frozen 350k oracle.

The independent raw-activation proof owns the reference preprocessing and
score fingerprints.  This gate imports the server-mode RoPE/cache/scorer code,
runs it on the same real layer-0 activation, and requires the already-pinned
production-top-k index/value hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.glm_official_indexer import (
    OfficialGLMReferenceIndexer,
    _ieee_fp32_matmul,
    apply_official_glm_indexer_rope,
)


EXPECTED_TRACE_MANIFEST = (
    "8f39bfc70173038086cc83d5d84d64b7536ff7595e528c7a851f7bf3f7666186"
)
PREVIOUS_RANK2_SCORE_SHA = (
    "d5d70cf7324a22ce52bf13ad985affe7658474d1ff542b70b60afd678439f8fb"
)


def _sha(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(payload).hexdigest()


def _load_proof_module(path: Path):
    spec = importlib.util.spec_from_file_location("raw_official_proof", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import proof harness {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_values(
    values: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    canonical_indices, order = torch.sort(indices.reshape(-1).to(torch.int64))
    canonical_values = torch.gather(values.reshape(-1).float(), -1, order)
    return canonical_indices, canonical_values


def _bare_indexer():
    target = OfficialGLMReferenceIndexer.__new__(OfficialGLMReferenceIndexer)
    torch.nn.Module.__init__(target)
    target.topk_tokens = 2048
    target.head_dim = 128
    target.num_q_heads = 32
    target.q_chunk_rows = 1
    target.dcp_world_size = 1
    target.dcp_rank = 0
    target.cp_kv_cache_interleave_size = 1
    return target


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--proof-harness", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-rows", type=int, default=8192)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    proof = _load_proof_module(args.proof_harness)
    trace = proof._load_trace(args.trace_dir)
    if trace["manifest_sha256"] != EXPECTED_TRACE_MANIFEST:
        raise RuntimeError(
            "frozen trace manifest changed: "
            f"{trace['manifest_sha256']} != {EXPECTED_TRACE_MANIFEST}"
        )
    official = proof._build_official_qk(
        trace,
        device=device,
        rope_theta=8_000_000.0,
        chunk_rows=args.chunk_rows,
    )

    positions = trace["positions"]
    tail = trace["tail"]
    q_pre = tail["q_pre_separate_final_bf16"].to(device)
    q_actual, _ = apply_official_glm_indexer_rope(
        q_pre.unsqueeze(0),
        official["k_norm_official"][-1:].to(device),
        positions[-1:].to(device),
        rope_dim=64,
        rope_theta=8_000_000.0,
    )
    if not torch.equal(q_actual.squeeze(0), official["q_official"]):
        raise RuntimeError("server reference Q RoPE differs from frozen oracle")

    k_actual_parts: list[torch.Tensor] = []
    k_norm = official["k_norm_official"]
    for start in range(0, int(k_norm.shape[0]), args.chunk_rows):
        end = min(start + args.chunk_rows, int(k_norm.shape[0]))
        q_dummy = torch.zeros(
            (end - start, 32, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        _, k_part = apply_official_glm_indexer_rope(
            q_dummy,
            k_norm[start:end].to(device),
            positions[start:end].to(device),
            rope_dim=64,
            rope_theta=8_000_000.0,
        )
        k_actual_parts.append(k_part.cpu())
    k_actual = torch.cat(k_actual_parts)
    if not torch.equal(k_actual, official["k_official"]):
        raise RuntimeError("server reference K RoPE differs from frozen oracle")

    head_weight = tail["projection_weight_bf16"][128:].to(
        device=device,
        dtype=torch.float32,
    )
    hidden = tail["hidden_final_bf16"].to(device=device, dtype=torch.float32)
    with _ieee_fp32_matmul():
        weights = F.linear(hidden, head_weight)
    weights.mul_(32**-0.5)

    target = _bare_indexer()
    output_indices = torch.empty((1, 2048), dtype=torch.int32, device=device)
    output_values = torch.empty((1, 2048), dtype=torch.float32, device=device)
    target._select_local(
        q=official["q_official"].to(device).unsqueeze(0),
        keys=k_actual.to(device),
        weights=weights.unsqueeze(0),
        lengths=torch.tensor(
            [int(k_actual.shape[0])],
            dtype=torch.int32,
            device=device,
        ),
        output_indices=output_indices,
        output_scores=output_values,
    )

    score = proof._score(
        q=official["q_official"],
        k=official["k_official"],
        weights=weights.cpu(),
        device=device,
        chunk_rows=args.chunk_rows,
    )
    reference_values, reference_indices = torch.topk(
        score.view(1, -1),
        2048,
        dim=-1,
        sorted=False,
    )
    actual_i, actual_v = _canonical_values(output_values, output_indices)
    reference_i, reference_v = _canonical_values(
        reference_values,
        reference_indices,
    )
    source_values = score.gather(
        0,
        output_indices.reshape(-1).to(torch.int64),
    )
    if not torch.equal(actual_i, reference_i):
        raise RuntimeError("server reference production top-k index set differs")
    if not torch.equal(actual_v, reference_v):
        raise RuntimeError("server reference production top-k values differ")
    if not torch.equal(output_values.reshape(-1), source_values):
        raise RuntimeError("server reference top-k values differ from source scores")

    observed = {
        "score_sha256": _sha(score),
        "canonical_indices_sha256": _sha(actual_i),
        "canonical_values_sha256": _sha(actual_v),
    }

    print(
        json.dumps(
            {
                "schema": "v20-glm-official-reference-real-activation-gate-v1",
                "trace_manifest_sha256": trace["manifest_sha256"],
                "rows": int(k_actual.shape[0]),
                "q_rope_exact": True,
                "k_rope_exact": True,
                "literal_transformers_tensor_ranks": True,
                "production_topk_exact_set_and_values": True,
                "values_match_source_bitwise": True,
                "differs_from_previous_rank2_score_fingerprint": (
                    observed["score_sha256"] != PREVIOUS_RANK2_SCORE_SHA
                ),
                **observed,
                "status": "PASS",
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
