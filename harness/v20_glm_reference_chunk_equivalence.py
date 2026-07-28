#!/usr/bin/env python3
"""Prove that reducing reference Q chunking preserves scorer/top-k output.

The diagnostic official scorer uses bounded Q-row chunks to avoid retaining a
full [S,H,T] score tensor. This proof compares the original 64-row chunk with
the proposed 16-row resource-fit chunk at production DCP-local geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import torch

from vllm.model_executor.layers.glm_official_indexer import (
    OfficialGLMReferenceIndexer,
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(payload).hexdigest()


def bare_indexer(*, topk: int, q_chunk_rows: int):
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
def run_once(
    *,
    q: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    target = bare_indexer(topk=topk, q_chunk_rows=chunk_rows)
    indices = torch.empty(
        (q.shape[0], topk), dtype=torch.int32, device=q.device
    )
    values = torch.empty(
        (q.shape[0], topk), dtype=torch.float32, device=q.device
    )
    torch.cuda.synchronize(q.device)
    torch.cuda.reset_peak_memory_stats(q.device)
    started = time.monotonic()
    target._select_local(
        q=q,
        keys=keys,
        weights=weights,
        lengths=lengths,
        output_indices=indices,
        output_scores=values,
    )
    torch.cuda.synchronize(q.device)
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(q.device))
    return indices, values, elapsed, peak


def canonicalize(
    indices: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(indices.to(torch.int64), dim=-1)
    return (
        torch.gather(indices, -1, order),
        torch.gather(values, -1, order),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--key-rows", type=int, default=85_932)
    parser.add_argument("--topk", type=int, default=2_048)
    parser.add_argument("--reference-chunk", type=int, default=64)
    parser.add_argument("--candidate-chunk", type=int, default=16)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2026072701, 2026072702, 2026072703]
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cases: list[dict[str, object]] = []
    all_exact = True

    for seed in args.seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        q = torch.randn(
            (args.rows, 32, 128),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        keys = torch.randn(
            (args.key_rows, 128),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        weights = torch.randn(
            (args.rows, 32),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        lengths = torch.arange(
            args.key_rows - args.rows + 1,
            args.key_rows + 1,
            dtype=torch.int32,
            device=device,
        )

        candidate = run_once(
            q=q,
            keys=keys,
            weights=weights,
            lengths=lengths,
            topk=args.topk,
            chunk_rows=args.candidate_chunk,
        )
        candidate_indices, candidate_values = canonicalize(
            candidate[0], candidate[1]
        )
        torch.cuda.empty_cache()
        reference = run_once(
            q=q,
            keys=keys,
            weights=weights,
            lengths=lengths,
            topk=args.topk,
            chunk_rows=args.reference_chunk,
        )
        reference_indices, reference_values = canonicalize(
            reference[0], reference[1]
        )

        membership_equal = torch.equal(
            candidate_indices, reference_indices
        )
        values_bitwise_equal = torch.equal(
            candidate_values, reference_values
        )
        changed_rows = int(
            torch.any(candidate_indices != reference_indices, dim=-1).sum()
        )
        finite = torch.isfinite(candidate_values) & torch.isfinite(
            reference_values
        )
        max_abs = (
            float(
                (candidate_values[finite] - reference_values[finite])
                .abs()
                .max()
                .item()
            )
            if bool(finite.any())
            else 0.0
        )
        exact = membership_equal and values_bitwise_equal
        all_exact &= exact
        cases.append(
            {
                "seed": seed,
                "membership_equal": membership_equal,
                "values_bitwise_equal": values_bitwise_equal,
                "changed_rows": changed_rows,
                "max_abs_score_delta": max_abs,
                "candidate_indices_sha256": tensor_sha256(candidate_indices),
                "reference_indices_sha256": tensor_sha256(reference_indices),
                "candidate_values_sha256": tensor_sha256(candidate_values),
                "reference_values_sha256": tensor_sha256(reference_values),
                "candidate_elapsed_s": round(candidate[2], 4),
                "reference_elapsed_s": round(reference[2], 4),
                "candidate_peak_allocated_bytes": candidate[3],
                "reference_peak_allocated_bytes": reference[3],
            }
        )
        del (
            q,
            keys,
            weights,
            lengths,
            candidate_indices,
            candidate_values,
            reference_indices,
            reference_values,
            candidate,
            reference,
        )
        torch.cuda.empty_cache()

    report = {
        "schema": "v20-glm-reference-chunk-equivalence-v1",
        "status": "PASS" if all_exact else "FAIL",
        "device": str(device),
        "rows": args.rows,
        "key_rows": args.key_rows,
        "heads": 32,
        "head_dim": 128,
        "topk": args.topk,
        "reference_chunk": args.reference_chunk,
        "candidate_chunk": args.candidate_chunk,
        "cases": cases,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
