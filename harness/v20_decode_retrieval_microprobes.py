#!/usr/bin/env python3
"""No-model GPU probes for the v20 decode/long-context regression.

Run this inside the exact v20 candidate image after the serving process has
stopped.  It exercises two source deltas without paying for a model boot:

1. BF16-weight fused MLA query assembly versus the prior staged
   ``torch.bmm + concat + FP8 quantize`` path.
2. The long-context paged two-level top-k fold at production-scale widths.

The script prints one JSON record per case and a final summary.  A top-k
disagreement exits non-zero.  Fused/staged differences are reported rather
than treated as a process failure because the purpose of that probe is to
measure whether the new reduction order crosses FP8 bins.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FusedQueryResult:
    heads: int
    m: int
    seed: int
    q_scale: float
    bf16_exact_fraction: float
    bf16_max_abs_error: float
    fp8_mismatches: int
    fp8_elements: int
    fp8_mismatch_fraction: float
    fp8_max_abs_error: float
    rope_suffix_mismatches: int
    bf16_static_fp8_mismatches: int
    bf16_static_fp8_mismatch_fraction: float
    bf16_static_fp8_max_abs_error: float
    direct_vs_bf16_static_fp8_mismatches: int
    direct_vs_bf16_static_fp8_mismatch_fraction: float
    retrieval_seq_len: int
    retrieval_topk: int
    retrieval_overlap_fraction: float
    bf16_static_retrieval_overlap_fraction: float


@dataclass(frozen=True)
class FusedQueryTimingResult:
    heads: int
    m: int
    seed: int
    q_scale: float
    iterations: int
    rounds: int
    staged_median_us: float
    fused_bf16_static_median_us: float
    fused_median_us: float
    fused_bf16_static_over_staged: float
    fused_over_staged: float


@dataclass(frozen=True)
class TopKResult:
    seq_len: int
    rows: int
    topk: int
    block_q: int
    block_k: int
    supertile_k: int
    path: str
    score_mode: str
    out_of_range: int
    duplicate_rows: int
    below_threshold: int
    value_mismatches: int
    passed: bool


def _emit(kind: str, payload: Any) -> None:
    body = asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
    print(json.dumps({"kind": kind, **body}, sort_keys=True), flush=True)


def _load_vllm_custom_ops() -> None:
    # Serving workers import the platform kernels before constructing MLA.
    # A bare proof entrypoint does not, so reproduce that registration step
    # before resolving torch.ops._C.
    import vllm._custom_ops  # noqa: F401

    required = ("safe_mla_query_bmm", "static_scaled_fp8_quant")
    missing = [name for name in required if not hasattr(torch.ops._C, name)]
    if missing:
        raise RuntimeError(
            "vLLM custom-op registration did not expose required operators: "
            + ", ".join(missing)
        )


def _capture_graph(fn) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _graph_median_us(
    graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
    rounds: int,
) -> float:
    samples: list[float] = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples)


def _benchmark_fused_query(
    *,
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: torch.Tensor,
    iterations: int,
    rounds: int,
) -> tuple[float, float, float]:
    from sparkinfer.gemm import mla_query_projection

    heads, m, _ = q_nope.shape
    projected = torch.empty(
        heads,
        m,
        512,
        device=q_nope.device,
        dtype=torch.bfloat16,
    )
    staged_bf16 = torch.empty(
        m,
        heads,
        576,
        device=q_nope.device,
        dtype=torch.bfloat16,
    )
    staged_fp8 = torch.empty(
        m,
        heads,
        576,
        device=q_nope.device,
        dtype=torch.float8_e4m3fn,
    )
    fused_bf16 = torch.empty_like(staged_bf16)
    fused_bf16_static_fp8 = torch.empty_like(staged_fp8)
    fused_out = torch.empty(
        m,
        heads,
        576,
        device=q_nope.device,
        dtype=torch.float8_e4m3fn,
    )
    if not hasattr(torch.ops._C, "safe_mla_query_bmm"):
        raise RuntimeError(
            "the exact staged-path proof requires torch.ops._C.safe_mla_query_bmm"
        )

    def staged() -> None:
        torch.ops._C.safe_mla_query_bmm(q_nope, weight, projected)
        torch.cat(
            (projected.transpose(0, 1), q_pe),
            dim=-1,
            out=staged_bf16,
        )
        torch.ops._C.static_scaled_fp8_quant(
            staged_fp8.view(m, -1),
            staged_bf16.view(m, -1),
            q_scale,
        )

    def fused() -> None:
        mla_query_projection.run(
            q_nope,
            weight,
            q_pe,
            fused_out,
            q_scale=q_scale,
        )

    def fused_bf16_static() -> None:
        mla_query_projection.run(
            q_nope,
            weight,
            q_pe,
            fused_bf16,
        )
        torch.ops._C.static_scaled_fp8_quant(
            fused_bf16_static_fp8.view(m, -1),
            fused_bf16.view(m, -1),
            q_scale,
        )

    staged_graph = _capture_graph(staged)
    fused_bf16_static_graph = _capture_graph(fused_bf16_static)
    fused_graph = _capture_graph(fused)
    staged_us = _graph_median_us(
        staged_graph, iterations=iterations, rounds=rounds
    )
    fused_bf16_static_us = _graph_median_us(
        fused_bf16_static_graph,
        iterations=iterations,
        rounds=rounds,
    )
    fused_us = _graph_median_us(
        fused_graph, iterations=iterations, rounds=rounds
    )
    return staged_us, fused_bf16_static_us, fused_us


def run_fused_query_probe(
    *,
    device: torch.device,
    heads: int,
    m_values: list[int],
    seeds: list[int],
    scales: list[float],
    timing_iterations: int,
    timing_rounds: int,
    retrieval_seq_len: int,
    retrieval_topk: int,
) -> tuple[list[FusedQueryResult], list[FusedQueryTimingResult]]:
    from sparkinfer.gemm import mla_query_projection

    results: list[FusedQueryResult] = []
    timing_results: list[FusedQueryTimingResult] = []
    retrieval_keys: dict[int, torch.Tensor] = {}
    for m in m_values:
        for seed in seeds:
            torch.manual_seed(seed)
            if seed not in retrieval_keys:
                key_generator = torch.Generator(device=device)
                key_generator.manual_seed(0xA77E_0000 + seed)
                retrieval_keys[seed] = (
                    torch.randn(
                        retrieval_seq_len,
                        576,
                        device=device,
                        dtype=torch.bfloat16,
                        generator=key_generator,
                    )
                    * 0.25
                )
            # Reproduce the model's real split-and-transpose view:
            # [M,H,256] -> nope [H,M,192] + RoPE [M,H,64].
            query_storage = (
                torch.randn(
                    m,
                    heads,
                    256,
                    device=device,
                    dtype=torch.bfloat16,
                )
                * 0.5
            )
            q_nope_storage, q_pe = query_storage.split((192, 64), dim=-1)
            q_nope = q_nope_storage.transpose(0, 1)
            assert not q_nope.is_contiguous()
            weight = (
                torch.randn(
                    heads, 192, 512, device=device, dtype=torch.bfloat16
                )
                * 0.05
            )
            staged_projected = torch.empty(
                heads,
                m,
                512,
                device=device,
                dtype=torch.bfloat16,
            )
            if not hasattr(torch.ops._C, "safe_mla_query_bmm"):
                raise RuntimeError(
                    "the exact staged-path proof requires "
                    "torch.ops._C.safe_mla_query_bmm"
                )
            torch.ops._C.safe_mla_query_bmm(
                q_nope,
                weight,
                staged_projected,
            )
            staged_bf16 = torch.cat(
                (staged_projected.transpose(0, 1), q_pe),
                dim=-1,
            )

            fused_bf16 = torch.empty_like(staged_bf16)
            mla_query_projection.run(
                q_nope,
                weight,
                q_pe,
                fused_bf16,
            )

            timing_scale = scales[0]
            timing_q_scale = torch.tensor(
                [timing_scale], device=device, dtype=torch.float32
            )
            staged_us, fused_bf16_static_us, fused_us = _benchmark_fused_query(
                q_nope=q_nope,
                weight=weight,
                q_pe=q_pe,
                q_scale=timing_q_scale,
                iterations=timing_iterations,
                rounds=timing_rounds,
            )
            timing_result = FusedQueryTimingResult(
                heads=heads,
                m=m,
                seed=seed,
                q_scale=timing_scale,
                iterations=timing_iterations,
                rounds=timing_rounds,
                staged_median_us=staged_us,
                fused_bf16_static_median_us=fused_bf16_static_us,
                fused_median_us=fused_us,
                fused_bf16_static_over_staged=(
                    fused_bf16_static_us / staged_us
                ),
                fused_over_staged=fused_us / staged_us,
            )
            timing_results.append(timing_result)
            _emit("fused_query_timing", timing_result)

            for scale_value in scales:
                q_scale = torch.tensor(
                    [scale_value], device=device, dtype=torch.float32
                )
                staged_fp8 = torch.empty_like(
                    staged_bf16,
                    dtype=torch.float8_e4m3fn,
                )
                torch.ops._C.static_scaled_fp8_quant(
                    staged_fp8.view(m, -1),
                    staged_bf16.view(m, -1),
                    q_scale,
                )
                fused_fp8 = torch.empty_like(
                    staged_bf16, dtype=torch.float8_e4m3fn
                )
                fused_bf16_static_fp8 = torch.empty_like(
                    staged_bf16, dtype=torch.float8_e4m3fn
                )
                torch.ops._C.static_scaled_fp8_quant(
                    fused_bf16_static_fp8.view(m, -1),
                    fused_bf16.view(m, -1),
                    q_scale,
                )
                mla_query_projection.run(
                    q_nope,
                    weight,
                    q_pe,
                    fused_fp8,
                    q_scale=q_scale,
                )
                torch.cuda.synchronize(device)

                bf16_equal = fused_bf16 == staged_bf16
                fp8_equal = fused_fp8.view(torch.uint8) == staged_fp8.view(
                    torch.uint8
                )
                bf16_static_fp8_equal = (
                    fused_bf16_static_fp8.view(torch.uint8)
                    == staged_fp8.view(torch.uint8)
                )
                direct_vs_bf16_static_equal = (
                    fused_fp8.view(torch.uint8)
                    == fused_bf16_static_fp8.view(torch.uint8)
                )
                bf16_abs = (
                    fused_bf16.float() - staged_bf16.float()
                ).abs()
                fp8_abs = (
                    fused_fp8.float() - staged_fp8.float()
                ).abs()
                bf16_static_fp8_abs = (
                    fused_bf16_static_fp8.float() - staged_fp8.float()
                ).abs()
                rope_equal = (
                    fused_fp8[..., 512:].view(torch.uint8)
                    == staged_fp8[..., 512:].view(torch.uint8)
                )
                keys = retrieval_keys[seed].float()
                staged_scores = staged_fp8[0].float() @ keys.T
                fused_scores = fused_fp8[0].float() @ keys.T
                fused_bf16_static_scores = (
                    fused_bf16_static_fp8[0].float() @ keys.T
                )
                staged_ids = torch.topk(
                    staged_scores,
                    retrieval_topk,
                    dim=1,
                    sorted=False,
                ).indices
                fused_ids = torch.topk(
                    fused_scores,
                    retrieval_topk,
                    dim=1,
                    sorted=False,
                ).indices
                fused_bf16_static_ids = torch.topk(
                    fused_bf16_static_scores,
                    retrieval_topk,
                    dim=1,
                    sorted=False,
                ).indices
                staged_mask = torch.zeros(
                    heads,
                    retrieval_seq_len,
                    device=device,
                    dtype=torch.bool,
                )
                staged_mask.scatter_(1, staged_ids, True)
                retrieval_overlap = float(
                    staged_mask.gather(1, fused_ids).float().mean().item()
                )
                bf16_static_retrieval_overlap = float(
                    staged_mask.gather(1, fused_bf16_static_ids)
                    .float()
                    .mean()
                    .item()
                )
                result = FusedQueryResult(
                    heads=heads,
                    m=m,
                    seed=seed,
                    q_scale=scale_value,
                    bf16_exact_fraction=float(
                        bf16_equal.float().mean().item()
                    ),
                    bf16_max_abs_error=float(bf16_abs.max().item()),
                    fp8_mismatches=int((~fp8_equal).sum().item()),
                    fp8_elements=int(fp8_equal.numel()),
                    fp8_mismatch_fraction=float(
                        (~fp8_equal).float().mean().item()
                    ),
                    fp8_max_abs_error=float(fp8_abs.max().item()),
                    rope_suffix_mismatches=int((~rope_equal).sum().item()),
                    bf16_static_fp8_mismatches=int(
                        (~bf16_static_fp8_equal).sum().item()
                    ),
                    bf16_static_fp8_mismatch_fraction=float(
                        (~bf16_static_fp8_equal).float().mean().item()
                    ),
                    bf16_static_fp8_max_abs_error=float(
                        bf16_static_fp8_abs.max().item()
                    ),
                    direct_vs_bf16_static_fp8_mismatches=int(
                        (~direct_vs_bf16_static_equal).sum().item()
                    ),
                    direct_vs_bf16_static_fp8_mismatch_fraction=float(
                        (~direct_vs_bf16_static_equal).float().mean().item()
                    ),
                    retrieval_seq_len=retrieval_seq_len,
                    retrieval_topk=retrieval_topk,
                    retrieval_overlap_fraction=retrieval_overlap,
                    bf16_static_retrieval_overlap_fraction=(
                        bf16_static_retrieval_overlap
                    ),
                )
                results.append(result)
                _emit("fused_query", result)
    return results, timing_results


def _make_scores(
    rows: int,
    seq_len: int,
    score_mode: str,
    device: torch.device,
) -> torch.Tensor:
    if score_mode == "monotonic":
        base = torch.linspace(
            0.25, 1.25, seq_len, device=device, dtype=torch.float32
        )
        # Each row has the same strict ordering but a distinct offset.
        return base.unsqueeze(0) + torch.arange(
            rows, device=device, dtype=torch.float32
        ).unsqueeze(1) * 2.0
    if score_mode == "clustered":
        # Many close values stress the coarse/fine radix bins without
        # introducing exact ties.
        base = torch.linspace(
            -2.0e-3, 2.0e-3, seq_len, device=device, dtype=torch.float32
        )
        wobble = torch.sin(
            torch.arange(seq_len, device=device, dtype=torch.float32)
            * 0.0137
        ) * 1.0e-5
        row_shift = torch.arange(
            rows, device=device, dtype=torch.float32
        ).unsqueeze(1) * 0.01
        return base.unsqueeze(0) + wobble.unsqueeze(0) + row_shift
    if score_mode == "random":
        return torch.randn(
            rows, seq_len, device=device, dtype=torch.float32
        )
    if score_mode == "fold_boundary":
        # Make the first 32k region dominate the tail. The two-level slice
        # fold must preserve its winners when combining slice candidates;
        # this route is first entered at 32,768 local tokens.
        positions = torch.arange(
            seq_len, device=device, dtype=torch.float32
        )
        first = positions < 32768
        base = torch.where(
            first,
            1.0 + positions * 1.0e-7,
            positions * 1.0e-7,
        )
        return base.unsqueeze(0) + torch.arange(
            rows, device=device, dtype=torch.float32
        ).unsqueeze(1) * 0.01
    if score_mode == "quantized_ties":
        # Indexer operands are FP8 even though accumulated logits are FP32.
        # Repeated levels exercise threshold bins with many exact ties. Values,
        # rather than arbitrary equal-score indices, remain the hard oracle.
        positions = torch.arange(
            seq_len, device=device, dtype=torch.int64
        )
        base = (positions % 257).to(torch.float32) * 1.0e-4
        return base.unsqueeze(0).expand(rows, -1).contiguous()
    raise ValueError(f"unknown score mode: {score_mode}")


def _tile_rows(
    row_logits: torch.Tensor,
    *,
    block_q: int,
    block_k: int,
) -> tuple[torch.Tensor, int]:
    rows, seq_len = row_logits.shape
    q_tiles = math.ceil(rows / block_q)
    k_tiles = math.ceil(seq_len / block_k)
    padded_rows = q_tiles * block_q
    padded_cols = k_tiles * block_k
    padded = torch.full(
        (padded_rows, padded_cols),
        float("-inf"),
        device=row_logits.device,
        dtype=torch.float32,
    )
    padded[:rows, :seq_len] = row_logits
    tiled = (
        padded.view(q_tiles, block_q, k_tiles, block_k)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(-1)
    )
    return tiled, k_tiles


def _run_production_paged_topk(
    row_logits: torch.Tensor,
    *,
    topk: int,
    block_q: int,
    block_k: int,
    supertile_k: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Mirror the production logical-index paged top-k.

    DCP4 asks the B12X indexer for logical indices. At 32,768 tokens this
    switches from a direct selection to the paged path's two-level slice fold,
    even before a second 32k supertile is needed. That is the structural
    boundary between the passing 50k global case and the first failing 150k
    global case. Testing only the older ping-pong carry path would miss it.
    """
    from sparkinfer.attention.nsa_indexer.tiled_topk import (
        run_row_topk,
        run_tiled_topk,
    )

    rows, seq_len = map(int, row_logits.shape)
    if supertile_k % block_k:
        raise ValueError("supertile_k must be divisible by block_k")
    num_chunks = math.ceil(seq_len / supertile_k)
    slice_tokens = 16384
    max_slices = 32
    lengths = torch.full(
        (rows,),
        seq_len,
        device=row_logits.device,
        dtype=torch.int32,
    )
    final_values = torch.empty(
        (rows, topk),
        device=row_logits.device,
        dtype=torch.float32,
    )
    final_indices = torch.empty(
        (rows, topk),
        device=row_logits.device,
        dtype=torch.int32,
    )
    num_k_tiles = supertile_k // block_k
    use_two_level = seq_len >= 2 * slice_tokens

    two_level_slices: list[tuple[int, int, int]] = []
    total_slices = 0
    if use_two_level:
        effective_slice_tokens = max(
            slice_tokens,
            math.ceil(math.ceil(seq_len / max_slices) / block_k) * block_k,
        )
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * supertile_k
            chunk_width = min(supertile_k, seq_len - chunk_start)
            chunk_splits = max(
                1, math.ceil(chunk_width / effective_slice_tokens)
            )
            split_extent = (
                math.ceil(math.ceil(chunk_width / chunk_splits) / block_k)
                * block_k
            )
            two_level_slices.append(
                (chunk_splits, total_slices, split_extent)
            )
            total_slices += chunk_splits
        fold_values = torch.empty(
            (rows * total_slices, topk),
            device=row_logits.device,
            dtype=torch.float32,
        )
        fold_indices = torch.empty(
            (rows * total_slices, topk),
            device=row_logits.device,
            dtype=torch.int32,
        )
        fold_lengths = torch.full(
            (rows,),
            total_slices * topk,
            device=row_logits.device,
            dtype=torch.int32,
        )
        carry_values = None
        carry_indices = None
    else:
        fold_values = None
        fold_indices = None
        fold_lengths = None
        carry_values = torch.empty(
            (2, rows, topk),
            device=row_logits.device,
            dtype=torch.float32,
        )
        carry_indices = torch.empty(
            (2, rows, topk),
            device=row_logits.device,
            dtype=torch.int32,
        )

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * supertile_k
        chunk_end = min(chunk_start + supertile_k, seq_len)
        chunk_width = chunk_end - chunk_start
        # The real scratch keeps the full supertile allocation on the final
        # short chunk; unwritten positions are -inf.
        chunk_logits = torch.full(
            (rows, supertile_k),
            float("-inf"),
            device=row_logits.device,
            dtype=torch.float32,
        )
        chunk_logits[:, : chunk_end - chunk_start] = row_logits[
            :, chunk_start:chunk_end
        ]
        tiled, observed_k_tiles = _tile_rows(
            chunk_logits,
            block_q=block_q,
            block_k=block_k,
        )
        if observed_k_tiles != num_k_tiles:
            raise RuntimeError(
                f"unexpected supertile geometry {observed_k_tiles} != "
                f"{num_k_tiles}"
            )
        if use_two_level:
            assert fold_values is not None
            assert fold_indices is not None
            chunk_splits, chunk_slice_base, split_extent = (
                two_level_slices[chunk_idx]
            )
            run_tiled_topk(
                tile_logits=tiled,
                k_start=None,
                lengths=lengths,
                topk=topk,
                block_q=block_q,
                block_k=block_k,
                output_values=fold_values,
                output_indices=fold_indices,
                num_k_tiles=num_k_tiles,
                input_index_offset=chunk_start,
                input_extent=split_extent,
                output_index_offset=chunk_start,
                zero_row_start=True,
                is_first=True,
                extent_splits=chunk_splits,
                output_row_stride=total_slices,
                output_row_base=chunk_slice_base,
            )
            continue

        assert carry_values is not None
        assert carry_indices is not None
        is_first = chunk_idx == 0
        is_last = chunk_idx == num_chunks - 1
        carry_in_values = (
            None if is_first else carry_values[(chunk_idx - 1) % 2]
        )
        carry_in_indices = (
            None if is_first else carry_indices[(chunk_idx - 1) % 2]
        )
        output_values = (
            final_values if is_last else carry_values[chunk_idx % 2]
        )
        output_indices = (
            final_indices if is_last else carry_indices[chunk_idx % 2]
        )
        run_tiled_topk(
            tile_logits=tiled,
            k_start=None,
            lengths=lengths,
            topk=topk,
            block_q=block_q,
            block_k=block_k,
            output_values=output_values,
            output_indices=output_indices,
            num_k_tiles=num_k_tiles,
            input_index_offset=chunk_start,
            input_extent=chunk_width,
            output_index_offset=chunk_start,
            zero_row_start=True,
            carry_values=carry_in_values,
            carry_indices=carry_in_indices,
            is_first=is_first,
        )

    if use_two_level:
        assert fold_values is not None
        assert fold_indices is not None
        assert fold_lengths is not None
        run_row_topk(
            row_logits=fold_values.view(rows, total_slices * topk),
            lengths=fold_lengths,
            topk=topk,
            output_values=final_values,
            output_indices=final_indices,
            output_gather_table=fold_indices.view(
                rows, total_slices * topk
            ),
        )
        return final_values, final_indices, "paged_two_level_slice_fold"
    return final_values, final_indices, "paged_streaming_carry"


def run_topk_probe(
    *,
    device: torch.device,
    seq_lengths: list[int],
    score_modes: list[str],
    rows: int = 32,
    topk: int = 2048,
    block_q: int = 32,
    block_k: int = 512,
    supertile_k: int = 32768,
) -> list[TopKResult]:
    results: list[TopKResult] = []
    for seq_len in seq_lengths:
        for score_mode in score_modes:
            torch.manual_seed(991 + seq_len)
            row_logits = _make_scores(rows, seq_len, score_mode, device)
            values, indices, path = _run_production_paged_topk(
                row_logits,
                topk=topk,
                block_q=block_q,
                block_k=block_k,
                supertile_k=supertile_k,
            )
            torch.cuda.synchronize(device)

            ref_values, _ = torch.topk(row_logits, topk, dim=1)
            kth = ref_values[:, -1:]
            out_of_range = int(
                ((indices < 0) | (indices >= seq_len)).sum().item()
            )
            safe_indices = indices.long().clamp(0, seq_len - 1)
            selected_scores = torch.gather(row_logits, 1, safe_indices)
            below_threshold = int(
                (selected_scores < (kth - 1.0e-6)).sum().item()
            )
            duplicate_rows = 0
            for row in indices.cpu().tolist():
                if len(set(row)) != len(row):
                    duplicate_rows += 1
            # Sort because valid top-k order is not part of the contract.
            value_mismatches = int(
                (
                    torch.sort(values, dim=1, descending=True).values
                    != torch.sort(ref_values, dim=1, descending=True).values
                )
                .sum()
                .item()
            )
            passed = (
                out_of_range == 0
                and duplicate_rows == 0
                and below_threshold == 0
                and value_mismatches == 0
            )
            result = TopKResult(
                seq_len=seq_len,
                rows=rows,
                topk=topk,
                block_q=block_q,
                block_k=block_k,
                supertile_k=supertile_k,
                path=path,
                score_mode=score_mode,
                out_of_range=out_of_range,
                duplicate_rows=duplicate_rows,
                below_threshold=below_threshold,
                value_mismatches=value_mismatches,
                passed=passed,
            )
            results.append(result)
            _emit("topk", result)
    return results


def _csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _csv_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


def _csv_strings(value: str) -> list[str]:
    return [part for part in value.split(",") if part]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe",
        choices=("all", "fused-query", "topk"),
        default="all",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--m-values", default="1,4,9,16,32")
    parser.add_argument("--seeds", default="47,131,991")
    parser.add_argument("--scales", default="0.02,0.037,0.05,0.1")
    parser.add_argument("--timing-iterations", type=int, default=200)
    parser.add_argument("--timing-rounds", type=int, default=5)
    parser.add_argument("--retrieval-seq-len", type=int, default=62504)
    parser.add_argument("--retrieval-topk", type=int, default=2048)
    parser.add_argument(
        "--topk-rows",
        # Single-token decode, the first uneven MTP/DCP shape, the production
        # capture ceiling, and one full selector tile.
        default="1,9,16,32",
    )
    parser.add_argument(
        "--seq-lengths",
        # Production is DCP4: these are the exact supertile boundary plus
        # representative per-rank local widths for the failing 150k/250k
        # prompts, including unequal final-rank tails.
        default="32767,32768,32769,36842,36843,61401,62503,62504",
    )
    parser.add_argument(
        "--score-modes",
        default=(
            "monotonic,clustered,random,fold_boundary,quantized_ties"
        ),
    )
    args = parser.parse_args()

    if args.retrieval_seq_len <= 0:
        parser.error("--retrieval-seq-len must be positive")
    if not 0 < args.retrieval_topk <= args.retrieval_seq_len:
        parser.error(
            "--retrieval-topk must be positive and no larger than "
            "--retrieval-seq-len"
        )
    if args.timing_iterations <= 0 or args.timing_rounds <= 0:
        parser.error("--timing-iterations and --timing-rounds must be positive")

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.probe in ("all", "fused-query"):
        _load_vllm_custom_ops()

    fused_results: list[FusedQueryResult] = []
    fused_timing_results: list[FusedQueryTimingResult] = []
    topk_results: list[TopKResult] = []
    if args.probe in ("all", "fused-query"):
        fused_results, fused_timing_results = run_fused_query_probe(
            device=device,
            heads=args.heads,
            m_values=_csv_ints(args.m_values),
            seeds=_csv_ints(args.seeds),
            scales=_csv_floats(args.scales),
            timing_iterations=args.timing_iterations,
            timing_rounds=args.timing_rounds,
            retrieval_seq_len=args.retrieval_seq_len,
            retrieval_topk=args.retrieval_topk,
        )
    if args.probe in ("all", "topk"):
        for rows in _csv_ints(args.topk_rows):
            topk_results.extend(
                run_topk_probe(
                    device=device,
                    seq_lengths=_csv_ints(args.seq_lengths),
                    score_modes=_csv_strings(args.score_modes),
                    rows=rows,
                )
            )

    summary = {
        "fused_query_cases": len(fused_results),
        "fused_query_cases_with_fp8_mismatch": sum(
            result.fp8_mismatches > 0 for result in fused_results
        ),
        "fused_query_max_fp8_mismatch_fraction": max(
            (
                result.fp8_mismatch_fraction
                for result in fused_results
            ),
            default=0.0,
        ),
        "fused_query_min_retrieval_overlap_fraction": min(
            (
                result.retrieval_overlap_fraction
                for result in fused_results
            ),
            default=1.0,
        ),
        "fused_query_max_bf16_static_fp8_mismatch_fraction": max(
            (
                result.bf16_static_fp8_mismatch_fraction
                for result in fused_results
            ),
            default=0.0,
        ),
        "fused_query_min_bf16_static_retrieval_overlap_fraction": min(
            (
                result.bf16_static_retrieval_overlap_fraction
                for result in fused_results
            ),
            default=1.0,
        ),
        "fused_query_max_direct_vs_bf16_static_fp8_mismatch_fraction": max(
            (
                result.direct_vs_bf16_static_fp8_mismatch_fraction
                for result in fused_results
            ),
            default=0.0,
        ),
        "fused_query_timing_cases": len(fused_timing_results),
        "fused_query_max_fused_over_staged": max(
            (
                result.fused_over_staged
                for result in fused_timing_results
            ),
            default=0.0,
        ),
        "fused_query_max_fused_bf16_static_over_staged": max(
            (
                result.fused_bf16_static_over_staged
                for result in fused_timing_results
            ),
            default=0.0,
        ),
        "topk_cases": len(topk_results),
        "topk_failures": sum(not result.passed for result in topk_results),
    }
    _emit("summary", summary)
    return 1 if summary["topk_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
