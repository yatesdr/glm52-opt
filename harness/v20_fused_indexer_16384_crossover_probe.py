#!/usr/bin/env python3
"""Probe the GLM fused-indexer merge crossover at 16,384 DCP-local tokens.

For the production GLM geometry (32 heads, top-k 2048, four query rows), the
fused paged indexer switches from its serial last-CTA merge to its cooperative
grid-barrier merge when the live local sequence length exceeds 16,384. Global
DCP4 contexts around 60k/70k straddle that exact boundary.

This no-model probe runs the same scorer inputs through forced-serial,
forced-cooperative, and auto-dispatch paths, including CUDA-graph replay. It
uses a production-capacity page table but materializes only the pages reachable
by the tested live lengths, keeping the GPU footprint small.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from sparkinfer.attention.nsa_indexer.fused_indexer import (
    _COOP_STATE_WORDS,
    _resolve_default_ctas_per_group,
    _resolve_default_merge_threshold,
    run_fused_paged_indexer,
)


PAGE_SIZE = 64
INDEX_DIM = 128


def _emit(record: dict[str, Any], output: Path | None) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        with output.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _same_topk(
    actual_indices: torch.Tensor,
    actual_values: torch.Tensor,
    reference_indices: torch.Tensor,
    reference_values: torch.Tensor,
    *,
    max_value_delta_allowed: float = 1e-4,
) -> tuple[bool, float]:
    actual_sorted = torch.sort(actual_indices, dim=-1).values
    reference_sorted = torch.sort(reference_indices, dim=-1).values
    same_indices = bool(torch.equal(actual_sorted, reference_sorted))
    actual_value_sorted = torch.sort(actual_values.float(), dim=-1).values
    reference_value_sorted = torch.sort(reference_values.float(), dim=-1).values
    max_value_delta = float(
        (actual_value_sorted - reference_value_sorted).abs().max().item()
    )
    return (
        same_indices and max_value_delta <= max_value_delta_allowed,
        max_value_delta,
    )


def _reference_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scales: torch.Tensor,
    page_table: torch.Tensor,
    live_length: int,
    topk: int,
    output_physical_slots: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PyTorch oracle for the fused scorer and index mapping."""
    rows = q_fp8.shape[0]
    pages = math.ceil(live_length / PAGE_SIZE)
    q_float = q_fp8.float()
    k_float = k_fp8.float()
    reference_indices = []
    reference_values = []
    for row in range(rows):
        row_pages = page_table[row, :pages].to(torch.int64)
        row_k = k_float[row_pages].reshape(-1, INDEX_DIM)[:live_length]
        row_scales = k_scales[row_pages].reshape(-1)[:live_length]
        logits = (
            torch.relu(
                torch.einsum("hd,td->ht", q_float[row], row_k)
            )
            * weights[row].unsqueeze(1)
        ).sum(dim=0) * row_scales
        values, logical = torch.topk(
            logits, topk, largest=True, sorted=False
        )
        if output_physical_slots:
            page_columns = torch.div(
                logical, PAGE_SIZE, rounding_mode="floor"
            )
            page_offsets = logical.remainder(PAGE_SIZE)
            physical_pages = page_table[row, page_columns]
            indices = physical_pages * PAGE_SIZE + page_offsets
        else:
            indices = logical.to(torch.int32)
        reference_indices.append(indices.to(torch.int32))
        reference_values.append(values)
    return torch.stack(reference_indices), torch.stack(reference_values)


def _make_scratch(
    *,
    rows: int,
    topk: int,
    ctas_per_group: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pack_elems = rows * ctas_per_group * topk
    return (
        torch.empty(pack_elems, dtype=torch.float32, device=device),
        torch.empty(pack_elems, dtype=torch.int32, device=device),
        torch.zeros(rows * _COOP_STATE_WORDS, dtype=torch.int32, device=device),
    )


@torch.inference_mode()
def run_probe(
    *,
    device: torch.device,
    rows: int,
    heads: int,
    topk: int,
    capacity_tokens: int,
    lengths: list[int],
    seed: int,
    output_physical_slots: bool,
) -> list[dict[str, Any]]:
    max_live_tokens = max(lengths)
    active_pages = math.ceil(max_live_tokens / PAGE_SIZE)
    capacity_pages = math.ceil(capacity_tokens / PAGE_SIZE)
    if capacity_pages < active_pages:
        raise ValueError("capacity must cover every live length")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    q_fp8 = (
        torch.randn(rows, heads, INDEX_DIM, generator=generator) / 3
    ).to(torch.float8_e4m3fn).to(device)
    weights = torch.randn(
        rows,
        heads,
        dtype=torch.float32,
        generator=generator,
    ).to(device)
    k_fp8 = (
        torch.randn(
            active_pages,
            PAGE_SIZE,
            INDEX_DIM,
            generator=generator,
        )
        / 3
    ).to(torch.float8_e4m3fn).to(device)
    k_scales = (
        torch.rand(
            active_pages,
            PAGE_SIZE,
            dtype=torch.float32,
            generator=generator,
        )
        + 0.1
    ).to(device)

    base_pages = torch.randperm(active_pages, generator=generator, dtype=torch.int32)
    page_table_cpu = torch.full(
        (rows, capacity_pages),
        -1,
        dtype=torch.int32,
    )
    for row in range(rows):
        page_table_cpu[row, :active_pages] = base_pages.roll(row * 13)
    page_table = page_table_cpu.to(device)
    seqlens = torch.full(
        (rows,),
        lengths[0],
        dtype=torch.int32,
        device=device,
    )

    ctas_per_group = _resolve_default_ctas_per_group(
        num_rows=rows,
        max_pages=capacity_pages,
        device=device,
    )
    auto_threshold = _resolve_default_merge_threshold(
        ctas_per_group=ctas_per_group,
        num_heads=heads,
        topk=topk,
    )
    if auto_threshold != 16384:
        raise RuntimeError(
            f"expected production crossover 16384, got {auto_threshold}"
        )

    q_bytes = q_fp8.view(torch.uint8)
    k_bytes = k_fp8.view(torch.uint8).contiguous()
    route_buffers = {}
    for route in ("serial", "cooperative", "auto"):
        out_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
        out_values = torch.empty((rows, topk), dtype=torch.float32, device=device)
        scratch = _make_scratch(
            rows=rows,
            topk=topk,
            ctas_per_group=ctas_per_group,
            device=device,
        )
        route_buffers[route] = (out_indices, out_values, *scratch)

    def launch(route: str) -> tuple[torch.Tensor, torch.Tensor]:
        out_indices, out_values, pack_values, pack_indices, merge_state = (
            route_buffers[route]
        )
        threshold = (
            capacity_tokens + PAGE_SIZE
            if route == "serial"
            else 0
            if route == "cooperative"
            else auto_threshold
        )
        return run_fused_paged_indexer(
            q_bytes=q_bytes,
            weights=weights,
            k_quant_bytes=k_bytes,
            k_scales=k_scales,
            real_page_table=page_table,
            seqlens=seqlens,
            num_heads=heads,
            topk=topk,
            out_indices=out_indices,
            out_values=out_values,
            ctas_per_group=ctas_per_group,
            merge_threshold=threshold,
            pack_values=pack_values,
            pack_indices=pack_indices,
            merge_state=merge_state,
            merge_state_preinitialized=True,
            output_physical_slots=output_physical_slots,
        )

    # Compile/warm every route before graph capture.
    for route in ("serial", "cooperative", "auto"):
        launch(route)
        torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch("auto")
    torch.cuda.synchronize(device)

    results = []
    reference_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    replay_order = [*lengths, *reversed(lengths), lengths[-1]]
    for iteration, live_length in enumerate(replay_order):
        seqlens.fill_(live_length)

        launch("serial")
        torch.cuda.synchronize(device)
        serial_indices = route_buffers["serial"][0].clone()
        serial_values = route_buffers["serial"][1].clone()
        reference = reference_cache.get(live_length)
        if reference is None:
            reference = _reference_topk(
                q_fp8=q_fp8,
                weights=weights,
                k_fp8=k_fp8,
                k_scales=k_scales,
                page_table=page_table,
                live_length=live_length,
                topk=topk,
                output_physical_slots=output_physical_slots,
            )
            reference_cache[live_length] = reference
        reference_indices, reference_values = reference
        serial_matches, serial_delta = _same_topk(
            serial_indices,
            serial_values,
            reference_indices,
            reference_values,
            max_value_delta_allowed=1e-2,
        )

        launch("cooperative")
        torch.cuda.synchronize(device)
        coop_matches, coop_delta = _same_topk(
            route_buffers["cooperative"][0],
            route_buffers["cooperative"][1],
            serial_indices,
            serial_values,
        )

        graph.replay()
        torch.cuda.synchronize(device)
        auto_matches, auto_delta = _same_topk(
            route_buffers["auto"][0],
            route_buffers["auto"][1],
            serial_indices,
            serial_values,
        )
        expected_auto_route = (
            "cooperative" if live_length > auto_threshold else "serial"
        )
        result = {
            "kind": "fused_indexer_16384_crossover",
            "iteration": iteration,
            "live_length": live_length,
            "global_dcp4_equivalent": live_length * 4,
            "rows": rows,
            "heads": heads,
            "topk": topk,
            "capacity_tokens": capacity_pages * PAGE_SIZE,
            "ctas_per_group": ctas_per_group,
            "auto_threshold": auto_threshold,
            "expected_auto_route": expected_auto_route,
            "output_physical_slots": output_physical_slots,
            "serial_matches_reference": serial_matches,
            "serial_reference_max_value_delta": serial_delta,
            "cooperative_matches_serial": coop_matches,
            "cooperative_max_value_delta": coop_delta,
            "auto_graph_matches_serial": auto_matches,
            "auto_graph_max_value_delta": auto_delta,
            "status": (
                "PASS"
                if serial_matches and coop_matches and auto_matches
                else "FAIL"
            ),
        }
        results.append(result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity-tokens", type=int, default=120000)
    parser.add_argument("--lengths", default="16383,16384,16385")
    parser.add_argument("--seeds", default="7,19,41")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _csv_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.output is not None:
        args.output.unlink(missing_ok=True)

    records = []
    for output_physical_slots in (False, True):
        for seed in _csv_ints(args.seeds):
            batch = run_probe(
                device=device,
                rows=args.rows,
                heads=args.heads,
                topk=args.topk,
                capacity_tokens=args.capacity_tokens,
                lengths=_csv_ints(args.lengths),
                seed=seed,
                output_physical_slots=output_physical_slots,
            )
            records.extend(batch)
            for record in batch:
                _emit(record, args.output)

    failed = sum(record["status"] != "PASS" for record in records)
    summary = {
        "kind": "summary",
        "cases": len(records),
        "failed": failed,
        "status": "PASS" if failed == 0 else "FAIL",
    }
    _emit(summary, args.output)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
