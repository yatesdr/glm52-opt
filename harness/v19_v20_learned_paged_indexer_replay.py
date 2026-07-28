#!/usr/bin/env python3
"""Replay a captured learned sparse-indexer row through v19 or v20.

The long-context trace stores the final query row from a production-shaped
prefill chunk.  This probe repeats that row to retain the production
``q_rows`` dispatch geometry, forces the production shared-page-table plan,
and checks the final output against a dense PyTorch reference computed from
the same captured FP8 query and packed FP8+scale K cache.

Run this script inside either a v19 (``b12x.attention.indexer``) or v20
(``sparkinfer.attention.nsa_indexer``) image.  Feeding the same trace to both
images is the causal discriminator: input bytes and geometry are held fixed,
so a result change belongs to the indexer implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


TRACE_SCHEMA = "v19-v20-longctx-indexer-boundary-v1"
RESULT_SCHEMA = "v19-v20-learned-paged-indexer-replay-v1"
PAGE_SIZE = 64
INDEX_HEAD_DIM = 128
SCALE_BYTES = 4


@dataclass(frozen=True)
class IndexerAPI:
    implementation: str
    caps: type
    source_layout_paged: str
    plan: Callable[..., Any]
    index_topk_fp8: Callable[..., torch.Tensor]
    paged_logits_reference: Callable[..., torch.Tensor]
    clear_caches: Callable[[], None]


def _load_indexer_api() -> IndexerAPI:
    try:
        from sparkinfer.attention.nsa_indexer import (
            Caps,
            SOURCE_LAYOUT_PAGED,
            clear_caches,
            index_topk_fp8,
            plan,
        )
        from sparkinfer.attention.nsa_indexer.paged import (
            paged_index_logits_reference,
        )

        return IndexerAPI(
            implementation="sparkinfer.attention.nsa_indexer",
            caps=Caps,
            source_layout_paged=SOURCE_LAYOUT_PAGED,
            plan=plan,
            index_topk_fp8=index_topk_fp8,
            paged_logits_reference=paged_index_logits_reference,
            clear_caches=clear_caches,
        )
    except ImportError as v20_error:
        try:
            from b12x.attention.indexer import (
                B12XIndexerScratchCaps,
                INDEXER_SOURCE_LAYOUT_PAGED,
                clear_indexer_caches,
                index_topk_fp8,
                paged_index_logits_reference,
                plan_indexer_scratch,
            )

            return IndexerAPI(
                implementation="b12x.attention.indexer",
                caps=B12XIndexerScratchCaps,
                source_layout_paged=INDEXER_SOURCE_LAYOUT_PAGED,
                plan=plan_indexer_scratch,
                index_topk_fp8=index_topk_fp8,
                paged_logits_reference=paged_index_logits_reference,
                clear_caches=clear_indexer_caches,
            )
        except ImportError as v19_error:
            raise RuntimeError(
                "neither the v20 sparkinfer nor v19 b12x indexer API imported; "
                f"v20_error={v20_error!r}; v19_error={v19_error!r}"
            ) from v19_error


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _set_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a_set = {int(value) for value in a.tolist() if int(value) >= 0}
    b_set = {int(value) for value in b.tolist() if int(value) >= 0}
    intersection = a_set & b_set
    union = a_set | b_set
    return {
        "a_count": len(a_set),
        "b_count": len(b_set),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": 1.0 if not union else len(intersection) / len(union),
        "set_exact": a_set == b_set,
    }


def _load_trace(path: Path) -> dict[str, Any]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("schema") != TRACE_SCHEMA:
        raise RuntimeError(
            f"{path}: expected trace schema {TRACE_SCHEMA!r}, "
            f"got {record.get('schema')!r}"
        )
    if record.get("stage") != "local":
        raise RuntimeError(f"{path}: expected local trace, got {record.get('stage')!r}")
    required = (
        "batch_tokens",
        "active_page_width",
        "q_fp8",
        "weights",
        "seq_len",
        "page_ids",
        "cache_pages",
        "topk_indices",
        "topk_scores",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise RuntimeError(f"{path}: missing fields {missing}")
    return record


def _validate_trace(record: dict[str, Any], *, topk: int) -> None:
    q = record["q_fp8"]
    weights = record["weights"]
    cache = record["cache_pages"]
    page_ids = record["page_ids"]
    seq_len = record["seq_len"]
    if q.ndim != 2 or tuple(q.shape) != (32, INDEX_HEAD_DIM):
        raise RuntimeError(f"unexpected q shape {tuple(q.shape)}")
    if q.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"unexpected q dtype {q.dtype}")
    if tuple(weights.shape) != (32,) or weights.dtype != torch.float32:
        raise RuntimeError(
            f"unexpected weights contract shape={tuple(weights.shape)} "
            f"dtype={weights.dtype}"
        )
    if cache.ndim != 3 or tuple(cache.shape[1:]) != (
        PAGE_SIZE,
        INDEX_HEAD_DIM + SCALE_BYTES,
    ):
        raise RuntimeError(f"unexpected cache shape {tuple(cache.shape)}")
    if cache.dtype != torch.uint8:
        raise RuntimeError(f"unexpected cache dtype {cache.dtype}")
    if page_ids.numel() != cache.shape[0]:
        raise RuntimeError(
            f"page/cache mismatch page_ids={page_ids.numel()} pages={cache.shape[0]}"
        )
    if int(record["active_page_width"]) != int(cache.shape[0]):
        raise RuntimeError(
            "active_page_width does not match captured logical cache pages: "
            f"{record['active_page_width']} vs {cache.shape[0]}"
        )
    if seq_len.numel() != 1 or int(seq_len.item()) <= 0:
        raise RuntimeError(f"unexpected seq_len {seq_len}")
    if int(seq_len.item()) > int(cache.shape[0]) * PAGE_SIZE:
        raise RuntimeError(
            f"seq_len {int(seq_len.item())} exceeds cache capacity "
            f"{int(cache.shape[0]) * PAGE_SIZE}"
        )
    if tuple(record["topk_indices"].shape) != (topk,):
        raise RuntimeError(
            f"captured top-k width {record['topk_indices'].shape} != {topk}"
        )


def _run_one(
    *,
    api: IndexerAPI,
    trace_path: Path,
    output_dir: Path,
    rows: int | None,
    topk: int,
    device: torch.device,
) -> dict[str, Any]:
    record = _load_trace(trace_path)
    _validate_trace(record, topk=topk)
    production_rows = int(record["batch_tokens"])
    replay_rows = production_rows if rows is None else int(rows)
    if replay_rows <= 0:
        raise ValueError(f"rows must be positive, got {replay_rows}")

    q_row = record["q_fp8"].to(device=device)
    weights_row = record["weights"].to(device=device)
    cache_pages = record["cache_pages"].contiguous().to(device=device)
    cache_flat = cache_pages.view(cache_pages.shape[0], -1).contiguous()
    page_width = int(cache_pages.shape[0])
    seq_len_value = int(record["seq_len"].item())

    # Preserve the production q_rows dispatch shape. Every indexer output row is
    # independent, so repeating the captured learned row keeps the final row's
    # mathematical input fixed while selecting the same block-Q kernels and
    # scratch geometry as the production invocation.
    q = q_row.unsqueeze(0).repeat(replay_rows, 1, 1).contiguous()
    weights = weights_row.unsqueeze(0).repeat(replay_rows, 1).contiguous()
    seq_lens = torch.full(
        (replay_rows,), seq_len_value, dtype=torch.int32, device=device
    )
    identity_table = torch.arange(
        page_width, dtype=torch.int32, device=device
    ).view(1, page_width)
    block_table = identity_table.expand(replay_rows, page_width)

    api.clear_caches()
    plan = api.plan(
        api.caps(
            device=device,
            source_layout=api.source_layout_paged,
            num_q_heads=int(q.shape[1]),
            max_q_rows=replay_rows,
            max_page_table_width=page_width,
            topk=topk,
            mode="prefill",
            shared_page_table=True,
        )
    )
    route = getattr(plan, "route", None)
    if route is None:
        route = getattr(getattr(plan, "layout", None), "route", None)
    if route != "packed_contiguous":
        raise RuntimeError(
            f"shared prefill plan must select packed_contiguous, got {route!r}"
        )
    scratch = [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    ]
    binding = plan.bind(
        scratch=scratch,
        real_page_table=block_table,
        cache_seqlens_int32=seq_lens,
        active_width=None,
        schedule_metadata=None,
        expected_num_q_heads=int(q.shape[1]),
        shared_page_table=True,
        output_physical_slots=False,
    )
    binding_route = getattr(binding, "route", None)
    if binding_route != "packed_contiguous":
        raise RuntimeError(
            f"shared prefill binding must select packed_contiguous, "
            f"got {binding_route!r}"
        )

    out_indices = torch.empty(
        (replay_rows, topk), dtype=torch.int32, device=device
    )
    out_scores = torch.empty(
        (replay_rows, topk), dtype=torch.float32, device=device
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    api.index_topk_fp8(
        q_fp8=q,
        weights=weights,
        index_k_cache=cache_flat,
        binding=binding,
        page_size=PAGE_SIZE,
        expected_num_q_heads=int(q.shape[1]),
        out_indices=out_indices,
        out_scores=out_scores,
    )
    torch.cuda.synchronize(device)
    kernel_seconds = time.perf_counter() - started

    replay_indices = out_indices[-1].detach()
    replay_scores = out_scores[-1].detach()

    # The dense reference intentionally runs one row. It shares only the public
    # byte-layout/math contract with the optimized selector, not its scorer,
    # tiling, radix selection, carry buffers, or dispatch.
    reference_q = q_row.unsqueeze(0).contiguous()
    reference_weights = weights_row.unsqueeze(0).contiguous()
    reference_table = identity_table.contiguous()
    reference_seq_len = torch.tensor(
        [seq_len_value], dtype=torch.int32, device=device
    )
    reference_logits = api.paged_logits_reference(
        q_fp8=reference_q,
        weights=reference_weights,
        index_k_cache=cache_flat,
        real_page_table=reference_table,
        query_row_to_batch=torch.zeros(1, dtype=torch.int32, device=device),
        seqlens_per_query=reference_seq_len,
    )[0]
    reference_scores, reference_indices = torch.topk(
        reference_logits, k=topk, largest=True, sorted=False
    )
    reference_indices = reference_indices.to(torch.int32)

    replay_indices_cpu = replay_indices.cpu()
    replay_scores_cpu = replay_scores.cpu()
    reference_indices_cpu = reference_indices.cpu()
    reference_scores_cpu = reference_scores.cpu()
    captured_indices_cpu = record["topk_indices"].cpu()
    captured_scores_cpu = record["topk_scores"].cpu()

    safe_replay_indices = replay_indices.long().clamp(
        min=0, max=reference_logits.numel() - 1
    )
    expected_replay_scores = reference_logits.index_select(
        0, safe_replay_indices
    )
    valid_replay = (replay_indices >= 0) & (
        replay_indices < reference_logits.numel()
    )
    replay_score_delta = torch.where(
        valid_replay,
        replay_scores - expected_replay_scores,
        torch.full_like(replay_scores, float("inf")),
    )
    oracle_kth = float(reference_scores.min().item())
    replay_min = float(replay_scores.min().item())
    strictly_better_omitted = int(
        torch.count_nonzero(reference_logits > replay_min).item()
        - torch.count_nonzero(replay_scores > replay_min).item()
    )

    tensor_output = {
        "schema": RESULT_SCHEMA,
        "implementation": api.implementation,
        "source_trace": str(trace_path),
        "rank": int(record["tp_rank"]),
        "replay_rows": replay_rows,
        "route": binding_route,
        "replay_indices": replay_indices_cpu,
        "replay_scores": replay_scores_cpu,
        "reference_indices": reference_indices_cpu,
        "reference_scores": reference_scores_cpu,
        "captured_indices": captured_indices_cpu,
        "captured_scores": captured_scores_cpu,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / f"tp{int(record['tp_rank'])}-replay.pt"
    torch.save(tensor_output, tensor_path)

    return {
        "rank": int(record["tp_rank"]),
        "source_trace": str(trace_path),
        "input": {
            "replay_rows": replay_rows,
            "production_rows": production_rows,
            "page_width": page_width,
            "seq_len": seq_len_value,
            "topk": topk,
            "q_sha256": _tensor_sha256(record["q_fp8"]),
            "weights_sha256": _tensor_sha256(record["weights"]),
            "cache_sha256": _tensor_sha256(record["cache_pages"]),
        },
        "execution": {
            "implementation": api.implementation,
            "route": binding_route,
            "kernel_seconds_including_first_compile": kernel_seconds,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(device),
        },
        "replay": {
            "indices_sha256": _tensor_sha256(replay_indices_cpu),
            "scores_sha256": _tensor_sha256(replay_scores_cpu),
            "valid_indices": int(valid_replay.sum().item()),
            "score_vs_reference_max_abs": float(
                replay_score_delta.abs().max().item()
            ),
            "score_vs_reference_mean_abs": float(
                replay_score_delta.abs().mean().item()
            ),
            "min_score": replay_min,
        },
        "reference": {
            "indices_sha256": _tensor_sha256(reference_indices_cpu),
            "scores_sha256": _tensor_sha256(reference_scores_cpu),
            "kth_score": oracle_kth,
            "strictly_better_omitted": max(strictly_better_omitted, 0),
        },
        "replay_vs_reference": _set_metrics(
            replay_indices_cpu, reference_indices_cpu
        ),
        "replay_vs_captured": _set_metrics(
            replay_indices_cpu, captured_indices_cpu
        ),
        "captured_vs_reference": _set_metrics(
            captured_indices_cpu, reference_indices_cpu
        ),
        "tensor_output": str(tensor_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-root",
        type=Path,
        required=True,
        help="directory containing tpN/layer00-indexer-local.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="TP ranks to replay",
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="override replay q_rows; default preserves captured batch_tokens",
    )
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the optimized indexer replay")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    api = _load_indexer_api()

    results = []
    for rank in args.ranks:
        trace_path = args.trace_root / f"tp{rank}" / "layer00-indexer-local.pt"
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        results.append(
            _run_one(
                api=api,
                trace_path=trace_path,
                output_dir=args.output_dir,
                rows=args.rows,
                topk=args.topk,
                device=device,
            )
        )

    report = {
        "schema": RESULT_SCHEMA,
        "implementation": api.implementation,
        "trace_root": str(args.trace_root),
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    summary_path = args.summary or (args.output_dir / "summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
