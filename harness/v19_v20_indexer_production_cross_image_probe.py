#!/usr/bin/env python3
"""Cross-image sparse-indexer fingerprint at the frozen 350k geometry.

The known-good v19 and failing v20 images expose the same fused-indexer API
under different package names.  This probe feeds both implementations
bit-identical, formula-generated inputs at the production failure shape:

* 1,711 query rows (343,727 prompt tokens modulo MNBT=2,048);
* 85,932 live rank-local tokens (ceil(343,727 / DCP4));
* 120,000-token rank-local capacity; and
* 32 indexer heads selecting top-k 2,048.

It fingerprints canonical ``(logical_index, score)`` pairs for forced-serial,
forced-cooperative, and automatic merge routing.  Index/score fingerprints
that differ across images localize the first numerical or selection delta to
the packaged indexer.  Equal fingerprints move the search upstream to the
real model-produced query, key, or per-head weight tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from typing import Any

import torch


PAGE_SIZE = 64
INDEX_DIM = 128
DEFAULT_ROWS = 1711
DEFAULT_LIVE_TOKENS = 85932
DEFAULT_CAPACITY_TOKENS = 120000
DEFAULT_HEADS = 32
DEFAULT_TOPK = 2048


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _load_api():
    try:
        import sparkinfer
        from sparkinfer.attention.nsa_indexer.fused_indexer import (
            _COOP_STATE_WORDS,
            _resolve_default_ctas_per_group,
            _resolve_default_merge_threshold,
            run_fused_paged_indexer,
        )

        return (
            "sparkinfer",
            str(getattr(sparkinfer, "__version__", "(unknown)")),
            _COOP_STATE_WORDS,
            _resolve_default_ctas_per_group,
            _resolve_default_merge_threshold,
            run_fused_paged_indexer,
        )
    except ImportError:
        import b12x
        from b12x.attention.indexer.fused_indexer import (
            _COOP_STATE_WORDS,
            _resolve_default_ctas_per_group,
            _resolve_default_merge_threshold,
            run_fused_paged_indexer,
        )

        return (
            "b12x",
            str(getattr(b12x, "__version__", "(unknown)")),
            _COOP_STATE_WORDS,
            _resolve_default_ctas_per_group,
            _resolve_default_merge_threshold,
            run_fused_paged_indexer,
        )


def _canonical(
    indices: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    canonical_indices, order = torch.sort(indices, dim=-1)
    canonical_values = torch.gather(values, -1, order.to(torch.int64))
    return canonical_indices, canonical_values


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--live-tokens", type=int, default=DEFAULT_LIVE_TOKENS)
    parser.add_argument(
        "--capacity-tokens",
        type=int,
        default=DEFAULT_CAPACITY_TOKENS,
    )
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    args = parser.parse_args()

    if args.live_tokens > args.capacity_tokens:
        raise ValueError("live tokens exceed capacity")
    if args.topk > args.live_tokens:
        raise ValueError("top-k exceeds live tokens")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    (
        package,
        version,
        coop_state_words,
        resolve_ctas,
        resolve_threshold,
        run_indexer,
    ) = _load_api()

    active_pages = math.ceil(args.live_tokens / PAGE_SIZE)
    capacity_pages = math.ceil(args.capacity_tokens / PAGE_SIZE)
    # Generate on CPU so the stream is independent of CUDA implementation
    # details, then fingerprint every resulting input.  A prior arithmetic
    # pattern was intentionally discarded because its short periods created
    # exact top-k ties and made valid tie-breaking choices look like drift.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260726)
    q_fp8 = (
        torch.randn(
            args.rows,
            args.heads,
            INDEX_DIM,
            generator=generator,
        )
        / 3.0
    ).to(torch.float8_e4m3fn).to(device)
    weights = torch.randn(
        args.rows,
        args.heads,
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
        / 3.0
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

    row_ids = torch.arange(args.rows, dtype=torch.int64, device=device)[:, None]
    page_ids = torch.arange(active_pages, dtype=torch.int64, device=device)[None, :]
    live_page_table = (page_ids + row_ids * 13).remainder(active_pages)
    page_table = torch.full(
        (args.rows, capacity_pages),
        -1,
        dtype=torch.int32,
        device=device,
    )
    page_table[:, :active_pages].copy_(live_page_table.to(torch.int32))
    seqlens = torch.full(
        (args.rows,),
        args.live_tokens,
        dtype=torch.int32,
        device=device,
    )

    ctas_per_group = resolve_ctas(
        num_rows=args.rows,
        max_pages=capacity_pages,
        device=device,
    )
    auto_threshold = resolve_threshold(
        ctas_per_group=ctas_per_group,
        num_heads=args.heads,
        topk=args.topk,
    )
    pack_elems = args.rows * ctas_per_group * args.topk
    pack_values = torch.empty(pack_elems, dtype=torch.float32, device=device)
    pack_indices = torch.empty(pack_elems, dtype=torch.int32, device=device)
    merge_state = torch.zeros(
        args.rows * coop_state_words,
        dtype=torch.int32,
        device=device,
    )
    out_indices = torch.empty(
        (args.rows, args.topk),
        dtype=torch.int32,
        device=device,
    )
    out_values = torch.empty(
        (args.rows, args.topk),
        dtype=torch.float32,
        device=device,
    )

    print(
        json.dumps(
            {
                "kind": "meta",
                "package": package,
                "package_version": version,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(device),
                "rows": args.rows,
                "heads": args.heads,
                "topk": args.topk,
                "live_tokens": args.live_tokens,
                "active_pages": active_pages,
                "capacity_tokens": capacity_pages * PAGE_SIZE,
                "capacity_pages": capacity_pages,
                "ctas_per_group": ctas_per_group,
                "auto_threshold": auto_threshold,
                "input_sha256": {
                    "q": _digest(q_fp8),
                    "weights": _digest(weights),
                    "k": _digest(k_fp8),
                    "k_scales": _digest(k_scales),
                    "page_table": _digest(page_table),
                    "seqlens": _digest(seqlens),
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )

    q_bytes = q_fp8.view(torch.uint8)
    k_bytes = k_fp8.view(torch.uint8).contiguous()
    serial_reference: tuple[torch.Tensor, torch.Tensor] | None = None
    failed = False
    for route in ("serial", "cooperative", "auto"):
        threshold = (
            capacity_pages * PAGE_SIZE + PAGE_SIZE
            if route == "serial"
            else 0
            if route == "cooperative"
            else auto_threshold
        )
        merge_state.zero_()
        torch.cuda.synchronize(device)
        started = time.monotonic()
        run_indexer(
            q_bytes=q_bytes,
            weights=weights,
            k_quant_bytes=k_bytes,
            k_scales=k_scales,
            real_page_table=page_table,
            seqlens=seqlens,
            num_heads=args.heads,
            topk=args.topk,
            out_indices=out_indices,
            out_values=out_values,
            ctas_per_group=ctas_per_group,
            merge_threshold=threshold,
            pack_values=pack_values,
            pack_indices=pack_indices,
            merge_state=merge_state,
            merge_state_preinitialized=True,
            output_physical_slots=False,
        )
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        canonical_indices, canonical_values = _canonical(out_indices, out_values)
        if serial_reference is None:
            serial_reference = (
                canonical_indices.clone(),
                canonical_values.clone(),
            )
            index_mismatches = 0
            score_max_abs_delta = 0.0
        else:
            serial_indices, serial_values = serial_reference
            index_mismatches = int(
                torch.count_nonzero(canonical_indices.ne(serial_indices)).item()
            )
            score_max_abs_delta = float(
                (canonical_values - serial_values).abs().max().item()
            )
        route_passed = index_mismatches == 0 and score_max_abs_delta <= 1e-4
        failed |= not route_passed
        print(
            json.dumps(
                {
                    "kind": "route",
                    "route": route,
                    "threshold": threshold,
                    "canonical_indices_sha256": _digest(canonical_indices),
                    "canonical_values_sha256": _digest(canonical_values),
                    "index_mismatches_vs_serial": index_mismatches,
                    "score_max_abs_delta_vs_serial": score_max_abs_delta,
                    "elapsed_seconds": elapsed,
                    "status": "PASS" if route_passed else "FAIL",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "kind": "summary",
                "package": package,
                "routes": 3,
                "status": "FAIL" if failed else "PASS",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
