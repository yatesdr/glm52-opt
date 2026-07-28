#!/usr/bin/env python3
"""Cross-image fingerprint for the exact v19/v20 sparse-MLA prefill seam.

The known-good v19 image and failing v20 image carry the same NVFP4+FP8-RoPE
record contract but different package names/revisions (``b12x`` versus
``sparkinfer``).  This probe constructs inputs without a PRNG, runs the
token-major extend path in either package, and emits comparable SHA-256
fingerprints for:

* the 368-byte KV records produced by the packaged writer;
* sparse-MLA attention output; and
* the downstream BF16 value projection.

Run the identical pinned script in both images.  Equal fingerprints exclude
the packaged record writer and sparse-attention kernel revision for these
production geometries.  A mismatch localizes the first differing stage.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import time
from dataclasses import asdict, dataclass

import torch


HEADS = 16
Q_DIM = 576
LATENT = 512
TOPK = 2048
PAGE = 64
RECORD_BYTES = 368
KV_TOKENS = 4096


@dataclass(frozen=True)
class Result:
    package: str
    package_version: str
    cap: int
    rows: int
    cache_sha256: str
    attention_sha256: str
    projection_sha256: str
    output_stride: tuple[int, ...]
    elapsed_seconds: float


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _exact_bf16(
    shape: tuple[int, ...],
    *,
    multiplier: int,
    offset: int,
    modulus: int,
    divisor: float,
    device: torch.device,
) -> torch.Tensor:
    numel = math.prod(shape)
    values = torch.arange(numel, dtype=torch.int64)
    values = ((values * multiplier + offset) % modulus) - modulus // 2
    return (values.to(torch.float32) / divisor).to(torch.bfloat16).reshape(
        shape
    ).to(device)


def _load_api():
    try:
        import sparkinfer
        from sparkinfer.attention._shared.mla.kv_cache import (
            concat_and_cache_nvfp4_mla_fp8_rope,
        )
        from sparkinfer.attention.sparse_mla import Caps, plan, run_extend

        return (
            "sparkinfer",
            str(getattr(sparkinfer, "__version__", "(unknown)")),
            Caps,
            plan,
            run_extend,
            concat_and_cache_nvfp4_mla_fp8_rope,
        )
    except ImportError:
        import b12x
        from b12x.attention.mla.kv_cache import (
            concat_and_cache_nvfp4_mla_fp8_rope,
        )
        from b12x.integration.mla import (
            B12XSparseMLAScratchCaps,
            plan_sparse_mla_scratch,
            sparse_mla_extend_forward,
        )

        return (
            "b12x",
            str(getattr(b12x, "__version__", "(unknown)")),
            B12XSparseMLAScratchCaps,
            plan_sparse_mla_scratch,
            sparse_mla_extend_forward,
            concat_and_cache_nvfp4_mla_fp8_rope,
        )


def _make_cache(device: torch.device, writer) -> torch.Tensor:
    kv_c = _exact_bf16(
        (KV_TOKENS, LATENT),
        multiplier=193,
        offset=17,
        modulus=1021,
        divisor=4096.0,
        device=device,
    )
    k_pe = _exact_bf16(
        (KV_TOKENS, Q_DIM - LATENT),
        multiplier=107,
        offset=29,
        modulus=509,
        divisor=2048.0,
        device=device,
    )
    cache = torch.empty(
        (KV_TOKENS // PAGE, PAGE, RECORD_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(KV_TOKENS, dtype=torch.int64, device=device)
    writer(kv_c, k_pe, cache, slots)
    torch.cuda.synchronize()
    return cache


def _run_case(
    *,
    package: str,
    package_version: str,
    Caps,
    plan,
    run_extend,
    cache: torch.Tensor,
    weight: torch.Tensor,
    cap: int,
    rows: int,
    case_index: int,
) -> Result:
    started = time.monotonic()
    device = cache.device
    q = _exact_bf16(
        (rows, HEADS, Q_DIM),
        multiplier=181 + 2 * case_index,
        offset=43 + case_index,
        modulus=2053,
        divisor=8192.0,
        device=device,
    )
    row_ids = torch.arange(rows, dtype=torch.int64, device=device)[:, None]
    col_ids = torch.arange(TOPK, dtype=torch.int64, device=device)[None, :]
    selected = (
        (row_ids * 104729 + col_ids * 15485863 + 20260726 + case_index)
        % KV_TOKENS
    ).to(torch.int32)
    active = torch.full((rows,), TOPK, dtype=torch.int32, device=device)
    cache_lens = torch.tensor([KV_TOKENS], dtype=torch.int32, device=device)

    caps_kwargs = dict(
        mode="extend",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=HEADS,
        head_dim=Q_DIM,
        v_head_dim=LATENT,
        max_width=TOPK,
        max_q_rows=cap,
        max_batch=1,
        page_size=PAGE,
    )
    if "head_major_output" in inspect.signature(Caps).parameters:
        caps_kwargs["head_major_output"] = False
    scratch_plan = plan(Caps(**caps_kwargs))
    (spec,) = scratch_plan.scratch_specs()
    storage = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = scratch_plan.bind(
        scratch=storage,
        q=q,
        selected_indices=selected,
        cache_seqlens_int32=cache_lens,
        nsa_cache_seqlens_int32=active,
    )
    attention = run_extend(
        binding=binding,
        kv_cache=cache,
        sm_scale=1.0 / math.sqrt(Q_DIM),
        latent_scale=1.0,
        v_head_dim=LATENT,
        scale_format=2,
        fp8_rope=True,
    )
    assert isinstance(attention, torch.Tensor)
    torch.cuda.synchronize()

    x = attention.view(rows, HEADS, LATENT).transpose(0, 1)
    projected = torch.empty(
        (rows, HEADS, LATENT), dtype=torch.bfloat16, device=device
    )
    torch.bmm(x, weight, out=projected.transpose(0, 1))
    torch.cuda.synchronize()

    return Result(
        package=package,
        package_version=package_version,
        cap=cap,
        rows=rows,
        cache_sha256=_digest(cache),
        attention_sha256=_digest(attention),
        projection_sha256=_digest(projected),
        output_stride=tuple(int(value) for value in attention.stride()),
        elapsed_seconds=time.monotonic() - started,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="CAP:ROWS; defaults to 2048:1711 and 3072:2735",
    )
    args = parser.parse_args()
    cases = args.case or ["2048:1711", "3072:2735"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    package, version, Caps, plan, run_extend, writer = _load_api()
    print(
        json.dumps(
            {
                "kind": "meta",
                "package": package,
                "package_version": version,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(device),
                "cases": cases,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    cache = _make_cache(device, writer)
    weight = _exact_bf16(
        (HEADS, LATENT, LATENT),
        multiplier=149,
        offset=71,
        modulus=4093,
        divisor=32768.0,
        device=device,
    )
    results: list[Result] = []
    for index, text in enumerate(cases):
        cap_text, rows_text = text.split(":", 1)
        cap, rows = int(cap_text), int(rows_text)
        if not 0 < rows <= cap:
            raise ValueError(f"invalid CAP:ROWS case {text!r}")
        result = _run_case(
            package=package,
            package_version=version,
            Caps=Caps,
            plan=plan,
            run_extend=run_extend,
            cache=cache,
            weight=weight,
            cap=cap,
            rows=rows,
            case_index=index,
        )
        results.append(result)
        print(
            json.dumps({"kind": "case", **asdict(result)}, sort_keys=True),
            flush=True,
        )
    print(
        json.dumps(
            {
                "kind": "summary",
                "package": package,
                "cases": len(results),
                "cache_sha256": _digest(cache),
                "verdict": "COMPLETE",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
