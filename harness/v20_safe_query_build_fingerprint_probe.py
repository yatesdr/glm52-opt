#!/usr/bin/env python3
"""Emit the 54-case post-FP8 safe-query fingerprint used by image builds.

This is a post-image, pre-push GPU gate. Docker build stages do not have a
CUDA driver, so running it inside a Dockerfile would be a false contract.
The build pipeline must run this probe against the completed local image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


TOKENS = (1, 4, 9, 16, 32, 3072)
SEEDS = (7, 19, 41)
Q_SCALES = (0.5, 1.0, 2.0)
HEADS = 8
Q_DIM = 192
ROPE_DIM = 64
LATENT_DIM = 512


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _emit(record: dict[str, Any], output: Path) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _load_ops(expected_stable_sha256: str) -> str:
    import vllm
    import vllm._custom_ops  # noqa: F401

    package_dir = Path(vllm.__file__).resolve().parent
    candidates = sorted(package_dir.glob("_C_stable_libtorch*.so"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one stable-libtorch extension, found {candidates}"
        )
    observed = _file_sha256(candidates[0])
    if observed != expected_stable_sha256:
        raise RuntimeError(
            "stable-libtorch byte pin mismatch: "
            f"expected={expected_stable_sha256} observed={observed}"
        )
    missing = [
        name
        for name in ("safe_mla_query_bmm", "static_scaled_fp8_quant")
        if not hasattr(torch.ops._C, name)
    ]
    if missing:
        raise RuntimeError("missing vLLM custom ops: " + ", ".join(missing))
    return observed


@torch.inference_mode()
def _run_case(
    *,
    device: torch.device,
    tokens: int,
    seed: int,
    q_scale_value: float,
    call_mode: str,
) -> dict[str, Any]:
    # Preserve Gate A's input-generation contract exactly. A change to this
    # sequence requires a new reviewed reference, not an in-place edit.
    torch.manual_seed(seed)
    query_storage = (
        torch.randn(
            tokens,
            HEADS,
            Q_DIM + ROPE_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.5
    )
    q_nope_storage, q_pe = query_storage.split((Q_DIM, ROPE_DIM), dim=-1)
    query = q_nope_storage.transpose(0, 1)
    weight = (
        torch.randn(
            HEADS,
            Q_DIM,
            LATENT_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.05
    )
    projected = torch.empty(
        HEADS,
        tokens,
        LATENT_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    assembled = torch.empty(
        tokens,
        HEADS,
        LATENT_DIM + ROPE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    quantized = torch.empty_like(assembled, dtype=torch.float8_e4m3fn)
    q_scale = torch.tensor([q_scale_value], dtype=torch.float32, device=device)

    if query.is_contiguous():
        raise AssertionError("production split/transpose query must be non-contiguous")
    if call_mode == "precise":
        # Requiring the four-argument schema makes an image that silently
        # drops the precision control fail before any fingerprint comparison.
        torch.ops._C.safe_mla_query_bmm(query, weight, projected, True)
    else:
        torch.ops._C.safe_mla_query_bmm(query, weight, projected)
    torch.cat((projected.transpose(0, 1), q_pe), dim=-1, out=assembled)
    torch.ops._C.static_scaled_fp8_quant(
        quantized.view(tokens, -1),
        assembled.view(tokens, -1),
        q_scale,
    )
    torch.cuda.synchronize(device)

    return {
        "kind": "safe_query_bmm_fingerprint",
        "tokens": tokens,
        "heads": HEADS,
        "seed": seed,
        "q_scale": q_scale_value,
        "bf16_sha256": _tensor_sha256(projected),
        "fp8_sha256": _tensor_sha256(quantized),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-stable-sha256", required=True)
    parser.add_argument("--call-mode", choices=("legacy", "precise"), default="precise")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = list(torch.cuda.get_device_capability(device))
    if capability != [12, 0]:
        raise SystemExit(f"reference requires compute capability 12.0, got {capability}")
    if torch.version.cuda != "13.2":
        raise SystemExit(f"reference requires torch CUDA 13.2, got {torch.version.cuda}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    stable_sha256 = _load_ops(args.expected_stable_sha256)
    count = 0
    for tokens in TOKENS:
        for seed in SEEDS:
            for q_scale in Q_SCALES:
                _emit(
                    _run_case(
                        device=device,
                        tokens=tokens,
                        seed=seed,
                        q_scale_value=q_scale,
                        call_mode=args.call_mode,
                    ),
                    args.output,
                )
                count += 1
    _emit(
        {
            "kind": "summary",
            "status": "PASS",
            "cases": count,
            "call_mode": args.call_mode,
            "platform_id": "sm120-cu132",
            "compute_capability": capability,
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "stable_libtorch_sha256": stable_sha256,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
