#!/usr/bin/env python3
"""Compare a traced embedding row with its checkpoint source row.

This probe is intentionally model-free.  It consumes the single-row layer-0
records emitted by the long-context trace and the safetensors shard containing
``model.embed_tokens.weight``.  Its purpose is to distinguish an input/token or
weight-loading mismatch from a transport-induced numerical difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    delta = actual_f32 - expected_f32
    expected_norm = torch.linalg.vector_norm(expected_f32)
    return {
        "exact": bool(torch.equal(actual, expected)),
        "changed_elements": int(torch.count_nonzero(actual != expected).item()),
        "numel": int(actual.numel()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(delta) / expected_norm).item()
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual_f32, expected_f32, dim=0
            ).item()
        ),
    }


def _load_hidden(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    record = torch.load(path, map_location="cpu", weights_only=False)
    hidden = record["hidden"].detach().contiguous()
    metadata = {
        "schema": record.get("schema"),
        "layer": int(record["layer"]),
        "stage": str(record["stage"]),
        "tp_rank": int(record["tp_rank"]),
        "batch_tokens": int(record["batch_tokens"]),
        "absolute_position": int(record["absolute_position"]),
        "dtype": str(hidden.dtype),
        "shape": list(hidden.shape),
        "sha256": _sha256(path),
    }
    return hidden, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v19-trace", required=True, type=Path)
    parser.add_argument("--v20-trace", required=True, type=Path)
    parser.add_argument("--checkpoint-shard", required=True, type=Path)
    parser.add_argument("--tensor-name", default="model.embed_tokens.weight")
    parser.add_argument("--token-id", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    v19, v19_metadata = _load_hidden(args.v19_trace)
    v20, v20_metadata = _load_hidden(args.v20_trace)
    if v19_metadata["absolute_position"] != v20_metadata["absolute_position"]:
        raise RuntimeError("trace records do not refer to the same absolute position")
    if v19.shape != v20.shape or v19.dtype != v20.dtype:
        raise RuntimeError("trace rows do not share shape and dtype")

    with safe_open(
        args.checkpoint_shard, framework="pt", device="cpu"
    ) as checkpoint:
        tensor_slice = checkpoint.get_slice(args.tensor_name)
        tensor_shape = list(tensor_slice.get_shape())
        expected = tensor_slice[args.token_id : args.token_id + 1][0]

    if expected.shape != v19.shape or expected.dtype != v19.dtype:
        raise RuntimeError("checkpoint row does not share trace shape and dtype")

    report = {
        "schema": "v19-v20-embedding-transport-probe-v1",
        "contract": {
            "tensor_name": args.tensor_name,
            "token_id": args.token_id,
            "checkpoint_tensor_shape": tensor_shape,
            "checkpoint_shard_sha256": _sha256(args.checkpoint_shard),
        },
        "v19": {
            "trace": v19_metadata,
            "vs_checkpoint": _comparison(v19, expected),
        },
        "v20": {
            "trace": v20_metadata,
            "vs_checkpoint": _comparison(v20, expected),
        },
        "v19_vs_v20": _comparison(v19, v20),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
