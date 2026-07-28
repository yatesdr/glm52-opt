#!/usr/bin/env python3
"""Dependency-free proof of the block-INT8 wire layout and error bound."""

from __future__ import annotations

import math
import random
import struct


BLOCK = 128
QMAX = 127


def bf16_round(value: float) -> float:
    """Round a finite float32 value to BF16, ties to even."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def quantize_block(values: list[float]) -> tuple[list[int], float]:
    assert len(values) == BLOCK
    amax = max(abs(value) for value in values)
    scale = amax / QMAX if amax else 1.0
    payload = [max(-QMAX, min(QMAX, round(value / scale))) for value in values]
    return payload, scale


def codec(values: list[float]) -> tuple[list[int], list[float], list[float]]:
    assert len(values) % BLOCK == 0
    payload: list[int] = []
    scales: list[float] = []
    materialized: list[float] = []
    for start in range(0, len(values), BLOCK):
        block = values[start : start + BLOCK]
        quantized, scale = quantize_block(block)
        raw = [value * scale for value in quantized]
        bound = max(abs(value) for value in block) / (2 * QMAX)
        assert max(abs(left - right) for left, right in zip(block, raw)) <= (
            bound + 1e-12
        )
        payload.extend(quantized)
        scales.append(scale)
        materialized.extend(bf16_round(value) for value in raw)
    return payload, scales, materialized


def cases() -> dict[str, list[float]]:
    rng = random.Random(0xB12)
    normal = [bf16_round(rng.gauss(0.0, 0.6)) for _ in range(4 * BLOCK)]
    outliers = [bf16_round(rng.gauss(0.0, 0.05)) for _ in range(4 * BLOCK)]
    for index, value in enumerate((8.0, -11.0, 4.5, -6.25)):
        outliers[index * BLOCK + 17] = value
    smooth = [
        bf16_round(math.sin(index * 0.017) * 0.75)
        for index in range(4 * BLOCK)
    ]
    return {
        "zeros": [0.0] * (4 * BLOCK),
        "normal": normal,
        "outliers": outliers,
        "smooth": smooth,
    }


def main() -> None:
    for name, values in cases().items():
        payload, scales, owner = codec(values)
        _, _, peer = codec(values)
        blocks = len(values) // BLOCK
        wire_bytes = len(payload) + len(scales) * 4
        assert wire_bytes == blocks * (BLOCK + 4)
        assert all(-QMAX <= value <= QMAX for value in payload)
        assert owner == peer
        max_abs = max(abs(left - right) for left, right in zip(values, owner))
        print(
            f"PASS case={name} blocks={blocks} wire_bytes={wire_bytes} "
            f"owner_peer_identical=1 max_abs_after_bf16={max_abs:.8f}"
        )


if __name__ == "__main__":
    main()
