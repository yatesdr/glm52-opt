#!/usr/bin/env python3
"""CPU proof for the v20 B12X MTP split-K precision regression.

The production decode merge reads BF16 partial value vectors and merges them
with FP32 arithmetic.  FP32 at the merge cannot recover information already
discarded when each partition was written to BF16.
"""

from __future__ import annotations

import struct


def f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def bf16_bits(value: float) -> int:
    bits = struct.unpack(">I", struct.pack(">f", f32(value)))[0]
    # Round-to-nearest-even before truncating the low 16 mantissa bits.
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def bf16(value: float) -> float:
    return struct.unpack(">f", struct.pack(">I", bf16_bits(value) << 16))[0]


def f32_mean(values: list[float]) -> float:
    total = f32(0.0)
    for value in values:
        total = f32(total + f32(value))
    return f32(total / len(values))


def main() -> None:
    # Equal partition LSEs reduce the production softmax merge to an arithmetic
    # mean.  The +/-1 partials model cancellation between independently
    # normalized value-vector partitions.
    partials = [1.0, -0.999] * 4
    single_pass = bf16(f32_mean(partials))
    split_partials = [bf16(value) for value in partials]
    split_merge = bf16(f32_mean(split_partials))

    assert single_pass != 0.0
    assert split_merge == 0.0
    assert bf16_bits(1.0) == 0x3F80
    assert bf16_bits(-0.999) == 0xBF80

    global_topk = 2048
    dcp_world_size = 4
    local_winners = global_topk // dcp_world_size
    candidates_per_chunk = 64
    active_partitions = local_winners // candidates_per_chunk
    forced_partitions = global_topk // candidates_per_chunk
    heads = 64
    value_dim = 512
    bf16_bytes = 2
    active_partial_bytes = (
        active_partitions * heads * value_dim * bf16_bytes
    )
    one_split_output_bytes = heads * value_dim * bf16_bytes

    assert local_winners == 512
    assert active_partitions == 8
    assert forced_partitions == 32
    assert active_partial_bytes == 524_288
    assert one_split_output_bytes == 65_536

    print("PASS: BF16 split merge is not single-pass equivalent")
    print(
        "numeric:",
        f"single_pass={single_pass:.9f}",
        f"split_merge={split_merge:.9f}",
        f"partial_bf16={[hex(bf16_bits(v)) for v in partials[:2]]}",
    )
    print(
        "production geometry:",
        f"global_topk={global_topk}",
        f"local_winners~={local_winners}",
        f"forced_partitions={forced_partitions}",
        f"active_partitions~={active_partitions}",
        f"active_partial_bytes/row={active_partial_bytes}",
        f"single_split_output_bytes/row={one_split_output_bytes}",
    )


if __name__ == "__main__":
    main()
