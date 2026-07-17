#!/usr/bin/env python3
"""CPU proof for request-major packed-CKV remapping and byte reads."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence


WORLD_SIZE = 4
BLOCK_SIZE = 64
VIRTUAL_BLOCK = WORLD_SIZE * BLOCK_SIZE
RECORD_BYTES = 368
SEQ_LENS = (260, 517)

# Indexed as [request][owner][virtual block]. Negative entries are holes.
BLOCK_TABLES = (
    (
        (31, 7),
        (4, 29),
        (18, 2),
        (-1, 11),
    ),
    (
        (23, 5, 41),
        (8, 37, 13),
        (44, -1, 6),
        (10, 1, 35),
    ),
)


def make_record(
    request: int, logical_id: int, owner: int, local_offset: int
) -> bytes:
    header = struct.pack("<IIII", request, logical_id, owner, local_offset)
    digest = hashlib.blake2s(header, digest_size=32).digest()
    material = header + digest + b"\x00\xff\x00\xff"
    repeats = (RECORD_BYTES + len(material) - 1) // len(material)
    return (material * repeats)[:RECORD_BYTES]


def build_caches() -> tuple[dict[int, tuple[bytes, ...]], ...]:
    caches: list[dict[int, tuple[bytes, ...]]] = [
        {} for _ in range(WORLD_SIZE)
    ]
    for request, rank_tables in enumerate(BLOCK_TABLES):
        for owner, physical_blocks in enumerate(rank_tables):
            for virtual_block, physical_block in enumerate(physical_blocks):
                if physical_block < 0:
                    continue
                assert physical_block not in caches[owner]
                page = tuple(
                    make_record(
                        request,
                        virtual_block * VIRTUAL_BLOCK
                        + local_offset * WORLD_SIZE
                        + owner,
                        owner,
                        local_offset,
                    )
                    for local_offset in range(BLOCK_SIZE)
                )
                caches[owner][physical_block] = page
    return tuple(caches)


def request_bases() -> tuple[tuple[int, ...], tuple[int, ...], int]:
    blocks = tuple(
        (seq_len + VIRTUAL_BLOCK - 1) // VIRTUAL_BLOCK
        for seq_len in SEQ_LENS
    )
    bases = [0]
    for count in blocks:
        bases.append(bases[-1] + count)
    return blocks, tuple(bases), bases[-1]


def pack_pages(
    caches: tuple[dict[int, tuple[bytes, ...]], ...],
    bases: tuple[int, ...],
    blocks: tuple[int, ...],
    packed_blocks: int,
) -> tuple[bytes, tuple[tuple[int, ...], ...]]:
    gathered = bytearray(
        WORLD_SIZE * packed_blocks * BLOCK_SIZE * RECORD_BYTES
    )
    validity = [[0] * packed_blocks for _ in range(WORLD_SIZE)]

    for request, block_count in enumerate(blocks):
        for virtual_block in range(block_count):
            packed_block = bases[request] + virtual_block
            for owner in range(WORLD_SIZE):
                physical_block = BLOCK_TABLES[request][owner][virtual_block]
                if physical_block < 0:
                    continue
                validity[owner][packed_block] = 1
                page = caches[owner][physical_block]
                page_index = owner * packed_blocks + packed_block
                start = page_index * BLOCK_SIZE * RECORD_BYTES
                gathered[start : start + BLOCK_SIZE * RECORD_BYTES] = (
                    b"".join(page)
                )
    return bytes(gathered), tuple(tuple(row) for row in validity)


def remap(
    request: int,
    logical_ids: Sequence[int],
    causal_length: int,
    blocks: tuple[int, ...],
    bases: tuple[int, ...],
    packed_blocks: int,
    validity: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    slots: list[int] = []
    kept_ids: list[int] = []
    hole_drops = 0

    for logical_id in logical_ids:
        if logical_id < 0 or logical_id >= causal_length:
            continue
        virtual_block = logical_id // VIRTUAL_BLOCK
        if virtual_block >= blocks[request]:
            continue
        owner = logical_id % WORLD_SIZE
        local_offset = (logical_id % VIRTUAL_BLOCK) // WORLD_SIZE
        packed_block = bases[request] + virtual_block
        if validity[owner][packed_block] == 0:
            hole_drops += 1
            continue
        slot = (
            (owner * packed_blocks + packed_block) * BLOCK_SIZE
            + local_offset
        )
        slots.append(slot)
        kept_ids.append(logical_id)

    slots.extend([-1] * (len(logical_ids) - len(slots)))
    return tuple(slots), tuple(kept_ids), hole_drops


def direct_read(
    caches: tuple[dict[int, tuple[bytes, ...]], ...],
    request: int,
    logical_id: int,
) -> bytes:
    owner = logical_id % WORLD_SIZE
    virtual_block = logical_id // VIRTUAL_BLOCK
    local_offset = (logical_id % VIRTUAL_BLOCK) // WORLD_SIZE
    physical_block = BLOCK_TABLES[request][owner][virtual_block]
    assert physical_block >= 0
    return caches[owner][physical_block][local_offset]


def main() -> None:
    caches = build_caches()
    blocks, bases, packed_blocks = request_bases()
    gathered, validity = pack_pages(caches, bases, blocks, packed_blocks)
    cases = (
        (
            0,
            (0, 1, 2, 3, 4, 252, 253, 254, 255, 256, 257, 258,
             259, 260, -1, 999),
            (0, 1, 2, 4, 252, 253, 254, 256, 257, 258, 259),
            (0, 320, 640, 1, 63, 383, 703, 64, 384, 704, 1024),
        ),
        (
            1,
            (0, 3, 252, 255, 256, 257, 258, 259, 508, 509, 510,
             511, 512, 513, 514, 515, 516, 517, -1, 999),
            (0, 3, 252, 255, 256, 257, 259, 508, 509, 511, 512,
             513, 514, 515, 516),
            (128, 1088, 191, 1151, 192, 512, 1152, 255, 575, 1215,
             256, 576, 896, 1216, 257),
        ),
    )

    total_kept = 0
    total_hole_drops = 0
    for request, logical_ids, expected_ids, expected_slots in cases:
        slots, kept_ids, hole_drops = remap(
            request,
            logical_ids,
            SEQ_LENS[request],
            blocks,
            bases,
            packed_blocks,
            validity,
        )
        valid_slots = tuple(slot for slot in slots if slot >= 0)
        assert kept_ids == expected_ids
        assert valid_slots == expected_slots
        assert len(valid_slots) == len(kept_ids)
        assert slots[len(valid_slots) :] == (-1,) * (
            len(slots) - len(valid_slots)
        )

        for logical_id, slot in zip(kept_ids, valid_slots):
            start = slot * RECORD_BYTES
            gathered_record = gathered[start : start + RECORD_BYTES]
            assert gathered_record == direct_read(
                caches, request, logical_id
            )
        total_kept += len(kept_ids)
        total_hole_drops += hole_drops
        print(
            f"remap request={request}: input={len(logical_ids)} "
            f"kept={len(kept_ids)} holes_dropped={hole_drops} "
            f"padded_invalid={len(logical_ids) - len(kept_ids)}"
        )

    assert blocks == (2, 3)
    assert bases == (0, 2, 5)
    assert total_hole_drops >= 4
    assert len(gathered) == (
        WORLD_SIZE * packed_blocks * BLOCK_SIZE * RECORD_BYTES
    )
    print(
        "remap layout: "
        f"shape=[{WORLD_SIZE},{packed_blocks},{BLOCK_SIZE},"
        f"{RECORD_BYTES}] bytes={len(gathered)} "
        f"kept={total_kept} holes_dropped={total_hole_drops}"
    )
    print("PASS remap gathered-view reads equal direct owner-shard reads")


if __name__ == "__main__":
    main()
