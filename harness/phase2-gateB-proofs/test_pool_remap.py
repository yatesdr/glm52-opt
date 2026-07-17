#!/usr/bin/env python3
"""CPU proof for the phase-2 physical-pool fallback and table remap."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence


WORLD_SIZE = 4
BLOCK_SIZE = 64
VIRTUAL_BLOCK = WORLD_SIZE * BLOCK_SIZE
RECORD_BYTES = 368
B_ACTIVE = 2403
POOL_PAGES = 17
BLOCKS_PER_REQUEST = (700, 650, 620, 560)


def make_record(owner: int, physical: int, local_offset: int) -> bytes:
    header = struct.pack("<IIII", owner, physical, local_offset, 0xC0FFEE)
    digest = hashlib.blake2s(header, digest_size=32).digest()
    material = header + digest + b"\x00\xff\xc0\x7f"
    repeats = (RECORD_BYTES + len(material) - 1) // len(material)
    return (material * repeats)[:RECORD_BYTES]


def build_caches() -> tuple[tuple[tuple[bytes, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(
                make_record(owner, physical, local_offset)
                for local_offset in range(BLOCK_SIZE)
            )
            for physical in range(POOL_PAGES)
        )
        for owner in range(WORLD_SIZE)
    )


def physical_for(request: int, owner: int, virtual_block: int) -> int:
    if (request * 17 + owner * 29 + virtual_block) % 113 == 0:
        return -1
    return (request * 3 + owner * 5 + virtual_block * 7) % POOL_PAGES


def build_owner_tables() -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(
                physical_for(request, owner, virtual_block)
                for virtual_block in range(BLOCKS_PER_REQUEST[request])
            )
            for request in range(len(BLOCKS_PER_REQUEST))
        )
        for owner in range(WORLD_SIZE)
    )


def gather_pool(
    caches: tuple[tuple[tuple[bytes, ...], ...], ...],
) -> bytes:
    return b"".join(
        record
        for owner_cache in caches
        for page in owner_cache
        for record in page
    )


def remap_pool(
    request: int,
    logical_ids: Sequence[int],
    owner_tables: tuple[tuple[tuple[int, ...], ...], ...],
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    slots: list[int] = []
    kept: list[int] = []
    holes = 0
    causal_length = BLOCKS_PER_REQUEST[request] * VIRTUAL_BLOCK - 37

    for logical_id in logical_ids:
        if logical_id < 0 or logical_id >= causal_length:
            continue
        virtual_block = logical_id // VIRTUAL_BLOCK
        owner = logical_id % WORLD_SIZE
        local_offset = (logical_id % VIRTUAL_BLOCK) // WORLD_SIZE
        physical = owner_tables[owner][request][virtual_block]
        if physical < 0 or physical >= POOL_PAGES:
            holes += 1
            continue
        slots.append(
            (owner * POOL_PAGES + physical) * BLOCK_SIZE + local_offset
        )
        kept.append(logical_id)

    slots.extend([-1] * (len(logical_ids) - len(slots)))
    return tuple(slots), tuple(kept), holes


def direct_read(
    caches: tuple[tuple[tuple[bytes, ...], ...], ...],
    owner_tables: tuple[tuple[tuple[int, ...], ...], ...],
    request: int,
    logical_id: int,
) -> bytes:
    virtual_block = logical_id // VIRTUAL_BLOCK
    owner = logical_id % WORLD_SIZE
    local_offset = (logical_id % VIRTUAL_BLOCK) // WORLD_SIZE
    physical = owner_tables[owner][request][virtual_block]
    assert 0 <= physical < POOL_PAGES
    return caches[owner][physical][local_offset]


def selected_ids(request: int) -> tuple[int, ...]:
    block_count = BLOCKS_PER_REQUEST[request]
    virtual_blocks = (
        0,
        1,
        16,
        17,
        112,
        113,
        block_count // 2,
        block_count - 2,
        block_count - 1,
    )
    result = [-1]
    for virtual_block in virtual_blocks:
        base = virtual_block * VIRTUAL_BLOCK
        result.extend((base, base + 1, base + 252, base + 255))
    result.extend(
        (
            block_count * VIRTUAL_BLOCK - 38,
            block_count * VIRTUAL_BLOCK - 37,
            block_count * VIRTUAL_BLOCK + 999,
        )
    )
    return tuple(result)


def main() -> None:
    logical_blocks = sum(BLOCKS_PER_REQUEST)
    assert logical_blocks == 2530
    assert logical_blocks > B_ACTIVE
    caches = build_caches()
    owner_tables = build_owner_tables()
    gathered = gather_pool(caches)
    assert len(gathered) == (
        WORLD_SIZE * POOL_PAGES * BLOCK_SIZE * RECORD_BYTES
    )

    checked = 0
    holes = 0
    aliased_entries = 0
    for owner in range(WORLD_SIZE):
        flattened = [
            physical
            for request_tables in owner_tables[owner]
            for physical in request_tables
            if physical >= 0
        ]
        aliased_entries += len(flattened) - len(set(flattened))
    assert aliased_entries > 9000

    for request in range(len(BLOCKS_PER_REQUEST)):
        inputs = selected_ids(request)
        slots, kept, request_holes = remap_pool(
            request, inputs, owner_tables
        )
        valid_slots = tuple(slot for slot in slots if slot >= 0)
        assert slots[len(valid_slots) :] == (-1,) * (
            len(slots) - len(valid_slots)
        )
        assert len(valid_slots) == len(kept)
        for logical_id, slot in zip(kept, valid_slots):
            begin = slot * RECORD_BYTES
            record = gathered[begin : begin + RECORD_BYTES]
            assert record == direct_read(
                caches, owner_tables, request, logical_id
            )
        checked += len(kept)
        holes += request_holes

    assert checked >= 100
    assert holes >= 1
    print(
        "pool route: "
        f"logical_B={logical_blocks} active_cap={B_ACTIVE} "
        f"physical_pages/rank={POOL_PAGES} gathered_bytes={len(gathered)}"
    )
    print(
        "pool aliases: "
        f"aliased_table_entries={aliased_entries} "
        f"direct_reads_checked={checked} holes_dropped={holes}"
    )
    print("PASS pool remap reads equal direct owner-specific physical reads")


if __name__ == "__main__":
    main()
