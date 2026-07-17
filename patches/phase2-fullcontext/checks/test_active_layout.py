#!/usr/bin/env python3
"""CPU proof that phase-2 NCCL concatenation equals Stage-3 packing."""

from __future__ import annotations

import hashlib
import struct


WORLD_SIZE = 4
BLOCK_SIZE = 64
VIRTUAL_BLOCK = WORLD_SIZE * BLOCK_SIZE
RECORD_BYTES = 368
SEQ_LENS = (260, 517, 101)
BLOCK_TABLES = (
    ((31, 7), (4, 29), (18, 2), (-1, 11)),
    ((23, 5, 41), (8, 37, 13), (44, -1, 6), (10, 1, 35)),
    ((12,), (30,), (-1,), (9,)),
)


def make_record(
    request: int, owner: int, physical: int, local_offset: int
) -> bytes:
    header = struct.pack("<IIII", request, owner, physical, local_offset)
    digest = hashlib.sha256(header).digest()
    material = header + digest + b"\x00\xff\x00\xff"
    repeats = (RECORD_BYTES + len(material) - 1) // len(material)
    return (material * repeats)[:RECORD_BYTES]


def request_layout() -> tuple[tuple[int, ...], tuple[int, ...], int]:
    blocks = tuple(
        (length + VIRTUAL_BLOCK - 1) // VIRTUAL_BLOCK
        for length in SEQ_LENS
    )
    bases = [0]
    for count in blocks:
        bases.append(bases[-1] + count)
    return blocks, tuple(bases), bases[-1]


def pack_rank(
    owner: int,
    blocks: tuple[int, ...],
    bases: tuple[int, ...],
    packed_blocks: int,
) -> tuple[bytes, bytes]:
    records = bytearray(packed_blocks * BLOCK_SIZE * RECORD_BYTES)
    validity = bytearray(packed_blocks)
    for request, count in enumerate(blocks):
        for virtual_block in range(count):
            packed_block = bases[request] + virtual_block
            physical = BLOCK_TABLES[request][owner][virtual_block]
            if physical < 0:
                continue
            validity[packed_block] = 1
            page = b"".join(
                make_record(request, owner, physical, local_offset)
                for local_offset in range(BLOCK_SIZE)
            )
            begin = packed_block * BLOCK_SIZE * RECORD_BYTES
            records[begin : begin + len(page)] = page
    return bytes(records), bytes(validity)


def stage3_reference(
    blocks: tuple[int, ...],
    bases: tuple[int, ...],
    packed_blocks: int,
) -> tuple[bytes, bytes]:
    records = bytearray(
        WORLD_SIZE * packed_blocks * BLOCK_SIZE * RECORD_BYTES
    )
    validity = bytearray(WORLD_SIZE * packed_blocks)
    for request, block_count in enumerate(blocks):
        for virtual_block in range(block_count):
            packed_block = bases[request] + virtual_block
            for owner in range(WORLD_SIZE):
                physical = BLOCK_TABLES[request][owner][virtual_block]
                if physical < 0:
                    continue
                validity[owner * packed_blocks + packed_block] = 1
                for local_offset in range(BLOCK_SIZE):
                    slot = (
                        (owner * packed_blocks + packed_block) * BLOCK_SIZE
                        + local_offset
                    )
                    begin = slot * RECORD_BYTES
                    records[begin : begin + RECORD_BYTES] = make_record(
                        request, owner, physical, local_offset
                    )
    return bytes(records), bytes(validity)


def phase2_two_collectives(
    blocks: tuple[int, ...],
    bases: tuple[int, ...],
    packed_blocks: int,
) -> tuple[bytes, bytes, tuple[int, ...]]:
    rank_payloads = tuple(
        pack_rank(owner, blocks, bases, packed_blocks)
        for owner in range(WORLD_SIZE)
    )
    record_lengths = tuple(len(payload[0]) for payload in rank_payloads)
    validity_lengths = tuple(len(payload[1]) for payload in rank_payloads)
    assert len(set(record_lengths)) == 1
    assert len(set(validity_lengths)) == 1
    return (
        b"".join(payload[0] for payload in rank_payloads),
        b"".join(payload[1] for payload in rank_payloads),
        record_lengths,
    )


def main() -> None:
    blocks, bases, packed_blocks = request_layout()
    assert blocks == (2, 3, 1)
    assert bases == (0, 2, 5, 6)
    reference_records, reference_validity = stage3_reference(
        blocks, bases, packed_blocks
    )
    phase2_records, phase2_validity, record_lengths = (
        phase2_two_collectives(blocks, bases, packed_blocks)
    )

    assert phase2_records == reference_records
    assert phase2_validity == reference_validity
    assert record_lengths == (
        packed_blocks * BLOCK_SIZE * RECORD_BYTES,
    ) * WORLD_SIZE
    holes = phase2_validity.count(0)
    valid = phase2_validity.count(1)
    assert holes == 3
    assert valid == WORLD_SIZE * packed_blocks - holes

    partial_tail_records = sum(
        VIRTUAL_BLOCK - (length % VIRTUAL_BLOCK)
        for length in SEQ_LENS
        if length % VIRTUAL_BLOCK
    )
    assert partial_tail_records > 0
    digest = hashlib.sha256(phase2_records + phase2_validity).hexdigest()
    print(
        "active layout: "
        f"B={packed_blocks} local_record_bytes={record_lengths[0]} "
        f"global_record_bytes={len(phase2_records)}"
    )
    print(
        "active validity: "
        f"valid_pages={valid} holes={holes} "
        f"tail_records_outside_causal={partial_tail_records} "
        f"sha256={digest[:16]}"
    )
    print("PASS NCCL-style rank concatenation equals Stage-3 gathered stream")


if __name__ == "__main__":
    main()
