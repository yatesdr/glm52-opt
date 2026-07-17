#!/usr/bin/env python3
"""CPU proof for rank-invariant phase-2 route and collective lengths."""

from __future__ import annotations

import random
from dataclasses import dataclass


WORLD_SIZE = 4
BLOCK_SIZE = 64
VIRTUAL_BLOCK = WORLD_SIZE * BLOCK_SIZE
RECORD_BYTES = 368
B_ACTIVE = 2403


@dataclass(frozen=True)
class GlobalMetadata:
    sequence_lengths: tuple[int, ...]
    physical_pages: int
    table_width: int


@dataclass(frozen=True)
class RouteSignature:
    route: str
    logical_blocks: int
    record_input_bytes: int
    record_output_bytes: int
    validity_input_bytes: int
    validity_output_bytes: int
    table_input_bytes: int
    table_output_bytes: int


class RouteSelector:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    @property
    def pool_warning_count(self) -> int:
        return len(self.warnings)

    def choose(self, metadata: GlobalMetadata) -> RouteSignature:
        request_count = len(metadata.sequence_lengths)
        logical_blocks = sum(
            (length + VIRTUAL_BLOCK - 1) // VIRTUAL_BLOCK
            for length in metadata.sequence_lengths
        )
        if logical_blocks <= B_ACTIVE:
            return RouteSignature(
                "active",
                logical_blocks,
                logical_blocks * BLOCK_SIZE * RECORD_BYTES,
                WORLD_SIZE * logical_blocks * BLOCK_SIZE * RECORD_BYTES,
                logical_blocks,
                WORLD_SIZE * logical_blocks,
                0,
                0,
            )

        table_input = request_count * metadata.table_width * 4
        signature = RouteSignature(
            "pool",
            logical_blocks,
            metadata.physical_pages * BLOCK_SIZE * RECORD_BYTES,
            WORLD_SIZE
            * metadata.physical_pages
            * BLOCK_SIZE
            * RECORD_BYTES,
            0,
            0,
            table_input,
            WORLD_SIZE * table_input,
        )
        if not self.warnings:
            self.warnings.append(
                "WARNING Packed-CKV physical-pool route activated "
                f"logical_B={logical_blocks} "
                f"gathered_bytes={signature.record_output_bytes}"
            )
        return signature


def metadata_for_blocks(
    rng: random.Random, block_counts: tuple[int, ...]
) -> GlobalMetadata:
    lengths = tuple(
        (blocks - 1) * VIRTUAL_BLOCK + rng.randint(1, VIRTUAL_BLOCK)
        for blocks in block_counts
    )
    return GlobalMetadata(
        sequence_lengths=lengths,
        physical_pages=rng.randint(1800, 2340),
        table_width=max(block_counts) + rng.randint(0, 7),
    )


def partition_blocks(
    rng: random.Random, total: int, request_count: int
) -> tuple[int, ...]:
    cuts = sorted(rng.sample(range(1, total), request_count - 1))
    points = (0, *cuts, total)
    return tuple(
        points[index + 1] - points[index]
        for index in range(request_count)
    )


def main() -> None:
    rng = random.Random(0x2403DC04)
    selectors = tuple(RouteSelector() for _ in range(WORLD_SIZE))
    cases = 0
    active_cases = 0
    pool_cases = 0

    totals = (1, 1875, 2348, 2402, 2403, 2404, 2530, 3000)
    metadata_cases: list[GlobalMetadata] = []
    for total in totals:
        request_count = min(8, total)
        metadata_cases.append(
            metadata_for_blocks(
                rng, partition_blocks(rng, total, request_count)
            )
        )
    for _ in range(1000):
        request_count = rng.randint(1, 8)
        total = rng.randint(request_count, 3200)
        metadata_cases.append(
            metadata_for_blocks(
                rng, partition_blocks(rng, total, request_count)
            )
        )

    for metadata in metadata_cases:
        # Owner tables intentionally differ; they are payload, not routing
        # inputs. Identical global metadata must still yield one signature.
        owner_table_checksums = tuple(
            cases * 100_000 + metadata.table_width * 10 + rank
            for rank in range(WORLD_SIZE)
        )
        assert len(set(owner_table_checksums)) == WORLD_SIZE
        signatures = tuple(
            selectors[rank].choose(metadata) for rank in range(WORLD_SIZE)
        )
        assert len(set(signatures)) == 1
        signature = signatures[0]
        if signature.route == "active":
            active_cases += 1
            assert signature.logical_blocks <= B_ACTIVE
            assert signature.validity_input_bytes == signature.logical_blocks
            assert signature.table_input_bytes == 0
        else:
            pool_cases += 1
            assert signature.logical_blocks > B_ACTIVE
            assert signature.record_input_bytes == (
                metadata.physical_pages * BLOCK_SIZE * RECORD_BYTES
            )
            assert signature.table_input_bytes == (
                len(metadata.sequence_lengths) * metadata.table_width * 4
            )
        assert signature.record_output_bytes == (
            WORLD_SIZE * signature.record_input_bytes
        )
        cases += 1

    assert active_cases > 0
    assert pool_cases > 0
    assert all(selector.pool_warning_count == 1 for selector in selectors)
    assert all(
        selector.warnings[0].startswith("WARNING ")
        and "gathered_bytes=" in selector.warnings[0]
        for selector in selectors
    )

    boundary_low = metadata_for_blocks(rng, (2403,))
    boundary_high = metadata_for_blocks(rng, (2404,))
    assert RouteSelector().choose(boundary_low).route == "active"
    assert RouteSelector().choose(boundary_high).route == "pool"
    print(
        "route determinism: "
        f"cases={cases} active={active_cases} pool={pool_cases} "
        f"boundary={B_ACTIVE}/{B_ACTIVE + 1}"
    )
    print(
        "pool observability: "
        f"once_per_process_warnings="
        f"{tuple(s.pool_warning_count for s in selectors)}"
    )
    print("PASS all ranks choose identical routes and collective byte lengths")


if __name__ == "__main__":
    main()
