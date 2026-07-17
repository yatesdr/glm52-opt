#!/usr/bin/env python3
"""CPU proof that query ownership can be inverted to packed-KV ownership."""

from __future__ import annotations

import math
import random
import struct
from collections.abc import Sequence
from dataclasses import dataclass


WORLD_SIZE = 4
LOCAL_HEADS = 2
GLOBAL_HEADS = WORLD_SIZE * LOCAL_HEADS
BLOCK_SIZE = 64
VIRTUAL_BLOCK = WORLD_SIZE * BLOCK_SIZE
QK_DIM = 5
VALUE_DIM = 4
ATOL = 5e-5
RTOL = 5e-5


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def random_vector(rng: random.Random, size: int) -> tuple[float, ...]:
    return tuple(f32(rng.uniform(-0.5, 0.5)) for _ in range(size))


@dataclass(frozen=True)
class CacheRecord:
    request: int
    logical_id: int
    keys: tuple[tuple[float, ...], ...]
    values: tuple[tuple[float, ...], ...]


SEQ_LENS = (263, 519)
BLOCK_TABLES = (
    (
        (19, 3),
        (7, 31),
        (22, 5),
        (14, 28),
    ),
    (
        (2, 41, 17),
        (25, 4, 38),
        (9, 33, 1),
        (36, 6, 20),
    ),
)


def build_caches() -> tuple[dict[int, tuple[CacheRecord, ...]], ...]:
    rng = random.Random(0xC0FFEE)
    caches: list[dict[int, tuple[CacheRecord, ...]]] = [
        {} for _ in range(WORLD_SIZE)
    ]
    for request, rank_tables in enumerate(BLOCK_TABLES):
        for owner, physical_blocks in enumerate(rank_tables):
            for virtual_block, physical_block in enumerate(physical_blocks):
                page: list[CacheRecord] = []
                for local_offset in range(BLOCK_SIZE):
                    logical_id = (
                        virtual_block * VIRTUAL_BLOCK
                        + local_offset * WORLD_SIZE
                        + owner
                    )
                    keys = tuple(
                        random_vector(rng, QK_DIM)
                        for _ in range(GLOBAL_HEADS)
                    )
                    values = tuple(
                        random_vector(rng, VALUE_DIM)
                        for _ in range(GLOBAL_HEADS)
                    )
                    page.append(
                        CacheRecord(request, logical_id, keys, values)
                    )
                assert physical_block not in caches[owner]
                caches[owner][physical_block] = tuple(page)
    return tuple(caches)


def lookup_record(
    caches: tuple[dict[int, tuple[CacheRecord, ...]], ...],
    request: int,
    logical_id: int,
) -> tuple[int, CacheRecord] | None:
    if logical_id < 0 or logical_id >= SEQ_LENS[request]:
        return None
    virtual_block = logical_id // VIRTUAL_BLOCK
    owner = logical_id % WORLD_SIZE
    local_offset = (logical_id % VIRTUAL_BLOCK) // WORLD_SIZE
    physical_block = BLOCK_TABLES[request][owner][virtual_block]
    record = caches[owner][physical_block][local_offset]
    assert record.request == request
    assert record.logical_id == logical_id
    return owner, record


def dot_f32(left: Sequence[float], right: Sequence[float]) -> float:
    total = f32(0.0)
    for lhs, rhs in zip(left, right):
        total = f32(total + f32(lhs * rhs))
    return total


def attention(
    query: Sequence[float],
    records: Sequence[CacheRecord],
    head: int,
) -> tuple[tuple[float, ...], float]:
    if not records:
        return (0.0,) * VALUE_DIM, -math.inf

    scale = f32(1.0 / math.sqrt(QK_DIM))
    scores = [
        f32(dot_f32(query, record.keys[head]) * scale) for record in records
    ]
    score_max = max(scores)
    weights = [f32(math.exp(f32(score - score_max))) for score in scores]
    denominator = f32(0.0)
    for weight in weights:
        denominator = f32(denominator + weight)

    output: list[float] = []
    for dimension in range(VALUE_DIM):
        numerator = f32(0.0)
        for weight, record in zip(weights, records):
            contribution = f32(weight * record.values[head][dimension])
            numerator = f32(numerator + contribution)
        output.append(f32(numerator / denominator))
    lse = f32(score_max + math.log(denominator))
    return tuple(output), lse


def merge_partial_attention(
    partials: Sequence[tuple[tuple[float, ...], float]],
) -> tuple[tuple[float, ...], float]:
    finite = [partial for partial in partials if math.isfinite(partial[1])]
    assert finite
    lse_max = max(lse for _, lse in finite)
    weights = [f32(math.exp(f32(lse - lse_max))) for _, lse in finite]
    denominator = f32(0.0)
    for weight in weights:
        denominator = f32(denominator + weight)

    output: list[float] = []
    for dimension in range(VALUE_DIM):
        numerator = f32(0.0)
        for weight, (partial_output, _) in zip(weights, finite):
            contribution = f32(weight * partial_output[dimension])
            numerator = f32(numerator + contribution)
        output.append(f32(numerator / denominator))
    lse = f32(lse_max + math.log(denominator))
    return tuple(output), lse


def close(left: float, right: float) -> bool:
    return abs(left - right) <= ATOL + RTOL * abs(right)


def local_context_lengths(sequence_length: int) -> tuple[int, ...]:
    return tuple(
        max(0, (sequence_length - owner + WORLD_SIZE - 1) // WORLD_SIZE)
        for owner in range(WORLD_SIZE)
    )


def main() -> None:
    caches = build_caches()
    rows = (
        (0, (0, 3, 4, 63, 64, 127, 128, 252, 253, 254, 255,
             256, 257, 258, 262, -1, 263, 999)),
        (1, (0, 1, 127, 255, 256, 257, 258, 511, 512, 513, 514,
             516, -1, 517, 999)),
    )

    query_rng = random.Random(0x51A6E)
    local_queries = tuple(
        tuple(
            tuple(random_vector(query_rng, QK_DIM) for _ in range(LOCAL_HEADS))
            for _ in rows
        )
        for _ in range(WORLD_SIZE)
    )
    gathered_queries = tuple(
        tuple(
            query
            for rank in range(WORLD_SIZE)
            for query in local_queries[rank][row_index]
        )
        for row_index in range(len(rows))
    )

    max_output_error = 0.0
    max_lse_error = 0.0
    invalid_entries = 0
    owner_counts_by_row: list[tuple[int, ...]] = []

    for row_index, (request, selected_ids) in enumerate(rows):
        owner_records: list[list[CacheRecord]] = [
            [] for _ in range(WORLD_SIZE)
        ]
        for logical_id in selected_ids:
            lookup = lookup_record(caches, request, logical_id)
            if lookup is None:
                invalid_entries += 1
                continue
            owner, record = lookup
            owner_records[owner].append(record)
        owner_counts_by_row.append(tuple(map(len, owner_records)))

        gathered_records = [
            record for owner_records_part in owner_records
            for record in owner_records_part
        ]
        for query_owner in range(WORLD_SIZE):
            for local_head in range(LOCAL_HEADS):
                global_head = query_owner * LOCAL_HEADS + local_head
                baseline_query = gathered_queries[row_index][global_head]
                proposed_query = local_queries[query_owner][row_index][local_head]
                assert baseline_query == proposed_query

                partials = tuple(
                    attention(baseline_query, records, global_head)
                    for records in owner_records
                )
                baseline_output, baseline_lse = merge_partial_attention(
                    partials
                )
                proposed_output, proposed_lse = attention(
                    proposed_query, gathered_records, global_head
                )

                for baseline, proposed in zip(
                    baseline_output, proposed_output
                ):
                    error = abs(baseline - proposed)
                    max_output_error = max(max_output_error, error)
                    assert close(baseline, proposed), (
                        query_owner,
                        local_head,
                        baseline,
                        proposed,
                    )
                lse_error = abs(baseline_lse - proposed_lse)
                max_lse_error = max(max_lse_error, lse_error)
                assert close(baseline_lse, proposed_lse)

    assert len(rows) > 1
    assert invalid_entries >= 4
    assert any(len(set(counts)) > 1 for counts in owner_counts_by_row)
    context_lengths = tuple(map(local_context_lengths, SEQ_LENS))
    assert all(len(set(lengths)) > 1 for lengths in context_lengths)
    for rank_tables in BLOCK_TABLES:
        for physical_blocks in rank_tables:
            assert any(
                right != left + 1
                for left, right in zip(physical_blocks, physical_blocks[1:])
            )

    print(
        "ownership inversion: "
        f"requests={len(rows)} local_lengths={context_lengths} "
        f"owner_counts={tuple(owner_counts_by_row)} "
        f"invalid_entries={invalid_entries}"
    )
    print(
        "ownership inversion errors: "
        f"max_output={max_output_error:.9g} "
        f"max_lse={max_lse_error:.9g} "
        f"atol={ATOL:g} rtol={RTOL:g}"
    )
    print("PASS end-to-end ownership-inversion equivalence")


if __name__ == "__main__":
    main()
