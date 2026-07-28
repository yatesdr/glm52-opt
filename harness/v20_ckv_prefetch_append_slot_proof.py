#!/usr/bin/env python3
"""CPU proof for v20 CKV-prefetch current-chunk slot placement.

The asynchronous CKV path gathers each future layer before that layer writes
the current chunk to its paged cache.  On consumption,
``_append_current_chunk_to_gathered`` reconstructs the missing records.  This
proof mirrors its integer mapping and compares it with the rank-packed layout
produced by a synchronous gather.

It intentionally proves only layout/ownership.  Quantization bytes and CUDA
stream/collective ordering require the separate GPU equivalence probe.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def owner_local(
    global_pos: int,
    *,
    world_size: int,
    interleave: int,
) -> tuple[int, int]:
    owner = (global_pos // interleave) % world_size
    local_pos = (
        global_pos // (world_size * interleave) * interleave
        + global_pos % interleave
    )
    return owner, local_pos


def rank_len(
    seq_len: int,
    rank: int,
    *,
    world_size: int,
    interleave: int,
) -> int:
    cycle = world_size * interleave
    full_cycles, remainder = divmod(seq_len, cycle)
    return (
        full_cycles * interleave
        + min(interleave, max(0, remainder - rank * interleave))
    )


def prove_case(
    seq_lens: tuple[int, ...],
    chunk_lens: tuple[int, ...],
    *,
    world_size: int,
    interleave: int,
    block_size: int,
) -> None:
    if len(seq_lens) != len(chunk_lens):
        raise AssertionError("sequence/chunk arity mismatch")
    if any(chunk < 0 or chunk > seq for seq, chunk in zip(seq_lens, chunk_lens)):
        raise AssertionError("invalid chunk length")

    num_reqs = len(seq_lens)
    rank_req_lens = [
        [
            rank_len(
                seq,
                rank,
                world_size=world_size,
                interleave=interleave,
            )
            for seq in seq_lens
        ]
        for rank in range(world_size)
    ]
    rank_req_starts = []
    for lengths in rank_req_lens:
        starts = [0]
        for length in lengths[:-1]:
            starts.append(starts[-1] + length)
        rank_req_starts.append(starts)
    padded_tokens = ceil_div(
        max(sum(lengths) for lengths in rank_req_lens),
        block_size,
    ) * block_size

    # Mirror _append_current_chunk_to_gathered exactly: request-major current
    # tokens, global_pos = final_seq_len - request_chunk_len + local_chunk_pos.
    query_starts = [0]
    for chunk_len in chunk_lens:
        query_starts.append(query_starts[-1] + chunk_len)
    req_ids = list(
        itertools.chain.from_iterable(
            itertools.repeat(req_id, chunk_len)
            for req_id, chunk_len in enumerate(chunk_lens)
        )
    )
    append_slots: dict[int, tuple[int, int]] = {}
    for t, req_id in enumerate(req_ids):
        req_chunk_start = query_starts[req_id]
        req_chunk_len = chunk_lens[req_id]
        global_pos = (
            seq_lens[req_id]
            - req_chunk_len
            + (t - req_chunk_start)
        )
        owner, local_pos = owner_local(
            global_pos,
            world_size=world_size,
            interleave=interleave,
        )
        slot = (
            owner * padded_tokens
            + rank_req_starts[owner][req_id]
            + local_pos
        )
        token = (req_id, global_pos)
        history_len_on_owner = rank_len(
            seq_lens[req_id] - req_chunk_len,
            owner,
            world_size=world_size,
            interleave=interleave,
        )
        final_len_on_owner = rank_req_lens[owner][req_id]
        if not history_len_on_owner <= local_pos < final_len_on_owner:
            raise AssertionError(
                "append did not land in the synchronous layout's current-chunk "
                f"suffix: {token=} {owner=} {local_pos=} "
                f"{history_len_on_owner=} {final_len_on_owner=}"
            )
        if slot in append_slots:
            raise AssertionError(
                f"append slot collision: {slot=} "
                f"{append_slots[slot]=} {token=}"
            )
        append_slots[slot] = token

    if len(append_slots) != sum(chunk_lens):
        raise AssertionError("append did not cover every current token exactly once")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    cases = 0
    for world_size in (2, 4, 8):
        for interleave in (1, 2, 8, 16, 64):
            for seq_lens, chunk_lens in (
                ((1,), (1,)),
                ((65,), (1,)),
                ((65,), (64,)),
                ((65_537,), (2048,)),
                ((343_727,), (1711,)),
                ((343_727,), (2048,)),
                ((17, 65), (3, 11)),
                ((4097, 8193, 16_385), (2048, 17, 1025)),
            ):
                prove_case(
                    seq_lens,
                    chunk_lens,
                    world_size=world_size,
                    interleave=interleave,
                    block_size=64,
                )
                cases += 1

    rng = random.Random(args.seed)
    for _ in range(args.random_cases):
        world_size = rng.choice((2, 4, 8))
        interleave = rng.choice((1, 2, 4, 8, 16, 32, 64))
        num_reqs = rng.randint(1, 16)
        seq_lens = tuple(rng.randint(1, 480_000) for _ in range(num_reqs))
        chunk_lens = tuple(
            rng.randint(1, min(seq_len, 64)) for seq_len in seq_lens
        )
        prove_case(
            seq_lens,
            chunk_lens,
            world_size=world_size,
            interleave=interleave,
            block_size=64,
        )
        cases += 1

    print(
        json.dumps(
            {
                "kind": "ckv_prefetch_append_slot_proof",
                "cases": cases,
                "random_seed": args.seed,
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
