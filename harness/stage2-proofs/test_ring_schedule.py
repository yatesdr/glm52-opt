#!/usr/bin/env python3
"""CPU proof for the packed-CKV three-slot all-gather schedule."""

from __future__ import annotations

from collections.abc import Sequence


WORLD_SIZE = 4
RECORD_BYTES = 368


class RingHarness:
    """Synchronous model of the proposed three receive slots per rank."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.slots = [
            [bytearray(capacity) for _ in range(WORLD_SIZE - 1)]
            for _ in range(WORLD_SIZE)
        ]

    def gather(self, payloads: Sequence[bytes]) -> tuple[bytes, ...]:
        if len(payloads) != WORLD_SIZE:
            raise ValueError("the proof is fixed to four ranks")
        if any(len(payload) > self.capacity for payload in payloads):
            raise ValueError("payload exceeds fixed slot capacity")

        lengths = tuple(len(payload) for payload in payloads)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)

        gathered = [bytearray(offsets[-1]) for _ in range(WORLD_SIZE)]
        for rank, payload in enumerate(payloads):
            gathered[rank][offsets[rank] : offsets[rank + 1]] = payload

        for step in range(WORLD_SIZE - 1):
            outbound: list[bytes] = []
            for rank in range(WORLD_SIZE):
                owner = (rank - step) % WORLD_SIZE
                if step == 0:
                    source = payloads[rank]
                else:
                    length = lengths[owner]
                    source = bytes(self.slots[rank][step - 1][:length])
                outbound.append(source)

            for sender, payload in enumerate(outbound):
                receiver = (sender + 1) % WORLD_SIZE
                owner = (sender - step) % WORLD_SIZE
                expected_owner = (receiver - step - 1) % WORLD_SIZE
                assert owner == expected_owner
                slot = self.slots[receiver][step]
                slot[: len(payload)] = payload
                gathered[receiver][offsets[owner] : offsets[owner + 1]] = slot[
                    : len(payload)
                ]

        return tuple(bytes(result) for result in gathered)


def make_payload(rank: int, records: int, tail: int) -> bytes:
    length = records * RECORD_BYTES + tail
    payload = bytearray(
        ((rank + 1) * 29 + offset * 73 + offset // 11) & 0xFF
        for offset in range(length)
    )
    markers = (
        b"\x00\x00\x00\x00",
        b"\xff\xff\xff\xff",
        b"\x00\x00\xc0\x7f",
        b"\x01\x00\x80\x7f",
    )
    for index, marker in enumerate(markers):
        start = index * 17
        if start + len(marker) <= length:
            payload[start : start + len(marker)] = marker
    return bytes(payload)


def assert_direct_concatenation(
    harness: RingHarness, payloads: tuple[bytes, ...]
) -> None:
    expected = b"".join(payloads)
    results = harness.gather(payloads)
    assert all(result == expected for result in results)


def main() -> None:
    large = (
        make_payload(0, 3, 7),
        make_payload(1, 1, 255),
        make_payload(2, 2, 1),
        make_payload(3, 4, 31),
    )
    small = (
        make_payload(0, 1, 3),
        make_payload(1, 0, 19),
        make_payload(2, 1, 5),
        make_payload(3, 0, 61),
    )
    assert all(len(after) < len(before) for before, after in zip(large, small))

    harness = RingHarness(max(map(len, large)))
    slot_ids = tuple(id(slot) for rank in harness.slots for slot in rank)

    assert_direct_concatenation(harness, large)
    first_call_slots = [
        [bytes(slot) for slot in rank_slots] for rank_slots in harness.slots
    ]
    assert_direct_concatenation(harness, large)
    assert tuple(id(slot) for rank in harness.slots for slot in rank) == slot_ids

    assert_direct_concatenation(harness, small)
    assert tuple(id(slot) for rank in harness.slots for slot in rank) == slot_ids

    stale_tails_checked = 0
    for receiver in range(WORLD_SIZE):
        for step in range(WORLD_SIZE - 1):
            owner = (receiver - step - 1) % WORLD_SIZE
            active = len(small[owner])
            assert bytes(harness.slots[receiver][step][active:]) == (
                first_call_slots[receiver][step][active:]
            )
            stale_tails_checked += 1

    print(
        "collective large: "
        f"lengths={tuple(map(len, large))} repeated_schedule=exact"
    )
    print(
        "collective smaller-after-larger: "
        f"lengths={tuple(map(len, small))} "
        f"receive_slots_per_rank={WORLD_SIZE - 1} "
        f"stale_tails_excluded={stale_tails_checked}"
    )
    print("PASS collective byte-exact ring schedule")


if __name__ == "__main__":
    main()
