#!/usr/bin/env python3
"""Dependency-free proof of the FP8 owner-shard consistency invariant.

This is a schedule/semantics proof, not a numerical FP8 emulator.  ``wire`` is
deliberately lossy so a pre-wire owner value differs from the value its peers
materialize.  The real CUDA codec is covered by the four-GPU gate in the patch
bundle.
"""

from __future__ import annotations

def wire(value: float) -> float:
    """Deterministic lossy stand-in for quantize/dequantize."""
    return round(value * 8.0) / 8.0


def make_inputs(world: int) -> list[list[float]]:
    return [
        [rank * 0.173 + shard * 0.097 + 0.011 for shard in range(world)]
        for rank in range(world)
    ]


def ring_owner_records(
    inputs: list[list[float]], mode: str
) -> list[tuple[int, float, float]]:
    """Run the source ring's reduce schedule through its final owner step.

    Each result is ``(owner_rank, pre_wire_owner_value, forwarded_payload)``.
    ``ring`` quantizes every partial; ``ag`` keeps the reduction unquantized
    and quantizes only the completed owner value.
    """
    world = len(inputs)
    working = [[0.0] * world for _ in range(world)]
    staged = [[0.0] * world for _ in range(world)]
    records: list[tuple[int, float, float] | None] = [None] * world

    for step in range(world - 1):
        sends: list[tuple[int, float]] = []
        for rank in range(world):
            send_shard = (rank - step) % world
            if mode == "ring":
                value = (
                    wire(inputs[rank][send_shard])
                    if step == 0
                    else staged[rank][send_shard]
                )
            else:
                value = (
                    inputs[rank][send_shard]
                    if step == 0
                    else working[rank][send_shard]
                )
            sends.append((send_shard, value))

        for rank in range(world):
            recv_shard = (rank - step - 1) % world
            sent_shard, incoming = sends[(rank - 1) % world]
            assert sent_shard == recv_shard
            reduced = inputs[rank][recv_shard] + incoming
            if mode == "ring":
                staged[rank][recv_shard] = wire(reduced)
            else:
                working[rank][recv_shard] = reduced
            if step == world - 2:
                payload = (
                    staged[rank][recv_shard]
                    if mode == "ring"
                    else wire(working[rank][recv_shard])
                )
                records[recv_shard] = (rank, reduced, payload)

    assert all(record is not None for record in records)
    return [record for record in records if record is not None]


def a2a_owner_records(inputs: list[list[float]]) -> list[tuple[int, float, float]]:
    """Run the a2a path's quantize-in, owner-accumulate, broadcast boundary."""
    world = len(inputs)
    records = []
    for shard in range(world):
        owner = shard
        reduced = inputs[owner][shard] + sum(
            wire(inputs[rank][shard]) for rank in range(world) if rank != owner
        )
        records.append((owner, reduced, wire(reduced)))
    return records


def materialize(
    records: list[tuple[int, float, float]], *, owner_roundtrip: bool
) -> list[list[float]]:
    """Materialize every completed output shard on every consumer rank."""
    world = len(records)
    outputs: list[list[float]] = []
    for consumer in range(world):
        rank_output = []
        for owner, owner_value, payload in records:
            if consumer == owner and not owner_roundtrip:
                rank_output.append(owner_value)
            else:
                rank_output.append(payload)
        outputs.append(rank_output)
    return outputs


def prove_mode(world: int, mode: str) -> None:
    inputs = make_inputs(world)
    records = (
        a2a_owner_records(inputs)
        if mode == "a2a"
        else ring_owner_records(inputs, mode)
    )
    assert all(owner_value != payload for _, owner_value, payload in records)
    legacy = materialize(records, owner_roundtrip=False)
    fixed = materialize(records, owner_roundtrip=True)

    # Every legacy rank retains a different pre-wire owner shard.
    assert len({tuple(row) for row in legacy}) == world
    for shard, (owner, owner_value, payload) in enumerate(records):
        assert legacy[owner][shard] == owner_value
        assert all(
            legacy[rank][shard] == payload
            for rank in range(world)
            if rank != owner
        )

    # The fix materializes the exact transmitted payload on every rank.
    expected = [payload for _, _, payload in records]
    assert fixed == [expected] * world
    print(
        f"PASS mode={mode} world={world}: legacy_unique={world} "
        "fixed_unique=1"
    )


def main() -> None:
    for world in (2, 4, 8):
        prove_mode(world, "ag")
        prove_mode(world, "ring")
        prove_mode(world, "a2a")


if __name__ == "__main__":
    main()
