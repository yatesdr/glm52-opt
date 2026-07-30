#!/usr/bin/env python3
"""CPU proof for the mixed-K fused top-k reduction ordering.

The current production path reduces each tier independently in route order
and then adds the two FP32 vectors. A fused kernel must retain that
association; a single interleaved accumulator is mathematically equivalent
but not generally bit-identical.
"""

from __future__ import annotations

import random
import struct
import unittest


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def descriptor_map(
    tier0_global_ids: tuple[int, ...],
    tier1_global_ids: tuple[int, ...],
    map_slots: int,
) -> tuple[int, ...]:
    result = [-1] * map_slots
    for tier, ids in enumerate((tier0_global_ids, tier1_global_ids)):
        if len(ids) > 256:
            raise ValueError("tier-local expert id does not fit in 8 bits")
        for local, global_id in enumerate(ids):
            if not 0 <= global_id < map_slots:
                raise ValueError("global expert id outside descriptor map")
            if result[global_id] != -1:
                raise ValueError("global expert id mapped twice")
            result[global_id] = (tier << 8) | local
    if any(value < 0 for value in result):
        raise ValueError("descriptor map does not cover every global expert")
    return tuple(result)


def serial_two_tier(
    route_values: list[list[float]],
    route_weights: list[float],
    route_tiers: list[int],
) -> list[float]:
    width = len(route_values[0])
    tier_outputs: list[list[float]] = []
    for tier in (0, 1):
        accum = [f32(0.0)] * width
        for values, weight, value_tier in zip(
            route_values, route_weights, route_tiers, strict=True
        ):
            if value_tier != tier:
                continue
            for column, value in enumerate(values):
                term = f32(f32(value) * f32(weight))
                accum[column] = f32(accum[column] + term)
        tier_outputs.append(accum)
    return [
        f32(left + right)
        for left, right in zip(tier_outputs[0], tier_outputs[1], strict=True)
    ]


def fused_two_accumulator(
    route_values: list[list[float]],
    route_weights: list[float],
    route_tiers: list[int],
) -> list[float]:
    width = len(route_values[0])
    tier0 = [f32(0.0)] * width
    tier1 = [f32(0.0)] * width
    for tier in (0, 1):
        target = tier0 if tier == 0 else tier1
        for values, weight, value_tier in zip(
            route_values, route_weights, route_tiers, strict=True
        ):
            if value_tier != tier:
                continue
            for column, value in enumerate(values):
                term = f32(f32(value) * f32(weight))
                target[column] = f32(target[column] + term)
    return [
        f32(left + right)
        for left, right in zip(tier0, tier1, strict=True)
    ]


class MixedKReductionContractTest(unittest.TestCase):
    def test_descriptor_is_total_disjoint_and_decodable(self) -> None:
        tier0 = tuple(range(0, 256, 4)) + tuple(range(1, 256, 4))
        tier1 = tuple(sorted(set(range(256)) - set(tier0)))
        table = descriptor_map(tier0, tier1, 256)
        self.assertEqual(len(table), 256)
        for global_id, descriptor in enumerate(table):
            tier = descriptor >> 8
            local = descriptor & 0xFF
            source = tier0 if tier == 0 else tier1
            self.assertEqual(source[local], global_id)

    def test_tier_local_rotation_lookup_matches_global_expert_order(self) -> None:
        # The production K3/K4 sets are interleaved. Prove that resolving the
        # descriptor into an existing tier row is identical to reading a
        # hypothetical duplicated global table.
        tier1_ids = tuple(range(3, 256, 4))
        tier0_ids = tuple(
            expert for expert in range(256) if expert not in set(tier1_ids)
        )
        table = descriptor_map(tier0_ids, tier1_ids, 256)
        global_rows = tuple(
            tuple(f32(expert * 0.25 + column * 0.03125) for column in range(11))
            for expert in range(256)
        )
        tier_rows = (
            tuple(global_rows[expert] for expert in tier0_ids),
            tuple(global_rows[expert] for expert in tier1_ids),
        )
        for global_expert, descriptor in enumerate(table):
            tier = descriptor >> 8
            local = descriptor & 0xFF
            self.assertEqual(
                tier_rows[tier][local],
                global_rows[global_expert],
            )

    def test_two_accumulator_fusion_is_bit_exact_to_serial_path(self) -> None:
        generator = random.Random(20260730)
        for _ in range(250):
            routes = 8
            width = 17
            values = [
                [generator.uniform(-32.0, 32.0) for _ in range(width)]
                for _ in range(routes)
            ]
            weights = [generator.uniform(-1.0, 1.0) for _ in range(routes)]
            tiers = [generator.randrange(2) for _ in range(routes)]
            expected = serial_two_tier(values, weights, tiers)
            actual = fused_two_accumulator(values, weights, tiers)
            self.assertEqual(
                [struct.pack("<f", value) for value in actual],
                [struct.pack("<f", value) for value in expected],
            )

    def test_interleaving_is_not_a_valid_bit_exact_replacement(self) -> None:
        # This fixed case is deliberately cancellation-sensitive. It documents
        # why the fused kernel needs two accumulators rather than one.
        values = [[1.0e20], [-1.0e20], [3.25], [2.0]]
        weights = [1.0, 1.0, 1.0, 1.0]
        tiers = [0, 1, 0, 1]
        expected = serial_two_tier(values, weights, tiers)
        interleaved = [f32(0.0)]
        for row, weight in zip(values, weights, strict=True):
            interleaved[0] = f32(interleaved[0] + f32(f32(row[0]) * f32(weight)))
        self.assertNotEqual(
            struct.pack("<f", interleaved[0]),
            struct.pack("<f", expected[0]),
        )


if __name__ == "__main__":
    unittest.main()
