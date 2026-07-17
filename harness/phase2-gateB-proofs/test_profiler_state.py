#!/usr/bin/env python3
"""CPU proof for the fixed-pool nested compute-profiler state machine."""

from __future__ import annotations

from dataclasses import dataclass


PHASES = (
    "layer_total",
    "mla_total",
    "indexer",
    "attention_path",
    "dcp_query_ag",
    "ckv_pack",
    "ckv_ag",
    "ckv_remap",
    "ckv_stage",
    "sparse_attn",
    "dcp_project",
    "dcp_lse_ag",
    "dcp_lse_correct",
    "dcp_output_rs",
    "o_proj_total",
    "tp_ar_attention",
    "moe_total",
    "tp_ar_moe",
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, milliseconds: float) -> None:
        self.now += milliseconds


@dataclass
class FakeEventPair:
    start: float | None = None
    end: float | None = None

    def elapsed(self) -> float:
        assert self.start is not None and self.end is not None
        return self.end - self.start


class FakeEventFactory:
    def __init__(self) -> None:
        self.created = 0
        self.sync_calls = 0

    def pair(self) -> FakeEventPair:
        self.created += 2
        return FakeEventPair()

    def synchronize(self) -> None:
        self.sync_calls += 1


class PhaseProfiler:
    def __init__(
        self,
        enabled: bool,
        factory: FakeEventFactory,
        clock: FakeClock,
        capacity: int,
    ) -> None:
        self.enabled = enabled
        self.factory = factory
        self.clock = clock
        self.capacity = capacity
        self.dead = False
        self.summary_lines = 0
        self.pairs = (
            {
                phase: tuple(factory.pair() for _ in range(capacity))
                for phase in PHASES
            }
            if enabled
            else {}
        )
        self.used = {phase: 0 for phase in PHASES} if enabled else {}
        self.open_index = (
            {phase: -1 for phase in PHASES} if enabled else {}
        )
        self.stack: list[str | None] = [None] * len(PHASES)
        self.depth = 0
        self.layer_count = 0
        self.layer_kind: list[str | None] = [None] * capacity
        self.attention_tp_calls = [0] * capacity
        self.moe_tp_calls = [0] * capacity
        self.attention_tp_bytes = [0] * capacity
        self.moe_tp_bytes = [0] * capacity

    def start(self, phase: str) -> None:
        if not self.enabled or self.dead:
            return
        if phase not in self.pairs or self.depth >= len(self.stack):
            self.dead = True
            return
        index = self.used[phase]
        if index >= self.capacity or self.open_index[phase] != -1:
            self.dead = True
            return
        pair = self.pairs[phase][index]
        pair.start = self.clock.now
        pair.end = None
        self.open_index[phase] = index
        self.stack[self.depth] = phase
        self.depth += 1

    def stop(self, phase: str) -> None:
        if not self.enabled or self.dead:
            return
        if self.depth == 0 or self.stack[self.depth - 1] != phase:
            self.dead = True
            return
        index = self.open_index[phase]
        if index < 0:
            self.dead = True
            return
        self.pairs[phase][index].end = self.clock.now
        self.open_index[phase] = -1
        self.used[phase] += 1
        self.depth -= 1
        self.stack[self.depth] = None

    def begin_layer(self, kind: str) -> None:
        if not self.enabled or self.dead:
            return
        if kind not in ("F", "S") or self.layer_count >= self.capacity:
            self.dead = True
            return
        self.layer_kind[self.layer_count] = kind
        self.start("layer_total")

    def record_tp_allreduce(self, byte_count: int) -> None:
        if not self.enabled or self.dead:
            return
        if self.layer_count >= self.capacity:
            self.dead = True
            return
        attention_ar = False
        attention_region = False
        moe_ar = False
        moe_region = False
        for index in range(self.depth):
            phase = self.stack[index]
            attention_ar = attention_ar or phase == "tp_ar_attention"
            attention_region = attention_region or phase == "o_proj_total"
            moe_ar = moe_ar or phase == "tp_ar_moe"
            moe_region = moe_region or phase == "moe_total"
        if attention_ar and attention_region:
            self.attention_tp_calls[self.layer_count] += 1
            self.attention_tp_bytes[self.layer_count] += byte_count
        elif moe_ar and moe_region:
            self.moe_tp_calls[self.layer_count] += 1
            self.moe_tp_bytes[self.layer_count] += byte_count
        else:
            self.dead = True

    def end_layer(self) -> None:
        if not self.enabled or self.dead:
            return
        self.stop("layer_total")
        if not self.dead:
            self.layer_count += 1

    def elapsed(self, phase: str) -> float:
        return sum(
            self.pairs[phase][index].elapsed()
            for index in range(self.used[phase])
        )

    def summary(
        self, expected_mean_layer_ms: float
    ) -> dict[str, float | int | bool]:
        if not self.enabled or self.dead:
            return {}
        self.dead = True
        self.factory.synchronize()
        elapsed = {phase: self.elapsed(phase) for phase in PHASES}
        attention_tags = sum(
            elapsed[phase]
            for phase in (
                "dcp_query_ag",
                "ckv_pack",
                "ckv_ag",
                "ckv_remap",
                "ckv_stage",
                "sparse_attn",
                "dcp_project",
                "dcp_lse_ag",
                "dcp_lse_correct",
                "dcp_output_rs",
            )
        )
        exclusive_sum = (
            elapsed["attention_path"]
            - attention_tags
            + elapsed["mla_total"]
            - elapsed["indexer"]
            - elapsed["attention_path"]
            - elapsed["o_proj_total"]
            + elapsed["o_proj_total"]
            - elapsed["tp_ar_attention"]
            + elapsed["moe_total"]
            - elapsed["tp_ar_moe"]
            + elapsed["layer_total"]
            - elapsed["mla_total"]
            - elapsed["moe_total"]
            + elapsed["indexer"]
            + attention_tags
            + elapsed["tp_ar_attention"]
            + elapsed["tp_ar_moe"]
        )
        classification_valid = all(
            self.attention_tp_calls[index] == 1
            and self.moe_tp_calls[index] == 1
            for index in range(self.layer_count)
        )
        mean_layer_ms = elapsed["layer_total"] / max(self.layer_count, 1)
        reproduction_error = abs(
            mean_layer_ms - expected_mean_layer_ms
        ) / expected_mean_layer_ms
        self.summary_lines += 1
        return {
            "layers": self.layer_count,
            "f_layers": sum(
                self.layer_kind[index] == "F"
                for index in range(self.layer_count)
            ),
            "s_layers": sum(
                self.layer_kind[index] == "S"
                for index in range(self.layer_count)
            ),
            "layer_total": elapsed["layer_total"],
            "exclusive_sum": exclusive_sum,
            "attention_local": elapsed["attention_path"] - attention_tags,
            "classification_valid": classification_valid,
            "reproduction_valid": reproduction_error <= 0.05,
            "reproduction_error": reproduction_error,
            "attention_tp_bytes": sum(self.attention_tp_bytes),
            "moe_tp_bytes": sum(self.moe_tp_bytes),
        }


def timed(profiler: PhaseProfiler, phase: str, duration: float) -> None:
    profiler.start(phase)
    profiler.clock.advance(duration)
    profiler.stop(phase)


def record_layer(
    profiler: PhaseProfiler,
    kind: str,
    route: str,
    *,
    extra_attention_ar: bool = False,
    tail_extra_ms: float = 0.0,
) -> None:
    profiler.begin_layer(kind)
    profiler.clock.advance(5)
    profiler.start("mla_total")
    profiler.clock.advance(2)
    if kind == "F":
        timed(profiler, "indexer", 5)
    else:
        profiler.clock.advance(5)
    profiler.start("attention_path")
    if route == "query":
        profiler.clock.advance(1)
        timed(profiler, "dcp_query_ag", 10)
        timed(profiler, "sparse_attn", 15)
        timed(profiler, "dcp_project", 2)
        timed(profiler, "dcp_lse_ag", 1)
        timed(profiler, "dcp_lse_correct", 3)
        timed(profiler, "dcp_output_rs", 4)
        profiler.clock.advance(4)
    elif route == "ckv":
        profiler.clock.advance(3)
        timed(profiler, "ckv_pack", 4)
        timed(profiler, "ckv_ag", 8)
        timed(profiler, "ckv_remap", 3)
        timed(profiler, "sparse_attn", 15)
        timed(profiler, "ckv_stage", 2)
        profiler.clock.advance(5)
    else:
        raise ValueError(route)
    profiler.stop("attention_path")
    profiler.start("o_proj_total")
    profiler.clock.advance(2)
    profiler.start("tp_ar_attention")
    profiler.record_tp_allreduce(36 * (1 << 20))
    if extra_attention_ar:
        profiler.record_tp_allreduce(36 * (1 << 20))
    profiler.clock.advance(3)
    profiler.stop("tp_ar_attention")
    profiler.clock.advance(3)
    profiler.stop("o_proj_total")
    profiler.stop("mla_total")
    profiler.start("moe_total")
    profiler.clock.advance(2)
    profiler.start("tp_ar_moe")
    profiler.record_tp_allreduce(24 * (1 << 20))
    profiler.clock.advance(4)
    profiler.stop("tp_ar_moe")
    profiler.clock.advance(29)
    profiler.stop("moe_total")
    profiler.clock.advance(5 + tail_extra_ms)
    profiler.end_layer()


def main() -> None:
    disabled_factory = FakeEventFactory()
    disabled = PhaseProfiler(False, disabled_factory, FakeClock(), 4)
    record_layer(disabled, "F", "query")
    assert disabled_factory.created == 0
    assert disabled.summary(100.0) == {}

    factory = FakeEventFactory()
    profiler = PhaseProfiler(True, factory, FakeClock(), 4)
    preallocated_events = factory.created
    assert preallocated_events == 2 * len(PHASES) * 4
    record_layer(profiler, "F", "query")
    record_layer(profiler, "S", "ckv")
    assert factory.created == preallocated_events
    result = profiler.summary(100.0)
    assert factory.sync_calls == 1
    assert profiler.summary_lines == 1
    assert result["layers"] == 2
    assert result["f_layers"] == 1
    assert result["s_layers"] == 1
    assert result["layer_total"] == 200.0
    assert result["exclusive_sum"] == result["layer_total"]
    assert result["attention_local"] == 13.0
    assert result["classification_valid"] is True
    assert result["reproduction_valid"] is True
    assert result["reproduction_error"] == 0.0
    assert result["attention_tp_bytes"] == 72 * (1 << 20)
    assert result["moe_tp_bytes"] == 48 * (1 << 20)

    faulty = PhaseProfiler(True, FakeEventFactory(), FakeClock(), 2)
    record_layer(faulty, "F", "query", extra_attention_ar=True)
    faulty_result = faulty.summary(100.0)
    assert faulty_result["classification_valid"] is False

    perturbed = PhaseProfiler(True, FakeEventFactory(), FakeClock(), 2)
    record_layer(perturbed, "F", "query", tail_extra_ms=6.0)
    perturbed_result = perturbed.summary(100.0)
    assert perturbed_result["reproduction_valid"] is False
    assert abs(perturbed_result["reproduction_error"] - 0.06) < 1e-9

    overflow = PhaseProfiler(True, FakeEventFactory(), FakeClock(), 1)
    record_layer(overflow, "F", "query")
    overflow.begin_layer("S")
    assert overflow.dead

    nesting = PhaseProfiler(True, FakeEventFactory(), FakeClock(), 1)
    nesting.start("layer_total")
    nesting.start("mla_total")
    nesting.stop("layer_total")
    assert nesting.dead

    print(
        "profiler accounting: "
        f"layers={result['layers']} F/S={result['f_layers']}/"
        f"{result['s_layers']} total={result['layer_total']:.1f}ms "
        f"exclusive={result['exclusive_sum']:.1f}ms"
    )
    print(
        "profiler discipline: "
        f"preallocated_events={preallocated_events} hot_allocations=0 "
        "syncs=1 reproduction_5pct=valid perturbation=detected "
        "tp_classification=valid faulty_tp=detected "
        "overflow=disabled nesting_error=disabled default_off=zero"
    )
    print("PASS profiler nesting, exclusives, counters, and self-disable")


if __name__ == "__main__":
    main()
