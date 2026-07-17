#!/usr/bin/env python3
"""CPU proof for the phase-2 direct-CUDA headroom escrow state machine."""

from __future__ import annotations

from dataclasses import dataclass, field


WORLD_SIZE = 4
MIB = 1 << 20
ESCROW_BYTES = 192 * MIB
HARD_GATE_BYTES = 150 * MIB


class FatalEscrowError(RuntimeError):
    pass


@dataclass
class FakeCuda:
    free_bytes: int
    malloc_calls: int = 0
    free_calls: int = 0
    next_pointer: int = 0x100000
    live: dict[int, int] = field(default_factory=dict)

    def mem_get_info(self) -> tuple[int, int]:
        return self.free_bytes, 96 * (1 << 30)

    def malloc(self, size: int) -> int:
        self.malloc_calls += 1
        if size > self.free_bytes:
            raise RuntimeError("simulated cudaMalloc OOM")
        pointer = self.next_pointer
        self.next_pointer += size + 256
        self.live[pointer] = size
        self.free_bytes -= size
        return pointer

    def free(self, pointer: int) -> None:
        if pointer not in self.live:
            raise AssertionError("double or foreign cudaFree")
        self.free_calls += 1
        self.free_bytes += self.live.pop(pointer)

    def consume(self, size: int) -> None:
        if size > self.free_bytes:
            raise RuntimeError("simulated resident allocation OOM")
        self.free_bytes -= size


@dataclass
class EscrowProcess:
    rank: int
    cuda: FakeCuda
    enabled: bool = True
    state: str = "new"
    pointer: int | None = None
    logs: list[str] = field(default_factory=list)


class EscrowCohort:
    def __init__(self, processes: tuple[EscrowProcess, ...]) -> None:
        if len(processes) != WORLD_SIZE:
            raise ValueError("escrow proof requires four ranks")
        self.processes = processes
        self.vote_calls = 0
        self.min_calls = 0

    def _fatal_all(self, message: str) -> None:
        for process in self.processes:
            if (
                process.pointer is not None
                and process.pointer in process.cuda.live
            ):
                process.cuda.free(process.pointer)
                process.pointer = None
            process.state = "dead"
        raise FatalEscrowError(message)

    def arm(self) -> None:
        if not any(process.enabled for process in self.processes):
            assert all(process.state == "new" for process in self.processes)
            return
        if not all(process.enabled for process in self.processes):
            self._fatal_all("phase-2 escrow enablement differs across ranks")

        votes: list[int] = []
        for process in self.processes:
            if process.state != "new":
                self._fatal_all("escrow arm called out of order")
            free_before, _ = process.cuda.mem_get_info()
            try:
                process.pointer = process.cuda.malloc(ESCROW_BYTES)
                vote = 1
            except RuntimeError:
                process.pointer = None
                vote = 0
            free_after, _ = process.cuda.mem_get_info()
            process.logs.append(
                "CKV escrow arm "
                f"rank={process.rank} driver_free_before={free_before} "
                f"driver_free_after={free_after} bytes={ESCROW_BYTES} "
                f"local_ok={vote}"
            )
            votes.append(vote)

        self.vote_calls += 1
        if min(votes) != 1:
            for process in self.processes:
                if process.pointer is not None:
                    process.cuda.free(process.pointer)
                    process.pointer = None
            self._fatal_all("at least one rank failed the escrow allocation")
        for process in self.processes:
            process.state = "armed"

    def assert_held_for_first_ckv(self) -> None:
        for process in self.processes:
            assert process.state == "armed"
            assert process.pointer in process.cuda.live

    def complete_first_ckv_and_probe_a(self) -> int:
        self.assert_held_for_first_ckv()
        frees: list[int] = []
        for process in self.processes:
            assert process.pointer is not None
            process.cuda.free(process.pointer)
            process.pointer = None
            process.state = "released_a"
            free_bytes, _ = process.cuda.mem_get_info()
            frees.append(free_bytes)
        self.min_calls += 1
        group_min = min(frees)
        for process in self.processes:
            process.logs.append(
                "CKV escrow probe=A "
                f"rank={process.rank} group_min_free={group_min}"
            )
        if group_min < HARD_GATE_BYTES:
            self._fatal_all("Probe A is below the 150 MiB hard gate")
        return group_min

    def probe_b_at_next_layer(self) -> int:
        if not all(
            process.state == "released_a" for process in self.processes
        ):
            self._fatal_all("Probe B occurred before escrow release/Probe A")
        frees = tuple(
            process.cuda.mem_get_info()[0] for process in self.processes
        )
        self.min_calls += 1
        group_min = min(frees)
        for process in self.processes:
            process.logs.append(
                "CKV escrow probe=B "
                f"rank={process.rank} group_min_free={group_min}"
            )
            process.state = "probed_b"
        if group_min < HARD_GATE_BYTES:
            self._fatal_all("Probe B is below the 150 MiB hard gate")
        return group_min


def make_cohort(free_mib: tuple[int, int, int, int]) -> EscrowCohort:
    return EscrowCohort(
        tuple(
            EscrowProcess(rank, FakeCuda(free * MIB))
            for rank, free in enumerate(free_mib)
        )
    )


def main() -> None:
    disabled = EscrowCohort(
        tuple(
            EscrowProcess(rank, FakeCuda(300 * MIB), enabled=False)
            for rank in range(WORLD_SIZE)
        )
    )
    disabled.arm()
    assert disabled.vote_calls == 0
    assert all(process.cuda.malloc_calls == 0 for process in disabled.processes)
    assert all(not process.logs for process in disabled.processes)

    success = make_cohort((320, 315, 310, 305))
    success.arm()
    success.assert_held_for_first_ckv()
    # Request/context residents absorb most unescrowed space while the direct
    # allocation remains untouchable by the simulated caching allocator.
    for process, held_target_mib in zip(
        success.processes, (58, 54, 51, 49)
    ):
        process.cuda.consume(
            process.cuda.free_bytes - held_target_mib * MIB
        )
    probe_a = success.complete_first_ckv_and_probe_a()
    assert probe_a == 241 * MIB
    for process, persistent_mib in zip(success.processes, (8, 11, 9, 12)):
        process.cuda.consume(persistent_mib * MIB)
    probe_b = success.probe_b_at_next_layer()
    assert probe_b == 229 * MIB
    assert success.vote_calls == 1
    assert success.min_calls == 2
    assert all(process.cuda.malloc_calls == 1 for process in success.processes)
    assert all(process.cuda.free_calls == 1 for process in success.processes)
    assert all(not process.cuda.live for process in success.processes)
    assert all(process.state == "probed_b" for process in success.processes)
    assert all(
        sum("CKV escrow arm" in line for line in process.logs) == 1
        and sum("probe=A" in line for line in process.logs) == 1
        and sum("probe=B" in line for line in process.logs) == 1
        and "driver_free_before=" in process.logs[0]
        and "driver_free_after=" in process.logs[0]
        for process in success.processes
    )

    allocation_failure = make_cohort((300, 300, 191, 300))
    try:
        allocation_failure.arm()
    except FatalEscrowError:
        pass
    else:
        raise AssertionError("rank-local allocation failure did not kill cohort")
    assert all(
        process.state == "dead" for process in allocation_failure.processes
    )
    assert allocation_failure.vote_calls == 1
    assert allocation_failure.processes[2].cuda.free_calls == 0
    assert all(
        process.cuda.free_calls == 1
        for process in (
            allocation_failure.processes[0],
            allocation_failure.processes[1],
            allocation_failure.processes[3],
        )
    )

    low_probe = make_cohort((250, 250, 250, 250))
    low_probe.arm()
    assert low_probe.complete_first_ckv_and_probe_a() == 250 * MIB
    low_probe.processes[2].cuda.consume(101 * MIB)
    try:
        low_probe.probe_b_at_next_layer()
    except FatalEscrowError:
        pass
    else:
        raise AssertionError("sub-150 MiB Probe B did not kill cohort")
    assert all(process.state == "dead" for process in low_probe.processes)

    bad_order = make_cohort((300, 300, 300, 300))
    bad_order.arm()
    try:
        bad_order.probe_b_at_next_layer()
    except FatalEscrowError:
        pass
    else:
        raise AssertionError("Probe B before Probe A did not fail closed")
    assert all(process.cuda.free_calls == 1 for process in bad_order.processes)
    assert all(not process.cuda.live for process in bad_order.processes)

    print(
        "escrow success: "
        f"bytes={ESCROW_BYTES} probe_a_min={probe_a} "
        f"probe_b_min={probe_b} malloc/free=1/1"
    )
    print(
        "escrow failures: allocation_vote=fatal low_probe=fatal "
        "bad_order=fatal disabled_activity=zero arm_log=present"
    )
    print("PASS escrow is one-shot, observable, group-safe, and fail-closed")


if __name__ == "__main__":
    main()
