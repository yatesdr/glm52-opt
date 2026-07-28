#!/usr/bin/env python3
"""Four-GPU eager/graph gate for the block-INT8 all-gather candidate."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_input(
    rank: int, device: torch.device, rows: int, iteration: int = 0
) -> torch.Tensor:
    hidden = 6144
    values = torch.arange(rows * hidden, device=device, dtype=torch.float32)
    values = values.reshape(rows, hidden)
    return (
        torch.sin(values * 0.001 + rank * 0.31 + iteration * 0.17) * 0.5
    ).to(torch.bfloat16)


def assert_output(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    rank: int,
    world: int,
    label: str,
) -> None:
    rank0_output = output.clone() if rank == 0 else torch.empty_like(output)
    dist.broadcast(rank0_output, src=0)
    if not torch.equal(output, rank0_output):
        mismatch = (output.float() - rank0_output.float()).abs()
        raise AssertionError(
            f"{label} rank={rank} differs from rank0: "
            f"count={int(torch.count_nonzero(mismatch).item())} "
            f"max_abs={float(mismatch.max().item())}"
        )

    error = (output.float() - reference).abs()
    max_abs = float(error.max().item())
    rmse = float(torch.sqrt(torch.mean(error.square())).item())
    if max_abs > 0.04 * world or rmse > 0.012 * world:
        raise AssertionError(
            f"{label} INT8 error band failed: max_abs={max_abs} rmse={rmse}"
        )
    if rank == 0:
        print(f"PASS {label} rank_equal=1 max_abs={max_abs:.8f} rmse={rmse:.8f}")


def worker(rank: int, world: int, port: int) -> None:
    os.environ["B12X_PCIE_DMA_FP8"] = "i8"
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    from b12x.distributed.pcie_dma import PCIeDmaAllReduce

    ring = PCIeDmaAllReduce(
        exchange_group=dist.group.WORLD,
        device=device,
        max_bytes=3072 * 6144 * 2,
        fp8="i8",
    )
    try:
        for rows in (512, 3072):
            inp = make_input(rank, device, rows)
            reference = inp.float()
            dist.all_reduce(reference)
            output = ring.all_reduce(inp)
            torch.cuda.synchronize(device)
            assert_output(
                output, reference, rank=rank, world=world, label=f"rows={rows}-eager"
            )

        rows = 512
        graph_input = make_input(rank, device, rows, 1)
        graph_output = torch.empty_like(graph_input)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ring.all_reduce(graph_input, out=graph_output)
        for iteration in range(2, 5):
            graph_input.copy_(make_input(rank, device, rows, iteration))
            reference = graph_input.float()
            dist.all_reduce(reference)
            graph.replay()
            torch.cuda.synchronize(device)
            assert_output(
                graph_output,
                reference,
                rank=rank,
                world=world,
                label=f"rows={rows}-graph-{iteration}",
            )
        dist.barrier()
    finally:
        ring.close()
        dist.destroy_process_group()


def main() -> None:
    world = 4
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        raise RuntimeError("this gate requires four visible CUDA GPUs")
    mp.spawn(worker, args=(world, free_port()), nprocs=world, join=True)


if __name__ == "__main__":
    main()
