import os
import time
import statistics
import torch
import torch.distributed as dist
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        h = self.norm(x)
        h = torch.nn.functional.gelu(self.fc1(h))
        return x + self.fc2(h)


class ToyModel(nn.Module):
    def __init__(self, dim: int = 1024, layers: int = 12, expansion: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MLPBlock(dim, expansion) for _ in range(layers)]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return local_rank, rank, world_size


def sync_grads_naive(model: nn.Module, world_size: int):
    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def sync_grads_flat(model: nn.Module, world_size: int):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    flat = torch.cat([g.reshape(-1) for g in grads])

    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)

    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset:offset+n].view_as(g))
        offset += n


def make_batch(batch: int, seq: int, dim: int, rank: int):
    # 每个 rank 用不同随机数据，模拟数据并行。
    g = torch.Generator(device="cuda")
    g.manual_seed(1234 + rank)
    return torch.randn(
        batch, seq, dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=g,
    )


def run_sync_strategy(
    strategy: str,
    rank: int,
    world_size: int,
    *,
    dim: int,
    layers: int,
    batch: int,
    seq: int,
    warmup: int,
    steps: int,
):
    torch.manual_seed(0)

    model = ToyModel(dim=dim, layers=layers).cuda().to(torch.bfloat16)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = make_batch(batch, seq, dim, rank)

    param_count = sum(p.numel() for p in model.parameters())
    grad_mib = param_count * 2 / (1024**2)  # BF16 gradient bytes.

    def sync():
        if strategy == "naive":
            sync_grads_naive(model, world_size)
        elif strategy == "flat":
            sync_grads_flat(model, world_size)
        else:
            raise ValueError(strategy)

    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        y = model(x)
        loss = y.float().square().mean()
        loss.backward()
        sync()
        optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()

    step_times = []
    comm_times = []

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dist.barrier()

        t0 = time.perf_counter()

        y = model(x)
        loss = y.float().square().mean()
        loss.backward()

        torch.cuda.synchronize()
        c0 = time.perf_counter()
        sync()
        torch.cuda.synchronize()
        c1 = time.perf_counter()

        optimizer.step()
        torch.cuda.synchronize()

        step_elapsed = time.perf_counter() - t0
        comm_elapsed = c1 - c0

        # 用最慢 rank 的时间。
        t = torch.tensor([step_elapsed, comm_elapsed], device="cuda", dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)

        step_times.append(float(t[0].item()))
        comm_times.append(float(t[1].item()))

    return {
        "strategy": strategy,
        "world_size": world_size,
        "param_million": param_count / 1e6,
        "grad_mib": grad_mib,
        "step_ms": statistics.median(step_times) * 1000,
        "comm_ms": statistics.median(comm_times) * 1000,
    }


import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("profiles/ddp_sync_bench.csv"))
    args = parser.parse_args()

    _, rank, world_size = setup()

    rows = []
    for strategy in ["naive", "flat"]:
        row = run_sync_strategy(
            strategy,
            rank,
            world_size,
            dim=args.dim,
            layers=args.layers,
            batch=args.batch,
            seq=args.seq,
            warmup=args.warmup,
            steps=args.steps,
        )
        rows.append(row)

        if rank == 0:
            print(
                f"{strategy:>6} | "
                f"params={row['param_million']:.1f}M | "
                f"grad={row['grad_mib']:.1f} MiB | "
                f"step={row['step_ms']:.3f} ms | "
                f"comm={row['comm_ms']:.3f} ms"
            )

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved: {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
