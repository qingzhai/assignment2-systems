import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size


def cleanup():
    dist.destroy_process_group()


def one_size(size_mib: int, warmup: int, reps: int, rank: int, world_size: int):
    # 用 float32，方便把 MiB 精确换成元素个数。
    numel = size_mib * 1024 * 1024 // 4
    x = torch.ones(numel, device="cuda", dtype=torch.float32)

    for _ in range(warmup):
        work = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
    torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        work = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # 用所有 rank 中最慢的那个时间，避免只看 rank0 偏乐观。
        t = torch.tensor([elapsed], device="cuda", dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        times.append(float(t.item()))

    median_s = statistics.median(times)
    bytes_count = size_mib * 1024 * 1024

    # 算法带宽：payload / latency。
    alg_bw_gbps = bytes_count / median_s / 1e9

    # 对 ring all-reduce，常用 bus bandwidth 近似：
    # alg_bw * 2*(N-1)/N
    bus_bw_gbps = alg_bw_gbps * 2 * (world_size - 1) / world_size

    return {
        "world_size": world_size,
        "size_mib": size_mib,
        "median_ms": median_s * 1000,
        "alg_bw_GBps": alg_bw_gbps,
        "bus_bw_GBps": bus_bw_gbps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("profiles/allreduce_bench.csv"))
    args = parser.parse_args()

    _, rank, world_size = setup()

    rows = []
    if rank == 0:
        print(f"world_size={world_size}")
        print(f"GPU={torch.cuda.get_device_name()}")

    for size in args.sizes_mib:
        row = one_size(size, args.warmup, args.reps, rank, world_size)
        rows.append(row)

        if rank == 0:
            print(
                f"{size:>5} MiB | "
                f"{row['median_ms']:>8.3f} ms | "
                f"alg_bw={row['alg_bw_GBps']:>7.2f} GB/s | "
                f"bus_bw={row['bus_bw_GBps']:>7.2f} GB/s"
            )

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved: {args.output}")

    cleanup()


if __name__ == "__main__":
    main()
