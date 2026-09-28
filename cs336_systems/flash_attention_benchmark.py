"""
Benchmark the learning Triton FlashAttention implementation.

This script measures:
1. Forward latency
2. Backward latency (forward is rebuilt outside the timed region)
3. CUDA allocated-memory behavior around forward/backward

Default settings:
    B = 8
    S = [256, 1024, 4096, 8192, 16384]
    D = [16, 32, 64, 128]
    dtype = bfloat16
    causal = True
    warmup = 5
    measurements = 100
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
from pathlib import Path

import torch

from cs336_systems.flash_attention_triton import flash_attention_triton


GB = 1024**3


def bytes_to_gb(num_bytes: int) -> float:
    """Convert bytes to GiB."""
    return num_bytes / GB


def benchmark_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    warmup: int,
    measurements: int,
) -> float:
    """
    Benchmark training-style forward latency.

    q/k/v require gradients, so the custom autograd Function still creates
    the normal training forward context.
    """
    for _ in range(warmup):
        out = flash_attention_triton(
            q,
            k,
            v,
            is_causal=is_causal,
        )
        del out

    torch.cuda.synchronize()

    times_ms: list[float] = []

    for _ in range(measurements):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        out = flash_attention_triton(
            q,
            k,
            v,
            is_causal=is_causal,
        )

        end.record()
        end.synchronize()

        times_ms.append(start.elapsed_time(end))

        del out

    return statistics.median(times_ms)


def benchmark_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    *,
    is_causal: bool,
    warmup: int,
    measurements: int,
) -> float:
    """
    Benchmark pure backward latency.

    Each backward requires a fresh graph, so forward is rebuilt before the
    timer starts. The reported backward_ms therefore excludes forward.
    """
    for _ in range(warmup):
        out = flash_attention_triton(
            q,
            k,
            v,
            is_causal=is_causal,
        )

        grads = torch.autograd.grad(
            outputs=out,
            inputs=(q, k, v),
            grad_outputs=dout,
            retain_graph=False,
            create_graph=False,
        )

        del grads
        del out

    torch.cuda.synchronize()

    times_ms: list[float] = []

    for _ in range(measurements):
        # Rebuild the graph outside the timed region.
        out = flash_attention_triton(
            q,
            k,
            v,
            is_causal=is_causal,
        )

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        grads = torch.autograd.grad(
            outputs=out,
            inputs=(q, k, v),
            grad_outputs=dout,
            retain_graph=False,
            create_graph=False,
        )

        end.record()
        end.synchronize()

        times_ms.append(start.elapsed_time(end))

        del grads
        del out

    return statistics.median(times_ms)


def probe_memory(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    *,
    is_causal: bool,
) -> dict[str, float]:
    """
    Measure one forward+backward pass.

    baseline_gb:
        Memory after q/k/v/dout already exist.

    before_backward_gb:
        Memory after forward, before backward.

    saved_for_backward_gb:
        before_backward - baseline.
        This is an empirical proxy for extra live CUDA tensor state left by
        forward. For FlashAttention it should scale roughly linearly with S.

    peak_gb:
        Peak total allocated CUDA memory during this probe.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    baseline_bytes = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()

    out = flash_attention_triton(
        q,
        k,
        v,
        is_causal=is_causal,
    )
    torch.cuda.synchronize()

    before_backward_bytes = torch.cuda.memory_allocated()

    grads = torch.autograd.grad(
        outputs=out,
        inputs=(q, k, v),
        grad_outputs=dout,
        retain_graph=False,
        create_graph=False,
    )
    torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated()

    del grads
    del out

    return {
        "baseline_gb": bytes_to_gb(baseline_bytes),
        "before_backward_gb": bytes_to_gb(before_backward_bytes),
        "saved_for_backward_gb": bytes_to_gb(
            max(0, before_backward_bytes - baseline_bytes)
        ),
        "peak_gb": bytes_to_gb(peak_bytes),
        "peak_over_baseline_gb": bytes_to_gb(
            max(0, peak_bytes - baseline_bytes)
        ),
    }


def run_one_configuration(
    *,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    warmup: int,
    measurements: int,
) -> dict[str, object]:
    """Benchmark one (B, S, D) configuration."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    q = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    k = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    v = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    dout = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        device="cuda",
        dtype=dtype,
    )

    memory = probe_memory(
        q,
        k,
        v,
        dout,
        is_causal=is_causal,
    )

    forward_ms = benchmark_forward(
        q,
        k,
        v,
        is_causal=is_causal,
        warmup=warmup,
        measurements=measurements,
    )

    backward_ms = benchmark_backward(
        q,
        k,
        v,
        dout,
        is_causal=is_causal,
        warmup=warmup,
        measurements=measurements,
    )

    row: dict[str, object] = {
        "implementation": "triton_flash_attention",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "causal": is_causal,
        "warmup": warmup,
        "measurements": measurements,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        **memory,
        "status": "ok",
        "error": "",
    }

    del q, k, v, dout

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return row


def print_result(row: dict[str, object]) -> None:
    """Print one benchmark row."""
    if row["status"] != "ok":
        print(
            f"B={row['batch_size']} "
            f"S={row['seq_len']} "
            f"D={row['head_dim']} "
            f"FAILED: {row['error']}"
        )
        return

    print(
        f"B={row['batch_size']:>2} "
        f"S={row['seq_len']:>5} "
        f"D={row['head_dim']:>3} | "
        f"fwd={row['forward_ms']:>9.3f} ms | "
        f"bwd={row['backward_ms']:>9.3f} ms | "
        f"before_bwd={row['before_backward_gb']:>7.3f} GB | "
        f"saved={row['saved_for_backward_gb']:>7.3f} GB | "
        f"peak={row['peak_gb']:>7.3f} GB"
    )


def save_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Save benchmark rows to CSV."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "implementation",
        "batch_size",
        "seq_len",
        "head_dim",
        "dtype",
        "causal",
        "warmup",
        "measurements",
        "forward_ms",
        "backward_ms",
        "baseline_gb",
        "before_backward_gb",
        "saved_for_backward_gb",
        "peak_gb",
        "peak_over_baseline_gb",
        "status",
        "error",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[
            256,
            1024,
            4096,
            8192,
            16384,
        ],
    )

    parser.add_argument(
        "--head-dims",
        type=int,
        nargs="+",
        default=[
            16,
            32,
            64,
            128,
        ],
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--measurements",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--non-causal",
        action="store_true",
        help="Benchmark non-causal attention. Default is causal.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "profiles/flash_attention_triton_full.csv"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    dtype = torch.bfloat16
    is_causal = not args.non_causal

    print("=" * 90)
    print("Triton FlashAttention benchmark")
    print("=" * 90)
    print(f"GPU:          {torch.cuda.get_device_name()}")
    print(f"torch:        {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"B:            {args.batch_size}")
    print(f"S:            {args.seq_lens}")
    print(f"D:            {args.head_dims}")
    print(f"dtype:        {dtype}")
    print(f"causal:       {is_causal}")
    print(f"warmup:       {args.warmup}")
    print(f"measurements: {args.measurements}")
    print(f"output:       {args.output}")
    print("=" * 90)

    rows: list[dict[str, object]] = []

    for seq_len in args.seq_lens:
        for head_dim in args.head_dims:
            print()
            print(
                f"Running B={args.batch_size}, "
                f"S={seq_len}, "
                f"D={head_dim} ..."
            )

            try:
                row = run_one_configuration(
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    dtype=dtype,
                    is_causal=is_causal,
                    warmup=args.warmup,
                    measurements=args.measurements,
                )

            except torch.cuda.OutOfMemoryError as exc:
                row = {
                    "implementation": "triton_flash_attention",
                    "batch_size": args.batch_size,
                    "seq_len": seq_len,
                    "head_dim": head_dim,
                    "dtype": "bfloat16",
                    "causal": is_causal,
                    "warmup": args.warmup,
                    "measurements": args.measurements,
                    "forward_ms": "",
                    "backward_ms": "",
                    "baseline_gb": "",
                    "before_backward_gb": "",
                    "saved_for_backward_gb": "",
                    "peak_gb": "",
                    "peak_over_baseline_gb": "",
                    "status": "oom",
                    "error": str(exc).replace("\n", " "),
                }

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            except Exception as exc:
                row = {
                    "implementation": "triton_flash_attention",
                    "batch_size": args.batch_size,
                    "seq_len": seq_len,
                    "head_dim": head_dim,
                    "dtype": "bfloat16",
                    "causal": is_causal,
                    "warmup": args.warmup,
                    "measurements": args.measurements,
                    "forward_ms": "",
                    "backward_ms": "",
                    "baseline_gb": "",
                    "before_backward_gb": "",
                    "saved_for_backward_gb": "",
                    "peak_gb": "",
                    "peak_over_baseline_gb": "",
                    "status": "error",
                    "error": repr(exc),
                }

                rows.append(row)
                save_csv(
                    rows,
                    args.output,
                )
                print_result(row)
                print(
                    f"\nPartial results saved to: {args.output}"
                )
                raise

            rows.append(row)

            # 每完成一个配置就保存一次，长实验被打断也不会全丢。
            save_csv(
                rows,
                args.output,
            )

            print_result(row)

    print()
    print("=" * 90)
    print("Benchmark finished.")
    print(f"Results saved to: {args.output}")
    print("=" * 90)


if __name__ == "__main__":
    main()
