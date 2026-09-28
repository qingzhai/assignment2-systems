# ============================================================
# torch.compile Attention Benchmark
#
# 和 pytorch_attention.py 使用完全相同的：
#
#   Attention implementation
#   input shapes
#   dtype
#   benchmark methodology
#
# 唯一的变化：
#
#       pytorch_attention
#
#              ↓
#
#       torch.compile(...)
#
# 这样我们才能公平比较：
#
#       eager PyTorch
#
#       vs
#
#       compiled PyTorch
# ============================================================


# argparse 解析命令行参数。
import argparse

# PyTorch。
import torch

# 从 vanilla benchmark 中复用所有公共逻辑。
from cs336_systems.pytorch_attention import (
    DEFAULT_HEAD_DIMS,
    DEFAULT_SEQ_LENS,
    benchmark_configuration,
    cleanup_cuda,
    parse_dtype,
    pytorch_attention,
    save_results,
)


# ============================================================
# 1. CLI
# ============================================================

def parse_args():
    """
    读取命令行参数。
    """

    # 创建 parser。
    parser = argparse.ArgumentParser()

    # Batch size。
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    # Sequence lengths。
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENS,
    )

    # Head dimensions。
    parser.add_argument(
        "--head-dims",
        type=int,
        nargs="+",
        default=DEFAULT_HEAD_DIMS,
    )

    # dtype。
    parser.add_argument(
        "--dtype",
        choices=[
            "float32",
            "bfloat16",
            "float16",
        ],
        default="float32",
    )

    # Warmup。
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
    )

    # 正式 measurement 数量。
    parser.add_argument(
        "--measurement-steps",
        type=int,
        default=100,
    )

    # CSV 输出位置。
    parser.add_argument(
        "--output",
        type=str,
        default="profiles/pytorch_attention_compile.csv",
    )

    # 返回。
    return parser.parse_args()


# ============================================================
# 2. main
# ============================================================

def main():
    """
    对显式 PyTorch Attention 使用 torch.compile，
    然后跑与 eager 完全相同的 benchmark。
    """

    # 解析命令行。
    args = parse_args()

    # 必须有 CUDA。
    if not torch.cuda.is_available():

        # 没有 GPU 直接退出。
        raise RuntimeError(
            "CUDA GPU is required."
        )

    # CUDA device。
    device = torch.device(
        "cuda"
    )

    # dtype。
    dtype = parse_dtype(
        args.dtype
    )

    # 打印 GPU。
    print(
        f"GPU: {torch.cuda.get_device_name()}"
    )

    # 打印 dtype。
    print(
        f"dtype: {dtype}"
    )

    # 打印 FP32 matmul precision。
    print(
        "float32 matmul precision:",
        torch.get_float32_matmul_precision(),
    )

    # 保存所有结果。
    results = []

    # ========================================================
    # Cartesian product
    # ========================================================

    for seq_len in args.seq_lens:

        for head_dim in args.head_dims:

            # 当前 configuration。
            print()
            print(
                "=" * 70
            )

            print(
                f"B={args.batch_size}, "
                f"S={seq_len}, "
                f"D={head_dim}"
            )

            # ------------------------------------------------
            # 每一个新的 shape 都创建一个新的 compiled fn。
            #
            # dynamic=False：
            #
            #   让编译器针对当前静态 shape 优化。
            #
            # fullgraph=True：
            #
            #   尽量让整个 Attention graph 一次编译，
            #   而不是发生 graph break。
            # ------------------------------------------------

            compiled_attention = torch.compile(
                pytorch_attention,
                dynamic=False,
                fullgraph=True,
            )

            # ------------------------------------------------
            # Benchmark
            # ------------------------------------------------

            try:

                # 这里 measure_first_call=True。
                #
                # 因为 compiled function 第一次执行时
                # 会包含：
                #
                # graph capture
                # compiler optimization
                # Triton/codegen
                # kernel compilation
                #
                # 所以我们单独记录 cold-start。
                result = benchmark_configuration(
                    attention_fn=compiled_attention,
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                    warmup_steps=args.warmup_steps,
                    measurement_steps=args.measurement_steps,
                    measure_first_call=True,
                )

                # 保存结果。
                results.append(
                    result
                )

                # 打印第一次调用。
                print(
                    f"first compiled call: "
                    f"{result['first_call_ms']:.3f} ms"
                )

                # Steady-state Forward。
                print(
                    f"forward steady-state:  "
                    f"{result['forward_mean_ms']:.3f} ms"
                )

                # Steady-state Backward。
                print(
                    f"backward steady-state: "
                    f"{result['backward_mean_ms']:.3f} ms"
                )

                # Forward 后显存。
                print(
                    f"before backward memory: "
                    f"{result['memory_before_backward_gb']:.3f} GB"
                )

                # Saved residual memory。
                print(
                    f"saved for backward: "
                    f"{result['saved_for_backward_gb']:.3f} GB"
                )

                # Peak。
                print(
                    f"peak memory: "
                    f"{result['peak_memory_gb']:.3f} GB"
                )

            # ------------------------------------------------
            # OOM
            # ------------------------------------------------

            except torch.OutOfMemoryError:

                # OOM row。
                result = {
                    "batch_size": args.batch_size,
                    "seq_len": seq_len,
                    "head_dim": head_dim,
                    "dtype": args.dtype,
                    "status": "OOM",
                    "first_call_ms": "",
                    "forward_mean_ms": "",
                    "forward_std_ms": "",
                    "backward_mean_ms": "",
                    "backward_std_ms": "",
                    "baseline_memory_gb": "",
                    "memory_before_backward_gb": "",
                    "saved_for_backward_gb": "",
                    "peak_memory_gb": "",
                }

                # 保存。
                results.append(
                    result
                )

                # 打印。
                print("OOM")

                # 清理 CUDA。
                cleanup_cuda()

            # ------------------------------------------------
            # 清理 compiler state
            # ------------------------------------------------

            finally:

                # 删除当前 compiled callable。
                del compiled_attention

                # 清除 Dynamo graph cache。
                #
                # 这样不同 shape 的实验更加独立。
                torch._dynamo.reset()

                # 清理 CUDA allocator。
                cleanup_cuda()

            # 每个 config 完成后立即保存。
            save_results(
                results,
                args.output,
            )

    # ========================================================
    # 完成
    # ========================================================

    print()
    print(
        f"Results saved to: {args.output}"
    )


# Python entry point。
if __name__ == "__main__":
    main()