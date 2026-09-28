# ============================================================
# PyTorch Attention Benchmark
#
# 作用：
#   1. 使用显式 PyTorch 算子实现普通 causal attention
#   2. Benchmark Forward
#   3. Benchmark Backward
#   4. 测量 backward 开始之前的显存
#   5. 对不同 sequence length / head dimension 做 sweep
#
# Attention:
#
#                Q K^T
#   P = softmax( ----- + causal mask )
#                sqrt(d)
#
#   O = P V
#
# 注意：
#   这里故意不使用：
#
#       torch.nn.functional.scaled_dot_product_attention
#
# 因为它可能自动派发到 fused / Flash Attention kernel，
# 就不能作为 vanilla PyTorch Attention baseline 了。
# ============================================================


# argparse 用来解析命令行参数。
import argparse

# csv 用来把 benchmark 结果保存成 CSV。
import csv

# gc 用来在每组实验结束后主动触发 Python 垃圾回收。
import gc

# math 用来计算 sqrt(d)。
import math

# statistics 用来计算 mean / std。
import statistics

# time 用来做 wall-clock timing。
import time

# Path 用来创建输出目录。
from pathlib import Path

# PyTorch。
import torch


# ============================================================
# 1. Assignment 默认 sweep
# ============================================================

# Assignment 要测试的 sequence lengths。
DEFAULT_SEQ_LENS = [
    256,
    1024,
    4096,
    8192,
    16384,
]

# Assignment 要测试的 embedding/head dimensions。
DEFAULT_HEAD_DIMS = [
    16,
    32,
    64,
    128,
]


# ============================================================
# 2. dtype helper
# ============================================================

def parse_dtype(dtype_name: str) -> torch.dtype:
    """
    将命令行中的 dtype 字符串转换成 torch.dtype。

    支持：

        float32
        bfloat16
        float16

    默认使用 float32。

    如果你希望继续使用我们前面 benchmark 的 BF16，
    可以运行：

        --dtype bfloat16
    """

    # 字符串 -> torch dtype 的映射。
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    # 返回对应 dtype。
    return mapping[dtype_name]


# ============================================================
# 3. Explicit PyTorch Attention
# ============================================================

def pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: torch.Tensor,
) -> torch.Tensor:
    """
    使用普通 PyTorch Tensor operations 实现 causal attention。

    输入 shape：

        q: [B, S, D]
        k: [B, S, D]
        v: [B, S, D]

    注意：

        没有 multi-head dimension。

    输出：

        output: [B, S, D]

    ----------------------------------------------------------

    实现：

        scores = Q K^T / sqrt(D)

        scores = causal_mask(scores)

        probabilities = softmax(scores)

        output = probabilities V

    ----------------------------------------------------------

    这里最重要的是：

        scores shape = [B, S, S]

    因此普通 Attention 会显式 materialize：

        O(B * S^2)

    大小的 attention matrix。

    当 S 很大时，
    这就是主要 memory bottleneck。
    """

    # 获取 head / embedding dimension D。
    d = q.shape[-1]

    # --------------------------------------------------------
    # Q K^T
    # --------------------------------------------------------

    # k.transpose(-2, -1):
    #
    # [B, S, D]
    #       ↓
    # [B, D, S]
    #
    # matmul 后：
    #
    # [B, S, D] @ [B, D, S]
    #
    #       ↓
    #
    # [B, S, S]
    scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    # Scaled Dot-Product Attention：
    #
    #               Q K^T
    # scores = ----------------
    #              sqrt(D)
    scores = scores * (
        1.0 / math.sqrt(d)
    )

    # --------------------------------------------------------
    # Causal Mask
    # --------------------------------------------------------

    # causal_mask 中：
    #
    # True  = future token，需要屏蔽
    # False = 可以访问
    #
    # 将 future positions 设置为 -inf。
    scores = scores.masked_fill(
        causal_mask,
        float("-inf"),
    )

    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    # 沿最后一个 key dimension 做 softmax。
    #
    # probabilities shape:
    #
    # [B, S, S]
    probabilities = torch.softmax(
        scores,
        dim=-1,
    )

    # --------------------------------------------------------
    # P V
    # --------------------------------------------------------

    # [B, S, S] @ [B, S, D]
    #
    #       ↓
    #
    # [B, S, D]
    output = torch.matmul(
        probabilities,
        v,
    )

    # 返回 Attention output。
    return output


# ============================================================
# 4. 创建输入
# ============================================================

def create_inputs(
    batch_size: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    创建 Attention 所需要的随机 Q / K / V。

    所有 Tensor：

        shape = [B, S, D]

    requires_grad=True：

        因为后面需要 benchmark backward。
    """

    # Q。
    q = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    # K。
    k = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    # V。
    v = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    # --------------------------------------------------------
    # Causal mask
    # --------------------------------------------------------

    # 创建：
    #
    # [S, S]
    #
    # 的上三角 mask。
    #
    # diagonal=1：
    #
    # 对角线本身允许访问，
    # 只屏蔽未来位置。
    causal_mask = torch.ones(
        seq_len,
        seq_len,
        device=device,
        dtype=torch.bool,
    ).triu(
        diagonal=1,
    )

    # output 与 q shape 一致，
    # 所以可以提前准备固定 grad_output。
    grad_output = torch.randn(
        batch_size,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )

    # 返回全部 Tensor。
    return q, k, v, causal_mask, grad_output


# ============================================================
# 5. 清除 gradients
# ============================================================

def clear_gradients(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    """
    删除上一轮 backward 产生的 gradients。

    使用 None 而不是 zero_()：

        不需要额外保留全零 gradient tensor。
    """

    # 删除 q gradient。
    q.grad = None

    # 删除 k gradient。
    k.grad = None

    # 删除 v gradient。
    v.grad = None


# ============================================================
# 6. Forward benchmark
# ============================================================

def benchmark_forward(
    attention_fn,
    q,
    k,
    v,
    causal_mask,
    warmup_steps: int,
    measurement_steps: int,
):
    """
    Benchmark Forward latency。

    注意：

        CUDA 是异步执行。

    所以每次 timing 都：

        synchronize
        ↓
        start timer
        ↓
        forward
        ↓
        synchronize
        ↓
        stop timer
    """

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(warmup_steps):

        # Forward。
        output = attention_fn(
            q,
            k,
            v,
            causal_mask,
        )

        # 等 GPU 真正执行完成。
        torch.cuda.synchronize()

        # 删除 output，
        # 这样对应 computation graph 可以被释放。
        del output

    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    # 保存所有 latency。
    timings_ms = []

    # 正式 benchmark。
    for _ in range(measurement_steps):

        # 确保之前 GPU queue 清空。
        torch.cuda.synchronize()

        # 开始 timing。
        start = time.perf_counter()

        # Forward。
        output = attention_fn(
            q,
            k,
            v,
            causal_mask,
        )

        # 等 forward 真正完成。
        torch.cuda.synchronize()

        # 结束 timing。
        end = time.perf_counter()

        # 转换成 ms。
        timings_ms.append(
            (end - start) * 1000
        )

        # 删除 computation graph。
        del output

    # 计算 mean。
    mean_ms = statistics.mean(
        timings_ms
    )

    # 计算 std。
    std_ms = (
        statistics.stdev(timings_ms)
        if len(timings_ms) > 1
        else 0.0
    )

    # 返回结果。
    return mean_ms, std_ms


# ============================================================
# 7. Backward benchmark
# ============================================================

def benchmark_backward(
    attention_fn,
    q,
    k,
    v,
    causal_mask,
    grad_output,
    warmup_steps: int,
    measurement_steps: int,
):
    """
    Benchmark Backward latency。

    Forward 不计入 backward timing。

    每一次：

        forward
          ↓
        synchronize

        start timer
          ↓
        backward
          ↓
        synchronize
          ↓
        stop timer

    这样我们测到的是纯 backward 时间。
    """

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(warmup_steps):

        # 清除上一轮 gradients。
        clear_gradients(
            q,
            k,
            v,
        )

        # 创建 fresh computation graph。
        output = attention_fn(
            q,
            k,
            v,
            causal_mask,
        )

        # Forward 完成。
        torch.cuda.synchronize()

        # Backward。
        output.backward(
            grad_output
        )

        # Backward 完成。
        torch.cuda.synchronize()

        # 删除 graph。
        del output

    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    # 保存每一次 backward latency。
    timings_ms = []

    # 正式测量。
    for _ in range(measurement_steps):

        # 删除上一轮 gradients。
        clear_gradients(
            q,
            k,
            v,
        )

        # 先做 Forward，
        # 为当前 backward 创建新的 computation graph。
        output = attention_fn(
            q,
            k,
            v,
            causal_mask,
        )

        # Forward 不计入 backward timing，
        # 因此这里先等待它完成。
        torch.cuda.synchronize()

        # 开始 timing。
        start = time.perf_counter()

        # Backward。
        output.backward(
            grad_output
        )

        # 等 backward 真正完成。
        torch.cuda.synchronize()

        # 结束 timing。
        end = time.perf_counter()

        # 秒 -> ms。
        timings_ms.append(
            (end - start) * 1000
        )

        # 删除 graph。
        del output

    # 平均值。
    mean_ms = statistics.mean(
        timings_ms
    )

    # 标准差。
    std_ms = (
        statistics.stdev(timings_ms)
        if len(timings_ms) > 1
        else 0.0
    )

    # 返回。
    return mean_ms, std_ms


# ============================================================
# 8. Memory probe
# ============================================================

def measure_memory(
    attention_fn,
    q,
    k,
    v,
    causal_mask,
    grad_output,
):
    """
    单独执行一次 Forward + Backward，
    用于观察 Attention 的 memory。

    重点：

        baseline_memory

            Q / K / V / mask 等输入已经存在时的显存。

        before_backward_memory

            Forward 已经结束，
            Autograd residuals 仍然活着时的显存。

        saved_for_backward

            before_backward - baseline

        peak_memory

            整个 Forward + Backward 期间峰值显存。
    """

    # 删除旧 gradient。
    clear_gradients(
        q,
        k,
        v,
    )

    # 清理 allocator 中未使用的 cache。
    torch.cuda.empty_cache()

    # GPU 同步。
    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    # 当前 active memory。
    baseline_memory = (
        torch.cuda.memory_allocated()
    )

    # 从这里重新统计 peak。
    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    # Forward。
    output = attention_fn(
        q,
        k,
        v,
        causal_mask,
    )

    # 等 GPU 完成。
    torch.cuda.synchronize()

    # 此时就是：
    #
    # backward 即将开始之前。
    before_backward_memory = (
        torch.cuda.memory_allocated()
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    # Backward。
    output.backward(
        grad_output
    )

    # 等 GPU 完成。
    torch.cuda.synchronize()

    # 整个 step 的 peak。
    peak_memory = (
        torch.cuda.max_memory_allocated()
    )

    # --------------------------------------------------------
    # Saved-for-backward delta
    # --------------------------------------------------------

    # 粗略表示 Forward 为 backward
    # 新增保留下来的 active memory。
    saved_for_backward = (
        before_backward_memory
        - baseline_memory
    )

    # 删除 output。
    del output

    # 清除 gradients。
    clear_gradients(
        q,
        k,
        v,
    )

    # 返回 memory 数据。
    return (
        baseline_memory,
        before_backward_memory,
        saved_for_backward,
        peak_memory,
    )


# ============================================================
# 9. 单个 configuration benchmark
# ============================================================

def benchmark_configuration(
    attention_fn,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup_steps: int,
    measurement_steps: int,
    measure_first_call: bool = False,
):
    """
    对一个：

        B
        S
        D

    configuration 做完整 benchmark。

    measure_first_call：

        False:
            普通 PyTorch eager。

        True:
            用于 torch.compile，
            额外记录第一次调用时间，
            方便观察 compilation cold start。
    """

    # 创建输入。
    q, k, v, causal_mask, grad_output = create_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    # 默认没有 cold-start 数据。
    first_call_ms = None

    # --------------------------------------------------------
    # Optional first call
    # --------------------------------------------------------

    if measure_first_call:

        # 等 GPU。
        torch.cuda.synchronize()

        # 开始。
        start = time.perf_counter()

        # 第一次调用 compiled function，
        # 通常会触发 graph capture + compile + codegen。
        output = attention_fn(
            q,
            k,
            v,
            causal_mask,
        )

        # 等第一次执行真正完成。
        torch.cuda.synchronize()

        # 结束。
        end = time.perf_counter()

        # 转成 ms。
        first_call_ms = (
            end - start
        ) * 1000

        # 删除 output。
        del output

    # --------------------------------------------------------
    # Forward timing
    # --------------------------------------------------------

    forward_mean_ms, forward_std_ms = benchmark_forward(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        causal_mask=causal_mask,
        warmup_steps=warmup_steps,
        measurement_steps=measurement_steps,
    )

    # --------------------------------------------------------
    # Backward timing
    # --------------------------------------------------------

    backward_mean_ms, backward_std_ms = benchmark_backward(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        causal_mask=causal_mask,
        grad_output=grad_output,
        warmup_steps=warmup_steps,
        measurement_steps=measurement_steps,
    )

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    (
        baseline_memory,
        before_backward_memory,
        saved_for_backward,
        peak_memory,
    ) = measure_memory(
        attention_fn=attention_fn,
        q=q,
        k=k,
        v=v,
        causal_mask=causal_mask,
        grad_output=grad_output,
    )

    # GiB conversion。
    gib = 1024 ** 3

    # 形成一行结果。
    result = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "status": "OK",
        "first_call_ms": (
            first_call_ms
            if first_call_ms is not None
            else ""
        ),
        "forward_mean_ms": forward_mean_ms,
        "forward_std_ms": forward_std_ms,
        "backward_mean_ms": backward_mean_ms,
        "backward_std_ms": backward_std_ms,
        "baseline_memory_gb": (
            baseline_memory / gib
        ),
        "memory_before_backward_gb": (
            before_backward_memory / gib
        ),
        "saved_for_backward_gb": (
            saved_for_backward / gib
        ),
        "peak_memory_gb": (
            peak_memory / gib
        ),
    }

    # 删除 Tensor references。
    del q
    del k
    del v
    del causal_mask
    del grad_output

    # Python GC。
    gc.collect()

    # 清理 GPU cache。
    torch.cuda.empty_cache()

    # 返回。
    return result


# ============================================================
# 10. 保存 CSV
# ============================================================

def save_results(
    results: list[dict],
    output_file: str,
):
    """
    将 benchmark rows 保存为 CSV。
    """

    # 转换成 Path。
    output_path = Path(
        output_file
    )

    # 创建父目录。
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 如果没有结果就不用写。
    if not results:
        return

    # 打开 CSV。
    with output_path.open(
        "w",
        newline="",
    ) as f:

        # 使用第一行的 keys 作为 columns。
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        # 写 header。
        writer.writeheader()

        # 写数据。
        writer.writerows(
            results
        )


# ============================================================
# 11. Cleanup helper
# ============================================================

def cleanup_cuda():
    """
    OOM 或 configuration 完成之后清理 CUDA。
    """

    # Python garbage collection。
    gc.collect()

    # 清理 PyTorch caching allocator。
    torch.cuda.empty_cache()

    # 等 GPU。
    torch.cuda.synchronize()


# ============================================================
# 12. CLI
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

    # Assignment 正式要求 100 次 measurement。
    parser.add_argument(
        "--measurement-steps",
        type=int,
        default=100,
    )

    # CSV 输出。
    parser.add_argument(
        "--output",
        type=str,
        default="profiles/pytorch_attention.csv",
    )

    # 返回 args。
    return parser.parse_args()


# ============================================================
# 13. main
# ============================================================

def main():
    """
    执行整个 Attention benchmark sweep。
    """

    # 解析 CLI。
    args = parse_args()

    # 本实验需要 CUDA。
    if not torch.cuda.is_available():
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

    # 打印环境。
    print(
        f"GPU: {torch.cuda.get_device_name()}"
    )

    # 打印 dtype。
    print(
        f"dtype: {dtype}"
    )

    # FP32 情况下顺便记录 matmul precision 配置。
    print(
        "float32 matmul precision:",
        torch.get_float32_matmul_precision(),
    )

    # 保存所有 rows。
    results = []

    # --------------------------------------------------------
    # Cartesian product
    # --------------------------------------------------------

    for seq_len in args.seq_lens:

        for head_dim in args.head_dims:

            # 当前 config。
            print()
            print(
                "=" * 70
            )

            print(
                f"B={args.batch_size}, "
                f"S={seq_len}, "
                f"D={head_dim}"
            )

            # -----------------------------------------------
            # 尝试运行。
            # -----------------------------------------------

            try:

                # Benchmark。
                result = benchmark_configuration(
                    attention_fn=pytorch_attention,
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                    warmup_steps=args.warmup_steps,
                    measurement_steps=args.measurement_steps,
                    measure_first_call=False,
                )

                # 保存。
                results.append(
                    result
                )

                # 打印。
                print(
                    f"forward:  "
                    f"{result['forward_mean_ms']:.3f} ms"
                )

                print(
                    f"backward: "
                    f"{result['backward_mean_ms']:.3f} ms"
                )

                print(
                    f"before backward memory: "
                    f"{result['memory_before_backward_gb']:.3f} GB"
                )

                print(
                    f"saved for backward: "
                    f"{result['saved_for_backward_gb']:.3f} GB"
                )

                print(
                    f"peak memory: "
                    f"{result['peak_memory_gb']:.3f} GB"
                )

            # -----------------------------------------------
            # OOM handling
            # -----------------------------------------------

            except torch.OutOfMemoryError:

                # 构造 OOM row。
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

                # 保存 OOM row。
                results.append(
                    result
                )

                # 打印。
                print("OOM")

                # 清理。
                cleanup_cuda()

            # 每一个 configuration 完成后，
            # 都立刻保存一次 CSV。
            #
            # 即使后面程序意外中断，
            # 前面结果也不会丢。
            save_results(
                results,
                args.output,
            )

    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    print()
    print(
        f"Results saved to: {args.output}"
    )


# Python entry point。
if __name__ == "__main__":
    main()