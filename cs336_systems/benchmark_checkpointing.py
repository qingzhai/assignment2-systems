# argparse 用来读取命令行参数。
import argparse

# statistics 用来计算多次测量的平均值和标准差。
import statistics

# timeit.default_timer 用来做高精度计时。
import timeit

# Path 用来创建 profiles 目录以及生成 snapshot 文件路径。
from pathlib import Path

# PyTorch。
import torch

# Cross Entropy Loss。
import torch.nn.functional as F

# PyTorch 官方 Activation Checkpointing API。
from torch.utils.checkpoint import checkpoint

# 直接复用我们之前 benchmark.py 中已经写好的基础设施。
from cs336_systems.benchmark import (
    MODEL_CONFIGS,
    autocast_context,
    choose_device,
    create_model,
    create_random_batch,
    synchronize,
)


# ============================================================
# 1. forward_model
# ============================================================

def forward_model(
    model,
    inputs: torch.Tensor,
    use_checkpoint: bool,
):
    """
    执行 Transformer Language Model 的 forward。

    和原始 BasicsTransformerLM.forward() 的整体结构保持一致：

        token embedding
            ↓
        TransformerBlock 0
            ↓
        TransformerBlock 1
            ↓
        ...
            ↓
        final RMSNorm
            ↓
        LM Head

    唯一的区别：

        use_checkpoint=False

            x = layer(x)

        use_checkpoint=True

            x = checkpoint(layer, x)

    也就是说：

    我们 checkpoint 的粒度是：

        一个完整 TransformerBlock。

    ------------------------------------------------------------

    Activation Checkpointing 的意义：

    普通 forward：

        Block 内部 intermediate activations
            ↓
        保存到 backward

    checkpoint：

        只保存 checkpoint boundary 所需要的信息
            ↓
        Block 内大量 intermediate activations 不长期保存
            ↓
        backward 时重新执行该 Block 的 forward
    """

    # --------------------------------------------------------
    # Token Embedding
    # --------------------------------------------------------

    # inputs shape:
    #
    # [batch_size, sequence_length]
    #
    # x shape:
    #
    # [batch_size, sequence_length, d_model]
    x = model.token_embeddings(inputs)

    # --------------------------------------------------------
    # Transformer Blocks
    # --------------------------------------------------------

    # 逐层执行 TransformerBlock。
    for layer in model.layers:

        # 如果开启 activation checkpointing。
        if use_checkpoint:

            # checkpoint() 不长期保存 Block 内部所有中间 activation。
            #
            # 在 backward 真正需要这些 tensor 时，
            # PyTorch 会重新执行 layer(x)。
            #
            # use_reentrant=False 是目前推荐的实现方式。
            x = checkpoint(
                layer,
                x,
                use_reentrant=False,
            )

        else:

            # 普通 forward。
            #
            # Autograd 会保存 backward 所需要的 residual tensors。
            x = layer(x)

    # --------------------------------------------------------
    # Final RMSNorm
    # --------------------------------------------------------

    # Transformer blocks 全部完成后，
    # 执行最后的 RMSNorm。
    x = model.ln_final(x)

    # --------------------------------------------------------
    # LM Head
    # --------------------------------------------------------

    # 投影到 vocabulary dimension。
    #
    # logits shape:
    #
    # [batch_size, sequence_length, vocab_size]
    logits = model.lm_head(x)

    # 返回 logits。
    return logits


# ============================================================
# 2. run_forward_backward
# ============================================================

def run_forward_backward(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    use_checkpoint: bool,
    collect_memory: bool = False,
):
    """
    执行一次完整的：

        Forward
            ↓
        Cross Entropy Loss
            ↓
        Backward

    不执行 optimizer.step()。

    原因：

        我们现在研究 Activation Checkpointing。

    Checkpointing 主要改变的是：

        Forward saved activations
        +
        Backward recomputation

    如果加入 AdamW：

        exp_avg
        exp_avg_sq

    会增加大量固定 optimizer-state memory，
    反而干扰我们观察 activation memory 的变化。

    ------------------------------------------------------------

    如果 collect_memory=True，
    还会记录：

        before_forward
        after_forward
        after_backward
        peak
    """

    # 每个 training step 开始前，
    # 删除上一次留下来的 gradients。
    #
    # set_to_none=True 不会创建全 0 gradient tensor，
    # 更适合显存分析。
    model.zero_grad(set_to_none=True)

    # 如果需要显存统计，
    # 先记录 forward 前的 active memory。
    if collect_memory:

        # CUDA allocated memory，单位 Bytes。
        before_forward = torch.cuda.memory_allocated(device)

    # ========================================================
    # Forward
    # ========================================================

    # 开启 BF16 autocast。
    #
    # 如果 --no-amp，
    # autocast_context 会退化成 nullcontext。
    with autocast_context(
        device=device,
        use_amp=use_amp,
    ):

        # 执行普通 Transformer forward
        # 或 checkpoint Transformer forward。
        logits = forward_model(
            model=model,
            inputs=inputs,
            use_checkpoint=use_checkpoint,
        )

    # 如果需要观察 Forward 结束后的真实 GPU memory，
    # 必须先 synchronize。
    if collect_memory:

        # 等 GPU forward kernels 完成。
        synchronize(device)

        # 此时普通模式下：
        #
        # 很多 Autograd residuals 仍然存在。
        #
        # checkpoint 模式下：
        #
        # 大量 Block 内部 residuals 没有长期保存。
        after_forward = torch.cuda.memory_allocated(device)

    # ========================================================
    # Loss
    # ========================================================

    # logits:
    #
    # [B, S, V]
    #
    # reshape:
    #
    # [B*S, V]
    logits_flat = logits.reshape(
        -1,
        logits.shape[-1],
    )

    # targets:
    #
    # [B, S]
    #
    # reshape:
    #
    # [B*S]
    targets_flat = targets.reshape(-1)

    # 计算 language modeling loss。
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
    )

    # ========================================================
    # Backward
    # ========================================================

    # 普通模式：
    #
    # backward 直接读取 forward 保存的 residual tensors。
    #
    # checkpoint 模式：
    #
    # backward 会重新运行一部分 TransformerBlock forward，
    # 重建需要的 intermediate activations。
    loss.backward()

    # 如果需要 memory 数据。
    if collect_memory:

        # 确保 backward 真正完成。
        synchronize(device)

        # 此时 gradients 已经创建。
        after_backward = torch.cuda.memory_allocated(device)

        # 获取整个 step 中出现过的最大 allocated memory。
        peak = torch.cuda.max_memory_allocated(device)

        # 返回 loss 和四个 memory 指标。
        return {
            "loss": loss.item(),
            "before_forward": before_forward,
            "after_forward": after_forward,
            "after_backward": after_backward,
            "peak": peak,
        }

    # 普通 timing 模式不需要返回 memory 信息。
    return None


# ============================================================
# 3. benchmark_runtime
# ============================================================

def benchmark_runtime(
    model,
    inputs,
    targets,
    device,
    use_amp,
    use_checkpoint,
    warmup_steps,
    measurement_steps,
):
    """
    Benchmark Forward + Backward wall-clock latency。

    关键原则：

        CUDA execution 是异步的。

    所以每次真正计时前后都必须：

        torch.cuda.synchronize()

    否则 CPU timer 可能只测到：

        kernel launch

    而不是：

        GPU 真正完成计算。
    """

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    # 先执行若干 warmup step。
    for _ in range(warmup_steps):

        # 执行完整 Forward + Backward。
        run_forward_backward(
            model=model,
            inputs=inputs,
            targets=targets,
            device=device,
            use_amp=use_amp,
            use_checkpoint=use_checkpoint,
            collect_memory=False,
        )

        # 等 GPU 真正完成。
        synchronize(device)

    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    # 保存每一次 measurement 的 latency。
    timings_ms = []

    # 执行多次 measurement。
    for _ in range(measurement_steps):

        # 确保之前 GPU queue 中没有遗留工作。
        synchronize(device)

        # 记录开始时间。
        start = timeit.default_timer()

        # 执行 Forward + Backward。
        run_forward_backward(
            model=model,
            inputs=inputs,
            targets=targets,
            device=device,
            use_amp=use_amp,
            use_checkpoint=use_checkpoint,
            collect_memory=False,
        )

        # CUDA 是异步的，
        # 所以一定要等待 GPU 真正执行完成。
        synchronize(device)

        # 记录结束时间。
        end = timeit.default_timer()

        # 秒转换成毫秒。
        elapsed_ms = (end - start) * 1000

        # 保存结果。
        timings_ms.append(elapsed_ms)

    # 计算平均时间。
    mean_ms = statistics.mean(timings_ms)

    # 如果只有一次 measurement，
    # statistics.stdev 会报错，
    # 因此单次时 std 设置为 0。
    std_ms = (
        statistics.stdev(timings_ms)
        if len(timings_ms) > 1
        else 0.0
    )

    # 返回全部 timing 信息。
    return timings_ms, mean_ms, std_ms


# ============================================================
# 4. profile_memory
# ============================================================

def profile_memory(
    model,
    inputs,
    targets,
    device,
    use_amp,
    use_checkpoint,
):
    """
    单独执行一次 Forward + Backward，
    用于准确观察显存。

    为什么不直接使用 timing loop 的 peak？

    因为：

        timing benchmark
        和
        memory profiling

    最好分开。

    这样：

        timing 测时间
        memory probe 测显存

    两者不会互相污染。
    """

    # 删除 timing benchmark 最后一次留下的 gradients。
    model.zero_grad(set_to_none=True)

    # 等待 GPU。
    synchronize(device)

    # 清除 CUDA caching allocator 中当前没有使用的缓存。
    #
    # 参数等活跃 Tensor 不会被删除。
    torch.cuda.empty_cache()

    # 再次同步。
    synchronize(device)

    # 从这里重新统计 peak memory。
    torch.cuda.reset_peak_memory_stats(device)

    # 执行一次真正的 memory probe。
    memory_stats = run_forward_backward(
        model=model,
        inputs=inputs,
        targets=targets,
        device=device,
        use_amp=use_amp,
        use_checkpoint=use_checkpoint,
        collect_memory=True,
    )

    # 返回 memory 数据。
    return memory_stats


# ============================================================
# 5. bytes_to_gb
# ============================================================

def bytes_to_gb(num_bytes: int) -> float:
    """
    将 Bytes 转成 GiB。
    """

    # 1 GiB = 1024^3 Bytes。
    return num_bytes / (1024 ** 3)


# ============================================================
# 6. dump_memory_snapshot
# ============================================================

def dump_memory_snapshot(
    model,
    inputs,
    targets,
    device,
    use_amp,
    use_checkpoint,
    output_dir,
    model_size,
    batch_size,
    context_length,
    max_entries,
):
    """
    可选生成 PyTorch memory snapshot。

    后续可以拖到：

        https://pytorch.org/memory_viz

    对比：

        no checkpoint
        vs
        checkpoint

    的 Active Memory Timeline。
    """

    # 创建输出目录。
    output_path = Path(output_dir)

    # 如果目录不存在则创建。
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 根据是否开启 checkpoint 设置文件名标签。
    checkpoint_name = (
        "checkpoint"
        if use_checkpoint
        else "no_checkpoint"
    )

    # 生成文件名。
    snapshot_path = output_path / (
        f"memory_ckpt_"
        f"{model_size}_"
        f"b{batch_size}_"
        f"s{context_length}_"
        f"{checkpoint_name}.pickle"
    )

    # 清除之前留下的 gradients。
    model.zero_grad(set_to_none=True)

    # GPU 同步。
    synchronize(device)

    # 清掉未使用 allocator cache。
    torch.cuda.empty_cache()

    # GPU 同步。
    synchronize(device)

    # 开启 PyTorch CUDA memory allocation history。
    torch.cuda.memory._record_memory_history(
        max_entries=max_entries,
    )

    # try/finally 保证最终关闭 memory history。
    try:

        # 运行一次 Forward + Backward。
        run_forward_backward(
            model=model,
            inputs=inputs,
            targets=targets,
            device=device,
            use_amp=use_amp,
            use_checkpoint=use_checkpoint,
            collect_memory=False,
        )

        # 等待 CUDA 完成。
        synchronize(device)

        # 保存 snapshot。
        torch.cuda.memory._dump_snapshot(
            str(snapshot_path)
        )

    finally:

        # 关闭 memory history。
        torch.cuda.memory._record_memory_history(
            enabled=None
        )

    # 返回 snapshot 文件路径。
    return snapshot_path


# ============================================================
# 7. parse_args
# ============================================================

def parse_args():
    """
    读取命令行参数。
    """

    # 创建 ArgumentParser。
    parser = argparse.ArgumentParser()

    # 模型规模。
    parser.add_argument(
        "--model-size",
        choices=MODEL_CONFIGS.keys(),
        default="large",
    )

    # Batch size。
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    # Context length。
    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
    )

    # Vocabulary size。
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
    )

    # Warmup 次数。
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
    )

    # Measurement 次数。
    parser.add_argument(
        "--measurement-steps",
        type=int,
        default=5,
    )

    # 运行设备。
    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "mps",
            "cpu",
        ],
        default="auto",
    )

    # 是否关闭 BF16 autocast。
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    # 是否开启 Activation Checkpointing。
    parser.add_argument(
        "--checkpoint",
        action="store_true",
    )

    # 是否额外 dump PyTorch memory snapshot。
    parser.add_argument(
        "--dump-snapshot",
        action="store_true",
    )

    # snapshot 输出目录。
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiles",
    )

    # memory history 最多记录多少事件。
    parser.add_argument(
        "--max-entries",
        type=int,
        default=200000,
    )

    # 返回解析结果。
    return parser.parse_args()


# ============================================================
# 8. main
# ============================================================

def main():
    """
    Activation Checkpointing benchmark 主程序。

    实验比较：

        no checkpoint

    vs

        checkpoint every TransformerBlock

    观察：

        runtime
        after-forward memory
        peak memory
    """

    # 读取参数。
    args = parse_args()

    # 自动选择 device。
    device = choose_device(args.device)

    # 本实验主要针对 CUDA。
    if device.type != "cuda":

        # 非 CUDA 环境直接报错。
        raise RuntimeError(
            "benchmark_checkpointing.py requires CUDA."
        )

    # --no-amp 没出现时，
    # 默认开启 BF16 autocast。
    use_amp = not args.no_amp

    # --------------------------------------------------------
    # 打印实验配置
    # --------------------------------------------------------

    # GPU。
    print(f"device: {device}")

    # 模型。
    print(f"model size: {args.model_size}")

    # Batch。
    print(f"batch size: {args.batch_size}")

    # Sequence length。
    print(f"context length: {args.context_length}")

    # BF16。
    print(f"BF16 autocast: {use_amp}")

    # Checkpoint。
    print(f"activation checkpointing: {args.checkpoint}")

    # --------------------------------------------------------
    # 创建模型
    # --------------------------------------------------------

    # 创建与 benchmark.py 相同的 Transformer。
    model = create_model(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        device=device,
    )

    # --------------------------------------------------------
    # 创建数据
    # --------------------------------------------------------

    # 随机 next-token prediction batch。
    inputs, targets = create_random_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=device,
    )

    # ========================================================
    # Runtime Benchmark
    # ========================================================

    # 测 Forward + Backward latency。
    timings_ms, mean_ms, std_ms = benchmark_runtime(
        model=model,
        inputs=inputs,
        targets=targets,
        device=device,
        use_amp=use_amp,
        use_checkpoint=args.checkpoint,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
    )

    # ========================================================
    # Memory Profiling
    # ========================================================

    # 单独再跑一次 step 测显存。
    memory_stats = profile_memory(
        model=model,
        inputs=inputs,
        targets=targets,
        device=device,
        use_amp=use_amp,
        use_checkpoint=args.checkpoint,
    )

    # ========================================================
    # 输出结果
    # ========================================================

    # 空行。
    print()

    # 打印每次 latency。
    print(
        "timings (ms):",
        [round(x, 3) for x in timings_ms],
    )

    # 平均 latency。
    print(
        f"forward + backward mean: "
        f"{mean_ms:.3f} ms"
    )

    # latency 标准差。
    print(
        f"forward + backward std:  "
        f"{std_ms:.3f} ms"
    )

    # 空行。
    print()

    # Forward 之前。
    print(
        "memory before forward: "
        f"{bytes_to_gb(memory_stats['before_forward']):.3f} GB"
    )

    # Forward 完成之后。
    print(
        "memory after forward:  "
        f"{bytes_to_gb(memory_stats['after_forward']):.3f} GB"
    )

    # Backward 完成之后。
    print(
        "memory after backward: "
        f"{bytes_to_gb(memory_stats['after_backward']):.3f} GB"
    )

    # Peak。
    print(
        "peak GPU memory:       "
        f"{bytes_to_gb(memory_stats['peak']):.3f} GB"
    )

    # ========================================================
    # Optional Memory Snapshot
    # ========================================================

    # 如果用户要求生成 pickle。
    if args.dump_snapshot:

        # 生成 snapshot。
        snapshot_path = dump_memory_snapshot(
            model=model,
            inputs=inputs,
            targets=targets,
            device=device,
            use_amp=use_amp,
            use_checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            model_size=args.model_size,
            batch_size=args.batch_size,
            context_length=args.context_length,
            max_entries=args.max_entries,
        )

        # 空行。
        print()

        # 打印路径。
        print(
            f"Memory snapshot saved to: "
            f"{snapshot_path}"
        )


# ============================================================
# 9. Python entry point
# ============================================================

# 直接运行当前 Python 文件时。
if __name__ == "__main__":

    # 执行主程序。
    main()