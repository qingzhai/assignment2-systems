# argparse 用来读取命令行参数。
import argparse

# Path 用来创建 profiles 目录以及拼接 snapshot 文件路径。
from pathlib import Path

# PyTorch。
import torch

# F 中包含 cross_entropy。
import torch.nn.functional as F

# 直接复用 benchmark.py 中已经写好的模型配置和辅助函数，
# 这样就不用在两个文件里维护两套模型配置。
from cs336_systems.benchmark import (
    MODEL_CONFIGS,
    autocast_context,
    choose_device,
    create_model,
    create_random_batch,
    synchronize,
)


# ============================================================
# 1. report_memory
# ============================================================

def report_memory(
    stage: str,
    device: torch.device,
):
    """
    打印某一个时刻的 CUDA 显存状态。

    我们主要观察三个指标：

        allocated:
            当前真正被 PyTorch Tensor 占用的显存。

        reserved:
            PyTorch CUDA allocator 已经向 CUDA 申请、
            但不一定全部正在被 Tensor 使用的显存。

        peak:
            从上一次 reset_peak_memory_stats() 开始，
            allocated 曾经达到过的最高值。

    ----------------------------------------------------------

    举例：

        allocated = 5 GB
        reserved  = 7 GB

    表示：

        PyTorch 当前 Tensor 真正用了约 5 GB，

        但是 allocator 手里一共留着约 7 GB，

        剩余约 2 GB 可能是缓存，
        以后再次申请 Tensor 时可以直接复用。
    """

    # 如果不是 CUDA，
    # PyTorch CUDA memory API 没有意义，
    # 所以直接返回。
    if device.type != "cuda":

        # 结束当前函数。
        return

    # 获得当前实际 allocated memory，单位 Bytes。
    allocated_bytes = torch.cuda.memory_allocated(device)

    # 获得 PyTorch allocator 当前 reserved memory，单位 Bytes。
    reserved_bytes = torch.cuda.memory_reserved(device)

    # 获得从上次 reset 后出现过的最大 allocated memory。
    peak_bytes = torch.cuda.max_memory_allocated(device)

    # Bytes 转换为 GiB。
    allocated_gb = allocated_bytes / (1024 ** 3)

    # Bytes 转换为 GiB。
    reserved_gb = reserved_bytes / (1024 ** 3)

    # Bytes 转换为 GiB。
    peak_gb = peak_bytes / (1024 ** 3)

    # 打印当前阶段名称。
    print(f"[{stage}]")

    # 打印当前真正被 Tensor 使用的显存。
    print(f"  allocated: {allocated_gb:.3f} GB")

    # 打印 allocator 当前保留的显存。
    print(f"  reserved:  {reserved_gb:.3f} GB")

    # 打印目前出现过的峰值显存。
    print(f"  peak:      {peak_gb:.3f} GB")


# ============================================================
# 2. run_memory_step
# ============================================================

def run_memory_step(
    model,
    optimizer,
    inputs,
    targets,
    mode: str,
    device: torch.device,
    use_amp: bool,
):
    """
    执行一次用于 memory profiling 的训练 step。

    支持三种模式：

        forward

            Forward

        forward_backward

            Forward
              ↓
            Loss
              ↓
            Backward

        full

            Forward
              ↓
            Loss
              ↓
            Backward
              ↓
            AdamW optimizer.step()

    ----------------------------------------------------------

    和 benchmark.py 最大的区别：

    benchmark.py 关心：

        每一步花了多少时间？

    benchmark_memory.py 关心：

        每一步分配了多少显存？
        哪些显存后来释放了？
        peak 出现在哪里？

    因此这里会在关键阶段调用 report_memory()。
    """

    # --------------------------------------------------------
    # Step 开始
    # --------------------------------------------------------

    # 如果涉及 backward，
    # 先保证 parameter.grad 为空。
    if mode != "forward":

        # set_to_none=True 不创建全 0 gradient tensor，
        # 而是直接把 grad 设置为 None。
        optimizer.zero_grad(set_to_none=True)

    # 确保之前所有 CUDA 工作已经完成。
    synchronize(device)

    # 打印 step 开始之前的显存。
    report_memory(
        stage="Before forward",
        device=device,
    )

    # ========================================================
    # Forward
    # ========================================================

    # 进入 autocast context。
    with autocast_context(
        device=device,
        use_amp=use_amp,
    ):

        # Transformer forward。
        #
        # inputs:
        # [batch_size, seq_len]
        #
        # logits:
        # [batch_size, seq_len, vocab_size]
        logits = model(inputs)

    # 等待 forward GPU kernels 真正执行结束。
    synchronize(device)

    # 此时训练 forward 为 backward 保存的 activation
    # 仍然存在，所以这里的 memory 很有参考价值。
    report_memory(
        stage="After forward",
        device=device,
    )

    # --------------------------------------------------------
    # 如果只分析 forward，
    # 到这里就可以结束。
    # --------------------------------------------------------

    if mode == "forward":

        # 返回。
        return

    # ========================================================
    # Loss
    # ========================================================

    # 再次进入 autocast context。
    with autocast_context(
        device=device,
        use_amp=use_amp,
    ):

        # logits 原始 shape：
        #
        # [B, S, V]
        #
        # reshape 后：
        #
        # [B*S, V]
        logits_flat = logits.reshape(
            -1,
            logits.shape[-1],
        )

        # targets 原始 shape：
        #
        # [B, S]
        #
        # reshape 后：
        #
        # [B*S]
        targets_flat = targets.reshape(-1)

        # 计算语言模型 Cross Entropy Loss。
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
        )

    # 等待 loss 相关 CUDA kernels 完成。
    synchronize(device)

    # 打印计算 loss 后的显存。
    report_memory(
        stage="After loss",
        device=device,
    )

    # ========================================================
    # Backward
    # ========================================================

    # 进行反向传播。
    loss.backward()

    # 等待 backward GPU kernels 完成。
    synchronize(device)

    # 到这里：
    #
    # 一部分 forward saved activations 已经释放，
    #
    # 同时 parameter gradients 已经创建。
    #
    # 因此这里非常值得和 After forward 对比。
    report_memory(
        stage="After backward",
        device=device,
    )

    # --------------------------------------------------------
    # 如果只分析 forward + backward，
    # 到这里结束。
    # --------------------------------------------------------

    if mode == "forward_backward":

        # 返回。
        return

    # ========================================================
    # Optimizer
    # ========================================================

    # 执行 AdamW 参数更新。
    #
    # 第一次 optimizer.step() 非常重要：
    #
    # AdamW 会在这里 lazy-create：
    #
    # exp_avg
    # exp_avg_sq
    #
    # 也就是我们之前说的 m 和 v。
    optimizer.step()

    # 等待 optimizer kernels 真正完成。
    synchronize(device)

    # 这里通常能看到显存出现明显增长，
    # 主要原因就是 Adam optimizer states。
    report_memory(
        stage="After optimizer",
        device=device,
    )


# ============================================================
# 3. build_snapshot_path
# ============================================================

def build_snapshot_path(
    output_dir: str,
    model_size: str,
    mode: str,
    batch_size: int,
    context_length: int,
    use_amp: bool,
):
    """
    自动生成 memory snapshot 文件名。

    例如：

        profiles/
        memory_large_full_b1_s256_bf16.pickle

    或：

        profiles/
        memory_large_forward_b1_s1024_fp32.pickle

    这样我们以后做 context length sweep 时，
    不容易把多个 snapshot 搞混。
    """

    # 将字符串路径转换成 Path 对象。
    output_path = Path(output_dir)

    # 如果目录不存在，就自动创建。
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 根据 AMP 是否开启决定精度标签。
    precision_name = (
        "bf16"
        if use_amp
        else "fp32"
    )

    # 拼接 snapshot 文件名。
    filename = (
        f"memory_"
        f"{model_size}_"
        f"{mode}_"
        f"b{batch_size}_"
        f"s{context_length}_"
        f"{precision_name}.pickle"
    )

    # 返回完整文件路径。
    return output_path / filename


# ============================================================
# 4. parse_args
# ============================================================

def parse_args():
    """
    读取命令行参数。

    示例：

        uv run python cs336_systems/benchmark_memory.py \
            --model-size large \
            --mode full \
            --batch-size 1 \
            --context-length 256
    """

    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Model size
    # --------------------------------------------------------

    # 添加模型规模参数。
    parser.add_argument(

        # 参数名。
        "--model-size",

        # 必须从 benchmark.py 的 MODEL_CONFIGS 中选择。
        choices=MODEL_CONFIGS.keys(),

        # 默认先用 tiny。
        default="tiny",
    )

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    # 添加运行模式。
    parser.add_argument(

        # 参数名。
        "--mode",

        # 支持三种模式。
        choices=[
            "forward",
            "forward_backward",
            "full",
        ],

        # 默认 forward。
        default="forward",
    )

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    # 添加 batch size。
    parser.add_argument(

        # 参数名。
        "--batch-size",

        # 整数。
        type=int,

        # 4090D 先从 1 开始。
        default=1,
    )

    # --------------------------------------------------------
    # Context length
    # --------------------------------------------------------

    # 添加 sequence/context length。
    parser.add_argument(

        # 参数名。
        "--context-length",

        # 整数。
        type=int,

        # 默认继续使用 baseline 的 256。
        default=256,
    )

    # --------------------------------------------------------
    # Vocabulary size
    # --------------------------------------------------------

    # 添加 vocabulary size。
    parser.add_argument(

        # 参数名。
        "--vocab-size",

        # 整数。
        type=int,

        # 与 benchmark.py 保持一致。
        default=10000,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    # 添加运行设备。
    parser.add_argument(

        # 参数名。
        "--device",

        # 支持这些设备。
        choices=[
            "auto",
            "cuda",
            "mps",
            "cpu",
        ],

        # 默认自动判断。
        default="auto",
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    # 添加关闭 AMP 的参数。
    parser.add_argument(

        # 使用 --no-amp 时关闭 BF16 autocast。
        "--no-amp",

        # 出现参数后变成 True。
        action="store_true",
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    # 添加 snapshot 输出目录。
    parser.add_argument(

        # 参数名。
        "--output-dir",

        # 字符串。
        type=str,

        # 默认放进 profiles。
        default="profiles",
    )

    # --------------------------------------------------------
    # Memory history entries
    # --------------------------------------------------------

    # 设置最多保存多少条显存 allocation/free event。
    parser.add_argument(

        # 参数名。
        "--max-entries",

        # 整数。
        type=int,

        # 单个 training step 使用 200000 已经很充足。
        default=200000,
    )

    # 返回解析后的参数。
    return parser.parse_args()


# ============================================================
# 5. main
# ============================================================

def main():
    """
    Memory profiling 程序入口。

    整体流程：

        创建模型
            ↓
        创建 optimizer
            ↓
        创建随机 batch
            ↓
        清理 allocator cache
            ↓
        开启 memory history
            ↓
        跑一次 forward / backward / optimizer
            ↓
        dump memory snapshot
            ↓
        关闭 memory history

    ----------------------------------------------------------

    注意：

    这里故意不执行 full-step warmup。

    因为如果提前调用过：

        optimizer.step()

    AdamW 的：

        exp_avg
        exp_avg_sq

    就已经创建完成。

    那么我们真正开始 recording 时，
    就看不到 optimizer state 从无到有的过程了。

    Memory profiling 和 latency benchmark 的目标不同，
    所以这里不需要照搬 benchmark.py 的 warmup。
    """

    # --------------------------------------------------------
    # 读取命令行参数
    # --------------------------------------------------------

    # 解析参数。
    args = parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    # 自动选择 device。
    device = choose_device(
        args.device
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    # 没有传 --no-amp 时默认开启 BF16 autocast。
    use_amp = not args.no_amp

    # --------------------------------------------------------
    # 检查 CUDA
    # --------------------------------------------------------

    # memory snapshot 是我们为了 CUDA GPU 做的实验。
    if device.type != "cuda":

        # 非 CUDA 环境直接报错，
        # 避免误以为 Mac MPS 也能生成 CUDA memory snapshot。
        raise RuntimeError(
            "benchmark_memory.py requires a CUDA GPU."
        )

    # --------------------------------------------------------
    # 打印当前实验配置
    # --------------------------------------------------------

    # 打印 GPU。
    print(f"device: {device}")

    # 打印模型规模。
    print(f"model size: {args.model_size}")

    # 打印模式。
    print(f"mode: {args.mode}")

    # 打印 batch size。
    print(f"batch size: {args.batch_size}")

    # 打印 context length。
    print(f"context length: {args.context_length}")

    # 打印 vocabulary size。
    print(f"vocab size: {args.vocab_size}")

    # 打印 BF16 autocast 状态。
    print(
        "BF16 autocast: "
        f"{use_amp}"
    )

    # --------------------------------------------------------
    # 创建模型
    # --------------------------------------------------------

    # 创建 Transformer，
    # 并直接放到 CUDA GPU。
    model = create_model(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        device=device,
    )

    # --------------------------------------------------------
    # 创建 optimizer
    # --------------------------------------------------------

    # 创建 AdamW。
    optimizer = torch.optim.AdamW(

        # optimizer 管理所有模型参数。
        model.parameters(),

        # 学习率在本实验里并不重要。
        lr=1e-3,
    )

    # 注意：
    #
    # 到这里 AdamW 还没有真正创建 m / v。
    #
    # 它们一般会等到第一次 optimizer.step()
    # 才 lazy initialization。

    # --------------------------------------------------------
    # 创建随机数据
    # --------------------------------------------------------

    # 创建随机 next-token prediction batch。
    inputs, targets = create_random_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=device,
    )

    # --------------------------------------------------------
    # 同步 CUDA
    # --------------------------------------------------------

    # 等待模型和数据相关的 CUDA 操作完成。
    synchronize(device)

    # --------------------------------------------------------
    # 清理未使用 allocator cache
    # --------------------------------------------------------

    # empty_cache 不会释放模型参数等仍然活跃的 Tensor。
    #
    # 它只是把 allocator 中当前没有使用的缓存块还给 CUDA。
    #
    # 这样 snapshot 起点相对更干净。
    torch.cuda.empty_cache()

    # 再次同步。
    synchronize(device)

    # --------------------------------------------------------
    # 重置 peak memory
    # --------------------------------------------------------

    # 从当前这个时刻重新开始统计 peak allocated memory。
    torch.cuda.reset_peak_memory_stats(device)

    # --------------------------------------------------------
    # 打印 baseline memory
    # --------------------------------------------------------

    # 此时主要已经存在：
    #
    # parameters
    # inputs
    # targets
    #
    # Adam m/v 尚未创建。
    report_memory(
        stage="Baseline before recording",
        device=device,
    )

    # --------------------------------------------------------
    # 构造 snapshot 路径
    # --------------------------------------------------------

    # 自动生成 snapshot 文件名。
    snapshot_path = build_snapshot_path(
        output_dir=args.output_dir,
        model_size=args.model_size,
        mode=args.mode,
        batch_size=args.batch_size,
        context_length=args.context_length,
        use_amp=use_amp,
    )

    # --------------------------------------------------------
    # 开始记录 CUDA memory history
    # --------------------------------------------------------

    # 开启 PyTorch CUDA allocator allocation/free history。
    #
    # 之后发生的 allocation 和 free
    # 都会被记录到 history 中。
    torch.cuda.memory._record_memory_history(

        # 最大保存 event 数量。
        max_entries=args.max_entries,
    )

    # 使用 try/finally，
    # 确保无论中间是否报错，
    # 最后都会关闭 memory history。
    try:

        # ----------------------------------------------------
        # 执行一次真正需要分析的 step
        # ----------------------------------------------------

        # 跑一次 forward / forward_backward / full。
        run_memory_step(
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            mode=args.mode,
            device=device,
            use_amp=use_amp,
        )

        # ----------------------------------------------------
        # 同步
        # ----------------------------------------------------

        # 确保所有 CUDA memory operation 都已经完成。
        synchronize(device)

        # ----------------------------------------------------
        # 输出最终 memory 状态
        # ----------------------------------------------------

        # 打印最终显存。
        report_memory(
            stage="End of step",
            device=device,
        )

        # ----------------------------------------------------
        # 保存 snapshot
        # ----------------------------------------------------

        # 将完整的 allocator state + allocation history
        # 保存为 pickle。
        torch.cuda.memory._dump_snapshot(
            str(snapshot_path)
        )

        # 告诉用户文件生成在哪里。
        print()

        # 打印 snapshot 路径。
        print(
            f"Memory snapshot saved to: "
            f"{snapshot_path}"
        )

        # 打印下一步应该做什么。
        print(
            "Open https://pytorch.org/memory_viz "
            "and drag this pickle file into the viewer."
        )

    finally:

        # ----------------------------------------------------
        # 停止记录 CUDA memory history
        # ----------------------------------------------------

        # enabled=None 表示关闭 memory history。
        torch.cuda.memory._record_memory_history(
            enabled=None
        )


# ============================================================
# 6. Python entry point
# ============================================================

# 当前文件被直接执行时。
if __name__ == "__main__":

    # 执行 main。
    main()