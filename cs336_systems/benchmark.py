# argparse 用来读取命令行参数，例如：
# --model-size small
# --mode forward
import argparse

# statistics 用来计算 benchmark 多次测量的平均值和标准差
import statistics

# timeit 提供高精度计时器
# benchmark 中比普通 time.time() 更合适
import timeit

# contextmanager 用来定义 NVTX 的上下文管理器
# 最后可以写成：
#
# with nvtx_range("Forward"):
#     ...
#
# 而不是每次手动 range_push / range_pop
from contextlib import contextmanager

# nullcontext 表示一个“什么都不做”的上下文管理器
# 当不是 CUDA 环境时，我们用它代替 autocast
from contextlib import nullcontext

# PyTorch
import torch

# F 中包含 cross_entropy 等常用函数
import torch.nn.functional as F

# 直接复用 CS336 A1 提供的 Transformer 实现
from cs336_basics.model import BasicsTransformerLM


# ============================================================
# 1. 模型配置
# ============================================================
#
# 我们没有机械照搬课程里特别大的模型。
#
# 原因：
#
# 我们目前主要使用 RTX 4090 / 4090D 24GB 做实验，
# 目标是学习：
#
# benchmark
# profiling
# GPU memory
# kernel
# Triton
#
# 而不是复现 Stanford 集群上的绝对性能数字。
#
# 四个规模足够我们观察：
#
# 模型变大
#     ↓
# 参数量增加
#     ↓
# FLOPs 增加
#     ↓
# 显存增加
#     ↓
# latency / GPU utilization 变化
#
# ============================================================

MODEL_CONFIGS = {

    # --------------------------------------------------------
    # tiny
    # --------------------------------------------------------
    #
    # 最小模型。
    #
    # 主要作用：
    # 1. 快速测试代码有没有问题
    # 2. 快速验证 CUDA / MPS
    # 3. 避免一开始就占很多显存
    #
    "tiny": {

        # Transformer hidden dimension
        "d_model": 512,

        # FFN 中间层维度
        "d_ff": 2048,

        # Transformer Block 数量
        "num_layers": 6,

        # Multi-Head Attention 的 head 数量
        "num_heads": 8,
    },

    # --------------------------------------------------------
    # small
    # --------------------------------------------------------
    "small": {

        # hidden dimension
        "d_model": 768,

        # FFN dimension
        "d_ff": 3072,

        # Transformer Block 数量
        "num_layers": 12,

        # Attention heads
        "num_heads": 12,
    },

    # --------------------------------------------------------
    # medium
    # --------------------------------------------------------
    "medium": {

        # hidden dimension
        "d_model": 1024,

        # FFN dimension
        "d_ff": 4096,

        # Transformer Block 数量
        "num_layers": 16,

        # Attention heads
        "num_heads": 16,
    },

    # --------------------------------------------------------
    # large
    # --------------------------------------------------------
    #
    # 我们目前最大的 benchmark 模型。
    #
    # 在 4090D 24GB 上：
    #
    # batch_size = 1
    # context_length = 256
    #
    # full training step 实测峰值显存约 10GB 左右，
    # 因此仍然有比较充足的余量。
    #
    "large": {

        # hidden dimension
        "d_model": 1280,

        # FFN dimension
        "d_ff": 5120,

        # Transformer Block 数量
        "num_layers": 20,

        # Attention heads
        "num_heads": 20,
    },
}


# ============================================================
# 2. NVTX Range
# ============================================================

@contextmanager
def nvtx_range(name: str, device: torch.device):
    """
    给一段代码添加 NVTX 标记。

    Nsight Systems 可以直接读取 NVTX range。

    比如：

        with nvtx_range("Forward", device):
            logits = model(inputs)

    在 Nsight Systems Timeline 中就会出现：

        [ Forward ]
        ███████████████

    这样我们就不需要面对一堆完全不知道属于哪里的
    CUDA kernels。

    ----------------------------------------------------------

    为什么需要判断 CUDA？

    torch.cuda.nvtx 是 CUDA 功能。

    我们 Mac 本地使用 MPS 测试代码时，
    不应该调用 CUDA NVTX。

    所以：

        CUDA:
            真正添加 NVTX

        MPS / CPU:
            什么都不做

    """

    # 判断当前是否运行在 NVIDIA CUDA GPU 上
    if device.type == "cuda":

        # 向 CUDA NVTX stack 中压入一个 range
        #
        # Nsight Systems 会记录这个名称
        torch.cuda.nvtx.range_push(name)

        try:

            # yield 表示真正执行 with 块内部的代码
            yield

        finally:

            # 无论内部代码是否报错，
            # 都把这个 NVTX range 正确结束
            torch.cuda.nvtx.range_pop()

    else:

        # Mac MPS / CPU 环境不需要 NVTX
        yield


# ============================================================
# 3. synchronize
# ============================================================

def synchronize(device: torch.device):
    """
    等待 accelerator 把前面提交的任务全部执行完。

    ----------------------------------------------------------

    CUDA 默认是异步执行：

        CPU
         │
         │ launch kernel
         ↓
        GPU 开始计算

    CPU 发出 kernel 后，
    并不一定会等 GPU 真正完成，
    而是继续执行下面的 Python。

    如果我们直接：

        start = timer()

        model(x)

        end = timer()

    CPU 很可能在 GPU 还没执行完成时就记录 end。

    那么测出来的就主要是：

        kernel launch 时间

    而不是：

        GPU 真正执行时间。

    所以 benchmark 时必须 synchronize。
    """

    # --------------------------------------------------------
    # NVIDIA CUDA GPU
    # --------------------------------------------------------

    if device.type == "cuda":

        # 等待当前 CUDA device 上前面所有任务完成
        torch.cuda.synchronize()

    # --------------------------------------------------------
    # Apple Silicon MPS
    # --------------------------------------------------------

    elif device.type == "mps":

        # MPS 同样具有异步执行特征
        torch.mps.synchronize()


# ============================================================
# 4. autocast_context
# ============================================================

def autocast_context(
    device: torch.device,
    use_amp: bool,
):
    """
    根据当前设备决定是否使用 BF16 autocast。

    ----------------------------------------------------------

    在 RTX 4090 / 4090D 上：

        FP32 parameter
              ↓
        autocast
              ↓
        很多矩阵运算使用 BF16
              ↓
        Tensor Core

    ----------------------------------------------------------

    注意：

    autocast 并不等于：

        把模型永久变成 BF16。

    模型 parameter 本身仍然可以是 FP32。

    只是执行适合的算子时，
    PyTorch 自动选择 BF16。

    这也是我们之前在 Nsight Systems 中看到大量：

        bfloat16_copy_kernel

    的一个重要背景。
    """

    # 只有 CUDA 环境并且用户没有关闭 AMP 时
    # 才开启 BF16 autocast
    if device.type == "cuda" and use_amp:

        # 返回 CUDA BF16 autocast context
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    # CPU / MPS 环境下不改变数据类型
    return nullcontext()


# ============================================================
# 5. create_model
# ============================================================

def create_model(
    model_size: str,
    vocab_size: int,
    context_length: int,
    device: torch.device,
):
    """
    根据指定配置创建 Transformer。

    输入：

        model_size:
            tiny / small / medium / large

        vocab_size:
            vocabulary 大小

        context_length:
            最大 sequence length

        device:
            cuda / mps / cpu

    输出：

        已经放到指定 device 上的 Transformer。
    """

    # 根据 model_size 查找对应配置
    config = MODEL_CONFIGS[model_size]

    # 创建 CS336 A1 提供的 Transformer LM
    model = BasicsTransformerLM(

        # vocabulary size
        vocab_size=vocab_size,

        # 最大上下文长度
        context_length=context_length,

        # Transformer hidden dimension
        d_model=config["d_model"],

        # Transformer Block 数量
        num_layers=config["num_layers"],

        # Attention head 数量
        num_heads=config["num_heads"],

        # FFN hidden dimension
        d_ff=config["d_ff"],

        # RoPE theta
        rope_theta=10000.0,
    )

    # 将模型从 CPU 移动到目标 device
    #
    # 在 Nsight Systems 第一份报告里，
    # 那个大约 2.2GB Host -> Device memcpy
    # 主要就是 large model 在这里发生的。
    model = model.to(device)

    # 设置为训练模式
    model.train()

    # 返回模型
    return model


# ============================================================
# 6. create_random_batch
# ============================================================

def create_random_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
):
    """
    创建随机的语言模型训练数据。

    ----------------------------------------------------------

    正常 next-token prediction：

        原始 token：

        token1 token2 token3 token4 token5

        inputs：

        token1 token2 token3 token4

        targets：

               token2 token3 token4 token5

    ----------------------------------------------------------

    Benchmark 不关心文本真正表示什么。

    我们只关心：

        tensor shape
        FLOPs
        latency
        memory

    所以直接生成随机 token id 即可。
    """

    # 生成随机 token ids
    #
    # shape:
    #
    # [batch_size, context_length + 1]
    #
    tokens = torch.randint(

        # token id 从 0 开始
        low=0,

        # 最大 token id 为 vocab_size - 1
        high=vocab_size,

        # 多生成一个 token，
        # 用来构造错位的 inputs / targets
        size=(batch_size, context_length + 1),

        # 直接在 GPU / MPS 上创建数据
        #
        # 避免 benchmark 内部出现额外：
        #
        # CPU → GPU
        #
        # 数据传输
        device=device,
    )

    # 去掉最后一个 token
    #
    # shape:
    #
    # [batch_size, context_length]
    #
    inputs = tokens[:, :-1]

    # 去掉第一个 token
    #
    # shape:
    #
    # [batch_size, context_length]
    #
    targets = tokens[:, 1:]

    # 返回输入和标签
    return inputs, targets


# ============================================================
# 7. run_step
# ============================================================

def run_step(
    model,
    optimizer,
    inputs,
    targets,
    mode: str,
    device: torch.device,
    use_amp: bool,
):
    """
    执行一次模型 step。

    支持三种模式：

    ----------------------------------------------------------

    forward

        Forward

    ----------------------------------------------------------

    forward_backward

        Forward
          ↓
        Loss
          ↓
        Backward

    ----------------------------------------------------------

    full

        Forward
          ↓
        Loss
          ↓
        Backward
          ↓
        Optimizer

    ----------------------------------------------------------

    每一部分都加入 NVTX range。

    所以 Nsight Systems 中可以直接看到：

        Forward
        Loss
        Backward
        Optimizer
    """

    # ========================================================
    # Forward
    # ========================================================

    # 给 Forward 添加 NVTX 标记
    with nvtx_range("Forward", device):

        # 进入 mixed precision context
        with autocast_context(device, use_amp):

            # Transformer forward
            #
            # 输入：
            #
            # [B, S]
            #
            # 输出 logits：
            #
            # [B, S, V]
            #
            logits = model(inputs)

    # --------------------------------------------------------
    # 如果只 benchmark forward
    # 到这里直接结束
    # --------------------------------------------------------

    if mode == "forward":

        return

    # ========================================================
    # Loss
    # ========================================================

    # 将 Loss computation 单独标记
    with nvtx_range("Loss", device):

        # loss 也放在 autocast context 中
        with autocast_context(device, use_amp):

            # logits 原始：
            #
            # [B, S, V]
            #
            # CrossEntropy 希望输入：
            #
            # [B*S, V]
            #
            logits_flat = logits.reshape(
                -1,
                logits.shape[-1],
            )

            # targets 原始：
            #
            # [B, S]
            #
            # reshape 后：
            #
            # [B*S]
            #
            targets_flat = targets.reshape(-1)

            # 计算 Cross Entropy Loss
            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
            )

    # ========================================================
    # Backward
    # ========================================================

    # 给整个 backward 添加 NVTX range
    with nvtx_range("Backward", device):

        # PyTorch autograd 根据 loss 反向传播
        #
        # 计算所有需要训练 parameter 的 gradient
        loss.backward()

    # --------------------------------------------------------
    # forward_backward 模式
    #
    # backward 完成以后就结束
    # --------------------------------------------------------

    if mode == "forward_backward":

        return

    # ========================================================
    # Optimizer
    # ========================================================

    # full 模式才会走到这里
    #
    # 给 optimizer.step() 添加单独 NVTX range
    with nvtx_range("Optimizer", device):

        # AdamW 根据 gradient 更新 parameters
        optimizer.step()


# ============================================================
# 8. benchmark
# ============================================================

def benchmark(
    model,
    optimizer,
    inputs,
    targets,
    mode: str,
    warmup_steps: int,
    measurement_steps: int,
    device: torch.device,
    use_amp: bool,
):
    """
    对指定模式进行 benchmark。

    整体流程：

        WARMUP
            ↓
        Warmup 0
        Warmup 1
        ...
            ↓
        MEASUREMENT
            ↓
        Measurement 0
        Measurement 1
        Measurement 2
        ...
            ↓
        mean / std

    ----------------------------------------------------------

    这次最重要的修改：

    我们给整个阶段加入 NVTX：

        WARMUP

        MEASUREMENT

    并给每一步加入：

        Warmup 0

        Measurement 0

    所以下一次打开 Nsight Systems，
    不需要再猜：

        “这一堆 kernel 到底是哪一次 forward？”

    Timeline 会直接告诉我们。
    """

    # ========================================================
    # Warmup
    # ========================================================

    # 给整个 warmup 阶段添加一个大 NVTX range
    with nvtx_range("WARMUP", device):

        # 执行 warmup_steps 次
        for step in range(warmup_steps):

            # 给当前 warmup step 单独打标记
            with nvtx_range(
                f"Warmup {step}",
                device,
            ):

                # 如果需要 backward，
                # 先清空上一轮 gradient
                if mode != "forward":

                    # set_to_none=True：
                    #
                    # 不真正把 gradient tensor 填成 0，
                    # 而是设成 None，
                    #
                    # 通常更省时间和 memory operation
                    optimizer.zero_grad(set_to_none=True)

                # 执行一次 step
                run_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=inputs,
                    targets=targets,
                    mode=mode,
                    device=device,
                    use_amp=use_amp,
                )

                # 确保 GPU 真正执行完
                synchronize(device)

    # ========================================================
    # Measurement
    # ========================================================

    # 保存每一次 measurement latency
    timings = []

    # --------------------------------------------------------
    # CUDA peak memory
    # --------------------------------------------------------

    if device.type == "cuda":

        # 清空 warmup 阶段留下来的 peak memory 统计
        #
        # 注意：
        #
        # 这不会释放 optimizer states / parameters，
        # 只是重置“最大显存记录”
        torch.cuda.reset_peak_memory_stats(device)

    # 给整个正式 measurement 区域打 NVTX 标记
    with nvtx_range("MEASUREMENT", device):

        # 正式测量 measurement_steps 次
        for step in range(measurement_steps):

            # 给每一次 measurement 单独打 NVTX 标记
            with nvtx_range(
                f"Measurement {step}",
                device,
            ):

                # ------------------------------------------------
                # Gradient reset
                # ------------------------------------------------

                if mode != "forward":

                    # zero_grad 放在 timer 之外
                    #
                    # 因为当前 benchmark 定义的是：
                    #
                    # forward
                    #
                    # forward + backward
                    #
                    # forward + backward + optimizer
                    #
                    # 而不是把 zero_grad 也算进去
                    optimizer.zero_grad(set_to_none=True)

                # ------------------------------------------------
                # Benchmark 开始前同步
                # ------------------------------------------------

                # 保证上一轮 GPU workload 已经完成
                synchronize(device)

                # CPU 记录开始时间
                start_time = timeit.default_timer()

                # ------------------------------------------------
                # 真正执行模型 step
                # ------------------------------------------------

                run_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=inputs,
                    targets=targets,
                    mode=mode,
                    device=device,
                    use_amp=use_amp,
                )

                # ------------------------------------------------
                # Benchmark 结束前同步
                # ------------------------------------------------

                # 等待 GPU 真正执行结束
                synchronize(device)

                # CPU 记录结束时间
                end_time = timeit.default_timer()

                # ------------------------------------------------
                # 计算 latency
                # ------------------------------------------------

                # 秒 → 毫秒
                elapsed_ms = (
                    end_time - start_time
                ) * 1000

                # 保存当前 measurement 结果
                timings.append(elapsed_ms)

    # ========================================================
    # Statistics
    # ========================================================

    # 计算平均 latency
    mean_ms = statistics.mean(timings)

    # 至少两次 measurement 才能计算 sample std
    if len(timings) > 1:

        # 计算标准差
        std_ms = statistics.stdev(timings)

    else:

        # 只有一次时直接设为 0
        std_ms = 0.0

    # ========================================================
    # GPU Peak Memory
    # ========================================================

    # 默认没有 GPU memory 数据
    peak_memory_gb = None

    # 如果运行在 CUDA 上
    if device.type == "cuda":

        # 获得 measurement 期间最大显存占用
        peak_memory_bytes = torch.cuda.max_memory_allocated(
            device
        )

        # Bytes → GiB
        peak_memory_gb = (
            peak_memory_bytes
            / (1024 ** 3)
        )

    # 返回 benchmark 结果
    return (
        mean_ms,
        std_ms,
        timings,
        peak_memory_gb,
    )


# ============================================================
# 9. parse_args
# ============================================================

def parse_args():
    """
    读取命令行参数。

    示例：

        uv run python cs336_systems/benchmark.py \
            --model-size large \
            --mode forward \
            --batch-size 1 \
            --context-length 256
    """

    # 创建 ArgumentParser
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Model size
    # --------------------------------------------------------

    parser.add_argument(

        # 命令行参数名称
        "--model-size",

        # 只能从 MODEL_CONFIGS 中选择
        choices=MODEL_CONFIGS.keys(),

        # 默认 tiny
        default="tiny",
    )

    # --------------------------------------------------------
    # Benchmark mode
    # --------------------------------------------------------

    parser.add_argument(

        # 参数名称
        "--mode",

        # 支持三种模式
        choices=[
            "forward",
            "forward_backward",
            "full",
        ],

        # 默认 forward
        default="forward",
    )

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    parser.add_argument(

        # 参数名称
        "--batch-size",

        # integer
        type=int,

        # 4090D 先从 batch=1 开始
        default=1,
    )

    # --------------------------------------------------------
    # Context length
    # --------------------------------------------------------

    parser.add_argument(

        # 参数名称
        "--context-length",

        # integer
        type=int,

        # 当前 baseline 使用 256
        default=256,
    )

    # --------------------------------------------------------
    # Vocabulary size
    # --------------------------------------------------------

    parser.add_argument(

        # 参数名称
        "--vocab-size",

        # integer
        type=int,

        # 自学 benchmark 使用 10000 足够
        #
        # 可以降低 embedding / LM head 显存
        default=10000,
    )

    # --------------------------------------------------------
    # Warmup steps
    # --------------------------------------------------------

    parser.add_argument(

        # 参数名称
        "--warmup-steps",

        # integer
        type=int,

        # 默认 5 次
        default=5,
    )

    # --------------------------------------------------------
    # Measurement steps
    # --------------------------------------------------------

    parser.add_argument(

        # 参数名称
        "--measurement-steps",

        # integer
        type=int,

        # 默认 10 次
        default=10,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    parser.add_argument(

        # 参数名称
        "--device",

        # 支持的 device
        choices=[
            "auto",
            "cuda",
            "mps",
            "cpu",
        ],

        # 默认自动选择
        default="auto",
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    parser.add_argument(

        # 出现 --no-amp 时关闭 AMP
        "--no-amp",

        # store_true：
        #
        # 默认 False
        #
        # 用户写 --no-amp 后变 True
        action="store_true",
    )

    # 返回所有参数
    return parser.parse_args()


# ============================================================
# 10. choose_device
# ============================================================

def choose_device(device_arg: str):
    """
    自动选择运行设备。

    优先级：

        CUDA
          ↓
        MPS
          ↓
        CPU
    """

    # 如果用户明确指定了 device
    if device_arg != "auto":

        # 直接使用用户指定的设备
        return torch.device(device_arg)

    # NVIDIA GPU 可用
    if torch.cuda.is_available():

        # 使用 CUDA
        return torch.device("cuda")

    # Apple Silicon MPS 可用
    if torch.backends.mps.is_available():

        # 使用 MPS
        return torch.device("mps")

    # 否则使用 CPU
    return torch.device("cpu")


# ============================================================
# 11. main
# ============================================================

def main():
    """
    Benchmark 程序入口。

    完整流程：

        parse args
            ↓
        choose device
            ↓
        create model
            ↓
        create optimizer
            ↓
        create random batch
            ↓
        warmup
            ↓
        measurement
            ↓
        mean / std / peak memory
    """

    # --------------------------------------------------------
    # 读取参数
    # --------------------------------------------------------

    args = parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = choose_device(
        args.device
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    # 默认开启 AMP
    #
    # 如果用户传：
    #
    # --no-amp
    #
    # 则关闭
    use_amp = not args.no_amp

    # --------------------------------------------------------
    # 打印实验配置
    # --------------------------------------------------------

    print(
        f"device: {device}"
    )

    print(
        f"model size: {args.model_size}"
    )

    print(
        f"mode: {args.mode}"
    )

    print(
        f"batch size: {args.batch_size}"
    )

    print(
        f"context length: {args.context_length}"
    )

    print(
        "BF16 autocast: "
        f"{device.type == 'cuda' and use_amp}"
    )

    # --------------------------------------------------------
    # 创建模型
    # --------------------------------------------------------

    model = create_model(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        device=device,
    )

    # --------------------------------------------------------
    # 创建 AdamW
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        # 管理模型全部参数
        model.parameters(),

        # Benchmark 不关心真正训练收敛，
        # 这里只需要一个合理 learning rate
        lr=1e-3,
    )

    # --------------------------------------------------------
    # 创建随机 batch
    # --------------------------------------------------------

    inputs, targets = create_random_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=device,
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    (
        mean_ms,
        std_ms,
        timings,
        peak_memory_gb,
    ) = benchmark(

        model=model,

        optimizer=optimizer,

        inputs=inputs,

        targets=targets,

        mode=args.mode,

        warmup_steps=args.warmup_steps,

        measurement_steps=args.measurement_steps,

        device=device,

        use_amp=use_amp,
    )

    # --------------------------------------------------------
    # 输出结果
    # --------------------------------------------------------

    # 空行
    print()

    # 每一次 latency
    print(
        "timings (ms):",
        [
            round(x, 3)
            for x in timings
        ],
    )

    # 平均 latency
    print(
        f"mean: {mean_ms:.3f} ms"
    )

    # 标准差
    print(
        f"std:  {std_ms:.3f} ms"
    )

    # CUDA 环境下输出 peak VRAM
    if peak_memory_gb is not None:

        print(
            "peak GPU memory: "
            f"{peak_memory_gb:.2f} GB"
        )


# ============================================================
# 12. Python entry point
# ============================================================

# 当前文件被直接执行时
if __name__ == "__main__":

    # 从 main 开始
    main()