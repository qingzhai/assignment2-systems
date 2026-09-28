# CS336 Spring 2026 Assignment 2: Systems — Self-Study Notes & Experiments

> **个人自学仓库 / Personal self-study repository**
>
> 本仓库基于 Stanford CS336 Spring 2026 Assignment 2: Systems，用作我的系统优化学习路线和实验记录。它不是 Stanford 官方 solution。

我没有按正式课程作业要求逐项完成 Assignment 2，而是选择性学习、实现对我有帮助的内容。这里的代码不保证通过全部官方 tests；部分代码没有运行官方 tests。一些实现只用于理解机制，是 learning implementation，并非 production-quality implementation。Benchmark 和 profiling 的配置根据我的个人 GPU 环境调整，仓库中的结果也只代表对应实验环境。

[CS336-2笔记.pdf](./CS336-2笔记.pdf) 是我在整个学习过程中持续整理的个人学习笔记。

## 学习内容 / Learning Path

### 1. Benchmarking & Profiling

- Runtime benchmarking：[cs336_systems/benchmark.py](./cs336_systems/benchmark.py)
- Memory profiling：[cs336_systems/benchmark_memory.py](./cs336_systems/benchmark_memory.py)
- Activation checkpointing 的运行时间与显存实验：[cs336_systems/benchmark_checkpointing.py](./cs336_systems/benchmark_checkpointing.py)

### 2. PyTorch Attention

- 显式 PyTorch 算子的 eager attention，以及 forward、backward 和显存测量：[cs336_systems/pytorch_attention.py](./cs336_systems/pytorch_attention.py)
- `torch.compile` 对照实验：[cs336_systems/pytorch_attention_compile.py](./cs336_systems/pytorch_attention_compile.py)
- 保留的 CSV 结果：[profiles/pytorch_attention_full.csv](./profiles/pytorch_attention_full.csv)、[profiles/pytorch_attention_compile_full.csv](./profiles/pytorch_attention_compile_full.csv)

### 3. FlashAttention

- Online softmax 与分块计算的纯 PyTorch 学习实现：[cs336_systems/flash_attention.py](./cs336_systems/flash_attention.py)
- Triton forward/backward 学习实现：[cs336_systems/flash_attention_triton.py](./cs336_systems/flash_attention_triton.py)
- Triton runtime、memory benchmark：[cs336_systems/flash_attention_benchmark.py](./cs336_systems/flash_attention_benchmark.py)；保留的结果：[profiles/flash_attention_triton_d64.csv](./profiles/flash_attention_triton_d64.csv)

### 4. Distributed Training

- NCCL All-Reduce benchmark：[cs336_systems/ddp_allreduce_bench.py](./cs336_systems/ddp_allreduce_bench.py)；结果：[profiles/allreduce_bench.csv](./profiles/allreduce_bench.csv)
- 逐参数的 naive gradient synchronization 与将梯度 flatten 成较大通信单元的对照：[cs336_systems/ddp_sync_bench.py](./cs336_systems/ddp_sync_bench.py)；保留的结果：[profiles/ddp_sync_many_small.csv](./profiles/ddp_sync_many_small.csv)
- 通信与计算重叠实验：[cs336_systems/ddp_overlap_bench.py](./cs336_systems/ddp_overlap_bench.py)；结果：[profiles/ddp_overlap_bench.csv](./profiles/ddp_overlap_bench.csv)
- 批量运行脚本：[cs336_systems/run_all_ddp.sh](./cs336_systems/run_all_ddp.sh)；实验环境记录：[profiles/environment.txt](./profiles/environment.txt)、[profiles/gpu_topology.txt](./profiles/gpu_topology.txt)、[profiles/nvidia_smi.txt](./profiles/nvidia_smi.txt)

### 5. Optimizer State Sharding / FSDP

- [cs336_systems/sharded_optimizer.py](./cs336_systems/sharded_optimizer.py)：ZeRO-1 风格的优化器状态分片学习实现，用来理解优化器状态分片、参数同步以及 ZeRO / FSDP 相关机制；仓库中没有完整的 FSDP 实现。

## 仓库使用说明

项目使用 `uv` 管理依赖，配置见 [pyproject.toml](./pyproject.toml) 和 [uv.lock](./uv.lock)。`cs336-basics/` 是课程提供的 Assignment 1 语言模型实现，`cs336_systems/` 存放本仓库的实验代码，`profiles/` 存放选择保留的小型 CSV 结果与环境信息。运行脚本前请阅读相应文件的参数；GPU、CUDA、NCCL 或 Triton 相关实验需要匹配的硬件与软件环境。`profiles/environment.txt` 记录的是部分个人 GPU 实验的环境，并不表示所有脚本都在该环境下完整验证。

原始 profiling 报告、显存快照、临时 smoke-test CSV 和未用于最终分析的 `profiles/ddp_sync_bench.csv` 不作为仓库结果提交。

## 来源与原始作业说明

本仓库来源于 **Stanford CS336 Spring 2026 Assignment 2: Systems**。完整作业说明见 [cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf)。课程代码和原始说明保留在仓库中；许可证见 [LICENSE](./LICENSE)（MIT License）。

原始作业使用 `uv` 管理依赖。`cs336-basics/` 包含课程提供的 Assignment 1 语言模型实现及其 `pyproject.toml`；如果要改用自己的 Assignment 1 实现，原始作业说明允许替换该目录，或修改顶层 `pyproject.toml` 中的依赖路径。`cs336_systems/` 是 Assignment 2 的实现目录；本仓库已在其中加入上述个人实验。可用 `uv run python` 启动项目环境并尝试导入 `cs336_basics`。

原始作业还提供 [test_and_make_submission.sh](./test_and_make_submission.sh)，用于课程提交时安装依赖、运行 tests 并生成提交压缩包。这里保留该脚本作为上游作业的一部分；本自学仓库不把它视为已完成课程提交流程的证明。原始作业说明建议发现 handout 或代码问题时提出 GitHub issue 或 pull request。
