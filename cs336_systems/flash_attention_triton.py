import math

import torch
import triton
import triton.language as tl


# ============================================================
# Triton FlashAttention: Forward + Backward
#
# 学习版目标：
# 1. 用 Triton 实现 tiled FlashAttention forward。
# 2. forward 使用 online softmax，不 materialize 完整 [S, S] 矩阵。
# 3. backward 不保存完整 P，而是利用 LSE 重新计算局部 P tile。
# 4. backward 分三步：
#       a) delta = rowsum(O * dO)
#       b) 一个 program 负责一个 K/V tile，计算 dK / dV
#       c) 一个 program 负责一个 Q tile，计算 dQ
#
# 当前支持：
#     Q/K/V: [B, S, D]
#     dtype: float16 / bfloat16
#     D: 16 / 32 / 64 / 128
#     causal / non-causal
#
# 说明：
#     这是学习版，不是极限性能版。
# ============================================================


@triton.jit
def _flash_attention_fwd_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    stride_qb,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_os,
    stride_od,
    sm_scale,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """一个 program 负责一个 batch 中的一块 Q，并扫描全部 K/V tiles。"""

    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # ------------------------------
    # Load Q tile
    # ------------------------------
    q_ptrs = (
        Q
        + pid_b * stride_qb
        + offs_m[:, None] * stride_qs
        + offs_d[None, :] * stride_qd
    )

    q_mask = (
        (offs_m[:, None] < N_CTX)
        & (offs_d[None, :] < HEAD_DIM)
    )

    q = tl.load(
        q_ptrs,
        mask=q_mask,
        other=0.0,
    )

    # ------------------------------
    # Online Softmax 状态
    # ------------------------------
    row_valid = offs_m < N_CTX

    m_i = tl.where(
        row_valid,
        -float("inf"),
        0.0,
    ).to(tl.float32)

    l_i = tl.where(
        row_valid,
        0.0,
        1.0,
    ).to(tl.float32)

    acc = tl.zeros(
        (BLOCK_M, BLOCK_D),
        dtype=tl.float32,
    )

    # ------------------------------
    # 扫描全部 K/V tiles
    # ------------------------------
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # 直接按 K^T 的形状 [D, BLOCK_N] load。
        k_ptrs = (
            K
            + pid_b * stride_kb
            + offs_d[:, None] * stride_kd
            + offs_n[None, :] * stride_ks
        )

        k_mask = (
            (offs_d[:, None] < HEAD_DIM)
            & (offs_n[None, :] < N_CTX)
        )

        k = tl.load(
            k_ptrs,
            mask=k_mask,
            other=0.0,
        )

        # S_ij = Q_i K_j^T / sqrt(D)
        qk = tl.dot(q, k)
        qk *= sm_scale

        valid_mask = (
            (offs_m[:, None] < N_CTX)
            & (offs_n[None, :] < N_CTX)
        )

        if IS_CAUSAL:
            valid_mask = (
                valid_mask
                & (offs_m[:, None] >= offs_n[None, :])
            )

        qk = tl.where(
            valid_mask,
            qk,
            -float("inf"),
        )

        # Online Softmax。
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)

        p = tl.exp(
            qk - m_new[:, None]
        )

        l_new = (
            alpha * l_i
            + tl.sum(p, axis=1)
        )

        # Load V tile: [BLOCK_N, D]
        v_ptrs = (
            V
            + pid_b * stride_vb
            + offs_n[:, None] * stride_vs
            + offs_d[None, :] * stride_vd
        )

        v_mask = (
            (offs_n[:, None] < N_CTX)
            & (offs_d[None, :] < HEAD_DIM)
        )

        v = tl.load(
            v_ptrs,
            mask=v_mask,
            other=0.0,
        )

        # 旧 acc 也要随着新的 running max 一起 rescale。
        acc *= alpha[:, None]

        # 当前 tile 对输出 numerator 的贡献。
        p_for_dot = p.to(v.dtype)
        acc += tl.dot(p_for_dot, v)

        m_i = m_new
        l_i = l_new

    # O_i = acc / l_i
    acc = acc / l_i[:, None]

    # ------------------------------
    # Store O
    # ------------------------------
    o_ptrs = (
        O
        + pid_b * stride_ob
        + offs_m[:, None] * stride_os
        + offs_d[None, :] * stride_od
    )

    o_mask = (
        (offs_m[:, None] < N_CTX)
        & (offs_d[None, :] < HEAD_DIM)
    )

    tl.store(
        o_ptrs,
        acc,
        mask=o_mask,
    )

    # LSE_i = log(sum_j exp(S_ij)) = m_i + log(l_i)
    lse = m_i + tl.log(l_i)

    lse_ptrs = (
        LSE
        + pid_b * N_CTX
        + offs_m
    )

    tl.store(
        lse_ptrs,
        lse,
        mask=offs_m < N_CTX,
    )


@triton.jit
def _flash_attention_bwd_delta_kernel(
    O,
    DO,
    DELTA,
    stride_ob,
    stride_os,
    stride_od,
    stride_dob,
    stride_dos,
    stride_dod,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """delta_i = sum_d O[i,d] * dO[i,d]。一个 program 负责一行。"""

    row = tl.program_id(0)

    pid_b = row // N_CTX
    pid_m = row % N_CTX

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    o_ptrs = (
        O
        + pid_b * stride_ob
        + pid_m * stride_os
        + offs_d * stride_od
    )

    do_ptrs = (
        DO
        + pid_b * stride_dob
        + pid_m * stride_dos
        + offs_d * stride_dod
    )

    o = tl.load(
        o_ptrs,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)

    do = tl.load(
        do_ptrs,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)

    delta = tl.sum(
        o * do,
        axis=0,
    )

    delta_ptr = (
        DELTA
        + pid_b * N_CTX
        + pid_m
    )

    tl.store(delta_ptr, delta)


@triton.jit
def _flash_attention_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    stride_qb,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vd,
    stride_dob,
    stride_dos,
    stride_dod,
    stride_dkb,
    stride_dks,
    stride_dkd,
    stride_dvb,
    stride_dvs,
    stride_dvd,
    sm_scale,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    一个 program 负责一个 batch 中的一块 K/V。
    扫描所有 Q tiles，最后独占写回对应 dK/dV tile。
    """

    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = (
        pid_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )

    offs_d = tl.arange(0, BLOCK_D)

    kv_mask = (
        (offs_n[:, None] < N_CTX)
        & (offs_d[None, :] < HEAD_DIM)
    )

    # 固定 K_j。
    k_ptrs = (
        K
        + pid_b * stride_kb
        + offs_n[:, None] * stride_ks
        + offs_d[None, :] * stride_kd
    )

    k = tl.load(
        k_ptrs,
        mask=kv_mask,
        other=0.0,
    )

    # 固定 V_j。
    v_ptrs = (
        V
        + pid_b * stride_vb
        + offs_n[:, None] * stride_vs
        + offs_d[None, :] * stride_vd
    )

    v = tl.load(
        v_ptrs,
        mask=kv_mask,
        other=0.0,
    )

    dk = tl.zeros(
        (BLOCK_N, BLOCK_D),
        dtype=tl.float32,
    )

    dv = tl.zeros(
        (BLOCK_N, BLOCK_D),
        dtype=tl.float32,
    )

    # 扫描所有 Q tiles。
    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m = (
            start_m
            + tl.arange(0, BLOCK_M)
        )

        q_mask = (
            (offs_m[:, None] < N_CTX)
            & (offs_d[None, :] < HEAD_DIM)
        )

        q_ptrs = (
            Q
            + pid_b * stride_qb
            + offs_m[:, None] * stride_qs
            + offs_d[None, :] * stride_qd
        )

        q = tl.load(
            q_ptrs,
            mask=q_mask,
            other=0.0,
        )

        do_ptrs = (
            DO
            + pid_b * stride_dob
            + offs_m[:, None] * stride_dos
            + offs_d[None, :] * stride_dod
        )

        do = tl.load(
            do_ptrs,
            mask=q_mask,
            other=0.0,
        )

        lse_ptrs = (
            LSE
            + pid_b * N_CTX
            + offs_m
        )

        lse_i = tl.load(
            lse_ptrs,
            mask=offs_m < N_CTX,
            other=0.0,
        )

        delta_ptrs = (
            DELTA
            + pid_b * N_CTX
            + offs_m
        )

        delta_i = tl.load(
            delta_ptrs,
            mask=offs_m < N_CTX,
            other=0.0,
        )

        # Recompute S_ij。
        qk = tl.dot(
            q,
            tl.trans(k),
        )
        qk *= sm_scale

        valid_mask = (
            (offs_m[:, None] < N_CTX)
            & (offs_n[None, :] < N_CTX)
        )

        if IS_CAUSAL:
            valid_mask = (
                valid_mask
                & (offs_m[:, None] >= offs_n[None, :])
            )

        qk = tl.where(
            valid_mask,
            qk,
            -float("inf"),
        )

        # Recompute P_ij = exp(S_ij - LSE_i)。
        p = tl.exp(
            qk - lse_i[:, None]
        )

        # dP = dO @ V^T。
        dp = tl.dot(
            do,
            tl.trans(v),
        )

        # dS = P * (dP - delta)。
        ds = (
            p
            * (
                dp
                - delta_i[:, None]
            )
        )

        # dK += dS^T @ Q * scale。
        ds_for_dot = ds.to(q.dtype)

        dk += (
            tl.dot(
                tl.trans(ds_for_dot),
                q,
            )
            * sm_scale
        )

        # dV += P^T @ dO。
        p_for_dot = p.to(do.dtype)

        dv += tl.dot(
            tl.trans(p_for_dot),
            do,
        )

    dk_ptrs = (
        DK
        + pid_b * stride_dkb
        + offs_n[:, None] * stride_dks
        + offs_d[None, :] * stride_dkd
    )

    dv_ptrs = (
        DV
        + pid_b * stride_dvb
        + offs_n[:, None] * stride_dvs
        + offs_d[None, :] * stride_dvd
    )

    tl.store(
        dk_ptrs,
        dk,
        mask=kv_mask,
    )

    tl.store(
        dv_ptrs,
        dv,
        mask=kv_mask,
    )


@triton.jit
def _flash_attention_bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    stride_qb,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vd,
    stride_dob,
    stride_dos,
    stride_dod,
    stride_dqb,
    stride_dqs,
    stride_dqd,
    sm_scale,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    一个 program 负责一个 batch 中的一块 Q。
    扫描全部 K/V tiles，最后独占写回对应 dQ tile。
    """

    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = (
        pid_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )

    offs_d = tl.arange(0, BLOCK_D)

    q_mask = (
        (offs_m[:, None] < N_CTX)
        & (offs_d[None, :] < HEAD_DIM)
    )

    q_ptrs = (
        Q
        + pid_b * stride_qb
        + offs_m[:, None] * stride_qs
        + offs_d[None, :] * stride_qd
    )

    q = tl.load(
        q_ptrs,
        mask=q_mask,
        other=0.0,
    )

    do_ptrs = (
        DO
        + pid_b * stride_dob
        + offs_m[:, None] * stride_dos
        + offs_d[None, :] * stride_dod
    )

    do = tl.load(
        do_ptrs,
        mask=q_mask,
        other=0.0,
    )

    lse_ptrs = (
        LSE
        + pid_b * N_CTX
        + offs_m
    )

    lse_i = tl.load(
        lse_ptrs,
        mask=offs_m < N_CTX,
        other=0.0,
    )

    delta_ptrs = (
        DELTA
        + pid_b * N_CTX
        + offs_m
    )

    delta_i = tl.load(
        delta_ptrs,
        mask=offs_m < N_CTX,
        other=0.0,
    )

    dq = tl.zeros(
        (BLOCK_M, BLOCK_D),
        dtype=tl.float32,
    )

    # 扫描所有 K/V tiles。
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = (
            start_n
            + tl.arange(0, BLOCK_N)
        )

        kv_mask = (
            (offs_n[:, None] < N_CTX)
            & (offs_d[None, :] < HEAD_DIM)
        )

        k_ptrs = (
            K
            + pid_b * stride_kb
            + offs_n[:, None] * stride_ks
            + offs_d[None, :] * stride_kd
        )

        k = tl.load(
            k_ptrs,
            mask=kv_mask,
            other=0.0,
        )

        v_ptrs = (
            V
            + pid_b * stride_vb
            + offs_n[:, None] * stride_vs
            + offs_d[None, :] * stride_vd
        )

        v = tl.load(
            v_ptrs,
            mask=kv_mask,
            other=0.0,
        )

        # Recompute score。
        qk = tl.dot(
            q,
            tl.trans(k),
        )
        qk *= sm_scale

        valid_mask = (
            (offs_m[:, None] < N_CTX)
            & (offs_n[None, :] < N_CTX)
        )

        if IS_CAUSAL:
            valid_mask = (
                valid_mask
                & (offs_m[:, None] >= offs_n[None, :])
            )

        qk = tl.where(
            valid_mask,
            qk,
            -float("inf"),
        )

        # Recompute P。
        p = tl.exp(
            qk - lse_i[:, None]
        )

        # dP = dO @ V^T。
        dp = tl.dot(
            do,
            tl.trans(v),
        )

        # dS = P * (dP - delta)。
        ds = (
            p
            * (
                dp
                - delta_i[:, None]
            )
        )

        # dQ += dS @ K * scale。
        ds_for_dot = ds.to(k.dtype)

        dq += (
            tl.dot(
                ds_for_dot,
                k,
            )
            * sm_scale
        )

    dq_ptrs = (
        DQ
        + pid_b * stride_dqb
        + offs_m[:, None] * stride_dqs
        + offs_d[None, :] * stride_dqd
    )

    tl.store(
        dq_ptrs,
        dq,
        mask=q_mask,
    )


# ============================================================
# Python wrappers
# ============================================================

def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    """检查当前学习版支持的输入。"""

    assert q.is_cuda
    assert k.is_cuda
    assert v.is_cuda

    assert q.ndim == 3
    assert k.ndim == 3
    assert v.ndim == 3

    assert q.shape == k.shape == v.shape

    _, _, D = q.shape

    assert D in {
        16,
        32,
        64,
        128,
    }, "当前学习版只支持 head_dim = 16/32/64/128"

    assert q.dtype in {
        torch.float16,
        torch.bfloat16,
    }

    assert k.dtype == q.dtype
    assert v.dtype == q.dtype


def flash_attention_triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    return_lse: bool = False,
):
    """Triton FlashAttention forward。当前支持 [B, S, D]。"""

    _validate_inputs(q, k, v)

    B, S, D = q.shape

    o = torch.empty_like(q)

    lse = torch.empty(
        (B, S),
        device=q.device,
        dtype=torch.float32,
    )

    # D=128 时一个 64x64 tile 对共享内存压力比较大。
    # 因此减小 K/V tile。
    BLOCK_M = 64

    if D == 128:
        BLOCK_N = 32
    else:
        BLOCK_N = 64

    BLOCK_D = D

    # 一个 program = 一个 batch + 一个 Q tile。
    grid = (
        triton.cdiv(S, BLOCK_M),
        B,
    )

    _flash_attention_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,

        q.stride(0),
        q.stride(1),
        q.stride(2),

        k.stride(0),
        k.stride(1),
        k.stride(2),

        v.stride(0),
        v.stride(1),
        v.stride(2),

        o.stride(0),
        o.stride(1),
        o.stride(2),

        1.0 / math.sqrt(D),

        N_CTX=S,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal,

        num_warps=4,
        num_stages=1,
    )

    if return_lse:
        return o, lse

    return o


def flash_attention_triton_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    is_causal: bool,
):
    """Triton FlashAttention backward，返回 dq/dk/dv。"""

    _validate_inputs(q, k, v)

    B, S, D = q.shape

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    delta = torch.empty(
        (B, S),
        device=q.device,
        dtype=torch.float32,
    )

    BLOCK_M = 64

    if D == 128:
        BLOCK_N = 32
    else:
        BLOCK_N = 64

    BLOCK_D = D

    # ------------------------------
    # Step 1: delta
    # ------------------------------
    delta_grid = (B * S,)

    _flash_attention_bwd_delta_kernel[delta_grid](
        o,
        do,
        delta,

        o.stride(0),
        o.stride(1),
        o.stride(2),

        do.stride(0),
        do.stride(1),
        do.stride(2),

        N_CTX=S,
        HEAD_DIM=D,
        BLOCK_D=BLOCK_D,

        num_warps=1,
    )

    # ------------------------------
    # Step 2: dK / dV
    # ------------------------------
    dkdv_grid = (
        triton.cdiv(S, BLOCK_N),
        B,
    )

    _flash_attention_bwd_dkdv_kernel[dkdv_grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk,
        dv,

        q.stride(0),
        q.stride(1),
        q.stride(2),

        k.stride(0),
        k.stride(1),
        k.stride(2),

        v.stride(0),
        v.stride(1),
        v.stride(2),

        do.stride(0),
        do.stride(1),
        do.stride(2),

        dk.stride(0),
        dk.stride(1),
        dk.stride(2),

        dv.stride(0),
        dv.stride(1),
        dv.stride(2),

        1.0 / math.sqrt(D),

        N_CTX=S,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal,

        num_warps=4,
        num_stages=1,
    )

    # ------------------------------
    # Step 3: dQ
    # ------------------------------
    dq_grid = (
        triton.cdiv(S, BLOCK_M),
        B,
    )

    _flash_attention_bwd_dq_kernel[dq_grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,

        q.stride(0),
        q.stride(1),
        q.stride(2),

        k.stride(0),
        k.stride(1),
        k.stride(2),

        v.stride(0),
        v.stride(1),
        v.stride(2),

        do.stride(0),
        do.stride(1),
        do.stride(2),

        dq.stride(0),
        dq.stride(1),
        dq.stride(2),

        1.0 / math.sqrt(D),

        N_CTX=S,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal,

        num_warps=4,
        num_stages=1,
    )

    return dq, dk, dv


# ============================================================
# PyTorch autograd interface
# ============================================================

class FlashAttentionTriton(torch.autograd.Function):
    """把 Triton forward/backward 接入 PyTorch 自动求导。"""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        is_causal=False,
    ):
        o, lse = flash_attention_triton_forward(
            q,
            k,
            v,
            is_causal=is_causal,
            return_lse=True,
        )

        # 不保存完整 P，只保存 backward 真正需要的信息。
        ctx.save_for_backward(
            q,
            k,
            v,
            o,
            lse,
        )

        ctx.is_causal = is_causal

        return o

    @staticmethod
    def backward(
        ctx,
        do,
    ):
        q, k, v, o, lse = ctx.saved_tensors

        dq, dk, dv = flash_attention_triton_backward(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            is_causal=ctx.is_causal,
        )

        return (
            dq,
            dk,
            dv,
            None,
        )


def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
):
    """用户侧调用接口。"""

    return FlashAttentionTriton.apply(
        q,
        k,
        v,
        is_causal,
    )


# ============================================================
# PyTorch reference
# ============================================================

def pytorch_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
):
    """
    普通显式 PyTorch Attention。

    只用来验证 Triton 版本的 forward / backward 正确性。
    """

    D = q.shape[-1]

    scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    )

    scores = (
        scores
        / math.sqrt(D)
    )

    if is_causal:
        S = q.shape[-2]

        mask = torch.ones(
            S,
            S,
            device=q.device,
            dtype=torch.bool,
        ).triu(diagonal=1)

        scores = scores.masked_fill(
            mask,
            -float("inf"),
        )

    probs = torch.softmax(
        scores,
        dim=-1,
    )

    return torch.matmul(
        probs,
        v,
    )


# ============================================================
# Forward + Backward sanity check
# ============================================================

def _run_sanity_check(
    is_causal: bool,
):
    """比较 Triton 与普通 PyTorch 的 O/dQ/dK/dV。"""

    torch.manual_seed(0)

    B = 2
    S = 256
    D = 64

    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(
        B,
        S,
        D,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    k = torch.randn(
        B,
        S,
        D,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    v = torch.randn(
        B,
        S,
        D,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    # Reference 使用独立的 leaf tensors。
    q_ref = (
        q.detach()
        .clone()
        .requires_grad_(True)
    )

    k_ref = (
        k.detach()
        .clone()
        .requires_grad_(True)
    )

    v_ref = (
        v.detach()
        .clone()
        .requires_grad_(True)
    )

    # 固定上游梯度，让两边 backward 完全可比。
    do = torch.randn(
        B,
        S,
        D,
        device=device,
        dtype=dtype,
    )

    out = flash_attention_triton(
        q,
        k,
        v,
        is_causal=is_causal,
    )

    out_ref = pytorch_attention_reference(
        q_ref,
        k_ref,
        v_ref,
        is_causal=is_causal,
    )

    torch.cuda.synchronize()

    out.backward(do)
    out_ref.backward(do)

    torch.cuda.synchronize()

    output_error = (
        out.float()
        - out_ref.float()
    ).abs().max().item()

    dq_error = (
        q.grad.float()
        - q_ref.grad.float()
    ).abs().max().item()

    dk_error = (
        k.grad.float()
        - k_ref.grad.float()
    ).abs().max().item()

    dv_error = (
        v.grad.float()
        - v_ref.grad.float()
    ).abs().max().item()

    print()
    print("=" * 70)
    print(f"causal = {is_causal}")
    print(f"forward max abs error: {output_error:.6f}")
    print(f"dQ max abs error:      {dq_error:.6f}")
    print(f"dK max abs error:      {dk_error:.6f}")
    print(f"dV max abs error:      {dv_error:.6f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    print("Triton FlashAttention Forward + Backward sanity check")
    print(f"GPU: {torch.cuda.get_device_name()}")

    _run_sanity_check(
        is_causal=False,
    )

    _run_sanity_check(
        is_causal=True,
    )
