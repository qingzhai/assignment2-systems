import math

import torch


class FlashAttentionPytorch(torch.autograd.Function):
    """
    Pure-PyTorch implementation of tiled FlashAttention.

    目的不是追求速度，而是：

    1. 用 PyTorch 把 FlashAttention forward 算法写清楚；
    2. 不 materialize 完整的 [S, S] attention matrix；
    3. forward 使用 online softmax；
    4. backward 使用 recomputation；
    5. backward 使用两遍 tile 扫描：
       - 第一遍计算 dK / dV
       - 第二遍计算 dQ

    输入：
        Q: [..., Nq, D]
        K: [..., Nk, D]
        V: [..., Nk, D]
        is_causal: bool

    输出：
        O: [..., Nq, D]
    """

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:

        # -----------------------------------------------------
        # 1. 基本 shape 检查
        # -----------------------------------------------------

        assert Q.shape[:-2] == K.shape[:-2] == V.shape[:-2]

        assert K.shape[-2] == V.shape[-2]

        assert Q.shape[-1] == K.shape[-1] == V.shape[-1]

        # Query sequence length。
        Nq = Q.shape[-2]

        # Key / Value sequence length。
        Nk = K.shape[-2]

        # Head dimension。
        D = Q.shape[-1]

        # Attention scaling。
        scale = 1.0 / math.sqrt(D)

        # -----------------------------------------------------
        # 2. 将所有 leading dimensions 合并成 batch
        # -----------------------------------------------------
        #
        # 这样既支持：
        #
        #   [B, S, D]
        #
        # 也支持：
        #
        #   [B, H, S, D]
        #
        # 最后统一处理为：
        #
        #   [B_flat, S, D]
        # -----------------------------------------------------

        leading_shape = Q.shape[:-2]

        Q_flat = Q.reshape(-1, Nq, D)
        K_flat = K.reshape(-1, Nk, D)
        V_flat = V.reshape(-1, Nk, D)

        B = Q_flat.shape[0]

        # -----------------------------------------------------
        # 3. 中间计算统一使用 FP32
        # -----------------------------------------------------
        #
        # 即使输入是 BF16：
        #
        # Q/K/V 输入本身仍然保持原 dtype；
        # 但是 m / l / acc 和 reference 计算用 FP32，
        # 数值稳定性更好。
        # -----------------------------------------------------

        Qf = Q_flat.float()
        Kf = K_flat.float()
        Vf = V_flat.float()

        # -----------------------------------------------------
        # 4. Tile size
        # -----------------------------------------------------

        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        # -----------------------------------------------------
        # 5. 输出 O
        # -----------------------------------------------------

        O = torch.empty(
            (B, Nq, D),
            device=Q.device,
            dtype=torch.float32,
        )

        # -----------------------------------------------------
        # 6. LogSumExp L
        # -----------------------------------------------------
        #
        # 每一个 query 保存一个：
        #
        # L_i = log sum_j exp(S_ij)
        #
        # shape:
        #
        # [B, Nq]
        #
        # backward 时可以利用它重新得到：
        #
        # P_ij = exp(S_ij - L_i)
        # -----------------------------------------------------

        L = torch.empty(
            (B, Nq),
            device=Q.device,
            dtype=torch.float32,
        )

        # =====================================================
        # 外层循环：遍历 Q tiles
        # =====================================================

        for q_start in range(0, Nq, Q_TILE_SIZE):

            q_end = min(
                q_start + Q_TILE_SIZE,
                Nq,
            )

            # 当前 Query tile：
            #
            # [B, Bq, D]
            Q_i = Qf[
                :,
                q_start:q_end,
                :,
            ]

            Bq = q_end - q_start

            # -------------------------------------------------
            # running max
            # -------------------------------------------------
            #
            # 每一个 query row 一个 max。
            #
            # [B, Bq]
            # -------------------------------------------------

            m_i = torch.full(
                (B, Bq),
                -float("inf"),
                device=Q.device,
                dtype=torch.float32,
            )

            # -------------------------------------------------
            # running denominator
            # -------------------------------------------------
            #
            # [B, Bq]
            # -------------------------------------------------

            l_i = torch.zeros(
                (B, Bq),
                device=Q.device,
                dtype=torch.float32,
            )

            # -------------------------------------------------
            # running output numerator
            # -------------------------------------------------
            #
            # [B, Bq, D]
            # -------------------------------------------------

            acc = torch.zeros(
                (B, Bq, D),
                device=Q.device,
                dtype=torch.float32,
            )

            # Query 的绝对位置。
            q_indices = torch.arange(
                q_start,
                q_end,
                device=Q.device,
            )

            # =================================================
            # 内层循环：遍历 K/V tiles
            # =================================================

            for k_start in range(0, Nk, K_TILE_SIZE):

                k_end = min(
                    k_start + K_TILE_SIZE,
                    Nk,
                )

                # causal 情况下：
                #
                # 如果当前 K tile 已经完全处于
                # 当前 Q tile 的未来，
                # 后面的 K tile 也全部是未来。
                #
                # 可以直接停止。
                if is_causal and k_start > q_end - 1:
                    break

                # 当前 K tile：
                #
                # [B, Bk, D]
                K_j = Kf[
                    :,
                    k_start:k_end,
                    :,
                ]

                # 当前 V tile：
                #
                # [B, Bk, D]
                V_j = Vf[
                    :,
                    k_start:k_end,
                    :,
                ]

                # ------------------------------------------------
                # 7. 当前 score tile
                # ------------------------------------------------
                #
                # Q_i:
                #   [B, Bq, D]
                #
                # K_j^T:
                #   [B, D, Bk]
                #
                # 得到：
                #
                #   [B, Bq, Bk]
                # ------------------------------------------------

                S_ij = torch.matmul(
                    Q_i,
                    K_j.transpose(-2, -1),
                )

                S_ij = S_ij * scale

                # ------------------------------------------------
                # 8. Causal mask
                # ------------------------------------------------

                if is_causal:

                    k_indices = torch.arange(
                        k_start,
                        k_end,
                        device=Q.device,
                    )

                    # shape:
                    #
                    # [Bq, Bk]
                    causal_mask = (
                        q_indices[:, None]
                        >=
                        k_indices[None, :]
                    )

                    S_ij = S_ij.masked_fill(
                        ~causal_mask.unsqueeze(0),
                        -float("inf"),
                    )

                # ------------------------------------------------
                # 9. 当前 tile 的 row max
                # ------------------------------------------------
                #
                # [B, Bq]
                # ------------------------------------------------

                m_tile = S_ij.max(
                    dim=-1,
                ).values

                # 新的 running max。
                m_new = torch.maximum(
                    m_i,
                    m_tile,
                )

                # ------------------------------------------------
                # 10. Rescale factor
                # ------------------------------------------------
                #
                # alpha =
                #
                # exp(m_old - m_new)
                #
                # 用来把旧的 l 和 acc
                # 转换到新的 softmax scale。
                # ------------------------------------------------

                alpha = torch.exp(
                    m_i - m_new
                )

                # ------------------------------------------------
                # 11. 当前 tile 的未归一化概率
                # ------------------------------------------------
                #
                # P_tilde =
                #
                # exp(S - m_new)
                #
                # m_new:
                #
                # [B, Bq]
                #
                # 扩一维：
                #
                # [B, Bq, 1]
                # ------------------------------------------------

                P_tilde = torch.exp(
                    S_ij
                    -
                    m_new.unsqueeze(-1)
                )

                # ------------------------------------------------
                # 12. 更新 denominator
                # ------------------------------------------------

                l_i = (
                    alpha * l_i
                    +
                    P_tilde.sum(dim=-1)
                )

                # ------------------------------------------------
                # 13. 更新 output numerator
                # ------------------------------------------------
                #
                # 旧 acc 需要一起 rescale。
                #
                # P_tilde:
                #   [B, Bq, Bk]
                #
                # V_j:
                #   [B, Bk, D]
                #
                # P_tilde @ V_j:
                #   [B, Bq, D]
                # ------------------------------------------------

                acc = (
                    alpha.unsqueeze(-1) * acc
                    +
                    torch.matmul(
                        P_tilde,
                        V_j,
                    )
                )

                # 更新 running max。
                m_i = m_new

            # =================================================
            # 当前 Q tile 所有 K/V 都处理完
            # =================================================

            # -------------------------------------------------
            # 14. 最终归一化
            # -------------------------------------------------
            #
            # O_i = acc / l
            # -------------------------------------------------

            O_i = (
                acc
                /
                l_i.unsqueeze(-1)
            )

            # 写入输出。
            O[
                :,
                q_start:q_end,
                :,
            ] = O_i

            # -------------------------------------------------
            # 15. 保存 logsumexp
            # -------------------------------------------------
            #
            # L = m + log(l)
            # -------------------------------------------------

            L[
                :,
                q_start:q_end,
            ] = (
                m_i
                +
                torch.log(l_i)
            )

        # -----------------------------------------------------
        # 16. 恢复原 shape
        # -----------------------------------------------------

        O = O.reshape(
            *leading_shape,
            Nq,
            D,
        )

        L = L.reshape(
            *leading_shape,
            Nq,
        )

        # 输出应该和 Q dtype 一致。
        O = O.to(Q.dtype)

        # -----------------------------------------------------
        # 17. 保存 backward 所需信息
        # -----------------------------------------------------

        ctx.save_for_backward(
            Q,
            K,
            V,
            O,
            L,
        )

        ctx.is_causal = is_causal

        ctx.q_tile_size = Q_TILE_SIZE
        ctx.k_tile_size = K_TILE_SIZE

        return O

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------
    @staticmethod
    def backward(
        ctx,
        dO: torch.Tensor,
    ):

        # 取回 forward 保存的数据。
        Q, K, V, O, L = ctx.saved_tensors

        is_causal = ctx.is_causal

        Q_TILE_SIZE = ctx.q_tile_size
        K_TILE_SIZE = ctx.k_tile_size

        # -----------------------------------------------------
        # Shape
        # -----------------------------------------------------

        Nq = Q.shape[-2]
        Nk = K.shape[-2]
        D = Q.shape[-1]

        scale = 1.0 / math.sqrt(D)

        leading_shape = Q.shape[:-2]

        # -----------------------------------------------------
        # Flatten leading dimensions
        # -----------------------------------------------------

        Qf = Q.reshape(
            -1,
            Nq,
            D,
        ).float()

        Kf = K.reshape(
            -1,
            Nk,
            D,
        ).float()

        Vf = V.reshape(
            -1,
            Nk,
            D,
        ).float()

        Of = O.reshape(
            -1,
            Nq,
            D,
        ).float()

        dOf = dO.reshape(
            -1,
            Nq,
            D,
        ).float()

        Lf = L.reshape(
            -1,
            Nq,
        ).float()

        B = Qf.shape[0]

        # -----------------------------------------------------
        # 1. Softmax backward row correction
        # -----------------------------------------------------
        #
        # D_i =
        #
        # sum_k O_ik * dO_ik
        #
        # shape:
        #
        # [B, Nq]
        # -----------------------------------------------------

        delta = (
            Of * dOf
        ).sum(
            dim=-1
        )

        # -----------------------------------------------------
        # Gradient buffers
        # -----------------------------------------------------

        dQ = torch.zeros_like(Qf)
        dK = torch.zeros_like(Kf)
        dV = torch.zeros_like(Vf)

        # =====================================================
        # PASS 1
        #
        # 每个 K/V tile 自己负责累加 dK / dV。
        #
        # 固定 K_j / V_j，
        # 遍历所有 Q_i。
        # =====================================================

        for k_start in range(0, Nk, K_TILE_SIZE):

            k_end = min(
                k_start + K_TILE_SIZE,
                Nk,
            )

            K_j = Kf[
                :,
                k_start:k_end,
                :,
            ]

            V_j = Vf[
                :,
                k_start:k_end,
                :,
            ]

            Bk = k_end - k_start

            # 当前 tile 的梯度 accumulator。
            dK_j = torch.zeros(
                (B, Bk, D),
                device=Q.device,
                dtype=torch.float32,
            )

            dV_j = torch.zeros(
                (B, Bk, D),
                device=Q.device,
                dtype=torch.float32,
            )

            k_indices = torch.arange(
                k_start,
                k_end,
                device=Q.device,
            )

            # 遍历所有 Q tiles。
            for q_start in range(
                0,
                Nq,
                Q_TILE_SIZE,
            ):

                q_end = min(
                    q_start + Q_TILE_SIZE,
                    Nq,
                )

                # causal：
                #
                # 当前 Q tile 完全在当前 K tile 之前，
                # 那么这些 key 全部不可见。
                if is_causal and q_end - 1 < k_start:
                    continue

                Q_i = Qf[
                    :,
                    q_start:q_end,
                    :,
                ]

                dO_i = dOf[
                    :,
                    q_start:q_end,
                    :,
                ]

                L_i = Lf[
                    :,
                    q_start:q_end,
                ]

                delta_i = delta[
                    :,
                    q_start:q_end,
                ]

                # ------------------------------------------------
                # 2. Recompute score tile
                # ------------------------------------------------

                S_ij = torch.matmul(
                    Q_i,
                    K_j.transpose(-2, -1),
                )

                S_ij = S_ij * scale

                # Causal mask。
                if is_causal:

                    q_indices = torch.arange(
                        q_start,
                        q_end,
                        device=Q.device,
                    )

                    causal_mask = (
                        q_indices[:, None]
                        >=
                        k_indices[None, :]
                    )

                    S_ij = S_ij.masked_fill(
                        ~causal_mask.unsqueeze(0),
                        -float("inf"),
                    )

                # ------------------------------------------------
                # 3. Recompute P tile
                # ------------------------------------------------
                #
                # P = exp(S - L)
                #
                # 不需要重新做完整 softmax。
                # ------------------------------------------------

                P_ij = torch.exp(
                    S_ij
                    -
                    L_i.unsqueeze(-1)
                )

                # ------------------------------------------------
                # 4. dP
                # ------------------------------------------------
                #
                # dP = dO @ V^T
                # ------------------------------------------------

                dP_ij = torch.matmul(
                    dO_i,
                    V_j.transpose(-2, -1),
                )

                # ------------------------------------------------
                # 5. dS
                # ------------------------------------------------
                #
                # dS =
                #
                # P * (dP - delta)
                # ------------------------------------------------

                dS_ij = (
                    P_ij
                    *
                    (
                        dP_ij
                        -
                        delta_i.unsqueeze(-1)
                    )
                )

                # ------------------------------------------------
                # 6. dK
                # ------------------------------------------------
                #
                # dK += dS^T @ Q * scale
                # ------------------------------------------------

                dK_j += (
                    torch.matmul(
                        dS_ij.transpose(-2, -1),
                        Q_i,
                    )
                    *
                    scale
                )

                # ------------------------------------------------
                # 7. dV
                # ------------------------------------------------
                #
                # dV += P^T @ dO
                # ------------------------------------------------

                dV_j += torch.matmul(
                    P_ij.transpose(-2, -1),
                    dO_i,
                )

            # 当前 K/V tile 的所有 Q contribution
            # 都累加完毕。
            dK[
                :,
                k_start:k_end,
                :,
            ] = dK_j

            dV[
                :,
                k_start:k_end,
                :,
            ] = dV_j

        # =====================================================
        # PASS 2
        #
        # 每一个 Q tile 自己负责 dQ。
        #
        # 固定 Q_i，
        # 遍历所有 K_j / V_j。
        # =====================================================

        for q_start in range(0, Nq, Q_TILE_SIZE):

            q_end = min(
                q_start + Q_TILE_SIZE,
                Nq,
            )

            Q_i = Qf[
                :,
                q_start:q_end,
                :,
            ]

            dO_i = dOf[
                :,
                q_start:q_end,
                :,
            ]

            L_i = Lf[
                :,
                q_start:q_end,
            ]

            delta_i = delta[
                :,
                q_start:q_end,
            ]

            Bq = q_end - q_start

            # 当前 Q tile 的 dQ accumulator。
            dQ_i = torch.zeros(
                (B, Bq, D),
                device=Q.device,
                dtype=torch.float32,
            )

            q_indices = torch.arange(
                q_start,
                q_end,
                device=Q.device,
            )

            # 遍历 K/V。
            for k_start in range(
                0,
                Nk,
                K_TILE_SIZE,
            ):

                k_end = min(
                    k_start + K_TILE_SIZE,
                    Nk,
                )

                if is_causal and k_start > q_end - 1:
                    break

                K_j = Kf[
                    :,
                    k_start:k_end,
                    :,
                ]

                V_j = Vf[
                    :,
                    k_start:k_end,
                    :,
                ]

                # ------------------------------------------------
                # 8. Recompute S
                # ------------------------------------------------

                S_ij = torch.matmul(
                    Q_i,
                    K_j.transpose(-2, -1),
                )

                S_ij = S_ij * scale

                # Causal mask。
                if is_causal:

                    k_indices = torch.arange(
                        k_start,
                        k_end,
                        device=Q.device,
                    )

                    causal_mask = (
                        q_indices[:, None]
                        >=
                        k_indices[None, :]
                    )

                    S_ij = S_ij.masked_fill(
                        ~causal_mask.unsqueeze(0),
                        -float("inf"),
                    )

                # ------------------------------------------------
                # 9. Recompute P
                # ------------------------------------------------

                P_ij = torch.exp(
                    S_ij
                    -
                    L_i.unsqueeze(-1)
                )

                # ------------------------------------------------
                # 10. dP
                # ------------------------------------------------

                dP_ij = torch.matmul(
                    dO_i,
                    V_j.transpose(-2, -1),
                )

                # ------------------------------------------------
                # 11. dS
                # ------------------------------------------------

                dS_ij = (
                    P_ij
                    *
                    (
                        dP_ij
                        -
                        delta_i.unsqueeze(-1)
                    )
                )

                # ------------------------------------------------
                # 12. dQ
                # ------------------------------------------------
                #
                # dQ += dS @ K * scale
                # ------------------------------------------------

                dQ_i += (
                    torch.matmul(
                        dS_ij,
                        K_j,
                    )
                    *
                    scale
                )

            # 写回 dQ。
            dQ[
                :,
                q_start:q_end,
                :,
            ] = dQ_i

        # -----------------------------------------------------
        # 13. 恢复原来的 shape 和 dtype
        # -----------------------------------------------------

        dQ = dQ.reshape_as(Q).to(Q.dtype)
        dK = dK.reshape_as(K).to(K.dtype)
        dV = dV.reshape_as(V).to(V.dtype)

        # forward 输入是：
        #
        # Q, K, V, is_causal
        #
        # 所以 backward 必须返回 4 个位置。
        #
        # bool 不需要梯度，所以最后是 None。
        return (
            dQ,
            dK,
            dV,
            None,
        )