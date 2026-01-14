# src/models/feature_split.py
import torch
import torch.nn as nn
from typing import Tuple


class FeatureSplit(nn.Module):
    """
    特征解耦模块 (Feature Splitter).

    职责:
    将输入的融合特征投影到两个独立的特征空间:
    1. Shared Space: 用于提取跨项目通用的语义特征 (供 Domain Disc 和 Classifier 使用).
    2. Private Space: 用于提取当前项目特有的语义特征 (仅供 Classifier 使用).

    结构:
    Input [B, in_dim]
       |
       +--> Shared Encoder (双层MLP+BN) -> Shared [B, shared_dim]
       |
       +--> Private Encoder (双层MLP+BN) -> Private [B, private_dim]
    """

    def __init__(
            self,
            in_dim: int,
            shared_dim: int,
            private_dim: int,
            dropout: float = 0.1
    ):
        """
        Args:
            in_dim: 输入特征维度 (e.g. CodeBERT dim + AST dim).
            shared_dim: 共享特征空间维度.
            private_dim: 私有特征空间维度.
            dropout: 投影后的 Dropout 概率.
        """
        super().__init__()

        self.in_dim = in_dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim

        # 1. 共享特征编码器 - 双层MLP + BatchNorm + ReLU
        self.shared_enc = nn.Sequential(
            nn.Linear(in_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),  # 关键：使用BN稳定训练
            nn.ReLU(),                   # 使用ReLU激活
            nn.Linear(shared_dim, shared_dim),  # 第二层增加非线性能力
            nn.Dropout(dropout)          # 保留Dropout防过拟合
        )

        # 2. 私有特征编码器 - 双层MLP + BatchNorm + ReLU
        self.private_enc = nn.Sequential(
            nn.Linear(in_dim, private_dim),
            nn.BatchNorm1d(private_dim),
            nn.ReLU(),
            nn.Linear(private_dim, private_dim),
            nn.Dropout(dropout)
        )

        # 3. 权重初始化 (Xavier Uniform)
        self._init_weights()

    def _init_weights(self):
        """应用 Xavier 初始化以加速收敛"""
        # 初始化共享编码器权重
        nn.init.xavier_uniform_(self.shared_enc[0].weight)
        nn.init.zeros_(self.shared_enc[0].bias)
        nn.init.xavier_uniform_(self.shared_enc[3].weight)
        nn.init.zeros_(self.shared_enc[3].bias)

        # 初始化私有编码器权重
        nn.init.xavier_uniform_(self.private_enc[0].weight)
        nn.init.zeros_(self.private_enc[0].bias)
        nn.init.xavier_uniform_(self.private_enc[3].weight)
        nn.init.zeros_(self.private_enc[3].bias)

    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        前向传播

        Args:
            x: 输入特征张量 [Batch, in_dim]

        Returns:
            (h_shared, h_private):
                h_shared: [Batch, shared_dim]
                h_private: [Batch, private_dim]
        """
        # 1. 防御性检查
        if x.dim() != 2:
            raise ValueError(f"FeatureSplit expects 2D input [Batch, Dim], got {x.dim()}D tensor.")

        if x.size(-1) != self.in_dim:
            raise ValueError(f"FeatureSplit expects input dim {self.in_dim}, got {x.size(-1)}.")

        # 2. 共享和私有特征提取
        h_s = self.shared_enc(x)
        h_p = self.private_enc(x)

        return h_s, h_p