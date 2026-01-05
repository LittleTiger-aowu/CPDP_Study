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
       +--> Linear -> Activation -> Dropout -> Shared [B, shared_dim]
       |
       +--> Linear -> Activation -> Dropout -> Private [B, private_dim]
    """

    def __init__(
            self,
            in_dim: int,
            shared_dim: int,
            private_dim: int,
            dropout: float = 0.1,
            activation: str = "gelu"
    ):
        """
        Args:
            in_dim: 输入特征维度 (e.g. CodeBERT dim + AST dim).
            shared_dim: 共享特征空间维度.
            private_dim: 私有特征空间维度.
            dropout: 投影后的 Dropout 概率.
            activation: 激活函数类型 ('gelu', 'relu', 'tanh').
        """
        super().__init__()

        self.in_dim = in_dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim

        # 1. 共享特征投影层
        self.shared_proj = nn.Linear(in_dim, shared_dim)

        # 2. 私有特征投影层
        self.private_proj = nn.Linear(in_dim, private_dim)

        # 3. 公共组件
        self.dropout = nn.Dropout(dropout)

        # 激活函数选择
        act_lower = activation.lower()
        if act_lower == "gelu":
            self.act = nn.GELU()
        elif act_lower == "relu":
            self.act = nn.ReLU()
        elif act_lower == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 4. 权重初始化 (Xavier Uniform)
        self._init_weights()

    def _init_weights(self):
        """应用 Xavier 初始化以加速收敛"""
        nn.init.xavier_uniform_(self.shared_proj.weight)
        nn.init.zeros_(self.shared_proj.bias)

        nn.init.xavier_uniform_(self.private_proj.weight)
        nn.init.zeros_(self.private_proj.bias)

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

        # 2. 共享分支计算
        h_s = self.shared_proj(x)
        h_s = self.act(h_s)
        h_s = self.dropout(h_s)

        # 3. 私有分支计算
        h_p = self.private_proj(x)
        h_p = self.act(h_p)
        h_p = self.dropout(h_p)

        return h_s, h_p