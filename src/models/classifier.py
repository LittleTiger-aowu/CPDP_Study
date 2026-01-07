import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union


class ClassifierHead(nn.Module):
    """
    通用分类头 (支持标准 Softmax 与 AM-Softmax).

    职责:
    1. 接收特征向量 [B, in_dim]
    2. (可选) 通过隐层投影提取特征
    3. 输出分类 Logits

    AM-Softmax 逻辑 (Scheme 1C):
    - Forward 输出 s * cos(theta)
    - 不接收 label，不在 Forward 中做 margin 惩罚
    - 仅输出 (logits, am_logits) tuple，供 train_step 处理
    """

    def __init__(
            self,
            in_dim: int,
            num_classes: int = 2,
            loss_type: str = "ce",
            hidden_dim: int = 0,
            dropout: float = 0.1,
            am_s: float = 30.0,
            am_m: float = 0.35
    ):
        """
        Args:
            in_dim: 输入特征维度.
            num_classes: 分类数量.
            loss_type: "ce" (CrossEntropy) 或 "am_softmax".
            hidden_dim: 若 > 0，在分类层前增加一个 Linear->GELU->Dropout 层.
            dropout: 隐层 dropout 概率.
            am_s: AM-Softmax 的缩放因子 (scale).
            am_m: AM-Softmax 的边界 (margin).
        """
        super().__init__()

        self.in_dim = in_dim
        self.num_classes = num_classes
        # [Fix 2] 配置归一化，提升容错
        self.loss_type = loss_type.lower()

        # 保存 AM-Softmax 参数供 train_step 读取
        self.am_s = float(am_s)
        self.am_m = float(am_m)

        # 1. 可选的隐层投影 (Feature Extractor)
        layers = []
        final_in_dim = in_dim

        if hidden_dim > 0:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            final_in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()

        # 2. 分类层 (Output Layer)
        if self.loss_type == "am_softmax":
            # AM-Softmax 不需要 Bias，权重需归一化
            # 我们用 Linear(bias=False) 来持有可学习参数 W
            self.fc = nn.Linear(final_in_dim, num_classes, bias=False)
        else:
            # 标准 CE 需要 Bias
            self.fc = nn.Linear(final_in_dim, num_classes, bias=True)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        if isinstance(self.fc, nn.Linear):
            nn.init.xavier_uniform_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.FloatTensor, labels: Union[torch.Tensor, None] = None):
        """
        Args:
            x: [Batch, in_dim]
            labels: [Batch] 训练时的标签张量，推理时为 None

        Returns:
            if loss_type == "ce":
                logits: [Batch, num_classes]
            if loss_type == "am_softmax":
                (logits, am_logits): Tuple, shapes [Batch, num_classes]
        """
        # 1. 防御性检查
        if x.dim() != 2:
            raise ValueError(f"ClassifierHead input must be 2D [B, Dim], got {x.dim()}D")

        if x.size(1) != self.in_dim:
            # 注意: 如果有 hidden_dim，这是指模块输入的校验
            raise ValueError(f"ClassifierHead input dim mismatch: expected {self.in_dim}, got {x.size(1)}")

        # 2. 特征提取 (若配置了 hidden_dim)
        h = self.feature_extractor(x)

        # 3. 分类计算
        if self.loss_type == "am_softmax":
            # ------------------------------------------------
            # AM-Softmax Logic (Scheme 1C)
            # ------------------------------------------------
            # 归一化输入特征 (h)
            # eps 防止除零
            h_norm = F.normalize(h, p=2, dim=1, eps=1e-12)

            # 归一化权重 (W)
            # self.fc.weight shape: [num_classes, final_in_dim]
            w_norm = F.normalize(self.fc.weight, p=2, dim=1, eps=1e-12)

            # 计算余弦相似度: cos = x_norm @ W_norm.T
            # F.linear(input, weight) = input @ weight.T
            cos_theta = F.linear(h_norm, w_norm)

            # 缩放: s * cos
            # 此时不做 margin，margin 在 loss 计算时基于 label 施加
            scaled_cosine = cos_theta * self.am_s

            # logits: 推理/统计使用 (不含 margin)
            # am_logits: 训练损失使用 (可加 margin)
            am_logits = scaled_cosine.clone()

            if self.training and labels is not None:
                if labels.dim() != 1:
                    raise ValueError(f"labels must be 1D [B], got {labels.shape}")
                if labels.size(0) != am_logits.size(0):
                    raise ValueError(
                        f"labels batch mismatch: {labels.size(0)} vs {am_logits.size(0)}"
                    )

                one_hot = torch.zeros_like(am_logits)
                one_hot.scatter_(1, labels.view(-1, 1), 1.0)
                am_logits = am_logits - one_hot * (self.am_m * self.am_s)

            return scaled_cosine, am_logits

        else:
            # ------------------------------------------------
            # Standard Cross Entropy Logic
            # ------------------------------------------------
            logits = self.fc(h)
            return logits
