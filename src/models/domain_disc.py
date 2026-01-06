# src/models/domain_disc.py
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    """
    GRL (Gradient Reversal Layer) 的 Autograd 实现.

    Forward: 恒等映射
    Backward: grad * (-lambda)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


def grl(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """GRL 便捷调用接口."""
    return GradientReversal.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    """
    域判别器 (Domain Discriminator) for DANN.

    职责:
    1. 接收 grl_lambda 并执行梯度反转.
    2. 预测域标签 (Source vs Target).
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int = 1024,
            dropout: float = 0.1,
            **kwargs
    ):
        """
        Args:
            in_dim: 输入特征维度 (shared_dim).
            hidden_dim: 隐层维度.
            dropout: Dropout 概率.
            **kwargs: 预留参数，用于扩展。
        """
        super().__init__()
        self.in_dim = in_dim
        self.num_domains = 2

        # 1. 构建 MLP 网络 (硬性要求 #5)
        # Linear -> ReLU -> Dropout -> Linear
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_domains)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        """
        Args:
            feat: [Batch, in_dim] 共享特征
            grl_lambda: GRL 反转强度

        Returns:
            domain_logits: [Batch, num_domains]
        """
        # 1. 防御性编程 (硬性要求 #6)
        if feat.dim() != 2:
            raise ValueError(f"DomainDiscriminator expects 2D input [B, Dim], got {feat.dim()}D")

        if feat.size(1) != self.in_dim:
            raise ValueError(f"DomainDiscriminator input dim mismatch: expected {self.in_dim}, got {feat.size(1)}")

        # 2. 梯度反转 (GRL)
        feat_grl = grl(feat, grl_lambda)

        # 3. 前向计算 MLP
        domain_logits = self.net(feat_grl)

        return domain_logits
