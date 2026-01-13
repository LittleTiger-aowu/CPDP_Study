from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoder_codebert import CodeBertEncoder


class AttentionPooling(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class FileSemanticEncoder(nn.Module):
    def __init__(
        self,
        pretrained_path: str,
        max_length: int = 512,
        pooling: str = "cls",
        window_pool: str = "attn",
    ) -> None:
        super().__init__()
        self.code_encoder = CodeBertEncoder(
            pretrained_path=pretrained_path,
            max_length=max_length,
            pooling=pooling,
        )
        self.output_dim = getattr(self.code_encoder, "output_dim", 768)
        self.window_pool = window_pool
        if window_pool == "attn":
            self.attn_pool = AttentionPooling(self.output_dim)
        elif window_pool == "mean":
            self.attn_pool = None
        else:
            raise ValueError(f"Unsupported window_pool: {window_pool}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        win_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.dim() != 3:
            raise ValueError("input_ids must be [B,W,L]")
        batch_size, windows, seq_len = input_ids.size()
        flat_ids = input_ids.view(batch_size * windows, seq_len)
        flat_mask = attention_mask.view(batch_size * windows, seq_len)

        pooled = self.code_encoder(input_ids=flat_ids, attention_mask=flat_mask)
        pooled = pooled.view(batch_size, windows, -1)

        if self.window_pool == "mean":
            win_mask_f = win_mask.float().unsqueeze(-1)
            denom = win_mask_f.sum(dim=1).clamp_min(1.0)
            return (pooled * win_mask_f).sum(dim=1) / denom

        return self.attn_pool(pooled, win_mask)


class FileStructureEncoder(nn.Module):
    def __init__(self, method_encoder: nn.Module, pool: str = "mean") -> None:
        super().__init__()
        self.method_encoder = method_encoder
        self.pool = pool

    def forward(self, method_inputs, method_mask: torch.Tensor) -> torch.Tensor:
        method_feats = self.method_encoder(method_inputs)
        if self.pool == "mean":
            mask_f = method_mask.float().unsqueeze(-1)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            return (method_feats * mask_f).sum(dim=1) / denom
        raise ValueError(f"Unsupported method pool: {self.pool}")
