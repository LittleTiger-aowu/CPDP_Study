"""Lightweight LoRA injection utilities for CodeBERT-style encoders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import torch
import torch.nn as nn


@dataclass
class LoraConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: Iterable[str] = ("query", "value")


class LoRALinear(nn.Module):
    """LoRA wrapper for Linear layers.

    Implements: W x + scale * (B(A(x)))
    where A: in_dim -> r, B: r -> out_dim.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA r must be > 0, got {r}")
        self.base = base
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(dropout)

        # LoRA weights
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)

        # Init: A ~ Kaiming, B ~ zeros (so initial delta is 0)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


def _matches_target(name: str, targets: Iterable[str]) -> bool:
    return any(t in name for t in targets)

def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _apply_to_encoder_layers(encoder: nn.Module, targets: Iterable[str], cfg: LoraConfig) -> int:
    """Traverse encoder layers and inject LoRA into matching Linear modules."""
    replaced = 0
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Linear) and _matches_target(name, targets):
            parent = encoder
            path = name.split(".")
            for part in path[:-1]:
                parent = getattr(parent, part)
            child_name = path[-1]
            base_layer = getattr(parent, child_name)
            if isinstance(base_layer, LoRALinear):
                continue
            _freeze_module(base_layer)
            setattr(parent, child_name, LoRALinear(base_layer, cfg.r, cfg.alpha, cfg.dropout))
            replaced += 1
    return replaced


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16,
    dropout: float = 0.05,
    target_modules: Union[Iterable[str], None] = None,
) -> nn.Module:
    """Inject LoRA modules into CodeBERT encoder layers.

    Freezes original Linear parameters and attaches a low-rank BA branch in
    parallel. Matching is done by module name substrings.
    """
    targets: List[str] = list(target_modules) if target_modules is not None else ["query", "value"]
    if not targets:
        return model

    cfg = LoraConfig(r=r, alpha=alpha, dropout=dropout, target_modules=targets)

    encoder = None

    # --- 修复逻辑开始 ---
    # 1. 针对 CodeBertEncoder 包装 AutoModel (RobertaModel) 的情况
    # 结构: wrapper.model -> RobertaModel -> .encoder
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        encoder = model.model.encoder

    # 2. 针对 CodeBertEncoder 包装 RobertaForSequenceClassification 的情况
    # 结构: wrapper.model -> RobertaFor... -> .roberta -> .encoder
    elif hasattr(model, "model") and hasattr(model.model, "roberta") and hasattr(model.model.roberta, "encoder"):
        encoder = model.model.roberta.encoder

    # 3. 针对直接传入 RobertaModel 的情况
    elif hasattr(model, "encoder"):
        encoder = model.encoder

    # 4. 针对直接传入 RobertaForSequenceClassification 的情况
    elif hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        encoder = model.roberta.encoder
    # --- 修复逻辑结束 ---

    if encoder is None:
        # 打印调试信息帮助定位
        structure_hint = f"Model type: {type(model)}\nAttrs: {list(model.__dict__.keys())}"
        if hasattr(model, "model"):
             structure_hint += f"\nInner Model type: {type(model.model)}\nInner Attrs: {list(model.model.__dict__.keys())}"

        raise ValueError(f"Unable to locate encoder on model for LoRA injection.\nDebug Info:\n{structure_hint}")

    _apply_to_encoder_layers(encoder, targets, cfg)
    return model