# src/data/collate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any
import torch


@dataclass
class CollateConfig:
    """
    配置类：仅传递 Collator 必需的参数，解耦全局 Config。
    """
    pad_token_id: int = 1  # Token Padding 值
    label_key: str = "target"  # 缺陷标签键名
    domain_key: str = "domain"  # 域标签键名
    use_ast: bool = False  # 是否拼接 AST 图数据


def _pad_1d(seqs: List[torch.Tensor], pad_value: int, dtype=torch.long) -> torch.Tensor:
    """
    将一维 Tensor列表 Pad 到 Batch 内最大长度。
    [Update] 自动感知输入 Tensor 的 device。
    """
    if len(seqs) == 0:
        raise ValueError("Empty batch in _pad_1d")

    lengths = [x.numel() for x in seqs]
    max_len = max(lengths)
    batch_size = len(seqs)

    # 自动获取 device (通常是 cpu，但为了兼容性读取第一个 tensor 的 device)
    device = seqs[0].device

    # 预分配内存
    out = torch.full((batch_size, max_len), pad_value, dtype=dtype, device=device)

    for i, x in enumerate(seqs):
        length = lengths[i]
        if length > 0:
            out[i, :length] = x.to(dtype)

    return out


def _ensure_2xE(edge_index: torch.Tensor) -> torch.Tensor:
    """
    标准化 edge_index 为 [2, E] 的 LongTensor。
    """
    if edge_index is None:
        return torch.zeros((2, 0), dtype=torch.long)

    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(f"edge_index must be torch.Tensor, got {type(edge_index)}")

    if edge_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

    return edge_index.to(dtype=torch.long)


def _check_key_consistency(samples: List[Dict], key: str) -> bool:
    """
    [New] 检查 Batch 内所有样本是否统一包含或不包含某个 key。
    Returns: True if key exists in all samples, False if in none.
    Raises: KeyError if inconsistent.
    """
    has_key = key in samples[0]
    for i, s in enumerate(samples):
        if (key in s) != has_key:
            raise KeyError(
                f"Inconsistent key '{key}' in batch. "
                f"Sample[0] has it: {has_key}, but Sample[{i}] {'has' if key in s else 'missing'} it."
            )
    return has_key


def _collate_ast_graph(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    AST 图的 Disjoint Union 拼接。
    包含维度一致性检查与索引越界防御。
    """
    xs: List[torch.Tensor] = []
    eis: List[torch.Tensor] = []
    batch_vecs: List[torch.Tensor] = []

    node_offset = 0

    # [New] 维度一致性锚点 (以第一个样本为准)
    expected_x_dim = samples[0]["ast_x"].dim()

    for i, s in enumerate(samples):
        # 1. 完整性检查
        if "ast_x" not in s or "ast_edge_index" not in s:
            missing = [k for k in ["ast_x", "ast_edge_index"] if k not in s]
            raise KeyError(f"AST enabled but sample[{i}] missing keys: {missing}")

        x = s["ast_x"]
        ei = _ensure_2xE(s["ast_edge_index"])

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"ast_x must be Tensor, got {type(x)} (sample[{i}])")

        # [New] 维度一致性强校验
        if x.dim() != expected_x_dim:
            raise ValueError(
                f"Mixed ast_x dims in batch! Sample[0] is {expected_x_dim}D, "
                f"but Sample[{i}] is {x.dim()}D. All samples must match (all node_ids or all features)."
            )

        # 允许 1D (Node IDs) 或 2D (Features)
        if x.dim() not in (1, 2):
            raise ValueError(f"ast_x must be 1D or 2D, got {tuple(x.shape)} (sample[{i}])")

        num_nodes = x.size(0)

        # 2. 节点与边处理
        if num_nodes == 0:
            # [New] 强制清空边：没有节点就没有边
            ei = torch.zeros((2, 0), dtype=torch.long)
        else:
            # [New] 索引越界检查 (Debug 神器)
            # 仅在有边时检查，避免空 Tensor 报错
            if ei.size(1) > 0:
                max_idx = ei.max().item()
                min_idx = ei.min().item()
                if max_idx >= num_nodes or min_idx < 0:
                    raise ValueError(
                        f"AST Edge Index Out of Bounds in Sample[{i}]: "
                        f"Nodes={num_nodes}, but found edge referencing index {max_idx} (max) / {min_idx} (min). "
                        "Check your Parser!"
                    )

                # Offset Shift
                ei = ei + node_offset

        xs.append(x)
        eis.append(ei)

        # 3. 生成归属向量
        batch_vecs.append(torch.full((num_nodes,), i, dtype=torch.long))
        node_offset += num_nodes

    # 4. 拼接大图
    if len(xs) > 0:
        ast_x = torch.cat(xs, dim=0)
    else:
        ast_x = torch.zeros((0,), dtype=torch.long)  # 空 Batch 防御

    if len(eis) > 0:
        ast_edge_index = torch.cat(eis, dim=1)
    else:
        ast_edge_index = torch.zeros((2, 0), dtype=torch.long)

    if len(batch_vecs) > 0:
        ast_batch = torch.cat(batch_vecs, dim=0)
    else:
        ast_batch = torch.zeros((0,), dtype=torch.long)

    return {
        "ast_x": ast_x,
        "ast_edge_index": ast_edge_index,
        "ast_batch": ast_batch,
    }


class Collator:
    """
    可调用对象，作为 DataLoader 的 collate_fn。
    """

    def __init__(self, cfg: CollateConfig):
        self.cfg = cfg

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(samples) == 0:
            raise ValueError("Empty samples passed to collate_fn")

        # -------------------------
        # 1. 文本特征 (Padding)
        # -------------------------
        try:
            input_ids_list = [s["input_ids"] for s in samples]
            attn_mask_list = [s["attention_mask"] for s in samples]
        except KeyError as e:
            raise KeyError(f"Sample missing core text keys: {e}")

        input_ids = _pad_1d(input_ids_list, pad_value=self.cfg.pad_token_id, dtype=torch.long)
        attention_mask = _pad_1d(attn_mask_list, pad_value=0, dtype=torch.long)

        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # [New] All-or-None 检查 token_type_ids
        if _check_key_consistency(samples, "token_type_ids"):
            batch["token_type_ids"] = _pad_1d(
                [s["token_type_ids"] for s in samples],
                pad_value=0,
                dtype=torch.long
            )

        # -------------------------
        # 2. 标签数据 (Stack)
        # -------------------------
        # [New] All-or-None 检查 label
        if _check_key_consistency(samples, self.cfg.label_key):
            batch[self.cfg.label_key] = torch.tensor(
                [int(s[self.cfg.label_key]) for s in samples],
                dtype=torch.long
            )

        # [New] All-or-None 检查 domain
        if _check_key_consistency(samples, self.cfg.domain_key):
            batch[self.cfg.domain_key] = torch.tensor(
                [int(s[self.cfg.domain_key]) for s in samples],
                dtype=torch.long
            )

        if _check_key_consistency(samples, "loc"):
            batch["loc"] = torch.tensor(
                [float(s["loc"]) for s in samples],
                dtype=torch.float32
            )

        # -------------------------
        # 3. AST 图结构 (Disjoint Union)
        # -------------------------
        if self.cfg.use_ast:
            ast_batch_dict = _collate_ast_graph(samples)
            batch.update(ast_batch_dict)

        return batch
