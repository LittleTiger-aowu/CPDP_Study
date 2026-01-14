import torch
import torch.nn as nn
import torch.nn.functional as F


class BetterGCNLayer(nn.Module):
    """
    改进版 GCN:
    1. 支持双向流 (Bidirectional Flow) - 通过添加反向边
    2. 强化残差 (Stronger Residual) - 保留原始信息
    3. 使用 Sum Aggregation 避免信息稀释
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = dropout
        # 增加一个非线性变换，防止多层线性塌缩
        self.act = nn.GELU() 

    def forward(self, x, edge_index):
        """
        x: [N, D] - 节点特征
        edge_index: [2, E] - 边索引 [src, dst]
        """
        # 1. 转换为无向图 (Add Reverse Edges)
        # 这样 Parent <-> Child 信息互通
        src, dst = edge_index[0], edge_index[1]
        # 拼接反向边: (src->dst) + (dst->src)
        full_src = torch.cat([src, dst], dim=0)
        full_dst = torch.cat([dst, src], dim=0)

        # 2. 消息传递 (Sum Aggregation)
        # 显式使用 source 的特征
        msg = x[full_src] 
        
        # 3. 聚合 - Sum Aggregation (保留关键信息，避免平均稀释)
        out = torch.zeros_like(x)
        out.index_add_(0, full_dst, msg)

        # 4. 度归一化 (可选，采用 Root-n 归一化)
        deg = torch.zeros(x.size(0), 1, device=x.device)
        ones = torch.ones(full_src.size(0), 1, device=x.device)
        deg.index_add_(0, full_dst, ones)
        deg = deg.clamp(min=1).sqrt() # 1/sqrt(deg)
        out = out / deg

        # 5. 线性变换
        out = self.linear(out)
        
        # 6. 强残差连接 (Input + GNN_Output)
        # 保留原始特征信息
        out = out + x 
        
        out = self.act(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class ASTEncoder(nn.Module):
    """
    改进的 AST 图编码器
    """

    def __init__(
            self,
            out_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            num_node_types: int = 300,  # 确保这个数字 > 你的 vocab.json 大小
            feature_dim: int = 768,
            strict_id_check: bool = True
    ):
        super().__init__()

        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        self.strict_id_check = strict_id_check

        # AST 节点的 Embedding
        self.embedding = nn.Embedding(num_node_types, hidden_dim)
        # 如果输入是特征向量而不是ID，则使用投影
        self.feat_proj = nn.Linear(feature_dim, hidden_dim)

        # GNN 层
        self.gnn_layers = nn.ModuleList([
            BetterGCNLayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 输出层
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)  # 加一个 LayerNorm 稳定训练

    def forward(self, x, edge_index, batch, batch_size=None):
        """
        Args:
            x: [TotalNodes] (Long) OR [TotalNodes, F] (Float) - 节点特征
            edge_index: [2, E] (Long) - 边索引
            batch: [TotalNodes] (Long) - 批次索引
            batch_size: 批次大小
        """
        # 0. 边界防御
        if x.numel() == 0:
            B_safe = batch_size if batch_size is not None else 0
            return torch.zeros(B_safe, self.out_dim, device=x.device)

        # 1. 节点特征编码 & 脏数据防御
        if x.dim() == 1:
            # 处理节点类型ID
            if self.strict_id_check:
                if x.max().item() >= self.num_node_types or x.min().item() < 0:
                    raise ValueError(
                        f"AST node_type_id out of range [0, {self.num_node_types}). "
                        f"Found max={x.max().item()}, min={x.min().item()}. "
                        "Check your Parser/Vocab or set strict_id_check=False."
                    )
            h = self.embedding(x)
        elif x.dim() == 2:
            # 处理节点特征向量
            h = self.feat_proj(x)
        else:
            raise ValueError(f"ASTEncoder input x must be 1D or 2D, got {x.dim()}D")

        # 2. GNN 传播
        for layer in self.gnn_layers:
            h = layer(h, edge_index)

        # 3. 全局池化 (Global Mean Pooling)
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1

        # 使用 Sum Pooling + 归一化 (保留更多信息)
        graph_feat = torch.zeros(batch_size, self.hidden_dim, device=h.device, dtype=h.dtype)
        graph_feat.index_add_(0, batch, h)
        
        # 按每个图的节点数进行归一化
        counts = torch.bincount(batch, minlength=batch_size).to(dtype=h.dtype).unsqueeze(1).clamp(min=1)
        graph_feat = graph_feat / counts

        # 4. 输出投影
        out = self.out_proj(graph_feat)
        out = self.norm(out)  # 归一化，方便和 CodeBERT 拼接
        
        return out
