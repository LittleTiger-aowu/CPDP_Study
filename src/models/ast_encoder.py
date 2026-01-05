import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCNLayer(nn.Module):
    """
    原生 PyTorch 实现的轻量级 GCN 层 (Mean Aggregation + Residual).

    Logic:
        1. Aggregate: h_agg = Mean_{j in N(i) U {i}} (h_j)
           (Includes self-loop in aggregation)
        2. Residual:  h_res = h_agg + h_i
           (Strengthens self-information to prevent over-smoothing)
        3. Update:    h_new = Dropout(ReLU(Linear(h_res)))
    """

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Args:
            x: [TotalNodes, in_dim]
            edge_index: [2, E_with_loops] (Source -> Target)
        """
        # 1. 准备数据
        # [Contract] edge_index[0]=Parent(Src), edge_index[1]=Child(Dst)
        # 语义: Child 节点聚合 Parent 节点的信息 (以及自己的信息)
        src, dst = edge_index[0], edge_index[1]

        # 2. 消息生成
        msg = x[src]  # [E, in_dim]

        # 3. 消息聚合 (Mean Aggregation)
        # 显式创建容器，避免 zeros_like 隐患，且 device/dtype 跟随输入
        out = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)

        # Scatter Add: out[dst] += msg
        out.index_add_(0, dst, msg)

        # 计算入度 (含自环) 用于平均
        ones = torch.ones(src.size(0), 1, device=x.device)
        deg = torch.zeros(x.size(0), 1, device=x.device)
        deg.index_add_(0, dst, ones)

        # 归一化
        deg = deg.clamp(min=1)
        out = out / deg

        # 4. 残差连接 (Residual Connection)
        # [Fix 2] 显式加回输入 x。虽然 Aggregation 含自环，但显式 Residual
        # 能显著缓解深层 GNN 的梯度消失和过平滑问题。
        out = out + x

        # 5. 变换与激活
        out = self.linear(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out


class ASTEncoder(nn.Module):
    """
    AST 图编码器 (Route B - Final Production Ready).

    Input Contract:
    - x: [TotalNodes] (Long) OR [TotalNodes, F] (Float)
    - edge_index: [2, E] (Long), Parent(0) -> Child(1)
    - batch: [TotalNodes] (Long)
    """

    def __init__(
            self,
            out_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            num_node_types: int = 300,
            feature_dim: int = 768,
            strict_id_check: bool = True  # [Fix 3] 脏数据防御开关
    ):
        super().__init__()

        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        self.strict_id_check = strict_id_check

        # 1. 输入适配
        self.embedding = nn.Embedding(num_node_types, hidden_dim)
        self.feat_proj = nn.Linear(feature_dim, hidden_dim)

        # 2. GNN 主干
        # 假设 hidden -> hidden 维度不变，便于残差连接
        self.gnn_layers = nn.ModuleList([
            SimpleGCNLayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 3. 输出层
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch, batch_size=None):
        """
        Args:
            batch_size (int, optional): 显式传入 Batch Size (必传，防止末尾空图 Bug)。
        """
        # -----------------------------------------------------------
        # 0. 边界条件处理
        # -----------------------------------------------------------
        if x.numel() == 0:
            B_safe = batch_size if batch_size is not None else 0
            return torch.zeros(B_safe, self.out_dim, device=x.device)

        # -----------------------------------------------------------
        # 1. 节点特征编码 & 脏数据防御
        # -----------------------------------------------------------
        if x.dim() == 1:
            # [Fix 3] 可配置的 ID 越界检查
            if self.strict_id_check:
                # 检查最大值和最小值，避免 Embedding lookup crash
                if x.max().item() >= self.num_node_types or x.min().item() < 0:
                    raise ValueError(
                        f"AST node_type_id out of range [0, {self.num_node_types}). "
                        f"Found max={x.max().item()}, min={x.min().item()}. "
                        "Check your Parser/Vocab or set strict_id_check=False."
                    )
            h = self.embedding(x)

        elif x.dim() == 2:
            h = self.feat_proj(x)
        else:
            raise ValueError(f"ASTEncoder input x must be 1D or 2D, got {x.dim()}D")

        # -----------------------------------------------------------
        # 2. 预处理: 添加自环 (Self-Loops)
        # -----------------------------------------------------------
        # 强制添加自环，确保每个节点在聚合时包含自身
        num_nodes = x.size(0)
        loop_index = torch.arange(num_nodes, device=x.device, dtype=torch.long)
        loop_index = torch.stack([loop_index, loop_index], dim=0)

        # 即使 edge_index 为空，也要有自环
        if edge_index.numel() == 0:
            edge_index_with_loop = loop_index
        else:
            edge_index_with_loop = torch.cat([edge_index, loop_index], dim=1)

        # -----------------------------------------------------------
        # 3. GNN 传播
        # -----------------------------------------------------------
        for layer in self.gnn_layers:
            h = layer(h, edge_index_with_loop)

        # -----------------------------------------------------------
        # 4. 全局池化 (Global Mean Pooling)
        # -----------------------------------------------------------
        # 必须使用传入的 batch_size 保证对齐
        if batch_size is None:
            # 仅做 fallback，生产环境应在 cpdp_model 保证传入
            batch_size = int(batch.max().item()) + 1

        graph_feat = torch.zeros(batch_size, self.hidden_dim, device=h.device, dtype=h.dtype)

        # Scatter Add
        graph_feat.index_add_(0, batch, h)

        # Mean Pooling Denominator
        counts = torch.bincount(batch, minlength=batch_size).to(dtype=h.dtype)
        counts = counts.unsqueeze(1).clamp(min=1)

        graph_feat = graph_feat / counts

        # -----------------------------------------------------------
        # 5. 最终投影
        # -----------------------------------------------------------
        out = self.out_proj(graph_feat)

        return out