# src/data/dataset.py
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

# 初始化 Logger
logger = logging.getLogger(__name__)


class CPDPDataset(Dataset):
    """
    CPDP 跨项目缺陷预测数据集 (Route B: AST Graph Support).

    职责:
    1. 读取 JSONL 数据 (Code + AST + Label).
    2. Tokenize 源代码 (CodeBERT).
    3. 转换 AST 为 PyG 风格的 Tensors (x, edge_index).
    4. 返回单样本 Dict (不含 Batch Padding).
    """

    def __init__(
            self,
            data_path: str,
            tokenizer,
            max_length: int = 224,
            label_key: str = "target",
            domain_key: str = "domain",
            code_key: str = "code",
            code_key_fallbacks: Optional[List[str]] = None,
            domain_key_fallbacks: Optional[List[str]] = None,
            domain_map: Optional[Dict[str, int]] = None,
            default_domain_value: Optional[int] = None,
            use_ast: bool = True,
            strict_graph_check: bool = True,
            in_memory: bool = True,
            ast_cache_dir: Optional[str] = None,
            ast_cache_fallback_to_jsonl: bool = True,
    ):
        """
        Args:
            data_path: JSONL 文件路径.
            tokenizer: HuggingFace Tokenizer 实例.
            max_length: 文本截断长度.
            label_key: 缺陷标签键名 (JSONL中的key).
            domain_key: 域标签键名 (JSONL中的key).
            code_key: 源代码键名 (JSONL中的key).
            code_key_fallbacks: 源代码备用键名列表.
            domain_key_fallbacks: 域标签备用键名列表.
            domain_map: 域标签字符串映射表 (例如 {"FFmpeg": 0, "QEMU": 1}).
            default_domain_value: 当 domain_key 不存在时使用的默认域标签.
            use_ast: 是否处理 AST 数据.
            strict_graph_check: 是否对 AST 边索引做越界检查.
            in_memory: 是否一次性加载到内存 (建议 True 以加速).
            ast_cache_dir: 离线 AST 缓存目录 (含 .pt 文件).
            ast_cache_fallback_to_jsonl: 缓存缺失时是否退回 JSONL AST 字段.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_key = label_key
        self.domain_key = domain_key
        fallback_code_keys = code_key_fallbacks or ["func"]
        self.code_keys = [code_key] + [k for k in fallback_code_keys if k != code_key]
        fallback_domain_keys = domain_key_fallbacks or []
        self.domain_keys = [domain_key] + [k for k in fallback_domain_keys if k != domain_key]
        self.domain_map = domain_map or {}
        self.default_domain_value = default_domain_value
        self.use_ast = use_ast
        self.strict_graph_check = strict_graph_check
        self.ast_cache_dir = Path(ast_cache_dir) if ast_cache_dir else None
        self.ast_cache_fallback_to_jsonl = ast_cache_fallback_to_jsonl

        self.data: List[Dict] = []

        # 1. 数据加载
        logger.info(f"Loading dataset from {data_path} (in_memory={in_memory})...")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                if in_memory:
                    self.data = [json.loads(line) for line in f if line.strip()]
                else:
                    # 对于超大数据集，可改为 seek 索引 (此处简化为内存加载)
                    # 若需 Lazy Loading，需自行实现 line offset index
                    self.data = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL format in {data_path}: {e}")

        logger.info(f"Loaded {len(self.data)} samples.")

    def _clean_code(self, code: str) -> str:
        # 参考 LineVul 的处理：移除多余的换行和特殊字符，保留语义结构
        # 1. 移除特殊不可见字符 (除了换行)
        code = "".join([c for c in code if c.isprintable() or c in ['\n', '\t', '\r']])
        # 2. 简单的空格规范化 (可选，取决于你的原始数据质量)
        # code = " ".join(code.split()) 
        return code

    def _get_code_text(self, item: Dict[str, Any]) -> str:
        for key in self.code_keys:
            if key in item:
                value = item.get(key, "")
                if value is None:
                    continue
                if isinstance(value, str):
                    # 对代码进行清洗
                    return self._clean_code(value)
                return str(value)
        return ""

    def _get_domain_value(self, item: Dict[str, Any]) -> int:
        raw_value = None
        for key in self.domain_keys:
            if key in item:
                raw_value = item.get(key)
                break

        if raw_value is None:
            if self.default_domain_value is not None:
                return int(self.default_domain_value)
            return 0

        if isinstance(raw_value, bool):
            return int(raw_value)
        if isinstance(raw_value, (int, float)):
            return int(raw_value)
        if isinstance(raw_value, str):
            if raw_value.isdigit() or (raw_value.startswith("-") and raw_value[1:].isdigit()):
                return int(raw_value)
            if raw_value in self.domain_map:
                return int(self.domain_map[raw_value])
            logger.warning(
                "Domain value '%s' could not be mapped. Using default (%s).",
                raw_value,
                self.default_domain_value if self.default_domain_value is not None else 0,
            )
            return int(self.default_domain_value) if self.default_domain_value is not None else 0

        return int(self.default_domain_value) if self.default_domain_value is not None else 0

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _sanitize_filename(value: Any) -> str:
        return "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in str(value)])

    def _get_ast_cache_path(self, item: Dict[str, Any], idx: int) -> Optional[Path]:
        if not self.ast_cache_dir:
            return None
        project = item.get("project", "unknown")
        commit_id = item.get("commit_id", "unknown")
        item_idx = item.get("idx", idx)
        safe_proj = self._sanitize_filename(project)
        safe_commit = self._sanitize_filename(commit_id)
        safe_idx = self._sanitize_filename(item_idx)
        filename = f"{safe_proj}__{safe_commit}__{safe_idx}.pt"
        return self.ast_cache_dir / filename

    def _validate_ast_edge_index(self, ast_edge_index: torch.Tensor, num_nodes: int) -> None:
        if not self.strict_graph_check or ast_edge_index.numel() == 0:
            return
        max_idx = ast_edge_index.max().item()
        min_idx = ast_edge_index.min().item()
        if max_idx >= num_nodes or min_idx < 0:
            msg = (f"AST Edge Index Out of Bounds: "
                   f"Node count={num_nodes}, but edge refs {max_idx}(max)/{min_idx}(min).")
            raise ValueError(msg)

    def _load_ast_from_cache(self, payload: Any) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(payload, dict):
            return None
        ast_x = payload.get("ast_x")
        ast_edge_index = payload.get("ast_edge_index")
        if ast_x is None or ast_edge_index is None:
            return None

        if not torch.is_tensor(ast_x):
            ast_x = torch.tensor(ast_x, dtype=torch.long)
        else:
            ast_x = ast_x.to(dtype=torch.long)

        if not torch.is_tensor(ast_edge_index):
            ast_edge_index = torch.tensor(ast_edge_index, dtype=torch.long)
        else:
            ast_edge_index = ast_edge_index.to(dtype=torch.long)

        if ast_edge_index.numel() == 0:
            ast_edge_index = torch.zeros((2, 0), dtype=torch.long)
        elif ast_edge_index.dim() == 2 and ast_edge_index.size(0) == 2:
            ast_edge_index = ast_edge_index.contiguous()
        elif ast_edge_index.dim() == 2 and ast_edge_index.size(1) == 2:
            ast_edge_index = ast_edge_index.t().contiguous()
        else:
            return None

        self._validate_ast_edge_index(ast_edge_index, ast_x.size(0))
        return {"ast_x": ast_x, "ast_edge_index": ast_edge_index}

    def _process_ast(self, node_types: List[int], edges: List[List[int]]) -> Dict[str, torch.Tensor]:
        """
        处理 AST 数据，转换为 Tensor，并执行防御性检查。
        """
        # 1. 节点特征 (ast_x)
        if not node_types:
            # 空图处理
            return {
                "ast_x": torch.tensor([], dtype=torch.long),
                "ast_edge_index": torch.zeros((2, 0), dtype=torch.long)
            }

        ast_x = torch.tensor(node_types, dtype=torch.long)
        num_nodes = len(node_types)

        # 2. 边索引 (ast_edge_index)
        if not edges:
            ast_edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            # 输入 edges 为 list of [parent, child] -> shape [E, 2]
            # 转置为 [2, E] -> row 0: parent, row 1: child
            # 严格对齐 GNN 聚合方向: Child aggregates Parent (Parent -> Child)
            edge_tensor = torch.tensor(edges, dtype=torch.long)

            # 防御: 确保是 [E, 2] 形状
            if edge_tensor.dim() != 2 or edge_tensor.size(1) != 2:
                # 可能是空边列表 [] 导致的 tensor([])，需再次检查
                if edge_tensor.numel() == 0:
                    ast_edge_index = torch.zeros((2, 0), dtype=torch.long)
                else:
                    raise ValueError(f"AST edges format error. Expected [[u,v],...], got shape {edge_tensor.shape}")
            else:
                ast_edge_index = edge_tensor.t().contiguous()  # [2, E]

            # 3. 严格图检查 (Debug/防崩)
            self._validate_ast_edge_index(ast_edge_index, num_nodes)

        return {
            "ast_x": ast_x,
            "ast_edge_index": ast_edge_index
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # -------------------------
        # 1. 文本处理 (Head+Tail Tokenization)
        # -------------------------
        code_text = self._get_code_text(item)
        loc = max(1, len([line for line in code_text.splitlines() if line.strip()]))
        # Tokenizer 调用: 不做 batch padding (padding=False)，由 collate 负责
        # 这里不启用 truncation，手动做 Head+Tail 截断
        encoding = self.tokenizer(
            code_text,
            truncation=False,
            padding=False,
            return_token_type_ids=True  # CodeBERT/BERT 需要，RoBERTa 不需要(但兼容)
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding.get("token_type_ids", [])

        max_len = self.max_length
        current_len = len(input_ids)
        if current_len > max_len:
            capacity = max_len - 2
            head_len = capacity // 2
            tail_len = capacity - head_len
            input_ids = (
                [input_ids[0]]
                + input_ids[1: 1 + head_len]
                + input_ids[-1 - tail_len: -1]
                + [input_ids[-1]]
            )
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)

        res = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            # 标签数据 (默认处理为标量，collate 会 stack)
            self.label_key: int(item.get(self.label_key, 0)),
            self.domain_key: self._get_domain_value(item),
            "loc": loc,
            "unit_id": item.get("idx", item.get("unit_id")),
            "project": item.get("project"),
        }

        # 兼容 RoBERTa (没有 token_type_ids) 和 BERT (有)
        if "token_type_ids" in encoding:
            res["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)

        # -------------------------
        # 2. AST 处理 (Optional)
        # -------------------------
        if self.use_ast:
            ast_data = None
            if self.ast_cache_dir:
                cache_path = self._get_ast_cache_path(item, idx)
                if cache_path and cache_path.exists():
                    payload = torch.load(cache_path, map_location="cpu")
                    ast_data = self._load_ast_from_cache(payload)
                elif not self.ast_cache_fallback_to_jsonl:
                    ast_data = {
                        "ast_x": torch.tensor([], dtype=torch.long),
                        "ast_edge_index": torch.zeros((2, 0), dtype=torch.long),
                    }

            if ast_data is None:
                # JSONL 键名契约: 'ast_node_types', 'ast_edges'
                raw_nodes = item.get("ast_node_types", [])
                raw_edges = item.get("ast_edges", [])
                ast_data = self._process_ast(raw_nodes, raw_edges)

            res.update(ast_data)

            # Dataset 不产生 ast_batch，这是 Collate 的职责

        return res


# -----------------------------------------------------------------------------
# 最小可运行示例 (Minimal Runnable Example)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import tempfile
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # 1. 模拟数据生成 (JSONL)
    print(">>> Generating dummy data...")
    dummy_data = [
        # 样本 1: 正常代码 + 简单 AST (0->1, 0->2)
        {
            "code": "def hello(): print('world')",
            "target": 1,
            "domain": 0,
            "ast_node_types": [10, 20, 30],
            "ast_edges": [[0, 1], [0, 2]]  # Parent->Child
        },
        # 样本 2: 空 AST (模拟解析失败或无结构)
        {
            "code": "x = 1",
            "target": 0,
            "domain": 1,
            "ast_node_types": [],
            "ast_edges": []
        }
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl', encoding='utf-8') as tmp:
        for d in dummy_data:
            tmp.write(json.dumps(d) + '\n')
        tmp_path = tmp.name

    # 2. 初始化组件
    print(">>> Initializing Tokenizer & Dataset...")
    # 使用轻量级 tokenizer 模拟
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    dataset = CPDPDataset(
        data_path=tmp_path,
        tokenizer=tokenizer,
        max_length=32,
        use_ast=True,
        strict_graph_check=True
    )


    # 3. 模拟 Collate (因无法 import src.data.collate，此处内联简化版)
    # 实际工程请使用 from src.data.collate import Collator, CollateConfig
    def dummy_collate_fn(samples):
        # 简化的 Disjoint Union 逻辑用于测试
        batch = {}
        # Text Stack
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            [s["input_ids"] for s in samples], batch_first=True, padding_value=1
        )
        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [s["attention_mask"] for s in samples], batch_first=True, padding_value=0
        )
        batch["target"] = torch.tensor([s["target"] for s in samples])

        # AST Disjoint Union
        xs, eis, batches = [], [], []
        offset = 0
        for i, s in enumerate(samples):
            x = s["ast_x"]
            ei = s["ast_edge_index"]
            if x.numel() > 0:
                xs.append(x)
                # Offset edges
                if ei.numel() > 0:
                    eis.append(ei + offset)
                else:
                    eis.append(torch.zeros(2, 0, dtype=torch.long))
                batches.append(torch.full((x.size(0),), i, dtype=torch.long))
                offset += x.size(0)
            else:
                # Empty graph handling
                pass

        if xs:
            batch["ast_x"] = torch.cat(xs)
            batch["ast_edge_index"] = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0).long()
            batch["ast_batch"] = torch.cat(batches)
        else:
            batch["ast_x"] = torch.zeros(0).long()
            batch["ast_edge_index"] = torch.zeros(2, 0).long()
            batch["ast_batch"] = torch.zeros(0).long()

        return batch


    # 4. DataLoader 测试
    print(">>> Creating DataLoader...")
    loader = DataLoader(dataset, batch_size=2, collate_fn=dummy_collate_fn)

    for batch in loader:
        print("\n=== Batch Shapes ===")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape} | dtype={v.dtype}")
            else:
                print(f"{k}: {v}")

        # 验证 AST 逻辑
        assert "ast_batch" in batch
        assert batch["ast_edge_index"].shape[0] == 2
        print("\n>>> Dataset Test Passed!")
        break

    # 清理
    os.remove(tmp_path)
