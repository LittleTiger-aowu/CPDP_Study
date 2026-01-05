import argparse
import json
import os
import torch
import hashlib
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from tree_sitter import Language, Parser


# ==========================================
# 1. 核心处理逻辑
# ==========================================

class ASTCacheBuilder:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 Parser
        self.parser = Parser()
        try:
            # 加载 .so 库
            lang = Language(args.lib_path, args.language)
            self.parser.set_language(lang)
        except Exception as e:
            raise RuntimeError(f"Failed to load tree-sitter language from {args.lib_path}. Error: {e}")

        self.vocab = {}
        self.vocab_path = Path(args.vocab_path) if args.vocab_path else self.out_dir / "vocab.json"

    def _get_ast_sequence(self, source_code_bytes):
        """
        解析源码，返回 (node_types, edges)
        edges: List[Tuple[parent_idx, child_idx]] (严格 Parent->Child)
        """
        tree = self.parser.parse(source_code_bytes)
        root = tree.root_node

        stack = [(root, -1)]
        node_types_list = []
        edges_list = []
        curr_idx = 0

        while stack:
            node, parent_idx = stack.pop()

            node_types_list.append(node.type)

            if parent_idx != -1:
                edges_list.append((parent_idx, curr_idx))

            # 倒序压栈，确保 DFS 顺序
            for child in reversed(node.children):
                stack.append((child, curr_idx))

            curr_idx += 1

        return node_types_list, edges_list

    def build_vocab(self):
        """
        第一遍扫描：构建全局 node_type 词表
        [Fix 1] 明确使用字典序排序，保证跨平台/多次运行的绝对确定性。
        """
        print(f"[Pass 1] Scanning for vocab from {self.args.jsonl_path}...")
        counter = Counter()

        # [Fix 2] 流式读取，避免一次性加载
        with open(self.args.jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(tqdm(f, desc="Building Vocab")):
                # 处理 max_samples
                if self.args.max_samples > 0 and line_idx >= self.args.max_samples:
                    break

                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    code = item.get('func', "")
                    if not code: continue

                    tree = self.parser.parse(bytes(code, "utf8"))
                    cursor = tree.walk()

                    # 高效遍历
                    visited_children = False
                    while True:
                        if not visited_children:
                            counter[cursor.node.type] += 1
                        if cursor.goto_first_child():
                            visited_children = False
                        elif cursor.goto_next_sibling():
                            visited_children = False
                        elif cursor.goto_parent():
                            visited_children = True
                        else:
                            break
                except Exception:
                    continue

                    # [Fix 1] 强制字典序排序 (Alphabetical Sort)
        sorted_types = sorted(counter.keys())
        self.vocab = {t: i + 1 for i, t in enumerate(sorted_types)}
        self.vocab["<unk>"] = 0

        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2)
        print(f"[Pass 1] Vocab saved to {self.vocab_path}, size: {len(self.vocab)} (Sorted Alphabetically)")

    def build_cache(self):
        """第二遍扫描：生成并保存 Tensor Cache"""
        if not self.vocab:
            if self.vocab_path.exists():
                with open(self.vocab_path, 'r') as f:
                    self.vocab = json.load(f)
                print(f"[Pass 2] Loaded existing vocab, size: {len(self.vocab)}")
            else:
                raise FileNotFoundError("Vocab not found. Run with vocab building first.")

        print(f"[Pass 2] Generating AST cache to {self.out_dir}...")

        bad_cases_path = self.out_dir / "bad_cases.jsonl"
        bad_file = open(bad_cases_path, 'w', encoding='utf-8')

        success_count = 0
        fail_count = 0

        # [Fix 2] 流式读取
        with open(self.args.jsonl_path, 'r', encoding='utf-8') as f:
            iterator = tqdm(f, desc="Processing")

            for line_idx, line in enumerate(iterator):
                if self.args.max_samples > 0 and line_idx >= self.args.max_samples:
                    break

                if not line.strip(): continue

                try:
                    item = json.loads(line)
                    code = item.get('func', "")
                    if not code: raise ValueError("Empty code")

                    # [Fix 3] 增强 Key 唯一性：加入内容 Hash
                    # 格式: {project}__{commit}__{idx}_{hash8}.pt
                    # 这样即使 idx 在不同版本数据中发生碰撞，Hash 也能保证唯一性
                    # Dataset 端加载时也需同样计算 hash，或者 Dataset 直接读文件名列表
                    # (此处假设 Dataset 能够根据 code 内容复现此 Key，或这是离线预处理一次性生成)
                    code_hash = hashlib.md5(code.encode('utf-8')).hexdigest()[:8]

                    # 优先使用 item['idx']，如果没有则用 line_idx，最后挂上 hash 双保险
                    idx_val = item.get('idx', line_idx)
                    file_key = f"{item.get('project', 'unk')}__{item.get('commit_id', 'unk')}__{idx_val}_{code_hash}"

                    save_path = self.out_dir / f"{file_key}.pt"

                    if not self.args.overwrite and save_path.exists():
                        continue

                    # 1. 提取结构
                    node_types_str, edges = self._get_ast_sequence(bytes(code, "utf8"))

                    # 2. Tensor 化
                    node_ids = [self.vocab.get(t, 0) for t in node_types_str]
                    ast_x = torch.tensor(node_ids, dtype=torch.long)

                    if len(edges) > 0:
                        # 转置为 [2, E]
                        edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    else:
                        edge_tensor = torch.zeros((2, 0), dtype=torch.long)

                    # 3. 构造 Cache 对象
                    data = {
                        "ast_x": ast_x,
                        "ast_edge_index": edge_tensor,
                        "meta": {
                            "project": item.get('project'),
                            "idx": idx_val,
                            "hash": code_hash,
                            "nodes_count": len(node_ids)
                        }
                    }

                    torch.save(data, save_path)
                    success_count += 1

                except Exception as e:
                    fail_count += 1
                    bad_record = {"line": line_idx, "error": str(e)}
                    bad_file.write(json.dumps(bad_record) + "\n")

                    if self.args.on_error == "empty":
                        # 错误时生成带 Hash 的空文件，防止 Dataset 加载 404
                        # 注意：如果不知道 Hash，Dataset 还是找不到，所以这里其实主要用于 debug
                        # 生产环境建议 on_error skip
                        pass

        bad_file.close()
        print(f"Done. Success: {success_count}, Failed: {fail_count}. Bad cases: {bad_cases_path}")


# ==========================================
# 2. CLI 入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Offline AST Cache (Route B)")

    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--lib_path", type=str, required=True)

    parser.add_argument("--language", type=str, default="c")
    parser.add_argument("--vocab_path", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--on_error", type=str, choices=["skip", "empty"], default="skip")  # 默认 skip 更安全
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_vocab", action="store_true")

    args = parser.parse_args()

    builder = ASTCacheBuilder(args)

    if not args.skip_vocab:
        builder.build_vocab()

    builder.build_cache()

# ==========================================
# 3. 验收测试 (Smoke Test)
# ==========================================
# 运行前请取消下方注释
"""
if __name__ == "__main__":
    # 使用方法:
    # python tools/build_ast_cache.py --jsonl_path ... --lib_path ...

    # 简单的逻辑验证脚本 (需要真实文件路径)
    pass
"""