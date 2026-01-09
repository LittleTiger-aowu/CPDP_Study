import argparse
import json
import os
from pathlib import Path
from collections import Counter

from tqdm import tqdm
import torch
from tree_sitter import Language, Parser


# ==========================================
# 1. AST 解析核心逻辑 (无状态)
# ==========================================

def parse_code_to_graph(code_bytes, parser):
    """
    解析代码字节流，返回节点类型列表和边列表 (Parent->Child)。

    Returns:
        node_types: List[str]
        edges: List[Tuple[int, int]] (source, target)
    """
    try:
        tree = parser.parse(code_bytes)
        root_node = tree.root_node

        if (root_node.end_byte - root_node.start_byte) == 0 or len(root_node.children) == 0:
            return [], []

        node_types = []
        edges = []
        stack = [(root_node, -1)]

        while stack:
            curr_node, parent_idx = stack.pop()
            curr_idx = len(node_types)
            node_types.append(curr_node.type)

            if parent_idx != -1:
                edges.append((parent_idx, curr_idx))

            for child in reversed(curr_node.children):
                stack.append((child, curr_idx))

        return node_types, edges

    except Exception as e:
        raise RuntimeError(f"Tree-sitter parse error: {str(e)}")


# ==========================================
# 2. 工具函数
# ==========================================

def sanitize_filename(s):
    """清理文件名中的非法字符"""
    return "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in str(s)])


def get_parser(lib_path, language_name):
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Language library not found: {lib_path}")

    lang = Language(lib_path, language_name)
    parser = Parser()
    parser.set_language(lang)
    return parser


# ==========================================
# 3. Pass 1: 构建 Vocab
# ==========================================

def build_vocab(args, parser):
    print("==> [Pass 1] Scanning for vocabulary...")
    type_counter = Counter()

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        iterator = tqdm(f, desc="Building Vocab", unit="lines")
        for i, line in enumerate(iterator):
            if args.max_samples and i >= args.max_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                code = item.get("func", "")
                if not code:
                    continue

                node_types, _ = parse_code_to_graph(code.encode("utf-8"), parser)
                type_counter.update(node_types)

            except json.JSONDecodeError:
                continue
            except Exception:
                continue

    sorted_types = sorted(type_counter.keys(), key=lambda k: (-type_counter[k], k))
    vocab = {t: idx for idx, t in enumerate(sorted_types)}

    vocab_path = Path(args.vocab_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    print(f"==> Vocab saved to {vocab_path} (Size: {len(vocab)})")
    return vocab


# ==========================================
# 4. Pass 2: 生成 Cache
# ==========================================

def build_cache(args, parser, vocab):
    print("==> [Pass 2] Generating AST cache...")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bad_cases_path = out_dir / "bad_cases.jsonl"
    bad_cases_file = open(bad_cases_path, "w", encoding="utf-8")

    success_count = 0
    fail_count = 0
    skip_count = 0

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        iterator = tqdm(f, desc="Generating Cache", unit="samples")

        for i, line in enumerate(iterator):
            if args.max_samples and i >= args.max_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                project = item.get("project", "unknown")
                commit_id = item.get("commit_id", "unknown")
                idx = item.get("idx", i)
                func_code = item.get("func", "")
            except json.JSONDecodeError:
                bad_cases_file.write(json.dumps({"line": i, "error": "JSONDecodeError"}) + "\n")
                fail_count += 1
                continue

            safe_proj = sanitize_filename(project)
            safe_commit = sanitize_filename(commit_id)
            safe_idx = sanitize_filename(idx)
            filename = f"{safe_proj}__{safe_commit}__{safe_idx}.pt"
            save_path = out_dir / filename

            if save_path.exists() and not args.overwrite:
                skip_count += 1
                continue

            try:
                node_types, edges = parse_code_to_graph(func_code.encode("utf-8"), parser)

                try:
                    x_ids = [vocab[t] for t in node_types]
                except KeyError as e:
                    raise ValueError(f"Unknown node type found in Pass 2: {e}")

                if len(x_ids) == 0:
                    ast_x = torch.tensor([], dtype=torch.long)
                    ast_edge_index = torch.empty((2, 0), dtype=torch.long)
                else:
                    ast_x = torch.tensor(x_ids, dtype=torch.long)
                    if len(edges) > 0:
                        ast_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    else:
                        ast_edge_index = torch.empty((2, 0), dtype=torch.long)

                payload = {
                    "ast_x": ast_x,
                    "ast_edge_index": ast_edge_index,
                    "meta": {
                        "project": project,
                        "commit_id": commit_id,
                        "idx": idx,
                    },
                }

                torch.save(payload, save_path)
                success_count += 1

            except Exception as e:
                fail_count += 1
                error_info = {
                    "project": project,
                    "commit_id": commit_id,
                    "idx": idx,
                    "error": str(e),
                }
                bad_cases_file.write(json.dumps(error_info) + "\n")

                if args.on_error == "skip":
                    continue
                if args.on_error == "empty":
                    payload = {
                        "ast_x": torch.tensor([], dtype=torch.long),
                        "ast_edge_index": torch.empty((2, 0), dtype=torch.long),
                        "meta": {"error": str(e)},
                    }
                    torch.save(payload, save_path)

    bad_cases_file.close()
    print(f"==> Done. Success: {success_count}, Failed: {fail_count}, Skipped: {skip_count}")
    print(f"==> Bad cases saved to {bad_cases_path}")


# ==========================================
# 5. Main Entry
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Offline AST Cache Builder")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for cache")
    parser.add_argument("--lib_path", type=str, required=True, help="Path to tree-sitter .so file")
    parser.add_argument("--language", type=str, default="c", help="Language name (e.g., c, java)")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Path to vocab.json (default: out_dir/vocab.json)",
    )
    parser.add_argument("--num_workers", type=int, default=1, help="Num workers (Not implemented for simplicity, use 1)")
    parser.add_argument("--max_samples", type=int, default=None, help="Debug limit")
    parser.add_argument("--on_error", type=str, choices=["skip", "empty"], default="skip")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")

    args = parser.parse_args()

    if args.vocab_path is None:
        args.vocab_path = os.path.join(args.out_dir, "vocab.json")

    ts_parser = get_parser(args.lib_path, args.language)

    if os.path.exists(args.vocab_path) and not args.overwrite:
        print(f"==> Loading existing vocab from {args.vocab_path}")
        with open(args.vocab_path, "r") as f:
            vocab = json.load(f)
    else:
        vocab = build_vocab(args, ts_parser)

    build_cache(args, ts_parser, vocab)


if __name__ == "__main__":
    main()
