import os
from typing import Dict, List, Optional, Tuple

from tree_sitter import Language, Parser

PAD_ID = 0
UNK_ID = 1


def _resolve_library_path(lib_path: Optional[str]) -> str:
    if lib_path and os.path.exists(lib_path):
        return lib_path
    candidates = [
        os.environ.get("SSE_TS_LIB"),
        os.path.join(os.path.dirname(__file__), "parser", "my-languages.so"),
        os.path.join(os.path.dirname(__file__), "parser", "my-languages.dylib"),
        os.path.join(os.path.dirname(__file__), "parser", "my-languages.dll"),
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "evaluators", "build", "my-languages.dll"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "未找到 tree-sitter 语言库，请通过 SSE_TS_LIB 指定路径。"
    )


def _load_language() -> Language:
    language_name = os.environ.get("SSE_TS_LANG", "c")
    lib_path = _resolve_library_path(None)
    return Language(lib_path, language_name)


_LANGUAGE = _load_language()
_PARSER = Parser()
_PARSER.set_language(_LANGUAGE)

NODE_KIND_MAP: Dict[str, int] = {
    _LANGUAGE.node_kind_for_id(i): i + 2 for i in range(_LANGUAGE.node_kind_count)
}
AST_VOCAB_SIZE = _LANGUAGE.node_kind_count + 2


def _count_named_nodes(node) -> int:
    count = 1 if node.is_named else 0
    for child in node.children:
        count += _count_named_nodes(child)
    return count


def _split_subtrees(node, max_nodes: int) -> List:
    if not node.is_named:
        return []
    total = _count_named_nodes(node)
    if total <= max_nodes:
        return [node]
    subtrees = []
    for child in node.children:
        if child.is_named:
            subtrees.extend(_split_subtrees(child, max_nodes))
    if not subtrees:
        subtrees.append(node)
    return subtrees


def _collect_leaf_paths(node, max_path_len: int) -> List[Tuple[int, List[int]]]:
    paths: List[Tuple[int, List[int]]] = []

    def dfs(cur, stack: List[str]):
        if cur.is_named:
            stack.append(cur.type)
        named_children = [child for child in cur.children if child.is_named]
        if not named_children:
            path = stack[:max_path_len]
            ids = [NODE_KIND_MAP.get(t, UNK_ID) for t in path]
            paths.append((cur.start_byte, ids))
        else:
            for child in named_children:
                dfs(child, stack[:])

    dfs(node, [])
    paths.sort(key=lambda x: x[0])
    return paths


def ParseToASTPath(code_str: str) -> List[List[List[int]]]:
    """
    返回子树路径序列: List[Subtree][Path][NodeTypeId]
    """
    max_subtree_nodes = int(os.environ.get("SSE_MAX_SUBTREE_NODES", 200))
    max_path_len = int(os.environ.get("SSE_MAX_PATH_LEN", 32))
    max_subtrees = int(os.environ.get("SSE_MAX_SUBTREES", 32))

    tree = _PARSER.parse(code_str.encode("utf-8"))
    root = tree.root_node
    subtrees = _split_subtrees(root, max_subtree_nodes)[:max_subtrees]

    results: List[List[List[int]]] = []
    for subtree in subtrees:
        leaf_paths = _collect_leaf_paths(subtree, max_path_len)
        results.append([path for _, path in leaf_paths])
    return results
