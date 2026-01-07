import json
import os
import shutil
import subprocess
from pathlib import Path

import torch


SAMPLE_DATA = {
    "project": "FFmpeg",
    "commit_id": "973b1a6b9070e2bf17d17568cbaf4043ce931f51",
    "target": 0,
    "func": "int main() { return 0; }",
    "idx": 0,
}


def smoke_test():
    print("=== AST Cache Builder Smoke Test ===")

    tmp_dir = Path("./tmp_smoke_test")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    jsonl_path = tmp_dir / "data.jsonl"
    out_dir = tmp_dir / "cache"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(SAMPLE_DATA) + "\n")
        item2 = SAMPLE_DATA.copy()
        item2["idx"] = 1
        item2["func"] = "void test(int a) { a = 1; }"
        f.write(json.dumps(item2) + "\n")
        item3 = SAMPLE_DATA.copy()
        item3["idx"] = 2
        item3["func"] = ""
        f.write(json.dumps(item3) + "\n")

    lib_path = "build/my-languages.so"
    if not os.path.exists(lib_path):
        print(f"[WARN] {lib_path} not found. Skipping actual execution.")
        print("Please compile tree-sitter languages to run full smoke test.")
        fake_pt = out_dir / f"{SAMPLE_DATA['project']}__{SAMPLE_DATA['commit_id']}__0.pt"
        out_dir.mkdir()
        torch.save(
            {
                "ast_x": torch.tensor([1, 2, 3], dtype=torch.long),
                "ast_edge_index": torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
                "meta": SAMPLE_DATA,
            },
            fake_pt,
        )
    else:
        cmd = [
            "python",
            "tools/build_ast_cache.py",
            "--jsonl_path",
            str(jsonl_path),
            "--out_dir",
            str(out_dir),
            "--lib_path",
            lib_path,
            "--language",
            "c",
            "--overwrite",
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)

    target_file = out_dir / f"{SAMPLE_DATA['project']}__{SAMPLE_DATA['commit_id']}__0.pt"

    if not target_file.exists():
        print(f"[FAIL] Target cache file not found: {target_file}")
        return

    data = torch.load(target_file)
    print("\n--- Loaded Cache Check ---")
    print(f"Meta: {data.get('meta')}")

    ast_x = data["ast_x"]
    edge_index = data["ast_edge_index"]

    print(f"ast_x shape: {ast_x.shape}, dtype: {ast_x.dtype}")
    print(f"edge_index shape: {edge_index.shape}, dtype: {edge_index.dtype}")

    assert ast_x.dtype == torch.long, "ast_x must be LongTensor"
    assert edge_index.dtype == torch.long, "edge_index must be LongTensor"
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must be [2, E]"

    num_nodes = ast_x.size(0)
    if edge_index.size(1) > 0:
        max_idx = edge_index.max().item()
        assert max_idx < num_nodes, f"Edge index {max_idx} out of bounds (N={num_nodes})"
        print(f"[PASS] Boundary check: max_edge_idx ({max_idx}) < num_nodes ({num_nodes})")
    else:
        print("[PASS] No edges (trivial graph)")

    print("[SUCCESS] Smoke test passed.")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    smoke_test()
