import argparse
import json
import random
import os
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {i + 1} in {path}")
    return data


def save_jsonl(data: List[Dict], path: Path, remove_target: bool = False):
    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            # 创建副本以免修改原始内存数据
            out_item = item.copy()
            if remove_target and 'target' in out_item:
                del out_item['target']
            f.write(json.dumps(out_item) + '\n')

    print(f"Saved {len(data)} records to {path}")


def get_stats(data: List[Dict]) -> str:
    total = len(data)
    if total == 0:
        return "Empty"
    # 检查是否有 target 字段
    if 'target' not in data[0]:
        return f"Total: {total} (Unlabeled)"

    pos = sum(1 for d in data if d.get('target', 0) == 1)
    neg = total - pos
    return f"Total: {total} | Bug: {pos} ({pos / total:.1%}) | Clean: {neg}"


def main():
    parser = argparse.ArgumentParser(description="Split datasets for CPDP (Source -> Target) experiments.")

    # 输入文件路径
    parser.add_argument("--src_file", type=str, required=True,
                        help="Path to full Source project JSONL (e.g., FFmpeg.jsonl)")
    parser.add_argument("--tgt_file", type=str, required=True,
                        help="Path to full Target project JSONL (e.g., Qemu.jsonl)")

    # 输出配置
    parser.add_argument("--out_root", type=str, default="src/data", help="Root directory for output data")

    # 划分比例
    parser.add_argument("--src_train_ratio", type=float, default=0.8,
                        help="Ratio of Source data used for Training (remainder for Valid)")
    parser.add_argument("--tgt_align_ratio", type=float, default=0.5,
                        help="Ratio of Target data used for Unlabeled Alignment (remainder for Test)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # 1. 设置随机种子
    random.seed(args.seed)

    # 2. 读取数据
    print(f"Loading Source: {args.src_file}...")
    src_data = load_jsonl(args.src_file)
    print(f"Loading Target: {args.tgt_file}...")
    tgt_data = load_jsonl(args.tgt_file)

    if not src_data or not tgt_data:
        raise ValueError("Source or Target data is empty!")

    # 3. 提取项目名称 (用于生成目录名)
    # 假设文件名就是项目名，或者从数据第一条记录读取 'project' 字段
    src_name = src_data[0].get("project", Path(args.src_file).stem)
    tgt_name = tgt_data[0].get("project", Path(args.tgt_file).stem)

    out_dir_name = f"cpdp_{src_name}_to_{tgt_name}"
    out_dir = Path(args.out_root) / out_dir_name

    print(f"\nTask: {src_name} -> {tgt_name}")
    print(f"Output Directory: {out_dir}\n")

    # 4. 打乱数据 (Shuffle)
    random.shuffle(src_data)
    random.shuffle(tgt_data)

    # 5. 划分源项目 (Source Split)
    # Train (e.g., 80%) / Valid (e.g., 20%)
    src_split_idx = int(len(src_data) * args.src_train_ratio)
    src_train = src_data[:src_split_idx]
    src_valid = src_data[src_split_idx:]

    # 6. 划分目标项目 (Target Split)
    # Unlabeled Alignment (e.g., 50%) / Test (e.g., 50%)
    tgt_split_idx = int(len(tgt_data) * args.tgt_align_ratio)
    tgt_unlabeled = tgt_data[:tgt_split_idx]
    tgt_test = tgt_data[tgt_split_idx:]

    # 7. 保存文件

    # A. train.jsonl (Source Train, Labeled)
    save_jsonl(src_train, out_dir / "train.jsonl", remove_target=False)

    # B. valid_src.jsonl (Source Valid, Labeled)
    save_jsonl(src_valid, out_dir / "valid_src.jsonl", remove_target=False)

    # C. valid_tgt_unlabeled.jsonl (Target Alignment, Unlabeled - Removes 'target')
    save_jsonl(tgt_unlabeled, out_dir / "valid_tgt_unlabeled.jsonl", remove_target=True)

    # D. test_tgt.jsonl (Target Test, Labeled)
    save_jsonl(tgt_test, out_dir / "test_tgt.jsonl", remove_target=False)

    # 8. 输出统计信息
    print("\nData Statistics:")
    print(f"{'File':<30} | {'Stats'}")
    print("-" * 60)
    print(f"{'train.jsonl':<30} | {get_stats(src_train)}")
    print(f"{'valid_src.jsonl':<30} | {get_stats(src_valid)}")
    print(f"{'valid_tgt_unlabeled.jsonl':<30} | {get_stats(tgt_unlabeled)} (Labels Removed)")
    print(f"{'test_tgt.jsonl':<30} | {get_stats(tgt_test)}")
    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()