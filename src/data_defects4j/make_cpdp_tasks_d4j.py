import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_by_bug(items: List[dict], valid_ratio: float, seed: int) -> (List[dict], List[dict]):
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for item in items:
        grouped[int(item.get("bug_id", 0))].append(item)

    bug_ids = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(bug_ids)

    valid_count = max(1, int(len(bug_ids) * valid_ratio)) if bug_ids else 0
    valid_ids = set(bug_ids[:valid_count])

    train_items: List[dict] = []
    valid_items: List[dict] = []
    for bug_id, group_items in grouped.items():
        if bug_id in valid_ids:
            valid_items.extend(group_items)
        else:
            train_items.extend(group_items)
    return train_items, valid_items


def assign_domain(rows: List[dict], domain: str) -> List[dict]:
    updated = []
    for row in rows:
        row = dict(row)
        row["domain"] = domain
        updated.append(row)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser("Build CPDP tasks from Defects4J file-level JSONL")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["pairwise", "lopo"], default="pairwise")
    parser.add_argument("--source-project", type=str, default=None)
    parser.add_argument("--target-project", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="src/data")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data = load_jsonl(Path(args.input_jsonl))

    by_project: Dict[str, List[dict]] = defaultdict(list)
    for row in data:
        by_project[str(row.get("project", "unknown"))].append(row)

    target_project = args.target_project
    if target_project not in by_project:
        raise ValueError(f"Target project '{target_project}' not found in dataset")

    if args.mode == "pairwise":
        if not args.source_project:
            raise ValueError("source-project is required for pairwise mode")
        source_projects = [args.source_project]
        dir_name = f"cpdp_d4j_{args.source_project}_to_{target_project}"
    else:
        source_projects = [p for p in by_project.keys() if p != target_project]
        dir_name = f"cpdp_d4j_lopo_{target_project}"

    source_rows = []
    for project in source_projects:
        source_rows.extend(by_project.get(project, []))
    target_rows = by_project[target_project]

    train_rows, valid_rows = split_by_bug(source_rows, args.valid_ratio, args.seed)

    train_rows = assign_domain(train_rows, "source")
    valid_rows = assign_domain(valid_rows, "source")
    test_rows = assign_domain(target_rows, "target")

    output_root = Path(args.output_root) / dir_name
    write_jsonl(output_root / "train.jsonl", train_rows)
    write_jsonl(output_root / "valid.jsonl", valid_rows)
    write_jsonl(output_root / "test.jsonl", test_rows)

    meta = {
        "mode": args.mode,
        "source_projects": source_projects,
        "target_project": target_project,
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "counts": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "test": len(test_rows),
        },
    }
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
