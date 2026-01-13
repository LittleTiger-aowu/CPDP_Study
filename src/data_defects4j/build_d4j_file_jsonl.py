import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def parse_bug_ids(raw: str) -> List[int]:
    raw = raw.strip()
    if not raw:
        return []
    bug_ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            bug_ids.extend(list(range(start, end + 1)))
        else:
            bug_ids.append(int(part))
    return sorted(set(bug_ids))


def run_cmd(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
    check: bool = True,
) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )
    return result.stdout.strip()


def _class_to_path(item: str) -> str:
    cleaned = item.strip()
    if cleaned.endswith(".java") or "/" in cleaned:
        return cleaned
    cleaned = cleaned.split("$")[0]
    return cleaned.replace(".", "/") + ".java"


def get_changed_files(project: str, bug_id: int, fixed_dir: Path, buggy_dir: Path, timeout: Optional[int]) -> List[str]:
    changed: List[str] = []
    try:
        output = run_cmd(
            ["defects4j", "export", "-p", "classes.modified", "-w", str(fixed_dir)],
            timeout=timeout,
        )
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            path = _class_to_path(line)
            changed.append(path)
    except subprocess.SubprocessError:
        changed = []

    if not changed:
        diff_output = run_cmd(
            ["diff", "-rq", str(buggy_dir), str(fixed_dir)],
            timeout=timeout,
            check=False,
        )
        for line in diff_output.splitlines():
            if ".java" not in line:
                continue
            if line.startswith("Files ") and " differ" in line:
                parts = line.split(" and ")
                left = parts[0].replace("Files ", "").strip()
                rel = os.path.relpath(left, buggy_dir)
                if rel.endswith(".java"):
                    changed.append(rel)
            elif line.startswith("Only in "):
                chunk = line.replace("Only in ", "")
                dir_part, file_part = chunk.split(":", 1)
                file_part = file_part.strip()
                if file_part.endswith(".java"):
                    full_path = Path(dir_part.strip()) / file_part
                    rel = os.path.relpath(full_path, buggy_dir)
                    changed.append(rel)

    unique = sorted({p for p in changed if p.endswith(".java")})
    existing = []
    for path in unique:
        if (buggy_dir / path).exists():
            existing.append(path)
    return existing


def list_java_files(root: Path) -> List[str]:
    return sorted([str(p.relative_to(root)) for p in root.rglob("*.java")])


def read_file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_methods_simple(code: str, max_methods: int) -> List[dict]:
    import re

    methods: List[dict] = []
    pattern = re.compile(
        r"(public|protected|private|static|final|native|synchronized|abstract|\s)+"
        r"[\w\<\>\[\]]+\s+(\w+)\s*\([^;]*\)\s*\{",
        re.MULTILINE,
    )
    for match in pattern.finditer(code):
        name = match.group(2)
        start = match.start()
        brace_count = 0
        end = None
        for idx in range(start, len(code)):
            if code[idx] == "{":
                brace_count += 1
            elif code[idx] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = idx + 1
                    break
        if end is None:
            continue
        method_code = code[start:end]
        start_line = code.count("\n", 0, start) + 1
        end_line = code.count("\n", 0, end) + 1
        methods.append(
            {
                "mid": f"m{len(methods)}",
                "name": name,
                "span": [start_line, end_line],
                "code": method_code,
            }
        )
        if len(methods) >= max_methods:
            break
    return methods


def build_samples(
    project: str,
    bug_id: int,
    buggy_dir: Path,
    fixed_dir: Path,
    neg_ratio: int,
    max_files_per_bug: Optional[int],
    extract_methods: bool,
    max_methods: int,
    seed: int,
    domain: str,
    timeout: Optional[int],
) -> List[dict]:
    changed_files = get_changed_files(project, bug_id, fixed_dir, buggy_dir, timeout)
    all_java = list_java_files(buggy_dir)

    changed_set = set(changed_files)
    pos_files = [f for f in all_java if f in changed_set]
    neg_files = [f for f in all_java if f not in changed_set]

    rng = random.Random(seed)
    if neg_ratio > 0:
        target_neg = min(len(neg_files), neg_ratio * max(1, len(pos_files)))
        neg_files = rng.sample(neg_files, target_neg) if neg_files else []

    if max_files_per_bug is not None:
        remaining = max(0, max_files_per_bug - len(pos_files))
        if remaining < len(neg_files):
            neg_files = rng.sample(neg_files, remaining)

    samples = []
    for file_path in pos_files + neg_files:
        code = read_file_text(buggy_dir / file_path)
        label = 1 if file_path in changed_set else 0
        unit_id = f"{project}-{bug_id}-buggy-{file_path}"
        sample = {
            "dataset": "defects4j",
            "project": project,
            "bug_id": bug_id,
            "version": "buggy",
            "file_path": file_path,
            "unit_id": unit_id,
            "language": "Java",
            "code": code,
            "label": label,
            "domain": domain,
        }
        if extract_methods:
            sample["methods"] = extract_methods_simple(code, max_methods)
        samples.append(sample)
    return samples


def checkout_version(project: str, bug_id: int, version: str, workdir: Path, timeout: Optional[int]) -> Path:
    checkout_dir = workdir / f"{project}_{bug_id}_{version}"
    if checkout_dir.exists():
        shutil.rmtree(checkout_dir)
    run_cmd(
        ["defects4j", "checkout", "-p", project, "-v", f"{bug_id}{version}", "-w", str(checkout_dir)],
        timeout=timeout,
    )
    return checkout_dir


def main() -> None:
    parser = argparse.ArgumentParser("Build Defects4J file-level JSONL")
    parser.add_argument("--d4j-root", type=str, required=False, default=".")
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--projects", type=str, required=True)
    parser.add_argument("--bug-ids", type=str, required=True)
    parser.add_argument("--out-jsonl", type=str, required=True)
    parser.add_argument("--neg-ratio", type=int, default=5)
    parser.add_argument("--max-files-per-bug", type=int, default=None)
    parser.add_argument("--extract-methods", action="store_true")
    parser.add_argument("--max-methods", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--domain", type=str, default="source")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    os.environ["DEFECTS4J_HOME"] = args.d4j_root

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    projects = [p.strip() for p in args.projects.split(",") if p.strip()]
    bug_ids = parse_bug_ids(args.bug_ids)

    samples: List[dict] = []
    for project in projects:
        for bug_id in bug_ids:
            buggy_dir = checkout_version(project, bug_id, "b", workdir, args.timeout)
            fixed_dir = checkout_version(project, bug_id, "f", workdir, args.timeout)
            samples.extend(
                build_samples(
                    project=project,
                    bug_id=bug_id,
                    buggy_dir=buggy_dir,
                    fixed_dir=fixed_dir,
                    neg_ratio=args.neg_ratio,
                    max_files_per_bug=args.max_files_per_bug,
                    extract_methods=args.extract_methods,
                    max_methods=args.max_methods,
                    seed=args.seed,
                    domain=args.domain,
                    timeout=args.timeout,
                )
            )

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
