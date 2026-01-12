from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser("Run CodeBERT naive transfer baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/naive_transfer_codebert.yaml",
        help="Path to naive transfer config YAML.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default=os.getcwd(),
        help="Project directory containing JSONL files (defaults to current directory).",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/naive_transfer_codebert.log",
        help="Log file path.",
    )
    args = parser.parse_args()

    command = [
        sys.executable,
        "src/run_experiment.py",
        "--config",
        args.config,
        "--project_dir",
        args.project_dir,
        "--log_path",
        args.log_path,
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
