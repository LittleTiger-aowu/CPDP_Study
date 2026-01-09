from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def set_nested(cfg: Dict[str, Any], keys: list[str], value: Any) -> None:
    cur = cfg
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def build_experiments() -> list[Dict[str, Any]]:
    return [
        {
            "name": "Exp1_Full_Method",
            "overrides": {
                ("model", "ast", "enable"): True,
                ("model", "feature_split", "enable"): True,
                ("model", "dann", "enable"): True,
                ("model", "ortho", "enable"): True,
            },
        },
        {
            "name": "Exp2_wo_Decoupling",
            "overrides": {
                ("model", "ast", "enable"): True,
                ("model", "feature_split", "enable"): False,
                ("model", "ortho", "enable"): False,
            },
        },
        {
            "name": "Exp3_wo_AST",
            "overrides": {
                ("model", "ast", "enable"): False,
                ("model", "feature_split", "enable"): True,
                ("model", "dann", "enable"): True,
                ("model", "ortho", "enable"): True,
            },
        },
        {
            "name": "Exp4_wo_DANN",
            "overrides": {
                ("model", "ast", "enable"): True,
                ("model", "feature_split", "enable"): True,
                ("model", "dann", "enable"): False,
                ("model", "ortho", "enable"): True,
            },
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser("Run CPDP ablation experiments")
    parser.add_argument("--config", type=str, default="src/configs/defaults.yaml")
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="logs/ablation")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    config_dir = os.path.join(args.log_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    experiments = build_experiments()

    for exp in experiments:
        exp_cfg = copy.deepcopy(base_cfg)
        for key_path, value in exp["overrides"].items():
            set_nested(exp_cfg, list(key_path), value)
        set_nested(exp_cfg, ["experiment", "run_name"], exp["name"])

        config_path = os.path.join(config_dir, f"{exp['name']}.yaml")
        save_yaml(exp_cfg, config_path)

        log_path = os.path.join(args.log_dir, f"{exp['name']}.log")
        command = [
            sys.executable,
            "src/run_experiment.py",
            "--config",
            config_path,
            "--project_dir",
            args.project_dir,
            "--log_path",
            log_path,
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
