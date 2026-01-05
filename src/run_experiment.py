import os
import argparse
import yaml
from datetime import datetime
from typing import Dict, Any, Union  # 添加 Union 导入

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (update or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base

def get_cfg(cfg: Dict[str, Any], path: str, default=None):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def setup_experiment_dir(base_dir: str, run_name: Union[str, None]) -> str:  # 修改类型注解
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def main():
    parser = argparse.ArgumentParser("CPDP Experiment Runner (Config Phase)")
    parser.add_argument("--defaults", type=str, default="configs/defaults.yaml")
    parser.add_argument("--ablations", type=str, default="configs/ablations.yaml")
    parser.add_argument("--ablation", type=str, default=None)  # 建议用 --ablation 更短
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.defaults)

    # apply ablation
    if args.ablation:
        abl = load_yaml(args.ablations)
        groups = abl.get("groups", abl)  # ✅ 兼容 groups/扁平两种格式

        if args.ablation not in groups:
            raise ValueError(f"Ablation '{args.ablation}' not found. Available: {list(groups.keys())}")

        deep_merge(cfg, groups[args.ablation])

    # output dir priority: CLI > cfg.experiment.output_dir > default
    base_dir = args.exp_dir or get_cfg(cfg, "experiment.output_dir", "experiments")
    run_name = args.run_name or get_cfg(cfg, "experiment.run_name", None)

    run_dir = setup_experiment_dir(base_dir, run_name)

    # persist merged config
    merged_path = os.path.join(run_dir, "config_merged.yaml")
    with open(merged_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # persist meta
    meta = {
        "defaults": args.defaults,
        "ablations": args.ablations,
        "ablation": args.ablation,
        "run_dir": run_dir,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "run_meta.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    print(f"[Init] run_dir: {run_dir}")
    print(f"[Init] merged config: {merged_path}")

if __name__ == "__main__":
    main()
