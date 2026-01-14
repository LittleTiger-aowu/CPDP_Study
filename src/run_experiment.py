#src/run_experiment.py
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import subprocess
from collections import Counter
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import yaml
import transformers
# --- 新增代码开始 ---
# 只有报错才显示，警告全部闭嘴
transformers.logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")
# --- 新增代码结束 ---
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from src.data.collate import Collator, CollateConfig
from src.data.dataset import CPDPDataset
from src.models.cpdp_model import CPDPModel
from src.models.lora import apply_lora, LoRALinear
from src.trainer import CPDPTrainer


def _resolve_device(device_cfg: str) -> torch.device:
    device_cfg = (device_cfg or "auto").lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_output_dir(cfg: Dict[str, Any], project_dir: str) -> str:
    exp_cfg = cfg.get("experiment", {})
    output_dir = exp_cfg.get("output_dir", "experiments")
    project_name = os.path.basename(os.path.abspath(project_dir))
    run_name = exp_cfg.get("run_name", "default")
    return os.path.join(output_dir, project_name, run_name)


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _check_data_paths(paths: Dict[str, str]) -> bool:
    ok = True
    for name, path in paths.items():
        try:
            with open(path, "r", encoding="utf-8"):
                pass
        except FileNotFoundError:
            logging.error("Missing data file for %s: %s", name, path)
            ok = False
    return ok


def _extract_labels(dataset: CPDPDataset, label_key: str) -> List[int]:
    labels = []
    for item in dataset.data:
        raw = item.get(label_key, 0)
        try:
            labels.append(int(raw))
        except (TypeError, ValueError):
            labels.append(0)
    return labels


def _compute_class_weights(labels: List[int], num_classes: int = 2) -> Tuple[List[float], List[float]]:
    counts = Counter(labels)
    total = sum(counts.values())
    class_weights = []
    for cls in range(num_classes):
        count = counts.get(cls, 0)
        if count <= 0:
            class_weights.append(0.0)
        else:
            class_weights.append(total / (num_classes * count))
    sample_weights = [class_weights[label] if label < num_classes else 0.0 for label in labels]
    return class_weights, sample_weights


def _validate_jsonl(
    path: str,
    required_keys: Dict[str, str],
    any_of_keys: Dict[str, list[str]] | None = None,
    optional_keys: Dict[str, str] | None = None,
) -> bool:
    ok = True
    any_of_keys = any_of_keys or {}
    optional_keys = optional_keys or {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.error("Invalid JSONL at %s line %d: %s", path, idx, exc)
                ok = False
                continue
            if not isinstance(obj, dict):
                logging.error("Invalid JSONL object at %s line %d (not a dict).", path, idx)
                ok = False
                continue
            for key, desc in required_keys.items():
                if key not in obj:
                    logging.error(
                        "Missing required key '%s' (%s) in %s line %d.",
                        key,
                        desc,
                        path,
                        idx,
                    )
                    ok = False
            for group_name, candidates in any_of_keys.items():
                if not any(key in obj for key in candidates):
                    logging.error(
                        "Missing required key '%s' (any of: %s) in %s line %d.",
                        group_name,
                        ", ".join(candidates),
                        path,
                        idx,
                    )
                    ok = False
            for key, desc in optional_keys.items():
                if key not in obj:
                    logging.debug(
                        "Optional key '%s' (%s) missing in %s line %d.",
                        key,
                        desc,
                        path,
                        idx,
                    )
    return ok


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    try:
        logging.info("Loading tokenizer from local cache: %s", model_name)
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except (OSError, ValueError) as exc:
        logging.warning("Local tokenizer load failed: %s", exc)
        logging.info("Falling back to online tokenizer load: %s", model_name)
        return AutoTokenizer.from_pretrained(model_name, local_files_only=False)


def _summarize_scores(
    probs: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
) -> Dict[str, float]:
    probs = probs.astype(float)
    labels = labels.astype(int)
    logits = logits.astype(float)
    pos_mask = labels == 1
    neg_mask = labels == 0
    return {
        "prob_mean": float(np.mean(probs)) if probs.size > 0 else float("nan"),
        "prob_std": float(np.std(probs)) if probs.size > 0 else float("nan"),
        "prob_p50": float(np.percentile(probs, 50)) if probs.size > 0 else float("nan"),
        "prob_p90": float(np.percentile(probs, 90)) if probs.size > 0 else float("nan"),
        "prob_p99": float(np.percentile(probs, 99)) if probs.size > 0 else float("nan"),
        "prob_pos_mean": float(np.mean(probs[pos_mask])) if np.any(pos_mask) else float("nan"),
        "prob_neg_mean": float(np.mean(probs[neg_mask])) if np.any(neg_mask) else float("nan"),
        "logit_mean": float(np.mean(logits)) if logits.size > 0 else float("nan"),
        "logit_std": float(np.std(logits)) if logits.size > 0 else float("nan"),
        "logit_p50": float(np.percentile(logits, 50)) if logits.size > 0 else float("nan"),
        "logit_p90": float(np.percentile(logits, 90)) if logits.size > 0 else float("nan"),
        "logit_p99": float(np.percentile(logits, 99)) if logits.size > 0 else float("nan"),
        "logit_pos_mean": float(np.mean(logits[pos_mask])) if np.any(pos_mask) else float("nan"),
        "logit_neg_mean": float(np.mean(logits[neg_mask])) if np.any(neg_mask) else float("nan"),
    }


def _save_score_artifacts(
    output_dir: str,
    rows: List[Dict[str, object]],
    probs: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    scores_path = os.path.join(output_dir, "test_scores.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(scores_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    stats = _summarize_scores(probs, labels, logits)
    stats_path = os.path.join(output_dir, "test_score_stats.csv")
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])


def _write_metadata(output_dir: str, payload: Dict[str, object]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser("CPDP Experiment Runner")
    parser.add_argument("--config", type=str, default="src/configs/defaults.yaml")
    parser.add_argument("--log_path", type=str, default="logs/experiment.log")
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--ast_cache_dir", type=str, default=None, help="AST cache directory. If not provided, will look for cache subdirectory in project_dir")
    args = parser.parse_args()

    setup_logging(args.log_path)

    cfg = load_yaml(args.config)
    exp_cfg = cfg.get("experiment", {})
    seed = int(exp_cfg.get("seed", 42))
    set_seed(seed)

    device = _resolve_device(exp_cfg.get("device", "auto"))
    logging.info("Using device: %s", device)
    output_dir = _resolve_output_dir(cfg, args.project_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 如果命令行提供了ast_cache_dir，则使用它，否则在项目目录下查找cache子目录
    ast_cache_dir = args.ast_cache_dir
    if not ast_cache_dir:
        # 尝试在项目目录下查找cache子目录
        possible_cache_dirs = [
            os.path.join(args.project_dir, "cache"),
            os.path.join(args.project_dir, "ast_cache"),
            os.path.join(args.project_dir, "data", "ast_cache")
        ]
        for cache_dir in possible_cache_dirs:
            if os.path.exists(cache_dir):
                ast_cache_dir = cache_dir
                logging.info("Found AST cache directory: %s", ast_cache_dir)
                break
        else:
            # 如果没找到，使用配置文件中的设置
            ast_cache_dir = cfg.get("data", {}).get("ast_cache_dir")
            logging.info("Using AST cache directory from config: %s", ast_cache_dir)
    else:
        logging.info("Using AST cache directory from command line: %s", ast_cache_dir)

    data_cfg = cfg.get("data", {})
    data_paths = {
        "train_source": data_cfg.get("train_jsonl") or os.path.join(args.project_dir, "train.jsonl"),
        "train_target": data_cfg.get("target_jsonl") or os.path.join(args.project_dir, "valid_tgt_unlabeled.jsonl"),
        "valid": data_cfg.get("valid_jsonl") or os.path.join(args.project_dir, "valid_src.jsonl"),
        "test": data_cfg.get("test_jsonl") or os.path.join(args.project_dir, "test_tgt.jsonl"),
    }

    if not _check_data_paths(data_paths):
        logging.error("Please ensure all required JSONL files exist before running.")
        return

    if os.path.abspath(data_paths["train_target"]) == os.path.abspath(data_paths["test"]):
        logging.error("Target-train and test JSONL paths must differ to avoid leakage.")
        return

    cfg.setdefault("data", {})
    cfg["data"]["train_jsonl"] = data_paths["train_source"]
    cfg["data"]["target_jsonl"] = data_paths["train_target"]
    cfg["data"]["valid_jsonl"] = data_paths["valid"]
    cfg["data"]["test_jsonl"] = data_paths["test"]

    data_cfg = cfg.get("data", {})
    label_key = data_cfg.get("label_key", "target")
    domain_key = data_cfg.get("domain_key", "domain")
    code_key = data_cfg.get("code_key", "code")
    code_key_fallbacks = data_cfg.get("code_key_fallbacks", ["func"])
    domain_key_fallbacks = data_cfg.get("domain_key_fallbacks", [])
    domain_map = data_cfg.get("domain_map", None)
    domain_key_required = bool(data_cfg.get("domain_key_required", False))
    # 优先使用命令行参数，然后是配置文件
    ast_cache_dir_final = ast_cache_dir or data_cfg.get("ast_cache_dir")
    ast_cache_fallback_to_jsonl = bool(data_cfg.get("ast_cache_fallback_to_jsonl", True))
    code_keys = [code_key] + [k for k in code_key_fallbacks if k != code_key]

    required_sets = {
        "train_source": {
            label_key: "defect label",
        },
        "valid": {
            label_key: "defect label",
        },
        "test": {
            label_key: "defect label",
        },
        "train_target": {
        },
    }
    any_of_keys = {
        name: {"code": code_keys}
        for name in required_sets
    }
    optional_domain = {domain_key: "domain label"} if not domain_key_required else {}
    if domain_key_required:
        for name in required_sets:
            required_sets[name][domain_key] = "domain label"

    for name, path in data_paths.items():
        if not _validate_jsonl(
            path,
            required_sets[name],
            any_of_keys=any_of_keys[name],
            optional_keys=optional_domain,
        ):
            logging.error("JSONL validation failed for %s (%s).", name, path)
            return

    # 1. 从配置(cfg)中读取你在 yaml 里写的路径
    model_path = cfg.get("model", {}).get("encoder", {}).get("pretrained_path", "microsoft/codebert-base")

    # 2. 使用读取到的路径加载 Tokenizer
    tokenizer = _load_tokenizer(model_path)

    use_ast = cfg.get("model", {}).get("ast", {}).get("enable", False)
    max_length = cfg.get("model", {}).get("encoder", {}).get("max_length", 224)

    train_source_set = CPDPDataset(
        data_path=data_paths["train_source"],
        tokenizer=tokenizer,
        max_length=max_length,
        label_key=label_key,
        domain_key=domain_key,
        code_key=code_key,
        code_key_fallbacks=code_key_fallbacks,
        domain_key_fallbacks=domain_key_fallbacks,
        domain_map=domain_map,
        default_domain_value=0,
        use_ast=use_ast,
        ast_cache_dir=ast_cache_dir_final,
        ast_cache_fallback_to_jsonl=ast_cache_fallback_to_jsonl,
    )
    train_target_set = CPDPDataset(
        data_path=data_paths["train_target"],
        tokenizer=tokenizer,
        max_length=max_length,
        label_key=label_key,
        domain_key=domain_key,
        code_key=code_key,
        code_key_fallbacks=code_key_fallbacks,
        domain_key_fallbacks=domain_key_fallbacks,
        domain_map=domain_map,
        default_domain_value=1,
        use_ast=use_ast,
        ast_cache_dir=ast_cache_dir_final,
        ast_cache_fallback_to_jsonl=ast_cache_fallback_to_jsonl,
    )
    valid_set = CPDPDataset(
        data_path=data_paths["valid"],
        tokenizer=tokenizer,
        max_length=max_length,
        label_key=label_key,
        domain_key=domain_key,
        code_key=code_key,
        code_key_fallbacks=code_key_fallbacks,
        domain_key_fallbacks=domain_key_fallbacks,
        domain_map=domain_map,
        default_domain_value=0,
        use_ast=use_ast,
        ast_cache_dir=ast_cache_dir_final,
        ast_cache_fallback_to_jsonl=ast_cache_fallback_to_jsonl,
    )
    test_set = CPDPDataset(
        data_path=data_paths["test"],
        tokenizer=tokenizer,
        max_length=max_length,
        label_key=label_key,
        domain_key=domain_key,
        code_key=code_key,
        code_key_fallbacks=code_key_fallbacks,
        domain_key_fallbacks=domain_key_fallbacks,
        domain_map=domain_map,
        default_domain_value=1,
        use_ast=use_ast,
        ast_cache_dir=ast_cache_dir_final,
        ast_cache_fallback_to_jsonl=ast_cache_fallback_to_jsonl,
    )

    collate_cfg = CollateConfig(
        pad_token_id=tokenizer.pad_token_id or 1,
        label_key=label_key,
        domain_key=domain_key,
        use_ast=use_ast,
    )
    collator = Collator(collate_cfg)

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(cfg.get("data", {}).get("num_workers", 0))
    prefetch_factor = cfg.get("data", {}).get("prefetch_factor", 2)
    imbalance_cfg = train_cfg.get("imbalance", {})
    imbalance_enabled = bool(imbalance_cfg.get("enable", False))
    imbalance_use_sampler = bool(imbalance_cfg.get("sampler", True))
    imbalance_use_loss_weight = bool(imbalance_cfg.get("loss_weight", True))
    train_sampler = None

    if imbalance_enabled:
        train_labels = _extract_labels(train_source_set, label_key)
        class_weights, sample_weights = _compute_class_weights(train_labels, num_classes=2)
        if imbalance_use_loss_weight:
            cfg.setdefault("train", {})["class_weights"] = class_weights
            logging.info("Class weights enabled: %s", class_weights)
        if imbalance_use_sampler:
            train_sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            logging.info("WeightedRandomSampler enabled with %d samples.", len(sample_weights))

    loader_kwargs = {"num_workers": num_workers, "collate_fn": collator}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_source_loader = DataLoader(
        train_source_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs,
    )
    train_target_loader = DataLoader(
        train_target_set,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = CPDPModel(cfg).to(device)
    lora_cfg = cfg.get("model", {}).get("lora", {})
    if lora_cfg.get("enable", False):
        already_applied = any(
            isinstance(module, LoRALinear) for module in model.code_encoder.modules()
        )
        if not already_applied:
            model.code_encoder = apply_lora(
                model.code_encoder,
                r=int(lora_cfg.get("r", 8)),
                alpha=float(lora_cfg.get("alpha", 16)),
                dropout=float(lora_cfg.get("dropout", 0.05)),
                target_modules=lora_cfg.get("target_modules", ["query", "value"]),
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    save_path = os.path.join(output_dir, "best_model.pt")
    trainer = CPDPTrainer(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        save_path=save_path,
    )

    trainer.train(
        source_loader=train_source_loader,
        target_loader=train_target_loader,
        valid_loader=valid_loader,
    )

    test_metrics = trainer.evaluate(test_loader)
    logging.info("Final Test Metrics: %s", test_metrics)

    model.eval()
    all_probs = []
    all_labels = []
    all_logits = []
    all_prob0 = []
    all_unit_ids = []
    all_projects = []
    all_domains = []
    all_locs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch, cfg, epoch_idx=0, grl_lambda=0.0)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            prob1 = probs[:, 1]
            prob0 = probs[:, 0]
            all_probs.append(prob1.detach().cpu().numpy())
            all_prob0.append(prob0.detach().cpu().numpy())
            all_logits.append(logits[:, 1].detach().cpu().numpy())
            label_key = cfg.get("data", {}).get("label_key", "target")
            if label_key in batch:
                all_labels.append(batch[label_key].detach().cpu().numpy())
            all_unit_ids.extend(batch.get("unit_id", [None] * len(prob1)))
            all_projects.extend(batch.get("project", [None] * len(prob1)))
            domain_key = cfg.get("data", {}).get("domain_key", "domain")
            if domain_key in batch:
                all_domains.extend(batch[domain_key].detach().cpu().numpy().tolist())
            else:
                all_domains.extend([None] * len(prob1))
            if "loc" in batch:
                all_locs.extend(batch["loc"].detach().cpu().numpy().tolist())
            else:
                all_locs.extend([1.0] * len(prob1))

    score_rows = None
    if all_probs and all_labels:
        probs_np = np.concatenate(all_probs, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        logits_np = np.concatenate(all_logits, axis=0)
        prob0_np = np.concatenate(all_prob0, axis=0)
        best_threshold = float(test_metrics.get("best_threshold", 0.5))
        rows = []
        for idx, (label, prob1, prob0, logit) in enumerate(
            zip(labels_np, probs_np, prob0_np, logits_np)
        ):
            row = {
                "idx": all_unit_ids[idx],
                "label": int(label),
                "prob": float(prob1),
                "prob0": float(prob0),
                "logit": float(logit),
                "pred_0p5": int(prob1 >= 0.5),
                "pred_best": int(prob1 >= best_threshold),
                "domain": all_domains[idx],
                "project": all_projects[idx],
                "split": "test",
                "loc": float(all_locs[idx]),
            }
            rows.append(row)
        _save_score_artifacts(output_dir, rows, probs_np, labels_np, logits_np)
        score_rows = rows

        git_sha = None
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            ).decode("utf-8").strip()
        except (OSError, subprocess.CalledProcessError):
            git_sha = None
        metadata = {
            "run_name": exp_cfg.get("run_name"),
            "seed": seed,
            "best_threshold": best_threshold,
            "output_dir": output_dir,
            "git_tag": git_sha,
        }
        _write_metadata(output_dir, metadata)

    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("run_name") or os.path.splitext(os.path.basename(args.config))[0]
    project_name = exp_cfg.get("project") or os.path.basename(os.path.abspath(args.project_dir))
    log_cfg = cfg.get("logging", {})
    results_csv = log_cfg.get("results_csv", "experiments/results.csv")
    results_path = results_csv
    if not os.path.isabs(results_csv):
        results_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), results_csv)
    results_row = {
        "Project": project_name,
        "F1": test_metrics.get("f1", 0.0),
        "MCC": test_metrics.get("mcc", 0.0),
        "AUC": test_metrics.get("auc", 0.0),
        "Recall": test_metrics.get("recall", 0.0),
        "PF": test_metrics.get("pf", 0.0),
        "Config_Name": exp_name,
    }

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    write_header = not os.path.exists(results_path) or os.path.getsize(results_path) == 0
    with open(results_path, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(results_row)

    if log_cfg.get("save_predictions", False) or log_cfg.get("save_embeddings", False):
        model.eval()
        all_probs = []
        all_labels = []
        all_embeddings = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(batch, cfg, epoch_idx=0, grl_lambda=0.0)
                probs = torch.softmax(outputs["logits"], dim=1)[:, 1]
                all_probs.append(probs.detach().cpu().numpy())
                label_key = cfg.get("data", {}).get("label_key", "target")
                if label_key in batch:
                    all_labels.append(batch[label_key].detach().cpu().numpy())
                if log_cfg.get("save_embeddings", False):
                    emb = outputs.get("features_shared")
                    if emb is not None:
                        all_embeddings.append(emb.detach().cpu().numpy())

        if log_cfg.get("save_predictions", False):
            pred_path = os.path.join(output_dir, "test_predictions.csv")
            if score_rows:
                fieldnames = list(score_rows[0].keys())
                with open(pred_path, "w", encoding="utf-8", newline="") as pred_file:
                    writer = csv.DictWriter(pred_file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(score_rows)
            else:
                probs_np = np.concatenate(all_probs, axis=0)
                labels_np = np.concatenate(all_labels, axis=0) if all_labels else None
                with open(pred_path, "w", encoding="utf-8", newline="") as pred_file:
                    writer = csv.writer(pred_file)
                    if labels_np is None:
                        writer.writerow(["prob"])
                        for prob in probs_np:
                            writer.writerow([float(prob)])
                    else:
                        writer.writerow(["label", "prob"])
                        for label, prob in zip(labels_np, probs_np):
                            writer.writerow([int(label), float(prob)])
            logging.info("Saved predictions to %s", pred_path)

        if log_cfg.get("save_embeddings", False) and all_embeddings:
            emb_path = os.path.join(output_dir, "test_embeddings.npy")
            np.save(emb_path, np.concatenate(all_embeddings, axis=0))
            logging.info("Saved embeddings to %s", emb_path)


if __name__ == "__main__":
    main()