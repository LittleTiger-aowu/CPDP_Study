from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import time
from collections import Counter
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from src.datasets_filelevel.collate_filelevel import FileCollateConfig, FileLevelCollator
from src.datasets_filelevel.file_jsonl_dataset import FileJsonlDataset
from src.models_filelevel.file_model_wrapper import FileModelWrapper
from src.trainer import CPDPTrainer, evaluate


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


def _extract_labels(dataset: FileJsonlDataset, label_key: str) -> List[int]:
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


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    try:
        logging.info("Loading tokenizer from local cache: %s", model_name)
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except (OSError, ValueError) as exc:
        logging.warning("Local tokenizer load failed: %s", exc)
        logging.info("Falling back to online tokenizer load: %s", model_name)
        return AutoTokenizer.from_pretrained(model_name, local_files_only=False)


def _save_metrics_csv(path: str, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def _save_epoch_metrics_csv(path: str, history: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not history:
        return
    fieldnames = [
        "epoch",
        "epoch_start_time",
        "epoch_end_time",
        "train_loss",
        "epoch_time_sec",
        "grl_lambda",
        "lr",
        "val_f1",
        "val_mcc",
        "val_auc",
        "val_pr_auc",
        "val_precision",
        "val_recall",
        "val_balanced_acc",
        "val_g_mean",
        "val_accuracy",
        "val_pf",
        "val_best_threshold",
        "val_recall_at_20_effort",
        "val_precision_at_20_effort",
        "best_f1_so_far",
        "best_mcc_so_far",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _create_run_dir(exp_cfg: Dict[str, Any]) -> str:
    output_dir = exp_cfg.get("output_dir", "experiments")
    run_name = exp_cfg.get("run_name", "run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser("Train Defects4J file-level CPDP")
    parser.add_argument("--config", type=str, default="src/configs/d4j_file/base.yaml")
    parser.add_argument("--log_path", type=str, default="logs/d4j_file_cpdp.log")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    exp_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})
    file_cfg = cfg.get("file_level", {})

    run_dir = _create_run_dir(exp_cfg)
    log_path = os.path.join(run_dir, os.path.basename(args.log_path))
    setup_logging(log_path)
    config_snapshot = os.path.join(run_dir, "config.yaml")
    with open(config_snapshot, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    set_seed(int(exp_cfg.get("seed", 2026)))
    device = _resolve_device(exp_cfg.get("device", "auto"))

    tokenizer_name = cfg["model"]["encoder"].get("pretrained_path", "microsoft/codebert-base")
    tokenizer = _load_tokenizer(tokenizer_name)

    domain_map = {"source": 0, "target": 1}
    train_dataset = FileJsonlDataset(
        data_path=data_cfg["train_jsonl"],
        label_key=data_cfg.get("label_key", "label"),
        domain_key=data_cfg.get("domain_key", "domain"),
        code_key=data_cfg.get("code_key", "code"),
        code_key_fallbacks=data_cfg.get("code_key_fallbacks", ["code"]),
        domain_map=domain_map,
    )
    valid_dataset = FileJsonlDataset(
        data_path=data_cfg["valid_jsonl"],
        label_key=data_cfg.get("label_key", "label"),
        domain_key=data_cfg.get("domain_key", "domain"),
        code_key=data_cfg.get("code_key", "code"),
        code_key_fallbacks=data_cfg.get("code_key_fallbacks", ["code"]),
        domain_map=domain_map,
    )
    target_dataset = FileJsonlDataset(
        data_path=data_cfg["target_jsonl"],
        label_key=data_cfg.get("label_key", "label"),
        domain_key=data_cfg.get("domain_key", "domain"),
        code_key=data_cfg.get("code_key", "code"),
        code_key_fallbacks=data_cfg.get("code_key_fallbacks", ["code"]),
        domain_map=domain_map,
    )
    test_dataset = FileJsonlDataset(
        data_path=data_cfg["test_jsonl"],
        label_key=data_cfg.get("label_key", "label"),
        domain_key=data_cfg.get("domain_key", "domain"),
        code_key=data_cfg.get("code_key", "code"),
        code_key_fallbacks=data_cfg.get("code_key_fallbacks", ["code"]),
        domain_map=domain_map,
    )

    collate_cfg = FileCollateConfig(
        pad_token_id=tokenizer.pad_token_id or 1,
        label_key=data_cfg.get("label_key", "label"),
        domain_key=data_cfg.get("domain_key", "domain"),
        max_len=int(file_cfg.get("max_len", 256)),
        stride=int(file_cfg.get("stride", 128)),
        max_windows=int(file_cfg.get("max_windows", 16)),
    )
    collator = FileLevelCollator(collate_cfg, tokenizer)

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))

    imbalance_cfg = train_cfg.get("imbalance", {})
    use_sampler = bool(imbalance_cfg.get("sampler", True))
    use_loss_weight = bool(imbalance_cfg.get("loss_weight", True))

    sampler = None
    if use_sampler:
        labels = _extract_labels(train_dataset, data_cfg.get("label_key", "label"))
        class_weights, sample_weights = _compute_class_weights(labels)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        if use_loss_weight:
            train_cfg["class_weights"] = class_weights

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collator,
        drop_last=False,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collator,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collator,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collator,
        drop_last=False,
    )

    model = FileModelWrapper(cfg)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    save_path_cfg = exp_cfg.get("save_path", "experiments/file_level_model.pt")
    save_basename = os.path.basename(save_path_cfg)
    save_path = os.path.join(run_dir, save_basename)
    trainer = CPDPTrainer(model=model, optimizer=optimizer, cfg=cfg, device=device, save_path=save_path)

    train_start = time.time()
    state = trainer.train(source_loader=train_loader, target_loader=target_loader, valid_loader=valid_loader)
    train_end = time.time()

    test_metrics = evaluate(model, test_loader, cfg, device)
    logging.info("Test metrics: %s", test_metrics)

    epoch_metrics_csv = os.path.join(run_dir, "epoch_metrics.csv")
    _save_epoch_metrics_csv(epoch_metrics_csv, state.history)

    results_csv_name = os.path.basename(cfg.get("logging", {}).get("results_csv", "final_metrics.csv"))
    results_csv = os.path.join(run_dir, results_csv_name)
    run_end = time.time()
    final_metrics = dict(test_metrics)
    final_metrics.update(
        {
            "best_f1": float(state.best_f1),
            "best_mcc": float(state.best_mcc),
            "train_duration_sec": float(train_end - train_start),
            "total_run_duration_sec": float(run_end - train_start),
        }
    )
    _save_metrics_csv(results_csv, final_metrics)


if __name__ == "__main__":
    main()
