from __future__ import annotations

import argparse
import logging
import os
import random
from typing import Dict, Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.collate import Collator, CollateConfig
from src.data.dataset import CPDPDataset
from src.models.cpdp_model import CPDPModel
from src.models.lora import apply_lora, LoRALinear
from src.trainer import CPDPTrainer


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


def main() -> None:
    parser = argparse.ArgumentParser("CPDP Experiment Runner")
    parser.add_argument("--config", type=str, default="src/configs/defaults.yaml")
    parser.add_argument("--log_path", type=str, default="logs/experiment.log")
    args = parser.parse_args()

    setup_logging(args.log_path)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    cfg = load_yaml(args.config)

    data_paths = {
        "train_source": "src/data/train.jsonl",
        "train_target": "src/data/test.jsonl",
        "valid": "src/data/valid.jsonl",
        "test": "src/data/test.jsonl",
    }

    if not _check_data_paths(data_paths):
        logging.error("Please ensure all required JSONL files exist before running.")
        return

    cfg.setdefault("data", {})
    cfg["data"]["train_jsonl"] = data_paths["train_source"]
    cfg["data"]["valid_jsonl"] = data_paths["valid"]
    cfg["data"]["test_jsonl"] = data_paths["test"]

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    use_ast = cfg.get("model", {}).get("ast", {}).get("enable", False)
    max_length = cfg.get("model", {}).get("encoder", {}).get("max_length", 224)
    label_key = cfg.get("data", {}).get("label_key", "target")
    domain_key = cfg.get("data", {}).get("domain_key", "domain")

    train_source_set = CPDPDataset(
        data_path=data_paths["train_source"],
        tokenizer=tokenizer,
        max_length=max_length,
        label_key=label_key,
        domain_key=domain_key,
        use_ast=use_ast,
    )
    train_target_set = CPDPDataset(
        data_path=data_paths["train_target"],
        tokenizer=tokenizer,
        max_length=max_length,
        label_key=label_key,
        domain_key=domain_key,
        use_ast=use_ast,
    )
    valid_set = CPDPDataset(
        data_path=data_paths["valid"],
        tokenizer=tokenizer,
        max_length=max_length,
        label_key=label_key,
        domain_key=domain_key,
        use_ast=use_ast,
    )
    test_set = CPDPDataset(
        data_path=data_paths["test"],
        tokenizer=tokenizer,
        max_length=max_length,
        label_key=label_key,
        domain_key=domain_key,
        use_ast=use_ast,
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

    loader_kwargs = {"num_workers": num_workers, "collate_fn": collator}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_source_loader = DataLoader(
        train_source_set,
        batch_size=batch_size,
        shuffle=True,
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

    save_path = os.path.join("logs", "best_model.pt")
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


if __name__ == "__main__":
    main()
