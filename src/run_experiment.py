from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser("CPDP Experiment Runner")
    parser.add_argument("--config", type=str, default="src/configs/defaults.yaml")
    parser.add_argument("--log_path", type=str, default="logs/experiment.log")
    parser.add_argument("--project_dir", type=str, required=True)
    args = parser.parse_args()

    setup_logging(args.log_path)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    cfg = load_yaml(args.config)

    data_paths = {
        "train_source": os.path.join(args.project_dir, "train.jsonl"),
        "train_target": os.path.join(args.project_dir, "valid_tgt_unlabeled.jsonl"),
        "valid": os.path.join(args.project_dir, "valid_src.jsonl"),
        "test": os.path.join(args.project_dir, "test_tgt.jsonl"),
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
    ast_cache_dir = data_cfg.get("ast_cache_dir")
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

    tokenizer = _load_tokenizer("microsoft/codebert-base")

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
        ast_cache_dir=ast_cache_dir,
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
        ast_cache_dir=ast_cache_dir,
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
        ast_cache_dir=ast_cache_dir,
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
        ast_cache_dir=ast_cache_dir,
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
