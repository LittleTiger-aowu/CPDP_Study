from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
import transformers
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from src.data.collate import Collator, CollateConfig
from src.data.dataset import CPDPDataset
from src.models.cpdp_model import CPDPModel
from src.models.lora import apply_lora, LoRALinear

transformers.logging.set_verbosity_error()


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


def _move_batch(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return moved


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    pf = fp / (fp + tn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    term1 = np.float64(tp + fp)
    term2 = np.float64(tp + fn)
    term3 = np.float64(tn + fp)
    term4 = np.float64(tn + fn)
    mcc_denom = np.sqrt(term1 * term2 * term3 * term4 + 1e-12)
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom

    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_score)) + 1
    pos = y_true == 1
    num_pos = np.sum(pos)
    num_neg = len(y_true) - num_pos
    if num_pos == 0 or num_neg == 0:
        auc = 0.0
    else:
        sum_ranks = np.sum(ranks[pos])
        auc = (sum_ranks - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg)

    return {
        "f1": float(f1),
        "mcc": float(mcc),
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "pf": float(pf),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, cfg: Dict, device: torch.device) -> Dict[str, float]:
    model.eval()
    label_key = cfg.get("data", {}).get("label_key", "target")

    all_labels = []
    all_scores = []
    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(batch, cfg, epoch_idx=0, grl_lambda=0.0)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_scores.append(probs.detach().cpu().numpy())
        all_labels.append(batch[label_key].detach().cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_score = np.concatenate(all_scores, axis=0)
    return _binary_metrics(y_true, y_score)


def train_source_only(
    model: torch.nn.Module,
    source_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Dict,
    device: torch.device,
    save_path: str,
) -> None:
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 1))
    eval_every = max(1, int(train_cfg.get("eval_every_epochs", 1)))
    early_cfg = train_cfg.get("early_stopping", {})
    early_enable = bool(early_cfg.get("enable", False))
    early_patience = int(early_cfg.get("patience", 0))
    early_metric = str(early_cfg.get("metric", "auc")).lower()
    save_best = bool(cfg.get("experiment", {}).get("save_best", True))
    best_metric = float("-inf")
    patience_counter = 0

    label_key = cfg.get("data", {}).get("label_key", "target")
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    log_every_steps = int(cfg.get("logging", {}).get("log_every_steps", 0))
    use_bf16 = bool(train_cfg.get("bf16", False))
    use_fp16 = bool(train_cfg.get("fp16", False))

    class_weights = cfg.get("train", {}).get("class_weights")
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion_cls = torch.nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    else:
        criterion_cls = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if cfg.get("model", {}).get("dann", {}).get("enable", False):
        raise ValueError("Source-only baseline does not allow DANN/CDAN alignment. Disable model.dann.enable.")

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_amp = torch.cuda.is_available() and (use_bf16 or use_fp16)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 and torch.cuda.is_available() else None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        num_steps = 0

        for step_idx, batch in enumerate(source_loader, start=1):
            batch = _move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = model(batch, cfg, epoch_idx=epoch, grl_lambda=0.0)
                logits = outputs["logits"]
                if "am_logits" in outputs:
                    loss_cls = criterion_cls(outputs["am_logits"], batch[label_key])
                else:
                    loss_cls = criterion_cls(logits, batch[label_key])
                loss = loss_cls / max(1, grad_accum_steps)

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step_idx % grad_accum_steps == 0:
                if max_grad_norm > 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            num_steps += 1
            if log_every_steps > 0 and step_idx % log_every_steps == 0:
                logging.info("Epoch %d step %d loss=%.6f", epoch + 1, step_idx, loss.item())

        if step_idx % grad_accum_steps != 0:
            if max_grad_norm > 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if eval_every > 0 and (epoch + 1) % eval_every != 0:
            continue

        metrics = evaluate(model, valid_loader, cfg, device)
        logging.info(
            "Epoch %d Valid: F1=%.4f, MCC=%.4f, AUC=%.4f",
            epoch + 1,
            metrics["f1"],
            metrics["mcc"],
            metrics["auc"],
        )
        current_metric = metrics.get(early_metric, metrics.get("auc", 0.0))
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            if save_best:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if early_enable and early_patience > 0 and patience_counter >= early_patience:
            logging.info("Early stopping triggered at epoch %d.", epoch + 1)
            break


def main() -> None:
    parser = argparse.ArgumentParser("CodeBERT Naive Transfer (source-only) runner")
    parser.add_argument("--config", type=str, default="src/configs/naive_transfer_codebert_source_only.yaml")
    parser.add_argument("--log_path", type=str, default="logs/naive_transfer_codebert_source_only.log")
    parser.add_argument("--project_dir", type=str, required=True)
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

    data_cfg = cfg.get("data", {})
    data_paths = {
        "train_source": data_cfg.get("train_jsonl") or os.path.join(args.project_dir, "train.jsonl"),
        "valid": data_cfg.get("valid_jsonl") or os.path.join(args.project_dir, "valid_src.jsonl"),
        "test": data_cfg.get("test_jsonl") or os.path.join(args.project_dir, "test_tgt.jsonl"),
    }

    if not _check_data_paths(data_paths):
        logging.error("Please ensure all required JSONL files exist before running.")
        return

    cfg.setdefault("data", {})
    cfg["data"]["train_jsonl"] = data_paths["train_source"]
    cfg["data"]["valid_jsonl"] = data_paths["valid"]
    cfg["data"]["test_jsonl"] = data_paths["test"]

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

    model_path = cfg.get("model", {}).get("encoder", {}).get("pretrained_path", "microsoft/codebert-base")
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
    logging.info("Naive transfer (source-only) training started.")
    train_source_only(
        model=model,
        source_loader=train_source_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        save_path=save_path,
    )

    test_metrics = evaluate(model, test_loader, cfg, device)
    logging.info("Final Test Metrics: %s", test_metrics)

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


if __name__ == "__main__":
    main()
