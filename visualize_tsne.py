from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.collate import Collator, CollateConfig
from src.data.dataset import CPDPDataset
from src.models.cpdp_model import CPDPModel


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    try:
        logging.info("Loading tokenizer from local cache: %s", model_name)
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except (OSError, ValueError) as exc:
        logging.warning("Local tokenizer load failed: %s", exc)
        logging.info("Falling back to online tokenizer load: %s", model_name)
        return AutoTokenizer.from_pretrained(model_name, local_files_only=False)


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def _collect_features(
    model: CPDPModel,
    loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    max_samples: int,
    domain_label: int,
) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    domains: List[np.ndarray] = []
    collected = 0

    for batch in loader:
        if collected >= max_samples:
            break
        batch = _move_batch(batch, device)
        outputs = model(batch, cfg, epoch_idx=0, grl_lambda=0.0)
        shared = outputs["features_shared"].detach().cpu().numpy()
        batch_size = shared.shape[0]
        remaining = max_samples - collected
        if batch_size > remaining:
            shared = shared[:remaining]
            batch_size = shared.shape[0]
        features.append(shared)
        domains.append(np.full(batch_size, domain_label, dtype=np.int64))
        collected += batch_size

    if not features:
        return np.empty((0, 0)), np.empty((0,), dtype=np.int64)

    return np.concatenate(features, axis=0), np.concatenate(domains, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser("Visualize shared features with t-SNE")
    parser.add_argument("--config", type=str, default="src/configs/defaults.yaml")
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="logs/best_model.pt")
    parser.add_argument("--output", type=str, default="tsne_shared_features.pdf")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=2000)
    args = parser.parse_args()

    setup_logging()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg.get("data", {})
    label_key = data_cfg.get("label_key", "target")
    domain_key = data_cfg.get("domain_key", "domain")
    code_key = data_cfg.get("code_key", "code")
    code_key_fallbacks = data_cfg.get("code_key_fallbacks", ["func"])
    domain_key_fallbacks = data_cfg.get("domain_key_fallbacks", [])
    domain_map = data_cfg.get("domain_map", None)
    ast_cache_dir = data_cfg.get("ast_cache_dir")
    ast_cache_fallback_to_jsonl = bool(data_cfg.get("ast_cache_fallback_to_jsonl", True))

    data_paths = {
        "source": os.path.join(args.project_dir, "valid_src.jsonl"),
        "target": os.path.join(args.project_dir, "test_tgt.jsonl"),
    }

    tokenizer = _load_tokenizer("microsoft/codebert-base")
    use_ast = cfg.get("model", {}).get("ast", {}).get("enable", False)
    max_length = cfg.get("model", {}).get("encoder", {}).get("max_length", 224)

    source_set = CPDPDataset(
        data_path=data_paths["source"],
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
    target_set = CPDPDataset(
        data_path=data_paths["target"],
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

    loader_kwargs = {"num_workers": 0, "collate_fn": collator}
    source_loader = DataLoader(source_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    target_loader = DataLoader(target_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = CPDPModel(cfg).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        source_features, source_domains = _collect_features(
            model,
            source_loader,
            cfg,
            device,
            max_samples=args.max_samples,
            domain_label=0,
        )
        target_features, target_domains = _collect_features(
            model,
            target_loader,
            cfg,
            device,
            max_samples=args.max_samples,
            domain_label=1,
        )

    if source_features.size == 0 or target_features.size == 0:
        raise ValueError("No features collected. Check the dataset paths and configuration.")

    features = np.concatenate([source_features, target_features], axis=0)
    domains = np.concatenate([source_domains, target_domains], axis=0)

    total_samples = features.shape[0]
    if total_samples < 2:
        raise ValueError("Not enough samples for t-SNE visualization.")

    max_perplexity = max(5, (total_samples - 1) // 3)
    perplexity = min(30, max_perplexity)
    tsne = TSNE(n_components=2, random_state=42, init="random", perplexity=perplexity)
    embedded = tsne.fit_transform(features)

    source_mask = domains == 0
    target_mask = domains == 1

    plt.figure(figsize=(7, 6))
    plt.scatter(embedded[source_mask, 0], embedded[source_mask, 1], s=10, alpha=0.7, label="Source")
    plt.scatter(embedded[target_mask, 0], embedded[target_mask, 1], s=10, alpha=0.7, label="Target")
    plt.legend()
    plt.title("t-SNE of Shared Features (Source vs Target)")
    plt.tight_layout()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(args.output)
    logging.info("Saved t-SNE plot to %s", args.output)


if __name__ == "__main__":
    main()
