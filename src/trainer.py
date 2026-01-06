from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.losses.orthogonal import orthogonal_loss


@dataclass
class TrainState:
    best_f1: float = 0.0
    best_mcc: float = 0.0


def compute_grl_lambda(epoch_idx: int, cfg: Dict) -> float:
    dann_cfg = cfg["model"].get("dann", {})
    grl_opts = dann_cfg.get("grl", {})
    schedule = grl_opts.get("schedule", "linear")
    warmup = int(grl_opts.get("warmup_epochs", 0))
    lambda_max = float(grl_opts.get("lambda_max", 1.0))

    if schedule == "constant":
        return lambda_max
    if warmup <= 0:
        return lambda_max

    progress = min(1.0, epoch_idx / warmup)
    return lambda_max * progress


def _iter_pairs(
    source_loader: DataLoader,
    target_loader: DataLoader,
) -> Iterable[Tuple[Dict, Dict]]:
    if len(source_loader) >= len(target_loader):
        src_iter = iter(source_loader)
        tgt_iter = cycle(target_loader)
        steps = len(source_loader)
    else:
        src_iter = cycle(source_loader)
        tgt_iter = iter(target_loader)
        steps = len(target_loader)

    for _ in range(steps):
        yield next(src_iter), next(tgt_iter)


def _move_batch(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return moved


def _apply_am_softmax_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    scale: float,
) -> torch.Tensor:
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)
    return logits - one_hot * (margin * scale)


def train_one_epoch(
    model: nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Dict,
    device: torch.device,
    epoch_idx: int,
) -> float:
    model.train()
    total_loss = 0.0
    num_steps = 0

    label_key = cfg.get("data", {}).get("label_key", "target")
    dann_cfg = cfg["model"].get("dann", {})
    use_dann = dann_cfg.get("enable", False) and dann_cfg.get("weight", 0.0) > 0
    dann_weight = float(dann_cfg.get("weight", 0.0))

    ortho_cfg = cfg["model"].get("ortho", {})
    use_ortho = ortho_cfg.get("enable", False) and ortho_cfg.get("weight", 0.0) > 0
    ortho_weight = float(ortho_cfg.get("weight", 0.0))
    ortho_mode = ortho_cfg.get("mode", "corr")

    grl_lambda = compute_grl_lambda(epoch_idx, cfg)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_dom = nn.CrossEntropyLoss()

    for src_batch, tgt_batch in _iter_pairs(source_loader, target_loader):
        src_batch = _move_batch(src_batch, device)
        tgt_batch = _move_batch(tgt_batch, device)

        optimizer.zero_grad(set_to_none=True)

        src_out = model(src_batch, cfg, epoch_idx=epoch_idx, grl_lambda=grl_lambda)
        tgt_out = model(tgt_batch, cfg, epoch_idx=epoch_idx, grl_lambda=grl_lambda)

        src_labels = src_batch[label_key]
        logits = src_out["logits"]

        if "am_logits" in src_out:
            am_logits = _apply_am_softmax_margin(
                src_out["am_logits"],
                src_labels,
                margin=model.classifier.am_m,
                scale=model.classifier.am_s,
            )
            loss_cls = criterion_cls(am_logits, src_labels)
        else:
            loss_cls = criterion_cls(logits, src_labels)

        loss_dom = torch.tensor(0.0, device=device)
        if use_dann:
            dom_src = src_out["domain_logits"]
            dom_tgt = tgt_out["domain_logits"]
            dom_labels_src = torch.zeros(dom_src.size(0), dtype=torch.long, device=device)
            dom_labels_tgt = torch.ones(dom_tgt.size(0), dtype=torch.long, device=device)
            loss_dom = criterion_dom(dom_src, dom_labels_src) + criterion_dom(dom_tgt, dom_labels_tgt)

        loss_ortho = torch.tensor(0.0, device=device)
        if use_ortho and src_out["features_private"] is not None:
            loss_ortho = orthogonal_loss(
                src_out["features_shared"], src_out["features_private"], mode=ortho_mode
            )
            if tgt_out["features_private"] is not None:
                loss_ortho = loss_ortho + orthogonal_loss(
                    tgt_out["features_shared"], tgt_out["features_private"], mode=ortho_mode
                )

        loss = loss_cls + dann_weight * loss_dom + ortho_weight * loss_ortho
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_steps += 1

    return total_loss / max(1, num_steps)


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-12)
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

    return {"f1": float(f1), "mcc": float(mcc), "auc": float(auc)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: Dict,
    device: torch.device,
) -> Dict[str, float]:
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


def train(
    model: nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Dict,
    device: torch.device,
    save_path: str,
) -> TrainState:
    epochs = int(cfg.get("train", {}).get("epochs", 1))
    state = TrainState()

    for epoch in range(epochs):
        train_one_epoch(
            model=model,
            source_loader=source_loader,
            target_loader=target_loader,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            epoch_idx=epoch,
        )

        metrics = evaluate(model, valid_loader, cfg, device)
        improved = False
        if metrics["f1"] > state.best_f1:
            state.best_f1 = metrics["f1"]
            improved = True
        if metrics["mcc"] > state.best_mcc:
            state.best_mcc = metrics["mcc"]
            improved = True
        if improved:
            torch.save(model.state_dict(), save_path)

    return state


class CPDPTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: Dict,
        device: torch.device,
        save_path: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device
        self.save_path = save_path

    def train(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> TrainState:
        return train(
            model=self.model,
            source_loader=source_loader,
            target_loader=target_loader,
            valid_loader=valid_loader,
            optimizer=self.optimizer,
            cfg=self.cfg,
            device=self.device,
            save_path=self.save_path,
        )

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        return evaluate(self.model, loader, self.cfg, self.device)
