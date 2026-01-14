from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, Iterator, List
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm  # <---【新增 1】引入进度条库

from src.losses.orthogonal import orthogonal_loss


@dataclass
class TrainState:
    best_f1: float = 0.0
    best_mcc: float = 0.0
    history: List[Dict[str, float]] = field(default_factory=list)


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


def _next_batch(loader: DataLoader, iterator: Iterator[Dict]) -> Tuple[Dict, Iterator[Dict]]:
    try:
        batch = next(iterator)
        return batch, iterator
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
        return batch, iterator


def _iter_pairs(
        source_loader: DataLoader,
        target_loader: DataLoader,
) -> Iterable[Tuple[Dict, Dict]]:
    src_iter = iter(source_loader)
    tgt_iter = iter(target_loader)
    steps = max(len(source_loader), len(target_loader))

    for _ in range(steps):
        src_batch, src_iter = _next_batch(source_loader, src_iter)
        tgt_batch, tgt_iter = _next_batch(target_loader, tgt_iter)
        yield src_batch, tgt_batch


def _move_batch(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return moved


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
    step_idx = 0

    label_key = cfg.get("data", {}).get("label_key", "target")
    train_cfg = cfg.get("train", {})
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    log_every_steps = int(cfg.get("logging", {}).get("log_every_steps", 0))
    use_bf16 = bool(train_cfg.get("bf16", False))
    use_fp16 = bool(train_cfg.get("fp16", False))
    dann_cfg = cfg["model"].get("dann", {})
    use_dann = dann_cfg.get("enable", False) and dann_cfg.get("weight", 0.0) > 0
    dann_weight = float(dann_cfg.get("weight", 0.0))

    ortho_cfg = cfg["model"].get("ortho", {})
    use_ortho = ortho_cfg.get("enable", False) and ortho_cfg.get("weight", 0.0) > 0
    ortho_weight = float(ortho_cfg.get("weight", 0.0))
    ortho_mode = ortho_cfg.get("mode", "corr")

    grl_lambda = compute_grl_lambda(epoch_idx, cfg)
    class_weights = cfg.get("train", {}).get("class_weights")
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion_cls = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    else:
        criterion_cls = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_dom = nn.CrossEntropyLoss()

    # ---【新增 2】计算总步数并包装进度条 ---
    steps = max(len(source_loader), len(target_loader))
    progress_bar = tqdm(_iter_pairs(source_loader, target_loader), total=steps, desc=f"Epoch {epoch_idx + 1}")

    # 使用 progress_bar 替代原来的 _iter_pairs(...)
    optimizer.zero_grad(set_to_none=True)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_amp = torch.cuda.is_available() and (use_bf16 or use_fp16)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 and torch.cuda.is_available() else None
    for src_batch, tgt_batch in progress_bar:
        src_batch = _move_batch(src_batch, device)
        tgt_batch = _move_batch(tgt_batch, device)

        step_idx += 1

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            src_out = model(src_batch, cfg, epoch_idx=epoch_idx, grl_lambda=grl_lambda)
            tgt_out = model(tgt_batch, cfg, epoch_idx=epoch_idx, grl_lambda=grl_lambda)

            src_labels = src_batch[label_key]
            logits = src_out["logits"]

            if "am_logits" in src_out:
                loss_cls = criterion_cls(src_out["am_logits"], src_labels)
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
            loss = loss / max(1, grad_accum_steps)

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

        # ---【新增 3】实时更新进度条上的 Loss 显示 ---
        progress_bar.set_postfix(loss=loss.item())
        if log_every_steps > 0 and step_idx % log_every_steps == 0:
            logging.info("Epoch %d step %d/%d loss=%.6f", epoch_idx + 1, step_idx, steps, loss.item())

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

    return total_loss / max(1, num_steps)


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    自动在验证集概率上搜索最佳 F1 阈值，并返回对应指标。
    """
    from sklearn.metrics import roc_auc_score

    # 1. 预计算 AUC (AUC 与阈值无关，算一次即可)
    num_pos = np.sum(y_true == 1)
    num_neg = np.sum(y_true == 0)
    if num_pos == 0 or num_neg == 0:
        auc = 0.0
    else:
        # 使用 argsort 计算 AUC
        order = np.argsort(y_score)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(y_score)) + 1
        sum_ranks = np.sum(ranks[y_true == 1])
        auc = (sum_ranks - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg)

    # 2. 动态搜索最佳阈值 (0.01 到 0.60，步长 0.01)
    best_f1 = -1.0
    best_metrics = {}

    # 跨项目场景下，由于分布偏移，最佳阈值通常偏低 (0.01-0.4 之间)
    thresholds = np.linspace(0.01, 0.60, 60)

    for th in thresholds:
        y_pred = (y_score >= th).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        if f1 > best_f1:
            best_f1 = f1
            # 计算当前最佳 F1 下的 MCC
            term1 = np.float64(tp + fp)
            term2 = np.float64(tp + fn)
            term3 = np.float64(tn + fp)
            term4 = np.float64(tn + fn)
            mcc_denom = np.sqrt(term1 * term2 * term3 * term4 + 1e-12)
            mcc = ((tp * tn) - (fp * fn)) / mcc_denom

            best_metrics = {
                "f1": float(f1),
                "mcc": float(mcc),
                "auc": float(auc),
                "precision": float(precision),
                "recall": float(recall),
                "accuracy": float((tp + tn) / (len(y_true) + 1e-12)),
                "pf": float(fp / (fp + tn + 1e-12)),
                "best_threshold": float(th)  # 记录是在哪个阈值下拿到的
            }

    return best_metrics


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

    # 你也可以在这里加一个 tqdm，不过验证集通常跑得快，不加也行
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
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 1))
    eval_every = max(1, int(train_cfg.get("eval_every_epochs", 1)))
    early_cfg = train_cfg.get("early_stopping", {})
    early_enable = bool(early_cfg.get("enable", False))
    early_patience = int(early_cfg.get("patience", 0))
    early_metric = str(early_cfg.get("metric", "auc")).lower()
    save_best = bool(cfg.get("experiment", {}).get("save_best", True))
    state = TrainState()
    history: List[Dict[str, float]] = []
    best_metric = float("-inf")
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_start_ts = datetime.now().isoformat(timespec="seconds")
        train_loss = train_one_epoch(
            model=model,
            source_loader=source_loader,
            target_loader=target_loader,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            epoch_idx=epoch,
        )
        epoch_end_ts = datetime.now().isoformat(timespec="seconds")
        epoch_time = time.time() - epoch_start
        lr = float(optimizer.param_groups[0].get("lr", 0.0)) if optimizer.param_groups else 0.0
        grl_lambda = compute_grl_lambda(epoch, cfg)
        record = {
            "epoch": epoch + 1,
            "epoch_start_time": epoch_start_ts,
            "epoch_end_time": epoch_end_ts,
            "train_loss": float(train_loss),
            "epoch_time_sec": float(epoch_time),
            "grl_lambda": float(grl_lambda),
            "lr": float(lr),
        }

        if eval_every > 0 and (epoch + 1) % eval_every != 0:
            history.append(record)
            continue

        metrics = evaluate(model, valid_loader, cfg, device)
        improved = False

        # 打印当前 Epoch 的评估结果
        print(f"Epoch {epoch + 1} Valid: F1={metrics['f1']:.4f}, MCC={metrics['mcc']:.4f}, AUC={metrics['auc']:.4f}")

        if metrics["f1"] > state.best_f1:
            state.best_f1 = metrics["f1"]
            improved = True
        if metrics["mcc"] > state.best_mcc:
            state.best_mcc = metrics["mcc"]
            improved = True
        current_metric = metrics.get(early_metric, metrics.get("auc", 0.0))
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            if save_best:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if improved and not save_best:
            torch.save(model.state_dict(), save_path)
        record.update(
            {
                "val_f1": float(metrics.get("f1", 0.0)),
                "val_mcc": float(metrics.get("mcc", 0.0)),
                "val_auc": float(metrics.get("auc", 0.0)),
                "val_precision": float(metrics.get("precision", 0.0)),
                "val_recall": float(metrics.get("recall", 0.0)),
                "val_accuracy": float(metrics.get("accuracy", 0.0)),
                "val_pf": float(metrics.get("pf", 0.0)),
                "val_best_threshold": float(metrics.get("best_threshold", 0.0)),
                "best_f1_so_far": float(state.best_f1),
                "best_mcc_so_far": float(state.best_mcc),
            }
        )
        history.append(record)

        if early_enable and early_patience > 0 and patience_counter >= early_patience:
            logging.info("Early stopping triggered at epoch %d.", epoch + 1)
            break

    state.history = history
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
