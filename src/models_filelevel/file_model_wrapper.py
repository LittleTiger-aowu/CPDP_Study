from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.classifier import ClassifierHead
from src.models.domain_disc import DomainDiscriminator
from src.models.feature_split import FeatureSplit
from src.models_filelevel.file_encoder import FileSemanticEncoder


class FileModelWrapper(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        enc_cfg = cfg["model"].get("encoder", {})
        file_cfg = cfg.get("file_level", {})
        window_pool = file_cfg.get("window_pool", "attn")

        self.semantic_encoder = FileSemanticEncoder(
            pretrained_path=enc_cfg.get("pretrained_path", "microsoft/codebert-base"),
            max_length=enc_cfg.get("max_length", 512),
            pooling=enc_cfg.get("pooling", "cls"),
            window_pool=window_pool,
        )
        merged_dim = getattr(self.semantic_encoder, "output_dim", 768)

        split_cfg = cfg["model"].get("feature_split", {})
        self.use_split = split_cfg.get("enable", True)
        if self.use_split:
            self.shared_dim = split_cfg.get("shared_dim", 256)
            self.private_dim = split_cfg.get("private_dim", 256)
            self.feature_splitter = FeatureSplit(
                in_dim=merged_dim,
                shared_dim=self.shared_dim,
                private_dim=self.private_dim,
            )
        else:
            self.shared_dim = merged_dim
            self.private_dim = 0
            self.feature_splitter = None

        clf_cfg = cfg["model"]["classifier"]
        clf_input_mode = split_cfg.get("clf_input", "shared")
        if not self.use_split and clf_input_mode in ["private", "concat"]:
            raise ValueError("Split disabled but classifier expects private/concat features.")

        if clf_input_mode == "shared":
            clf_in_dim = self.shared_dim
        elif clf_input_mode == "private":
            clf_in_dim = self.private_dim
        elif clf_input_mode == "concat":
            clf_in_dim = self.shared_dim + self.private_dim
        else:
            raise ValueError(f"Unknown clf_input mode: {clf_input_mode}")

        self.classifier = ClassifierHead(
            in_dim=clf_in_dim,
            num_classes=clf_cfg.get("num_classes", 2),
            loss_type=clf_cfg.get("loss_type", "ce"),
            am_s=clf_cfg.get("am_softmax", {}).get("scale", 30.0),
            am_m=clf_cfg.get("am_softmax", {}).get("margin", 0.35),
        )

        self.num_classes = clf_cfg.get("num_classes", 2)
        self.clf_input_mode = clf_input_mode

        dann_cfg = cfg["model"].get("dann", {})
        self.use_dann = dann_cfg.get("enable", False) and dann_cfg.get("weight", 0.0) > 0
        if self.use_dann:
            disc_opts = dann_cfg.get("disc", {})
            self.domain_disc = DomainDiscriminator(
                in_dim=self.shared_dim * self.num_classes,
                hidden_dim=disc_opts.get("hidden_dim", 1024),
                dropout=disc_opts.get("dropout", 0.1),
            )

    def forward(self, batch: dict, cfg: dict, epoch_idx: int = None, grl_lambda: Optional[float] = None) -> dict:
        if "input_ids" not in batch or "attention_mask" not in batch or "win_mask" not in batch:
            raise KeyError("Batch missing input_ids/attention_mask/win_mask")

        code_feat = self.semantic_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            win_mask=batch["win_mask"],
        )

        if self.use_split:
            h_s, h_p = self.feature_splitter(code_feat)
        else:
            h_s = code_feat
            h_p = None

        if self.use_split:
            if self.clf_input_mode == "shared":
                clf_in = h_s
            elif self.clf_input_mode == "private":
                clf_in = h_p
            else:
                clf_in = torch.cat([h_s, h_p], dim=-1)
        else:
            clf_in = code_feat

        label_key = cfg.get("data", {}).get("label_key", "labels") if isinstance(cfg, dict) else "labels"
        labels = batch.get(label_key, None)
        cls_out = self.classifier(clf_in, labels=labels)

        if isinstance(cls_out, tuple):
            logits, am_logits = cls_out
        else:
            logits = cls_out
            am_logits = None

        domain_logits = None
        if self.use_dann:
            if grl_lambda is None:
                raise ValueError("grl_lambda must be provided when CDAN is enabled.")
            softmax_probs = torch.softmax(logits, dim=-1)
            cdan_feat = torch.bmm(h_s.unsqueeze(2), softmax_probs.unsqueeze(1)).view(h_s.size(0), -1)
            domain_logits = self.domain_disc(cdan_feat, grl_lambda)

        output = {
            "logits": logits,
            "domain_logits": domain_logits,
            "features_shared": h_s,
            "features_private": h_p,
            "h": code_feat,
        }
        if am_logits is not None:
            output["am_logits"] = am_logits
        return output
