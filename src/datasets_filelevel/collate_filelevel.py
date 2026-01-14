from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from src.utils.chunking import chunk_text


@dataclass
class FileCollateConfig:
    pad_token_id: int = 1
    label_key: str = "label"
    domain_key: str = "domain"
    max_len: int = 256
    stride: int = 128
    max_windows: int = 16


class FileLevelCollator:
    def __init__(self, cfg: FileCollateConfig, tokenizer) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(samples) == 0:
            raise ValueError("Empty samples passed to collate_fn")

        input_ids_list = []
        attention_mask_list = []
        win_masks = []
        labels = []
        domain_labels = []
        methods = []
        locs = []

        for sample in samples:
            code = sample.get("code", "")
            input_ids, attention_mask, win_mask = chunk_text(
                tokenizer=self.tokenizer,
                text=code,
                max_len=self.cfg.max_len,
                stride=self.cfg.stride,
                max_windows=self.cfg.max_windows,
            )
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            win_masks.append(win_mask)

            labels.append(int(sample.get(self.cfg.label_key, 0)))
            domain_labels.append(int(sample.get(self.cfg.domain_key, 0)))
            locs.append(int(sample.get("loc", 1)))
            if "methods" in sample:
                methods.append(sample["methods"])

        batch = {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "win_mask": torch.stack(win_masks, dim=0),
            self.cfg.label_key: torch.tensor(labels, dtype=torch.long),
            "domain_labels": torch.tensor(domain_labels, dtype=torch.long),
            "loc": torch.tensor(locs, dtype=torch.float32),
        }
        if methods:
            batch["methods"] = methods
        return batch
