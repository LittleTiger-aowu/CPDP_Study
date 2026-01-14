import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FileJsonlDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        label_key: str = "label",
        domain_key: str = "domain",
        code_key: str = "code",
        code_key_fallbacks: Optional[List[str]] = None,
        domain_key_fallbacks: Optional[List[str]] = None,
        domain_map: Optional[Dict[str, int]] = None,
        default_domain_value: Optional[int] = None,
        in_memory: bool = True,
    ) -> None:
        super().__init__()
        self.label_key = label_key
        self.domain_key = domain_key
        fallback_code_keys = code_key_fallbacks or []
        self.code_keys = [code_key] + [k for k in fallback_code_keys if k != code_key]
        fallback_domain_keys = domain_key_fallbacks or []
        self.domain_keys = [domain_key] + [k for k in fallback_domain_keys if k != domain_key]
        self.domain_map = domain_map or {}
        self.default_domain_value = default_domain_value
        self.data: List[Dict[str, Any]] = []

        data_path = Path(data_path)
        logger.info("Loading file-level dataset from %s (in_memory=%s)", data_path, in_memory)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with data_path.open("r", encoding="utf-8") as f:
            if in_memory:
                self.data = [json.loads(line) for line in f if line.strip()]
            else:
                self.data = [json.loads(line) for line in f if line.strip()]

        logger.info("Loaded %d samples.", len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def _get_code_text(self, item: Dict[str, Any]) -> str:
        for key in self.code_keys:
            if key in item:
                value = item.get(key, "")
                if value is None:
                    continue
                if isinstance(value, str):
                    return value
                return str(value)
        return ""

    def _get_domain_value(self, item: Dict[str, Any]) -> int:
        raw_value = None
        for key in self.domain_keys:
            if key in item:
                raw_value = item.get(key)
                break

        if raw_value is None:
            if self.default_domain_value is not None:
                return int(self.default_domain_value)
            return 0

        if isinstance(raw_value, bool):
            return int(raw_value)
        if isinstance(raw_value, (int, float)):
            return int(raw_value)
        if isinstance(raw_value, str):
            if raw_value.isdigit() or (raw_value.startswith("-") and raw_value[1:].isdigit()):
                return int(raw_value)
            if raw_value in self.domain_map:
                return int(self.domain_map[raw_value])
            logger.warning(
                "Domain value '%s' could not be mapped. Using default (%s).",
                raw_value,
                self.default_domain_value if self.default_domain_value is not None else 0,
            )
            return int(self.default_domain_value) if self.default_domain_value is not None else 0

        return int(self.default_domain_value) if self.default_domain_value is not None else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        code = self._get_code_text(item)
        label = int(item.get(self.label_key, 0))
        domain_value = self._get_domain_value(item)
        loc = max(1, len([line for line in code.splitlines() if line.strip()]))

        sample = {
            "code": code,
            "label": label,
            "domain": domain_value,
            "unit_id": item.get("unit_id"),
            "loc": loc,
            "project": item.get("project"),
        }
        if "methods" in item:
            sample["methods"] = item["methods"]
        return sample
