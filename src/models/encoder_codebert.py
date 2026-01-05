# src/models/encoder_codebert.py
import os
import torch
import torch.nn as nn
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)


class CodeBertEncoder(nn.Module):
    """
    CodeBERT 编码器封装 (Final Enhanced Version).

    [Final Polish V2]
    1. 加载策略增强: 若本地路径存在但加载失败，直接报错，不再 Fallback 到 Online (防止掩盖文件损坏)。
    2. 调试信息增强: forward 重试失败时，输出包含 Shape/ModelName/ExceptionContext 的详细报错。
    3. 显式校验: 保持之前的 if raise 校验逻辑。
    """

    def __init__(
            self,
            pretrained_path: str,
            max_length: int = 512,
            pooling: str = "cls"
    ):
        """
        Args:
            pretrained_path: HuggingFace 模型名称或本地路径.
            max_length: (仅记录).
            pooling: 'cls' 或 'mean'.
        """
        super().__init__()
        self.pooling = pooling.lower()
        if self.pooling not in ["cls", "mean"]:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        # 1. 优先离线加载策略 (Enhanced)
        try:
            logger.info(f"[CodeBertEncoder] Attempting offline load from '{pretrained_path}'...")
            self.model = AutoModel.from_pretrained(
                pretrained_path,
                local_files_only=True,
                trust_remote_code=False
            )
        except (OSError, ValueError) as e:
            # [Fix 1] 更严格的本地判定
            # 只要路径存在 (无论是文件还是目录)，都视为用户意图是加载本地文件
            # 如果加载失败，说明文件损坏或格式不对，直接报错，不要尝试 Online (防止误下载同名 Hub 模型)
            if os.path.exists(pretrained_path):
                logger.error(
                    f"[CodeBertEncoder] Path '{pretrained_path}' exists locally but load failed. Check file integrity.")
                raise e
            else:
                # 只有当本地完全找不到该路径时，才认为是 Hub ID，尝试联网
                logger.warning(
                    f"[CodeBertEncoder] Offline load failed & path not found. Trying online load for '{pretrained_path}'...")
                self.model = AutoModel.from_pretrained(
                    pretrained_path,
                    local_files_only=False,
                    trust_remote_code=False
                )

        self.output_dim = self.model.config.hidden_size

        # Dropout
        dropout_prob = getattr(self.model.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            token_type_ids: torch.LongTensor = None
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids: [Batch, SeqLen]
            attention_mask: [Batch, SeqLen]
            token_type_ids: [Batch, SeqLen] (Optional)
        """
        # 1. 显式校验
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D, got {input_ids.dim()}D")

        if input_ids.shape != attention_mask.shape:
            raise ValueError(f"Shape mismatch: input_ids {input_ids.shape} vs attention_mask {attention_mask.shape}")

        if token_type_ids is not None:
            if token_type_ids.shape != input_ids.shape:
                raise ValueError(
                    f"Shape mismatch: token_type_ids {token_type_ids.shape} vs input_ids {input_ids.shape}")

        # 2. 构造输入与鲁棒调用 (Enhanced Retry)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        try:
            outputs = self.model(**inputs)
        except TypeError as first_err:
            # 仅当输入包含 token_type_ids 时才尝试重试
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
                try:
                    outputs = self.model(**inputs)
                except TypeError as retry_err:
                    # [Fix 2] 重试仍失败，抛出详细上下文
                    model_name = getattr(self.model, "name_or_path", "unknown")
                    raise RuntimeError(
                        f"[CodeBertEncoder] Forward failed after removing token_type_ids.\n"
                        f"Model: {model_name}\n"
                        f"Input Shapes: input_ids={input_ids.shape}\n"
                        f"Original Error: {first_err}\n"
                        f"Retry Error: {retry_err}"
                    ) from retry_err
            else:
                # 如果原本就没有 token_type_ids 却报错，直接抛出
                raise first_err

        last_hidden_state = outputs.last_hidden_state

        # 3. 池化策略
        if self.pooling == "cls":
            pooled_output = last_hidden_state[:, 0, :]

        elif self.pooling == "mean":
            # Broadcasting Masked Mean
            input_mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_embeddings / sum_mask

        # 4. Dropout
        pooled_output = self.dropout(pooled_output)

        return pooled_output