from typing import Optional

import torch
import torch.nn as nn
import logging

# -----------------------------------------------------------------------------
# 模块导入
# -----------------------------------------------------------------------------
try:
    from src.models.encoder_codebert import CodeBertEncoder
    from src.models.lora import apply_lora
    from src.models.ast_encoder import ASTEncoder
    from src.models.feature_split import FeatureSplit
    from src.models.classifier import ClassifierHead
    from src.models.domain_disc import DomainDiscriminator
    from src.models.fusion import CrossModalFusion  # 新增导入
except ImportError as e:
    raise ImportError(f"CPDPModel 缺少必要子模块，请检查 src/models/ 目录是否完整: {e}")

# [修正1] 使用标准 Logger
logger = logging.getLogger(__name__)


class CPDPModel(nn.Module):
    """
    CPDP 核心模型组装类 (Route B: Disjoint Union Graph Support).

    [Final Polish V2]
    - Logging: 使用 logging 模块替代 print。
    - Config: 严格校验 encoder.name，防止实验配置名不副实。
    - Consistency: split关闭时 private_dim=0，与 forward 输出保持一致。
    """

    def __init__(self, cfg: dict):
        super().__init__()

        # 1. 初始化 Code Encoder
        # -----------------------------------------------------------
        enc_cfg = cfg["model"].get("encoder", {})

        # [修正2] 严格校验 Encoder 名称 (防止配置错配)
        # 当前实现仅支持 CodeBERT/RoBERTa 架构，若配置了其他名称(如 unixcoder)应直接报错
        expected_name = enc_cfg.get("name", "codebert").lower()
        if "codebert" not in expected_name and "roberta" not in expected_name:
            raise ValueError(
                f"Encoder config name '{expected_name}' implies a model architecture not supported by 'CodeBertEncoder'. "
                "Current implementation strictly wraps CodeBERT/RoBERTa."
            )

        self.code_encoder = CodeBertEncoder(
            pretrained_path=enc_cfg.get("pretrained_path", "microsoft/codebert-base"),
            max_length=enc_cfg.get("max_length", 512),
            pooling=enc_cfg.get("pooling", "cls")
        )
        self.code_dim = getattr(self.code_encoder, "output_dim", 768)

        # 1.1 冻结策略
        freeze_n = enc_cfg.get("freeze_n_layers", 0)
        if freeze_n > 0:
            self._freeze_layers(freeze_n)

        # 1.2 LoRA 注入
        lora_cfg = cfg["model"].get("lora", {})
        if lora_cfg.get("enable", False):
            self.code_encoder = apply_lora(
                self.code_encoder,
                r=lora_cfg.get("r", 8),
                alpha=lora_cfg.get("alpha", 16),
                dropout=lora_cfg.get("dropout", 0.05),
                target_modules=lora_cfg.get("target_modules", ["query", "value"])
            )

        # 2. 初始化 AST Encoder (适配 Route B)
        # -----------------------------------------------------------
        ast_cfg = cfg["model"].get("ast", {})
        self.use_ast = ast_cfg.get("enable", False)
        self.ast_fusion = ast_cfg.get("fusion", "concat")

        current_dim = self.code_dim

        if self.use_ast:
            ast_dim = ast_cfg.get("dim", 128)
            # ASTEncoder 内部根据 out_dim 初始化 GNN/MLP
            self.ast_encoder = ASTEncoder(out_dim=ast_dim)
            
            # 【修改点 1】初始化 Fusion 模块
            # 当使用 cross_attention 融合时，融合后的维度等于 fusion_dim
            if self.ast_fusion == "cross_attention":
                self.fusion_dim = ast_cfg.get("fusion_dim", 256)
                self.fusion_module = CrossModalFusion(
                    semantic_dim=self.code_dim,       # CodeBERT 维度
                    struct_dim=ast_dim, 
                    hidden_dim=self.fusion_dim
                )
                current_dim = self.fusion_dim  # 更新当前特征维度
            else:
                if self.ast_fusion == "concat":
                    current_dim += ast_dim
                elif self.ast_fusion == "sum":
                    if current_dim != ast_dim:
                        raise ValueError(f"AST fusion 'sum' requires dim match: Code({current_dim}) vs AST({ast_dim})")
                else:
                    raise ValueError(f"Unknown AST fusion mode: {self.ast_fusion}")

        # 3. 特征解耦 (Feature Split)
        # -----------------------------------------------------------
        split_cfg = cfg["model"].get("feature_split", {})
        self.use_split = split_cfg.get("enable", True)

        if self.use_split:
            self.shared_dim = split_cfg.get("shared_dim", 256)
            self.private_dim = split_cfg.get("private_dim", 256)

            # [修正1] 使用 Logger
            logger.info(
                f"[CPDPModel] Feature Split Enabled: Projecting In({current_dim}) -> Shared({self.shared_dim}) / Private({self.private_dim})")

            self.feature_splitter = FeatureSplit(
                in_dim=current_dim,
                shared_dim=self.shared_dim,
                private_dim=self.private_dim
            )
        else:
            # [修正3] 属性一致性: 未解耦时 private_dim 置 0
            self.shared_dim = current_dim
            self.private_dim = 0
            logger.info(f"[CPDPModel] Feature Split Disabled: Passthrough dim={current_dim}")

        # 4. 缺陷分类器
        # -----------------------------------------------------------
        clf_cfg = cfg["model"]["classifier"]
        clf_input_mode = split_cfg.get("clf_input", "shared")

        # 配置互斥检查: 未开启 Split 时，不允许使用 Private/Concat 模式
        if not self.use_split:
            if clf_input_mode in ["private", "concat"]:
                raise ValueError(
                    f"Configuration Error: feature_split.enable=False but clf_input='{clf_input_mode}'. "
                    "When split is disabled, only 'shared' (meaning original features) is allowed."
                )

        # 计算分类器输入维度
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

        # 5. 域对抗鉴别器 (CDAN)
        # -----------------------------------------------------------
        dann_cfg = cfg["model"].get("dann", {})
        # 严格激活条件: enable=True 且 weight>0
        self.use_dann = dann_cfg.get("enable", False) and dann_cfg.get("weight", 0.0) > 0

        if self.use_dann:
            disc_opts = dann_cfg.get("disc", {})
            self.domain_disc = DomainDiscriminator(
                in_dim=self.shared_dim * self.num_classes,
                hidden_dim=disc_opts.get("hidden_dim", 1024),
                dropout=disc_opts.get("dropout", 0.1)
            )

    def _freeze_layers(self, n_layers):
        """适配 HF Roberta/Bert 冻结"""
        base_model = getattr(self.code_encoder, "model", None) or getattr(self.code_encoder, "bert", None)

        if base_model and hasattr(base_model, "roberta"):
            encoder = base_model.roberta.encoder
            embeddings = base_model.roberta.embeddings
        elif base_model and hasattr(base_model, "encoder"):
            encoder = base_model.encoder
            embeddings = getattr(base_model, "embeddings", None)
        else:
            logger.warning("Could not locate encoder to freeze. Skipping.")
            return

        if hasattr(encoder, "layer"):
            frozen = 0
            for i in range(min(n_layers, len(encoder.layer))):
                for param in encoder.layer[i].parameters():
                    param.requires_grad = False
                frozen += 1
            if embeddings:
                for param in embeddings.parameters():
                    param.requires_grad = False
            logger.info(f"[CPDPModel] Frozen embeddings + {frozen} encoder layers.")

    def forward(self, batch: dict, cfg: dict, epoch_idx: int = None, grl_lambda: Optional[float] = None) -> dict:
        """
        Args:
            batch: Data batch (must contain ast_x/ast_edge_index/ast_batch if ast enabled)
            cfg: Global config
            epoch_idx: Used for GRL schedule
        """
        # 1. 文本编码
        if "input_ids" not in batch or "attention_mask" not in batch:
            raise KeyError("Batch missing 'input_ids' or 'attention_mask'.")

        code_out = self.code_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None)
        )
        
        # 【修改点 2】解包，获取序列特征和注意力权重
        code_seq = code_out["sequence"]  # [B, Seq, 768]
        code_pooled = code_out["pooled"]  # [B, 768]
        code_attentions = code_out.get("attentions", None)  # 可能为 None

        merged_feat = code_pooled

        # 2. AST 编码与融合 (Route B 实现)
        if self.use_ast:
            # 强校验: 检查 Route B 必需的图结构字段
            required_ast_keys = ["ast_x", "ast_edge_index", "ast_batch"]
            missing_keys = [k for k in required_ast_keys if k not in batch]
            if missing_keys:
                raise KeyError(
                    f"AST enabled but missing keys: {missing_keys}. "
                    f"Expected {required_ast_keys} in batch (Route B Disjoint Union format)."
                )

            # [修正点 1] 获取当前真实的 Batch Size (以 input_ids 为准，这是最可靠的)
            current_bs = batch["input_ids"].size(0)

            # [修正点 2] 显式传参，并传入 batch_size
            # 不要用 ast_in 字典再解包了，直接传参更清晰，且 IDE 能跳转
            ast_feat = self.ast_encoder(
                x=batch["ast_x"],
                edge_index=batch["ast_edge_index"],
                batch=batch["ast_batch"],
                batch_size=current_bs  # <--- 必须加这个，防止末尾空图导致崩坏
            )
            
            # 【修改点 3】使用 Cross Attention 融合
            if self.ast_fusion == "cross_attention":
                # 替代原来的 torch.cat，使用 Cross Attention 融合
                merged_feat = self.fusion_module(code_seq, ast_feat, code_attentions)  # -> [B, 256]
            elif self.ast_fusion == "concat":
                merged_feat = torch.cat([code_pooled, ast_feat], dim=-1)
            elif self.ast_fusion == "sum":
                merged_feat = code_pooled + ast_feat

        # 3. 特征解耦
        if self.use_split:
            h_s, h_p = self.feature_splitter(merged_feat)
        else:
            # Split 关闭时，h_p 为 None，h_s 为原特征
            h_s = merged_feat
            h_p = None

        # 4. 缺陷分类
        if self.use_split:
            if self.clf_input_mode == "shared":
                clf_in = h_s
            elif self.clf_input_mode == "private":
                clf_in = h_p
            else:
                clf_in = torch.cat([h_s, h_p], dim=-1)
        else:
            clf_in = merged_feat

        label_key = cfg.get("data", {}).get("label_key", "target") if isinstance(cfg, dict) else "target"
        labels = batch.get(label_key, None)
        cls_out = self.classifier(clf_in, labels=labels)

        # 拆包处理 (兼容 AM-Softmax 返回 tuple 的情况)
        if isinstance(cls_out, tuple):
            logits, am_logits = cls_out
        else:
            logits = cls_out
            am_logits = None

        # 5. 域对抗 (CDAN)
        domain_logits = None
        if self.use_dann:
            if grl_lambda is None:
                raise ValueError("grl_lambda must be provided when CDAN is enabled.")
            softmax_probs = torch.softmax(logits, dim=-1)
            cdan_feat = torch.bmm(h_s.unsqueeze(2), softmax_probs.unsqueeze(1)).view(h_s.size(0), -1)
            domain_logits = self.domain_disc(cdan_feat, grl_lambda)

        # 6. 构建输出 (清洗 None 值防止下游 crash)
        output = {
            "logits": logits,
            "features_shared": h_s,
            "features_private": h_p,
            "domain_logits": domain_logits
        }

        if am_logits is not None:
            output["am_logits"] = am_logits

        return output
