"""L2 模型配置 (dataclass + JSON round-trip)

零重写: 配置驱动设计参考 HuggingFace ``transformers.PretrainedConfig`` 范式。

可验证:
- INV-13 参数量精确守恒 (config 决定参数量, 见 transformer.count_params)
- R-L2-05 两次加载 bitwise 一致 (json round-trip)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class ModelConfig:
    """Transformer 解码器配置。所有字段可 JSON 往返。"""

    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int | None = None      # GQA; None ⇒ 等于 n_heads (=MHA)
    ffn_mult: int = 4                  # d_ff = d_model * ffn_mult
    rope_base: float = 10000.0
    rope_scaling: float = 1.0          # NTK-aware 缩放因子
    norm_eps: float = 1e-6
    use_moe: bool = False
    n_experts: int = 0
    moe_topk: int = 1
    tie_word_embeddings: bool = True   # INV-12 权重绑定
    max_seq_len: int = 1024
    activation: str = "swiglu"         # swiglu | gelu
    use_lora: bool = False             # LoRA 低秩适配 (Hu et al. 2022)
    lora_r: int = 4                     # LoRA 秩 r
    lora_alpha: int = 8                 # LoRA 缩放 α (scale = α/r)
    use_qat: bool = False              # 量化感知训练 (Jacob et al. 2018): 训练期对线性权重做 STE 伪量化
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        # 整除约束 (GQA 要求 n_heads % n_kv_heads == 0)
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        if self.use_moe:
            assert self.n_experts >= 2 and self.moe_topk >= 1 and self.moe_topk < self.n_experts
        if self.use_lora:
            assert self.lora_r >= 1 and self.lora_alpha >= 1, "lora_r/lora_alpha 必须 ≥ 1"

    # 派生属性
    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        return self.d_model * self.ffn_mult

    @property
    def n_repeats(self) -> int:
        return self.n_heads // self.n_kv_heads

    # JSON 往返 (INV: 两次加载 bitwise 一致)
    def to_json(self, path: str | None = None) -> str:
        s = json.dumps(asdict(self), sort_keys=True, indent=2, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    @classmethod
    def from_json(cls, source: str | dict) -> "ModelConfig":
        if isinstance(source, str):
            obj = json.loads(source)
        else:
            obj = source
        allowed = {f for f in cls.__dataclass_fields__}
        obj = {k: v for k, v in obj.items() if k in allowed}
        return cls(**obj)

    def clone(self) -> "ModelConfig":
        return ModelConfig.from_json(self.to_json())
