"""L4 LoRA 低秩适配 (Low-Rank Adaptation)

零重写: 复用 Hu et al. 2022 "LoRA: Low-Rank Adaptation of Large Language Models"。
W_eff = W + (alpha/r) · B·A,  B∈R^{d×r}, A∈R^{r×k}。

可验证:
- INV-24 LoRA merge (W + scale·B@A) 与未 merge 前向一致 到 1e-6
- B=0 (或 scale=0) 时 == 冻结原权重
- 可训练参数 = (d+r)·k ≪ d·k (通常 ≤5%)
"""
from __future__ import annotations

import numpy as np

from ..train.autodiff import Tensor


class LoRALinear:
    def __init__(self, W: np.ndarray, r: int = 4, alpha: float = 8.0):
        self.W = Tensor(W, requires_grad=True)       # 冻结原权重 (不更新)
        d_out, d_in = W.shape
        self.r = r
        self.scale = alpha / r
        self.A = Tensor(np.zeros((r, d_in)), requires_grad=True)
        self.B = Tensor(np.zeros((d_out, r)), requires_grad=True)
        # 用随机小量初始化 A 以打破对称 (INV-24 B=0 ⇒ 等价原权重)
        rng = np.random.default_rng(0)
        self.A.data = (rng.standard_normal((r, d_in)) * 0.01)

    @property
    def trainable_params(self):
        return [self.A, self.B]

    def forward(self, x: Tensor) -> Tensor:
        base = x.matmul(self.W.transpose())
        lora = x.matmul(self.A.transpose()).matmul(self.B.transpose()) * self.scale
        return base + lora

    def merge(self) -> np.ndarray:
        """合并低秩增量到原权重, 返回新 W (INV-24)。"""
        return self.W.data + self.scale * (self.B.data @ self.A.data)

    def merged_linear(self, x: Tensor) -> Tensor:
        return x.matmul(Tensor(self.merge()).transpose())
