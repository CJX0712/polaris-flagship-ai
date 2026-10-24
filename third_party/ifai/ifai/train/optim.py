"""L4 优化器与调度 (AdamW / 余弦+预热 / 全局范数裁剪)

零重写:
- AdamW (Loshchilov & Hutter 2019, ICLR) — 解耦权重衰减
- 余弦退火 + 线性预热 (cosine annealing + linear warmup, 见 HuggingFace
  ``get_cosine_schedule_with_warmup``)
- 全局范数梯度裁剪 (Pascanu et al. 2013)

可验证:
- INV-21 AdamW 单步 == 解析公式 到 1e-6；weight_decay=0 时 == Adam
- R-L4-02 余弦调度 == 解析 到 1e-9
- INV-23 裁剪后全局范数 ≤ max_norm+1e-6；未超限时梯度 bitwise 不变
"""
from __future__ import annotations

import numpy as np

from ..train.autodiff import Tensor


class GlobalStep:
    """学习率调度状态机 (线性预热 + 余弦退火)。"""

    def __init__(self, lr: float, total_steps: int, warmup_steps: int = 0,
                 min_lr: float = 0.0):
        self.lr0 = lr
        self.total = max(total_steps, 1)
        self.warmup = warmup_steps
        self.min_lr = min_lr
        self.t = 0

    def step(self) -> float:
        lr = cosine_schedule(self.lr0, self.t, self.total, self.warmup, self.min_lr)
        self.t += 1
        return lr


def cosine_schedule(lr0: float, step: int, total: int,
                    warmup: int = 0, min_lr: float = 0.0) -> float:
    """线性预热 + 余弦退火。解析公式, 供 INV(R-L4-02) 对照。"""
    if step < warmup and warmup > 0:
        return lr0 * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr0 - min_lr) * (1.0 + np.cos(np.pi * progress))


class AdamW:
    """解耦权重衰减 AdamW。"""

    def __init__(self, params: list, lr: float = 1e-3, betas=(0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        bc1 = 1 - self.b1 ** self.step_count
        bc2 = 1 - self.b2 ** self.step_count
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / bc1
            vhat = self.v[i] / bc2
            # 解耦权重衰减
            p.data = p.data - self.lr * (mhat / (np.sqrt(vhat) + self.eps)
                                         + self.wd * p.data)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


def clip_global_norm(params: list, max_norm: float, norm_type: float = 2.0):
    """全局范数裁剪。返回 (total_norm, triggered)。INV-23。"""
    norms = [np.linalg.norm(p.grad.ravel(), ord=norm_type) ** norm_type
             for p in params if p.grad is not None]
    total = float(np.sum(norms) ** (1.0 / norm_type))
    if max_norm > 0 and total > max_norm:
        scale = max_norm / (total + 1e-12)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
        return total, True
    return total, False
