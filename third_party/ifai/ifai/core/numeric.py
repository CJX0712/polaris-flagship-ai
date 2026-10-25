"""L1 数值稳定工具 (Numerical stability primitives)

零重写: 复用业界标准技巧
- logsumexp (机器学习数值标准, 见 Murphy 2012 "Machine Learning: a Probabilistic
  Perspective" §2.5.4) — 防 exp 上溢/下溢
- 稳定 softmax — 平移不变性 (INV-09)
- 安全交叉熵 — 防止 log(0)

可验证不变量:
- INV-08  logsumexp 稳定式 == 朴素 ``log(sum(exp(x)))`` 到 1e-6 (解析解对照)
- INV-09  softmax(x+c) == softmax(x)  (平移不变)
- R-L1-03 logits ∈ [-1e4,1e4] 无 inf/nan
"""
from __future__ import annotations

import numpy as np


def logsumexp(x: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """数值稳定的 log(sum(exp(x)))。

    公式: max + log(sum(exp(x - max))) — 经典两路分解 (INV-08)。
    """
    x = np.asarray(x, dtype=np.float64)
    xmax = np.max(x, axis=axis, keepdims=True)
    # 处理全 -inf 的情形，避免 nan
    xmax_safe = np.where(np.isfinite(xmax), xmax, 0.0)
    out = xmax_safe + np.log(np.sum(np.exp(x - xmax_safe), axis=axis, keepdims=True))
    if not keepdims:
        # 只压掉被约简的轴: 无轴 squeeze 会把 size-1 的批次维一并压掉
        # (如 (1,C) 沿 axis=1 → (1,1) → 0 维), 破坏 ``lse[:, None]`` 等下游用法
        out = np.squeeze(out, axis=axis) if axis is not None else out.reshape(())
    return out.astype(np.float64)


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """平移不变的 softmax (INV-09)。"""
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def safe_cross_entropy(logits: np.ndarray, targets: np.ndarray,
                       ignore_index: int = -100, label_smoothing: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """返回 (loss 标量, 对 logits 的梯度)。

    - 用 logsumexp 稳定计算 log-prob (INV-08)
    - 平滑: (1-ls)*onehot + ls/C
    - 对标量/一维 target 与 2D [N,C] 均支持
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    if logits.ndim == 1:
        logits = logits[None]
        targets = np.atleast_1d(targets)
    N, C = logits.shape
    lse = logsumexp(logits, axis=1)                 # [N]
    log_probs = logits - lse[:, None]               # [N,C]
    mask = targets != ignore_index
    rows = np.where(mask, targets, 0)

    # 构造目标分布 q
    q = np.zeros((N, C), dtype=np.float64)
    q[np.arange(N), rows] = 1.0
    if label_smoothing > 0:
        q = (1.0 - label_smoothing) * q + label_smoothing / C

    ce = -np.sum(q * log_probs, axis=1)             # [N]
    loss = np.sum(ce * mask) / max(mask.sum(), 1)

    # dL/dlogits = p - q，其中 p = exp(log_probs)
    p = np.exp(log_probs)
    grad = (p - q) / max(mask.sum(), 1)
    grad = grad * mask[:, None]
    return float(np.sum(loss)), grad


def clip_by_norm(grad: np.ndarray, max_norm: float, norm_type: float = 2.0):
    """全局范数裁剪，返回 (裁剪后梯度, 是否触发裁剪)。INV-23 由 train 层实现全局版。"""
    g = np.asarray(grad, dtype=np.float64)
    norm = float(np.linalg.norm(g.ravel(), ord=norm_type))
    if norm > max_norm > 0:
        return g * (max_norm / norm), True
    return g, False
