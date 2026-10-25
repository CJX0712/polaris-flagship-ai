"""L8 困惑度评测 (Perplexity)

零重写: 困惑度沿用 Jelinek et al. 1977 提出的语言模型评价标准
``PPL = exp(mean NLL)`` —— 衡量模型对测试序列的"惊讶度", 是 NLP 评估的
事实标准度量。

可验证 (INV-39): PPL == exp(mean NLL), 与手工公式一致到 1e-6。
"""
from __future__ import annotations

import numpy as np

from ..core import numeric


def perplexity(model, token_ids: list[int]) -> float:
    """给定 token 序列计算困惑度 (逐 token 用 logits_next, 因果)。"""
    ids = list(token_ids)
    total_nll = 0.0
    n = 0
    for i in range(1, len(ids)):
        logits = model.logits_next(ids[:i])[0] if False else model.forward(np.array([ids[:i]]))[0, -1]
        # 数值稳定 NLL
        lse = numeric.logsumexp(logits)
        logp = logits[ids[i]] - lse
        total_nll += -float(logp)
        n += 1
    mean_nll = total_nll / max(n, 1)
    return float(np.exp(mean_nll))


def perplexity_manual(model, token_ids: list[int]) -> float:
    """纯手工 exp(mean NLL) 参考 (INV-39 对照)。"""
    ids = list(token_ids)
    nll = 0.0
    n = 0
    for i in range(1, len(ids)):
        logits = model.forward(np.array([ids[:i]]))[0, -1]
        m = np.max(logits)
        logp = (logits[ids[i]] - m) - np.log(np.sum(np.exp(logits - m)))
        nll += -logp
        n += 1
    return float(np.exp(nll / max(n, 1)))
