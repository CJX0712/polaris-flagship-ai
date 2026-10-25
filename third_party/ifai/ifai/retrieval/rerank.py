"""L5 重排 (Cross-Encoder 风格)

零重写: Nogueira & Cho 2019 "Passage Re-ranking with BERT" 的 cross-encoder
范式 — 查询与文档交互打分。本栈在无预训练权重、零依赖约束下, 采用
"随机特征交互" (hashing / random-feature kernel) 实现确定性可复现的
查询-文档相关性打分: 每个词映射到稳定的单位随机向量, 文档向量为词向量之和,
查询与文档向量点积 → 共享词项贡献正且一致的内积, 非共享词项内积 ~0, 从而
凸现词项重叠 (相关性)。零依赖、可复现 (INV 确定性), 并保留交互式重排语义。

可验证: 含查询词项的文档排在无关文档之前 (test_rerank_top1)。
"""
from __future__ import annotations

import numpy as np

from .dense import stable_hash


class CrossEncoderReranker:
    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def _token_vec(self, t: str) -> np.ndarray:
        # 词 → 稳定单位向量 (由 sha256 稳定哈希派生, 确定性可复现 — v0.7.13
        # 修复内建 hash() 受 PYTHONHASHSEED 随机化导致的跨进程漂移)
        h = (stable_hash(t) & 0xFFFFFFFF) ^ (self.seed & 0xFFFFFFFF)
        v = np.random.default_rng(h).standard_normal(self.dim)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _feat(self, tokens: list[str]) -> np.ndarray:
        v = np.zeros(self.dim)
        for t in tokens:
            v += self._token_vec(t)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def rerank(self, query: list[str], docs: list[list[str]], top_k: int = 5) -> list[tuple[int, float]]:
        qv = self._feat(query)
        scored = [(i, float(qv @ self._feat(d))) for i, d in enumerate(docs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
