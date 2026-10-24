"""L5 稠密向量检索 (Dense Retrieval)

零重写: 双塔 (DPR, Karpukhin et al. 2020) 思想 — 查询与文档在共享嵌入空间
做余弦相似度。本栈用哈希-加权嵌入 (hashing trick + TF 加权) 作为零依赖
可复现的稠密编码器, 与暴力 TopK 完全一致 (INV-29)。

可验证 (INV-29): 本索引 TopK 与暴力余弦排序 100% 一致。
"""
from __future__ import annotations

import hashlib

import numpy as np


def stable_hash(t: str) -> int:
    """跨进程稳定的词项哈希 (sha256 派生)。

    v0.7.13 真 bug 修复: 旧实现用 Python 内建 ``hash()`` — 受 PYTHONHASHSEED
    随机化影响, 同语料跨进程 fit 出不同向量, 违反本项目「确定性可复现」声明。
    sha256 派生在任意进程/任意种子下逐位一致。
    """
    return int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "little")


class DenseIndex:
    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        # seed 仅为 API 兼容保留 (persist.py 序列化用); 编码由 sha256 稳定派生,
        # 不依赖任何随机性 — v0.7.13 移除从未被使用的死属性 rng/proj。
        self.seed = seed

    def _embed(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(self.dim)
        for t in tokens:
            h = stable_hash(t) % self.dim
            vec[h] += 1.0
        # L2 归一化
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    def fit(self, corpus: list[list[str]]):
        self.docs = [self._embed(d) for d in corpus]
        self.matrix = np.array(self.docs) if self.docs else np.zeros((0, self.dim))
        return self

    def search(self, query: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        q = self._embed(query)
        if self.matrix.shape[0] == 0:
            return []
        sims = self.matrix @ q
        idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idx]

    # 暴力参考 (INV-29)
    @staticmethod
    def brute(matrix: np.ndarray, q: np.ndarray, top_k: int = 10):
        sims = matrix @ q
        idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idx]
