"""L5 MMR — Maximal Marginal Relevance 多样性重排

零重写 · 集大成
----------------
Priors:
  - Carbonell & Goldstein. *The Use of MMR, Diversity-Based Reranking for
    Reordering Documents and Producing Summaries.* SIGIR 1998.
    → 本模块的数学基底（贪心选择，见下）。
  - 与 ``ifai/retrieval/rerank.py`` 的 Cross-Encoder 重排**互补而非重复**：
    Cross-Encoder 优化「单篇相关性」，MMR 优化「集合的相关性/多样性权衡」。
    二者可级联：多路召回 → RRF/QFR 融合 → MMR 多样性裁剪 → Cross-Encoder 精排。

数学定义（Carbonell & Goldstein 1998）::

    MMR = argmax_{d ∈ R∖S}  λ·sim(d, Q) − (1−λ)·max_{s∈S} sim(d, s)

``λ ∈ [0,1]``：1.0 = 纯相关性（退化为原始排序），0.0 = 纯多样性。

工程要点
--------
- 相似度矩阵用 numpy 一次矩阵乘算完，避免 O(k·|S|·d) 的 Python 三重循环
- 选择过程是贪心的（论文本身即贪心），对输入顺序不敏感
- **同分裁决按 doc_id 字典序**，保证跨进程可复现

可验证（INV-49 ~ INV-50）
------------------------
INV-49 λ=1.0 恒等：MMR 退化为原始相关度排序
INV-50 多样性单调：λ 降低 ⟹ 结果的平均两两相似度不升
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["mmr_rerank", "pairwise_cosine", "avg_pairwise_similarity"]


def pairwise_cosine(vectors: np.ndarray) -> np.ndarray:
    """计算向量集合的两两余弦相似度矩阵。

    零向量安全：模长为 0 的行/列结果置 0（而非 NaN）。
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms <= 1e-12, 1.0, norms)
    unit = vectors / safe
    sim = unit @ unit.T
    # 零向量行的相似度强制为 0
    zero_mask = (norms <= 1e-12).ravel()
    if zero_mask.any():
        sim[zero_mask, :] = 0.0
        sim[:, zero_mask] = 0.0
    return sim


def mmr_rerank(
    doc_ids: Sequence[int],
    scores: Sequence[float],
    vectors: np.ndarray,
    *,
    top_k: int = 5,
    lambda_mult: float = 0.7,
    candidate_k: int = 40,
) -> list[tuple[int, float]]:
    """对候选文档做 MMR 重排。

    参数
    ----
    doc_ids:
        候选文档 id（与 ``scores`` / ``vectors`` 行对齐）。
    scores:
        相关性分数（通常来自 RRF/QFR 融合结果）。
    vectors:
        文档向量矩阵，shape ``(n, dim)``。
    top_k:
        最终返回条数。
    lambda_mult:
        1.0 = 纯相关性（INV-49 恒等式），0.0 = 纯多样性。
    candidate_k:
        进入 MMR 的候选上限。超过先用相关度截断 —— 这是 Carbonell 原文
        即采用的技巧，避免在大量低分文档上浪费计算。

    不变量
    ------
    INV-49: ``lambda_mult == 1.0`` 时结果等于按分数降序的前 top_k 个。
    INV-50: ``lambda_mult`` 降低 ⟹ 结果平均两两相似度不升。
    确定性: 相同输入 → 相同输出（同分按 doc_id 字典序）。
    """
    n = len(doc_ids)
    if n == 0 or top_k <= 0:
        return []
    if lambda_mult >= 1.0:
        # INV-49 的直接短路：纯相关性 = 原始排序
        order = sorted(range(n), key=lambda i: (-scores[i], doc_ids[i]))
        return [(doc_ids[i], float(scores[i])) for i in order[: min(top_k, n)]]

    # 先用相关度截断到候选集（Carbonell 原文技巧）
    pre_order = sorted(range(n), key=lambda i: (-scores[i], doc_ids[i]))[:candidate_k]
    m = len(pre_order)
    k = min(top_k, m)
    if k <= 0:
        return []

    cand_ids = [doc_ids[i] for i in pre_order]
    cand_scores = [float(scores[i]) for i in pre_order]
    cand_vecs = vectors[pre_order]
    sim = pairwise_cosine(cand_vecs)

    selected: list[int] = []
    remaining = list(range(m))
    while len(selected) < k and remaining:
        best_pos = -1
        best_val = float("-inf")
        for pos in remaining:
            if not selected:
                mmr = cand_scores[pos]
            else:
                max_sim = max(sim[pos][j] for j in selected)
                mmr = lambda_mult * cand_scores[pos] - (1.0 - lambda_mult) * max_sim
            # 严格大于才替换 → 同分保留先遇到的（已按 -score, id 预排序）
            if mmr > best_val:
                best_val = mmr
                best_pos = pos
        selected.append(best_pos)
        remaining.remove(best_pos)

    return [(cand_ids[p], cand_scores[p]) for p in selected]


def avg_pairwise_similarity(vectors: np.ndarray) -> float:
    """一组向量的平均两两余弦相似度（多样性度量，越小越多样）。"""
    m = len(vectors)
    if m < 2:
        return 0.0
    sim = pairwise_cosine(vectors)
    total = float(sim.sum() - np.trace(sim))
    pairs = m * (m - 1)
    return total / pairs if pairs else 0.0
