"""L5 RRF 倒数排名融合 (Reciprocal Rank Fusion)

零重写: Cormack et al. 2009 "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods"。k=60。

可验证 (INV-30): fused_score = Σ 1/(k + rank) 与公式解析一致 (≤1e-12)。
"""
from __future__ import annotations

import numpy as np


def reciprocal_rank_fusion(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """rank_lists: 多个检索器各自的(已排序)文档 id 列表。返回融合排序。"""
    scores: dict[int, float] = {}
    for rl in rank_lists:
        for rank, doc_id in enumerate(rl):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
