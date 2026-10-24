"""L5 HNSW 近似最近邻 (Hierarchical Navigable Small World)

零重写: Malkov & Yashunin 2018 "Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs"。

可验证 (INV-28): 与暴力 TopK 相比 recall@10 ≥ 0.95, QPS ≥ 暴力 5×。

实现要点: 多层图, 随机 level (ln 分布), 贪心搜索 + ef 候选集。
"""
from __future__ import annotations

import heapq
import math
import random

import numpy as np


class HNSW:
    def __init__(self, M: int = 16, ef_construction: int = 100, ef_search: int = 64,
                 ml: float | None = None, seed: int = 0):
        self.M = M
        self.efC = ef_construction
        self.efS = ef_search
        self.ml = ml or 1.0 / math.log(M)
        self.rng = random.Random(seed)
        self.data: list[np.ndarray] = []
        self.layers: dict[int, list[int]] = {}     # node -> neighbor ids per layer
        self.max_level = -1
        self.enter = None

    def _dist(self, a: int, b: int) -> float:
        return float(np.linalg.norm(self.data[a] - self.data[b]))

    def _level(self) -> int:
        return int(-math.log(self.rng.random()) * self.ml)

    def add(self, vec: np.ndarray):
        idx = len(self.data)
        self.data.append(np.asarray(vec, dtype=np.float64))
        level = self._level()
        self.layers[idx] = []
        # v0.7.13 真 bug 修复 (图不连通 → recall 恰 = 可达节点比例):
        # 旧实现当新节点提升 max_level 时 enter=idx, 且插入搜索从"它自己"
        # (零边) 出发 → 候选=自身 → 自环、无真实边 → enter 跳到孤立点,
        # 此后所有插入连进新分量, 旧分量整体失联 (400 节点实测 40 个不可达,
        # recall 恒 0.900 且 efS 拉满无效)。修复: 先捕获旧 enter 作为
        # 各层插入搜索的统一入口 — 新节点经旧 enter 的连通分量建立真实边。
        prev_enter = self.enter
        if self.enter is None or level > self.max_level:
            self.max_level = max(self.max_level, level)
            self.enter = idx
        ep = [prev_enter] if prev_enter is not None else []
        for lc in range(min(level, self.max_level), -1, -1):
            if not ep:
                ep = [self.enter] if self.enter is not None else []
            # 贪心搜 efC 候选
            candidates = self._search_layer(self.data[idx], ep, self.efC, lc)
            # 连接 M 最近邻 (排除自身, 防自环)
            neighbors = [e for e in candidates if e != idx][:self.M]
            self.layers[idx] = self.layers.get(idx, [])  # 占位
            for nb in neighbors:
                self._connect(idx, nb, lc)
            ep = neighbors
        return idx

    def _connect(self, a: int, b: int, layer: int):
        # 简易双向连接 (未做邻居裁剪, 足以保证高召回)
        if a == b:
            return  # v0.7.13: 自环守卫 (自环不提供导航边, 且污染邻居列表)
        self.layers.setdefault(a, [])
        self.layers.setdefault(b, [])
        if b not in self.layers[a]:
            self.layers[a].append(b)
        if a not in self.layers[b]:
            self.layers[b].append(a)

    def _search_layer(self, q: np.ndarray, entry: list[int], ef: int, layer: int) -> list[int]:
        visited = set(entry)
        # 候选最小堆 (距离, id)
        candidates = [(self._dist_id(q, e), e) for e in entry]
        # 结果最大堆 (保存最小 ef 个)
        results = [(-self._dist_id(q, e), e) for e in entry]
        heapq.heapify(candidates)
        heapq.heapify(results)
        while candidates:
            d, cur = heapq.heappop(candidates)
            # 若最远结果已比候选最近还近, 提前停止
            far = -results[0][0] if results else -1
            if d > far and len(results) >= ef:
                break
            for nb in self.layers.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    dn = self._dist_id(q, nb)
                    far = -results[0][0] if results else -1
                    if len(results) < ef or dn < far:
                        heapq.heappush(candidates, (dn, nb))
                        heapq.heappush(results, (-dn, nb))
                        if len(results) > ef:
                            heapq.heappop(results)
        return [e for _, e in sorted(results, reverse=True)]

    def _dist_id(self, q, i):
        return float(np.linalg.norm(q - self.data[i]))

    def build(self, data: np.ndarray):
        for v in data:
            self.add(v)
        return self

    def search(self, q: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if self.enter is None:
            return []
        res = self._search_layer(np.asarray(q, dtype=np.float64), [self.enter], max(self.efS, k), self.max_level)
        out = [(i, float(np.linalg.norm(np.asarray(q) - self.data[i]))) for i in res[:k]]
        return sorted(out, key=lambda x: x[1])[:k]
