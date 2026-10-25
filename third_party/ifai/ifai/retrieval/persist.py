"""L5 索引持久化 (Index Persistence) — 落盘 / 重载后检索结果 100% 一致

零重写 (集大成):
- **FAISS** (Johnson, Douze & Jégou 2019 "Billion-scale similarity search with
  GPUs") 与 **Annoy / HNSW** 的"索引落盘 + 重新 mmap"范式: 索引是**可序列化
  的状态**, 重载后与落盘前是同一份检索结构。
- 稀疏侧复用 **BM25** (Robertson & Zaragoza 2009), 稠密侧复用 **DPR 双塔**
  (Karpukhin et al. 2020) 的向量检索, 融合复用 **RRF** (Cormack et al. 2009)。
- 完整性校验沿用 L2 ``Transformer.save`` 的 **sha256 manifest** 做法
  (R-L2-06 同款: 逐文件 sha256 写入 manifest.json, 加载时校验)。

可验证不变量:
- **INV-32 索引持久化一致性**: ``save() → load()`` 后, 对同一批 query 的
  Top-K 结果 (doc_id 与分数) 与落盘前 **100% 一致** (逐项相等, 而非仅排序相同)。
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .bm25 import BM25
from .dense import DenseIndex
from .rrf import reciprocal_rank_fusion

INDEX_VERSION = 1


@dataclass
class Hit:
    """一条检索结果。

    Attributes:
        doc_id: 文档序号。
        score: 融合分数 (RRF, Cormack et al. 2009)。
        text: 文档原文 (可选, 由 ``texts`` 回填)。
        meta: 文档元数据 (可选)。
    """

    doc_id: int
    score: float
    text: str = ""
    meta: dict = field(default_factory=dict)

    def key(self) -> tuple:
        """用于一致性比对的键 (INV-32: id + 分数逐项相同)。"""
        return (self.doc_id, float(self.score), self.text)

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "score": self.score,
                "text": self.text, "meta": dict(self.meta)}


class PersistentIndex:
    """BM25 + 稠密双路 RRF 融合索引, 支持落盘与重载 (INV-32)。

    用法::

        idx = PersistentIndex().fit(corpus, texts=raw_texts)
        idx.save("assets/index/demo")
        idx2 = PersistentIndex.load("assets/index/demo")
        assert idx.search(q, 10) == idx2.search(q, 10)     # INV-32
    """

    def __init__(self, dim: int = 256, seed: int = 0, k1: float = 1.5,
                 b: float = 0.75, rrf_k: int = 60):
        self.dim = int(dim)
        self.seed = int(seed)
        self.k1 = float(k1)
        self.b = float(b)
        self.rrf_k = int(rrf_k)
        self.bm25: Optional[BM25] = None
        self.dense: Optional[DenseIndex] = None
        self.corpus: List[List[str]] = []
        self.texts: List[str] = []
        self.metas: List[dict] = []

    # ---------------- 构建 ----------------
    def fit(self, corpus: Sequence[Sequence[str]], texts: Optional[Sequence[str]] = None,
            metas: Optional[Sequence[dict]] = None) -> "PersistentIndex":
        """用分词后的语料构建索引。

        Args:
            corpus: 分词后的文档列表 (``[[tok, ...], ...]``)。
            texts: 文档原文 (用于 ``Hit.text`` 回填; 缺省用空格拼接 token)。
            metas: 每篇文档的元数据。

        Returns:
            self (便于链式调用)。
        """
        self.corpus = [list(d) for d in corpus]
        self.texts = list(texts) if texts is not None else [" ".join(d) for d in self.corpus]
        self.metas = [dict(m) for m in metas] if metas is not None else [{} for _ in self.corpus]
        self.bm25 = BM25(k1=self.k1, b=self.b).fit(self.corpus)
        self.dense = DenseIndex(dim=self.dim, seed=self.seed).fit(self.corpus)
        return self

    # ---------------- 检索 ----------------
    def search(self, query: Sequence[str], top_k: int = 10) -> List[Hit]:
        """双路检索 + RRF 融合, 返回 Top-K (INV-32 的比对对象)。"""
        if self.bm25 is None or self.dense is None:
            raise RuntimeError("索引未构建或未加载: 请先 fit() / load()")
        if not self.corpus:
            return []
        pool = max(int(top_k), 10)
        bm = self.bm25.search(list(query), top_k=pool)
        dn = self.dense.search(list(query), top_k=pool)
        fused = reciprocal_rank_fusion([[int(i) for i, _ in bm],
                                        [int(i) for i, _ in dn]], k=self.rrf_k)
        return [Hit(doc_id=int(i), score=float(s), text=self.texts[i],
                    meta=dict(self.metas[i])) for i, s in fused[:top_k]]

    def search_keys(self, query: Sequence[str], top_k: int = 10) -> List[tuple]:
        """检索结果的紧凑键 ``[(doc_id, score), ...]`` (便于一致性断言)。"""
        return [(h.doc_id, h.score) for h in self.search(query, top_k=top_k)]

    # ---------------- 落盘 / 重载 (INV-32) ----------------
    def save(self, dir_path: str) -> str:
        """落盘: ``index.json`` (稀疏侧状态 + 原文) + ``vectors.npz`` (稠密侧)。

        同时写入 ``manifest.json`` 记录各文件 sha256 (R-L2-06 同款完整性校验)。
        """
        if self.bm25 is None or self.dense is None:
            raise RuntimeError("索引未构建: 请先 fit()")
        os.makedirs(dir_path, exist_ok=True)
        state = {
            "version": INDEX_VERSION,
            "dim": self.dim,
            "seed": self.seed,
            "k1": self.k1,
            "b": self.b,
            "rrf_k": self.rrf_k,
            "bm25": self._bm25_state(),
            "texts": self.texts,
            "metas": self.metas,
        }
        idx_path = os.path.join(dir_path, "index.json")
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
        vec_path = os.path.join(dir_path, "vectors.npz")
        # v0.7.13: 移除死字段 proj (编码由 sha256 稳定派生, 无随机投影);
        # schema 变更 — 向量本身逐位不变 (sha256 稳定哈希下重 fit 亦逐位一致)
        np.savez(vec_path,
                 matrix=np.asarray(self.dense.matrix, dtype=np.float64))
        manifest = {
            "version": INDEX_VERSION,
            "n_docs": len(self.corpus),
            "sha256": {},
        }
        for name in ("index.json", "vectors.npz"):
            p = os.path.join(dir_path, name)
            with open(p, "rb") as f:
                manifest["sha256"][name] = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(dir_path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return dir_path

    def _bm25_state(self) -> dict:
        """导出 BM25 的完整状态 (docs / df / idf / doc_len / avgdl)。"""
        assert self.bm25 is not None
        return {
            "k1": self.bm25.k1,
            "b": self.bm25.b,
            "docs": [list(d) for d in self.bm25.docs],
            "doc_len": [int(x) for x in self.bm25.doc_len],
            "avgdl": float(self.bm25.avgdl),
            "df": {str(k): int(v) for k, v in self.bm25.df.items()},
            "idf": {str(k): float(v) for k, v in self.bm25.idf.items()},
        }

    @classmethod
    def load(cls, dir_path: str, verify: bool = True) -> "PersistentIndex":
        """重载索引; 重载后检索结果与落盘前 100% 一致 (INV-32)。

        Args:
            dir_path: :meth:`save` 写入的目录。
            verify: True 时校验 manifest 中的 sha256 (防止索引被篡改/损坏)。

        Raises:
            AssertionError: sha256 校验失败。
        """
        with open(os.path.join(dir_path, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        if verify:
            for name, digest in manifest.get("sha256", {}).items():
                p = os.path.join(dir_path, name)
                with open(p, "rb") as f:
                    assert hashlib.sha256(f.read()).hexdigest() == digest, \
                        f"索引文件校验失败 (sha256 mismatch): {name}"
        with open(os.path.join(dir_path, "index.json"), encoding="utf-8") as f:
            state = json.load(f)
        obj = cls(dim=int(state["dim"]), seed=int(state["seed"]),
                  k1=float(state["k1"]), b=float(state["b"]),
                  rrf_k=int(state["rrf_k"]))
        obj.corpus = [list(d) for d in state["bm25"]["docs"]]
        obj.texts = list(state.get("texts") or [])
        obj.metas = [dict(m) for m in state.get("metas") or []]
        obj.bm25 = cls._bm25_from_state(state["bm25"])
        data = np.load(os.path.join(dir_path, "vectors.npz"))
        dense = DenseIndex(dim=obj.dim, seed=obj.seed)
        dense.docs = [np.asarray(data["matrix"][i], dtype=np.float64)
                      for i in range(data["matrix"].shape[0])]
        dense.matrix = np.asarray(data["matrix"], dtype=np.float64)
        obj.dense = dense
        return obj

    @staticmethod
    def _bm25_from_state(st: dict) -> BM25:
        """按保存的状态逐字段重建 BM25 (不重新 fit, 保证分数逐位相同)。"""
        m = BM25(k1=float(st["k1"]), b=float(st["b"]))
        m.docs = [list(d) for d in st["docs"]]
        m.doc_len = [int(x) for x in st["doc_len"]]
        m.avgdl = float(st["avgdl"])
        m.df = Counter({str(k): int(v) for k, v in st["df"].items()})
        m.idf = {str(k): float(v) for k, v in st["idf"].items()}
        return m

    # ---------------- 一致性自检 (INV-32) ----------------
    def verify_reload(self, dir_path: str, queries: Sequence[Sequence[str]],
                      top_k: int = 10) -> dict:
        """落盘 → 重载 → 逐 query 比对 Top-K, 返回一致性报告。

        Returns:
            dict: ``consistent`` (bool) / ``n_queries`` / ``n_mismatches`` /
            ``mismatches`` (前 5 条差异常规信息) / ``max_score_diff``。
        """
        self.save(dir_path)
        reloaded = PersistentIndex.load(dir_path, verify=True)
        mismatches: List[dict] = []
        worst = 0.0
        for qi, q in enumerate(queries):
            a = self.search_keys(q, top_k=top_k)
            b = reloaded.search_keys(q, top_k=top_k)
            if a != b:
                mismatches.append({"query_index": qi, "before": a[:3], "after": b[:3]})
            for (ia, sa), (ib, sb) in zip(a, b):
                if ia == ib:
                    worst = max(worst, abs(sa - sb))
                else:
                    worst = float("inf")
        return {
            "consistent": len(mismatches) == 0,
            "n_queries": len(list(queries)),
            "n_mismatches": len(mismatches),
            "mismatches": mismatches[:5],
            "max_score_diff": worst,
        }


def save_index(index: PersistentIndex, dir_path: str) -> str:
    """函数式入口: 落盘。"""
    return index.save(dir_path)


def load_index(dir_path: str, verify: bool = True) -> PersistentIndex:
    """函数式入口: 重载 (INV-32)。"""
    return PersistentIndex.load(dir_path, verify=verify)


def verify_persistence(index: PersistentIndex, dir_path: str,
                       queries: Sequence[Sequence[str]], top_k: int = 10) -> dict:
    """函数式入口: INV-32 一致性报告。"""
    return index.verify_reload(dir_path, queries, top_k=top_k)
