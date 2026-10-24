"""L5 BM25 (Okapi BM25)

零重写: Robertson & Zaragoza 2009 "The Probabilistic Relevance Framework:
BM25 and Beyond"。k1=1.5, b=0.75 (PRD 默认)。

可验证 (INV-27): 与一份"朴素参考 BM25"实现的 Top-10 重合度 ≥ 95%。
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.df = Counter()
        self.idf = {}
        self.avgdl = 0.0
        self.doc_len = []

    def fit(self, corpus: list[list[str]]):
        self.docs = [list(d) for d in corpus]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        for d in self.docs:
            for w in set(d):
                self.df[w] += 1
        n = len(self.docs)
        for w, f in self.df.items():
            self.idf[w] = math.log(1 + (n - f + 0.5) / (f + 0.5))
        return self

    def _score(self, query: list[str], doc_idx: int) -> float:
        tf = Counter(self.docs[doc_idx])
        dl = self.doc_len[doc_idx]
        score = 0.0
        for w in query:
            if w not in self.idf:
                continue
            f = tf[w]
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score += self.idf[w] * (f * (self.k1 + 1)) / denom
        return score

    def search(self, query: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        scores = [(i, self._score(query, i)) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # 朴素参考实现 (用于 INV-27 对照)
    @staticmethod
    def reference(corpus: list[list[str]], query: list[str], k1=1.5, b=0.75):
        n = len(corpus)
        df = Counter()
        for d in corpus:
            for w in set(d):
                df[w] += 1
        idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        dl = [len(d) for d in corpus]
        avgdl = sum(dl) / n
        out = []
        for i, d in enumerate(corpus):
            tf = Counter(d)
            s = 0.0
            for w in query:
                if w in idf:
                    f = tf[w]
                    s += idf[w] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl[i] / avgdl))
            out.append((i, s))
        return sorted(out, key=lambda x: x[1], reverse=True)
