"""共识层 —— 让「一次生成」升级为「可度量的可信生成」。

零重写 · 出处
------------
两件世界顶级成果，Polaris 只做编排与可证化，不重写其算法：

1. **Self-Consistency**（Wang et al. 2022, arXiv:2203.11171）：
   对同一 prompt 采样 k 次（temperature>0），按**多数投票**选答案，
   替代朴素贪心解码。这篇论文的核心发现是：*复杂推理任务的正确率，
   与采样答案集合的自洽度强正相关* —— 也就是说，**一致性本身是可测量的
   置信度**。Polaris 把它做成 ``agreement ∈ [0,1]`` 的一等公民字段。

2. **交叉验证（Cross-Validation of Implementations）**：
   FlagshipAI（CJX0712/FlagshipAI，作者晨星）用 PyTorch 参照实现
   交叉验证自建算子 —— 同一个不变量，两个独立实现必须对上。
   Polaris 把这条铁律抬到引擎层：**同一 prompt 交给 N 个独立引擎，
   产出不一致就是可疑的**，无论其中任何一个自称多强。

相似度度量也不自造：**Jaccard 相似系数**（Jaccard 1912,
``J(A,B) = |A∩B| / |A∪B|``）跑在**字符 bigram** 集合上 ——
选择 bigram 而非 word token 的原因很实在：中文没有空格分词，
字符 bigram 是语言无关的标准做法（与 BLEU / chrF 的字符 n-gram 传统同源，
见 Popović 2015 chrF）。

可证不变量（本模块自带门禁 C1–C5，全部机检）
-------------------------------------------
- **C1** ``0 ≤ agreement ≤ 1``
- **C2** ``1 ≤ support ≤ total``
- **C3** 全票一致 ⇒ ``agreement == 1.0``
- **C4** 两两全不同 ⇒ ``agreement == 1/total``
- **C5** 确定性：固定 seed 下两次 ``consensus()`` 结果指纹相同
- **C6** 平局打破确定：同票时取字典序最小者（与采样顺序无关）
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from ..engines.base import EngineAdapter, EngineResult


# ------------------------------------------------------------------ 度量基元


def normalize(text: str) -> str:
    """投票用的文本归一：去首尾空白 + 折叠内部空白（不改动大小写/语义）。"""
    return " ".join(text.split())


def bigrams(text: str) -> set[str]:
    """字符 bigram 集合（长度 < 2 时退化为单字符集合）。

    语言无关：中文无空格分词也能用；与 chrF/BLEU 的字符 n-gram 传统同源。
    """
    t = normalize(text)
    if len(t) < 2:
        return {t} if t else set()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard 相似系数（Jaccard 1912）：``|A∩B| / |A∪B|``，空集对空集记 1.0。"""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def similarity(x: str, y: str) -> float:
    """两段文本的相似度 ∈ [0,1]（字符 bigram Jaccard）。"""
    return jaccard(bigrams(x), bigrams(y))


# ------------------------------------------------------------------ 结果类型


@dataclass
class ConsensusResult:
    """一次共识的结果。

    Attributes:
        text: 当选文本（票数最高；同票取字典序最小 —— C6 确定性）。
        agreement: 置信度 ∈ [0,1]，= 当选者票数 / 总票数。
        support: 当选者票数。
        total: 总票数（样本数或引擎数）。
        method: ``"self-consistency"`` 或 ``"cross-engine"``。
        mean_pairwise: 两两相似度均值 ∈ [0,1]（样本高度离散时显著低于 agreement）。
        votes: ``[(text, count), ...]``（票数降序，同票字典序升序）。
        reports: 每次调用的明细（引擎名、指纹、耗时、是否降级）。
        gates: C1–C5 门禁报告。
    """

    text: str
    agreement: float
    support: int
    total: int
    method: str
    mean_pairwise: float
    votes: list[tuple[str, int]] = field(default_factory=list)
    reports: list[dict[str, Any]] = field(default_factory=list)
    gates: list[dict[str, Any]] = field(default_factory=list)

    def fingerprint(self) -> str:
        payload = repr((self.text, self.method, self.agreement, self.support, self.total))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "agreement": round(self.agreement, 6),
            "support": self.support,
            "total": self.total,
            "method": self.method,
            "mean_pairwise": round(self.mean_pairwise, 6),
            "votes": [{"text": t, "count": c} for t, c in self.votes],
            "reports": self.reports,
            "gates": self.gates,
            "fingerprint": self.fingerprint(),
        }


# ------------------------------------------------------------------ 投票核心


def majority_vote(texts: list[str]) -> tuple[str, int, list[tuple[str, int]]]:
    """多数投票（Self-Consistency 的解码步）。

    同票时取**字典序最小**者 —— 使结果与采样顺序无关，满足 C6 确定性。

    Returns:
        ``(winner, support, votes)``；``votes`` 按 (-count, text) 排序。
    """
    counter = Counter(normalize(t) for t in texts if normalize(t))
    if not counter:
        return "", 0, []
    votes = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    winner, support = votes[0]
    return winner, support, votes


def mean_pairwise_similarity(texts: list[str]) -> float:
    """两两相似度均值（O(n²)，n 小 —— k 一般 ≤ 16，代价可接受）。"""
    xs = [t for t in texts if normalize(t)]
    if len(xs) < 2:
        return 1.0
    grams = [bigrams(t) for t in xs]
    total, pairs = 0.0, 0
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            total += jaccard(grams[i], grams[j])
            pairs += 1
    return total / pairs if pairs else 1.0


# ------------------------------------------------------------------ 门禁


def consensus_gates(
    agreement: float,
    support: int,
    total: int,
    texts: list[str],
) -> list[dict[str, Any]]:
    """C1–C4 门禁（C5 由调用方跨次比对，C6 由 majority_vote 结构性保证）。"""
    gates: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    add("C1", 0.0 <= agreement <= 1.0, f"agreement={agreement:.6f} in [0,1]")
    add("C2", total == 0 or 1 <= support <= total, f"support={support} total={total}")
    uniq = len({normalize(t) for t in texts if normalize(t)})
    if total > 0 and uniq == 1:
        add("C3", abs(agreement - 1.0) < 1e-12, f"unanimous -> agreement={agreement}")
    else:
        add("C3", True, "n/a (not unanimous)")
    if total > 0 and uniq == total:
        expected = 1.0 / total
        add("C4", abs(agreement - expected) < 1e-12, f"all-distinct -> {expected}")
    else:
        add("C4", True, "n/a (not all-distinct)")
    return gates


# ------------------------------------------------------------------ 两种共识


class Consensus:
    """共识器：把「一次生成」变成「可度量的可信生成」。"""

    def self_consistency(
        self,
        adapter: EngineAdapter,
        prompt: str,
        *,
        k: int = 5,
        max_tokens: int = 64,
        temperature: float = 0.8,
        seed: int = 0,
    ) -> ConsensusResult:
        """Self-Consistency（Wang et al. 2022）：同引擎采样 k 次后多数投票。

        每次采样用 ``seed + i`` —— 固定 seed 序列 ⇒ 可复现（C5 的前提）。
        """
        k = max(int(k), 1)
        texts: list[str] = []
        reports: list[dict[str, Any]] = []
        for i in range(k):
            t0 = time.perf_counter()
            r: EngineResult = adapter.complete(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed + i,
            )
            texts.append(r.text)
            reports.append(
                {
                    "engine": r.engine,
                    "index": i,
                    "seed": seed + i,
                    "result_fp": r.fingerprint(),
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                    "degraded": r.degraded,
                }
            )
        winner, support, votes = majority_vote(texts)
        agreement = support / k if k else 0.0
        return ConsensusResult(
            text=winner,
            agreement=agreement,
            support=support,
            total=k,
            method="self-consistency",
            mean_pairwise=mean_pairwise_similarity(texts),
            votes=votes,
            reports=reports,
            gates=consensus_gates(agreement, support, k, texts),
        )

    def cross_engine(
        self,
        adapters: list[EngineAdapter],
        prompt: str,
        *,
        max_tokens: int = 64,
        seed: int = 0,
    ) -> ConsensusResult:
        """多引擎交叉验证：同 prompt 交 N 个独立引擎，产出不一致即可疑。

        全部以 ``temperature=0`` 贪心解码 —— 交叉验证要求**可比**，
        随机采样会让「不一致」失去诊断意义。
        """
        # 按引擎名排序 ⇒ 顺序确定（C5）
        ordered = sorted(adapters, key=lambda a: a.name)
        texts: list[str] = []
        reports: list[dict[str, Any]] = []
        for a in ordered:
            t0 = time.perf_counter()
            r: EngineResult = a.complete(
                prompt, max_tokens=max_tokens, temperature=0.0, seed=seed
            )
            texts.append(r.text)
            reports.append(
                {
                    "engine": r.engine,
                    "kind": a.kind,
                    "result_fp": r.fingerprint(),
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                    "degraded": r.degraded,
                }
            )
        winner, support, votes = majority_vote(texts)
        total = len(ordered)
        agreement = support / total if total else 0.0
        return ConsensusResult(
            text=winner,
            agreement=agreement,
            support=support,
            total=total,
            method="cross-engine",
            mean_pairwise=mean_pairwise_similarity(texts),
            votes=votes,
            reports=reports,
            gates=consensus_gates(agreement, support, total, texts),
        )


__all__ = [
    "Consensus",
    "ConsensusResult",
    "consensus_gates",
    "majority_vote",
    "mean_pairwise_similarity",
    "normalize",
    "bigrams",
    "jaccard",
    "similarity",
]
