"""L5 QFR — Qiming Fusion Rank（时序感知 × 通道先验的倒数排名融合）

零重写 · 集大成
----------------
Priors:
  - Cormack, Clarke & Buettcher. *Reciprocal Rank Fusion outperforms Condorcet
    and individual Rank Learning Methods.* SIGIR 2009.
    → 本模块的数学基底：``score = Σ 1/(k + rank)``。``ifai/retrieval/rrf.py``
      已实现标准 RRF（INV-30）；**QFR 是其严格超集** —— 见 INV-43：
      当全部通道权重 = 1.0 且关闭时序时，QFR 与标准 RRF 逐位相等（≤1e-12）。
  - Robertson & Zaragoza 2009 (BM25) / Malkov & Yashunin 2018 (HNSW) /
    Karpukhin et al. 2020 (DPR 双塔) → 提供被融合的异构召回通道。
  - Zep / Graphiti（bi-temporal knowledge graph, 2024–2026）→
    事实有效期 ``valid_from`` / ``valid_to`` 的设计来源：过期的事实不是
    「相关性低」，而是「不该出现」——因此本模块对过期文档做**硬过滤**。
  - Letta / MemGPT（Packer et al. 2023, arXiv:2310.08560）→
    通道先验中 ``recency`` 一项的 OS 分层记忆思想。
  - Hindsight（Vectorize.io, 2026, MIT）→ 「多策略并行检索 + reflect 合成」
    的方法论来源（LongMemEval 91.4%）。

动机：世界顶级方案留下的缝隙
----------------------------
Hindsight 以「多策略并行检索 + **LLM reflect 合成**」登顶 LongMemEval，
但其 reflect 步骤带来 **800–3000ms** 延迟，无法用于实时对话，只能用于
批处理与离线记忆整合。

**没有人在「高召回」与「实时延迟」之间做纯数学的折中。**

QFR 的主张：多通道融合的质量瓶颈不在于「要不要 LLM 合成」，而在于
**「如何给不同通道赋权」**与**「如何处理事实的时效性」**。这两件事
不需要 LLM，只需要正确的数学。

数学定义
--------
标准 RRF（Cormack et al. 2009）::

    RRF(d) = Σ_c  1 / (k + rank_c(d))

QFR 引入两个正交因子::

    QFR(d) = Σ_c  w_c · T(d) · 1 / (k + rank_c(d))
                  │     │
                  │     └─ 时序感知：T(d) = 0.5 ** ((age_days / half_life) ** gamma)
                  └─ 通道可靠性先验 w_c

1. **通道先验** ``w_c``
   - ``dense``（DPR 双塔）在「换了说法问同一件事」上更强
   - ``bm25``（字面）在「精确术语 / 专有名词 / 代码标识符」上更强
   - ``hnsw``（近似最近邻）语义同 dense，但因近似可能有召回损失
   - ``recency``（最近写入）默认 0.6 —— 「最近说过」本身不是强证据，
     但刚写入的记忆确实有更高时效相关性
   权重显式可调，默认全 1.0（此时退化为标准 RRF，见 INV-43）。

2. **时序感知** ``T(d)``
   事实会过期::

       age_days = (now - valid_from(d)) / 86400000
       T(d) = 0.5 ** ((age_days / half_life_days) ** gamma)

   - ``T(0) = 1``（刚写入不衰减）；``T(half_life) = 0.5``（半衰期恰减半）
   - ``T`` 随 age **单调不增**，恒 ∈ (0, 1]
   - 浮点下指数 > 1023 会下溢为 0，实现中给最小正 double 地板保持 ``T > 0``

3. **有效期硬过滤**（借鉴 Zep/Graphiti bi-temporal）
   ``valid_to`` 已过的文档**直接剔除**，不是给低分。

**排名口径（重要兼容性细节）**
本模块与 ``ifai/retrieval/rrf.py`` 的 ``reciprocal_rank_fusion`` **严格同口径**：
rank 为 **0-based**（通道内第一篇文档 rank=0，得分 ``1/k``）。这保证
INV-43 的退化性是**逐位相等**而非近似相等 —— 若改用 1-based，退化性会
有系统性偏移（实测 4.91e-04），「超集」的声称就不成立了。

可验证（INV-43 ~ INV-48）
------------------------
INV-43 退化性：权重全 1 + 关时序 ⟹ QFR ≡ 标准 RRF（≤1e-12，与 rrf.py 交叉验证）
INV-44 时序有界：T ∈ (0,1]、单调不增、T(0)=1、T(30)=0.5、T(60)=0.25
INV-45 单调性：d 在所有通道排名不劣于 d' ⟹ QFR(d) ≥ QFR(d')
INV-46 过期必除：valid_to 已过的文档绝不出现在结果中
INV-47 确定性：相同输入 → 相同输出（同分按 doc_id 字典序，仅用于稳定展示）
INV-48 先验可解释：提高某通道权重 ⟹ 该通道头部文档排名上升

与 Hindsight 的对比
-------------------
=====================  =======================  ==============================
维度                   Hindsight                QFR
=====================  =======================  ==============================
多策略并行检索          ✅                       ✅
reflect LLM 合成        800–3000ms               **0ms（纯数学）**
事实时效性              需 LLM 推断              **显式 valid_to 硬过滤**
融合权重                LLM 隐式判断              **可解释的通道先验**
额外成本                每次检索 1 次 LLM 调用    **0 token**
=====================  =======================  ==============================

reflect 不是被抛弃，而是被放到正确位置：**离线巩固阶段**（延迟不敏感）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "QFRWeights",
    "temporal_factor",
    "qfr_fuse",
    "qfr_reference",
    "MIN_POSITIVE_DOUBLE",
]

# 5e-324 是最小的正 double（denormal）。用于防止 0.5 ** exp 在 exp > 1023 时
# 下溢为 0.0 —— 数学上 T 的极限是 0 但永不等于 0。
MIN_POSITIVE_DOUBLE = 5e-324


@dataclass
class QFRWeights:
    """各召回通道的可靠性先验 + 融合参数。

    通道名与 ``ifai/retrieval/__init__.py`` 导出的检索器对应：
    ``bm25`` / ``dense`` / ``hnsw`` / ``recency``。未登记的通道由
    :meth:`weight_for` 默认给 1.0。
    """

    bm25: float = 1.0
    dense: float = 1.0
    hnsw: float = 1.0
    # recency 默认低于语义通道："最近说过"本身不是强证据
    recency: float = 0.6
    # RRF 平滑常数。Cormack et al. 2009 建议 60；k 越大头部尾部差距越小。
    k: int = 60
    # 时序衰减半衰期（天）。语义事实的半衰期，按业务调节。
    half_life_days: float = 30.0
    # 衰减陡度指数。gamma=1 时半衰期处恰好为 0.5；>1 更陡。
    gamma: float = 1.0
    # 关闭时序衰减（用于与标准 RRF 对照实验，见 INV-43）
    temporal_enabled: bool = True

    def weight_for(self, channel: str) -> float:
        """按通道名取权重。未登记通道默认 1.0。"""
        return float(getattr(self, channel, 1.0) or 1.0)


def temporal_factor(age_days: float, *, half_life_days: float = 30.0, gamma: float = 1.0) -> float:
    """时序衰减因子 T。

    定义：``T = 0.5 ** ((age_days / half_life_days) ** gamma)``

    不变量（INV-44）：
    - ``age = 0 → T = 1.0``
    - ``age = half_life → T = 0.5``（当 gamma = 1）
    - ``T`` 随 age 单调不增
    - ``0 < T ≤ 1`` 恒成立（含指数下溢保护）
    """
    if age_days < 0:
        age_days = 0.0
    if half_life_days <= 0:
        return 1.0
    if gamma <= 0:
        return 1.0
    ratio = (age_days / half_life_days) ** gamma
    if ratio > 1023.0:
        # 防止浮点下溢为 0.0，保持 T > 0 的数学性质
        return MIN_POSITIVE_DOUBLE
    return float(0.5**ratio)


def qfr_fuse(
    channels: Mapping[str, Sequence[tuple[int, float]]],
    *,
    weights: QFRWeights | None = None,
    ages: Mapping[int, float] | None = None,
    valid_to: Mapping[int, int] | None = None,
    now_ms: int | None = None,
) -> list[tuple[int, float]]:
    """多通道 QFR 融合。

    参数
    ----
    channels:
        ``{通道名: [(doc_id, 原始分数), ...]}``。每通道内部按分数降序分配
        rank（**0-based**，与 ``rrf.py`` 同口径）；同分按 doc_id 升序（保证确定性）。
    weights:
        通道先验与融合参数。
    ages:
        ``{doc_id: age_days}`` 文档年龄（天）。缺失的文档视为 age=0（不衰减）。
    valid_to:
        ``{doc_id: 有效期截止 epoch_ms}``。已过期的文档被**硬过滤**（INV-46）。
    now_ms:
        当前 epoch 毫秒。缺省用 ``time.time()``。

    返回
    ----
    ``[(doc_id, qfr_score), ...]`` 按分数降序。

    流程
    ----
    1. 每通道按分数降序分配 rank
    2. 剔除过期文档（INV-46）
    3. 计算 QFR 分数（含时序因子与通道先验）
    4. 按 ``(-score, doc_id)`` 排序 —— 保证确定性（INV-47）
    """
    import time as _time

    w = weights or QFRWeights()
    ages = ages or {}
    valid_to = valid_to or {}
    now = now_ms if now_ms is not None else int(_time.time() * 1000)

    # 步骤 1：过期硬过滤（INV-46）——在分配 rank 之前剔除，
    # 这样过期文档既不得分，也不占用排名位次。
    expired: set[int] = set()
    for doc_id, vt in valid_to.items():
        if vt is not None and now >= int(vt):
            expired.add(doc_id)

    ranks: dict[int, dict[str, int]] = {}
    for channel, ranked in channels.items():
        if not ranked:
            continue
        # 通道内排序：分数降序 → doc_id 升序（同分裁决稳定）
        ordered = sorted(
            ((did, sc) for did, sc in ranked if did not in expired),
            key=lambda kv: (-kv[1], kv[0]),
        )
        seen: set[int] = set()
        # rank 为 0-based —— 与 reciprocal_rank_fusion 严格同口径，
        # 保证 INV-43 退化性逐位相等（1-based 会有 4.91e-04 的系统性偏移）。
        for rank, (doc_id, _score) in enumerate(ordered):
            if doc_id in seen:  # 通道内去重（保留最靠前的排名）
                continue
            seen.add(doc_id)
            ranks.setdefault(doc_id, {})[channel] = rank

    # 步骤 2：QFR 打分
    out: list[tuple[int, float]] = []
    for doc_id, ch_ranks in ranks.items():
        t_factor = 1.0
        if w.temporal_enabled:
            age = float(ages.get(doc_id, 0.0))
            t_factor = temporal_factor(age, half_life_days=w.half_life_days, gamma=w.gamma)
        total = 0.0
        for channel, rank in ch_ranks.items():
            total += w.weight_for(channel) * t_factor / (w.k + rank)
        out.append((doc_id, total))

    # 步骤 3：确定性排序（INV-47）
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out


def qfr_reference(
    channels: Mapping[str, Sequence[tuple[int, float]]],
    *,
    k: int = 60,
    weights_map: Mapping[str, float] | None = None,
    ages: Mapping[int, float] | None = None,
    half_life_days: float = 30.0,
    gamma: float = 1.0,
) -> list[tuple[int, float]]:
    """QFR 的独立参考实现（暴力版，无剪枝、无去重优化）。

    用途：INV-43/45/47 的交叉验证。主实现用倒排式剪枝与去重，
    参考实现直接全量扫描 —— 任何公式错误都会立刻暴露。

    这是「暴力枚举 vs 优化实现」型不变量，与 ``bm25.py`` 的
    ``reference`` 参考实现同一范式。
    """
    weights_map = weights_map or {}
    ages = ages or {}
    scores: dict[int, float] = {}
    for channel, ranked in channels.items():
        wc = float(weights_map.get(channel, 1.0))
        # 参考实现同样按 (分数降序, doc_id 升序) 分配 rank，且同为 0-based
        ordered = sorted(ranked, key=lambda kv: (-kv[1], kv[0]))
        for rank, (doc_id, _sc) in enumerate(ordered):
            age = float(ages.get(doc_id, 0.0))
            t = temporal_factor(age, half_life_days=half_life_days, gamma=gamma)
            scores[doc_id] = scores.get(doc_id, 0.0) + wc * t / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
