"""L5 文本分块 (Chunking) — 固定窗口 / 递归字符切分 / 重叠窗口

零重写 (集大成):
- **LangChain ``RecursiveCharacterTextSplitter``**: 递归分隔符优先级
  (``\\n\\n`` → ``\\n`` → 句号 → 空格 → 字符) 的"在自然边界处断开"思想。
- **重叠窗口 (sliding window with overlap)**: 检索领域的经典做法 (见
  Robertson & Zaragoza 2009 的段落级检索讨论), 避免答案跨块被截断。
- 覆盖完整性 (INV-31) 借鉴 **Manning et al. 2008 "Introduction to Information
  Retrieval"** 的"索引必须覆盖全文"要求: 切分只允许重排与重叠, 不允许丢字。

可验证不变量:
- **INV-31 分块覆盖完整性**: 按 offset 去重叠拼接后 == 原文 (归一化后 100%);
  ``overlap=0`` 时块文本的**直接串联**即原文 (零丢字、零重复)。
  每个块携带 ``start``/``end`` 原文偏移, 保证"边界不丢不重"可逐块核验。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

# 递归分隔符优先级 (由"最强语义边界"到"最弱")
DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]

_WS_RE = re.compile(r"\s+")


@dataclass
class Chunk:
    """一个文本块。

    Attributes:
        text: 块文本 (恒等于 ``原文[start:end]``)。
        start: 在原文中的起始偏移 (含)。
        end: 在原文中的结束偏移 (不含)。
        index: 块序号 (从 0 开始)。
        meta: 附加元数据 (如 ``{"overlap": 64}``)。
    """

    text: str
    start: int
    end: int
    index: int
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)

    @property
    def length(self) -> int:
        """块长度 (字符数)。"""
        return len(self.text)

    def to_dict(self) -> dict:
        return {"text": self.text, "start": self.start, "end": self.end,
                "index": self.index, "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(text=d["text"], start=int(d["start"]), end=int(d["end"]),
                   index=int(d["index"]), meta=dict(d.get("meta") or {}))


def normalize_text(text: str) -> str:
    """归一化: 折叠所有空白为单空格并去首尾 (INV-31 判据用的比较口径)。"""
    return _WS_RE.sub(" ", str(text)).strip()


def reconstruct(chunks: Sequence[Chunk]) -> str:
    """按 offset 去重叠拼接块文本, 还原原文 (INV-31)。

    相邻块若有重叠 (``prev.end > cur.start``), 则跳过当前块的前
    ``prev.end - cur.start`` 个字符, 从而既不丢字也不重复。
    """
    parts: List[str] = []
    prev_end: Optional[int] = None
    for ch in chunks:
        if prev_end is None:
            parts.append(ch.text)
        else:
            skip = max(0, prev_end - ch.start)
            parts.append(ch.text[skip:] if skip < len(ch.text) else "")
        prev_end = ch.end
    return "".join(parts)


def _snap_end(text: str, start: int, end: int, separators: Sequence[str],
              min_keep: int) -> int:
    """把切分点 ``end`` 回退到最近的语义分隔符之后 (递归字符切分的核心)。

    Args:
        text: 原文。
        start: 块起点 (含)。
        end: 窗口原始终点 (不含)。
        separators: 分隔符优先级列表。
        min_keep: 最小保留长度; 回退后不足该长度则**不回退** (避免产生碎块)。

    Returns:
        回退后的终点 (不含), 恒满足 ``start < ret <= end``。
    """
    if end >= len(text):
        return len(text)
    best, best_len = -1, 0
    for sep in separators:
        if not sep:
            continue
        i = text.rfind(sep, start, end)
        if i > best:
            best, best_len = i, len(sep)
    if best < 0:
        return end
    cand = best + best_len
    if cand <= start + min_keep:
        return max(end, start + 1)
    return cand


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64,
               strategy: str = "fixed",
               separators: Optional[Sequence[str]] = None) -> List[Chunk]:
    """把 ``text`` 切成覆盖完整的块 (INV-31)。

    Args:
        text: 原文 (任意 Unicode, 含中文)。
        chunk_size: 目标块长 (字符数)。
        overlap: 相邻块重叠长度, 必须满足 ``0 <= overlap < chunk_size``。
        strategy: ``"fixed"`` 定长窗口 / ``"recursive"`` 递归字符切分
            (在分隔符处回退, 块边界更自然)。
        separators: 自定义分隔符优先级 (仅 ``recursive`` 使用)。

    Returns:
        :class:`Chunk` 列表; 相邻块按 offset 连续铺满 ``[0, len(text))``。

    Raises:
        ValueError: 参数非法 (块长非正 / overlap 越界 / 未知策略)。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")
    if strategy not in ("fixed", "recursive"):
        raise ValueError(f"未知切分策略: {strategy}")
    text = str(text)
    seps = list(separators) if separators is not None else list(DEFAULT_SEPARATORS)
    if not text:
        return []
    min_keep = max(1, int(chunk_size * 0.5)) if strategy == "recursive" else 1
    chunks: List[Chunk] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if strategy == "recursive":
            end = _snap_end(text, start, end, seps, min_keep)
        if end <= start:                       # 防御: 保证单调前进
            end = min(start + chunk_size, n)
        chunks.append(Chunk(text=text[start:end], start=start, end=end,
                            index=len(chunks),
                            meta={"overlap": int(overlap), "strategy": strategy}))
        if end >= n:
            break
        # 下一块起点: 回退 overlap 个字符形成重叠; 保证严格前进 (不丢字)
        nxt = end - overlap
        start = nxt if nxt > start else end
    return chunks


def verify_coverage(text: str, chunks: Sequence[Chunk],
                    normalized: bool = True) -> bool:
    """INV-31 判据: 拼接后的文本是否 == 原文。

    Args:
        text: 原文。
        chunks: 待核验的块序列。
        normalized: True 时按 :func:`normalize_text` 口径比较 (PRD 判据);
            False 时要求严格逐字符相等。

    Returns:
        True 表示覆盖 100% (无丢字、无重复)。
    """
    joined = reconstruct(chunks)
    if normalized:
        return normalize_text(joined) == normalize_text(text)
    return joined == text


def coverage_report(text: str, chunks: Sequence[Chunk]) -> dict:
    """返回覆盖完整性报告, 便于断言与调试。

    Returns:
        dict: ``n_chunks`` / ``covered_chars`` (去重后的净覆盖字符数) /
        ``n_text_chars`` / ``exact`` / ``normalized`` / ``offsets_valid`` /
        ``contiguous`` (块是否连续铺满, 无空洞)。
    """
    n = len(text)
    covered = 0
    prev_end = 0
    contiguous = True
    offsets_valid = True
    for ch in chunks:
        if text[ch.start:ch.end] != ch.text:
            offsets_valid = False
        if ch.start > prev_end:
            contiguous = False
        if ch.end > prev_end:
            covered += ch.end - max(prev_end, ch.start)
            prev_end = ch.end
    return {
        "n_chunks": len(chunks),
        "covered_chars": covered,
        "n_text_chars": n,
        "exact": reconstruct(chunks) == text,
        "normalized": verify_coverage(text, chunks, normalized=True),
        "offsets_valid": offsets_valid,
        "contiguous": contiguous and (prev_end >= n if chunks else n == 0),
    }


class TextChunker:
    """面向对象封装: 固定配置复用, 并提供覆盖自检。"""

    def __init__(self, chunk_size: int = 512, overlap: int = 64,
                 strategy: str = "fixed",
                 separators: Optional[Sequence[str]] = None):
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.strategy = strategy
        self.separators = separators

    def split(self, text: str) -> List[Chunk]:
        """切分文本为覆盖完整的块 (INV-31)。"""
        return chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap,
                          strategy=self.strategy, separators=self.separators)

    def reconstruct(self, chunks: Sequence[Chunk]) -> str:
        """去重叠拼接还原原文。"""
        return reconstruct(chunks)

    def verify(self, text: str, chunks: Optional[Sequence[Chunk]] = None) -> dict:
        """覆盖完整性报告 (块未提供时内部先切分)。"""
        cs = chunks if chunks is not None else self.split(text)
        return coverage_report(text, cs)

    def split_many(self, texts: Iterable[str]) -> List[List[Chunk]]:
        """批量切分 (便于构建语料级索引)。"""
        return [self.split(t) for t in texts]
