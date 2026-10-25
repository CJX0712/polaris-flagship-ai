"""L3 分块 prefill (Chunked Prefill)

零重写 (集大成):
- **vLLM / PagedAttention** (Kwon et al. 2023 "Efficient Memory Management for
  Large Language Model Serving with PagedAttention"): 长 prompt 切成固定大小
  (默认 512) 的块逐块前向, 峰值激活只与块长相关, 而非总长; 与 **Orca**
  (Yu et al. 2022) 的迭代级调度组合后, prefill 可与 decode 交错执行。
- **FlashAttention** (Dao et al. 2022) 的在线 softmax 语义: 分块只是**重排**
  同一份 ``(i) -> logits`` 计算, 不做任何跨块归一化或近似。
- 历史 KV 的组织复用 L3 的 :class:`ifai.engine.kvcache.PagedKVCache` 语义
  (INV-15: 分页 KV ≡ 稠密 KV, 仅重排不改变数值)。

可验证不变量:
- **INV-20 分块 prefill ≡ 整段 prefill**: 逐块累积前向 (历史 KV 逐块拼接)
  与一次性整段前向的 logits 逐位比较, ``max_abs_diff ≤ 1e-5``。
  等价性来源: 位置 i 的 query 在两种路径下看到**完全相同、同序**的 key 集合
  ``0..i`` (causal masking, INV-11), 故 softmax 输入逐位相同。

契约: ``model.forward(ids, position_ids=..., past_kv=..., return_kv=True)``
返回 ``(logits, present_kv)``; :class:`ifai.models.transformer.Transformer`
满足该契约 (``present_kv`` 为每层 ``(K, V)``, 形状 ``[B, T_total, n_kv, d_head]``)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_CHUNK_SIZE = 512
DEFAULT_TOL = 1e-5

KVPair = Tuple[np.ndarray, np.ndarray]


@dataclass
class PrefillResult:
    """一次 prefill 的产物。

    Attributes:
        logits: 形状 ``[B, T, V]`` 的 logits (``return_hidden=True`` 时为隐状态)。
        kv_cache: 每层 ``(K, V)`` 列表, 形状 ``[B, T, n_kv, d_head]``。
        n_chunks: 实际切分的块数 (整段 prefill 为 1)。
        chunk_size: 本次使用的块长。
    """

    logits: np.ndarray
    kv_cache: List[KVPair] = field(default_factory=list)
    n_chunks: int = 1
    chunk_size: int = 0

    @property
    def n_tokens(self) -> int:
        """prefill 的 token 数 (不含批维)。"""
        return int(self.logits.shape[1])

    def last_logits(self) -> np.ndarray:
        """返回最后一个位置的 logits ``[B, V]`` (decode 的起点)。"""
        return self.logits[:, -1, :]


class ChunkedPrefill:
    """分块 prefill 与整段 prefill 的统一入口 (INV-20)。

    用法::

        prefill = ChunkedPrefill(model, chunk_size=512)
        res = prefill.run(ids)                 # 分块累积前向
        ref = prefill.run_whole(ids)           # 整段前向 (基准)
        report = prefill.verify(ids)           # {"max_abs_diff": ..., "equal": True}
    """

    def __init__(self, model, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 tol: float = DEFAULT_TOL):
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        self.model = model
        self.chunk_size = int(chunk_size)
        self.tol = float(tol)

    # ---------------- 核心前向 ----------------
    def run(self, token_ids: np.ndarray, chunk_size: Optional[int] = None,
            return_hidden: bool = False) -> PrefillResult:
        """分块累积前向: 每块只看历史 KV, 峰值激活与块长成正比。

        Args:
            token_ids: ``[T]`` 或 ``[B, T]`` 的 token id。
            chunk_size: 覆盖默认块长 (调试/测试用)。
            return_hidden: True 时返回最后隐状态而非 logits。

        Returns:
            :class:`PrefillResult`。
        """
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None]
        B, T = ids.shape
        cs = int(chunk_size) if chunk_size else self.chunk_size
        cs = max(1, min(cs, T))
        kv: Optional[List[KVPair]] = None
        parts: List[np.ndarray] = []
        n_chunks = 0
        for start in range(0, T, cs):
            end = min(start + cs, T)
            chunk = ids[:, start:end]
            # 绝对位置: 保证 RoPE 与整段 prefill 一致 (INV-03/INV-20)
            position_ids = np.arange(start, end, dtype=np.int64)
            out, present = self.model.forward(
                chunk, position_ids=position_ids, past_kv=kv,
                return_hidden=return_hidden, return_kv=True)
            parts.append(np.asarray(out))
            kv = present
            n_chunks += 1
        logits = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        return PrefillResult(logits=logits, kv_cache=kv or [],
                             n_chunks=n_chunks, chunk_size=cs)

    def run_whole(self, token_ids: np.ndarray,
                  return_hidden: bool = False) -> PrefillResult:
        """整段一次性前向 (INV-20 的对照基准)。"""
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None]
        T = ids.shape[1]
        out, present = self.model.forward(
            ids, position_ids=np.arange(T, dtype=np.int64),
            return_hidden=return_hidden, return_kv=True)
        return PrefillResult(logits=np.asarray(out), kv_cache=list(present),
                             n_chunks=1, chunk_size=T)

    def last_logits(self, token_ids: np.ndarray,
                    chunk_size: Optional[int] = None) -> np.ndarray:
        """分块 prefill 后取末位 logits ``[B, V]`` (供 decode 使用)。"""
        return self.run(token_ids, chunk_size=chunk_size).last_logits()

    # ---------------- 等价性验证 (INV-20) ----------------
    def verify(self, token_ids: np.ndarray, chunk_size: Optional[int] = None,
               tol: Optional[float] = None) -> dict:
        """对比分块 prefill 与整段 prefill, 返回可断言的报告。

        Returns:
            dict: ``max_abs_diff`` / ``equal`` (是否 ≤ tol) / ``tol`` /
            ``n_chunks`` / ``chunk_size`` / ``n_tokens``。
        """
        tol = self.tol if tol is None else float(tol)
        chunked = self.run(token_ids, chunk_size=chunk_size)
        whole = self.run_whole(token_ids)
        diff = max_abs_diff(chunked.logits, whole.logits)
        return {
            "max_abs_diff": diff,
            "equal": bool(diff <= tol),
            "tol": tol,
            "n_chunks": chunked.n_chunks,
            "chunk_size": chunked.chunk_size,
            "n_tokens": chunked.n_tokens,
        }

    def verify_kv(self, token_ids: np.ndarray,
                  chunk_size: Optional[int] = None,
                  tol: Optional[float] = None) -> dict:
        """分块与整段两条路径的 KV cache 是否逐位等价 (INV-20 的更强判据)。

        KV 相同 ⇒ 后续 decode 阶段也能逐 token 对齐 (INV-16 的前提)。
        """
        tol = self.tol if tol is None else float(tol)
        chunked = self.run(token_ids, chunk_size=chunk_size)
        whole = self.run_whole(token_ids)
        diffs: List[float] = []
        for (kc, vc), (kw, vw) in zip(chunked.kv_cache, whole.kv_cache):
            diffs.append(max_abs_diff(kc, kw))
            diffs.append(max_abs_diff(vc, vw))
        worst = max(diffs) if diffs else 0.0
        return {"max_abs_diff": worst, "equal": bool(worst <= tol), "tol": tol,
                "n_layers": len(diffs)}


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """两个数组的最大绝对差 (形状不一致时返回 ``inf``)。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def chunked_prefill(model, token_ids: np.ndarray,
                    chunk_size: int = DEFAULT_CHUNK_SIZE) -> PrefillResult:
    """函数式入口: 一次分块 prefill。"""
    return ChunkedPrefill(model, chunk_size=chunk_size).run(token_ids)


def full_prefill(model, token_ids: np.ndarray) -> PrefillResult:
    """函数式入口: 整段 prefill (基准)。"""
    return ChunkedPrefill(model).run_whole(token_ids)


def verify_prefill(model, token_ids: np.ndarray,
                   chunk_size: int = DEFAULT_CHUNK_SIZE,
                   tol: float = DEFAULT_TOL) -> dict:
    """函数式入口: INV-20 等价性报告。"""
    return ChunkedPrefill(model, chunk_size=chunk_size, tol=tol).verify(token_ids)
