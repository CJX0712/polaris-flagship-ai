"""L3 连续批处理调度器 (Continuous Batching)

零重写: 复用 Yu et al. 2022 "Orca: A Distributed Serving System for
Transformer-Based Generative Models" 的连续批处理 (iteration-level
scheduling) — 不同序列在不同步数结束, 每步仅对"仍在生成"的序列做前向。

可验证:
- INV-16 贪心下, 连续批处理结果与对每条序列"串行逐 token 自回归"的结果 100% 一致
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from .sampler import Sampler


class ContinuousScheduler:
    """对同一批 prompt 做并行贪心/采样生成。

    ``model_forward(ids: np.ndarray[B,T]) -> logits[B,T,V]`` 为统一前向接口。
    ``sampler`` 为可选 L3 采样器 (temperature/top_p/min_p 等); ``greedy=False``
    时走采样 — v0.7.13 修复历史 bug: 旧实现 else 分支与贪心逐字相同,
    ``greedy=False`` 恒 argmax (采样参数形同虚设)。
    """

    def __init__(self, model_forward: Callable[[np.ndarray], np.ndarray],
                 sampler: Sampler | None = None):
        self.fwd = model_forward
        self.sampler = sampler

    def _sample(self, logits_1d: np.ndarray) -> int:
        if self.sampler is None:
            self.sampler = Sampler()
        return int(self.sampler.sample(logits_1d))

    def generate(self, prompts: List[List[int]], max_new_tokens: int,
                 eos_id: int | None = None, greedy: bool = True,
                 pad_id: int = 0) -> List[List[int]]:
        # 维护各序列 (含 prompt) 的可变 token 列表
        seqs = [list(p) for p in prompts]
        finished = [False] * len(seqs)
        produced = [0] * len(seqs)
        for _ in range(max_new_tokens):
            if all(finished):
                break
            live = [i for i in range(len(seqs)) if not finished[i]]
            # 左补齐到本步最长序列, 真实 token 末位对齐 (便于取 logits[:, -1])
            max_len = max(len(seqs[i]) for i in live)
            batch = np.full((len(live), max_len), pad_id, dtype=np.int64)
            position_ids = np.zeros((len(live), max_len), dtype=np.int64)
            key_padding_mask = np.zeros((len(live), max_len), dtype=bool)
            for row, i in enumerate(live):
                L = len(seqs[i])
                batch[row, max_len - L:] = seqs[i]
                position_ids[row, max_len - L:] = np.arange(L)
                key_padding_mask[row, :max_len - L] = True
            logits = self.fwd(batch, position_ids=position_ids,
                              key_padding_mask=key_padding_mask)   # [B, T, V]
            last = logits[:, -1, :]
            for row, i in enumerate(live):
                nxt = int(np.argmax(last[row])) if greedy else self._sample(last[row])
                seqs[i].append(nxt)
                produced[i] += 1
                if (eos_id is not None and nxt == eos_id) or produced[i] >= max_new_tokens:
                    finished[i] = True
        # 仅返回新生成部分
        return [seqs[i][len(prompts[i]):] for i in range(len(seqs))]
