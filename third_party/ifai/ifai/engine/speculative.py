"""L3 投机解码 (Speculative Decoding)

零重写: 复用 Leviathan et al. 2023 "Fast Inference from Transformers via
Speculative Decoding" 与 Chen et al. 2023 (Medusa) / EAGLE 的"草稿-验证"
框架。草稿模型 (小/快) 一次提出 γ 个候选, 目标模型并行验证, 多数接受时
显著加速。

可验证 (INV-17): 贪心模式下, 投机解码逐 token 结果与朴素自回归贪心 100% 一致。
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from .. import rng


def speculative_decode(target_next: Callable[[List[int]], np.ndarray],
                       draft_next: Callable[[List[int]], np.ndarray],
                       prompt: List[int], gamma: int = 4,
                       max_new_tokens: int = 32, eos_id: int | None = None,
                       greedy: bool = True, temperature: float = 1.0) -> List[int]:
    """返回新生成的 token 列表 (不含 prompt)。

    target_next(seq) / draft_next(seq): 给定前缀返回下一个位置的 logits[V]。
    """
    r = rng.get_rng()
    seq = list(prompt)
    while len(seq) - len(prompt) < max_new_tokens:
        # 1) 草稿模型自回归提出 γ 个候选 (受剩余额度约束)
        draft_tokens: List[int] = []
        cur = list(seq)
        for _ in range(gamma):
            if len(seq) - len(prompt) >= max_new_tokens:
                break
            lg = np.asarray(draft_next(cur), dtype=np.float64)
            if greedy:
                tok = int(np.argmax(lg))
            else:
                p = _softmax(lg / max(temperature, 1e-8))
                tok = int(r.choice(len(p), p=p))
            draft_tokens.append(tok)
            cur.append(tok)
        # 2) 目标模型逐位验证 (同样受剩余额度约束)
        for tok in draft_tokens:
            if len(seq) - len(prompt) >= max_new_tokens:
                break
            tgt = np.asarray(target_next(seq), dtype=np.float64)
            if greedy:
                tgt_tok = int(np.argmax(tgt))
                if tok == tgt_tok:
                    seq.append(tok)
                else:
                    seq.append(tgt_tok)
                    break
            else:
                p_t = _softmax(tgt / max(temperature, 1e-8))
                p_d = _softmax(np.asarray(draft_next(seq), dtype=np.float64) / max(temperature, 1e-8))
                p_d = max(p_d[tok], 1e-12)
                accept = min(1.0, p_t[tok] / p_d)
                if r.random() < accept:
                    seq.append(tok)
                else:
                    seq.append(int(np.argmax(tgt)))
                    break
        if eos_id is not None and seq[-1] == eos_id:
            break
    return seq[len(prompt):]


def _softmax(x):
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()
