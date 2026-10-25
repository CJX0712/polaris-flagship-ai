"""L2 分词器 — 字节级 BPE (Byte-Pair Encoding)

零重写: 复用 Sennrich et al. 2015 "Neural Machine Translation of Rare Words
with Subword Units" 的 BPE 合并算法；字符级符号空间规避 UTF-8 边界问题
(与 GPT-2 byte-level 思路一致)。

可验证: 同一语料训练 + 同一随机种子 ⇒ 词表/merges 一致 (INV-40 派生)；
encode→decode 保真 (round-trip)。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import List

_SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class Tokenizer:
    def __init__(self, vocab_size: int = 32000, specials: List[str] | None = None):
        self.vocab_size_target = vocab_size
        self.specials = (specials or []) + [s for s in _SPECIALS if s not in (specials or [])]
        self.merges: List[tuple] = []               # [(a,b), ...] 有序
        self.merge_ranks: dict = {}
        self.token2id: dict = {}
        self.id2token: dict = {}

    # ---------------- 训练 ----------------
    def train(self, corpus: str, min_freq: int = 2):
        # 初始符号: 语料中所有字符
        chars = sorted(set(corpus))
        symbols = list(self.specials) + chars
        self.token2id = {s: i for i, s in enumerate(symbols)}
        self.id2token = {i: s for s, i in self.token2id.items()}

        words = re.findall(r"\S+", corpus)  # 以空白切分
        # 转成 symbol 序列 (每个字符一个 symbol)
        def to_syms(w):
            return list(w)

        word_syms = [to_syms(w) for w in words]
        max_merges = max(0, self.vocab_size_target - len(self.token2id))
        for _ in range(max_merges):
            pair_freq = Counter()
            for syms in word_syms:
                for a, b in zip(syms[:-1], syms[1:]):
                    pair_freq[(a, b)] += 1
            if not pair_freq:
                break
            # 取频率最高且 ≥ min_freq
            best = max((p for p, c in pair_freq.items() if c >= min_freq),
                       key=lambda p: pair_freq[p], default=None)
            if best is None:
                break
            a, b = best
            merged = a + b
            self.merges.append((a, b))
            self.merge_ranks[(a, b)] = len(self.merges)
            new_id = len(self.token2id)
            self.token2id[merged] = new_id
            self.id2token[new_id] = merged
            # 应用该合并到语料
            for syms in word_syms:
                i = 0
                new_syms = []
                while i < len(syms):
                    if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                        new_syms.append(merged)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                syms[:] = new_syms
        return self

    # ---------------- 编解码 ----------------
    def _apply_merges(self, syms: List[str]) -> List[str]:
        while True:
            # 找到最高优先级的相邻对
            best_i, best_rank = -1, None
            for i in range(len(syms) - 1):
                key = (syms[i], syms[i + 1])
                if key in self.merge_ranks:
                    r = self.merge_ranks[key]
                    if best_rank is None or r < best_rank:
                        best_rank, best_i = r, i
            if best_i < 0:
                break
            a, b = syms[best_i], syms[best_i + 1]
            syms[best_i:best_i + 2] = [a + b]
        return syms

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for w in re.findall(r"\S+", text):
            syms = self._apply_merges(list(w))
            for s in syms:
                if s in self.token2id:
                    ids.append(self.token2id[s])
                else:
                    ids.append(self.token2id["<unk>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id2token.get(i, "<unk>") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    # ---------------- 序列化 ----------------
    def to_json(self, path: str | None = None) -> str:
        obj = {"vocab_size_target": self.vocab_size_target,
               "specials": self.specials,
               "merges": self.merges,
               "token2id": self.token2id}
        s = json.dumps(obj, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    @classmethod
    def from_json(cls, source: str | dict) -> "Tokenizer":
        obj = json.loads(source) if isinstance(source, str) else source
        t = cls(vocab_size=obj["vocab_size_target"], specials=obj["specials"])
        t.merges = [tuple(m) for m in obj["merges"]]
        t.merge_ranks = {tuple(m): i for i, m in enumerate(t.merges)}
        t.token2id = {k: int(v) for k, v in obj["token2id"].items()}
        t.id2token = {int(v): k for k, v in obj["token2id"].items()}
        return t
