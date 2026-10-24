"""L6 记忆 (Memory) — 情景 + 语义分层

零重写: 借鉴 Packer et al. 2023 "MemGPT" 的"记忆分层"思想 —— 情景记忆
(episodic, 带时间戳的事件流) 与语义记忆 (semantic, 键值知识)。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Episode:
    ts: float
    text: str


class MemoryStore:
    def __init__(self):
        self.episodic: list[Episode] = []
        self.semantic: dict[str, str] = {}

    def remember_event(self, text: str) -> None:
        self.episodic.append(Episode(time.time(), text))

    def remember_fact(self, key: str, value: str) -> None:
        self.semantic[key] = value

    def recall_events(self, query: str = "", limit: int = 10) -> list[str]:
        out = [e.text for e in self.episodic if (not query or query in e.text)]
        return out[-limit:]

    def recall_fact(self, key: str) -> str | None:
        return self.semantic.get(key)

    def __len__(self):
        return len(self.episodic) + len(self.semantic)
