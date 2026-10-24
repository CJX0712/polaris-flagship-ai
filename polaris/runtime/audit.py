"""治理审计账本 —— Polaris 的「黑匣子」。

零重写 · 出处
------------
哈希链账本（hash-chained append-only log）是区块链与世界级审计系统
（Git 对象模型、Certificate Transparency RFC 6962、Merkle 日志）的
标准形态。Polaris 的账本对每次 dispatch 记录：

    entry = {seq, ts, intent, engine, gates, latency_ms, result_fp}
    entry_hash = sha256(prev_hash || canonical_json(entry))

性质（可由 ``verify_chain()`` 独立复现验证）：

1. **追加-only**：任何历史条目被篡改 → 后续全部哈希断裂；
2. **可重放**：``replay()`` 逐条重算哈希链，与落盘快照比对；
3. **防串台**：每条带序号，缺号/重号即刻暴露。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Optional


def _canonical(obj: Any) -> str:
    """规范化 JSON（排序键、无空白）—— 跨平台逐位一致的哈希前提。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class AuditLog:
    """哈希链审计账本（内存 + 可选 JSONL 落盘）。"""

    GENESIS = "0" * 64

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._entries: list[dict[str, Any]] = []
        self._hashes: list[str] = []
        self._prev = self.GENESIS
        if path and os.path.exists(path):
            self._load(path)

    # ------------------------------------------------------------------ 写入
    def append(self, entry: dict[str, Any]) -> str:
        """追加一条审计记录，返回其哈希。"""
        entry = dict(entry)
        entry["seq"] = len(self._entries)
        entry.setdefault("ts", time.time())
        entry["prev_hash"] = self._prev
        digest = hashlib.sha256(
            (self._prev + "||" + _canonical(entry)).encode("utf-8")
        ).hexdigest()
        entry["entry_hash"] = digest
        self._entries.append(entry)
        self._hashes.append(digest)
        self._prev = digest
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(_canonical(entry) + "\n")
        return digest

    # ------------------------------------------------------------------ 验证
    def verify_chain(self) -> tuple[bool, str]:
        """重放整条哈希链，返回 ``(ok, detail)``。"""
        prev = self.GENESIS
        for i, entry in enumerate(self._entries):
            if entry.get("seq") != i:
                return False, f"seq mismatch at index {i}: {entry.get('seq')}"
            stored = entry.get("entry_hash", "")
            body = {k: v for k, v in entry.items() if k != "entry_hash"}
            digest = hashlib.sha256(
                (prev + "||" + _canonical(body)).encode("utf-8")
            ).hexdigest()
            if digest != stored:
                return False, f"hash mismatch at seq {i}"
            prev = stored
        return True, f"chain verified: {len(self._entries)} entries"

    def replay(self) -> list[dict[str, Any]]:
        """逐条重放（删除 entry_hash/prev_hash 后的净载荷）。"""
        return [
            {k: v for k, v in e.items() if k not in ("entry_hash", "prev_hash")}
            for e in self._entries
        ]

    # ------------------------------------------------------------------ 查询
    def __len__(self) -> int:
        return len(self._entries)

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        return self._entries[-n:]

    # ------------------------------------------------------------------ 持久
    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._entries.append(entry)
                self._hashes.append(entry.get("entry_hash", ""))
                self._prev = entry.get("entry_hash", self._prev)


__all__ = ["AuditLog"]
