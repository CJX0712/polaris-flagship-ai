"""Mock 引擎 —— 测试与离线演示专用。

零重写 · 出处
------------
测试替身（test double）是 xUnit / pytest 生态的世界标准实践；
本引擎只做一件事：**确定性地**把 prompt 映射为可预期文本，
使门禁测试不依赖任何外部进程。
"""

from __future__ import annotations

from .base import EngineAdapter, EngineResult


class MockEngine(EngineAdapter):
    """确定性 echo 引擎：``complete("x") -> "mock:x"``。

    用于：
    1. 调度/门禁/审计链路的单元测试（不依赖 llama.cpp / 模型文件）；
    2. 无任何引擎可用时的最后占位（调用方会看到 ``degraded=True``）。
    """

    name = "mock"
    kind = "mock"
    capabilities = {"modalities": ["text"], "deterministic": True, "maxTokens": 1024}

    def __init__(self, prefix: str = "mock"):
        self._prefix = prefix

    def available(self) -> bool:
        return True

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> EngineResult:
        text = f"{self._prefix}:{prompt}"
        if max_tokens and len(text) > max_tokens:
            text = text[:max_tokens]
        return EngineResult(
            text=text,
            engine=self.name,
            tokens=None,
            degraded=True,
            meta={"adapter": "mock", "echo_of": prompt},
        )


__all__ = ["MockEngine"]
