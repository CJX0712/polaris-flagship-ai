"""引擎注册表 —— Polaris 的「设备管理器」。

零重写 · 出处
------------
注册表模式（registry pattern）是 Python 世界（``importlib``、
``setuptools entry_points``、``pluggy``）与 LLM 服务生态
（LiteLLM 的 provider registry、vLLM 的 engine 注册）的标准形态。
本模块只做最朴素的那一件事：名字 → 适配器实例 的唯一真相源。
"""

from __future__ import annotations

from ..engines.base import EngineAdapter


class EngineRegistry:
    """引擎注册表：全局唯一的「名字 → 适配器」映射。

    用法::

        reg = EngineRegistry()
        reg.register(LlamaCppAdapter())
        reg.register(BuiltinEngineAdapter())
        reg.available()          # -> ["llama.cpp", "ifai-builtin"]
        reg.get("llama.cpp")     # -> adapter | None
    """

    def __init__(self) -> None:
        self._adapters: dict[str, EngineAdapter] = {}

    def register(self, adapter: EngineAdapter) -> EngineAdapter:
        """注册引擎（重名覆盖，后注册者生效 —— 与插件生态一致）。"""
        if not adapter.name:
            raise ValueError("adapter.name 不能为空")
        self._adapters[adapter.name] = adapter
        return adapter

    def get(self, name: str) -> EngineAdapter | None:
        return self._adapters.get(name)

    def names(self) -> list[str]:
        return sorted(self._adapters)

    def available(self) -> list[str]:
        """当前可用的引擎名列表（有序，确定性）。"""
        return sorted(n for n, a in self._adapters.items() if a.available())

    def all(self) -> list[EngineAdapter]:
        return [self._adapters[n] for n in sorted(self._adapters)]

    def __contains__(self, name: str) -> bool:
        return name in self._adapters

    def __len__(self) -> int:
        return len(self._adapters)


__all__ = ["EngineRegistry"]
