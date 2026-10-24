"""引擎层公开接口。

零重写 · 出处：北极星 POLARIS 的三级引擎体系（对应 NEXUS 的引擎哲学）：

1. ``LlamaCppAdapter``  —— 世界顶级外部引擎（llama.cpp），智力巅峰；
2. ``BuiltinEngineAdapter`` —— 内置旗舰（ifai 全栈），确定性回退；
3. ``MockEngine`` —— 测试替身，链路验证。

调度策略见 ``polaris.runtime.scheduler``。
"""

from __future__ import annotations

from .base import EngineAdapter, EngineResult, is_finite_number, validate_result
from .builtin import BuiltinEngineAdapter
from .llamacpp import DEFAULT_BASE_URL, ENV_URL, LlamaCppAdapter
from .mock import MockEngine

__all__ = [
    "EngineAdapter",
    "EngineResult",
    "validate_result",
    "is_finite_number",
    "LlamaCppAdapter",
    "BuiltinEngineAdapter",
    "MockEngine",
    "DEFAULT_BASE_URL",
    "ENV_URL",
]
