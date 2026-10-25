"""引擎层公开接口。

零重写 · 出处：北极星 POLARIS 的引擎体系（对应 NEXUS 的引擎哲学）：

1. ``OpenAICompatEngine``  —— **引擎联邦**：一份 OpenAI Completions 契约
   吃下 llama.cpp / vLLM / SGLang / Ollama / LM Studio / TGI / MLX /
   TensorRT-LLM / NIM / 任意兼容网关（``PRESETS`` 为预设表，
   ``autodiscover()`` 发现式注册）；
2. ``LlamaCppAdapter``  —— 世界顶级外部引擎（llama.cpp）的具名入口；
3. ``BuiltinEngineAdapter`` —— 内置旗舰（ifai 全栈），确定性回退；
4. ``MockEngine`` —— 测试替身，链路验证。

调度策略见 ``polaris.runtime.scheduler``；交叉验证见 ``polaris.runtime.consensus``。
"""

from __future__ import annotations

from .base import EngineAdapter, EngineResult, is_finite_number, validate_result
from .builtin import BuiltinEngineAdapter
from .federation import (
    PRESETS,
    EnginePreset,
    OpenAICompatEngine,
    autodiscover,
    build_engine,
    preset_table,
)
from .llamacpp import DEFAULT_BASE_URL, ENV_URL, LlamaCppAdapter
from .mock import MockEngine

__all__ = [
    "EngineAdapter",
    "EngineResult",
    "validate_result",
    "is_finite_number",
    "OpenAICompatEngine",
    "EnginePreset",
    "PRESETS",
    "build_engine",
    "autodiscover",
    "preset_table",
    "LlamaCppAdapter",
    "BuiltinEngineAdapter",
    "MockEngine",
    "DEFAULT_BASE_URL",
    "ENV_URL",
]
