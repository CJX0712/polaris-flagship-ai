"""北极星 POLARIS · 晨星旗舰 AI 操作系统。

零重写 · 集大成 · 可运行 · 可复现 · 可验证

    from polaris import PolarisOS
    from polaris.engines import LlamaCppAdapter, BuiltinEngineAdapter

    os_ = PolarisOS()
    os_.register_engine(LlamaCppAdapter())          # 世界顶级外部引擎
    os_.register_engine(BuiltinEngineAdapter())     # 内置旗舰回退
    print(os_.dispatch("你好", max_tokens=32).text)

作者：晨星 (Morningstar) · CJX0712 · Apache-2.0
"""

from __future__ import annotations

from ._vendor_path import ensure_vendor_path

ensure_vendor_path()

__version__ = "1.0.0"
__author__ = "晨星 (Morningstar)"

from .engines import (  # noqa: E402,F401
    BuiltinEngineAdapter,
    EngineAdapter,
    EngineResult,
    LlamaCppAdapter,
    MockEngine,
)
from .runtime import (  # noqa: E402,F401
    AuditLog,
    EngineRegistry,
    GateViolation,
    PolarisOS,
    Scheduler,
)

__all__ = [
    "__version__",
    "__author__",
    "PolarisOS",
    "GateViolation",
    "EngineRegistry",
    "Scheduler",
    "AuditLog",
    "EngineAdapter",
    "EngineResult",
    "LlamaCppAdapter",
    "BuiltinEngineAdapter",
    "MockEngine",
]
