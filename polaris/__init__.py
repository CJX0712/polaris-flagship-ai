"""北极星 POLARIS · 晨星旗舰 AI 操作系统。

零重写 · 集大成 · 可运行 · 可复现 · 可验证

    from polaris import PolarisOS
    from polaris.engines import LlamaCppAdapter, BuiltinEngineAdapter

    os_ = PolarisOS()
    os_.register_engine(LlamaCppAdapter())          # 世界顶级外部引擎
    os_.register_engine(BuiltinEngineAdapter())     # 内置旗舰回退
    print(os_.dispatch("你好", max_tokens=32).text)

    # 可度量的可信生成：Self-Consistency（Wang et al. 2022）
    cr = os_.self_consistency("1+1=?", k=5)
    print(cr.agreement)                             # 置信度 ∈ [0,1]

    # 意图 → 1024 微指令 ISA 程序（确定性子集）
    prog = os_.compile_intent("把这个向量归一化再做 softmax")
    print(prog.run([1.0, 2.0, 3.0]))

作者：晨星 (Morningstar) · CJX0712 · Apache-2.0
"""

from __future__ import annotations

from ._vendor_path import ensure_vendor_path

ensure_vendor_path()

__version__ = "1.1.0"
__author__ = "晨星 (Morningstar)"

from .compiler import (  # noqa: E402,F401
    ISAProgram,
    NoIntentMatch,
    ProgramError,
    compile_intent,
    route,
)
from .engines import (  # noqa: E402,F401
    PRESETS,
    BuiltinEngineAdapter,
    EngineAdapter,
    EngineResult,
    LlamaCppAdapter,
    MockEngine,
    OpenAICompatEngine,
    autodiscover,
    build_engine,
    preset_table,
)
from .runtime import (  # noqa: E402,F401
    AuditLog,
    Consensus,
    ConsensusResult,
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
    "OpenAICompatEngine",
    "PRESETS",
    "build_engine",
    "autodiscover",
    "preset_table",
    "BuiltinEngineAdapter",
    "MockEngine",
    "Consensus",
    "ConsensusResult",
    "ISAProgram",
    "compile_intent",
    "route",
    "NoIntentMatch",
    "ProgramError",
]
