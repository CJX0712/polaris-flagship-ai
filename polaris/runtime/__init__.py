"""操作系统层（runtime）公开接口。

零重写 · 出处：四件套 + 门面，对应 NEXUS 的 L5/L6/L8 在 Python 侧的合流：

- ``registry``   引擎注册表（设备管理器）
- ``scheduler``  约束调度器（按确定性/上下文/优先级选引擎）
- ``gates``      运行时门禁 G1 确定性 / G2 有限性 / G3 契约
- ``audit``      哈希链审计账本（追加-only，可独立重放验证）
- ``orchestrator`` PolarisOS 门面（单一入口）
"""

from __future__ import annotations

from .audit import AuditLog
from .gates import (
    GateReport,
    gate_g1_determinism,
    gate_g2_finiteness,
    gate_g3_contract,
    run_gates,
)
from .orchestrator import GateViolation, PolarisOS
from .registry import EngineRegistry
from .scheduler import DispatchRequest, DispatchDecision, NoEngineAvailable, Scheduler

__all__ = [
    "AuditLog",
    "GateReport",
    "run_gates",
    "gate_g1_determinism",
    "gate_g2_finiteness",
    "gate_g3_contract",
    "PolarisOS",
    "GateViolation",
    "EngineRegistry",
    "DispatchRequest",
    "DispatchDecision",
    "NoEngineAvailable",
    "Scheduler",
]
