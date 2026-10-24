"""PolarisOS —— 旗舰 AI 操作系统门面。

这是 Polaris 的单一入口，对应 NEXUS 的 API 门面（L7）。把舰队全部
顶级件编排为一个可编程系统：

    polaris = PolarisOS()
    polaris.register_engine(LlamaCppAdapter())        # 世界顶级外部引擎
    polaris.register_engine(BuiltinEngineAdapter())   # 内置旗舰回退
    result = polaris.dispatch("你好", max_tokens=32)  # 调度+门禁+审计
    polaris.run_isa("smax.softmax", x)                # 1024 微指令直执
    polaris.isa_stats()                               # ISA 统计

零重写 · 出处
------------
- 编排契约：NEXUS（CJX0712/nexus-flagship, MIT，作者晨星）的
  ``registerEngine / dispatch / run / isa / benchmark`` 门面；
- 1024 微指令：ifai L10（移植自 tianji-ai，MIT），经本门面原样暴露；
- 引擎/门禁/审计：本仓 ``polaris.engines`` / ``polaris.runtime``。
"""

from __future__ import annotations

import time
from typing import Any, Optional

from .._vendor_path import ensure_vendor_path

ensure_vendor_path()

from ifai.l10 import FAMILIES, FAMILY_KIND, MICROOPS, count_ops  # noqa: E402

from ..engines.base import EngineAdapter, EngineResult  # noqa: E402
from .audit import AuditLog  # noqa: E402
from .gates import GateReport, run_gates  # noqa: E402
from .registry import EngineRegistry  # noqa: E402
from .scheduler import DispatchRequest, NoEngineAvailable, Scheduler  # noqa: E402


class PolarisOS:
    """北极星旗舰 AI 操作系统。

    Args:
        audit_path: 审计账本落盘路径（JSONL）。None 则仅内存。
        g1_replay: None=自动（外部引擎跳过、内置/mock 执行），True/False 强制。
    """

    def __init__(self, audit_path: Optional[str] = None, g1_replay: Optional[bool] = None):
        self.registry = EngineRegistry()
        self.scheduler = Scheduler(self.registry)
        self.audit = AuditLog(path=audit_path)
        self.g1_replay = g1_replay

    # ---------------------------------------------------------------- 引擎
    def register_engine(self, adapter: EngineAdapter) -> EngineAdapter:
        """注册引擎（返回适配器，便于链式）。"""
        return self.registry.register(adapter)

    def engines(self) -> list[dict[str, Any]]:
        """全部引擎的自描述（doctor / engines 消费）。"""
        return [a.describe() for a in self.registry.all()]

    # ---------------------------------------------------------------- 调度
    def dispatch(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 0,
        require_deterministic: bool = True,
        preferred: Optional[str] = None,
    ) -> EngineResult:
        """端到端：调度 → 生成 → 门禁 → 审计。

        Raises:
            NoEngineAvailable: 无引擎满足约束（诚实失败）。
            GateViolation: 门禁拦截（结果不可信，绝不放行）。
        """
        request = DispatchRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            require_deterministic=require_deterministic,
            preferred=preferred,
        )
        t0 = time.perf_counter()
        decision = self.scheduler.decide(request)
        adapter = decision.adapter

        result = adapter.complete(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )

        # G1 重放：内置/mock 引擎双调比对指纹（外部引擎默认跳过，
        # 其确定性由 llama.cpp 自身保证，双调成本不诚实）。
        replay: Optional[EngineResult] = None
        do_replay = self.g1_replay
        if do_replay is None:
            do_replay = adapter.kind in ("builtin-flagship", "mock")
        if do_replay:
            replay = adapter.complete(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )

        reports: list[GateReport] = run_gates(
            result,
            registry_names=set(self.registry.names()),
            max_tokens=max_tokens,
            replay=replay,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self.audit.append(
            {
                "intent": "dispatch",
                "prompt_fp": _fp(prompt),
                "engine": result.engine,
                "kind": adapter.kind,
                "reason": decision.reason,
                "gates": [r.as_dict() for r in reports],
                "latency_ms": round(latency_ms, 3),
                "result_fp": result.fingerprint(),
                "degraded": result.degraded,
            }
        )

        failed = [r for r in reports if not r.passed]
        if failed:
            raise GateViolation(result, failed)

        result.meta["dispatch"] = {
            "reason": decision.reason,
            "candidates": decision.candidates,
            "gates": [r.as_dict() for r in reports],
            "latency_ms": round(latency_ms, 3),
        }
        return result

    # ---------------------------------------------------------------- ISA
    def run_isa(self, op_name: str, *args: Any) -> Any:
        """直接执行一条 1024 微指令 ISA 成员（如 ``smax.softmax``）。

        零重写：``ifai.l10.MICROOPS`` 原样暴露，1024 条全部可调用。
        """
        op = MICROOPS.get(op_name)
        if op is None:
            raise KeyError(
                f"unknown micro-op {op_name!r}; use isa_stats() to list families"
            )
        if op.arity == 1:
            return op.fn(args[0])
        if op.arity == 2:
            return op.fn(args[0], args[1])
        raise ValueError(f"unsupported arity {op.arity} for {op_name}")

    def isa_stats(self) -> dict[str, Any]:
        """ISA 统计：1024 条 / 32 家族 / core-sweep 分类（诚实记账）。"""
        kinds: dict[str, int] = {}
        for kind in FAMILY_KIND.values():
            kinds[kind] = kinds.get(kind, 0) + 1
        return {
            "total_ops": count_ops(),
            "families": len(FAMILIES),
            "family_kinds": kinds,
            "family_names": sorted(FAMILIES),
        }

    def isa_gates(self) -> dict[str, Any]:
        """跑 ifai L10 全量微指令门禁（冒烟 + 数学性质断言）。"""
        from ifai.l10 import run_all

        return run_all()

    # ---------------------------------------------------------------- 审计
    def audit_tail(self, n: int = 10) -> list[dict[str, Any]]:
        return self.audit.tail(n)

    def audit_verify(self) -> tuple[bool, str]:
        return self.audit.verify_chain()


class GateViolation(RuntimeError):
    """门禁拦截：产出未通过 G1/G2/G3，系统拒绝放行（绝不静默造假）。"""

    def __init__(self, result: EngineResult, failed: list[GateReport]):
        self.result = result
        self.failed = failed
        detail = "; ".join(f"{r.gate}: {r.detail}" for r in failed)
        super().__init__(f"gate violation -> {detail}")


def _fp(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


__all__ = ["PolarisOS", "GateViolation"]
