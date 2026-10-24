"""约束调度器 —— Polaris 的「CPU 调度器」。

零重写 · 出处
------------
调度思想取自两个世界顶级系统：

1. **llama.cpp / vLLM 的引擎选择策略**：按请求约束（模态、上下文长度、
   确定性要求）匹配后端能力，能力不足者出局；
2. **NEXUS 的 Dispatch 契约**（nexus-flagship, MIT，作者晨星）：
   「优先世界顶级外部引擎，无外部引擎时退回内置确定性参考引擎」。

Polaris 调度器是二者在 Python 侧的合流，策略如下：

1. 请求要求确定性（``require_deterministic=True``，默认）→ 只考虑
   声明了 ``deterministic: True`` 且 ``available()`` 为真的引擎；
2. 同类引擎中按优先级排序：world-class-external > builtin-flagship > mock；
3. 无任何引擎满足 → 抛出 ``NoEngineAvailable``（诚实失败，不静默造假）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..engines.base import EngineAdapter
from .registry import EngineRegistry


class NoEngineAvailable(RuntimeError):
    """没有任何引擎满足请求约束（诚实失败）。"""


#: 引擎类别优先级（世界顶级外部引擎优先，与 NEXUS Dispatch 一致）
_KIND_PRIORITY = {
    "world-class-external": 0,
    "builtin-flagship": 1,
    "mock": 2,
}


@dataclass
class DispatchRequest:
    """一次调度请求。

    Attributes:
        prompt: 输入提示。
        max_tokens: 最大生成 token 数。
        temperature: 采样温度（0 = 贪心确定性）。
        seed: 随机种子（G1 的前提）。
        require_deterministic: 是否要求确定性输出（默认 True）。
        preferred: 偏好引擎名（满足约束时优先，不满足则忽略）。
    """

    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    seed: int = 0
    require_deterministic: bool = True
    preferred: Optional[str] = None


@dataclass
class DispatchDecision:
    """调度决策（含理由，供审计与 ``--explain`` 消费）。"""

    adapter: EngineAdapter
    reason: str
    candidates: list[str]


class Scheduler:
    """约束调度器。"""

    def __init__(self, registry: EngineRegistry):
        self.registry = registry

    def decide(self, request: DispatchRequest) -> DispatchDecision:
        """按约束选出唯一引擎（确定性决策，同输入同决策）。"""
        candidates: list[str] = []
        for adapter in self.registry.all():
            caps = adapter.capabilities or {}
            if not adapter.available():
                continue
            if request.require_deterministic and not caps.get("deterministic", False):
                continue
            if request.temperature == 0.0 and not caps.get("deterministic", False):
                continue
            max_ctx = caps.get("maxTokens")
            if isinstance(max_ctx, int) and request.max_tokens > max_ctx:
                continue
            candidates.append(adapter.name)

        # 偏好优先（仅当偏好引擎本身是合格候选）
        if request.preferred and request.preferred in candidates:
            chosen = request.preferred
            reason = f"preferred engine {chosen!r} satisfies all constraints"
            return DispatchDecision(self.registry.get(chosen), reason, candidates)

        if not candidates:
            raise NoEngineAvailable(
                "no engine satisfies constraints: "
                f"deterministic={request.require_deterministic}, "
                f"max_tokens={request.max_tokens}; "
                f"registered={self.registry.names()}, "
                f"available={self.registry.available()}"
            )

        chosen = sorted(
            candidates,
            key=lambda n: (
                _KIND_PRIORITY.get(self.registry.get(n).kind, 99),
                n,
            ),
        )[0]
        kind = self.registry.get(chosen).kind
        reason = (
            f"constraint-satisfying engine {chosen!r} selected by kind priority "
            f"(kind={kind}, deterministic-first, external-world-class preferred)"
        )
        return DispatchDecision(self.registry.get(chosen), reason, candidates)


__all__ = [
    "NoEngineAvailable",
    "DispatchRequest",
    "DispatchDecision",
    "Scheduler",
]
