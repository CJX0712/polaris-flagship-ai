"""引擎契约层 —— Polaris 操作系统的「硬件抽象」。

零重写 · 出处
------------
契约设计参照三个世界顶级形态，不重写其中任何一个：

1. **llama.cpp server 的 OpenAI 兼容面**（ggerganov/llama.cpp, MIT）：
   外部世界级引擎统一以 ``POST {base}/v1/completions`` 接入，
   字段与 OpenAI Completions API 一致 —— 生态零改动的正解。
2. **OpenAI Python SDK 的极简范式**：``complete(prompt, max_tokens,
   temperature) -> 结构化结果``。
3. **NEXUS 旗舰操作系统**（CJX0712/nexus-flagship, MIT，作者晨星）的
   ``EngineAdapter`` 注册/调度契约 —— 本模块是其 Python 侧对应物。

Polaris 自身只定义契约与编排；「智力」永远来自被接入的世界顶级引擎。
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class EngineResult:
    """一次引擎调用的标准化产出。

    Attributes:
        text: 生成文本（经门禁清洗后的最终文本）。
        engine: 产出该结果的引擎名（必须已注册）。
        tokens: 生成的 token id 序列（可缺失，取决于引擎）。
        degraded: True 表示结果来自降级路径（如无外部引擎时的内置回退），
            **如实标记，不伪装成世界级引擎产出**。
        meta: 引擎附带的原始元数据（模型名、耗时、finish_reason…）。
    """

    text: str
    engine: str
    tokens: Optional[list[int]] = None
    degraded: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """结果的确定性指纹（用于 G1 门禁与审计账本）。"""
        payload = repr((self.text, self.engine, self.tokens)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


class EngineAdapter(ABC):
    """世界顶级引擎的统一适配契约。

    任何引擎 —— 本地 llama.cpp、远程 vLLM/SGLang、内置 ifai 全栈、
    测试 mock —— 只要实现本契约即可接入 Polaris，经同一套门禁治理。
    """

    #: 引擎名（注册表主键，全局唯一）
    name: str = ""
    #: 引擎类别：world-class-external（外部世界级）/ builtin-flagship（内置旗舰）/ mock（测试）
    kind: str = ""
    #: 能力声明（调度器据此匹配请求约束）
    capabilities: dict[str, Any] = {}

    @abstractmethod
    def available(self) -> bool:
        """引擎当前是否可用（发现式探测，不抛异常）。"""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> EngineResult:
        """执行一次补全，返回标准化结果。

        实现方必须保证：temperature=0 时输出确定可复现（G1 门禁的前提）。
        """

    def describe(self) -> dict[str, Any]:
        """引擎的自描述（doctor / engines 子命令消费）。"""
        return {
            "name": self.name,
            "kind": self.kind,
            "capabilities": dict(self.capabilities),
            "available": self.available(),
        }


def validate_result(result: EngineResult, registry_names: set[str]) -> list[str]:
    """契约级**结构**校验：返回违规清单（空 = 通过）。

    世界顶级工程的常识：信任但验证 —— 引擎产出先过契约校验，再过门禁。
    内容语义（NaN/Inf 字样泄漏）归 G2 门禁，本层只管结构完整性。
    """
    problems: list[str] = []
    if not isinstance(result, EngineResult):
        return ["result is not an EngineResult"]
    if not result.text:
        problems.append("empty text")
    if result.engine not in registry_names:
        problems.append(f"engine {result.engine!r} not registered")
    for ch in result.text:
        if ch in ("\ufffd",):
            problems.append("text contains replacement character")
            break
    if result.tokens is not None and len(result.tokens) > 1_000_000:
        problems.append("tokens implausibly long")
    return problems


def is_finite_number(x: float) -> bool:
    """有限数判据（G2 门禁的原子判据，math.isfinite 的世界标准语义）。"""
    return isinstance(x, (int, float)) and math.isfinite(x)


__all__ = [
    "EngineResult",
    "EngineAdapter",
    "validate_result",
    "is_finite_number",
]
