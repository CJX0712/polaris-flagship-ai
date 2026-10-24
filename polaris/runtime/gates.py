"""运行时门禁 —— Polaris 的「尺子」。

零重写 · 出处
------------
门禁思想继承自同门旗舰 **AXIOM**（axiom-ai，64 项数学不变量）与
**NEXUS**（nexus-flagship，G1 确定性 / G2 有限性 / G3 不变量三重门禁）。
判据全部取自公开数学标准，不发明新数学：

- **G1 确定性**：同 seed 同输入 → 输出指纹逐位一致（可复现性的定义）；
- **G2 有限性**：输出非空、长度受约束、无 NaN/Inf 字样泄漏（IEEE 754
  有限性判据的字符串侧防御）；
- **G3 契约**：``EngineResult`` 字段完整、引擎已注册（trust-but-verify）。

每条门禁返回 ``GateReport(gate, passed, detail)``，失败即拦截并记审计。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..engines.base import EngineResult, validate_result

#: NaN/Inf 数值字样判据（词边界，避免误伤 "Nanning" 等地名/普通词）
_NANINF_RE = re.compile(r"\b(nan|inf|-inf|\+inf)\b", re.IGNORECASE)


@dataclass
class GateReport:
    """单条门禁的判定结果。"""

    gate: str          # "G1" | "G2" | "G3"
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict:
        return {"gate": self.gate, "passed": self.passed, "detail": self.detail}


def gate_g1_determinism(first: EngineResult, replay: EngineResult) -> GateReport:
    """G1：同 seed 重放必须逐位一致（以结果指纹计）。"""
    ok = first.fingerprint() == replay.fingerprint()
    return GateReport(
        "G1",
        ok,
        "" if ok else f"fingerprint mismatch: {first.fingerprint()} vs {replay.fingerprint()}",
    )


def gate_g2_finiteness(result: EngineResult, max_tokens: int) -> GateReport:
    """G2：输出必须非空、不超长、无 NaN/Inf 字样。"""
    text = result.text
    if not text or not text.strip():
        return GateReport("G2", False, "empty output")
    if max_tokens > 0 and len(text) > max_tokens * 8:
        # 宽松上界：1 token 至多约 8 字符（中文/BPE 的保守估计）
        return GateReport("G2", False, f"output too long: {len(text)} > {max_tokens * 8}")
    m = _NANINF_RE.search(text)
    if m:
        return GateReport("G2", False, f"NaN/Inf literal leaked into text: {m.group(0)!r}")
    return GateReport("G2", True)


def gate_g3_contract(result: EngineResult, registry_names: set[str]) -> GateReport:
    """G3：契约校验（字段完整 + 引擎已注册 + 无泄漏字符）。"""
    problems = validate_result(result, registry_names)
    return GateReport("G3", not problems, "; ".join(problems))


def run_gates(
    result: EngineResult,
    *,
    registry_names: set[str],
    max_tokens: int = 64,
    replay: EngineResult | None = None,
) -> list[GateReport]:
    """执行全部门禁；``replay`` 提供时追加 G1 确定性判定。"""
    reports = [
        gate_g2_finiteness(result, max_tokens),
        gate_g3_contract(result, registry_names),
    ]
    if replay is not None:
        reports.append(gate_g1_determinism(result, replay))
    return reports


__all__ = [
    "GateReport",
    "gate_g1_determinism",
    "gate_g2_finiteness",
    "gate_g3_contract",
    "run_gates",
]
