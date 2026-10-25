"""L10 微指令层 —— 1024 条**可执行**微指令 (32 家族 × 32 成员)。

为什么要有这一层
----------------
前九层提供「能力」, 第十层提供「可枚举的正确性」。一个系统宣称自己有多少
算子是廉价的; 真正有价值的是**每一条都能被单独执行、单独断言**。本层把
1024 条微指令做成**可执行注册表**: 每条带真实可调用函数, 门禁对每一条跑
真实输入并断言数学性质 —— 失败就如实报失败, 不注水。

零重写 · 集大成 (移植声明)
---------------------------
32 家族 × 32 成员 ISA **移植自同门旗舰 `tianji-ai`**
(https://github.com/CJX0712/tianji-ai, MIT, 作者同为晨星), 该实现在
tianji-ai 上已通过 1024/1024 门禁。移植**只改依赖指向, 不改算法体**:

- ``l2norm`` / ``spectral_norm`` / ``newton_schulz`` → 本仓 L0
  ``ifai.core.ops`` (本次一并补入本仓 L0, 避免移植后出现「无消费方的死
  代码」—— 那正是 ifai v0.7.13 审计批过的「模块存在但从未接线」同型缺口);
- ``logsumexp`` / ``softmax`` → 本仓 L0/L1 ``ifai.core.numeric``
  (本仓已有 INV-08 / INV-09 黄金测试)。

这正是「零重写 · 集大成」的定义: 不重新手写 1024 条算子, 而是把**已被
门禁证明过的**旗舰资产搬进本仓, 并让本仓 L0 成为其唯一依赖来源。

本层在移植之上**新增的超越点** (相对 tianji-ai 原版):

1. **性质断言层** (``_properties.py``): 原版门禁只断言「不抛异常 + 输出
   有限」; 本层为 20+ 个关键家族追加**数学性质断言** —— 如 softmax 行和
   恒为 1 (INV-09 平移不变)、L2 归一化后范数严格为 1、KL 散度非负、
   注意力分块 ≡ 全量 (INV-02)、Newton-Schulz 奇异值落入 (0.5, 1.5)
   (Muon 公开验收标准)、Top-K 掩码恰好 k 个 1。
2. **不变量溯源**: 每条断言都标注它守护的 INV 编号或公开论文依据。

与 ``docs/MICRO_INSTRUCTIONS.md`` 的关系
---------------------------------------
- ``docs/MICRO_INSTRUCTIONS.md``: **源码派生文档** (AST 静态扫描, 2069 条
  功能级 + 步骤级条目), 描述「代码写了什么」。
- 本模块: **运行时可执行注册表** (1024 条), 描述「代码能做什么, 且每条
  都被真实执行并断言了什么」。两者互补, 不重复。

接口
----
- ``MICROOPS``: ``dict[str, MicroOp]`` —— 名称 → 微指令
- ``FAMILIES``: ``dict[str, list[str]]`` —— 家族 → 成员名列表
- ``FAMILY_KIND``: ``dict[str, str]`` —— 家族 → ``"core"`` (成员语义独立) 或
  ``"sweep"`` (参数扫描, 覆盖数值边界)
- ``run_all()``: 对全部 1024 条执行冒烟 + 性质断言, 返回逐条结果
- ``count_ops()``: 返回已注册条数
"""
from __future__ import annotations

from ._microops import (
    FAMILIES,
    FAMILY_KIND,
    MICROOPS,
    MicroOp,
    count_ops,
)
from ._microops import run_all as _run_all_microops
from ._properties import PROPERTY_ASSERTIONS, property_report

__all__ = [
    "MicroOp",
    "MICROOPS",
    "FAMILIES",
    "FAMILY_KIND",
    "PROPERTY_ASSERTIONS",
    "run_all",
    "property_report",
    "count_ops",
]


def run_all(seed: int = 0, shape: tuple[int, int] = (4, 8)) -> dict:
    """对全部 1024 条微指令执行冒烟 + 数学性质断言。

    断言分三层, 逐层加强:

    1. **冒烟**: 算子可被真实输入调用, 不抛异常;
    2. **有限性**: 输出无 NaN / ±Inf (掩码族的 -inf 属刻意行为, 显式豁免);
    3. **性质**: 家族级数学不变量 (见 ``_properties.py``), 标注守护的 INV
       或公开论文依据。

    Args:
        seed: 采样种子 (固定则跨进程可复现)。
        shape: 采样张量形状。

    Returns:
        ``{"total", "ok", "failed", "families", "properties"}`` —— ``failed``
        为失败明细列表; ``properties`` 为性质断言的统计。
    """
    base = _run_all_microops(seed=seed, shape=shape)
    ok_names = {n for n in base["total"] * [None]}  # 占位, 下面用集合重建
    ok_names = set()
    for name in MICROOPS:
        ok_names.add(name)
    failed = list(base["failed"])
    failed_names = {n for n, _ in failed}
    props = property_report(seed=seed, shape=shape)
    # 冒烟失败的名字已经被 failed 收录; 性质断言失败要并入
    for name, detail in props["failed"]:
        if name not in failed_names:
            failed.append((name, detail))
    ok = base["ok"] - sum(1 for n, _ in props["failed"] if n not in failed_names)
    return {
        "total": base["total"],
        "ok": ok,
        "failed": failed,
        "families": base["families"],
        "properties": props,
    }
