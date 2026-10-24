"""端到端演示 —— 一条意图走完 Polaris 全链路。

零重写 · 出处：演示链路与 dispatch 完全同一条代码路径（无演示专用
后门），所调引擎/ISA/门禁/审计均为 Polaris 生产路径的同一实现。

演示链路：

    意图 → 调度器选引擎 → 引擎生成 → G1/G2/G3 门禁 → 哈希链审计

用法::

    python -m polaris.demo            # 完整演示
    python -m polaris.demo --quick    # 跳过现场训练，仅演示引擎/ISA/审计
"""

from __future__ import annotations

import argparse
import sys

from . import PolarisOS
from .engines import BuiltinEngineAdapter, LlamaCppAdapter, MockEngine


def _hr(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def run_demo(quick: bool = False) -> int:
    os_ = PolarisOS()

    _hr("第 0 步 · 引擎登记（世界顶级外部引擎 + 内置旗舰 + 测试替身）")
    os_.register_engine(LlamaCppAdapter())
    os_.register_engine(BuiltinEngineAdapter(allow_autotrain=not quick))
    os_.register_engine(MockEngine())
    for e in os_.engines():
        mark = "OK" if e["available"] else "--"
        print(f"  [{mark}] {e['name']:<16} kind={e['kind']:<22} caps={e['capabilities']}")

    _hr("第 1 步 · 1024 微指令 ISA（复用 ifai L10，零重写）")
    stats = os_.isa_stats()
    print(f"  微指令总数: {stats['total_ops']}  家族数: {stats['families']}")
    print(f"  家族分类（诚实记账）: {stats['family_kinds']}")
    print(f"  家族列表: {', '.join(stats['family_names'][:8])} ...")

    _hr("第 2 步 · 单条微指令直执（smax.softmax）")
    out = os_.run_isa("smax.softmax", [0.1, 0.2, 0.7])
    print(f"  smax.softmax([0.1, 0.2, 0.7]) -> 行和 = {out.sum():.12f}")

    _hr("第 3 步 · 全量微指令门禁（冒烟 + 数学性质断言）")
    gates = os_.isa_gates()
    print(f"  总条数 {gates['total']} · 通过 {gates['ok']} · 失败 {len(gates['failed'])}")
    props = gates.get("properties", {})
    if isinstance(props, dict) and "checked" in props:
        print(
            f"  性质断言: {props.get('passed', '?')}/{props.get('checked', '?')} "
            f"(skipped={props.get('skipped', '?')})"
        )

    _hr("第 4 步 · 端到端 dispatch（调度 + 生成 + 门禁 + 审计）")
    for prompt in ("人工智能", "注意力机制"):
        r = os_.dispatch(prompt, max_tokens=24)
        d = r.meta.get("dispatch", {})
        print(f"  输入: {prompt!r}")
        print(f"  引擎: {r.engine} (degraded={r.degraded})")
        print(f"  调度理由: {d.get('reason', '')}")
        print(f"  门禁: {[g['gate'] + '=' + ('PASS' if g['passed'] else 'FAIL') for g in d.get('gates', [])]}")
        print(f"  输出: {r.text[:60]!r}")
        print(f"  耗时: {d.get('latency_ms', 0):.1f} ms")
        print()

    _hr("第 5 步 · 哈希链审计账本（防篡改验证）")
    ok, detail = os_.audit_verify()
    print(f"  账本校验: {'PASS' if ok else 'FAIL'} - {detail}")
    print(f"  末条记录: {os_.audit_tail(1)[0]['entry_hash'][:32]}...")

    _hr("演示完成 · 北极星 POLARIS - 零重写 · 集大成 · 闭环交付")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="polaris.demo", description="Polaris 端到端演示")
    ap.add_argument("--quick", action="store_true", help="跳过现场训练（快）")
    args = ap.parse_args(argv)
    return run_demo(quick=args.quick)


if __name__ == "__main__":
    sys.exit(main())
