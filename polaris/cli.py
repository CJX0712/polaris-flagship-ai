"""Polaris 命令行 —— 操作系统的 shell。

子命令::

    polaris doctor                     环境自检（numpy / ifai / 引擎 / ISA）
    polaris engines                    列出全部引擎与可用性
    polaris dispatch "提示"            端到端：调度+门禁+审计
    polaris isa                        ISA 统计 + 全量微指令门禁
    polaris isa --family smax          列出家族成员
    polaris isa --run smax.softmax     执行单条微指令
    polaris train-tiny                现场训练内置微型模型（ifai 官方路径）
    polaris demo                      端到端演示
    polaris audit --verify             审计账本校验

零重写：训练走 ifai 官方 ``cmd_train``；ISA 走 ifai L10 注册表。
"""

from __future__ import annotations

import argparse
import json
import sys

from . import PolarisOS, __version__
from .engines import BuiltinEngineAdapter, LlamaCppAdapter, MockEngine


def _build_os(with_autotrain: bool = False) -> PolarisOS:
    os_ = PolarisOS()
    os_.register_engine(LlamaCppAdapter())
    os_.register_engine(BuiltinEngineAdapter(allow_autotrain=with_autotrain))
    os_.register_engine(MockEngine())
    return os_


def cmd_doctor(_args) -> int:
    print("北极星 POLARIS · 环境自检")
    print(f"  polaris 版本 : {__version__}")
    try:
        import numpy as np

        print(f"  numpy       : {np.__version__}")
    except ImportError:
        print("  numpy       : 缺失 (pip install numpy)")
        return 1
    try:
        import ifai

        print(f"  ifai(vendor): {ifai.__version__} (third_party/ifai)")
    except ImportError:
        print("  ifai(vendor): 不可导入 - third_party/ifai 缺损")
        return 1
    os_ = _build_os()
    for e in os_.engines():
        mark = "OK" if e["available"] else "--"
        print(f"  引擎[{mark}]    : {e['name']} ({e['kind']})")
    stats = os_.isa_stats()
    print(f"  ISA         : {stats['total_ops']} 微指令 / {stats['families']} 家族")
    print("  结论: 环境就绪（无 llama.cpp 时系统经内置引擎降级运行，功能不减、语义如实标记）")
    return 0


def cmd_engines(_args) -> int:
    os_ = _build_os()
    print(json.dumps(os_.engines(), ensure_ascii=False, indent=2))
    return 0


def cmd_dispatch(args) -> int:
    os_ = _build_os(with_autotrain=True)
    try:
        r = os_.dispatch(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
    except Exception as exc:  # NoEngineAvailable / GateViolation 如实呈现
        print(f"dispatch 失败（诚实失败）: {exc}", file=sys.stderr)
        return 2
    d = r.meta.get("dispatch", {})
    print(f"engine : {r.engine} (degraded={r.degraded})")
    print(f"reason : {d.get('reason', '')}")
    print(f"gates  : {[g['gate'] + '=' + ('PASS' if g['passed'] else 'FAIL') for g in d.get('gates', [])]}")
    print(f"latency: {d.get('latency_ms', 0):.1f} ms")
    print(f"output : {r.text}")
    return 0


def cmd_isa(args) -> int:
    os_ = _build_os()
    if args.family:
        from ifai.l10 import FAMILIES

        members = FAMILIES.get(args.family)
        if members is None:
            print(f"未知家族 {args.family!r}; 可用: {', '.join(sorted(FAMILIES))}", file=sys.stderr)
            return 2
        print(f"家族 {args.family}: {len(members)} 条")
        for m in members:
            print(f"  {m}")
        return 0
    if args.run:
        import numpy as np

        op = args.run
        try:
            if op.startswith(("smax.", "distp.", "enty.")):
                x = np.array([0.1, 0.2, 0.7])
                out = os_.run_isa(op, x)
            else:
                out = os_.run_isa(op, np.eye(3))
        except KeyError as exc:
            print(f"未知微指令: {exc}", file=sys.stderr)
            return 2
        print(f"{op} ->")
        print(out)
        return 0
    stats = os_.isa_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n运行全量微指令门禁（冒烟 + 数学性质断言）...")
    gates = os_.isa_gates()
    props = gates.get("properties", {})
    print(
        f"结果: total={gates['total']} ok={gates['ok']} "
        f"failed={len(gates['failed'])} "
        f"properties={props.get('passed', '?')}/{props.get('checked', '?')} "
        f"skipped={props.get('skipped', '?')}"
    )
    return 0 if gates["ok"] == gates["total"] and not gates["failed"] else 3


def cmd_train_tiny(args) -> int:
    import argparse as ap

    from ifai.cli import cmd_train

    cmd_train(
        ap.Namespace(
            seed=args.seed, corpus=None, vocab=512, d_model=128,
            layers=3, heads=4, kv_heads=2, ffn_mult=4,
            seq_len=128, steps=args.steps, lr=3e-3, out=args.out,
        )
    )
    return 0


def cmd_demo(args) -> int:
    from .demo import run_demo

    return run_demo(quick=args.quick)


def cmd_audit(args) -> int:
    os_ = _build_os()
    if not len(os_.audit):
        print("审计账本为空（本次会话尚无 dispatch）")
        return 0
    if args.verify:
        ok, detail = os_.audit_verify()
        print(f"{'PASS' if ok else 'FAIL'}: {detail}")
        return 0 if ok else 4
    for entry in os_.audit_tail(args.tail):
        print(json.dumps(entry, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="polaris",
        description="北极星 POLARIS - 晨星旗舰 AI 操作系统（llama.cpp x 1024 微指令 ISA x 可证门禁）",
    )
    ap.add_argument("--version", action="version", version=f"polaris {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="环境自检")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("engines", help="列出引擎")
    p.set_defaults(fn=cmd_engines)

    p = sub.add_parser("dispatch", help="端到端推理")
    p.add_argument("prompt")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_dispatch)

    p = sub.add_parser("isa", help="1024 微指令 ISA")
    p.add_argument("--family", default=None)
    p.add_argument("--run", default=None)
    p.set_defaults(fn=cmd_isa)

    p = sub.add_parser("train-tiny", help="现场训练内置微型模型")
    p.add_argument("--out", default="./.polaris_model")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_train_tiny)

    p = sub.add_parser("demo", help="端到端演示")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("audit", help="审计账本")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--tail", type=int, default=10)
    p.set_defaults(fn=cmd_audit)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
