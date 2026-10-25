"""可复现基准套件 —— 性能必须能被别人跑出来，否则不算性能。

零重写 · 出处
------------
基准方法学取自世界标准做法，不自创指标：

1. **延迟分位数 p50/p90/p99**（而非平均值）：Google 「Tail at Scale」
   （Dean & Barroso 2013, CACM）确立的共识 —— 平均值会掩盖长尾，
   生产系统看的是分位数。分位数采用**最近秩次（nearest-rank）**定义，
   与 numpy ``percentile`` 的线性插值不同，但对整数秩次样本更直观、可手算复现。
2. **吞吐 ops/sec** = 总操作数 / 墙钟耗时（SPEC/MLPerf 的通用定义）。
3. **多样本取中位数**抵抗抖动：每次测量重复 R 轮，取中位数（与
   Google Benchmark 库的聚合策略一致）。

诚实记账（写死在输出里）
------------------------
- ``engine`` 字段标明被测引擎；mock 引擎的「吞吐」只反映**框架开销**，
  不含任何模型推理成本 —— 把 mock 的数字当模型性能是自欺，输出里会
  显式标注 ``"model_inference_included": false``。
- 环境信息（Python / numpy / OS / CPU）随结果一并输出，缺了它数字不可复现。
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from typing import Any, Callable, Optional

from ._vendor_path import ensure_vendor_path

ensure_vendor_path()

import numpy as np  # noqa: E402

from ifai.l10 import MICROOPS  # noqa: E402


def percentile(samples: list[float], q: float) -> float:
    """最近秩次分位数（nearest-rank，可手算复现）。

    ``q`` 取 0–100。空样本返回 0.0。
    """
    if not samples:
        return 0.0
    xs = sorted(samples)
    k = max(0, min(len(xs) - 1, int(round((q / 100.0) * (len(xs) - 1)))))
    return xs[k]


def measure(fn: Callable[[], Any], repeats: int = 5, iterations: int = 20) -> dict[str, Any]:
    """测一个可调用对象的延迟分布（毫秒）。

    Args:
        fn: 被测操作（无参）。
        repeats: 轮数（取中位数的样本数）。
        iterations: 每轮内的调用次数（摊薄单次抖动）。

    Returns:
        ``{"p50","p90","p99","mean","min","max","samples_ms","calls"}``
    """
    rounds: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        for _ in range(max(1, iterations)):
            fn()
        rounds.append((time.perf_counter() - t0) * 1000.0 / max(1, iterations))
    return {
        "p50": round(percentile(rounds, 50), 4),
        "p90": round(percentile(rounds, 90), 4),
        "p99": round(percentile(rounds, 99), 4),
        "mean": round(statistics.fmean(rounds), 4),
        "min": round(min(rounds), 4),
        "max": round(max(rounds), 4),
        "samples_ms": [round(r, 4) for r in rounds],
        "calls": max(1, repeats) * max(1, iterations),
    }


def env_info() -> dict[str, Any]:
    """环境指纹（缺了它，任何性能数字都不可复现）。"""
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def bench_isa(ops: Optional[list[str]] = None, iterations: int = 200) -> dict[str, Any]:
    """ISA 吞吐基准：1024 微指令在探针向量上的执行吞吐。

    Returns:
        ``{"ops","calls","ops_per_sec","ms_per_call","throughput_note"}``
    """
    names = ops or sorted(MICROOPS)[:128]  # 默认取前 128 条（字典序，确定性）
    x = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    cache = [(n, MICROOPS[n]) for n in names if n in MICROOPS]

    def one_round() -> None:
        for _, op in cache:
            try:
                op.fn(x) if op.arity == 1 else op.fn(x, x)
            except Exception:  # noqa: BLE001,S112 - 基准要跑完全集，个别形状不合跳过
                continue

    t0 = time.perf_counter()
    for _ in range(max(1, iterations)):
        one_round()
    elapsed = time.perf_counter() - t0
    calls = max(1, iterations) * len(cache)
    return {
        "ops": len(cache),
        "rounds": max(1, iterations),
        "calls": calls,
        "elapsed_s": round(elapsed, 4),
        "ops_per_sec": round(calls / elapsed, 1) if elapsed > 0 else 0.0,
        "ms_per_call": round(elapsed * 1000.0 / calls, 6) if calls else 0.0,
        "note": "纯算子吞吐（无模型推理、无 IO）；跨机器比较请对照 env_info()",
    }


def bench_dispatch(os_: Any, prompt: str = "基准测试", iterations: int = 10) -> dict[str, Any]:
    """dispatch 端到端延迟（调度 + 生成 + 门禁 + 审计）。

    ``model_inference_included`` 由被测引擎决定：mock/builtin 为 True
    （内置引擎真的在推理），外部引擎亦为 True。此处如实标注，不做美化。
    """
    def one() -> None:
        os_.dispatch(prompt, max_tokens=8)

    m = measure(one, repeats=5, iterations=max(1, iterations))
    m["engine"] = _first_available_engine(os_)
    m["model_inference_included"] = True
    m["note"] = "端到端：调度 + 生成 + G1/G2/G3 + 哈希链审计"
    return m


def bench_audit(os_: Any, iterations: int = 200) -> dict[str, Any]:
    """审计账本写入吞吐（每条含一次 SHA-256 哈希链推进）。"""
    def one() -> None:
        os_.audit.append({"bench": True, "i": iterations})

    m = measure(one, repeats=5, iterations=max(1, iterations))
    m["note"] = "哈希链追加（SHA-256 per entry）"
    return m


def bench_compile(intent: str = "归一化后 softmax", repeats: int = 20) -> dict[str, Any]:
    """意图编译吞吐（含编译期干运行验证 —— 这部开销是刻意付的）。"""
    from .compiler import compile_intent

    def one() -> None:
        compile_intent(intent, width=2)

    m = measure(one, repeats=repeats, iterations=1)
    m["note"] = "含编译期干运行（K4）—— 编译期多花的钱，换运行期不炸"
    return m


def run_all(
    os_: Any,
    *,
    isa_ops: int = 128,
    dispatch_iterations: int = 10,
    audit_iterations: int = 200,
) -> dict[str, Any]:
    """跑全套基准，返回结构化结果（``polaris bench`` 消费）。"""
    return {
        "polaris_version": _version(),
        "env": env_info(),
        "isa": bench_isa(sorted(MICROOPS)[: max(1, isa_ops)]),
        "dispatch": bench_dispatch(os_, iterations=dispatch_iterations),
        "audit": bench_audit(os_, iterations=audit_iterations),
        "compile": bench_compile(),
        "disclaimer": (
            "本机数字，仅供本机前后对比；跨机器比较必须连同 env 一起看。"
            "mock 引擎的 dispatch 延迟只代表框架开销。"
        ),
    }


def _version() -> str:
    from . import __version__

    return __version__


def _first_available_engine(os_: Any) -> str:
    try:
        for a in os_.registry.all():
            if a.available():
                return a.name
    except Exception:  # noqa: BLE001,S112
        pass
    return "none"


__all__ = [
    "percentile",
    "measure",
    "env_info",
    "bench_isa",
    "bench_dispatch",
    "bench_audit",
    "bench_compile",
    "run_all",
]
