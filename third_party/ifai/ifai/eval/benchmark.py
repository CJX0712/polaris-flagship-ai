"""L8 吞吐 / 延迟基准 (Benchmark)

零重写: 复用机器学习系统评估惯例 (如 MLPerf 的吞吐/延迟基准方法论) ——
固定 prompt 与步数, 多次运行统计均值/变异系数 (CV), 作为回归基线与
性能门禁。

可验证 (R-L8-02): 3 次运行变异系数 CV ≤ 15%, 结果写入 BASELINE.md。
"""
from __future__ import annotations

import numpy as np
import time


def benchmark_generation(generate_fn, prompt: str, max_new_tokens: int = 32,
                         runs: int = 3) -> dict:
    """generate_fn(prompt, n) -> 生成的 token 数; 返回吞吐统计。"""
    times, toks = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        n = generate_fn(prompt, max_new_tokens)
        dt = time.perf_counter() - t0
        times.append(dt)
        toks.append(n)
    tps = [t / max(dt, 1e-6) for t, dt in zip(toks, times)]
    arr = np.array(tps)
    cv = float(arr.std() / arr.mean()) if arr.mean() else 0.0
    return {"runs": runs, "tok_per_s_mean": float(arr.mean()),
            "tok_per_s_min": float(arr.min()), "cv": cv,
            "max_new_tokens": max_new_tokens}


def write_baseline(path: str, stats: dict) -> None:
    lines = ["# 性能基线 (BASELINE)", "",
             f"- tok/s mean: {stats['tok_per_s_mean']:.2f}",
             f"- tok/s min:  {stats['tok_per_s_min']:.2f}",
             f"- CV: {stats['cv']*100:.1f}%",
             f"- max_new_tokens: {stats['max_new_tokens']}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
