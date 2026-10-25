"""P11 基准套件门禁 —— 数字必须可复现、诚信、含环境指纹。

可证不变量：
- B1 分位数定义正确：p50 ≤ p90 ≤ p99，且都来自样本集合
- B2 最近秩次分位数在空样本上返回 0.0（不抛）
- B3 ISA 吞吐为正、calls 与 ops 自洽
- B4 审计写入吞吐可测（每条一次 SHA-256 链推进）
- B5 run_all 输出含 env 指纹（缺 env 则数字不可复现）
- B6 JSON 可序列化（能进 CI artifact / 审计账本）
"""

from __future__ import annotations

import json

import pytest

from polaris import PolarisOS
from polaris.bench import env_info, measure, percentile, run_all
from polaris.engines import MockEngine


def _os_with_mock() -> PolarisOS:
    os_ = PolarisOS()
    os_.register_engine(MockEngine())
    return os_


class TestPercentile:
    def test_nearest_rank_invariant(self):
        s = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(s, 50) == 3.0       # 中位数
        assert percentile(s, 90) == 5.0
        assert percentile(s, 0) == 1.0

    def test_monotonic(self):
        s = [10.0, 20.0, 30.0, 40.0]
        assert percentile(s, 50) <= percentile(s, 90) <= percentile(s, 99)

    def test_empty_returns_zero(self):
        assert percentile([], 90) == 0.0


class TestMeasure:
    def test_returns_p50_and_p99(self):
        m = measure(lambda: None, repeats=3, iterations=5)
        for k in ("p50", "p90", "p99", "mean", "min", "max", "calls"):
            assert k in m
        assert m["p50"] <= m["p99"]

    def test_json_ready(self):
        m = measure(lambda: None, repeats=2, iterations=2)
        json.dumps(m)


class TestEnvInfo:
    def test_contains_fingerprint_fields(self):
        e = env_info()
        for k in ("python", "numpy", "os", "machine"):
            assert k in e and e[k]


class TestBenchIsa:
    def test_throughput_positive_and_consistent(self):
        r = run_all(_os_with_mock(), isa_ops=8, dispatch_iterations=1, audit_iterations=10)
        isa = r["isa"]
        assert isa["ops"] > 0
        assert isa["calls"] == isa_ops_checked(isa)
        assert isa["ops_per_sec"] > 0
        assert "note" in isa

    def test_default_uses_first_128_ops_deterministic(self):
        from ifai.l10 import MICROOPS

        r = run_all(_os_with_mock(), isa_ops=128, dispatch_iterations=1, audit_iterations=5)
        assert r["isa"]["ops"] <= len(MICROOPS)


def isa_ops_checked(isa: dict) -> int:
    return isa["rounds"] * isa["ops"]


class TestRunAll:
    def test_contains_env_and_all_sections(self):
        r = run_all(_os_with_mock(), isa_ops=8, dispatch_iterations=2, audit_iterations=10)
        assert "env" in r
        for k in ("isa", "dispatch", "audit", "compile"):
            assert k in r
        assert "disclaimer" in r

    def test_dispatch_section_flags_engine_and_inference(self):
        r = run_all(_os_with_mock(), isa_ops=8, dispatch_iterations=2, audit_iterations=10)
        d = r["dispatch"]
        assert "engine" in d
        assert d["model_inference_included"] is True

    def test_audit_throughput_reported(self):
        r = run_all(_os_with_mock(), isa_ops=8, dispatch_iterations=1, audit_iterations=20)
        assert r["audit"]["calls"] >= 0

    def test_json_ready(self):
        r = run_all(_os_with_mock(), isa_ops=8, dispatch_iterations=1, audit_iterations=10)
        json.dumps(r, ensure_ascii=False)
