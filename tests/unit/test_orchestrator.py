"""门面端到端测试：dispatch 全链路 + 门禁拦截 + 审计联动。"""

from __future__ import annotations

import pytest

from polaris import GateViolation, PolarisOS
from polaris.engines import EngineAdapter, EngineResult, MockEngine


class _BadEngine(EngineAdapter):
    """恶意/畸形引擎：产出空文本，必须被 G2 拦截。"""

    name = "bad"
    kind = "mock"
    capabilities = {"deterministic": True, "maxTokens": 1024}

    def available(self):
        return True

    def complete(self, prompt, *, max_tokens=64, temperature=0.0, seed=0):
        return EngineResult(text="   ", engine="bad", degraded=True)


class _NondeterministicEngine(EngineAdapter):
    """非确定性引擎：两次输出不同，必须被 G1 抓住。"""

    name = "flaky"
    kind = "mock"
    capabilities = {"deterministic": True, "maxTokens": 1024}

    def __init__(self):
        self._n = 0

    def available(self):
        return True

    def complete(self, prompt, *, max_tokens=64, temperature=0.0, seed=0):
        self._n += 1
        return EngineResult(text=f"out-{self._n}", engine="flaky")


class TestDispatchE2E:
    """dispatch = 调度 → 生成 → 门禁 → 审计，一条代码路径。"""

    def test_dispatch_mock_green(self, os_with_mock):
        r = os_with_mock.dispatch("你好", max_tokens=16)
        assert r.engine == "mock"
        d = r.meta["dispatch"]
        assert [g["gate"] for g in d["gates"]] == ["G2", "G3", "G1"]
        assert all(g["passed"] for g in d["gates"])
        assert d["latency_ms"] >= 0

    def test_dispatch_writes_audit(self, os_with_mock):
        os_with_mock.dispatch("第一条", max_tokens=8)
        os_with_mock.dispatch("第二条", max_tokens=8)
        assert len(os_with_mock.audit) == 2
        ok, detail = os_with_mock.audit_verify()
        assert ok, detail
        entry = os_with_mock.audit_tail(1)[0]
        assert entry["intent"] == "dispatch"
        assert entry["engine"] == "mock"
        assert entry["gates"]

    def test_gate_violation_blocks_bad_engine(self):
        os_ = PolarisOS()
        os_.register_engine(_BadEngine())
        with pytest.raises(GateViolation) as exc_info:
            os_.dispatch("hi", max_tokens=8)
        failed = [r.gate for r in exc_info.value.failed]
        assert "G2" in failed  # 空输出被拦截，绝不放行

    def test_g1_catches_nondeterministic_engine(self):
        os_ = PolarisOS()
        os_.register_engine(_NondeterministicEngine())
        # kind="mock" → G1 重放自动开启，两次输出指纹不一致必被抓住
        with pytest.raises(GateViolation) as exc_info:
            os_.dispatch("hi", max_tokens=8)
        failed = [r.gate for r in exc_info.value.failed]
        assert "G1" in failed

    def test_no_engine_honest_failure(self):
        os_ = PolarisOS()  # 空系统
        with pytest.raises(RuntimeError, match="no engine satisfies"):
            os_.dispatch("hi")

    def test_preferred_engine(self, os_full):
        r = os_full.dispatch("你好", max_tokens=8, preferred="mock")
        assert r.engine == "mock"
        assert "preferred" in r.meta["dispatch"]["reason"]

    def test_degraded_flag_honest(self, os_with_mock):
        r = os_with_mock.dispatch("hi", max_tokens=8)
        assert r.degraded is True  # mock 产出必须如实标记降级


class TestOSSmoke:
    """门面其余公开面。"""

    def test_engines_describe(self, os_full):
        descs = os_full.engines()
        names = {d["name"] for d in descs}
        assert {"llama.cpp", "ifai-builtin", "mock"} <= names

    def test_isa_stats_and_run(self, os_with_mock):
        assert os_with_mock.isa_stats()["total_ops"] == 1024
        import numpy as np

        out = os_with_mock.run_isa("smax.softmax", np.array([1.0, 1.0]))
        assert abs(float(out[0]) - 0.5) < 1e-12
