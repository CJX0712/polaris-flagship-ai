"""操作系统层测试：注册表 / 调度 / 门禁 / 审计（P2-P5）。"""

from __future__ import annotations

import copy

import pytest

from polaris.engines import BuiltinEngineAdapter, EngineResult, MockEngine
from polaris.runtime import (
    AuditLog,
    EngineRegistry,
    GateReport,
    NoEngineAvailable,
    Scheduler,
    gate_g1_determinism,
    gate_g2_finiteness,
    gate_g3_contract,
)


class TestP2Registry:
    """P2：注册表是唯一的「名字 → 适配器」真相源。"""

    def test_register_get_names(self):
        reg = EngineRegistry()
        reg.register(MockEngine())
        assert reg.names() == ["mock"]
        assert reg.get("mock") is not None
        assert reg.get("ghost") is None
        assert "mock" in reg
        assert len(reg) == 1

    def test_available_reflects_probe(self):
        from polaris.engines import LlamaCppAdapter

        reg = EngineRegistry()
        reg.register(MockEngine())
        reg.register(LlamaCppAdapter(base_url="http://127.0.0.1:59999", timeout=1.0))
        assert reg.available() == ["mock"]  # llama.cpp 不在线，不伪装

    def test_register_rejects_empty_name(self):
        reg = EngineRegistry()

        class Nameless(MockEngine):
            name = ""

        with pytest.raises(ValueError):
            reg.register(Nameless())


class TestP3Scheduler:
    """P3：约束调度 —— 确定性优先、外部世界级优先、偏好生效、诚实失败。"""

    def _sched(self, preferred=None):
        reg = EngineRegistry()
        reg.register(MockEngine())
        reg.register(BuiltinEngineAdapter(allow_autotrain=False))
        return Scheduler(reg)

    def test_builtin_beats_mock(self):
        from polaris.runtime import DispatchRequest

        s = self._sched()
        d = s.decide(DispatchRequest(prompt="hi"))
        assert d.adapter.name == "ifai-builtin"  # kind 优先级：builtin > mock
        assert d.candidates  # 候选集如实记录

    def test_preferred_honored(self):
        from polaris.runtime import DispatchRequest

        s = self._sched()
        d = s.decide(DispatchRequest(prompt="hi", preferred="mock"))
        assert d.adapter.name == "mock"
        assert "preferred" in d.reason

    def test_no_engine_raises_honestly(self):
        from polaris.runtime import DispatchRequest

        reg = EngineRegistry()  # 空注册表
        s = Scheduler(reg)
        with pytest.raises(NoEngineAvailable):
            s.decide(DispatchRequest(prompt="hi"))

    def test_max_tokens_constraint(self):
        from polaris.runtime import DispatchRequest

        reg = EngineRegistry()
        reg.register(MockEngine())  # mock maxTokens=1024
        s = Scheduler(reg)
        with pytest.raises(NoEngineAvailable):
            s.decide(DispatchRequest(prompt="hi", max_tokens=99999))


class TestP4Gates:
    """P4：门禁三判据 —— G1 确定性 / G2 有限性 / G3 契约。"""

    def test_g1_mismatch_caught(self):
        a = EngineResult(text="x", engine="mock")
        b = EngineResult(text="y", engine="mock")
        assert gate_g1_determinism(a, b).passed is False

    def test_g1_match(self):
        a = EngineResult(text="x", engine="mock")
        assert gate_g1_determinism(a, a).passed is True

    def test_g2_empty(self):
        r = EngineResult(text="   ", engine="mock")
        assert gate_g2_finiteness(r, 64).passed is False

    def test_g2_too_long(self):
        r = EngineResult(text="x" * 10_000, engine="mock")
        assert gate_g2_finiteness(r, 8).passed is False

    def test_g2_nan_leak(self):
        r = EngineResult(text="the loss is nan now", engine="mock")
        assert gate_g2_finiteness(r, 64).passed is False

    def test_g3_unregistered(self):
        r = EngineResult(text="ok", engine="ghost")
        rep = gate_g3_contract(r, {"mock"})
        assert rep.passed is False

    def test_gate_report_dict(self):
        rep = GateReport("G1", True, "")
        assert rep.as_dict() == {"gate": "G1", "passed": True, "detail": ""}


class TestP5AuditChain:
    """P5：哈希链账本 —— 追加-only、篡改即断、可重放。"""

    def test_append_and_verify(self):
        log = AuditLog()
        h = log.append({"intent": "dispatch", "engine": "mock"})
        log.append({"intent": "isa", "engine": "-"})
        ok, detail = log.verify_chain()
        assert ok, detail
        assert len(h) == 64

    def test_tamper_breaks_chain(self):
        log = AuditLog()
        log.append({"intent": "dispatch", "engine": "mock", "v": 1})
        log.append({"intent": "dispatch", "engine": "mock", "v": 2})
        # 篡改历史条目（攻击者视角）
        log._entries[0]["v"] = 999
        ok, detail = log.verify_chain()
        assert ok is False
        assert "hash mismatch" in detail

    def test_seq_gap_breaks_chain(self):
        log = AuditLog()
        log.append({"intent": "a"})
        log.append({"intent": "b"})
        log._entries[1]["seq"] = 99  # 序号断裂
        ok, _ = log.verify_chain()
        assert ok is False

    def test_replay_payload_clean(self):
        log = AuditLog()
        log.append({"intent": "dispatch", "engine": "mock"})
        replay = log.replay()
        assert len(replay) == 1
        assert "entry_hash" not in replay[0]
        assert "prev_hash" not in replay[0]

    def test_persist_and_reload(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path=str(path))
        log.append({"intent": "dispatch", "engine": "mock"})
        log.append({"intent": "isa", "engine": "-"})
        log2 = AuditLog(path=str(path))
        ok, detail = log2.verify_chain()
        assert ok, detail
        assert len(log2) == 2
