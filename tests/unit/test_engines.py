"""引擎契约测试（Polaris 门禁 P1：契约 fuzz + 确定性 + 诚实降级）。"""

from __future__ import annotations

import pytest

from polaris.engines import (
    BuiltinEngineAdapter,
    EngineResult,
    LlamaCppAdapter,
    MockEngine,
    validate_result,
)

# 畸形输入集：空串、超长、unicode、emoji、控制符、NaN 字样、注入尝试
FUZZ_PROMPTS = [
    "",
    " ",
    "x" * 4096,
    "你好，世界",
    "🌌✨🜲",
    "line1\nline2\ttab",
    "NaN inf -inf",
    "'); DROP TABLE users; --",
    "\x00\x01\x02",
    "混合 mixed 123 !@#$%^&*()",
]


class TestP1ContractFuzz:
    """P1：任何引擎对畸形输入不得抛异常，且产出过契约校验。"""

    @pytest.mark.parametrize("prompt", FUZZ_PROMPTS)
    def test_mock_engine_fuzz(self, prompt):
        eng = MockEngine()
        r = eng.complete(prompt, max_tokens=64)
        assert isinstance(r, EngineResult)
        assert r.engine == "mock"
        assert r.degraded is True  # mock 必须如实标记降级
        problems = validate_result(r, {"mock"})
        assert problems == [], f"contract violations for {prompt!r}: {problems}"

    @pytest.mark.parametrize("prompt", FUZZ_PROMPTS)
    def test_builtin_no_model_no_crash(self, prompt):
        """无模型且禁止训练时，内置引擎返回诚实占位而非崩溃。"""
        eng = BuiltinEngineAdapter(model_dir=None, allow_autotrain=False)
        r = eng.complete(prompt, max_tokens=64)
        assert r.degraded is True
        assert r.text  # 非空（G2 的前提）


class TestDeterminism:
    """确定性：同输入两次调用指纹一致（G1 的引擎侧前提）。"""

    def test_mock_deterministic(self):
        eng = MockEngine()
        a = eng.complete("确定性问题", max_tokens=32)
        b = eng.complete("确定性问题", max_tokens=32)
        assert a.fingerprint() == b.fingerprint()

    def test_mock_truncates_to_max_tokens(self):
        eng = MockEngine()
        r = eng.complete("abcdefghij", max_tokens=4)
        assert len(r.text) <= 4


class TestLlamaCppDiscovery:
    """llama.cpp 适配器的发现式语义（无 server 时必须如实不可用）。"""

    def test_unavailable_without_server(self):
        a = LlamaCppAdapter(base_url="http://127.0.0.1:59999", timeout=1.0)
        assert a.available() is False  # 不抛异常、不伪装可用

    def test_complete_without_server_raises(self):
        a = LlamaCppAdapter(base_url="http://127.0.0.1:59999", timeout=1.0)
        with pytest.raises(Exception):
            a.complete("hi", max_tokens=4)

    def test_describe_includes_base_url(self):
        a = LlamaCppAdapter(base_url="http://127.0.0.1:8080")
        d = a.describe()
        assert d["name"] == "llama.cpp"
        assert d["kind"] == "world-class-external"
        assert d["base_url"] == "http://127.0.0.1:8080"


class TestValidateResult:
    """契约校验器必须抓住三类造假。"""

    def test_catches_empty_text(self):
        r = EngineResult(text="", engine="mock")
        assert "empty text" in validate_result(r, {"mock"})

    def test_catches_unregistered_engine(self):
        r = EngineResult(text="ok", engine="ghost")
        assert any("not registered" in p for p in validate_result(r, {"mock"}))

    def test_passes_clean_result(self):
        r = EngineResult(text="正常输出", engine="mock")
        assert validate_result(r, {"mock"}) == []

    def test_nan_literal_is_g2_duty_not_contract(self):
        """NaN 字样属内容语义，由 G2 门禁拦截（见 test_runtime.test_g2_nan_leak）。"""
        r = EngineResult(text="result is nan", engine="mock")
        assert validate_result(r, {"mock"}) == []  # 契约层不误伤
