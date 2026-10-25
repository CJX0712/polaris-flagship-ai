"""P8 引擎联邦门禁 —— 一份契约统治 N 个世界顶级引擎。

可证不变量：
- 预设库结构完整（键唯一、必填字段齐、能力声明含 deterministic）
- 未知预设 → KeyError（诚实失败，不回退到「随便一个引擎」）
- 离线时 available() 为 False（绝不谎报在线）
- llama.cpp 具名适配器与联邦预设指向同一套契约
"""

from __future__ import annotations

import pytest

from polaris.engines import (
    PRESETS,
    LlamaCppAdapter,
    OpenAICompatEngine,
    autodiscover,
    build_engine,
    preset_table,
)


class TestPresetLibrary:
    def test_presets_non_empty_and_unique(self):
        assert len(PRESETS) >= 8
        assert len(set(PRESETS)) == len(PRESETS)

    @pytest.mark.parametrize("key", sorted(PRESETS))
    def test_preset_fields_complete(self, key):
        p = PRESETS[key]
        assert p.key == key
        assert p.kind == "world-class-external"
        assert p.default_url.startswith("http")
        assert p.env.startswith("POLARIS_")
        assert p.health_paths, f"{key} 必须有健康端点"
        assert "deterministic" in p.capabilities
        assert p.cite.startswith("https://")

    def test_preset_table_rows_match_presets(self):
        rows = preset_table()
        assert {r["preset"] for r in rows} == set(PRESETS)
        for r in rows:
            assert r["health"] and "," in r["health"] or r["health"]

    def test_known_world_class_engines_present(self):
        for key in ("llama.cpp", "vllm", "sglang", "ollama", "tensorrt-llm"):
            assert key in PRESETS, f"{key} 必须在预设库里"


class TestOpenAICompatEngine:
    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            OpenAICompatEngine("definitely-not-an-engine")

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("POLARIS_VLLM_URL", "http://10.0.0.7:9000")
        e = build_engine("vllm")
        assert e.base_url == "http://10.0.0.7:9000"
        assert e.name == "vllm"

    def test_explicit_base_url_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("POLARIS_VLLM_URL", "http://10.0.0.7:9000")
        e = build_engine("vllm", base_url="http://127.0.0.1:1234/")
        assert e.base_url == "http://127.0.0.1:1234"

    def test_offline_engine_reports_unavailable(self):
        # 指向一个必然拒绝连接的端口（RFC 5737 文档保留地址 + 冷端口）
        e = OpenAICompatEngine("vllm", base_url="http://127.0.0.1:9", timeout=0.4)
        assert e.available() is False

    def test_describe_carries_citation_and_url(self):
        e = build_engine("sglang")
        d = e.describe()
        assert d["preset"] == "sglang"
        assert d["cite"] == PRESETS["sglang"].cite
        assert d["base_url"] == PRESETS["sglang"].default_url
        assert d["available"] in (True, False)

    def test_capabilities_copied_per_instance(self):
        a = build_engine("vllm")
        b = build_engine("vllm")
        a.capabilities["maxTokens"] = 1
        assert b.capabilities["maxTokens"] != 1  # 实例间不串味


class TestLlamaCppFederationIdentity:
    def test_llamacpp_is_a_federation_engine(self):
        a = LlamaCppAdapter()
        assert isinstance(a, OpenAICompatEngine)

    def test_llamacpp_defaults_match_preset(self):
        a = LlamaCppAdapter()
        assert a.name == "llama.cpp"
        assert a.base_url == PRESETS["llama.cpp"].default_url
        assert a.capabilities["quantized"] is True


class TestAutodiscover:
    def test_discover_offline_returns_empty(self):
        # 全部预设都指向未起服的本地端口 ⇒ 应当为空，绝不伪造
        found = autodiscover(presets=["vllm"], timeout=0.3)
        assert isinstance(found, list)
        assert all(isinstance(e, OpenAICompatEngine) for e in found)

    def test_discover_unknown_preset_skipped(self):
        found = autodiscover(presets=["nope"], timeout=0.2)
        assert found == []
