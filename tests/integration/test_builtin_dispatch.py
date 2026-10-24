"""集成测试：内置引擎真实 dispatch（现场训练微型模型，ifai 官方路径）。

本文件涉及真实训练（默认 40 步，CPU 秒级），验证：
1. 内置引擎端到端产出（degraded=False，真实模型生成）；
2. G1 确定性：两次 dispatch 结果指纹逐位一致；
3. 审计账本与门禁全链路联动。

跳过条件：设置 ``POLARIS_SKIP_SLOW=1`` 时跳过（CI 快速通道）。
"""

from __future__ import annotations

import os

import pytest

import polaris  # noqa: F401
from polaris import PolarisOS
from polaris.engines import BuiltinEngineAdapter

pytestmark = pytest.mark.skipif(
    os.environ.get("POLARIS_SKIP_SLOW") == "1",
    reason="POLARIS_SKIP_SLOW=1（快速通道）",
)


@pytest.fixture(scope="module")
def trained_os(tmp_path_factory):
    model_dir = str(tmp_path_factory.mktemp("polaris_model"))
    os_ = PolarisOS()
    os_.register_engine(BuiltinEngineAdapter(model_dir=model_dir, train_steps=40))
    return os_


class TestBuiltinDispatch:
    def test_end_to_end_green(self, trained_os):
        r = trained_os.dispatch("人工智能", max_tokens=16)
        assert r.engine == "ifai-builtin"
        assert r.degraded is False  # 真实模型产出，非占位
        d = r.meta["dispatch"]
        assert all(g["passed"] for g in d["gates"])
        assert r.text.strip()

    def test_determinism_bitwise(self, trained_os):
        a = trained_os.dispatch("注意力机制", max_tokens=16)
        b = trained_os.dispatch("注意力机制", max_tokens=16)
        assert a.fingerprint() == b.fingerprint()  # G1：逐位一致

    def test_audit_chain_after_dispatch(self, trained_os):
        trained_os.dispatch("检索", max_tokens=8)
        ok, detail = trained_os.audit_verify()
        assert ok, detail
        entry = trained_os.audit_tail(1)[0]
        assert entry["engine"] == "ifai-builtin"
        assert entry["degraded"] is False
