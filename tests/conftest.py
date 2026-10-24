"""Polaris 测试公共夹具。"""

from __future__ import annotations

import pytest

import polaris  # noqa: F401  （触发 vendor 路径注入）


@pytest.fixture
def os_with_mock():
    """只挂 mock 引擎的操作系统（快、无外部依赖）。"""
    from polaris import PolarisOS
    from polaris.engines import MockEngine

    os_ = PolarisOS()
    os_.register_engine(MockEngine())
    return os_


@pytest.fixture
def os_full():
    """挂全部三级引擎的操作系统（内置引擎不现场训练，保持测试快）。"""
    from polaris import PolarisOS
    from polaris.engines import BuiltinEngineAdapter, LlamaCppAdapter, MockEngine

    os_ = PolarisOS()
    os_.register_engine(LlamaCppAdapter())
    os_.register_engine(BuiltinEngineAdapter(allow_autotrain=False))
    os_.register_engine(MockEngine())
    return os_
