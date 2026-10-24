"""ISA 复用完整性测试（P6：1024 条微指令经 Polaris 全部可调用、全部门禁可复现）。"""

from __future__ import annotations

import numpy as np
import pytest

import polaris  # noqa: F401
from polaris import PolarisOS
from polaris.engines import MockEngine


@pytest.fixture(scope="module")
def os_():
    o = PolarisOS()
    o.register_engine(MockEngine())
    return o


class TestISAStats:
    """ISA 统计必须诚实：恰好 1024 条、32 家族、core/sweep 分开记账。"""

    def test_exactly_1024(self, os_):
        stats = os_.isa_stats()
        assert stats["total_ops"] == 1024

    def test_32_families(self, os_):
        stats = os_.isa_stats()
        assert stats["families"] == 32

    def test_core_sweep_honest_accounting(self, os_):
        stats = os_.isa_stats()
        assert stats["family_kinds"] == {"core": 15, "sweep": 17}

    def test_family_names_sorted(self, os_):
        stats = os_.isa_stats()
        assert stats["family_names"] == sorted(stats["family_names"])


class TestISARun:
    """单条微指令经 Polaris 门面直执。"""

    def test_run_softmax_sums_to_one(self, os_):
        out = os_.run_isa("smax.softmax", np.array([0.1, 0.2, 0.7]))
        assert abs(float(out.sum()) - 1.0) < 1e-12

    def test_run_unknown_op_raises(self, os_):
        with pytest.raises(KeyError):
            os_.run_isa("nope.nope", np.array([1.0]))

    def test_l2norm_unit(self, os_):
        out = os_.run_isa("norm.l2norm", np.array([3.0, 4.0]))
        assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-12


class TestISAFullGates:
    """全量 1024 条冒烟 + 数学性质断言（与 ifai 上游门禁同源）。"""

    def test_run_all_1024_green(self, os_):
        gates = os_.isa_gates()
        assert gates["total"] == 1024
        assert gates["ok"] == 1024
        assert gates["failed"] == []

    def test_property_assertions_green(self, os_):
        gates = os_.isa_gates()
        props = gates["properties"]
        assert props["failed"] == []
        assert props["passed"] > 0
        assert props["checked"] >= props["passed"]
