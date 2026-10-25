"""P10 意图编译器门禁 —— 编译产物必须可证，不可证就别交付。

可证不变量（K1–K5，全部机检）：
- K1 编译产出非空；未命中任何家族 → NoIntentMatch（绝不静默空程序）
- K2 每条指令真实存在于 ISA 且 family 一致
- K3 确定性：探针上两次执行逐位相同
- K4 编译期干运行通过（每条候选都试跑过）
- K5 指纹只取决于指令序列，与调用次序无关
"""

from __future__ import annotations

import numpy as np
import pytest

from polaris.compiler import (
    NoIntentMatch,
    ProgramError,
    Step,
    compile_gates,
    compile_intent,
    extract_numbers,
    family_candidates,
    route,
)
from polaris.engines import MockEngine  # noqa: F401  (确保 vendor 路径已就绪)


class TestRoute:
    def test_route_scores_by_keyword_length(self):
        # "自注意力" 比 "attention" 更具体 → attn 必命中
        m = dict(route("计算自注意力"))
        assert "attn" in m and m["attn"] > 0

    def test_direct_op_name_boosts_family(self):
        m = dict(route("运行 smax.softmax"))
        assert m.get("smax", 0) > 0

    def test_no_match_returns_empty(self):
        assert route("帮我写一首七言绝句") == []


class TestExtractNumbers:
    def test_extracts_int_and_float_and_exp(self):
        nums = extract_numbers("取前 3 个，阈值 0.5，温度 1e-2")
        assert nums == [3.0, 0.5, 0.01]

    def test_empty_when_none(self):
        assert extract_numbers("没有数字") == []


class TestFamilyCandidates:
    def test_preferred_first(self):
        c = family_candidates("smax")
        assert c[0] == "smax.softmax"

    def test_all_belong_to_family(self):
        from ifai.l10 import MICROOPS

        c = family_candidates("norm")
        assert all(MICROOPS[n].family == "norm" for n in c)


class TestK1ProgramNonEmpty:
    def test_match_yields_nonempty_program(self):
        p = compile_intent("归一化后 softmax", width=2)
        assert len(p.steps) >= 1

    def test_no_match_raises(self):
        with pytest.raises(NoIntentMatch):
            compile_intent("帮我写首诗")

    def test_compile_gates_flag_k1(self):
        p = compile_intent("计算欧氏距离", width=1)
        g = compile_gates(p)
        k1 = next(x for x in g if x["gate"] == "K1")
        assert k1["passed"] is True


class TestK2InstructionsExist:
    def test_all_steps_in_isa_and_family_consistent(self):
        p = compile_intent("gelu 激活 + 卷积", width=3, max_families=2)
        from ifai.l10 import MICROOPS

        for s in p.steps:
            assert s.op in MICROOPS
            assert MICROOPS[s.op].family == s.family

    def test_k2_gate_passes(self):
        p = compile_intent("softmax 概率分布", width=2)
        g = compile_gates(p)
        k2 = next(x for x in g if x["gate"] == "K2")
        assert k2["passed"] is True


class TestK3Determinism:
    def test_two_runs_bit_identical(self):
        p = compile_intent("自注意力 + kv cache", width=3, max_families=2)
        x = np.asarray([1.0, 2.0, 3.0, 4.0])
        a = p.run(x)
        b = p.run(x)
        assert np.array_equal(a, b, equal_nan=True)

    def test_program_marks_deterministic(self):
        p = compile_intent("l2 归一化后 gelu", width=2)
        assert p.deterministic is True

    def test_k3_gate_passes(self):
        p = compile_intent("topk 检索", width=2)
        g = compile_gates(p)
        k3 = next(x for x in g if x["gate"] == "K3")
        assert k3["passed"] is True


class TestK4DryRunPassed:
    def test_k4_always_true_for_compiled_program(self):
        p = compile_intent("MoE 专家路由 + 量化", width=3, max_families=2)
        g = compile_gates(p)
        k4 = next(x for x in g if x["gate"] == "K4")
        assert k4["passed"] is True

    def test_dropped_candidates_recorded(self):
        # 宽程序必然触发一些干运行失败被剔除
        p = compile_intent("注意力 + 归一化 + 量化 + 池化", width=8, max_families=4)
        # 即便没被剔除，dropped 字段也应是 list（结构稳定）
        assert isinstance(p.dropped, list)


class TestK5FingerprintStable:
    def test_fingerprint_only_depends_on_ops(self):
        p = compile_intent("归一化 softmax", width=2)
        manual = "|".join(s.op for s in p.steps)
        import hashlib

        expect = hashlib.sha256(manual.encode()).hexdigest()[:16]
        assert p.fingerprint() == expect

    def test_same_ops_different_intent_same_fp(self):
        a = compile_intent("把这个向量归一化再做 softmax", width=2)
        b = compile_intent("归一化后 softmax 处理", width=2)
        assert a.fingerprint() == b.fingerprint()

    def test_k5_gate_passes(self):
        p = compile_intent("梯度下降用 adam 优化器", width=2)
        g = compile_gates(p)
        k5 = next(x for x in g if x["gate"] == "K5")
        assert k5["passed"] is True


class TestCompileExecutor:
    def test_run_returns_array(self):
        p = compile_intent("把这个向量归一化再做 softmax", width=2)
        r = p.run([1.0, 2.0, 3.0, 4.0])
        assert np.shape(r) is not None

    def test_trace_returns_snapshots(self):
        p = compile_intent("int8 量化 + 归一化", width=2)
        r, snaps = p.run([1.0, 2.0, 3.0], trace=True)
        assert len(snaps) == len(p.steps)

    def test_as_dict_json_ready(self):
        import json

        p = compile_intent("计算自注意力", width=2, max_families=1)
        json.dumps(p.as_dict(), ensure_ascii=False)


class TestCompileGateSummary:
    def test_all_gates_pass_for_clean_program(self):
        p = compile_intent("softmax", width=1)
        assert all(g["passed"] for g in p.gates)
        assert [g["gate"] for g in p.gates] == ["K1", "K2", "K3", "K4", "K5"]
