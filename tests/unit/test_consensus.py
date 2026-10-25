"""P9 共识门禁 —— 一致性必须是可测量的数，而不是形容词。

可证不变量：
- C1 agreement ∈ [0,1]
- C2 1 ≤ support ≤ total
- C3 全票一致 ⇒ agreement == 1.0
- C4 两两全不同 ⇒ agreement == 1/n
- C5 固定 seed 下两次共识指纹相同
- C6 平局打破与顺序无关（同票取字典序最小）
- Jaccard 三条基本性质：自反=1、对称、空对空=1
"""

from __future__ import annotations

import pytest

from polaris.engines import MockEngine
from polaris.runtime.consensus import (
    Consensus,
    bigrams,
    consensus_gates,
    jaccard,
    majority_vote,
    mean_pairwise_similarity,
    similarity,
)


class _ScriptedEngine(MockEngine):
    """按脚本回放文本的引擎（把「引擎」这一变量钉死，只测共识逻辑）。"""

    name = "scripted"
    kind = "mock"

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.i = 0
        self.calls = 0

    def complete(self, prompt, *, max_tokens=64, temperature=0.0, seed=0):
        self.calls += 1
        text = self.script[self.i % len(self.script)]
        self.i += 1
        from polaris.engines import EngineResult

        return EngineResult(text=text, engine=self.name)


class TestJaccard:
    def test_identical_is_one(self):
        assert similarity("人工智能", "人工智能") == 1.0

    def test_symmetric(self):
        a, b = "深度学习模型", "机器学习模型"
        assert similarity(a, b) == pytest.approx(similarity(b, a))

    def test_disjoint_is_zero(self):
        assert similarity("abc", "xyz") == 0.0

    def test_empty_pairs(self):
        assert jaccard(set(), set()) == 1.0
        assert similarity("", "") == 1.0

    def test_bigrams_of_short_text(self):
        assert bigrams("中") == {"中"}
        assert bigrams("") == set()


class TestMajorityVote:
    def test_unanimous(self):
        winner, support, votes = majority_vote(["A", "A", "A"])
        assert winner == "A" and support == 3
        assert votes == [("A", 3)]

    def test_all_distinct_support_is_one(self):
        winner, support, _ = majority_vote(["A", "B", "C"])
        assert support == 1
        assert winner == "A"  # 同票取字典序最小 —— C6

    def test_tie_break_is_order_independent(self):
        w1, _, _ = majority_vote(["B", "A", "B", "A"])
        w2, _, _ = majority_vote(["A", "B", "A", "B"])
        assert w1 == w2 == "A"

    def test_whitespace_normalized(self):
        winner, support, _ = majority_vote(["  A  ", "A"])
        assert winner == "A" and support == 2

    def test_empty_input(self):
        assert majority_vote([]) == ("", 0, [])


class TestMeanPairwise:
    def test_single_sample_is_one(self):
        assert mean_pairwise_similarity(["A"]) == 1.0

    def test_all_same_is_one(self):
        assert mean_pairwise_similarity(["AB", "AB", "AB"]) == pytest.approx(1.0)

    def test_bounded(self):
        v = mean_pairwise_similarity(["人工智能", "机器学习", "区块链"])
        assert 0.0 <= v <= 1.0


class TestConsensusGates:
    def test_unanimous_gates(self):
        g = consensus_gates(1.0, 3, 3, ["A", "A", "A"])
        assert all(x["passed"] for x in g)
        assert [x["gate"] for x in g] == ["C1", "C2", "C3", "C4"]

    def test_all_distinct_gates(self):
        g = consensus_gates(1 / 3, 1, 3, ["A", "B", "C"])
        assert all(x["passed"] for x in g)

    def test_out_of_range_agreement_fails_c1(self):
        g = consensus_gates(1.5, 3, 3, ["A", "A", "A"])
        assert not next(x for x in g if x["gate"] == "C1")["passed"]


class TestSelfConsistency:
    def test_unanimous_samples(self):
        e = _ScriptedEngine(["同一个答案"])
        cr = Consensus().self_consistency(e, "q", k=4, temperature=0.8)
        assert cr.method == "self-consistency"
        assert cr.agreement == 1.0
        assert cr.support == 4 and cr.total == 4
        assert cr.mean_pairwise == pytest.approx(1.0)
        assert all(g["passed"] for g in cr.gates)

    def test_all_distinct_samples(self):
        e = _ScriptedEngine(["甲", "乙", "丙", "丁"])
        cr = Consensus().self_consistency(e, "q", k=4)
        assert cr.agreement == pytest.approx(0.25)
        assert cr.support == 1
        assert all(g["passed"] for g in cr.gates)

    def test_seed_sequence_is_deterministic_c5(self):
        # 同一脚本引擎 + 同一 seed ⇒ 两次共识指纹相同
        e1 = _ScriptedEngine(["甲", "乙", "甲", "丙"])
        e2 = _ScriptedEngine(["甲", "乙", "甲", "丙"])
        c = Consensus()
        a = c.self_consistency(e1, "q", k=4, seed=7)
        b = c.self_consistency(e2, "q", k=4, seed=7)
        assert a.fingerprint() == b.fingerprint()
        assert a.text == b.text

    def test_k_calls_are_actually_made(self):
        e = _ScriptedEngine(["A"])
        Consensus().self_consistency(e, "q", k=3)
        assert e.calls == 3

    def test_reports_record_seed_and_latency(self):
        e = _ScriptedEngine(["A"])
        cr = Consensus().self_consistency(e, "q", k=2, seed=10)
        assert [r["seed"] for r in cr.reports] == [10, 11]
        assert all(r["latency_ms"] >= 0 for r in cr.reports)


class TestCrossEngine:
    def test_two_engines_agree(self):
        a = _ScriptedEngine(["共识答案"])
        a.name = "engine-a"
        b = _ScriptedEngine(["共识答案"])
        b.name = "engine-b"
        cr = Consensus().cross_engine([a, b], "q")
        assert cr.method == "cross-engine"
        assert cr.agreement == 1.0
        assert cr.total == 2

    def test_two_engines_disagree(self):
        a = _ScriptedEngine(["甲方案"])
        a.name = "engine-a"
        b = _ScriptedEngine(["乙方案"])
        b.name = "engine-b"
        cr = Consensus().cross_engine([a, b], "q")
        assert cr.agreement == 0.5
        assert cr.support == 1
        assert cr.mean_pairwise < 1.0

    def test_engine_order_does_not_change_result(self):
        a = _ScriptedEngine(["甲方案"])
        a.name = "engine-a"
        b = _ScriptedEngine(["乙方案"])
        b.name = "engine-b"
        c = Consensus()
        r1 = c.cross_engine([a, b], "q")
        a2 = _ScriptedEngine(["甲方案"])
        a2.name = "engine-a"
        b2 = _ScriptedEngine(["乙方案"])
        b2.name = "engine-b"
        r2 = c.cross_engine([b2, a2], "q")
        assert r1.fingerprint() == r2.fingerprint()

    def test_as_dict_is_json_ready(self):
        import json

        e = _ScriptedEngine(["A"])
        cr = Consensus().cross_engine([e], "q")
        json.dumps(cr.as_dict(), ensure_ascii=False)
