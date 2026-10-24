"""L6 轨迹记录与回放 (Trace / Deterministic Replay)

零重写 (集大成):
- **ReAct** (Yao et al. 2022 "Synergizing Reasoning and Acting in Language
  Models"): 轨迹的基本单元即 ``Thought / Action / Observation`` 三元组, 本模块
  把它结构化落盘, 而不是散落在 prompt 里。
- **Reflexion** (Shinn et al. 2023): 轨迹可被复盘、被当作下一轮的输入 — 这正
  是"可回放"要满足的前提 (轨迹必须是**完整且可复现**的记录)。
- **OpenTelemetry 语义约定**: 轨迹带 ``trace_id`` / step index / 时间戳, 便于
  跨服务串联; 比对时排除时间戳, 只比语义内容 (避免破坏确定性判据)。
- 确定性同 seed 重放沿用 **MLPerf / 可复现训练** 的共识 (与 INV-40 同口径):
  同 seed ⇒ 同随机流 ⇒ 同轨迹。

可验证不变量:
- **INV-35 轨迹可回放**: 同 seed 重放 ≡ 原始轨迹 (100% 一致)。比对口径为
  ``Trace.fingerprint()`` (逐步 Thought/Action/Action-args/Observation 的
  sha256), 与逐步 :meth:`TraceStep.key` 双保险。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .react import ReActAgent
from .tools import ToolRegistry

_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n\s*(?:Action|Finish|Observation)\s*:|$)", re.S)
_FINISH_RE = re.compile(r"Finish:\s*(.*)", re.S)


def _stable_hash(text: str) -> int:
    """跨进程稳定的字符串哈希 (避免内置 ``hash()`` 的 PYTHONHASHSEED 随机化)。"""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _extract_thought(raw: str) -> str:
    """从 LLM 原始输出中抽取 ``Thought:`` 内容 (缺失时返回空串)。"""
    m = _THOUGHT_RE.search(raw or "")
    return m.group(1).strip() if m else ""


@dataclass
class TraceStep:
    """一步 ReAct 记录: Thought / Action / Observation。"""

    index: int
    thought: str
    action: str                      # 工具名 / "finish" / "none"
    action_args: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    raw: str = ""                    # LLM 原始输出 (回放脚本用)
    ts: Optional[float] = None       # 时间戳: 不参与一致性比对

    def key(self) -> tuple:
        """语义键 (排除时间戳), 用于逐步比对。"""
        return (int(self.index), self.thought, self.action,
                json.dumps(self.action_args, sort_keys=True, ensure_ascii=False),
                self.observation)

    def to_dict(self) -> dict:
        return {"index": self.index, "thought": self.thought, "action": self.action,
                "action_args": dict(self.action_args), "observation": self.observation,
                "raw": self.raw, "ts": self.ts}

    @classmethod
    def from_dict(cls, d: dict) -> "TraceStep":
        return cls(index=int(d["index"]), thought=d.get("thought", ""),
                   action=d.get("action", "none"),
                   action_args=dict(d.get("action_args") or {}),
                   observation=d.get("observation", ""), raw=d.get("raw", ""),
                   ts=d.get("ts"))


@dataclass
class Trace:
    """一次 ReAct 运行的完整轨迹。"""

    question: str
    seed: int = 0
    steps: List[TraceStep] = field(default_factory=list)
    final_answer: str = ""
    trace_id: str = ""

    def __len__(self) -> int:
        return len(self.steps)

    def fingerprint(self) -> str:
        """轨迹指纹: 逐步语义键的 sha256 (INV-35 的主判据)。"""
        payload = json.dumps(
            {"question": self.question, "seed": int(self.seed),
             "steps": [list(s.key()) for s in self.steps],
             "final_answer": self.final_answer},
            ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def same_steps(self, other: "Trace") -> bool:
        """逐步比对 (排除时间戳), 与 :meth:`fingerprint` 互为双保险。"""
        if len(self.steps) != len(other.steps):
            return False
        return all(a.key() == b.key() for a, b in zip(self.steps, other.steps))

    def to_dict(self) -> dict:
        return {"question": self.question, "seed": int(self.seed),
                "trace_id": self.trace_id, "final_answer": self.final_answer,
                "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        return cls(question=d["question"], seed=int(d.get("seed", 0)),
                   trace_id=d.get("trace_id", ""),
                   final_answer=d.get("final_answer", ""),
                   steps=[TraceStep.from_dict(s) for s in d.get("steps", [])])

    def to_json(self, path: Optional[str] = None) -> str:
        """序列化为 JSON (``path`` 给出时同时落盘)。"""
        s = json.dumps(self.to_dict(), ensure_ascii=False, indent=1, sort_keys=True)
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    @classmethod
    def from_json(cls, source: str) -> "Trace":
        """从 JSON 字符串或文件路径重建轨迹。"""
        if os.path.exists(source) and len(source) < 4096:
            with open(source, "r", encoding="utf-8") as f:
                source = f.read()
        return cls.from_dict(json.loads(source))


class SeededLLM:
    """确定性 LLM 存根: 由 ``(seed, 步序, 上下文)`` 经 PCG64 决定 Thought/Action。

    仅用于"轨迹可回放"的可复现验证与演示 (真实部署替换为真实 LLM)。
    每一步的随机流独立派生 (``np.random.default_rng([seed, step, ctx_hash])``),
    因此**同 seed → 同决策序列 → 同轨迹** (INV-35 / INV-40 同口径)。
    """

    def __init__(self, tools: ToolRegistry, seed: int = 0, max_steps: int = 6,
                 finish_prob: float = 0.4):
        self.tools = tools
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.finish_prob = float(finish_prob)

    def __call__(self, messages: List[dict]) -> str:
        step = sum(1 for m in messages if m.get("role") == "assistant")
        ctx = messages[-1]["content"] if messages else ""
        r = np.random.default_rng([self.seed, step, _stable_hash(str(ctx)) & 0xFFFFFFFF])
        specs = self.tools.describe()
        name = specs[int(r.integers(0, len(specs)))]["name"] if specs else "none"
        thought = f"Thought: 第 {step + 1} 步, 先用 {name} 获取信息再作答。"
        last_obs = ""
        for m in reversed(messages):
            if m.get("role") == "user" and str(m.get("content", "")).startswith("Observation:"):
                last_obs = str(m["content"])
                break
        if not specs or step >= self.max_steps - 1 or r.random() < self.finish_prob:
            return f"{thought}\nFinish: {self._answer(last_obs)}"
        spec = self.tools.get(name)
        args = self._plan_args(spec.schema if spec else {}, r, ctx) if spec else {}
        args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
        return f"{thought}\nAction: {name}({args_str})"

    @staticmethod
    def _answer(last_obs: str) -> str:
        """由最近一次 Observation 生成确定性最终答案。"""
        m = re.search(r"Observation:\s*(.*)", last_obs or "", re.S)
        return (m.group(1).strip() if m else "暂无观察结果").strip() or "暂无观察结果"

    @staticmethod
    def _plan_args(schema: dict, r: np.random.Generator, ctx: str) -> Dict[str, Any]:
        """按 JSON Schema 生成确定性参数 (INV-33 保证可被校验通过)。"""
        args: Dict[str, Any] = {}
        props = schema.get("properties", {})
        for key, spec in props.items():
            t = spec.get("type", "string")
            if t == "integer":
                args[key] = int(r.integers(0, 100))
            elif t == "number":
                args[key] = float(round(float(r.random()), 4))
            elif t == "boolean":
                args[key] = bool(int(r.integers(0, 2)))
            elif spec.get("enum"):
                args[key] = spec["enum"][int(r.integers(0, len(spec["enum"])))]
            elif key == "expression":
                a, b = int(r.integers(1, 50)), int(r.integers(1, 50))
                op = ["+", "-", "*"][int(r.integers(0, 3))]
                args[key] = f"({a}{op}{b})"
            else:
                args[key] = f"q{int(r.integers(0, 1000))}"
        return args


class TracedReActAgent(ReActAgent):
    """ReAct 智能体的"带轨迹"版本: 逐步记录 Thought/Action/Observation (INV-35)。"""

    def __init__(self, llm_fn: Callable[[List[dict]], str], tools: ToolRegistry,
                 max_steps: int = 6, memory=None):
        super().__init__(llm_fn, tools, max_steps, memory)
        self.trace: Optional[Trace] = None

    def run_traced(self, question: str, seed: int = 0,
                   reflexion: bool = False) -> Trace:
        """运行并逐步记录轨迹。

        Args:
            question: 用户问题。
            seed: 记录进轨迹的随机种子 (回放时用于复现同一随机流)。
            reflexion: 未 Finish 时是否追加一次 Reflexion 反思 (Shinn et al. 2023)。

        Returns:
            :class:`Trace`, 同时存入 ``self.trace``。
        """
        self.history = [{"role": "system", "content": self._system()},
                        {"role": "user", "content": question}]
        steps: List[TraceStep] = []
        final = ""
        for step in range(self.max_steps):
            out = self.llm(self.history)
            kind, name, payload = self._parse_action(out)
            thought = _extract_thought(out)
            self.history.append({"role": "assistant", "content": out})
            if kind == "finish":
                steps.append(TraceStep(index=step, thought=thought, action="finish",
                                       action_args={}, observation=f"Finish: {payload}",
                                       raw=out, ts=time.time()))
                final = str(payload)
                if self.memory is not None:
                    self.memory.remember_event(f"Q:{question} A:{payload}")
                break
            args = payload if isinstance(payload, dict) else {}
            if kind == "action" and name in self.tools.tools:
                spec = self.tools.get(name)
                errs = spec.validate(args) if spec is not None else ["unknown tool"]
                if errs:
                    res: Any = {"error": "; ".join(errs)}
                else:
                    res = spec(args) if spec is not None else {"error": "unknown tool"}
                obs = f"Observation: {json.dumps(res, ensure_ascii=False)}"
            else:
                obs = "Observation: 无效动作, 请重试。"
            self.history.append({"role": "user", "content": obs})
            steps.append(TraceStep(index=step, thought=thought,
                                   action=str(name if kind == "action" else "none"),
                                   action_args=args, observation=obs, raw=out,
                                   ts=time.time()))
        if not final and reflexion:
            self.history.append({"role": "user",
                                 "content": "Reflection: 你尚未给出最终答案, 请复盘并 Finish。"})
            out = self.llm(self.history)
            kind, name, payload = self._parse_action(out)
            thought = _extract_thought(out)
            step = len(steps)
            if kind == "finish":
                final = str(payload)
                steps.append(TraceStep(index=step, thought=thought, action="finish",
                                       action_args={}, observation=f"Finish: {payload}",
                                       raw=out, ts=time.time()))
            else:
                steps.append(TraceStep(index=step, thought=thought,
                                       action=str(name or "none"), action_args={},
                                       observation="Observation: 无效动作, 请重试。",
                                       raw=out, ts=time.time()))
        trace = Trace(question=question, seed=int(seed), steps=steps,
                      final_answer=final or "未能在步数内完成。",
                      trace_id=hashlib.sha256(
                          f"{question}|{seed}".encode("utf-8")).hexdigest()[:16])
        self.trace = trace
        return trace


def replay_trace(trace: Trace, tools: ToolRegistry,
                 max_steps: Optional[int] = None) -> Trace:
    """按轨迹记录的 Action 重放 (不再让 LLM 重新决策), 返回重放轨迹。

    Args:
        trace: 原始轨迹 (提供每一步的 ``raw`` 输出与工具调用)。
        tools: 工具注册表 (重放时实际执行工具, 校验 Observation 是否复现)。
        max_steps: 覆盖步数上限 (默认取轨迹长度)。

    Returns:
        重放得到的 :class:`Trace`; 与 ``trace`` 逐步相同即满足 INV-35。
    """
    script: Dict[int, str] = {s.index: s.raw for s in trace.steps}

    def scripted_llm(messages: List[dict]) -> str:
        step = sum(1 for m in messages if m.get("role") == "assistant")
        if step in script:
            return script[step]
        last = trace.steps[-1].raw if trace.steps else "Finish: "
        return f"{last}\nFinish: {trace.final_answer}" if "Finish" not in last else last

    agent = TracedReActAgent(scripted_llm, tools,
                             max_steps=int(max_steps or max(len(trace.steps), 1)))
    return agent.run_traced(trace.question, seed=trace.seed)


def compare_traces(a: Trace, b: Trace) -> bool:
    """INV-35 判据: 两条轨迹是否 100% 一致 (指纹 + 逐步双判据)。"""
    return a.fingerprint() == b.fingerprint() and a.same_steps(b)


def record_and_replay(question: str, tools: ToolRegistry, seed: int = 0,
                      max_steps: int = 6) -> dict:
    """一键自检: 同 seed 录制 → 重放 → JSON 往返, 返回一致性报告。

    Returns:
        dict: ``identical`` / ``fingerprint`` / ``n_steps`` / ``json_roundtrip`` /
        ``seed_sensitive`` (不同 seed 是否会产生不同轨迹, 判别力佐证)。
    """
    agent = TracedReActAgent(SeededLLM(tools, seed=seed, max_steps=max_steps), tools,
                             max_steps=max_steps)
    original = agent.run_traced(question, seed=seed)
    again = TracedReActAgent(SeededLLM(tools, seed=seed, max_steps=max_steps), tools,
                             max_steps=max_steps).run_traced(question, seed=seed)
    replayed = replay_trace(original, tools, max_steps=max_steps)
    restored = Trace.from_json(original.to_json())
    fps = {TracedReActAgent(SeededLLM(tools, seed=s, max_steps=max_steps), tools,
                            max_steps=max_steps).run_traced(question, seed=s).fingerprint()
           for s in range(5)}
    return {
        "identical": bool(compare_traces(original, again)
                          and compare_traces(original, replayed)),
        "fingerprint": original.fingerprint(),
        "n_steps": len(original.steps),
        "json_roundtrip": bool(compare_traces(original, restored)),
        "seed_sensitive": len(fps) >= 2,
    }
