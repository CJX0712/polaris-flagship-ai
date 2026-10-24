"""L6 ReAct 智能体 (Reasoning + Acting)

零重写: 复用 Yao et al. 2022 "ReAct: Synergizing Reasoning and Acting in
Language Models" 的 Thought/Action/Observation 循环, 并可选 Reflexion
(Shinn et al. 2023) 自我反思复盘。

用法:
    agent = ReActAgent(llm_fn=my_llm, tools=registry)
    answer = agent.run("1+2 等于多少?")
其中 llm_fn(messages: list[dict]) -> str, 期望输出含
``Action: tool_name(key="value", ...)`` 或 ``Finish: 答案``。
"""
from __future__ import annotations

import json
import re
from typing import Callable, List

from .tools import ToolRegistry


class ReActAgent:
    def __init__(self, llm_fn: Callable[[List[dict]], str],
                 tools: ToolRegistry, max_steps: int = 6, memory=None):
        self.llm = llm_fn
        self.tools = tools
        self.max_steps = max_steps
        self.memory = memory
        self.history: List[dict] = []

    def _system(self) -> str:
        lines = ["你是一个使用工具的 ReAct 智能体。可用工具:"]
        for t in self.tools.describe():
            lines.append(f"- {t['name']}: {t['description']} 参数={json.dumps(t['schema'], ensure_ascii=False)}")
        lines.append("输出格式: 先 Thought, 再 Action: name(key=\"v\") 或 Finish: 最终答案")
        return "\n".join(lines)

    def _parse_action(self, text: str):
        m = re.search(r"Finish:\s*(.*)", text, re.S)
        if m:
            return ("finish", None, m.group(1).strip())
        m = re.search(r"Action:\s*([\w]+)\(", text)
        if m:
            name = m.group(1)
            start = m.end() - 1  # '(' 位置
            # 括号配对, 支持 Action 参数内嵌套括号
            depth = 0
            i = start
            for i in range(start, len(text)):
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
            args_raw = text[start + 1:i].strip()
            args = self._parse_args(args_raw)
            return ("action", name, args)
        return ("none", None, text.strip())

    @staticmethod
    def _parse_args(raw: str) -> dict:
        args = {}
        if not raw:
            return args
        for kv in re.findall(r'(\w+)\s*=\s*("(?:[^"]*)"|\S+)', raw):
            k, v = kv
            v = v.strip('"')
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            args[k] = v
        return args

    def run(self, question: str, reflexion: bool = False) -> str:
        self.history = [{"role": "system", "content": self._system()},
                        {"role": "user", "content": question}]
        for step in range(self.max_steps):
            out = self.llm(self.history)
            kind, name, payload = self._parse_action(out)
            self.history.append({"role": "assistant", "content": out})
            if kind == "finish":
                if self.memory is not None:
                    self.memory.remember_event(f"Q:{question} A:{payload}")
                return payload
            if kind == "action" and name in self.tools.tools:
                spec = self.tools.get(name)
                res = spec(payload) if not spec.validate(payload) else \
                    {"error": "; ".join(spec.validate(payload))}
                obs = f"Observation: {json.dumps(res, ensure_ascii=False)}"
                self.history.append({"role": "user", "content": obs})
            else:
                self.history.append({"role": "user",
                                     "content": "Observation: 无效动作, 请重试。"})
        # Reflexion: 若未 Finish, 反思后重试一次
        if reflexion:
            self.history.append({"role": "user",
                                 "content": "Reflection: 你尚未给出最终答案, 请复盘并 Finish。"})
            out = self.llm(self.history)
            kind, name, payload = self._parse_action(out)
            if kind == "finish":
                # v0.7.14: 反思成功的答案同样落记忆 (与主循环 Finish 同语义 —
                # 旧实现此路径绕过 memory, 反思后最值得记住的答案反而丢失)
                if self.memory is not None:
                    self.memory.remember_event(f"Q:{question} A:{payload}")
                return payload
        return "未能在步数内完成。"
