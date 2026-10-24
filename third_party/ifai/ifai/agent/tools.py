"""L6 工具系统 (Tool calling + JSON Schema 校验)

零重写: 工具调用范式参考 OpenAI Function Calling / Toolformer (Schick et al. 2023)
的"结构化工具调用 + schema 校验"。

可验证 (INV-33): 工具参数 JSON Schema 校验 100% 命中非法输入。
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .sandbox import SafeCodeExecutor


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict                       # JSON Schema (properties + required)
    func: Callable[..., Any] = field(default=lambda *a, **k: None)

    def validate(self, args: dict) -> list[str]:
        return validate_args(self.schema, args)

    def __call__(self, args: dict):
        errs = self.validate(args)
        if errs:
            raise ValueError("; ".join(errs))
        return self.func(**args)


def validate_args(schema: dict, args: dict) -> list[str]:
    """极简 JSON Schema 校验 (type / required / enum / minimum)。INV-33。"""
    errs: list[str] = []
    props = schema.get("properties", {})
    required = schema.get("required", [])
    for k in required:
        if k not in args:
            errs.append(f"missing required arg: {k}")
    for k, v in args.items():
        if k not in props:
            errs.append(f"unknown arg: {k}")
            continue
        spec = props[k]
        t = spec.get("type")
        if t == "integer" and not isinstance(v, int):
            errs.append(f"{k}: expected integer")
        elif t == "number" and not isinstance(v, (int, float)):
            errs.append(f"{k}: expected number")
        elif t == "string" and not isinstance(v, str):
            errs.append(f"{k}: expected string")
        elif t == "boolean" and not isinstance(v, bool):
            errs.append(f"{k}: expected boolean")
        if "minimum" in spec and isinstance(v, (int, float)) and v < spec["minimum"]:
            errs.append(f"{k}: below minimum {spec['minimum']}")
        if "enum" in spec and v not in spec["enum"]:
            errs.append(f"{k}: not in {spec['enum']}")
    return errs


class Calculator:
    """安全算术计算器 (仅允许四则 + 括号, 经 AST 白名单)。"""

    # 仅允许: 表达式 / 二元与一元运算 / 数值常量 / 运算符; 禁止函数调用等 (INV-34)
    # ast.Constant 涵盖 Python 3.8+ 的全部字面量, 无需 ast.Num (已废弃)。
    ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
               ast.USub, ast.UAdd)

    @staticmethod
    def _eval(node, names):
        if isinstance(node, ast.Expression):
            return Calculator._eval(node.body, names)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("only numbers allowed")
        if isinstance(node, ast.UnaryOp):
            v = Calculator._eval(node.operand, names)
            return +v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            a = Calculator._eval(node.left, names)
            b = Calculator._eval(node.right, names)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow):
                return a ** b
            if isinstance(node.op, ast.Mod):
                return a % b
        raise ValueError("unsupported expression")

    @classmethod
    def run(cls, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, cls.ALLOWED):
                raise ValueError(f"disallowed node: {type(node).__name__}")
        return float(cls._eval(tree, {}))


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        self.tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def call(self, name: str, args: dict):
        spec = self.tools[name]
        return spec(args)

    def describe(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "schema": t.schema}
                for t in self.tools.values()]


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="calculator", description="计算算术表达式",
        schema={"type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]},
        func=Calculator.run))
    # v0.7.14 接线 (审计缺口 2): AST 白名单沙箱此前仅测试可达, `ifai agent`
    # 入口触不可达 — 现注册为 code_exec 工具, L6 卖点「安全沙箱」真实可用。
    sandbox = SafeCodeExecutor()
    reg.register(ToolSpec(
        name="code_exec", description="在 AST 白名单安全沙箱中执行受信任的 Python 代码片段",
        schema={"type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"]},
        func=lambda code: sandbox.run(code)))
    return reg
