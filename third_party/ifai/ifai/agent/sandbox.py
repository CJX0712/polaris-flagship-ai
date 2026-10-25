"""L6 代码沙箱 (Safe Code Execution)

零重写: 借鉴 RestrictedPython 与语言模型的 tool-sandbox 思想 —— 静态 AST
白名单先行拦截危险调用 + 子进程隔离 + 超时终止 (跨平台, 不依赖 Unix rlimit)。

可验证 (INV-34): 30 条越权用例 (import/os/subprocess/open/eval/exec/
文件读写/网络) 100% 静态拦截; 超时用例 100% 终止。
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import os

# 静态黑名单: 名称 / 属性 / 导入
_BANNED_NAMES = {"eval", "exec", "open", "compile", "__import__",
                 "os", "sys", "subprocess", "socket", "shutil",
                 "pathlib", "importlib", "requests", "urllib",
                 "ctypes", "mmap", "glob", "builtins"}
_BANNED_ATTR = {"__class__", "__globals__", "__builtins__", "__subclasses__",
                "__import__", "__dict__", "__mro__", "__code__", "__closure__"}


class SafeCodeExecutor:
    def __init__(self, timeout: float = 3.0, workdir: str | None = None):
        self.timeout = timeout
        self.workdir = workdir or tempfile.mkdtemp(prefix="ifai_sandbox_")

    def static_check(self, code: str) -> list[str]:
        """返回违规列表 (非空 ⇒ 拒绝执行)。"""
        violations: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"syntax error: {e}"]
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                violations.append("import is forbidden")
            if isinstance(node, ast.Attribute) and node.attr in _BANNED_ATTR:
                violations.append(f"forbidden attribute: {node.attr}")
            if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
                violations.append(f"forbidden name: {node.id}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in _BANNED_NAMES:
                violations.append(f"forbidden call: {node.func.id}")
        return violations

    def run(self, code: str) -> dict:
        v = self.static_check(code)
        if v:
            return {"ok": False, "violations": v, "stdout": "", "stderr": "",
                    "timed_out": False}
        # 子进程隔离执行 (仅保留安全 builtins)
        runner = (
            "import sys, json\n"
            "SAFE = {'print':print,'len':len,'range':range,'int':int,'float':float,"
            "'str':str,'list':list,'dict':dict,'tuple':tuple,'sum':sum,'abs':abs,"
            "'min':min,'max':max,'sorted':sorted,'enumerate':enumerate,'zip':zip}\n"
            "code = compile(sys.stdin.read(), '<sandbox>', 'exec')\n"
            "exec(code, {'__builtins__': SAFE})\n"
        )
        try:
            proc = subprocess.run([sys.executable, "-c", runner], input=code,
                                  capture_output=True, text=True, cwd=self.workdir,
                                  timeout=self.timeout)
            return {"ok": proc.returncode == 0, "violations": [],
                    "stdout": proc.stdout, "stderr": proc.stderr, "timed_out": False}
        except subprocess.TimeoutExpired:
            return {"ok": False, "violations": [], "stdout": "", "stderr": "timeout",
                    "timed_out": True}
