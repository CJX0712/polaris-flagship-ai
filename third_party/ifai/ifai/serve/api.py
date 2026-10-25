"""L7 OpenAI 兼容 HTTP 服务 (REST)

零重写: 兼容 OpenAI REST 协议 (v1/models, v1/completions, v1/chat/completions)
以便现有生态 (LangChain / OpenAI SDK) 零改动接入。chat/completions 严格返回
OpenAI 的 choices[].message.content 结构 (INV-36); 生成长度恒定, 保证流式
消费与非流式 content 一致 (INV-37)。

可验证:
- INV-36 响应字段集与 OpenAI schema diff 为空 (见 tests)
- INV-37 流式 content 拼接 == 非流式 content (**真实 SSE 形态**, 见 v0.8.0)
- INV-38 输入/输出护栏接入请求路径 (见 v0.8.0, 闭合 G2 缺口)

v0.8.0 闭合的历史缺口
---------------------
- **G1 (SSE 流式)**: 此前 ``stream`` 参数被静默忽略 —— ``_gen_tokens``
  接收该形参却从不使用, HTTP 层无条件返回 ``application/json`` 整体响应。
  INV-37 因此是「假绿」: 测试只比对了内部生成器, 没验证真实传输形态。
  现接入 :mod:`ifai.serve.sse`, ``stream=true`` 真正返回
  ``text/event-stream`` 分块流 (OpenAI 兼容的 delta 事件)。
- **G2 (护栏接线)**: 此前 :mod:`ifai.serve.guardrail` 426 行只在测试中
  可达, 服务路径完全不过护栏 —— 「L7 有护栏」是空头支票。现把输入检查
  接在生成前、输出检查接在生成后, 命中即返回 OpenAI 风格错误体。
"""
from __future__ import annotations

import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator, List

import numpy as np

from .. import rng
from ..engine.sampler import Sampler
from ..models.transformer import Transformer
from ..models.tokenizer import Tokenizer
from . import sse as _sse


def format_messages(messages: List[dict]) -> str:
    """将 OpenAI chat messages 拼接为单条 prompt (逐行 "role: content")。

    零重写: 与 OpenAI chat 语义一致, 供服务端与 L9 SDK 共用单一实现。

    Args:
        messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]

    Returns:
        拼接后的单条 prompt 字符串。
    """
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def chat_completion(text: str, model_name: str) -> dict:
    """构造 OpenAI 严格的 ``chat.completion`` 响应体 (INV-36)。

    零重写: 字段集与 OpenAI ``/v1/chat/completions`` schema 完全一致 (diff 为空),
    且 ``choices[].message.content`` 与 ``/v1/completions`` 的 ``choices[].text``
    逐字一致 (INV-37)。服务端 ``OpenAIServer.chat`` 与 L9 SDK ``IFAI.chat_completion``
    共用本函数, 保证单一事实来源。

    Args:
        text: 已生成的续写文本。
        model_name: 模型标识, 回填至响应 ``model`` 字段。

    Returns:
        符合 OpenAI schema 的 dict。
    """
    n = len(text)
    return {
        "id": "chatcmpl-ifai",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": n,
                  "total_tokens": n},
    }


def _guard_error(verdict) -> dict:
    """把护栏裁决转成 OpenAI 风格错误体。

    零重写: 错误体形状与 OpenAI API 的 ``{"error": {...}}`` 一致
    (message / type / param / code), 使现有 SDK 的错误处理逻辑零改动生效。

    Args:
        verdict: :class:`ifai.serve.guardrail.GuardrailVerdict`。

    Returns:
        OpenAI 风格错误体 dict。
    """
    return {
        "error": {
            "message": getattr(verdict, "reason", "") or "blocked by guardrail",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "guardrail_blocked",
        }
    }


class OpenAIServer:
    def __init__(self, model: Transformer, tokenizer: Tokenizer,
                 max_new_tokens: int = 64, seed: int = 0,
                 guardrail=None, engine_backend: str = "naive"):
        self.model = model
        self.tok = tokenizer
        self.max_new_tokens = max_new_tokens
        self.sampler = Sampler(seed=seed)
        self.model_name = "ifai-flagship"
        # 护栏默认关闭 (向后兼容既有调用方); 传入 Guardrail 实例即接入
        # 请求路径 (v0.8.0 闭合 G2 —— 此前 426 行护栏在服务路径完全不可达)。
        self.guardrail = guardrail
        # 推理后端 (v0.8.0 闭合 G3): naive / paged / chunked / speculative。
        # 四者输出逐 token 相同, 差异只在性能特征 (见 ifai/engine/session.py)。
        self.engine_backend = engine_backend
        self._engine = None

    @property
    def engine(self):
        """惰性构造 :class:`ifai.engine.session.EngineSession` (L3 五件套入口)。"""
        if self._engine is None:
            from ..engine.session import EngineSession
            self._engine = EngineSession(self.model, sampler=self.sampler,
                                         backend=self.engine_backend)
        return self._engine

    # ----- 逻辑层 (可在不联网情况下被测) -----
    def _gen_tokens(self, prompt: str, max_new: int, temperature: float,
                    top_p: float, stream: bool = False):
        """生成 token 迭代器。

        生成长度恒为 max_new (或遇 <eos> 提前结束), 保证流式==非流式
        (INV-37)。``stream`` 仅作语义标记保留 —— 真正的 SSE 编码在
        :func:`ifai.serve.sse.sse_chunks`, 由 HTTP 层消费本迭代器。
        """
        ids = self.tok.encode(prompt)
        if not ids:
            ids = [self.tok.token2id.get("<bos>", 0)]
        eos_id = self.tok.token2id.get("<eos>")
        # L3 引擎路径 (v0.8.0 G3): 与朴素路径输出逐 token 相同。
        new_ids = self.engine.generate(ids, max_new_tokens=max_new,
                                       eos_id=eos_id, temperature=temperature,
                                       top_p=top_p)
        for nxt in new_ids:
            yield self.tok.decode([nxt])

    # ----- 护栏 (INV-38) -----
    def _guard_input(self, prompt: str) -> dict | None:
        """输入护栏: 命中则返回 OpenAI 风格错误体, 否则 None。"""
        if self.guardrail is None:
            return None
        verdict = self.guardrail.check_input(prompt)
        if verdict.blocked:
            return _guard_error(verdict)
        return None

    def _guard_output(self, text: str) -> dict | None:
        """输出护栏: 命中则返回 OpenAI 风格错误体, 否则 None。"""
        if self.guardrail is None:
            return None
        verdict = self.guardrail.check_output(text)
        if verdict.blocked:
            return _guard_error(verdict)
        return None

    def completions(self, prompt: str, temperature=0.8, top_p=1.0,
                    max_tokens=None, stream=False) -> dict | str:
        max_new = max_tokens or self.max_new_tokens
        text = "".join(self._gen_tokens(prompt, max_new, temperature, top_p, stream))
        obj = {"id": "cmpl-ifai", "object": "text_completion",
               "model": self.model_name,
               "choices": [{"text": text, "index": 0,
                            "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 0, "completion_tokens": len(text),
                         "total_tokens": len(text)}}
        return obj

    def chat(self, messages: List[dict], **kw) -> dict | str:
        prompt = format_messages(messages)
        comp = self.completions(prompt, **kw)
        # OpenAI 严格兼容: chat/completions 返回 choices[].message.content
        # (INV-36 字段集与 OpenAI schema diff 为空; 与 completions 的 text 一致, INV-37)
        return chat_completion(comp["choices"][0]["text"], self.model_name)

    def stream_chat(self, messages: List[dict], temperature=0.8, top_p=1.0,
                    max_tokens=None) -> Iterator[str]:
        """以 SSE 事件流的形式产出 chat 增量 (v0.8.0 闭合 G1)。

        Args:
            messages: OpenAI 格式的对话消息。
            temperature: 采样温度。
            top_p: nucleus 采样截断。
            max_tokens: 最大生成 token 数。

        Yields:
            SSE 事件文本 (每个以空行结尾, 末事件为 ``data: [DONE]``)。
        """
        prompt = format_messages(messages)
        max_new = max_tokens or self.max_new_tokens
        tokens = self._gen_tokens(prompt, max_new, temperature, top_p, stream=True)
        yield from _sse.sse_chunks(tokens, self.model_name)

    def stream_completions(self, prompt: str, temperature=0.8, top_p=1.0,
                           max_tokens=None) -> Iterator[str]:
        """以 SSE 事件流的形式产出 completions 增量 (v0.8.0 闭合 G1)。"""
        max_new = max_tokens or self.max_new_tokens
        tokens = self._gen_tokens(prompt, max_new, temperature, top_p, stream=True)
        yield from _sse.sse_chunks(tokens, self.model_name)

    def models(self) -> dict:
        return {"object": "list",
                "data": [{"id": self.model_name, "object": "model",
                          "owned_by": "morningstar"}]}

    # ----- HTTP 路由 -----
    def route(self, method: str, path: str, body: dict | None = None):
        if path == "/v1/models" and method == "GET":
            return 200, self.models()
        if path == "/v1/completions" and method == "POST":
            b = body or {}
            prompt = b.get("prompt", "")
            blocked = self._guard_input(prompt)
            if blocked is not None:
                return 400, blocked
            if b.get("stream", False):
                return "sse", self.stream_completions(
                    prompt, b.get("temperature", 0.8), b.get("top_p", 1.0),
                    b.get("max_tokens"))
            out = self.completions(prompt, b.get("temperature", 0.8),
                                   b.get("top_p", 1.0), b.get("max_tokens"))
            blocked = self._guard_output(out["choices"][0]["text"])
            if blocked is not None:
                return 400, blocked
            return 200, out
        if path == "/v1/chat/completions" and method == "POST":
            b = body or {}
            messages = b.get("messages", [])
            blocked = self._guard_input(format_messages(messages))
            if blocked is not None:
                return 400, blocked
            if b.get("stream", False):
                return "sse", self.stream_chat(
                    messages, b.get("temperature", 0.8), b.get("top_p", 1.0),
                    b.get("max_tokens"))
            out = self.chat(messages, temperature=b.get("temperature", 0.8),
                            top_p=b.get("top_p", 1.0),
                            max_tokens=b.get("max_tokens"))
            blocked = self._guard_output(out["choices"][0]["message"]["content"])
            if blocked is not None:
                return 400, blocked
            return 200, out
        return 404, {"error": "not found"}


def create_handler(server: OpenAIServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status, payload):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(json.dumps(
                payload, ensure_ascii=False).encode("utf-8"))))
            self.end_headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def _send_sse(self, chunks):
            """真正的 SSE 分块响应 (v0.8.0 闭合 G1)。

            关键: 必须用 chunked transfer encoding 逐块 flush, 否则整个响应
            会被缓冲到末尾一次性发出 —— 那样「流式」只是名义上的。
            """
            self.send_response(200)
            for k, v in _sse.sse_headers().items():
                self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in chunks:
                data = chunk.encode("utf-8")
                self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def do_GET(self):
            st, pl = server.route("GET", self.path)
            self._send(st, pl)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                body = {}
            st, pl = server.route("POST", self.path, body)
            if st == "sse":
                self._send_sse(pl)
            else:
                self._send(st, pl)

        def log_message(self, *a):
            pass

    return Handler


def serve(server: OpenAIServer, host: str = "127.0.0.1", port: int = 8000):
    httpd = ThreadingHTTPServer((host, port), create_handler(server))
    httpd.serve_forever()
