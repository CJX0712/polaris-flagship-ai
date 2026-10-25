"""L7 SSE 编码 —— 真正的 Server-Sent Events 流式输出。

零重写
------
SSE 协议本身按 **W3C/WHATWG HTML Living Standard** 的 "Server-sent events"
章节实现 (text/event-stream 媒体类型, ``data:``/``event:``/``id:`` 字段,
空行结束一个事件, ``[DONE]`` 收尾), 不自创协议; OpenAI 兼容的**分块形状**
(chunked delta 事件) 直接复用 OpenAI ``/v1/chat/completions`` ``stream=true``
的线上格式, 因此 LangChain / OpenAI SDK 零改动即可消费。

闭合的历史缺口 (G1)
-------------------
v0.7.13 全栈审计记录: ``serve/api.py`` 的 ``stream`` 参数**被静默忽略** ——
``_gen_tokens`` 接收 ``stream`` 形参却从不使用, HTTP 层也无条件返回
``application/json`` 整体响应。于是 INV-37 (「流式 content 拼接 == 非流式
content」) 变成**假绿**: 测试只比对了内部生成器, 没有验证真实的 SSE 传输
形态 —— 流式能力从未真正可达。

本模块提供缺失的那一层:
- :func:`sse_chunks` —— 把 token 迭代器编码为 SSE 字节流 (含 ``[DONE]``);
- :func:`iter_sse_lines` —— 解析 SSE 字节流为事件字典 (供客户端/测试消费);
- :func:`sse_headers` —— 标准 SSE 响应头 (含禁缓存与禁代理缓冲)。

可验证不变量:
- **INV-37 (真实形态)**: 从 SSE 字节流解出的所有 ``delta.content`` 拼接,
  **逐字等于**非流式响应的 ``choices[0].message.content``;
- **INV-37 (协议形态)**: 事件以空行分隔, 末事件为 ``data: [DONE]``;
- 媒体类型为 ``text/event-stream``。
"""
from __future__ import annotations

import json
from typing import Iterable, Iterator

__all__ = [
    "SSE_MEDIA_TYPE",
    "DONE_PAYLOAD",
    "sse_headers",
    "encode_sse_event",
    "sse_chat_chunk",
    "sse_chunks",
    "iter_sse_lines",
    "parse_sse_stream",
    "collect_sse_content",
]

#: SSE 媒体类型 (WHATWG HTML Living Standard §server-sent-events)。
SSE_MEDIA_TYPE = "text/event-stream; charset=utf-8"

#: OpenAI 兼容流的收尾哨兵。
DONE_PAYLOAD = "[DONE]"


def sse_headers() -> dict[str, str]:
    """返回标准 SSE 响应头。

    - ``Cache-Control: no-cache`` —— 流不可缓存;
    - ``Connection: keep-alive`` —— 保持长连接;
    - ``X-Accel-Buffering: no`` —— 禁用 nginx 等反向代理的缓冲, 否则
      增量 token 会被攒成一整块, 流式退化为非流式 (这是最常见的
      「流式不流式」现场坑)。
    """
    return {
        "Content-Type": SSE_MEDIA_TYPE,
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def encode_sse_event(data: str, event: str | None = None) -> str:
    """把一条负载编码为 SSE 事件文本 (以空行结尾)。

    Args:
        data: 事件负载 (已序列化的 JSON 字符串, 或 ``[DONE]``)。
        event: 可选事件名。

    Returns:
        形如 ``"data: {...}\\n\\n"`` 的文本。
    """
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


def sse_chat_chunk(content: str, model: str, chunk_id: str = "chatcmpl-ifai",
                   created: int | None = None, finish_reason: str | None = None) -> str:
    """构造一个 OpenAI 兼容的流式分块负载 (JSON 字符串)。

    形状与 OpenAI ``stream=true`` 的 ``chat.completion.chunk`` 一致:
    ``choices[0].delta.content`` 承载增量文本; 末块用 ``finish_reason``
    标记停止。

    Args:
        content: 本次增量文本 (可为空串)。
        model: 模型标识。
        chunk_id: 响应 id。
        created: Unix 时间戳 (默认当前时间)。
        finish_reason: 结束原因; ``None`` 表示仍在生成。

    Returns:
        JSON 字符串 (调用方用 :func:`encode_sse_event` 包成 SSE 事件)。
    """
    import time
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(created if created is not None else time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return json.dumps(payload, ensure_ascii=False)


def sse_chunks(tokens: Iterable[str], model: str, chunk_id: str = "chatcmpl-ifai",
               created: int | None = None) -> Iterator[str]:
    """把 token 迭代器编码为完整 SSE 事件流。

    事件序列:
    1. 每个 token 一个 ``chat.completion.chunk`` 事件 (增量 content);
    2. 一个 ``finish_reason="stop"`` 的末块事件;
    3. ``data: [DONE]`` 收尾。

    Args:
        tokens: token 文本迭代器。
        model: 模型标识。
        chunk_id: 响应 id。
        created: Unix 时间戳。

    Yields:
        SSE 事件文本 (每个都以空行结尾)。
    """
    ts = created if created is not None else 0
    for tok in tokens:
        yield encode_sse_event(sse_chat_chunk(tok, model, chunk_id, ts))
    yield encode_sse_event(sse_chat_chunk("", model, chunk_id, ts,
                                         finish_reason="stop"))
    yield encode_sse_event(DONE_PAYLOAD)


def iter_sse_lines(stream: bytes | str) -> Iterator[str]:
    """把 SSE 字节流拆成逐行 (按 ``\\n`` 切分, 去掉尾部 ``\\r``)。

    Args:
        stream: SSE 原始流 (bytes 或 str)。

    Yields:
        逐行文本 (不含换行符)。
    """
    if isinstance(stream, bytes):
        text = stream.decode("utf-8")
    else:
        text = stream
    for line in text.split("\n"):
        yield line.rstrip("\r")


def parse_sse_stream(stream: bytes | str) -> list[dict]:
    """解析 SSE 字节流为事件字典列表。

    每个事件字典形如 ``{"event": str | None, "data": str}``; ``data`` 为
    ``[DONE]`` 时原样保留 (调用方可据此判断收尾)。

    Args:
        stream: SSE 原始流。

    Returns:
        事件字典列表。
    """
    events: list[dict] = []
    name: str | None = None
    data_lines: list[str] = []
    for line in iter_sse_lines(stream):
        if line == "":
            # 空行 = 一个事件结束
            if data_lines:
                events.append({"event": name,
                               "data": "\n".join(data_lines)})
            name = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue  # 注释行 (keep-alive 心跳)
        if ":" in line:
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                name = value
            elif field == "data":
                data_lines.append(value)
    if data_lines:  # 流末尾没有空行的情况
        events.append({"event": name, "data": "\n".join(data_lines)})
    return events


def collect_sse_content(stream: bytes | str) -> str:
    """从 SSE 字节流中收集全部增量文本并拼接。

    这是 INV-37 的**客户端视角**: 拼接结果必须逐字等于非流式响应的
    ``choices[0].message.content``。

    Args:
        stream: SSE 原始流。

    Returns:
        拼接后的文本。
    """
    parts: list[str] = []
    for ev in parse_sse_stream(stream):
        data = ev["data"]
        if data == DONE_PAYLOAD:
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        for ch in obj.get("choices", []):
            delta = ch.get("delta", {})
            content = delta.get("content")
            if content:
                parts.append(content)
    return "".join(parts)
