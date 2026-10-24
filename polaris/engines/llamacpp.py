"""llama.cpp 引擎适配器 —— 真实世界顶级智力的接入面。

零重写 · 出处
------------
**llama.cpp**（ggerganov/llama.cpp, MIT）是当今世界最著名的开源 LLM 推理
引擎（GGUF 量化、CPU/GPU/Metal/Vulkan 全后端、OpenAI 兼容 server）。
本适配器**不重写引擎的任何一行**，只做三件事：

1. **发现**：按环境变量 / 默认地址探测本地或远程 ``llama-server``；
2. **适配**：把 Polaris 契约翻译为 llama.cpp 的 OpenAI 兼容 HTTP 调用；
3. **降级**：不可用时 ``available()`` 如实返回 False，由调度器切内置引擎。

HTTP 客户端用标准库 ``urllib`` —— 零第三方依赖，与 Polaris 铁律一致。
llama.cpp server 的 OpenAI 兼容契约见其 ``tools/server`` 的公开文档。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

from .base import EngineAdapter, EngineResult

#: 发现顺序：环境变量优先，其次本地默认端口（llama-server 默认 8080）
ENV_URL = "POLARIS_LLAMA_CPP_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"


class LlamaCppAdapter(EngineAdapter):
    """经 HTTP 接入 llama.cpp server（OpenAI 兼容面）。

    Args:
        base_url: llama-server 根地址，如 ``http://127.0.0.1:8080``。
            默认读取 ``POLARIS_LLAMA_CPP_URL``，再退到本地 8080。
        timeout: 单次 HTTP 超时（秒）。``available()`` 探测用其 1/4。
        model_hint: 记入审计的模型标识（server 未回报时使用）。

    示例::

        # 终端一：llama-server -m model.gguf --port 8080
        # 终端二：
        from polaris.engines import LlamaCppAdapter
        a = LlamaCppAdapter()
        a.available()          # True（server 在线时）
        a.complete("你好", max_tokens=32)
    """

    name = "llama.cpp"
    kind = "world-class-external"
    capabilities = {
        "modalities": ["text"],
        "deterministic": True,   # temperature=0 时 llama.cpp 贪心解码确定性
        "maxTokens": 131072,     # 取决于所载模型的 ctx，保守声明
        "quantized": True,       # GGUF 量化家族
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        model_hint: str = "llama.cpp-server",
    ):
        self.base_url = (base_url or os.environ.get(ENV_URL) or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout)
        self.model_hint = model_hint

    # ------------------------------------------------------------------ 发现
    def available(self) -> bool:
        """探测 ``GET {base}/health``（llama-server 的健康端点）。"""
        try:
            req = urllib.request.Request(self.base_url + "/health", method="GET")
            with urllib.request.urlopen(req, timeout=max(self.timeout / 4, 1.0)) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    # ------------------------------------------------------------------ 适配
    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> EngineResult:
        """经 OpenAI 兼容 ``/v1/completions`` 调用 llama.cpp。

        零重写：请求/响应字段完全遵循 OpenAI Completions API（llama.cpp
        server 的公开兼容契约），Polaris 不做任何字段发明。
        """
        payload = {
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/v1/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        choice = (data.get("choices") or [{}])[0]
        text = choice.get("text", "") or ""
        meta: dict[str, Any] = {
            "adapter": "llama.cpp-http",
            "base_url": self.base_url,
            "model": data.get("model", self.model_hint),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage", {}),
        }
        return EngineResult(text=text, engine=self.name, tokens=None, degraded=False, meta=meta)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["base_url"] = self.base_url
        return d


__all__ = ["LlamaCppAdapter", "DEFAULT_BASE_URL", "ENV_URL"]
