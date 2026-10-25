"""llama.cpp 引擎适配器 —— 真实世界顶级智力的接入面。

零重写 · 出处
------------
**llama.cpp**（ggerganov/llama.cpp, MIT）是当今世界最著名的开源 LLM 推理
引擎（GGUF 量化、CPU/GPU/Metal/Vulkan 全后端、OpenAI 兼容 server）。
本适配器**不重写引擎的任何一行**，只做三件事：

1. **发现**：按环境变量 / 默认地址探测本地或远程 ``llama-server``；
2. **适配**：把 Polaris 契约翻译为 llama.cpp 的 OpenAI 兼容 HTTP 调用；
3. **降级**：不可用时 ``available()`` 如实返回 False，由调度器切内置引擎。

v1.1.0 起本类改为**继承** ``OpenAICompatEngine`` —— llama.cpp 与其他世界
顶级引擎（vLLM / SGLang / …）本就说同一种协议，把共同逻辑归并到一处，
比各写一遍更接近「零重写」。对外签名与行为完全不变。

HTTP 客户端用标准库 ``urllib`` —— 零第三方依赖，与 Polaris 铁律一致。
llama.cpp server 的 OpenAI 兼容契约见其 ``tools/server`` 的公开文档。
"""

from __future__ import annotations

import os
from typing import Optional

from .federation import PRESETS, OpenAICompatEngine

#: 发现顺序：环境变量优先，其次本地默认端口（llama-server 默认 8080）
ENV_URL = "POLARIS_LLAMA_CPP_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"


class LlamaCppAdapter(OpenAICompatEngine):
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

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        model_hint: str = "llama.cpp-server",
    ):
        super().__init__(
            "llama.cpp",
            base_url=base_url or os.environ.get(ENV_URL),
            timeout=timeout,
            model_hint=model_hint,
        )


__all__ = ["LlamaCppAdapter", "DEFAULT_BASE_URL", "ENV_URL", "PRESETS"]
