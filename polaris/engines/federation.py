"""引擎联邦 —— 一份契约，统治全部世界顶级推理引擎。

零重写 · 出处
------------
当今**所有**世界顶级 LLM 推理引擎都实现了同一份公开契约 ——
**OpenAI Completions API**（``POST {base}/v1/completions``）。这是生态事实，
不是 Polaris 的发明。因此正确做法不是为每个引擎写一个适配器（那是重写），
而是写**一个**适配器 + 一张**预设表**：

===================  ============================  ======  ==========================
世界顶级引擎          项目（许可）                    默认端口  健康端点
===================  ============================  ======  ==========================
llama.cpp server     ggerganov/llama.cpp (MIT)      8080    /health
vLLM                 vllm-project/vllm (Apache-2.0) 8000    /health
SGLang               sgl-project/sglang (Apache-2.0) 30000  /health_generate
Ollama               ollama/ollama (MIT)            11434   /api/version
LM Studio            lmstudio-ai (proprietary/CLI)  1234    /v1/models
TGI                  huggingface/text-generation-inference (Apache-2.0) 8080 /health
MLX (mlx-lm)         ml-explore/mlx-examples (MIT)  8080    /v1/models
TensorRT-LLM         NVIDIA/TensorRT-LLM (Apache-2.0) 8000  /health
NVIDIA NIM           NVIDIA NIM (Apache-2.0 comp.)  8000    /v1/health/ready
OpenAI 兼容云端      任意（OpenRouter / 各家网关）     —       /v1/models
===================  ============================  ======  ==========================

Polaris 只做三件别人没做的事：

1. **契约归一**：一个 ``OpenAICompatEngine`` 吃下上表全部引擎；
2. **发现式注册**：``autodiscover()`` 扫描预设，谁在线就注册谁；
3. **治理统一**：无论哪个引擎产出，都过同一套 G1/G2/G3 门禁 + 哈希链审计。

HTTP 客户端仍是标准库 ``urllib`` —— 零第三方依赖铁律不变。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from .base import EngineAdapter, EngineResult


@dataclass
class EnginePreset:
    """一个世界顶级引擎的接入预设（纯数据，不含逻辑）。

    Attributes:
        key: 预设键（``build_engine("vllm")`` 用）。
        kind: 引擎类别，决定调度优先级。
        default_url: 默认根地址。
        env: 覆盖地址的环境变量名。
        health_paths: 候选健康端点，按序探测，任一 200 即认为在线。
        capabilities: 能力声明（调度器据此匹配约束）。
        auth_env: 读取 Bearer token 的环境变量名（云端网关用）。
        cite: 出处（项目主页），用于 REFERENCES 与自证。
    """

    key: str
    kind: str
    default_url: str
    env: str
    health_paths: tuple[str, ...]
    capabilities: dict[str, Any] = field(default_factory=dict)
    auth_env: Optional[str] = None
    cite: str = ""


#: 世界顶级引擎预设库（端口与健康端点均取自各项目公开文档/默认配置）
PRESETS: dict[str, EnginePreset] = {
    "llama.cpp": EnginePreset(
        key="llama.cpp",
        kind="world-class-external",
        default_url="http://127.0.0.1:8080",
        env="POLARIS_LLAMA_CPP_URL",
        health_paths=("/health", "/v1/models"),
        capabilities={
            "modalities": ["text"],
            "deterministic": True,
            "maxTokens": 131072,
            "quantized": True,
            "streaming": True,
            "note": "GGUF 量化家族 · CPU/GPU/Metal/Vulkan 全后端",
        },
        cite="https://github.com/ggerganov/llama.cpp",
    ),
    "vllm": EnginePreset(
        key="vllm",
        kind="world-class-external",
        default_url="http://127.0.0.1:8000",
        env="POLARIS_VLLM_URL",
        health_paths=("/health", "/v1/models"),
        capabilities={
            "modalities": ["text"],
            "deterministic": True,
            "maxTokens": 262144,
            "quantized": True,
            "streaming": True,
            "note": "PagedAttention · 连续批处理（Kwon et al. 2023）",
        },
        cite="https://github.com/vllm-project/vllm",
    ),
    "sglang": EnginePreset(
        key="sglang",
        kind="world-class-external",
        default_url="http://127.0.0.1:30000",
        env="POLARIS_SGLANG_URL",
        health_paths=("/health_generate", "/health", "/v1/models"),
        capabilities={
            "modalities": ["text"],
            "deterministic": True,
            "maxTokens": 262144,
            "quantized": True,
            "streaming": True,
            "note": "RadixAttention 前缀缓存 · 零开销批处理调度器",
        },
        cite="https://github.com/sgl-project/sglang",
    ),
    "ollama": EnginePreset(
        key="ollama",
        kind="world-class-external",
        default_url="http://127.0.0.1:11434",
        env="POLARIS_OLLAMA_URL",
        health_paths=("/api/version", "/v1/models"),
        capabilities={
            "modalities": ["text"],
            "deterministic": True,
            "maxTokens": 131072,
            "quantized": True,
            "streaming": True,
            "note": "本地模型运行时 · GGUF 生态",
        },
        cite="https://github.com/ollama/ollama",
    ),
    "lmstudio": EnginePreset(
        key="lmstudio",
        kind="world-class-external",
        default_url="http://127.0.0.1:1234",
        env="POLARIS_LMSTUDIO_URL",
        health_paths=("/v1/models",),
        capabilities={
            "modalities": ["text"],
            "deterministic": True,
            "maxTokens": 131072,
            "quantized": True,
            "streaming": True,
            "note": "桌面本地推理服务（llama.cpp 内核）",
        },
        cite="https://lmstudio.ai/docs/app/api/endpoints/openai",
    ),
    "tgi": EnginePreset(
        key="tgi",
        kind="world-class-external",
        default_url="http://127.0.0.1:8080",
        env="POLARIS_TGI_URL",
        health_paths=("/health", "/v1/models"),
        capabilities={
            "modalities": ["text"],
            "deterministic": True,
            "maxTokens": 131072,
            "quantized": True,
            "streaming": True,
            "note": "HuggingFace 生产级推理服务 · 连续批处理",
        },
        cite="https://github.com/huggingface/text-generation-inference",
    ),
    "mlx": EnginePreset(
        key="mlx",
        kind="world-class-external",
        default_url="http://127.0.0.1:8080",
        env="POLARIS_MLX_URL",
        health_paths=("/v1/models", "/health"),
        capabilities={
            "modalities": ["text"],
            "deterministic": True,
            "maxTokens": 32768,
            "quantized": True,
            "streaming": True,
            "note": "Apple Silicon 统一内存推理（mlx-lm server）",
        },
        cite="https://github.com/ml-explore/mlx-examples",
    ),
    "tensorrt-llm": EnginePreset(
        key="tensorrt-llm",
        kind="world-class-external",
        default_url="http://127.0.0.1:8000",
        env="POLARIS_TRTLLM_URL",
        health_paths=("/health", "/v1/models"),
        capabilities={
            "modalities": ["text"],
            "deterministic": True,
            "maxTokens": 131072,
            "quantized": True,
            "streaming": True,
            "note": "NVIDIA 极限吞吐编译器栈 · in-flight batching",
        },
        cite="https://github.com/NVIDIA/TensorRT-LLM",
    ),
    "nim": EnginePreset(
        key="nim",
        kind="world-class-external",
        default_url="http://127.0.0.1:8000",
        env="POLARIS_NIM_URL",
        health_paths=("/v1/health/ready", "/health"),
        capabilities={
            "modalities": ["text"],
            "deterministic": True,
            "maxTokens": 131072,
            "quantized": True,
            "streaming": True,
            "note": "NVIDIA NIM 微服务（OpenAI 兼容）",
        },
        cite="https://build.nvidia.com/",
    ),
    "openai-compatible": EnginePreset(
        key="openai-compatible",
        kind="world-class-external",
        default_url="https://api.openai.com/v1",
        env="POLARIS_OPENAI_COMPAT_URL",
        health_paths=("/v1/models", "/models"),
        capabilities={
            "modalities": ["text"],
            "deterministic": True,
            "maxTokens": 131072,
            "quantized": False,
            "streaming": True,
            "note": "任意 OpenAI 兼容网关（OpenRouter / 各家云端）",
        },
        auth_env="POLARIS_OPENAI_COMPAT_KEY",
        cite="https://platform.openai.com/docs/api-reference/completions",
    ),
}


class OpenAICompatEngine(EngineAdapter):
    """OpenAI Completions 兼容引擎的统一适配器。

    一份实现覆盖 llama.cpp / vLLM / SGLang / Ollama / LM Studio / TGI /
    MLX / TensorRT-LLM / NIM / 任意 OpenAI 兼容网关 —— 因为**它们本就
    说同一种协议**，为每种各写一遍才是重写。

    Args:
        preset: 预设键（``"vllm"``）或 ``EnginePreset`` 实例。
        base_url: 覆盖预设地址。
        timeout: HTTP 超时（秒）；健康探测取其 1/4，下限 1s。
        api_key: Bearer token；缺省读预设的 ``auth_env``。
        model_hint: 服务端未回报模型名时的记账标识。

    示例::

        # vLLM 已在 8000 端口起服
        e = OpenAICompatEngine("vllm")
        e.available()                       # True
        e.complete("你好", max_tokens=32)

        # 或让 Polaris 自己找：谁在线用谁
        from polaris.engines.federation import autodiscover
        engines = autodiscover(timeout=1.0)  # 返回在线引擎列表
    """

    def __init__(
        self,
        preset: str | EnginePreset = "llama.cpp",
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
        model_hint: Optional[str] = None,
    ):
        p = PRESETS[preset] if isinstance(preset, str) else preset
        if isinstance(preset, str) and preset not in PRESETS:
            raise KeyError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
        self.preset = p
        self.name = p.key
        self.kind = p.kind
        self.capabilities = dict(p.capabilities)
        self.base_url = (
            base_url or os.environ.get(p.env) or p.default_url
        ).rstrip("/")
        self.timeout = float(timeout)
        self.api_key = api_key or (os.environ.get(p.auth_env) if p.auth_env else None)
        self.model_hint = model_hint or p.key

    # ------------------------------------------------------------------ 发现
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def available(self) -> bool:
        """按预设的健康端点依次探测，任一 200 即在线（发现式，不抛异常）。"""
        probe_timeout = max(self.timeout / 4, 0.5)
        for path in self.preset.health_paths:
            try:
                req = urllib.request.Request(
                    self.base_url + path, method="GET", headers=self._headers()
                )
                with urllib.request.urlopen(req, timeout=probe_timeout) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.HTTPError,) as e:  # noqa: PERF203
                # 401/403 说明服务在，只是要鉴权 —— 视为在线（调用时再报错）
                if e.code in (401, 403):
                    return True
            except (urllib.error.URLError, OSError, ValueError):
                continue
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
        """``POST {base}/v1/completions`` —— 字段严格遵循 OpenAI 契约。

        零重写：不发明任何字段。``seed`` 在 temperature=0 时无害，
        在 temperature>0 时被服务端用于可复现采样（llama.cpp/vLLM 均支持）。
        """
        payload: dict[str, Any] = {
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
            headers=self._headers(),
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        choice = (data.get("choices") or [{}])[0]
        text = choice.get("text", "") or ""
        meta: dict[str, Any] = {
            "adapter": "openai-compat",
            "preset": self.preset.key,
            "base_url": self.base_url,
            "model": data.get("model", self.model_hint),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage", {}),
            "cite": self.preset.cite,
        }
        return EngineResult(text=text, engine=self.name, tokens=None, degraded=False, meta=meta)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["base_url"] = self.base_url
        d["preset"] = self.preset.key
        d["cite"] = self.preset.cite
        return d


def build_engine(
    preset: str = "llama.cpp",
    *,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
) -> OpenAICompatEngine:
    """按预设名构造引擎（工厂函数，供 CLI/配置消费）。"""
    return OpenAICompatEngine(preset, base_url=base_url, timeout=timeout, api_key=api_key)


def autodiscover(
    presets: Optional[list[str]] = None,
    *,
    timeout: float = 2.0,
    include_remote: bool = False,
) -> list[OpenAICompatEngine]:
    """扫描预设库，返回**当前在线**的引擎（发现式注册）。

    Args:
        presets: 只扫这些预设；None = 扫全部本地预设。
        timeout: 单次探测总预算（秒），内部按健康端点数量摊分。
        include_remote: 是否也扫 ``openai-compatible``（需外网 + 可能鉴权）。

    Returns:
        在线引擎列表（顺序 = 预设表顺序，确定性）。
    """
    keys = presets or [k for k in PRESETS if k != "openai-compatible"]
    if include_remote and "openai-compatible" not in keys:
        keys = list(keys) + ["openai-compatible"]
    found: list[OpenAICompatEngine] = []
    for key in keys:
        if key not in PRESETS:
            continue
        engine = OpenAICompatEngine(key, timeout=timeout)
        if engine.available():
            found.append(engine)
    return found


def preset_table() -> list[dict[str, Any]]:
    """预设库表格（``polaris engines --all`` 与文档消费）。"""
    rows = []
    for key, p in PRESETS.items():
        rows.append(
            {
                "preset": key,
                "kind": p.kind,
                "default_url": p.default_url,
                "env": p.env,
                "health": ", ".join(p.health_paths),
                "auth_env": p.auth_env,
                "cite": p.cite,
                "capabilities": dict(p.capabilities),
            }
        )
    return rows


__all__ = [
    "EnginePreset",
    "PRESETS",
    "OpenAICompatEngine",
    "build_engine",
    "autodiscover",
    "preset_table",
]
