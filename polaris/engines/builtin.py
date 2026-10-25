"""内置旗舰引擎 —— ifai 全栈的确定性回退。

零重写 · 出处
------------
「内置引擎」= 同门旗舰 **ifai**（innovation-flagship-ai v0.8.0，
Apache-2.0，作者晨星，vendor 于 ``third_party/ifai/``）经 Polaris 契约
薄封装后的形态。本模块**不重写任何模型/训练/采样代码**：

- 有模型目录 → 直接驱动 ``ifai.sdk.IFAI``（其生成路径与服务端同源）；
- 无模型目录 → 调用 ifai 自己的训练命令 ``ifai.cli.cmd_train``
  现场训练一个微型模型（ifai 官方 demo 的同款做法，MICRO_CORPUS 语料）；
- 两者皆不可用 → 诚实返回 ``degraded=True`` 的确定性占位文本。

设计动机（NEXUS 的铁律）：**系统永远必须能跑**。外部世界级引擎
（llama.cpp）缺席时，内置旗舰保证 Polaris 离线可运行、可复现、可验证。
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

from .._vendor_path import ensure_vendor_path

ensure_vendor_path()

from ifai.sdk import IFAI  # noqa: E402  （vendor 路径注入后再导入）

from .base import EngineAdapter, EngineResult  # noqa: E402


def _default_model_dir() -> Optional[str]:
    """默认模型目录：工作区下已存在 ``.polaris_model`` 就直接复用。

    这一条很实际：现场训练约 16 秒，每次 CLI 调用重训一遍是纯浪费。
    已训练过 ⇒ 直接加载；没有 ⇒ 返回 None，由 ``_ensure_model`` 现场训练。
    """
    path = os.path.join(os.getcwd(), ".polaris_model")
    if os.path.exists(os.path.join(path, "manifest.json")):
        return path
    return None


class BuiltinEngineAdapter(EngineAdapter):
    """ifai 内置引擎适配器（确定性回退保证）。

    Args:
        model_dir: 已训练模型目录（含 manifest.json）。None 时优先复用
            ``./.polaris_model``，仍无则按需现场训练。
        seed: 全局种子（经 ifai 全局 RNG，保证逐位可复现）。
        allow_autotrain: 无模型时是否现场训练微型模型（默认允许）。
        train_steps: 现场训练步数（默认 120，CPU 秒级）。
    """

    name = "ifai-builtin"
    kind = "builtin-flagship"
    capabilities = {
        "modalities": ["text"],
        "deterministic": True,
        "maxTokens": 512,
        "trained": False,
    }

    def __init__(
        self,
        model_dir: Optional[str] = None,
        seed: int = 0,
        allow_autotrain: bool = True,
        train_steps: int = 120,
    ):
        # 能力声明按实例拷贝 —— 类属性是全局共享的，就地改会污染其他实例
        self.capabilities = dict(self.capabilities)
        if model_dir is not None:
            self.model_dir = model_dir
        elif allow_autotrain:
            # 允许训练时才自动拾取已有模型（避免重复训练）；否则尊重「不训练」语义
            self.model_dir = _default_model_dir()
        else:
            self.model_dir = None  # 无模型且禁止训练 ⇒ 诚实占位，绝不静默加载
        self.seed = int(seed)
        self.allow_autotrain = allow_autotrain
        self.train_steps = int(train_steps)
        self._ifai: Optional[IFAI] = None
        self._autotrained = False

    # ------------------------------------------------------------------ 发现
    def available(self) -> bool:
        """内置引擎永远可用 —— 它是系统「永远能跑」的最后保证。"""
        return True

    # ------------------------------------------------------------------ 训练
    def _ensure_model(self) -> Optional[IFAI]:
        """确保有一个可用的 IFAI 实例（无模型则按需现场训练）。

        零重写：训练直接调用 ifai 官方的 ``cmd_train``（与 ``ifai train``
        同一实现），本模块只负责编排「何时训练」。
        """
        if self._ifai is not None:
            return self._ifai
        if self.model_dir and os.path.exists(os.path.join(self.model_dir, "manifest.json")):
            self._ifai = IFAI(model_dir=self.model_dir, seed=self.seed)
            return self._ifai
        if self.allow_autotrain:
            from ifai.cli import cmd_train  # ifai 官方训练命令，零重写

            out = self.model_dir or os.path.join(os.getcwd(), ".polaris_model")
            self.model_dir = out
            os.makedirs(out, exist_ok=True)
            cmd_train(
                argparse.Namespace(
                    seed=self.seed, corpus=None, vocab=512, d_model=128,
                    layers=3, heads=4, kv_heads=2, ffn_mult=4,
                    seq_len=128, steps=self.train_steps, lr=3e-3, out=out,
                )
            )
            self._ifai = IFAI(model_dir=out, seed=self.seed)
            self._autotrained = True
            self.capabilities["trained"] = True
            return self._ifai
        return None

    # ------------------------------------------------------------------ 适配
    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> EngineResult:
        """经 ifai SDK 生成（temperature=0 走贪心，确定性可复现）。"""
        ai = self._ensure_model()
        if ai is None:
            return EngineResult(
                text="[polaris] 内置引擎未训练且不允许现场训练；请先 `polaris train-tiny`。",
                engine=self.name,
                degraded=True,
                meta={"adapter": "ifai-builtin", "reason": "no-model"},
            )
        text = ai.chat(prompt, max_new=max_tokens, temperature=temperature)
        return EngineResult(
            text=text,
            engine=self.name,
            tokens=None,
            degraded=False,
            meta={
                "adapter": "ifai-builtin",
                "model_dir": self.model_dir,
                "autotrained": self._autotrained,
                "seed": seed,
            },
        )


__all__ = ["BuiltinEngineAdapter"]
