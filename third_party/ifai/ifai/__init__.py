"""创新旗舰 AI · Innovation Flagship AI

十层架构、零重写集大成、CPU 可运行可复现可验证的旗舰级 AI 系统。

作者: 晨星 (Morningstar) / GitHub CJX0712
许可: Apache-2.0
"""
from __future__ import annotations

__version__ = "0.8.0"
__author__ = "晨星 (Morningstar)"

from . import rng  # noqa: F401  全局统一随机源

__all__ = ["__version__", "__author__", "rng"]
