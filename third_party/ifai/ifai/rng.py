"""L0/LX 全局统一随机源 (Global RNG)

零重写原则: 复用 NumPy 官方 ``numpy.random.Generator`` (PCG64, O'Neill 2014)，
不重写 PRNG。所有模块通过 ``get_rng()`` 取用同一全局实例，保证 ``--seed N``
两次运行 bitwise 一致 (INV-40)。

可验证: ``seed(0); a=get_rng().random(); seed(0); b=get_rng().random()`` ⇒ a==b。
"""
from __future__ import annotations

import numpy as np

_SEED: int | None = None
_RNG: np.random.Generator | None = None


def seed(value: int = 0) -> None:
    """设置全局随机种子并复位所有派生源。"""
    global _SEED, _RNG
    _SEED = int(value)
    _RNG = np.random.default_rng(_SEED)


def get_rng() -> np.random.Generator:
    """返回全局 ``Generator``；未初始化时默认 seed(0)。"""
    global _RNG
    if _RNG is None:
        seed(0)
    assert _RNG is not None
    return _RNG


def get_seed() -> int | None:
    """返回当前种子 (None 表示未设置)。"""
    return _SEED


def spawn(child_seed: int) -> np.random.Generator:
    """派生一个独立的子随机源 (用于并行/隔离场景)，不影响全局源。"""
    return np.random.default_rng((_SEED if _SEED is not None else 0) ^ int(child_seed))
