"""L0/L1 量化器 (Quantization) — RTN per-channel / per-token

零重写:
- LLM.int8 (Dettmers et al. 2022) 的 per-channel symmetric 量化思想
- RTN (round-to-nearest) 作为 INT8/INT4 基线量化，GPTQ (Frantar et al. 2022)
  与 AWQ (Lin et al. 2023) 在本栈作为可选进阶, 见 ``ifai/models`` 的进阶注释。

可验证不变量:
- INV-03 重建误差 ≤ scale/2 + 1e-6 (round-to-nearest 的解析误差界)
- INV-07 (L0-03) INT8 per-channel 重建误差 ≤ scale/2
- R-L1-02 int4 pack 逐位一致 (pack ↔ unpack 往返 equals)
"""
from __future__ import annotations

import numpy as np


def quantize_int8_per_channel(x: np.ndarray, axis: int = -1):
    """对称 per-channel RTN 量化到 INT8。

    返回 (qint8, scale) — scale 为每通道的绝对值最大值 / 127。
    """
    x = np.asarray(x, dtype=np.float64)
    max_abs = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = np.where(max_abs == 0, 1.0, max_abs) / 127.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float64)


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * np.asarray(scale, dtype=np.float64)


def quantize_int4_per_channel(x: np.ndarray, axis: int = -1):
    """对称 per-channel RTN 量化到 INT4 (范围 [-7, 7], 15 电平)。

    返回 (q, scale) — q 以 int8 容器存放 int4 值 ([-7,7]); scale 为每通道
    绝对值最大值 / 7。与 ``pack_int4`` 配合可获得 4-bit 存储 (先量化后打包)。
    重建误差界同 INV-07 形式: |x − dequant| ≤ scale/2。
    """
    x = np.asarray(x, dtype=np.float64)
    max_abs = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = np.where(max_abs == 0, 1.0, max_abs) / 7.0
    q = np.clip(np.round(x / scale), -7, 7).astype(np.int8)
    return q, scale.astype(np.float64)


def dequantize_int4(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * np.asarray(scale, dtype=np.float64)


def pack_int4(x: np.ndarray) -> np.ndarray:
    """把 int8 范围 [-8,7] 的数组打包成 uint8 (每字节两个数, 低4位先)。"""
    x = np.asarray(x, dtype=np.int8).ravel()
    x = np.clip(x, -8, 7).astype(np.int8)
    pad = (len(x) + 1) // 2 * 2 - len(x)
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=np.int8)])
    lo = x[0::2].astype(np.uint8) & 0x0F
    hi = x[1::2].astype(np.uint8) & 0x0F
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_int4(packed: np.ndarray, n: int | None = None) -> np.ndarray:
    """解包 int4 (符号扩展回 int8 范围 [-8,7])。"""
    p = np.asarray(packed, dtype=np.uint8)
    lo = (p & 0x0F).astype(np.int8)
    hi = ((p >> 4) & 0x0F).astype(np.int8)
    x = np.empty(len(p) * 2, dtype=np.int8)
    x[0::2] = lo
    x[1::2] = hi
    # 符号扩展: 0b1000(=8) 以上视为负 → 减去 16
    x = np.where(x >= 8, x - 16, x).astype(np.int8)
    return x[:n] if n is not None else x


def quantize_fp16(x: np.ndarray):
    """模拟 fp16 往返 (numpy 无纯 fp16 计算，这里用 float16 dtype 模拟量化误差)。"""
    x = np.asarray(x, dtype=np.float32)
    return x.astype(np.float16).astype(np.float32)
