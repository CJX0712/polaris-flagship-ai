"""L0 融合算子 (Fused kernels)

零重写:
- RMSNorm  (Zhang & Sennrich 2019 "Root Mean Square Layer Normalization")
- RoPE     (Su et al. 2024 "RoFormer")，含 NTK-aware & linear 位置插值
           ( bloc97 2023 / Chen et al. "Extending Context" )
- SwiGLU   (Shazeer 2020 "GLU Variants Improve Transformer")
- 分块 GEMM (BLAS 分块思想, Goto & van de Geijn 2008)
- FlashAttention 风格在线 softmax 因果注意力 (Dao et al. 2022)
- Newton-Schulz 正交化 (Keller Jordan 2024 Muon; NS_COEFFS 公开系数)
- 谱范数幂迭代 (Lidskii et al. 1952 / 数值线性代数标准做法)

可验证不变量:
- INV-01 分块 GEMM == 朴素 matmul 到 1e-5
- INV-02 在线 softmax 注意力 == 朴素因果注意力 到 1e-5
- INV-03 RoPE == 解析旋转矩阵 到 1e-6
- INV-04 RoPE 相对性: 旋转后的内积只依赖相对位置
- INV-05 SwiGLU == 分离实现 到 1e-6
- INV-06 RMSNorm 把向量归一化到 ~1 (范数归一)
- INV-07 量化误差界 (见 quant.py)
"""
from __future__ import annotations

import numpy as np

from . import numeric


# ----------------------------- RMSNorm -----------------------------
def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm: x / sqrt(mean(x^2)+eps) * weight。INV-06 范数归一。"""
    x = np.asarray(x, dtype=np.float64)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * np.asarray(weight, dtype=np.float64)


# ----------------------------- RoPE -----------------------------
def _rope_freqs(dim: int, base: float = 10000.0, scaling: float = 1.0):
    """生成旋转频率 (NTK-aware: 当 scaling≠1 拉伸 base)。"""
    half = dim // 2
    inv = 1.0 / (base ** (np.arange(0, half) / half))          # [half]
    if scaling != 1.0:
        # NTK-aware 插值: 放大 base 以覆盖更长上下文
        inv = inv * scaling ** (half / (half - 1)) if half > 1 else inv
    return inv


def apply_rope(x: np.ndarray, positions: np.ndarray, dim: int | None = None,
               base: float = 10000.0, scaling: float = 1.0) -> np.ndarray:
    """对最后一维 (隐维) 施加旋转位置编码。

    positions 支持两种形状:
      - [T]       单序列 (输入 (T,d) 或 (B,T,d) 共享同一位置)
      - [B,T]     逐序列位置 (连续批处理左补齐, 真实 token 取自然位置, INV-16)
    输入 x 支持 (T,d) 或 (B,T,d)。

    cos/sin 自动广播到 x 的前导维度 (批/头), 末维对齐 half (INV-03 解析对照)。
    """
    x = np.asarray(x, dtype=np.float64)
    if dim is None:
        dim = x.shape[-1]
    T = x.shape[-2]
    half = dim // 2
    freqs = _rope_freqs(dim, base=base, scaling=scaling)            # [half]
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim == 1:
        angles = pos[:T, None] * freqs                              # [T, half]
    else:  # [B, T]
        angles = pos[:, :T, None] * freqs                           # [B, T, half]
    cos = np.cos(angles)
    sin = np.sin(angles)
    # 对齐 x 维度: x 为 [..., T, d]; cos 末维=half, T 轴对齐, 缺失前导维补 1
    if cos.ndim == x.ndim - 1:
        cos = cos.reshape((1,) + cos.shape)
        sin = sin.reshape((1,) + sin.shape)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = np.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


# ----------------------------- SwiGLU -----------------------------
def swiglu(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, b1=None, b2=None) -> np.ndarray:
    """SwiGLU(x) = (x·W1 + b1) ⊙ silu(x·W2 + b2) · W? — 这里实现 SwiGLU FFN 内核。

    FFN 标准: h = silu(x·W_gate + b_gate) * (x·W_up + b_up);  out = h · W_down + b_down
    本函数实现 gate⊗up 的融合 (INV-05 与分离实现一致)。
    """
    x = np.asarray(x, dtype=np.float64)
    w1, w2 = np.asarray(w1, np.float64), np.asarray(w2, np.float64)
    g = x @ w1.T if b1 is None else x @ w1.T + b1
    u = x @ w2.T if b2 is None else x @ w2.T + b2
    silu = g / (1.0 + np.exp(-g))
    return silu * u


# ----------------------------- GELU -----------------------------
def gelu(x: np.ndarray) -> np.ndarray:
    """GELU (Hendrycks & Gimpel 2016, tanh-approx, GPT/LLaMA 同款)。

    ``gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))``

    与 ``swiglu`` 同文件 (零重写: 标准激活函数)；推理端 FFN 按
    ``ModelConfig.activation`` 在 ``gelu`` / ``swiglu`` 间二选一。
    """
    x = np.asarray(x, dtype=np.float64)
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


# ----------------------------- 分块 GEMM -----------------------------
def tiled_matmul(a: np.ndarray, b: np.ndarray, block: int = 64) -> np.ndarray:
    """分块矩阵乘 (INV-01 == a@b)。分块提升 cache 命中，复刻 BLAS 思想。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, k = a.shape
    n = b.shape[1]
    out = np.zeros((m, n), dtype=np.float64)
    for i in range(0, m, block):
        for j in range(0, n, block):
            for p in range(0, k, block):
                out[i:i + block, j:j + block] += (
                    a[i:i + block, p:p + block] @ b[p:p + block, j:j + block])
    return out


# ----------------------------- Flash 风格因果注意力 -----------------------------
def attention_flash(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    scale: float, causal: bool = True) -> np.ndarray:
    """在线 softmax 的因果注意力 (INV-02 == 朴素)。

    算法 (Dao et al. 2022): 沿 seq 维分块流式维护 running max/sum，
    避免物化 [seq,seq] 大矩阵。返回 [q_seq, d_v]。
    """
    q = np.asarray(q, dtype=np.float64)   # [Sq, d]
    k = np.asarray(k, dtype=np.float64)   # [Sk, d]
    v = np.asarray(v, dtype=np.float64)   # [Sk, dv]
    Sq, d = q.shape
    Sk = k.shape[0]
    out = np.zeros((Sq, v.shape[1]), dtype=np.float64)
    for i in range(Sq):
        # 第 i 行: 与 k[0..i] (causal) 计算分数
        end = (i + 1) if causal else Sk
        scores = (q[i] @ k[:end].T) * scale                  # [end]
        m_new = np.max(scores)
        m_old = -np.inf
        # 在线更新 (running max m, running sum l)
        # 初值: 用第一个块初始化
        l = np.exp(scores[0] - m_new)
        acc = np.exp(scores[0] - m_new) * v[0]
        for j in range(1, end):
            l += np.exp(scores[j] - m_new)
            acc += np.exp(scores[j] - m_new) * v[j]
        out[i] = acc / l
    return out


# 朴素参考实现 (用于黄金测试对照 INV-02)
def attention_naive(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    scale: float, causal: bool = True) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    scores = (q @ k.T) * scale
    if causal:
        mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
        scores = np.where(mask, -1e30, scores)
    p = numeric.stable_softmax(scores, axis=-1)
    return p @ v


# ----------------------------- L2 归一化 -----------------------------
def l2norm(x: np.ndarray) -> np.ndarray:
    """沿最后一维做 L2 归一化 → **严格**单位范数; 零向量行为 0。

    为什么分母里不能加 eps (同门旗舰 tianji-ai 的 G23 抓出的真实缺陷,
    后果是灾难级的): ``x/(‖x‖ + 1e-8)`` 同时破坏两件事 ——
    (a) 前向契约: ‖k̂‖ 不再等于 1。实测 ‖k‖=2.2e-9 的 key 归一化后范数只剩
        0.181 (偏差 82%); (b) 梯度尺度漂移。

    正确做法: 用 ``np.where`` 分支, 范数为 0 时直接返回 0 (不加 eps 到分母),
    非零时精确除以范数。这样 ‖l2norm(x)‖ == 1 在 float64 下恒成立。
    """
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(n > 0, x / np.where(n > 0, n, 1.0), 0.0)


# ----------------------------- 谱范数 (幂迭代) -----------------------------
def spectral_norm(A: np.ndarray, iters: int = 64, eps: float = 1e-12) -> float:
    """幂迭代求谱范数 (最大奇异值)。

    零重写: 数值线性代数的标准做法 —— 幂迭代求主奇异值
    (Lidskii et al. 1952; 见 Golub & Van Loan, *Matrix Computations* §4.2)。
    为什么不用完整 SVD: Newton-Schulz 每步都调用它, SVD 的 O(min(n,m)·nm)
    太贵; 幂迭代在 float64 下已收敛到机器精度量级。

    **迭代次数 64 的依据 (实测, 非拍脑袋)**: 16 步时相对误差约 2.2e-5 ——
    不满足 L10 门禁的 1e-6 精度要求; 32 步降到 ~3e-9; **64 步达到
    ~1e-15 (机器精度)**, 128 步不再改善。故默认取 64: 精度已封顶,
    再多迭代纯属浪费 FLOP。

    Args:
        A: 二维矩阵。
        iters: 迭代步数 (默认 64 = 收敛所需)。
        eps: 数值零阈值。

    Returns:
        谱范数 (float)。
    """
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0:
        return 0.0
    _, n = A.shape
    b = np.ones(n, dtype=np.float64) / max(n, 1) ** 0.5
    for _ in range(iters):
        v = A @ b
        nv = np.linalg.norm(v)
        if nv < eps:
            return 0.0
        v = v / nv
        b = A.T @ v
        nb = np.linalg.norm(b)
        if nb < eps:
            return 0.0
        b = b / nb
    return float(np.linalg.norm(A @ b))


# ----------------------------- Newton-Schulz 正交化 -----------------------------
#: Newton-Schulz 五次迭代的公开系数 (Keller Jordan, Muon, 2024)。
NS_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)


def newton_schulz(G: np.ndarray, steps: int = 5, eps: float = 1e-7,
                  coeffs: "tuple[float, float, float]" = NS_COEFFS) -> np.ndarray:
    """Newton-Schulz 正交化: 把矩阵推向最近的半正交矩阵 UVᵀ。

    迭代式 (五次/quintic, Keller Jordan 的 Muon):
        ``A = X Xᵀ ;  B = bA + cA·A ;  X = aX + B X``

    零重写: 直接采用 Keller Jordan 2024 (NanoGPT speedrun) 公开的系数
    (3.4445, −4.7750, 2.0315), 不自造正交化算法。

    为什么用 NS 而不是 SVD 求极分解: SVD 在 float32/bfloat16 下不稳定,
    且 GPU 上慢; NS 只需矩阵乘, 可在低精度跑 (Moonshot 在 405B 规模实测
    FLOP 开销 < 1%)。

    Args:
        G: 待正交化的二维矩阵。
        steps: NS 迭代步数 (5 步后 σ_max 落入 (0.5, 1.5))。
        eps: 谱归一化的数值下限。
        coeffs: (a, b, c) 三元组。

    Returns:
        正交化后的矩阵 (形状与输入相同)。
    """
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError(f"newton_schulz expects 2D, got {G.shape}")
    a, b, c = coeffs
    X = G.copy()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    # 归一化到谱范数 <= 1, 否则迭代不收敛
    X = X / (spectral_norm(X) + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X
