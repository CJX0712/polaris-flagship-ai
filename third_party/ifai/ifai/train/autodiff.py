"""L4 手写自动微分 (Reverse-mode autodiff)

零重写: 复用深度学习标准反向模式自动微分 (Griewank & Walther 2008
"Evaluating Derivatives")；用于 INV-22 手写反向传播 vs 有限差分 校验，
并驱动可训练微型 Transformer 的训练。

实现算子: matmul, add, mul, sub, neg, sigmoid, tanh, relu, exp, log,
mean, sum, transpose, pow, getitem, softmax, repeat。每个算子实现精确
解析梯度。
"""
from __future__ import annotations

import numpy as np

from ..core import quant


def _unbroadcast(grad, shape):
    """把梯度累加值收缩回 ``shape`` (处理 numpy 广播的反向, INV-22)。"""
    grad = np.asarray(grad)
    if grad.ndim > len(shape):
        diff = grad.ndim - len(shape)
        grad = grad.sum(axis=tuple(range(diff)))
    for i in range(len(shape)):
        if grad.ndim > i and shape[i] == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = list(_children)

    @property
    def shape(self):
        return self.data.shape

    def _set(self, val):
        self.data = np.asarray(val, dtype=np.float64)

    # ---- 基础运算 ----
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _b():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _b
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _b():
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _b
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data / other.data, (self, other), "/")

        def _b():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(-self.data / (other.data ** 2) * out.grad, other.data.shape)
        out._backward = _b
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(other.data / self.data, (self, other), "/")

        def _b():
            if self.requires_grad:
                self.grad += _unbroadcast(-other.data / (self.data ** 2) * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad / self.data, other.data.shape)
        out._backward = _b
        return out

    def __neg__(self):
        out = Tensor(-self.data, (self,), "neg")

        def _b():
            if self.requires_grad:
                self.grad -= out.grad
        out._backward = _b
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _b():
            a, b = self.data, other.data
            if self.requires_grad:
                if b.ndim == 1:
                    self.grad += np.outer(out.grad, b)
                else:
                    da = np.matmul(out.grad, np.swapaxes(b, -1, -2))
                    self.grad += _unbroadcast(da, a.shape)
            if other.requires_grad:
                if b.ndim == 1:
                    other.grad += a.T @ out.grad
                else:
                    db = np.matmul(np.swapaxes(a, -1, -2), out.grad)
                    other.grad += _unbroadcast(db, b.shape)
        out._backward = _b
        return out

    def fake_quant_int8_per_channel(self, axis: int = -1) -> "Tensor":
        """STE 伪量化 (QAT, Jacob et al. 2018): 前向 = dequant(quant(W)),
        反向对 W 恒等直通 (straight-through estimator, 忽略 d(scale)/dW)。

        零重写: 复用推理端同一 per-channel RTN 方案 (``core/quant.py``,
        LLM.int8 Dettmers et al. 2022, INV-07) — 训练期与部署期的量化语义
        逐字一致。冻结权重 (requires_grad=False) 自动跳过梯度回传。
        """
        q, scale = quant.quantize_int8_per_channel(self.data, axis=axis)
        dq = quant.dequantize_int8(q, scale)
        out = Tensor(dq, (self,), "fake_quant", requires_grad=True)

        def _b():
            if self.requires_grad:
                self.grad += out.grad          # STE: 恒等直通
        out._backward = _b
        return out

    def transpose(self, axes=None):
        out = Tensor(np.transpose(self.data, axes), (self,), "T")

        def _b():
            if self.requires_grad:
                if axes is None:
                    inv = tuple(reversed(range(self.data.ndim)))
                else:
                    inv = tuple(np.argsort(axes))
                self.grad += np.transpose(out.grad, inv)
        out._backward = _b
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _b():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _b
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _b():
            if self.requires_grad:
                self.grad += s * (1 - s) * out.grad
        out._backward = _b
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _b():
            if self.requires_grad:
                self.grad += (1 - t * t) * out.grad
        out._backward = _b
        return out

    def gelu(self):
        """可微 GELU (Hendrycks & Gimpel 2016, tanh-approx, GPT/LLaMA 同款)。

        前向: ``gelu(x) ≈ 0.5*x*(1 + tanh(u))``,
        其中 ``u = sqrt(2/pi)*(x + 0.044715*x**3)``。

        反向解析式:
        ``gelu'(x) = 0.5*(1+tanh(u)) + 0.5*x*(1-tanh(u)**2)*sqrt(2/pi)*(1+0.134145*x**2)``
        (0.134145 = 3·0.044715, 即 0.044715·x**3 对外层 x 求导的系数)。

        训练端 FFN 按 ``ModelConfig.activation`` 在 gelu / SwiGLU(silu) 间二选一；
        与推理端 ``ifai.core.ops.gelu`` 数值完全一致 (INV: 训练/推理激活函数对齐)。
        """
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        u = c * (x + 0.044715 * x ** 3)
        t = np.tanh(u)
        out_data = 0.5 * x * (1.0 + t)
        out = Tensor(out_data, (self,), "gelu")

        def _b():
            if self.requires_grad:
                dgelu = (0.5 * (1.0 + t)
                         + 0.5 * x * (1.0 - t * t) * c * (1.0 + 0.134145 * x * x))
                self.grad += dgelu * out.grad
        out._backward = _b
        return out

    def relu(self):
        m = self.data > 0
        out = Tensor(np.where(m, self.data, 0.0), (self,), "relu")

        def _b():
            if self.requires_grad:
                self.grad += m * out.grad
        out._backward = _b
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _b():
            if self.requires_grad:
                self.grad += e * out.grad
        out._backward = _b
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _b():
            if self.requires_grad:
                self.grad += out.grad / self.data
        out._backward = _b
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), "mean")

        def _b():
            if self.requires_grad:
                denom = self.data.size if axis is None else self.data.shape[axis]
                if axis is None:
                    self.grad += np.ones_like(self.data) * (out.grad / denom)
                elif keepdims:
                    self.grad += np.broadcast_to(out.grad, self.data.shape) / denom
                else:
                    shape = [1] * self.data.ndim
                    shape[axis] = self.data.shape[axis]
                    self.grad += np.broadcast_to(np.expand_dims(out.grad, axis), self.data.shape) / denom
        out._backward = _b
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum")

        def _b():
            if self.requires_grad:
                if axis is None:
                    self.grad += np.ones_like(self.data) * out.grad
                elif keepdims:
                    self.grad += np.broadcast_to(out.grad, self.data.shape)
                else:
                    self.grad += np.broadcast_to(np.expand_dims(out.grad, axis), self.data.shape)
        out._backward = _b
        return out

    def max(self, axis=None, keepdims=False):
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), (self,), "max")

        def _b():
            if self.requires_grad:
                if axis is None:
                    mx = self.data.max()
                    mask = (self.data == mx)
                    self.grad += mask * out.grad / max(int(mask.sum()), 1)
                else:
                    mx = np.max(self.data, axis=axis, keepdims=True)
                    mask = (self.data == mx)
                    if keepdims:
                        self.grad += mask * np.broadcast_to(out.grad, self.data.shape)
                    else:
                        self.grad += mask * np.broadcast_to(np.expand_dims(out.grad, axis), self.data.shape)
        out._backward = _b
        return out

    def pow(self, p):
        out = Tensor(self.data ** p, (self,), "pow")

        def _b():
            if self.requires_grad:
                self.grad += p * (self.data ** (p - 1)) * out.grad
        out._backward = _b
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), "getitem")

        def _b():
            if self.requires_grad:
                g = np.zeros_like(self.data)
                g[idx] = out.grad
                self.grad += g
        out._backward = _b
        return out

    def softmax(self, axis: int = -1):
        """可微 softmax (数值稳定: max 作为常数 detach)。

        反向解析式 (Bridle 1990 / 标准反向模式): ``dL/dx = p ⊙ (g − Σ(g·p))``。
        掩码位 (输入 -1e30 → p=0) 反向自动归零, 与因果掩码天然兼容。
        """
        m = self.data.max(axis=axis, keepdims=True)
        e = np.exp(self.data - m)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(p, (self,), "softmax")

        def _b():
            if self.requires_grad:
                g = p * (out.grad - (out.grad * p).sum(axis=axis, keepdims=True))
                self.grad += g
        out._backward = _b
        return out

    def repeat(self, repeats: int, axis: int):
        """沿 ``axis`` 把每个元素连续重复 ``repeats`` 次 (np.repeat 语义)。

        反向: 把梯度 reshape 为 ``(…, n, repeats, …)`` 后沿 repeats 轴求和。
        用途: GQA 的 kv 头扩展 (与推理端 ``np.repeat`` 同语义, INV-10)。
        """
        out = Tensor(np.repeat(self.data, repeats, axis=axis), (self,), "repeat")

        def _b():
            if self.requires_grad:
                g = out.grad
                n = self.data.shape[axis]
                new_shape = g.shape[:axis] + (n, repeats) + g.shape[axis + 1:]
                self.grad += g.reshape(new_shape).sum(axis=axis + 1)
        out._backward = _b
        return out

    @staticmethod
    def attention(q: "Tensor", k: "Tensor", v: "Tensor", scale: float,
                  bias: np.ndarray | None = None) -> "Tensor":
        """批量缩放点积注意力 (自定义单节点: BLAS 前向 + 标准解析反向)。

        q/k/v: ``[B,H,T,dh]``; ``bias``: ``[T,T]`` 掩码偏置 (如因果 -1e30)。
        前向: ``scores = (q·kᵀ)·scale + bias; P = softmax(scores); ctx = P·v``。
        反向 (标准 reverse-mode 解析式, Vaswani et al. 2017 / 所有主流框架
        同款): ``dV = Pᵀ·g; dP = g·Vᵀ; dS = P⊙(dP − Σ(dP⊙P));
        dQ = dS·K·scale; dK = dSᵀ·Q·scale``。
        掩码位 P=0 ⇒ dS=0, bias 为常数无需梯度。
        """
        qd, kd, vd = q.data, k.data, v.data
        s = (qd @ kd.swapaxes(-1, -2)) * scale                     # [B,H,T,T]
        if bias is not None:
            s = s + bias
        m = s.max(axis=-1, keepdims=True)
        e = np.exp(s - m)
        p = e / e.sum(axis=-1, keepdims=True)
        ctx = p @ vd
        out = Tensor(ctx, (q, k, v), "attention")

        def _b():
            g = out.grad
            if v.requires_grad:
                v.grad += p.swapaxes(-1, -2) @ g
            if q.requires_grad or k.requires_grad:
                dp = g @ vd.swapaxes(-1, -2)
                ds = p * (dp - (dp * p).sum(axis=-1, keepdims=True))
                if q.requires_grad:
                    q.grad += (ds @ kd) * scale
                if k.requires_grad:
                    k.grad += (ds.swapaxes(-1, -2) @ qd) * scale
        out._backward = _b
        return out

    # ---- 反向传播 ----
    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for ch in v._prev:
                    build(ch)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape), requires_grad=False)

    @staticmethod
    def parameter(data):
        return Tensor(data, requires_grad=True)
