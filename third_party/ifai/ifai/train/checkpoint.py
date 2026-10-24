"""L4 梯度检查点 (Gradient / Activation Checkpointing)

零重写 (集大成):
- **Chen et al. 2016** "Training Deep Nets with Sublinear Memory Cost": 前向
  只保存分段边界 (检查点) 的激活, 反向传播到该段时**重算**段内激活, 以
  ``O(√n)`` (或分段常数) 的激活内存换取一次额外的前向。
- 工程范式沿用 **HuggingFace Transformers** 的 ``gradient_checkpointing_enable``
  与 **torch.utils.checkpoint** 的语义: ``checkpoint(fn, *args)`` 返回一个
  "数值相同但已脱离原图"的张量, 其反向钩子负责重算。
- 反向传播本身仍是 **reverse-mode autodiff** (Griewank & Walther 2008),
  复用 L4 :mod:`ifai.train.autodiff` 的 :class:`Tensor`。

可验证不变量:
- **INV-25 梯度检查点: loss/grad ≡ 无检查点** (``diff ≤ 1e-5``)。
  等价性来源: 被检查点的段是**确定性**函数 (无 dropout / 无随机分支), 重算
  的前向与原始前向逐位相同, 因此重算图的反向梯度与原始图严格一致。

用法::

    from ifai.train.checkpoint import checkpoint
    y = checkpoint(lambda t: self._layer_forward(t, 0), x)   # 丢弃段内激活
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from .autodiff import Tensor


def _topo_from(root: Tensor) -> list:
    """拓扑排序: 保证子节点先于父节点被反向调用。"""
    topo: list = []
    visited = set()

    def build(v: Tensor) -> None:
        if v in visited:
            return
        visited.add(v)
        for ch in v._prev:
            build(ch)
        topo.append(v)
    build(root)
    return topo


def backward_from(root: Tensor, grad: np.ndarray) -> None:
    """从 ``root`` 以给定上游梯度做**局部**反向 (不重置全局图)。

    这是重算路径的关键: 重算出的子图与外层图只共享"段输入"与"参数",
    因此只应对子图调用各节点的 ``_backward``, 让梯度累加到共享节点上。
    """
    grad = np.asarray(grad, dtype=np.float64)
    root.grad = np.broadcast_to(grad, root.data.shape).astype(np.float64).copy()
    for node in reversed(_topo_from(root)):
        node._backward()


def checkpoint(fn: Callable[..., Tensor], *args, enabled: bool = True) -> Tensor:
    """以检查点方式执行 ``fn(*args)``: 前向丢弃段内激活, 反向时重算。

    Args:
        fn: 被检查点的确定性函数 (不得含随机性/副作用), 返回 :class:`Tensor`。
        *args: 段输入 (Tensor 或常量); 保存下来供重算使用。
        enabled: False 时退化为直接调用 (用于对照实验, INV-25)。

    Returns:
        与 ``fn(*args)`` 数值相同的 :class:`Tensor`; 该 Tensor 是新叶子节点,
        反向时触发一次段内重算。
    """
    if not enabled:
        return fn(*args)
    # 1) 前向: 仍要算出输出数值 (这部分无法省), 但段内中间激活随后被丢弃
    out = fn(*args)
    if not isinstance(out, Tensor):
        raise TypeError("checkpoint 要求 fn 返回 ifai.train.autodiff.Tensor")
    # 2) 切断计算图: 新叶子节点只携带数值
    y = Tensor(np.array(out.data, dtype=np.float64, copy=True),
               requires_grad=out.requires_grad)
    y._prev = []

    def _recompute_backward() -> None:
        # 3) 反向: 用保存的输入重算前向, 得到局部图后从 y.grad 反向
        recomputed = fn(*args)
        backward_from(recomputed, y.grad)

    y._backward = _recompute_backward
    return y


def checkpointed(fn: Callable[..., Tensor] | None = None, *, enabled: bool = True):
    """装饰器版本: ``@checkpointed`` 令整个函数体走检查点路径。

    用法::

        @checkpointed
        def block(x):
            ...
    """
    def wrap(f: Callable[..., Tensor]) -> Callable[..., Tensor]:
        def inner(*args, **kwargs):
            if kwargs:
                raise TypeError("checkpointed 装饰器不支持关键字参数 (请用 functools.partial)")
            return checkpoint(f, *args, enabled=enabled)
        inner.__name__ = getattr(f, "__name__", "checkpointed")
        inner.__doc__ = f.__doc__
        return inner
    if fn is not None:
        return wrap(fn)
    return wrap


def grad_diff(a: Dict[str, Tensor], b: Dict[str, Tensor]) -> float:
    """两组同名参数梯度的最大绝对差 (INV-25 判据之一)。

    缺失梯度视为 0; 返回 ``inf`` 表示形状不一致。
    """
    worst = 0.0
    for key in sorted(set(a) | set(b)):
        ta, tb = a.get(key), b.get(key)
        if ta is None or tb is None:
            return float("inf")
        ga = np.zeros_like(ta.data) if ta.grad is None else np.asarray(ta.grad)
        gb = np.zeros_like(tb.data) if tb.grad is None else np.asarray(tb.grad)
        if ga.shape != gb.shape:
            return float("inf")
        if ga.size:
            worst = max(worst, float(np.max(np.abs(ga - gb))))
    return worst


def verify_checkpoint(loss_fn: Callable[[bool], Tensor],
                      params_fn: Callable[[], Dict[str, Tensor]],
                      tol: float = 1e-5) -> Tuple[bool, dict]:
    """INV-25 一键自检: 开/关检查点两次前向反向, 对比 loss 与梯度。

    Args:
        loss_fn: ``loss_fn(enabled: bool) -> Tensor``, 内部决定是否用检查点,
            且**必须重新构建计算图** (每次调用都是全新前向)。
        params_fn: 返回当前参数字典 (用于取梯度)。
        tol: 判据阈值, 默认 1e-5 (PRD INV-25)。

    Returns:
        ``(ok, report)``; ``report`` 含 ``loss_plain`` / ``loss_ckpt`` /
        ``loss_diff`` / ``grad_diff`` / ``tol``。
    """
    def zero_grad(params: Dict[str, Tensor]) -> None:
        for p in params.values():
            if p.requires_grad and p.grad is not None:
                p.grad = np.zeros_like(p.data)

    plain_params = params_fn()
    zero_grad(plain_params)
    loss_plain_t = loss_fn(False)
    loss_plain_t.backward()
    loss_plain = float(loss_plain_t.data)
    plain_snapshot = {k: np.array(v.grad) if v.grad is not None else np.zeros_like(v.data)
                      for k, v in plain_params.items()}

    ckpt_params = params_fn()
    zero_grad(ckpt_params)
    loss_ckpt_t = loss_fn(True)
    loss_ckpt_t.backward()
    loss_ckpt = float(loss_ckpt_t.data)
    ckpt_snapshot = {k: np.array(v.grad) if v.grad is not None else np.zeros_like(v.data)
                     for k, v in ckpt_params.items()}

    gd = 0.0
    for key in sorted(set(plain_snapshot) | set(ckpt_snapshot)):
        ga, gb = plain_snapshot.get(key), ckpt_snapshot.get(key)
        if ga is None or gb is None or ga.shape != gb.shape:
            gd = float("inf")
            break
        if ga.size:
            gd = max(gd, float(np.max(np.abs(ga - gb))))
    ld = abs(loss_plain - loss_ckpt)
    ok = bool(ld <= tol and gd <= tol)
    return ok, {"loss_plain": loss_plain, "loss_ckpt": loss_ckpt,
                "loss_diff": ld, "grad_diff": gd, "tol": tol}


def checkpoint_segments(n_items: int, n_segments: int) -> Sequence[Tuple[int, int]]:
    """把 ``n_items`` 项 (如层) 均分为 ``n_segments`` 个检查点段, 返回区间列表。

    零重写: 即 Chen et al. 2016 的 "sqrt(n) 段" 策略的可配置版本 —— 每段只
    保存边界激活, 段数越多内存越省、重算越多。
    """
    if n_items <= 0:
        return []
    n_segments = max(1, min(int(n_segments), n_items))
    bounds = [round(i * n_items / n_segments) for i in range(n_segments + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(n_segments) if bounds[i] < bounds[i + 1]]
