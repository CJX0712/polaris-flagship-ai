"""L10 支撑设施 —— 确定性采样器与数学不变量断言。

零重写 · 集大成: 采样器与断言器是 L10 微指令的「测试夹具」层。设计锚定同门
旗舰 ``flagship-ai`` 的 L10-microisa (双种子抽查 + 谓词断言) 与 ``tianji-ai``
的 L10-microsops (家族级性质断言), 迁到 Python/NumPy。

关键约定
--------
1. **确定性**: 全部采样器用固定种子 ``20260924`` 的 ``np.random.Generator``,
   与全局 RNG 解耦 —— 保证同一条微指令在任何机器、任何进程给出同一输入。
2. **语义匹配**: 概率族喂非负分布, 矩阵族喂方阵, SPD 族喂对称正定矩阵 ——
   否则「不变量」只是形式上的检查, 没有数学意义。
3. **断言分级**: 家族级断言 (该家族共有的数学性质) + 成员级断言 (该成员
   独有的性质)。二者都在 ``run_all`` 中执行, 失败即如实报失败。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "SEED",
    "rng",
    "vec",
    "mat",
    "spd",
    "dist",
    "ints",
    "pos_mat",
    "to_text",
    "to_arr",
    "check_close",
    "check_nonneg",
    "check_unit_norm",
    "check_softmax",
    "check_symmetric",
    "check_psd",
    "check_shape_kept",
    "check_identity",
    "check_posdef",
    "check_sorted",
    "check_unit_sum",
    "check_integer",
    "check_in_range",
    "check_corr",
]

#: 全 L10 统一种子 —— 保证跨机器/跨进程可复现。
SEED = 20260924


def rng() -> np.random.Generator:
    """返回固定种子的独立随机源 (不触碰全局状态)。"""
    return np.random.default_rng(SEED)


# ----------------------------- 采样器 -----------------------------

def vec(n: int = 8) -> np.ndarray:
    """标准正态向量。"""
    return rng().standard_normal(n)


def mat(n: int = 4) -> np.ndarray:
    """标准正态方阵。"""
    return rng().standard_normal((n, n))


def spd(n: int = 4) -> np.ndarray:
    """对称正定矩阵 (Cholesky / 二次型 / 谱类算子的前提)。"""
    a = rng().standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def pos_mat(n: int = 4) -> np.ndarray:
    """元素严格为正的矩阵 (log / 概率类逐元素算子的前提)。"""
    return np.abs(rng().standard_normal((n, n))) + 0.5


def dist(n: int = 8) -> np.ndarray:
    """非负且和为 1 的概率分布。"""
    p = rng().random(n) + 0.05
    return p / p.sum()


def ints(n: int = 8, hi: int = 4) -> np.ndarray:
    """整数向量 (索引 / 计数类算子的前提)。"""
    return rng().integers(0, hi, size=n)


# ----------------------------- 断言器 -----------------------------

def check_close(out, ref: float, tol: float) -> None:
    """断言标量输出等于参考值 (容差内)。"""
    got = float(np.asarray(out).ravel()[0])
    if abs(got - ref) > tol:
        raise AssertionError(f"{got:.10g} ≠ {ref:.10g} (容差 {tol:.3g})")


def check_nonneg(out, inputs) -> None:
    """断言输出非负 (熵 / 散度 / 范数的基本性质)。"""
    if np.any(np.asarray(out) < -1e-12):
        raise AssertionError("输出应为非负")


def check_unit_norm(out, inputs) -> None:
    """断言输出末维 L2 范数为 1。"""
    n = np.linalg.norm(np.asarray(out), axis=-1)
    if not np.allclose(n, 1.0, atol=1e-6):
        raise AssertionError(f"归一化后范数 ≠ 1: {n}")


def check_unit_sum(out, inputs) -> None:
    """断言输出末维和为 1 (概率分布族)。"""
    s = np.asarray(out).sum(axis=-1)
    if not np.allclose(s, 1.0, atol=1e-9):
        raise AssertionError(f"和 ≠ 1: {s}")


def check_softmax(out, inputs) -> None:
    """断言 softmax 族: 非负且行和为 1。"""
    o = np.asarray(out)
    if np.any(o < -1e-12):
        raise AssertionError("softmax 输出含负值")
    check_unit_sum(o, inputs)


def check_symmetric(out, inputs) -> None:
    """断言输出矩阵对称。"""
    o = np.asarray(out)
    if o.ndim == 2 and not np.allclose(o, o.T, atol=1e-9):
        raise AssertionError("输出非对称")


def check_psd(out, inputs) -> None:
    """断言输出矩阵半正定 (最小特征值 ≥ -tol)。"""
    o = np.asarray(out)
    if o.ndim == 2 and o.shape[0] == o.shape[1]:
        w = np.linalg.eigvalsh((o + o.T) / 2)
        if np.any(w < -1e-8):
            raise AssertionError(f"输出非半正定: min eig={w.min():.3e}")


def check_posdef(out, inputs) -> None:
    """断言输出矩阵正定 (Cholesky 可分解)。"""
    o = np.asarray(out)
    if o.ndim == 2 and o.shape[0] == o.shape[1]:
        w = np.linalg.eigvalsh((o + o.T) / 2)
        if np.any(w <= 0):
            raise AssertionError(f"输出非正定: min eig={w.min():.3e}")


def check_shape_kept(out, inputs) -> None:
    """断言输出与首个输入同形。"""
    if np.asarray(out).shape != np.asarray(inputs[0]).shape:
        raise AssertionError("形状不守恒")


def check_identity(out, inputs) -> None:
    """断言输出与输入逐位一致 (幂等 / 自反族)。"""
    if not np.array_equal(np.asarray(out), np.asarray(inputs[0])):
        raise AssertionError("输出与输入不逐位一致")


def check_sorted(out, inputs) -> None:
    """断言输出单调不减 (排序族)。"""
    o = np.asarray(out)
    if o.ndim >= 1 and np.any(np.diff(o.ravel()) < -1e-12):
        raise AssertionError("输出非单调不减")


def check_integer(out, inputs) -> None:
    """断言输出为整数值 (索引 / 计数族)。"""
    o = np.asarray(out)
    if not np.allclose(o, np.round(o)):
        raise AssertionError("输出应为整数")


def check_in_range(out, inputs) -> None:
    """断言输出落在 [0, 1]。"""
    o = np.asarray(out)
    if np.any(o < -1e-12) or np.any(o > 1 + 1e-12):
        raise AssertionError("输出超出 [0,1]")


def check_corr(out, inputs) -> None:
    """断言输出为合法相关系数 (|r| ≤ 1)。"""
    o = np.asarray(out)
    if np.any(np.abs(o) > 1 + 1e-9):
        raise AssertionError("相关系数超出 [-1,1]")
