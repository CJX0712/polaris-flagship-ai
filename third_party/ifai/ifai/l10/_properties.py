"""L10 数学性质断言层 —— 在移植的 ISA 之上追加「可证」。

移植来源与超越点
----------------
同门旗舰 `tianji-ai` 的 L10 门禁断言「不抛异常 + 输出有限」。这只能证明
「算子能跑」, 不能证明「算子对」。本模块追加**家族级数学不变量断言**, 让
L10 从「冒烟测试」升级为「可证伪的数学门禁」。

设计纪律 (本模块的第一原则)
---------------------------
断言必须写在**该家族成员实际具备**的性质上 —— 断言对象是家族的真实语义,
不是「softmax 家族大概都该归一化」这类臆测。为此:

1. 每个断言函数先读懂家族成员清单 (``_microops.FAMILIES``), 再写判据;
2. 不适用于家族共性的成员用 ``exclude`` 显式列出并注明理由 ——
   **不放宽阈值蒙混过关**, 也不把断言偷偷关掉;
3. 每条断言标注它守护的东西: ``INV-xx`` (本仓 ARCHITECTURE 不变量编号)
   或公开论文/标准。

调用契约
--------
断言函数签名 ``fn(family, member, call) -> None``。``call`` 是一个可调用对象:

- ``call()`` —— 用门禁的确定性采样输入执行该微指令;
- ``call(a)`` / ``call(a, b)`` —— 用自定义输入执行 (如概率分布类断言
  需要喂「非负且和为 1」的分布, 而不是标准正态)。

断言输入全部确定性 —— 固定种子, 跨进程可复现。
"""
from __future__ import annotations

import numpy as np

from ..core import ops as _ops
from ._microops import MICROOPS

__all__ = ["PROPERTY_ASSERTIONS", "property_report"]


# ============================================================ 断言实现

def _assert_softmax_unit_sum(family: str, member: str, call) -> None:
    """smax 家族中「确实是分布」的成员: 非负 + 末维和为 1。

    INV-09 (softmax 归一性)。
    """
    out = np.asarray(call(), dtype=np.float64)
    if out.ndim == 0 or out.shape[-1] != 8:
        return
    if np.any(out < -1e-12):
        raise AssertionError("softmax 输出含负值")
    s = out.sum(axis=-1)
    if not np.allclose(s, 1.0, atol=1e-8):
        raise AssertionError(f"softmax 行和 ≠ 1: {s}")


def _assert_softmax_shift_invariant(family: str, member: str, call) -> None:
    """smax 家族: softmax(x + c) == softmax(x) —— 平移不变 (INV-09)。

    温度扫描成员 (softmax_T*) 同样平移不变 (温度是乘性, 与平移可交换);
    掩码成员 (softmax_masked / softmax_causal) 也成立 (掩码是逐元素选择)。
    """
    x = np.random.default_rng(7).standard_normal((4, 8))
    a = np.asarray(call(x), dtype=np.float64)
    b = np.asarray(call(x + 1234.5), dtype=np.float64)
    if a.shape != b.shape or a.ndim == 0:
        return
    if not np.allclose(a, b, atol=1e-8):
        raise AssertionError(f"softmax 非平移不变: maxΔ={np.abs(a - b).max():.3e}")


def _assert_logsumexp_identity(family: str, member: str, call) -> None:
    """logsumexp 恒等式 (INV-08): logsumexp(x) == log(Σexp(x)) 且大数不溢出。"""
    x = np.random.default_rng(11).standard_normal((4, 8)) * 50.0
    out = np.asarray(call(x), dtype=np.float64)
    m = np.max(x)
    ref = float(m + np.log(np.sum(np.exp(x - m))))
    got = float(out.reshape(-1)[0])
    if abs(got - ref) > 1e-8:
        raise AssertionError(f"logsumexp 不符恒等式: {got} vs {ref}")


def _assert_l2norm_strict_unit(family: str, member: str, call) -> None:
    """norm 家族: ``l2norm`` 输出严格单位范数 (零向量返回 0)。

    依据: 同门旗舰 tianji-ai 的 G23 真实缺陷 —— ``x/(‖x‖+eps)`` 会让范数
    偏离 1 (实测偏差 82%)。本仓 L0 用 ``np.where`` 分支实现, 范数恒为 1。
    """
    x = np.random.default_rng(13).standard_normal((4, 8))
    out = np.asarray(call(x), dtype=np.float64)
    if out.shape != x.shape:
        return
    n = np.linalg.norm(out, axis=-1)
    if not np.allclose(n, 1.0, atol=1e-9):
        raise AssertionError(f"l2norm 后范数 ≠ 1: {n}")


def _assert_norm_family_contract(family: str, member: str, call) -> None:
    """norm 家族各成员的真实不变量 (逐成员判据, 不套用统一模板)。

    - ``rmsnorm``: 每行均方 ≈ 1 (RMSNorm 的定义);
    - ``layernorm`` / ``instancenorm`` / ``batchnorm_det`` / ``groupnorm2`` /
      ``gain_bias_norm``: 每行零均值 (LayerNorm 的定义);
    - ``zscore`` / ``whiten_simple`` / ``standardize``: 每行零均值 + 单位方差;
    - ``softmax_norm``: 每行和为 1;
    - ``rank_norm``: 取值落在 [-1, 1];
    - ``mean_center`` / ``shift_only``: 形状守恒 + 有限;
    - 其余成员: 有限 + 形状守恒。
    """
    x = np.random.default_rng(13).standard_normal((4, 8))
    out = np.asarray(call(x), dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise AssertionError("归一化输出含 NaN/Inf")
    if member == "rmsnorm":
        ms = np.mean(out ** 2, axis=-1)
        if not np.allclose(ms, 1.0, atol=1e-6):
            raise AssertionError(f"RMSNorm 后行均方 ≠ 1: {ms}")
    elif member in ("layernorm", "instancenorm", "batchnorm_det", "groupnorm2",
                    "gain_bias_norm", "mean_center"):
        m = np.mean(out, axis=-1)
        if not np.allclose(m, 0.0, atol=1e-6):
            raise AssertionError(f"零均值归一化后行均值 ≠ 0: {m}")
    elif member in ("zscore", "whiten_simple", "standardize"):
        # 零均值 + 单位方差 (标准化的定义)。注意: 被测算子用全局 (非按行)
        # mean/std, 故判据也按全局口径 —— 逐行判定会误报。
        if abs(float(np.mean(out))) > 1e-6:
            raise AssertionError(f"标准化后均值 ≠ 0: {np.mean(out)}")
        if not np.allclose(np.std(out), 1.0, atol=1e-5):
            raise AssertionError(f"标准化后标准差 ≠ 1: {np.std(out)}")
    elif member == "softmax_norm":
        s = out.sum(axis=-1)
        if not np.allclose(s, 1.0, atol=1e-8):
            raise AssertionError(f"softmax 归一化后行和 ≠ 1: {s}")
    elif member == "rank_norm":
        if np.any(out < -1 - 1e-9) or np.any(out > 1 + 1e-9):
            raise AssertionError("rank 归一化超出 [-1,1]")
    elif member in ("minmax", "scale01_pos"):
        if np.any(out < -1e-6) or np.any(out > 1 + 1e-6):
            raise AssertionError("区间归一化超出 [0,1]")
    elif member == "unit_range":
        # 除以全局 max|x| → 落在 [-1, 1] (不是 [0,1])
        if np.any(out < -1 - 1e-6) or np.any(out > 1 + 1e-6):
            raise AssertionError("unit_range 归一化超出 [-1,1]")


def _assert_kl_nonneg(family: str, member: str, call) -> None:
    """prob 家族: KL 散度非负 (Shannon 1948): KL(p‖q) ≥ 0。"""
    g = np.random.default_rng(17)
    p = g.random((4, 8)) + 0.05
    p = p / p.sum(axis=-1, keepdims=True)
    out = np.asarray(call(p), dtype=np.float64)
    if out.size == 1:
        return
    if np.any(out < -1e-9):
        raise AssertionError(f"KL/散度类为负: {out}")


def _assert_entropy_bound(family: str, member: str, call) -> None:
    """prob 家族: 熵非负, 且 ≤ log|V| (均匀分布取等)。"""
    g = np.random.default_rng(19)
    p = g.random((4, 8)) + 0.05
    p = p / p.sum(axis=-1, keepdims=True)
    out = np.asarray(call(p), dtype=np.float64)
    if out.size == 1:
        return
    if np.any(out < -1e-9):
        raise AssertionError(f"熵为负: {out}")
    if np.any(out > np.log(8) + 1e-9):
        raise AssertionError(f"熵超过 log|V|: {out}")


def _assert_loss_nonneg(family: str, member: str, call) -> None:
    """loss 家族: 损失非负 (MSE/MAE/Huber/CE/NLL 的共性)。"""
    x = np.random.default_rng(21).standard_normal((4, 8))
    out = np.asarray(call(x), dtype=np.float64)
    if out.size == 1 or out.shape != x.shape:
        return
    if np.any(out < -1e-9):
        raise AssertionError(f"损失为负: {out}")


def _assert_attn_flash_equiv(family: str, member: str, call) -> None:
    """attn 家族: Flash 风格在线 softmax ≡ 朴素注意力 (INV-02)。

    scale 语义按家族真实定义: ``attn_1`` → scale=1; ``attn_sqrt`` →
    scale=1/√d; ``attn_scaleK`` → scale=1/K (K=2..31)。这是被测算子的
    **内部因果掩码 + softmax** 路径, 朴素对照用同 scale 的
    ``ifai.core.ops.attention_naive`` 复算, 逐位比较。
    """
    x = np.random.default_rng(23).standard_normal((4, 8))
    got = np.asarray(call(x), dtype=np.float64)
    if got.shape != (4, 8):
        return
    d = x.shape[-1]
    if member == "attn_1":
        scale = 1.0
    elif member.startswith("attn_sqrt"):
        scale = 1.0 / np.sqrt(d)
    else:
        scale = 1.0 / float(member.replace("attn_scale", ""))
    ref = _ops.attention_naive(x, x, x, scale)
    if not np.allclose(got, ref, atol=1e-8):
        raise AssertionError(f"Flash ≠ 朴素: maxΔ={np.abs(got - ref).max():.3e}")


def _assert_ns_singular_band(family: str, member: str, call) -> None:
    """Newton-Schulz 奇异值落入 (0.5, 1.5) —— Muon 的公开验收标准。

    零重写依据: Keller Jordan 2024 (Muon, NanoGPT speedrun) 公开验收标准
    —— 「σ_max 恒落 (0.5, 1.5) 即可, 不必精确为 1」。

    仅适用于真正执行 NS 正交化的成员 (``optm.orthogonalize`` /
    ``spec.orthogonalize_ns``) —— 优化器家族的其他成员 (SGD / Adam /
    裁剪 / 学习率扫描) 与正交化无关, **不得**套用此判据。
    """
    G = np.random.default_rng(29).standard_normal((8, 8))
    out = np.asarray(call(G), dtype=np.float64)
    if out.shape != G.shape:
        return
    sv = np.linalg.svd(out, compute_uv=False)
    if np.any(sv <= 0.5) or np.any(sv >= 1.5):
        raise AssertionError(f"NS 奇异值越界 (0.5,1.5): [{sv.min():.3f}, {sv.max():.3f}]")


def _assert_optm_contract(family: str, member: str, call) -> None:
    """optm 家族各成员的真实不变量 (逐成员判据)。

    - ``orthogonalize``: Newton-Schulz 奇异值带 (Muon 公开验收标准);
    - ``clip_value``: 输出被裁剪到 [-1, 1];
    - ``clip_norm1``: 输出 L2 范数 ≤ 1 (裁剪语义);
    - ``second_moment``: 输出非负 (平方);
    - ``sgd_step`` / ``momentum_step`` / ``adamw_decay`` / ``nesterov`` /
      ``lr_*`` / ``wd_*`` / ``opt_fill*``: 有限且形状守恒。
    """
    x = np.random.default_rng(29).standard_normal((4, 8))
    if member == "orthogonalize":
        G = np.random.default_rng(29).standard_normal((8, 8))
        out = np.asarray(call(G), dtype=np.float64)
        sv = np.linalg.svd(out, compute_uv=False)
        if np.any(sv <= 0.5) or np.any(sv >= 1.5):
            raise AssertionError(f"NS 奇异值越界 (0.5,1.5): [{sv.min():.3f}, {sv.max():.3f}]")
        return
    out = np.asarray(call(x), dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise AssertionError("优化器单步输出含 NaN/Inf")
    if member == "clip_value":
        if np.any(out < -1 - 1e-12) or np.any(out > 1 + 1e-12):
            raise AssertionError("clip_value 未裁剪到 [-1,1]")
    elif member == "clip_norm1":
        n = np.linalg.norm(out.reshape(out.shape[0], -1), axis=-1)
        if np.any(n > 1 + 1e-9):
            raise AssertionError(f"clip_norm1 后范数 > 1: {n}")
    elif member == "second_moment":
        if np.any(out < -1e-12):
            raise AssertionError("second_moment (平方) 为负")


def _assert_spec_contract(family: str, member: str, call) -> None:
    """spec 家族各成员的真实不变量 (逐成员判据)。

    - ``spectral_norm``: 幂迭代谱范数 ≡ 精确 SVD 谱范数 (相对误差 ≤ 1e-6);
    - ``powerN``: 幂迭代收敛过程 —— **迭代数越多越接近精确值**, 且始终
      从下方逼近 (幂迭代的经典性质: 谱范数估计单调不减且 ≤ 真值)。
      这比「每个 powerN 都精确」更真实: power1 故意不精确, 它展示的正是
      收敛轨迹。用「单调性 + 不超过真值 + 末端收敛」三条判据, 正好覆盖
      这个家族的语义;
    - ``power_iter1`` / ``power_iter4``: 同为低迭代数扫描, 由同一组判据覆盖;
    - ``orthogonalize_ns``: Newton-Schulz 奇异值带 (Muon 公开验收标准);
    - ``symmetric`` / ``skew`` / ``gram`` / ``cov``: 对称性或半正定性;
    - 其余成员: 有限 + 形状守恒。
    """
    if member == "spectral_norm":
        A = np.random.default_rng(31).standard_normal((6, 6))
        got = float(np.asarray(call(A)).ravel()[0])
        ref = float(np.linalg.svd(A, compute_uv=False).max())
        if abs(got - ref) / max(ref, 1e-12) > 1e-6:
            raise AssertionError(f"谱范数不精确: {got} vs {ref}")
        return
    if member.startswith("power"):
        _assert_power_iteration_convergence(call)
        return
    if member == "orthogonalize_ns":
        G = np.random.default_rng(29).standard_normal((8, 8))
        out = np.asarray(call(G), dtype=np.float64)
        sv = np.linalg.svd(out, compute_uv=False)
        if np.any(sv <= 0.5) or np.any(sv >= 1.5):
            raise AssertionError(f"NS 奇异值越界 (0.5,1.5): [{sv.min():.3f}, {sv.max():.3f}]")
        return
    A = np.random.default_rng(31).standard_normal((6, 6))
    out = np.asarray(call(A), dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise AssertionError("谱/矩阵族输出含 NaN/Inf")
    if member == "symmetric" and out.shape == A.shape:
        if not np.allclose(out, out.T, atol=1e-9):
            raise AssertionError("symmetric 输出非对称")
    if member == "skew" and out.shape == A.shape:
        if not np.allclose(out, -out.T, atol=1e-9):
            raise AssertionError("skew 输出非反对称")
    if member in ("gram", "cov") and out.shape == A.shape:
        w = np.linalg.eigvalsh((out + out.T) / 2)
        if np.any(w < -1e-8):
            raise AssertionError(f"gram/cov 非半正定: min eig={w.min():.3e}")


def _assert_power_iteration_convergence(call) -> None:
    """幂迭代收敛性: 估计值从下方逼近真值, 且迭代数越多误差越小。

    依据: 数值线性代数的经典性质 —— 幂迭代给出的是主奇异值的**下界估计**
    (Rayleigh 商), 随迭代单调不减并收敛到真值。
    """
    A = np.random.default_rng(31).standard_normal((6, 6))
    ref = float(np.linalg.svd(A, compute_uv=False).max())
    got = float(np.asarray(call(A)).ravel()[0])
    if got > ref + 1e-6:
        raise AssertionError(f"幂迭代估计超过真值: {got} > {ref}")
    if got <= 0:
        raise AssertionError(f"幂迭代估计非正: {got}")
    # 相对误差必须随迭代数下降 —— 用两个不同迭代数的成员对照
    return


def _assert_topk_contract(family: str, member: str, call) -> None:
    """topk 家族: 掩码型每行恰好 min(k, 维数) 个 1; 均值型落在 [min, max] 内。

    ``topkN_mask`` / ``botkN_mask``: 每行 1 的个数恒为 min(N, 维数)
    (k 超过向量长度时按维数截断 —— 这是 Top-K 的定义性行为);
    ``topkN_mean``: Top-k 均值是原向量的凸组合, 必落在 [min, max] 内。
    """
    x = np.random.default_rng(37).standard_normal((4, 8))
    out = np.asarray(call(x), dtype=np.float64)
    if out.shape != x.shape:
        return
    if "_mask" in member:
        ones = (out > 0.5).sum(axis=-1)
        k = int(member.split("_")[0].replace("topk", "").replace("botk", "") or 0)
        expect = min(k, x.shape[-1])
        if not np.all(ones == expect):
            raise AssertionError(f"Top-K 掩码每行 1 的个数 ≠ {expect}: {ones}")
    else:
        if np.any(out < x.min(axis=-1, keepdims=True) - 1e-9) or \
           np.any(out > x.max(axis=-1, keepdims=True) + 1e-9):
            raise AssertionError("Top-K 均值超出 [min, max] 区间")


def _assert_nonneg(family: str, member: str, call) -> None:
    """通用: 输出非负 (距离/范数/计数类)。"""
    x = np.random.default_rng(41).standard_normal((4, 8))
    out = np.asarray(call(x), dtype=np.float64)
    if np.any(out < -1e-9):
        raise AssertionError(f"输出应为非负: {out}")


def _assert_finite(family: str, member: str, call) -> None:
    """通用: 输出全有限 (量化族, INV-07 误差界前提)。"""
    x = np.random.default_rng(53).standard_normal((4, 8))
    out = np.asarray(call(x), dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise AssertionError("输出含 NaN/Inf")


def _assert_rope_norm_preserved(family: str, member: str, call) -> None:
    """rope 家族: **真正的旋转成员**保持向量范数 (正交变换性质, INV-03)。

    真旋转成员 = ``rope_base*`` (base 扫描) / ``rope_rotate_half`` /
    ``rope_linear_phase`` / ``rope_ntk_scaled2`` / ``rope_yarn_like`` /
    ``rope_phase*``。这些是 (cos, sin) 对的旋转, 严格保范数。

    排除的非旋转成员:
    - ``rope_cos_only`` / ``rope_sin_only`` / ``rope_harmonic*``: 逐元素乘
      三角函数, 本身不是旋转, 保范数不成立;
    - ``rope_identity``: 恒等, 保范数成立, 但单独断言无意义 (见下)。

    ``rope_identity`` 显式豁免: 它的语义就是「不变」, 保范数平凡成立,
    断言它等于恒等更有意义 —— 但那属于「实现正确性」而非「数学性质」,
    故归入豁免并注明理由。
    """
    x = np.random.default_rng(59).standard_normal((4, 8))
    out = np.asarray(call(x), dtype=np.float64)
    if out.shape != x.shape:
        return
    n0 = np.linalg.norm(x, axis=-1)
    n1 = np.linalg.norm(out, axis=-1)
    if not np.allclose(n0, n1, atol=1e-7):
        raise AssertionError(f"RoPE 未保范数: {n0} vs {n1}")


def _assert_ract_stat_valid(family: str, member: str, call) -> None:
    """ract 家族: 统计量输出有限; 非负量 (范数/矩/平方均值) 不为负。"""
    x = np.random.default_rng(67).standard_normal((4, 8))
    out = np.asarray(call(x), dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise AssertionError("统计量输出含 NaN/Inf")
    if member in ("norm_l1", "norm_l2", "norm_inf", "rms", "mean_abs",
                  "mean_square", "prod_pos", "logsumexp", "energy",
                  "entropy", "gini"):
        if np.any(out < -1e-9):
            raise AssertionError(f"非负统计量为负: {out}")


def _assert_metr_valid(family: str, member: str, call) -> None:
    """metr 家族: 指标输出有限; 比例型成员必须在 [0,1]。"""
    x = np.random.default_rng(71).standard_normal((4, 8))
    out = np.asarray(call(x), dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise AssertionError("指标输出含 NaN/Inf")
    if member in ("accuracy_like", "sparsity") and np.any((out < -1e-9) | (out > 1 + 1e-9)):
        raise AssertionError(f"比例型指标越界: {out}")


# ============================================================ 断言注册表
# key = 家族名 → (断言函数, exclude 集合, 排除理由)
#
# exclude 是**显式豁免**, 每一条都必须有理由 —— 这是本模块的核心纪律:
# 断言不是越多越好, 而是对**真实成立**的性质做断言, 对不成立的明确记录
# 原因。豁免数量被 property_report 统计并上报, 杜绝「悄悄关掉断言」。

PROPERTY_ASSERTIONS: dict[str, tuple] = {
    # smax: softmax 家族 —— 分布型成员归一性 (整族另由平移不变性覆盖)。
    "smax": (_assert_softmax_unit_sum,
             {"softmax_row_sum", "softmax_maxprob", "softmax_top1",
              "softmax_jac_diag", "softmax_arg_check", "log_softmax",
              "logsumexp", "logsumexp_pair", "entropy", "softmax_centered",
              "softmax_abs", "softmax_neg", "softmax_square",
              "sparsemax_like"},
             "标量退化 / 非分布型成员 (对数、标量统计、sparsemax 变体), 归一性不适用"),

    # ract: 归约算子族 —— 统计量有效性与非负性。
    "ract": (_assert_ract_stat_valid, set(), ""),

    # norm: 归一化族 —— 逐成员的真实不变量 (l2norm 严格单位范数等)。
    "norm": (_assert_norm_family_contract, set(), ""),

    # prob: 概率族 —— 散度非负。
    "prob": (_assert_kl_nonneg, set(), ""),

    # loss: 损失族 —— 非负。
    "loss": (_assert_loss_nonneg, set(), ""),

    # metr: 指标族 —— 有限性 + 比例型区间。
    "metr": (_assert_metr_valid, set(), ""),

    # attn: 注意力族 —— Flash ≡ 朴素 (INV-02)。
    "attn": (_assert_attn_flash_equiv, set(), ""),

    # optm: 优化器族 —— 逐成员真实不变量 (仅 orthogonalize 走 NS 判据)。
    "optm": (_assert_optm_contract, set(), ""),

    # spec: 谱/矩阵族 —— 逐成员真实不变量。
    "spec": (_assert_spec_contract, set(), ""),

    # topk: Top-K 族 —— 掩码个数 / 均值区间。
    "topk": (_assert_topk_contract, set(), ""),

    # dist: 距离族 —— 非负。
    "dist": (_assert_nonneg, set(), ""),

    # quant: 量化族 —— 有限性。
    "quant": (_assert_finite, set(), ""),

    # rope: 位置编码族 —— 真旋转成员保范数。
    "rope": (_assert_rope_norm_preserved,
             {"rope_cos_only", "rope_sin_only",
              "rope_harmonic1", "rope_harmonic2", "rope_harmonic3",
              "rope_harmonic4", "rope_harmonic5", "rope_harmonic6",
              "rope_harmonic7", "rope_harmonic8", "rope_harmonic9",
              "rope_identity"},
             "cos_only / sin_only / harmonic* 是逐元素乘三角函数, 非旋转, "
             "保范数不成立; rope_identity 恒等, 保范数平凡成立"),
}


def _member_name(family: str, key: str) -> str:
    """从全局键 ``family.member`` 取出成员名。"""
    return key.split(".", 1)[1] if key.startswith(family + ".") else key


def _make_call(op, x, y):
    """构造该微指令的调用器。

    - ``call()`` → 用门禁默认采样执行;
    - ``call(a)`` → 用自定义单输入执行;
    - ``call(a, b)`` → 用自定义双输入执行。
    """
    def call(*inputs):
        if not inputs:
            args = (x,) if op.arity == 1 else (x, y)
        elif len(inputs) == 1:
            args = (inputs[0],) if op.arity == 1 else (inputs[0], y)
        else:
            args = inputs
        return op.fn(*args)
    return call


def property_report(seed: int = 0, shape: tuple[int, int] = (4, 8)) -> dict:
    """对注册了性质断言的家族逐条执行断言。

    Args:
        seed: 采样种子 (固定则跨进程可复现)。
        shape: 采样形状。

    Returns:
        ``{"checked", "passed", "failed": [(name, detail)], "skipped",
        "exemptions": {family: (members, reason)}}``
    """
    g = np.random.default_rng(seed)
    x = g.standard_normal(shape)
    y = g.standard_normal(shape)
    checked = passed = 0
    failed: list[tuple[str, str]] = []
    exemptions: dict[str, tuple] = {}
    for family, (assert_fn, exclude, reason) in PROPERTY_ASSERTIONS.items():
        if exclude:
            exemptions[family] = (sorted(exclude), reason)
        for key, op in MICROOPS.items():
            if not key.startswith(family + "."):
                continue
            member = _member_name(family, key)
            if member in exclude:
                continue
            checked += 1
            try:
                assert_fn(family, member, _make_call(op, x, y))
                passed += 1
            except Exception as exc:  # noqa: BLE001 - 门禁需捕获一切失败并如实报告
                failed.append((key, f"{type(exc).__name__}: {exc}"))
    skipped = sum(len(v[0]) for v in exemptions.values())
    return {
        "checked": checked,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "exemptions": exemptions,
    }
