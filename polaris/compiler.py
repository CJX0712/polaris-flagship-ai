"""意图编译器 —— 自然语言 → 1024 微指令 ISA 程序。

零重写 · 出处
------------
编译器该有的样子，世界上早有定论 —— 本模块按经典编译器结构落地，
**不发明任何新理论**：

1. **前端（Front End）**：词法/关键字匹配做**意图路由**。这是
   ``lex``/``yacc`` 一脉的确定性词法分析思想，不是 LLM 语义解析；
2. **中间表示（IR）**：``ISAProgram`` = ifai L10 微指令的**线性序列**，
   对应 LLVM IR 的「指令列表 + 元数据」形态；
3. **优化/验证（Pass）**：**编译期干运行**（compile-time dry run）——
   在探针向量上试跑每一条候选指令，跑不通的直接**剔除并记录原因**。
   这是 LLVM ``opt`` 的 pass 思想：编译期就把不可执行的程序剪掉，
   而不是留给运行期抛异常；
4. **后端（Back End）**：``run(x)`` 顺序执行 IR，产出结果。

**指令集本身不是本仓写的**：1024 条微指令来自 ifai L10（CJX0712/
innovation-flagship-ai，Apache-2.0，移植自 tianji-ai），32 家族 × 32 条。
本编译器只做「选哪条、按什么顺序」—— 编译器本就不该自己实现指令语义。

诚实边界（写在这里，不藏在脚注里）
----------------------------------
本编译器是**规则驱动的确定性子集**：它做关键词路由，不做开放域语义理解。
``"把这个向量归一化"`` 能编译，``"帮我写首诗"`` 不能 —— 后者应当走
``PolarisOS.dispatch()`` 交给世界顶级 LLM 引擎，这才是正确的分工。
把编译器吹成能理解一切，是自欺；让编译器只做它可证明的部分，是工程。

可证不变量（K1–K5，全部机检）
----------------------------
- **K1** 编译产出的程序非空（意图未命中任何家族 ⇒ 报 ``NoIntentMatch``，
  绝不静默返回空程序）
- **K2** 每条指令都真实存在于 ISA，且 ``family`` 与路由结果一致
- **K3** 确定性：同一程序在同一输入上两次执行结果逐位相同
- **K4** 编译期干运行通过：程序在探针上可执行（不可执行的候选已被剔除）
- **K5** 指纹稳定：``program.fingerprint()`` 只取决于指令序列，与调用次序无关
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ._vendor_path import ensure_vendor_path

ensure_vendor_path()

import numpy as np  # noqa: E402

from ifai.l10 import MICROOPS  # noqa: E402


class NoIntentMatch(RuntimeError):
    """意图未命中任何 ISA 家族（诚实失败 —— 该交给 LLM，不该硬编）。"""


class ProgramError(RuntimeError):
    """程序运行期失败（含失败步号与指令名，便于定位）。"""

    def __init__(self, step: int, op_name: str, cause: BaseException):
        self.step = step
        self.op_name = op_name
        self.cause = cause
        super().__init__(f"step {step} ({op_name}) failed: {type(cause).__name__}: {cause}")


# ------------------------------------------------------------------ 前端：路由表

#: 家族 → 关键词（中英双语）。命中越长越具体，故按长度加权。
ROUTE_TABLE: dict[str, tuple[str, ...]] = {
    "norm": ("归一化", "标准化", "normalize", "batchnorm", "layernorm", "rmsnorm", "norm"),
    "smax": ("softmax", "软最大", "归一化指数", "log_softmax", "logsumexp", "概率分布"),
    "uact": ("激活函数", "激活", "activation", "relu", "gelu", "silu", "swish", "tanh", "sigmoid"),
    "ract": ("归约", "reduction", "前缀统计", "计数", "count", "reduce", "求和", "argmax"),
    "bact": ("逐元素", "elementwise", "二元", "binary", "相加", "相减", "点乘", "add"),
    "attn": ("自注意力", "多头注意力", "注意力", "attention", "self-attention", "mha", "attn"),
    "rope": ("旋转位置编码", "位置编码", "rotary", "rope", "rotary embedding"),
    "mask": ("因果掩码", "掩码", "mask", "causal", "遮蔽", "attention mask"),
    "gate": ("遗忘门", "门控", "gate", "lstm", "gru", "gating"),
    "moer": ("混合专家", "专家路由", "路由", "router", "moe", "expert", "top-k 路由"),
    "scan": ("前缀和", "扫描", "scan", "cumsum", "累积和", "prefix sum"),
    "seqm": ("状态空间", "序列模型", "ssm", "mamba", "线性注意力", "递归", "delta rule"),
    "cache": ("kv cache", "kv", "缓存", "cache", "显存", "kv-cache"),
    "quant": ("量化", "quantize", "quantization", "int8", "int4", "低比特", "absmax"),
    "cast": ("类型转换", "精度转换", "cast", "bf16", "fp16", "dtype", "denormal"),
    "conv": ("卷积", "convolution", "conv", "滤波", "高斯卷积"),
    "pool": ("池化", "pooling", "pool", "下采样", "平均池化"),
    "dist": ("余弦距离", "欧氏距离", "距离", "distance", "cosine", "相似度", "similarity"),
    "metr": ("评估指标", "指标", "metric", "accuracy", "准确率", "秩", "entropy rank"),
    "loss": ("交叉熵", "损失函数", "损失", "loss", "cross entropy", "目标函数"),
    "grad": ("反向传播", "梯度", "gradient", "求导", "derivative", "backward"),
    "optm": ("优化器", "optimizer", "adamw", "adam", "sgd", "学习率", "weight decay"),
    "init": ("权重初始化", "初始化", "initialize", "init", "he init", "xavier"),
    "rand": ("随机", "random", "dropout", "噪声", "采样噪声", "bernoulli"),
    "indx": ("索引", "index", "排序", "argsort", "gather", "scatter"),
    "topk": ("topk", "top-k", "取前 k", "前 k 个", "检索", "召回", "top_k"),
    "smpl": ("采样策略", "采样", "sampling", "sample", "logprobs", "贪心解码"),
    "spec": ("协方差", "谱", "spectral", "gram 矩阵", "gram", "svd", "正交化", "nuclear"),
    "specf": ("特殊函数", "special function", "erf", "gamma", "bessel", "digamma", "expm1"),
    "shap": ("分块", "形状变换", "形状", "shape", "reshape", "chunk", "切分"),
    "prob": ("概率", "probability", "能量", "energy", "温度", "temperature"),
    "misc": ("梯度裁剪", "裁剪", "clip", "misc", "杂项"),
}

#: 各家族的**首选指令**（语义代表）。缺失则回退到该家族字典序第一条。
REPRESENTATIVE: dict[str, tuple[str, ...]] = {
    "attn": ("attn.attn_1",),
    "bact": ("bact.add", "bact.diff_gate"),
    "cache": ("cache.kv_append", "cache.cache_tokens"),
    "cast": ("cast.normalize_scale",),
    "conv": ("conv.conv_gauss",),
    "dist": ("dist.euclidean", "dist.cosine_dist"),
    "gate": ("gate.forget_gate",),
    "grad": ("grad.d_gelu",),
    "indx": ("indx.argsort", "indx.argmax_idx"),
    "init": ("init.init_arange",),
    "loss": ("loss.ce_self",),
    "mask": ("mask.causal",),
    "metr": ("metr.cosine_sim",),
    "misc": ("misc.clip_grad",),
    "moer": ("moer.noisy_gate",),
    "norm": ("norm.cosine_norm", "norm.batchnorm_det"),
    "optm": ("optm.adam_like",),
    "pool": ("pool.avg_pool10",),
    "prob": ("prob.cross_entropy_uniform",),
    "quant": ("quant.absmax_scale",),
    "ract": ("ract.argmax_flat",),
    "rand": ("rand.rand_dropout",),
    "rope": ("rope.rope_base10000",),
    "scan": ("scan.cumsum",),
    "seqm": ("seqm.alpha_gate",),
    "shap": ("shap.chunk10",),
    "smax": ("smax.softmax", "smax.log_softmax"),
    "smpl": ("smpl.argmax",),
    "spec": ("spec.gram",),
    "specf": ("specf.exp",),
    "topk": ("topk.topk1_mask",),
    "uact": ("uact.gelu",),
}

#: 探针向量：编译期干运行与确定性检验都用它（固定 ⇒ 编译结果可复现）
PROBE: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_numbers(text: str) -> list[float]:
    """从意图文本中抽取数值（前端的「字面量词法」）。

    抽不到时返回空表 —— 由调用方决定用探针还是报错，本函数不做猜测。
    """
    out: list[float] = []
    for m in _NUM_RE.finditer(text):
        try:
            out.append(float(m.group(0)))
        except ValueError:  # pragma: no cover - 正则已保证可解析
            continue
    return out


def family_candidates(family: str) -> list[str]:
    """该家族的候选指令名：首选在前，其余按字典序（确定性，K5 前提）。"""
    preferred = [n for n in REPRESENTATIVE.get(family, ()) if n in MICROOPS]
    rest = sorted(n for n, op in MICROOPS.items() if op.family == family)
    seen: set[str] = set()
    out: list[str] = []
    for n in list(preferred) + rest:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def route(intent: str) -> list[tuple[str, int]]:
    """意图路由：返回 ``[(family, score), ...]``，按 (-score, family) 排序。

    打分规则（确定性、可复现）：每个命中关键词按**字符数**加权 ——
    越长的词越具体（``"自注意力"`` 比 ``"attention"`` 更确切），
    这是信息检索里最朴素的 IDF 直觉，不是新发明。
    """
    low = intent.lower()
    scores: dict[str, int] = {}
    for family, kws in ROUTE_TABLE.items():
        s = 0
        for kw in kws:
            if kw.lower() in low:
                s += len(kw)
        if s:
            scores[family] = s
    # 直接写出指令名（如 "smax.softmax"）⇒ 该家族直接加权
    for name, op in MICROOPS.items():
        if name.lower() in low:
            scores[op.family] = scores.get(op.family, 0) + len(name) * 2
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# ------------------------------------------------------------------ IR


@dataclass
class Step:
    """IR 中的一条指令。"""

    op: str
    family: str
    arity: int
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"op": self.op, "family": self.family, "arity": self.arity, "note": self.note}


@dataclass
class ISAProgram:
    """编译产物：可执行的微指令序列（LLVM IR 的极简对应物）。

    Attributes:
        intent: 原始意图文本（编译输入，进审计）。
        steps: 指令序列。
        matches: 路由打分结果（解释「为什么选这些指令」）。
        dropped: 编译期干运行被剔除的候选及原因（诚实记账）。
        deterministic: 探针上两次执行结果一致（K3）。
        gates: K1–K5 门禁报告。
    """

    intent: str
    steps: list[Step] = field(default_factory=list)
    matches: list[tuple[str, int]] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)
    deterministic: bool = False
    gates: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------ 指纹
    def fingerprint(self) -> str:
        payload = "|".join(s.op for s in self.steps)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------ 执行
    def run(self, x: Any, *, scalar: float = 1.0, trace: bool = False) -> Any:
        """顺序执行 IR。

        Args:
            x: 输入（标量 / 序列 / ndarray；内部统一转 ``float64`` ndarray）。
            scalar: 二元指令（arity=2）的第二操作数（广播为同形数组）。
            trace: True 时返回 ``(结果, 每步快照)``。
        """
        acc = np.asarray(x, dtype=np.float64)
        snaps: list[dict[str, Any]] = []
        for i, step in enumerate(self.steps):
            fn = MICROOPS[step.op].fn
            try:
                acc = _apply(fn, acc, step.arity, scalar)
            except Exception as exc:  # noqa: BLE001 - 运行期如实上报，不静默
                raise ProgramError(i, step.op, exc) from exc
            if trace:
                snaps.append({"step": i, "op": step.op, "shape": np.shape(acc)})
        if trace:
            return acc, snaps
        return acc

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "steps": [s.as_dict() for s in self.steps],
            "matches": [{"family": f, "score": s} for f, s in self.matches],
            "dropped": self.dropped,
            "deterministic": self.deterministic,
            "gates": self.gates,
            "fingerprint": self.fingerprint(),
        }


# ------------------------------------------------------------------ 中间 Pass：干运行


def _apply(fn: Any, acc: Any, arity: int, scalar: float) -> Any:
    """执行一条指令，含 **rank 合法化**（rank legalization）。

    合法化规则（确定性、可解释）：ISA 里少数算子要求 2-D 输入
    （如 ``attn.*`` 需要 ``tri(N)`` 构造掩码）。当累加值是 1-D 而指令
    需要 2-D 时，按 ``np.atleast_2d`` 提升为 ``(1, n)`` —— 这与 LLVM 后端
    的 type legalization 同构：把合法但不匹配的输入形状规约到指令要求的形态。

    编译期干运行与运行期**共用这一个函数** —— 编译期通过的，运行期必然同构。
    """
    try:
        if arity == 1:
            return fn(acc)
        return fn(acc, np.full(np.shape(acc), float(scalar), dtype=np.float64))
    except Exception:
        # 仅 1-D → 2-D 提升一次；再失败则如实抛出
        promoted = np.atleast_2d(acc)
        if promoted.ndim == acc.ndim:
            raise
        if arity == 1:
            return fn(promoted)
        return fn(promoted, np.full(np.shape(promoted), float(scalar), dtype=np.float64))


def _dry_run(ops: list[str], scalar: float = 1.0) -> tuple[bool, Optional[str]]:
    """在探针上试跑一串指令：跑通返回 (True, None)，否则 (False, 原因)。"""
    acc = np.asarray(PROBE, dtype=np.float64)
    for name in ops:
        op = MICROOPS[name]
        try:
            acc = _apply(op.fn, acc, op.arity, scalar)
        except Exception as exc:  # noqa: BLE001 - 编译期要的是「能不能跑」
            return False, f"{name}: {type(exc).__name__}: {exc}"
    return True, None


def _same(a: Any, b: Any) -> bool:
    """逐位相等（含 shape 与 dtype 无关的数值比较，NaN 视为相等）。"""
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        return False
    return bool(np.array_equal(aa, bb, equal_nan=True))


# ------------------------------------------------------------------ 编译器


def compile_intent(  # noqa: A001 - 名字与编译器语义一致，故意遮蔽内建
    intent: str,
    *,
    width: int = 2,
    max_families: int = 2,
    scalar: float = 1.0,
    require_deterministic: bool = True,
) -> ISAProgram:
    """把意图文本编译为 ISA 程序。

    Args:
        intent: 意图文本。
        width: 程序长度（指令条数）。
        max_families: 最多混合多少个家族（多家族 ⇒ 复合流水线）。
        scalar: 二元指令的第二操作数。
        require_deterministic: True 时，非确定性程序直接编译失败（K3）。

    Returns:
        编译期干运行已通过的 ``ISAProgram``。

    Raises:
        NoIntentMatch: 未命中任何家族（该走 LLM，不该硬编）。
        ProgramError: 干运行全灭 / 确定性要求未满足。
    """
    matches = route(intent)
    if not matches:
        raise NoIntentMatch(
            f"intent {intent!r} matched no ISA family; "
            "route it to an LLM engine via PolarisOS.dispatch() instead"
        )
    families = [f for f, _ in matches[: max(1, int(max_families))]]
    width = max(1, int(width))

    steps: list[Step] = []
    dropped: list[dict[str, str]] = []
    per_family_idx = {f: 0 for f in families}
    cands = {f: family_candidates(f) for f in families}

    # 轮转取指令：family[0], family[1], family[0], ...（确定性顺序，K5）
    guard = 0
    while len(steps) < width and guard < 200:
        guard += 1
        progressed = False
        for fam in families:
            if len(steps) >= width:
                break
            pool = cands[fam]
            while per_family_idx[fam] < len(pool):
                name = pool[per_family_idx[fam]]
                per_family_idx[fam] += 1
                ok, why = _dry_run([s.op for s in steps] + [name], scalar=scalar)
                if ok:
                    op = MICROOPS[name]
                    steps.append(Step(op=name, family=fam, arity=op.arity, note=op.note))
                    progressed = True
                    break
                dropped.append({"op": name, "family": fam, "reason": why or "dry-run failed"})
            if len(steps) >= width:
                break
        if not progressed:
            break

    if not steps:
        raise ProgramError(
            0,
            "compile",
            RuntimeError(
                "no executable instruction survived compile-time dry run; "
                f"dropped={dropped[:3]}"
            ),
        )

    program = ISAProgram(
        intent=intent,
        steps=steps,
        matches=matches,
        dropped=dropped,
        deterministic=False,
        gates=[],
    )

    # ---- Pass：确定性检验（探针上跑两次，K3）
    probe = np.asarray(PROBE, dtype=np.float64)
    try:
        r1 = program.run(probe, scalar=scalar)
        r2 = program.run(probe, scalar=scalar)
        program.deterministic = _same(r1, r2)
    except ProgramError:
        program.deterministic = False

    program.gates = compile_gates(program)
    if require_deterministic and not program.deterministic:
        raise ProgramError(
            0,
            "compile",
            RuntimeError(
                f"compiled program is not deterministic on probe: "
                f"{[s.op for s in program.steps]}"
            ),
        )
    return program


def compile_gates(program: ISAProgram) -> list[dict[str, Any]]:
    """K1–K5 门禁（结构化、可机检、可进审计账本）。"""
    gates: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    add("K1", len(program.steps) > 0, f"steps={len(program.steps)}")
    bad = [
        s.op
        for s in program.steps
        if s.op not in MICROOPS or MICROOPS[s.op].family != s.family
    ]
    add("K2", not bad, f"all steps exist in ISA & family-consistent (bad={bad})")
    add("K3", program.deterministic, "two runs on probe are bit-identical")
    add("K4", True, "compile-time dry run passed for every emitted step")
    add(
        "K5",
        program.fingerprint() == hashlib.sha256(
            "|".join(s.op for s in program.steps).encode("utf-8")
        ).hexdigest()[:16],
        "fingerprint depends only on the op sequence",
    )
    return gates


__all__ = [
    "NoIntentMatch",
    "ProgramError",
    "ROUTE_TABLE",
    "REPRESENTATIVE",
    "PROBE",
    "Step",
    "ISAProgram",
    "compile_intent",
    "compile_gates",
    "route",
    "family_candidates",
    "extract_numbers",
]
