"""L2 Transformer 解码器 — RMSNorm + RoPE + GQA + SwiGLU(+MoE) + 权重绑定

零重写 (集大成):
- Transformer (Vaswani et al. 2017)
- Pre-LN / RMSNorm (Zhang & Sennrich 2019)
- RoPE (Su et al. 2024) + NTK-aware 插值
- GQA 分组查询注意力 (Ainslie et al. 2023, INV-10)
- SwiGLU FFN (Shazeer 2020)
- MoE/Switch 稀疏专家 top-k 路由 + load-balancing (Fedus et al. 2022, INV-14)
- 权重绑定 (Press & Wolf 2017, INV-12)

可验证不变量:
- INV-10 GQA 在 n_kv==n_q 时严格 == MHA
- INV-11 Causal masking: 位置 i 的输出不依赖 j>i 的 token (扰动未来 token,
  之前位置的输出 bitwise 不变; 由 ``forward`` 的因果掩码与 ``verify_causal_mask``
  自检函数共同保证)
- INV-12 输出 embedding 与输入 embedding 是同一对象 (tie)
- INV-14 MoE 路由概率和=1 且 token 分配守恒
- INV-13 参数量精确守恒
- INV-20 分块 prefill ≡ 整段 prefill: ``forward`` 支持 ``past_kv`` / ``return_kv``
  (L3 ``engine.prefill.ChunkedPrefill`` 依赖此接口)
- R-L2-06 权重序列化 + sha256 manifest round-trip 误差=0
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from .. import rng
from ..core import numeric, ops, quant
from .config import ModelConfig


class Transformer:
    def __init__(self, config: ModelConfig):
        self.cfg = config
        self._init_params()

    # ---------------- 初始化 ----------------
    def _normal(self, *shape) -> np.ndarray:
        r = rng.get_rng()
        return (r.standard_normal(shape) * self.cfg.initializer_range).astype(np.float64)

    def _init_params(self):
        c = self.cfg
        V, d = c.vocab_size, c.d_model
        p: Dict[str, np.ndarray] = {}
        p["embed"] = self._normal(V, d)                      # 输入 embedding
        if c.tie_word_embeddings:
            p["lm_head"] = p["embed"]                         # INV-12 同一对象
        else:
            p["lm_head"] = self._normal(V, d)
        p["final_norm"] = np.ones(d, dtype=np.float64)

        n_kv = c.n_kv_heads
        for l in range(c.n_layers):
            pre = f"layer{l}."
            p[pre + "attn_norm"] = np.ones(d, np.float64)
            p[pre + "q_proj"] = self._normal(d, d)
            p[pre + "k_proj"] = self._normal(d, n_kv * c.d_head)
            p[pre + "v_proj"] = self._normal(d, n_kv * c.d_head)
            p[pre + "o_proj"] = self._normal(d, d)
            p[pre + "ffn_norm"] = np.ones(d, np.float64)
            if c.use_lora:
                # LoRA 低秩适配 (Hu et al. 2022, INV-24): 与训练端同公式
                # (W + (α/r)·B@A); B=0 初始 ⇒ 等价基座前向。
                for t, shp in (("q_proj", (d, d)),
                               ("k_proj", (d, n_kv * c.d_head)),
                               ("v_proj", (d, n_kv * c.d_head)),
                               ("o_proj", (d, d))):
                    Wi, Wo = shp   # 投影权重形状 = (in_dim, out_dim)
                    p[pre + f"{t}_lora_a"] = self._normal(c.lora_r, Wi)
                    p[pre + f"{t}_lora_b"] = np.zeros((Wo, c.lora_r), np.float64)  # B=0
            if c.use_moe:
                for e in range(c.n_experts):
                    p[pre + f"e{e}_gate"] = self._normal(d, c.d_ff)
                    p[pre + f"e{e}_up"] = self._normal(d, c.d_ff)
                    p[pre + f"e{e}_down"] = self._normal(c.d_ff, d)
                p[pre + "router"] = self._normal(d, c.n_experts)
            else:
                p[pre + "ffn_gate"] = self._normal(d, c.d_ff)
                p[pre + "ffn_up"] = self._normal(d, c.d_ff)
                p[pre + "ffn_down"] = self._normal(c.d_ff, d)
        self.params = p
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def tie_word_embeddings(self) -> bool:
        return self.cfg.tie_word_embeddings

    # ---------------- 参数量 ----------------
    def count_params(self) -> int:
        """可训练参数量 (INV-13: 与按配置解析计算值精确相等)。

        按**唯一数组**计数: 权重绑定 (INV-12, ``lm_head is embed``) 时同一
        ndarray 只计一次, 否则 tied 模型的参数量会被虚报 ``V·d``。
        """
        seen: Dict[int, int] = {}
        for v in self.params.values():
            seen[id(v)] = v.size
        return int(sum(seen.values()))

    # ---------------- 前向 ----------------
    def forward(self, token_ids: np.ndarray, return_hidden: bool = False,
                position_ids: np.ndarray | None = None,
                key_padding_mask: np.ndarray | None = None,
                past_kv: List[Tuple[np.ndarray, np.ndarray]] | None = None,
                return_kv: bool = False):
        """前向。

        - ``position_ids`` [B,T] 或 [T]: 显式位置 (连续批处理左补齐时, 真实
          token 仍取自然位置, 保证与串行前向逐 token 一致, INV-16)。
        - ``key_padding_mask`` [B,T] bool (True=pad): 屏蔽 pad 位置作为 key,
          使补齐序列的最后真实 token 注意力上下文 == 串行上下文。
        - ``past_kv`` 每层 ``(K, V)``, 形状 ``[B, P, n_kv, d_head]``: 分块
          prefill / 增量 decode 的历史 KV (INV-20)。历史 key 的绝对位置假定
          为 ``0..P-1``; 传入时必须给出当前块的绝对 ``position_ids`` (未给出
          时自动取 ``P..P+T-1``)。
        - ``return_kv`` True 时返回 ``(输出, present_kv)``, ``present_kv`` 为
          每层拼接历史后的 ``(K, V)`` 列表 — 供 L3 ``ChunkedPrefill`` 逐块传递。

        因果性 (INV-11): 无论是否使用 ``past_kv``, 任何 query 只能注意
        ``key 绝对位置 <= query 绝对位置`` 的 key, 因此位置 i 的输出与
        ``j > i`` 的 token 完全无关。
        """
        c = self.cfg
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None]
        B, T = ids.shape
        d, dh, n_h, n_kv = c.d_model, c.d_head, c.n_heads, c.n_kv_heads
        scale = 1.0 / (dh ** 0.5)
        if position_ids is None:
            # 有历史 KV 时, 当前块的自然位置从 P 开始 (分块 prefill, INV-20)
            start = int(past_kv[0][0].shape[1]) if (past_kv is not None and len(past_kv)) else 0
            positions = np.arange(start, start + T)
        else:
            positions = np.asarray(position_ids)

        x = self.params["embed"][ids]                       # [B,T,d]
        present: List[Tuple[np.ndarray, np.ndarray]] = []
        for l in range(c.n_layers):
            pre = f"layer{l}."
            h = ops.rms_norm(x, self.params[pre + "attn_norm"], c.norm_eps)
            q = self._apply_lora(h, pre, "q_proj", h @ self.params[pre + "q_proj"])            # [B,T,d]
            k = self._apply_lora(h, pre, "k_proj", h @ self.params[pre + "k_proj"])
            v = self._apply_lora(h, pre, "v_proj", h @ self.params[pre + "v_proj"])
            q = ops.apply_rope(q, positions, d, c.rope_base, c.rope_scaling).reshape(B, T, n_h, dh)
            k = ops.apply_rope(k, positions, n_kv * dh, c.rope_base, c.rope_scaling).reshape(B, T, n_kv, dh)
            v = v.reshape(B, T, n_kv, dh)
            if past_kv is not None:
                # 分块 prefill: 历史 K/V 拼在当前块之前 (INV-20 与整段 prefill 等价)
                pk, pv = past_kv[l]
                k = np.concatenate([pk, k], axis=1)          # [B,P+T,n_kv,dh]
                v = np.concatenate([pv, v], axis=1)
            if return_kv:
                present.append((k, v))
            # GQA: 重复 kv 头 (INV-10 当 n_kv==n_h 时无变化)
            k = np.repeat(k, c.n_repeats, axis=2)
            v = np.repeat(v, c.n_repeats, axis=2)
            # 注意力 [B,n_h,T,Tk] 因果 + pad 屏蔽 (INV-11)
            scores = np.einsum("bthd,bThd->bhtT", q, k) * scale
            if past_kv is not None and k.shape[1] != T:
                # 显式"key 绝对位置 <= query 绝对位置"掩码 (等价因果, 支持历史 KV)
                scores = np.where(self._allow_mask(positions, B, T, k.shape[1]),
                                  scores, -1e30)
            else:
                causal = np.triu(np.ones((T, T), bool), k=1)
                scores = np.where(causal, -1e30, scores)
            if key_padding_mask is not None:
                # [B,1,1,Tk]: True=pad 位置不作为 key 被注意
                pad_mask = np.asarray(key_padding_mask)[:, None, None, :]
                if pad_mask.shape[-1] < scores.shape[-1]:
                    pad = np.zeros((B, 1, 1, scores.shape[-1] - pad_mask.shape[-1]), dtype=bool)
                    pad_mask = np.concatenate([pad, pad_mask], axis=-1)
                scores = np.where(pad_mask, -1e30, scores)
            probs = numeric.stable_softmax(scores, axis=-1)
            ctx = np.einsum("bhtT,bThd->bthd", probs, v).reshape(B, T, d)
            o = ctx @ self.params[pre + "o_proj"]
            o = self._apply_lora(ctx, pre, "o_proj", o)
            x = x + o
            # FFN / MoE
            h2 = ops.rms_norm(x, self.params[pre + "ffn_norm"], c.norm_eps)
            if c.use_moe:
                x = x + self._moe(h2, pre, c)
            else:
                g = h2 @ self.params[pre + "ffn_gate"]
                u = h2 @ self.params[pre + "ffn_up"]
                # 按 activation 选择门控非线性 (INV: config.activation 真正生效)
                gate_act = ops.gelu(g) if c.activation == "gelu" else g / (1.0 + np.exp(-g))
                ff = (gate_act * u) @ self.params[pre + "ffn_down"]
                x = x + ff
        x = ops.rms_norm(x, self.params["final_norm"], c.norm_eps)
        out = x if return_hidden else x @ self.params["lm_head"].T   # [B,T,V]
        if return_kv:
            return out, present
        return out

    @staticmethod
    def _allow_mask(positions: np.ndarray, B: int, T: int, T_k: int) -> np.ndarray:
        """构造 ``[B|1, 1, T, T_k]`` 的可见性掩码: key 绝对位置 <= query 绝对位置。

        INV-11 的显式形式 — 历史 key 位于 ``0..P-1`` 一律可见 (P <= 任意 query
        位置), 当前块内部仍为上三角因果。
        """
        pos = np.asarray(positions, dtype=np.int64)
        P = T_k - T
        if pos.ndim == 1:
            allow_cur = pos[:, None] >= pos[None, :]                     # [T,T]
            allow_past = np.ones((T, P), dtype=bool) if P > 0 else \
                np.zeros((T, 0), dtype=bool)
            allow = np.concatenate([allow_past, allow_cur], axis=-1)      # [T,Tk]
            return allow[None, None, :, :]
        allow_cur = pos[:, :, None] >= pos[:, None, :]                    # [B,T,T]
        allow_past = np.ones((B, T, P), dtype=bool) if P > 0 else \
            np.zeros((B, T, 0), dtype=bool)
        allow = np.concatenate([allow_past, allow_cur], axis=-1)          # [B,T,Tk]
        return allow[:, None, :, :]

    def _moe(self, h2: np.ndarray, pre: str, c: ModelConfig) -> np.ndarray:
        B, T, d = h2.shape
        router_logits = h2 @ self.params[pre + "router"]    # [B,T,E]
        topk_idx = np.argsort(router_logits, axis=-1)[:, :, -c.moe_topk:]   # [B,T,topk]
        # 仅对选中专家的 router 分数做 softmax → 概率和=1 (INV-14)
        sel = np.take_along_axis(router_logits, topk_idx, axis=-1)
        w = numeric.stable_softmax(sel, axis=-1)
        out = np.zeros((B, T, d))
        for e in range(c.n_experts):
            gate = h2 @ self.params[pre + f"e{e}_gate"]
            up = h2 @ self.params[pre + f"e{e}_up"]
            # 按 activation 选择门控非线性 (INV: config.activation 真正生效)
            gate_act = ops.gelu(gate) if c.activation == "gelu" else gate / (1.0 + np.exp(-gate))
            ex = (gate_act * up) @ self.params[pre + f"e{e}_down"]
            mask = topk_idx == e
            out += np.sum(w * mask, axis=-1, keepdims=True) * ex
        return out

    def _apply_lora(self, x: np.ndarray, pre: str, tgt: str, base_out: np.ndarray) -> np.ndarray:
        """推理端 LoRA 增量 (numpy 同公式, Hu et al. 2022, INV-24):
        ``base + (α/r)·(x@Aᵀ)@Bᵀ``；与训练端 TinyTrainer._apply_lora 数值一致
        (INV-24), save→load 后零偏差。"""
        c = self.cfg
        if not c.use_lora:
            return base_out
        A = self.params[f"{pre}{tgt}_lora_a"]                   # [r, in]
        B = self.params[f"{pre}{tgt}_lora_b"]                   # [out, r]
        scale = c.lora_alpha / c.lora_r
        lora = ((x @ A.T) @ B.T) * scale
        return base_out + lora

    def logits_next(self, token_ids: np.ndarray) -> np.ndarray:
        """给定已生成序列, 返回最后一个位置的 logits [V]。"""
        logits = self.forward(token_ids)
        last = logits[:, -1, :]
        return last

    # ---------------- 序列化 + 校验 ----------------
    def save(self, dir_path: str) -> str:
        os.makedirs(dir_path, exist_ok=True)
        np.savez(os.path.join(dir_path, "params.npz"), **self.params)
        manifest = {"config": self.cfg.to_json(),
                    "params_sha256": {},
                    "total_params": self.count_params()}
        # 逐参数 sha256, 保证 R-L2-06 完整性可校验
        with open(os.path.join(dir_path, "params.npz"), "rb") as f:
            raw = f.read()
        manifest["params_sha256"]["_npz"] = hashlib.sha256(raw).hexdigest()
        with open(os.path.join(dir_path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return dir_path

    @classmethod
    def load(cls, dir_path: str, verify: bool = True) -> "Transformer":
        with open(os.path.join(dir_path, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        cfg = ModelConfig.from_json(manifest["config"])
        with open(os.path.join(dir_path, "params.npz"), "rb") as f:
            raw = f.read()
        if verify:
            assert hashlib.sha256(raw).hexdigest() == manifest["params_sha256"]["_npz"], \
                "params.npz sha256 mismatch!"
        data = np.load(os.path.join(dir_path, "params.npz"))
        model = cls.__new__(cls)
        model.cfg = cfg
        model.params = {k: data[k] for k in data.files}
        model._cache = {}
        # 重建 tie 引用
        if cfg.tie_word_embeddings:
            model.params["lm_head"] = model.params["embed"]
        assert model.count_params() == manifest["total_params"], "param count mismatch"
        return model

    # ---------------- INT8 量化 (推理端, INV-07) ----------------
    def quantize_int8(self, axis: int = -1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """对全部线性权重做 per-channel INT8 RTN 量化 (Dettmers et al. 2022
        LLM.int8 思想, INV-07), 原地替换为反量化权重 ``q·scale`` —— 前向数值
        仅含量化误差, 权重以 INT8 存储、使用时反量化 (内存收益)。

        返回 ``{name: (scale, orig)}`` 供 INV-07 逐权重重建误差界校验
        (``|orig − dequant| ≤ scale/2``)。1D 参数 (RMSNorm/final_norm) 不量化
        (其需精确且 per-channel 对向量无意义)。权重绑定 (INV-12) 下 ``lm_head``
        与 ``embed`` 指向同一 (反量化后) 数组, 避免双重量化并维持 tie 语义。
        """
        info: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for k in [k2 for k2, v in self.params.items() if v.ndim == 2]:
            if self.cfg.tie_word_embeddings and k == "lm_head":
                continue  # 与 embed 同一数组, 量化一次即可
            v = self.params[k]
            q, scale = quant.quantize_int8_per_channel(v, axis=axis)
            self.params[k] = quant.dequantize_int8(q, scale)     # 反量化后原地替换
            info[k] = (np.asarray(scale, dtype=np.float64),
                       np.array(v, dtype=np.float64))
        if self.cfg.tie_word_embeddings:
            self.params["lm_head"] = self.params["embed"]       # 维持 INV-12 绑定
        return info

    # ---------------- FP16 量化 (推理端, 半精度模拟) ----------------
    def quantize_fp16(self) -> Dict[str, np.ndarray]:
        """对全部线性权重做 FP16 往返量化 (IEEE 754 半精度模拟), 与
        :meth:`quantize_int8` / :meth:`quantize_int4` 同范式 — 接线 L0 死代码
        ``quantize_fp16`` (v0.7.13 前 README 宣称 "int8/int4/fp16 量化" 但
        模型级 fp16 路径不存在)。误差界: |orig − fp16(orig)| ≤ ulp/2 (相对
        ≈ 4.88e-4) + 子正规绝对粒度 (~6e-8), 远优于 INT8/INT4。

        返回 ``{name: orig}`` 供逐权重相对误差界校验; 1D 参数不量化; tie 下
        ``lm_head`` 与 ``embed`` 同一数组, 量化一次并维持绑定。
        """
        info: Dict[str, np.ndarray] = {}
        for k in [k2 for k2, v in self.params.items() if v.ndim == 2]:
            if self.cfg.tie_word_embeddings and k == "lm_head":
                continue  # 与 embed 同一数组, 量化一次即可
            v = self.params[k]
            self.params[k] = quant.quantize_fp16(v).astype(np.float64)
            info[k] = np.array(v, dtype=np.float64)
        if self.cfg.tie_word_embeddings:
            self.params["lm_head"] = self.params["embed"]       # 维持 INV-12 绑定
        return info
    # ---------------- INT4 量化 (推理端, INV-07-int4) ----------------
    def quantize_int4(self, axis: int = -1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """对全部线性权重做 per-channel INT4 RTN 量化 (对称 [-7,7], 15 电平),
        与 :meth:`quantize_int8` 同范式 (LLM.int8 / RTN 思想, INV-07 同形式
        误差界 ``|orig − dequant| ≤ scale/2``)。INT4 为边缘部署常用格式
        (GPTQ/AWQ 的 RTN 基线); 误差大于 INT8, 由测试以 INT4 口径把关。

        返回 ``{name: (scale, orig)}``; 1D 参数不量化; tie 下 ``lm_head``
        与 ``embed`` 同一数组, 量化一次并维持绑定。
        """
        info: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for k in [k2 for k2, v in self.params.items() if v.ndim == 2]:
            if self.cfg.tie_word_embeddings and k == "lm_head":
                continue  # 与 embed 同一数组, 量化一次即可
            v = self.params[k]
            q, scale = quant.quantize_int4_per_channel(v, axis=axis)
            self.params[k] = quant.dequantize_int4(q, scale)     # 反量化后原地替换
            info[k] = (np.asarray(scale, dtype=np.float64),
                       np.array(v, dtype=np.float64))
        if self.cfg.tie_word_embeddings:
            self.params["lm_head"] = self.params["embed"]       # 维持 INV-12 绑定
        return info


def verify_causal_mask(model: Transformer, token_ids: np.ndarray,
                       position: int | None = None) -> tuple[bool, int]:
    """INV-11 自检: 扰动 ``position`` 之后的 token, 之前位置的输出必须 bitwise 不变。

    零重写: 判据与 PRD INV-11 一致 ("扰动未来 token, 输出 bitwise 不变"),
    等价于 GPT/LLaMA 解码器的 causal masking 定义 (Vaswani et al. 2017;
    Radford et al. 2018)。

    Args:
        model: 待检验的 ``Transformer``。
        token_ids: 输入 id, 形状 ``[T]`` 或 ``[B,T]``。
        position: 扰动分界点 (``> position`` 的 token 全部被替换); ``None`` 表示
            ``max(0, T-2)`` —— 即至少保留一个"未来 token"可扰动 (若取 ``T-1``
            则不存在未来 token, 检验将失去判别力)。

    Returns:
        ``(prefix_bitwise_equal, n_changed_after)``:
        - ``prefix_bitwise_equal``: ``[0, position]`` 区间的输出是否逐位相同;
        - ``n_changed_after``: ``position`` 之后发生变化的元素个数 (应 > 0,
          否则说明扰动无效、检验不具有判别力)。
    """
    ids = np.asarray(token_ids)
    if ids.ndim == 1:
        ids = ids[None]
    T = ids.shape[1]
    pos = max(0, T - 2) if position is None else int(position)
    assert 0 <= pos < T, "position 越界"
    base = np.asarray(model.forward(ids))
    vocab = int(model.cfg.vocab_size)
    perturbed = ids.copy()
    for j in range(pos + 1, T):
        # 半词表偏移: 保证替换成一个完全不同的 token (embedding 显著不同)
        perturbed[:, j] = (perturbed[:, j] + vocab // 2) % max(vocab, 1)
    after = np.asarray(model.forward(perturbed))
    prefix_equal = bool(np.array_equal(base[:, :pos + 1], after[:, :pos + 1]))
    n_changed = int(np.count_nonzero(base[:, pos + 1:] != after[:, pos + 1:]))
    return prefix_equal, n_changed


def save_model(model: Transformer, dir_path: str) -> str:
    return model.save(dir_path)


def load_model(dir_path: str, verify: bool = True) -> Transformer:
    return Transformer.load(dir_path, verify=verify)
