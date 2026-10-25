"""L4 TinyTrainer — autograd 驱动的可训练微型 Transformer

目标: 在数十秒内于 CPU 上完成一个微型模型的训练 (INV-26 损失下降)，
产出与 ``ifai.models.Transformer`` 完全兼容的参数文件 (同名 + 同结构),
使 ``ifai demo`` 可直接加载推理。

零重写: 训练循环范式参考 Karpathy "nanogpt" (极简 GPT 训练循环);
梯度检查点 (INV-25) 复用 Chen et al. 2016 的重计算思想, 见
``ifai/train/checkpoint.py``。

可验证不变量:
- INV-25 梯度检查点: loss/grad ≡ 无检查点 (``diff ≤ 1e-5``), 由
  ``gradient_checkpointing`` 开关与 ``ifai.train.checkpoint.verify_checkpoint``
  自检。
- INV-26 训练损失单调性 (滑动平滑后)。
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List

import numpy as np

from .. import rng
from ..core import numeric, ops
from ..models.config import ModelConfig
from .autodiff import Tensor
from .checkpoint import checkpoint
from .optim import AdamW, GlobalStep, clip_global_norm


class TinyTrainer:
    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.params: Dict[str, Tensor] = {}
        # 梯度检查点开关 (INV-25): 开启后每层前向不保留层内激活, 反向时重算
        self.gradient_checkpointing = False
        self._build()

    def _P(self, name, shape):
        r = rng.get_rng()
        data = (r.standard_normal(shape) * self.cfg.initializer_range).astype(np.float64)
        self.params[name] = Tensor(data, requires_grad=True)

    def _build(self):
        c = self.cfg
        V, d = c.vocab_size, c.d_model
        self._P("embed", (V, d))
        if c.tie_word_embeddings:
            self.params["lm_head"] = self.params["embed"]
        else:
            self._P("lm_head", (V, d))
        self._P("final_norm", (d,))
        n_kv = c.n_kv_heads
        for l in range(c.n_layers):
            p = f"layer{l}."
            self._P(p + "attn_norm", (d,))
            self._P(p + "q_proj", (d, d))
            self._P(p + "k_proj", (d, n_kv * c.d_head))
            self._P(p + "v_proj", (d, n_kv * c.d_head))
            self._P(p + "o_proj", (d, d))
            self._P(p + "ffn_norm", (d,))
            if c.use_moe:
                # MoE/Switch 稀疏专家 (Fedus et al. 2022, INV-14): 每专家独立
                # gate/up/down + 路由权重; 与推理端 Transformer._moe 同名同结构,
                # 保证 save→load 后前向零偏差。
                for e in range(c.n_experts):
                    self._P(p + f"e{e}_gate", (d, c.d_ff))
                    self._P(p + f"e{e}_up", (d, c.d_ff))
                    self._P(p + f"e{e}_down", (c.d_ff, d))
                self._P(p + "router", (d, c.n_experts))
            else:
                self._P(p + "ffn_gate", (d, c.d_ff))
                self._P(p + "ffn_up", (d, c.d_ff))
                self._P(p + "ffn_down", (c.d_ff, d))
            if c.use_lora:
                # LoRA 低秩适配 (Hu et al. 2022, INV-24): 仅训练 A/B, 冻结目标基座。
                # 与推理端 Transformer 同名同结构 (W + (α/r)·B@A), save→load 零偏差。
                for t, shp in (("q_proj", (d, d)),
                               ("k_proj", (d, n_kv * c.d_head)),
                               ("v_proj", (d, n_kv * c.d_head)),
                               ("o_proj", (d, d))):
                    Wi, Wo = shp   # 投影权重形状 = (in_dim, out_dim)
                    self._P(p + f"{t}_lora_a", (c.lora_r, Wi))
                    # B 初始化为 0 (INV-24: B=0 ⇒ LoRA 增量≡0 ⇒ 等价基座前向)
                    self.params[p + f"{t}_lora_b"] = Tensor(
                        np.zeros((Wo, c.lora_r), dtype=np.float64), requires_grad=True)
                    self.params[p + t].requires_grad = False   # 冻结基座 (LoRA 语义)

    # ---------------- autograd 前向 ----------------
    def _rope(self, x: Tensor, pos: np.ndarray, dim: int) -> Tensor:
        """RoPE 旋转 (单节点自定义算子, 反向 = 正交旋转转置)。

        前向: 偶/奇交织两路 ``rot1 = x1·cos − x2·sin, rot2 = x1·sin + x2·cos``
        (Su et al. 2021 RoFormer)。旋转矩阵正交 ⇒ 反向即转置:
        ``dx1 = g1·cos + g2·sin, dx2 = −g1·sin + g2·cos``。
        v0.7.0: 融合为单节点 — 旧 11 节点表达版除节点开销外还存在
        ``_prev`` 漏 rot1/rot2 致 q/k 梯度全零的缺陷 (已修复并回归防护)。
        """
        half = dim // 2
        # 零重写: 复用推理端同一 NTK-aware 频率生成 (INV-03 解析对照口径一致),
        # 避免训练/推理 RoPE 分叉导致 save→load 后可复现性破坏 (rope_scaling≠1 时)。
        freqs = ops._rope_freqs(dim, base=self.cfg.rope_base,
                                scaling=self.cfg.rope_scaling)
        angles = np.outer(pos, freqs)                       # [T, half]
        cos = np.cos(angles)[None]                          # [1,T,half]
        sin = np.sin(angles)[None]
        x1 = x.data[..., 0::2]
        x2 = x.data[..., 1::2]
        out = np.empty_like(x.data)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        out_t = Tensor(out, (x,), "rope")

        def _b():
            if x.requires_grad:
                g = out_t.grad
                g1, g2 = g[..., 0::2], g[..., 1::2]
                dx = np.empty_like(x.data)
                dx[..., 0::2] = g1 * cos + g2 * sin
                dx[..., 1::2] = -g1 * sin + g2 * cos
                x.grad += dx
        out_t._backward = _b
        return out_t

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """开启/关闭梯度检查点 (INV-25)。"""
        self.gradient_checkpointing = bool(enabled)

    @staticmethod
    def _causal_bias(T: int) -> np.ndarray:
        """因果掩码偏置 [T,T]: 上三角 -1e30 (经 softmax 后概率严格为 0)。"""
        mask = np.triu(np.ones((T, T), bool), 1)
        return np.where(mask, -1e30, 0.0)

    def _bias(self, T: int) -> np.ndarray:
        """因果偏置缓存 (同 T 复用, 免每层每步重建)。"""
        cache = getattr(self, "_bias_cache", None)
        if cache is None:
            cache = self._bias_cache = {}
        b = cache.get(T)
        if b is None:
            b = cache[T] = self._causal_bias(T)
        return b

    def _layer_forward(self, x: Tensor, layer: int) -> Tensor:
        """单层 DecoderBlock 前向 (供梯度检查点按层分段重算, INV-25)。

        v0.7.0: 注意力改为批量可微 GQA (全程 autograd, 无逐头 Python 循环)。
        修复历史缺陷: 旧实现经 ``.data`` 取 raw numpy 后 ``Tensor(ctx)``
        重建叶子, 切断了 attn_norm/q_proj/k_proj/v_proj 的梯度链 —
        注意力参数从未收到梯度。
        """
        c = self.cfg
        B, T = x.data.shape[0], x.data.shape[1]
        d, dh, n_h, n_kv = c.d_model, c.d_head, c.n_heads, c.n_kv_heads
        pos = np.arange(T)
        p = f"layer{layer}."
        # RMSNorm
        ms = (x * x).mean(axis=-1, keepdims=True)
        h = x / (ms + c.norm_eps).pow(0.5) * self.params[p + "attn_norm"]
        q = self._apply_lora(h, p, "q_proj", h.matmul(self._wq(p + "q_proj")))
        k = self._apply_lora(h, p, "k_proj", h.matmul(self._wq(p + "k_proj")))
        v = self._apply_lora(h, p, "v_proj", h.matmul(self._wq(p + "v_proj")))
        # RoPE (单节点) + 头重排 (reshape/transpose 元操作)
        q = self._rope(q, pos, d).reshape(B, T, n_h, dh).transpose((0, 2, 1, 3))
        k = self._rope(k, pos, n_kv * dh).reshape(B, T, n_kv, dh).transpose((0, 2, 1, 3))
        v = v.reshape(B, T, n_kv, dh).transpose((0, 2, 1, 3))
        # GQA: kv 头扩展 (与推理端 np.repeat 同语义, INV-10)
        k = k.repeat(c.n_repeats, axis=1)
        v = v.repeat(c.n_repeats, axis=1)
        # 批量因果注意力 (单节点自定义算子: BLAS 前向 + 标准解析反向)
        scale = 1.0 / dh ** 0.5
        ctx = Tensor.attention(q, k, v, scale, bias=self._bias(T))  # [B,H,T,dh]
        ctx = ctx.transpose((0, 2, 1, 3)).reshape(B, T, d)
        o = ctx.matmul(self._wq(p + "o_proj"))
        o = self._apply_lora(ctx, p, "o_proj", o)
        x = x + o
        ms2 = (x * x).mean(axis=-1, keepdims=True)
        h2 = x / (ms2 + c.norm_eps).pow(0.5) * self.params[p + "ffn_norm"]
        if c.use_moe:
            x = x + self._moe_forward(h2, layer)
        else:
            g = h2.matmul(self._wq(p + "ffn_gate"))
            u = h2.matmul(self._wq(p + "ffn_up"))
            # 按 activation 选择门控非线性 (INV: config.activation 真正生效)
            # 默认 "swiglu" 保持 silu 逻辑完全不变; "gelu" 走可微 gelu。
            gate_act = g.gelu() if c.activation == "gelu" else g * g.sigmoid()
            ff = (gate_act * u).matmul(self._wq(p + "ffn_down"))
            x = x + ff
        return x

    def _moe_forward(self, h2: "Tensor", layer: int) -> "Tensor":
        """MoE 前向 (可微, 严格镜像推理端 ``Transformer._moe`` 以保证
        train→inference save→load 零偏差, INV-14 路由守恒)。

        零重写: Switch/Top-k MoE (Fedus et al. 2022); 仅对选中专家的
        router 分数做 softmax ⇒ 路由概率和=1 (INV-14)。反向经 ``getitem``
        高级索引散射回 ``router``, 经专家 matmul 回各专家权重。
        """
        c = self.cfg
        B, T, d = h2.data.shape
        E, K = c.n_experts, c.moe_topk
        router = self._wq(f"layer{layer}.router")
        router_logits = h2.matmul(router)                     # [B,T,E] Tensor
        rl = router_logits.data
        topk_idx = np.argsort(rl, axis=-1)[:, :, -K:]          # [B,T,K] 离散路由
        # 可微 gather 选中专家的 router logits (反向经 getitem 散射回 router)
        bi = np.arange(B)[:, None, None]
        ti = np.arange(T)[None, :, None]
        sel = router_logits[bi, ti, topk_idx]                  # [B,T,K] Tensor
        w = sel.softmax(axis=-1)                              # [B,T,K] 概率和=1
        out = Tensor.zeros(B, T, d)
        for e in range(E):
            gate = h2.matmul(self._wq(f"layer{layer}.e{e}_gate"))
            up = h2.matmul(self._wq(f"layer{layer}.e{e}_up"))
            gate_act = gate.gelu() if c.activation == "gelu" else gate * gate.sigmoid()
            ex = (gate_act * up).matmul(self._wq(f"layer{layer}.e{e}_down"))  # [B,T,d]
            mask = (topk_idx == e)                            # [B,T,K] 离散
            coeff = (w * Tensor(mask, requires_grad=False)).sum(axis=-1, keepdims=True)  # [B,T,1]
            out = out + coeff * ex
        return out

    def _wq(self, name: str) -> "Tensor":
        """QAT 门控 (Jacob et al. 2018): ``use_qat`` 时返回权重的 STE 伪量化
        节点 (前向 dequant(quant(W)), 反向恒等直通), 否则原样返回 — 默认
        关闭时训练图与数值逐字节不变。"""
        w = self.params[name]
        return w.fake_quant_int8_per_channel() if self.cfg.use_qat else w

    def _apply_lora(self, x: "Tensor", pre: str, tgt: str, base_out: "Tensor") -> "Tensor":
        """LoRA 低秩增量 (Hu et al. 2022, INV-24): ``base + (α/r)·(x@Aᵀ)@Bᵀ``。

        与推理端 ``Transformer`` 同公式 (INV-24); ``B=0`` 初始时增量≡0 ⇒ 等价
        基座前向。冻结目标基座 (q/k/v/o), 仅 A/B 可训练。
        """
        c = self.cfg
        if not c.use_lora:
            return base_out
        A = self._wq(f"{pre}{tgt}_lora_a")                      # [r, in]
        B = self._wq(f"{pre}{tgt}_lora_b")                      # [out, r]
        scale = c.lora_alpha / c.lora_r
        lora = (x.matmul(A.transpose())).matmul(B.transpose()) * scale
        return base_out + lora

    def forward(self, ids: np.ndarray) -> Tensor:
        c = self.cfg
        ids = np.asarray(ids)
        if ids.ndim == 1:
            ids = ids[None]
        x = self._wq("embed")[ids]                        # [B,T,d]
        for l in range(c.n_layers):
            if self.gradient_checkpointing:
                # 按层检查点: 前向丢弃层内激活, 反向时重算 (INV-25)
                x = checkpoint(lambda t, i=l: self._layer_forward(t, i), x)
            else:
                x = self._layer_forward(x, l)
        ms3 = (x * x).mean(axis=-1, keepdims=True)
        x = x / (ms3 + c.norm_eps).pow(0.5) * self.params["final_norm"]
        logits = x.matmul(self._wq("lm_head").transpose())     # [B,T,V]
        return logits

    def loss(self, ids: np.ndarray, targets: np.ndarray) -> Tensor:
        logits = self.forward(ids)
        # 交叉熵 (autograd 路径)
        logits_flat = logits.reshape(-1, self.cfg.vocab_size)
        tgt = targets.reshape(-1)
        # 数值稳定 log-softmax (max 作为常数 detach, 保证梯度正确)
        m = logits_flat.data.max(axis=-1, keepdims=True)
        z = logits_flat - Tensor(m, requires_grad=False)
        logsum = z.exp().sum(axis=-1, keepdims=True).log()
        lse = logsum + Tensor(m, requires_grad=False)      # 还原真实 logsumexp
        logp = logits_flat - lse                            # = log-softmax (梯度正确)
        nll = -logp[np.arange(len(tgt)), tgt].mean()
        return nll

    def train(self, ids_corpus: np.ndarray, steps: int = 200,
              lr: float = 3e-3, batch_size: int = 8, seq_len: int = 32,
              grad_clip: float = 1.0, warmup: int = 10) -> List[float]:
        params = [p for p in self.params.values() if p.requires_grad]
        opt = AdamW(params, lr=lr, weight_decay=0.01)
        sched = GlobalStep(lr, steps, warmup_steps=warmup)
        losses = []
        N = len(ids_corpus) - seq_len
        for s in range(steps):
            opt.zero_grad()
            opt.lr = sched.step()
            idx = rng.get_rng().integers(0, max(N, 1), size=batch_size)
            xb = np.stack([ids_corpus[i:i + seq_len] for i in idx])
            yb = np.stack([ids_corpus[i + 1:i + seq_len + 1] for i in idx])
            loss_t = self.loss(xb, yb)
            loss_t.backward()
            clip_global_norm(params, grad_clip)
            opt.step()
            losses.append(float(loss_t.data))
        return losses

    def to_numpy_params(self) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in self.params.items():
            out[k] = np.array(v.data, dtype=np.float64)
        # tie 引用一致性: 推理端会 rebind, 但这里保存独立数组
        if self.cfg.tie_word_embeddings:
            out["lm_head"] = out["embed"].copy()
        return out

    def count_params(self) -> int:
        """可训练参数量 (与 ``Transformer.count_params`` 同口径, INV-13)。

        权重绑定 (INV-12) 时 ``lm_head`` 与 ``embed`` 共享同一数组,
        只计一次; 保证 save→load 后 ``Transformer.load`` 的
        ``count_params() == manifest["total_params"]`` 断言成立。
        """
        skip = {"lm_head"} if self.cfg.tie_word_embeddings else set()
        return int(sum(v.data.size for k, v in self.params.items() if k not in skip))

    def save(self, dir_path: str) -> str:
        os.makedirs(dir_path, exist_ok=True)
        params = self.to_numpy_params()
        np.savez(os.path.join(dir_path, "params.npz"), **params)
        with open(os.path.join(dir_path, "params.npz"), "rb") as f:
            raw = f.read()
        manifest = {
            "config": self.cfg.to_json(),
            "params_sha256": {"_npz": hashlib.sha256(raw).hexdigest()},
            "total_params": self.count_params(),
            "trained_by": "TinyTrainer",
        }
        with open(os.path.join(dir_path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return dir_path
