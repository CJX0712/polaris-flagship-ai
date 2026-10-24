# 架构文档 · ARCHITECTURE

> 北极星 POLARIS · 十二层映射、模块依赖、数据流、舰队关系
> 作者：晨星 (Morningstar) · Apache-2.0

---

## 1. 总览：Polaris 在晨星旗舰舰队中的位置

```
                    晨星旗舰舰队（apex-flagship, 305 仓）
                                │
        ┌───────────────────────┼────────────────────────┐
        │                       │                        │
   模型/内核件              操作系统件                 验证件
   ┌─────────┐            ┌──────────────┐          ┌─────────┐
   │ TIANJI  │            │    NEXUS     │          │  AXIOM  │
   │ 混合架构 │◄──────────┤  JS 操作系统  │          │ JS 尺子 │
   │ LLM全栈 │            └──────────────┘          └─────────┘
   ├─────────┤                   ▲                        ▲
   │FlagshipAI│                  │                        │
   │FlashAttn │          ┌───────┴───────┐                │
   │ +PyTorch │          │  ★ POLARIS ★  │                │
   │ 交叉验证 │          │ Python 操作系统│                │
   └─────────┘          └───────────────┘                │
        │                       │                        │
        └─── ifai（Python 十层全栈 + 1024 ISA）────────────┘
                    （vendor：third_party/ifai/）
```

**Polaris 的独特站位**：舰队此前的操作系统件（NEXUS/APEX-Ω）在 JS 侧；Python 侧有全栈（ifai）、有模型（TIANJI/FlagshipAI），但缺一个把「真实世界顶级引擎 + 全栈 + 验证」编排起来的**操作系统**。Polaris 补上这一块，且全部件零重写复用。

---

## 2. 十二层架构

| 层 | 名称 | 实现 | Polaris 的处置 |
|---|---|---|---|
| L0 | 算子内核 | `ifai.core`（在线 softmax、logsumexp、量化、Newton–Schulz） | vendor 原样 |
| L1 | 数值/RNG | `ifai.rng`（PCG64，bitwise 可复现） | vendor 原样 |
| L2 | 模型架构 | `ifai.models`（RMSNorm/RoPE/GQA/SwiGLU/MoE） | vendor 原样 |
| L3 | 推理引擎 | `ifai.engine`（KV-Cache/连续批处理/投机解码） | vendor 原样 |
| L4 | 训练/微调 | `ifai.train`（TinyTrainer/LoRA/QAT/手写反向） | vendor 原样 |
| L5 | 检索/记忆 | `ifai.retrieval`（BM25+稠密+HNSW+RRF+QFR+MMR） | vendor 原样 |
| L6 | 智能体 | `ifai.agent`（ReAct/Reflexion/MemGPT/沙箱） | vendor 原样 |
| L7 | 服务 | `ifai.serve`（OpenAI 兼容/真 SSE/护栏） | vendor 原样 |
| L8 | 评测 | `ifai.eval` | vendor 原样 |
| L9 | 应用 | `ifai.cli` / `ifai.sdk` | vendor 原样 |
| L10 | **微指令 ISA** | `ifai.l10`（1024 条，32 家族，390 性质断言） | vendor 原样，经 `run_isa` 暴露 |
| L11 | **引擎层** | `polaris.engines` | **Polaris 薄适配** |
| L12 | **操作系统层** | `polaris.runtime` | **Polaris 原创** |

---

## 3. Polaris 自有模块（L11/L12）依赖图

```
polaris/
├── __init__.py            # 公开面：PolarisOS / 引擎 / runtime 四件套
├── _vendor_path.py        # vendor 路径注入（ifai import 的前提）
├── cli.py                 # shell：doctor/engines/dispatch/isa/train-tiny/demo/audit
├── demo.py                # 端到端演示（与 dispatch 同一代码路径）
├── engines/
│   ├── base.py            # EngineAdapter 契约 + EngineResult + validate_result
│   ├── llamacpp.py        # llama.cpp HTTP 适配（urllib，零依赖）
│   ├── builtin.py         # ifai 内置引擎（无模型则调 ifai cmd_train 现场训练）
│   └── mock.py            # 测试替身
└── runtime/
    ├── registry.py        # 引擎注册表（设备管理器）
    ├── scheduler.py       # 约束调度器（确定性优先/外部优先/偏好/诚实失败）
    ├── gates.py           # G1 确定性 / G2 有限性 / G3 契约
    ├── audit.py           # 哈希链账本（追加-only，可重放验证）
    └── orchestrator.py    # PolarisOS 门面（dispatch/run_isa/isa_stats/audit_*）
```

依赖方向严格单向：`engines → runtime → (cli/demo)`；`runtime.orchestrator` 只依赖 engines 契约与 ifai L10 注册表，**不依赖任何引擎具体实现**（新引擎可零改核心接入）。

---

## 4. dispatch 数据流（一条意图的完整生命周期）

```
prompt
  │
  ▼
┌─────────────┐   DispatchRequest(prompt, max_tokens, temperature,
│ Scheduler   │◄  require_deterministic, preferred)
│ 约束调度     │   策略：确定性候选过滤 → kind 优先级
└─────┬───────┘   （world-class-external > builtin > mock）
      │ DispatchDecision(adapter, reason, candidates)
      ▼
┌─────────────┐   adapter.complete(prompt, max_tokens, temperature, seed)
│  Engine     │──► EngineResult(text, engine, tokens, degraded, meta)
└─────┬───────┘
      │
      ▼
┌─────────────┐   G2 有限性 ─┐
│   Gates     │   G3 契约   ─┤─ 任一失败 ⇒ GateViolation（绝不放行）
│  G1 确定性   │   G1 重放   ─┘   （G1 对 builtin/mock 自动开启）
└─────┬───────┘
      │ 全过
      ▼
┌─────────────┐   entry = {seq, ts, intent, engine, gates, latency, fp}
│  AuditLog   │──► sha256(prev_hash ‖ canonical_json(entry)) 追加-only
└─────┬───────┘
      ▼
   EngineResult（meta.dispatch 附带调度理由/门禁明细/耗时）
```

---

## 5. 引擎发现与降级时序

```
polaris doctor
  ├─ llama.cpp      : GET http://127.0.0.1:8080/health（200 ⇒ 可用）
  ├─ ifai-builtin   : 恒可用（无模型则现场训练微型模型，ifai cmd_train）
  └─ mock           : 恒可用（仅测试）

dispatch("…")
  ├─ llama.cpp 在线      ⇒ 选中 llama.cpp（kind 优先级最高）
  ├─ llama.cpp 缺席      ⇒ 选中 ifai-builtin（真实模型生成，degraded=False）
  └─ 内置亦不可训练      ⇒ 选中 mock（degraded=True，如实标记）
```

**诚实记账**：降级不是静默的——`EngineResult.degraded` 与审计账本的 `degraded` 字段双重记录，任何「用 mock 冒充世界级引擎」的路径在机制上不存在。

---

## 6. 与上游（ifai）的边界

| 事项 | Polaris | ifai（vendor） |
|---|---|---|
| 模型/训练/推理代码 | 不写 | 唯一实现源 |
| ISA 1024 条 | 不写，经 `run_isa` 暴露 | 唯一实现源 |
| 引擎适配/调度/门禁/审计 | **原创** | 不涉及 |
| 修改上游 | 禁止（P7b 哈希审计） | — |
| 授权 | Apache-2.0 | 原始 Apache-2.0 保留 |

---

## 7. 已知边界（不掩饰）

1. **llama.cpp 需自备**：Polaris 不打包模型权重与 llama.cpp 二进制；`doctor` 会如实报告其在线状态。
2. **内置引擎是微型模型**：离线回退的生成质量限于 ifai MICRO_CORPUS 域（中文 AI 科普语料），定位是「可复现的确定性证明路径」，不是质量路径——质量路径是 llama.cpp。
3. **G1 对外部引擎默认跳过重放**：llama.cpp 的确定性由其自身保证，双调成本不诚实；如需强制重放，`PolarisOS(g1_replay=True)`。
4. **审计账本为单机 JSONL**：多机场景需外接日志系统（路线图）。
