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
| L11 | **引擎联邦** | `polaris.engines`（10 个世界顶级引擎预设 + ifai 内置 + mock） | **Polaris 薄适配**（一份 OpenAI 兼容契约） |
| L12 | **操作系统层** | `polaris.runtime` | **Polaris 原创** |

> 另有**工具层**（非独立层，是横切能力）：`polaris.compiler`（意图编译器）、`polaris.runtime.consensus`（共识层）、`polaris.bench`（基准套件）。它们分别消费 L10 的 ISA、L11 的引擎、L12 的 runtime，因此放在 L12 之上的应用侧。

---

## 3. Polaris 自有模块（L11/L12）依赖图

```
polaris/
├── __init__.py            # 公开面：PolarisOS / 引擎 / runtime / compiler
├── _vendor_path.py        # vendor 路径注入（ifai import 的前提）
├── cli.py                 # shell：doctor/engines/consensus/compile/bench/dispatch/isa/demo/audit
├── demo.py                # 端到端演示（与 dispatch 同一代码路径）
├── compiler.py            # 意图编译器：自然语言 → ISA 程序（前端/IR/Pass/后端）
├── bench.py               # 可复现基准套件（分位数/吞吐/env 指纹）
├── engines/
│   ├── base.py            # EngineAdapter 契约 + EngineResult + validate_result
│   ├── federation.py      # 引擎联邦：OpenAI 兼容适配器 + 10 预设库 + autodiscover
│   ├── llamacpp.py        # llama.cpp 适配（继承 federation 的 OpenAICompatEngine）
│   ├── builtin.py         # ifai 内置引擎（无模型则调 ifai cmd_train 现场训练）
│   └── mock.py            # 测试替身
└── runtime/
    ├── registry.py        # 引擎注册表（设备管理器）
    ├── scheduler.py       # 约束调度器（确定性优先/外部优先/偏好/诚实失败）
    ├── gates.py           # G1 确定性 / G2 有限性 / G3 契约
    ├── audit.py           # 哈希链账本（追加-only，可重放验证）
    ├── consensus.py       # 共识层：Self-Consistency + 多引擎交叉验证
    └── orchestrator.py    # PolarisOS 门面（dispatch/run_isa/isa_stats/audit_*）
```

依赖方向严格单向：`engines → runtime → (cli/demo)`；`runtime.orchestrator` 只依赖 engines 契约与 ifai L10 注册表，**不依赖任何引擎具体实现**（新引擎可零改核心接入）。`compiler` 与 `bench` 是叶子工具，不被 runtime 依赖，避免循环。

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

## 5b. 引擎联邦：一份契约，N 个世界顶级引擎

```
build_engine("sglang")
  │
  ├─ PRESETS["sglang"]            # 预设库（10 条）：默认地址/健康端点/环境变量/引用 URL
  │     └─ 覆盖优先级：显式 base_url > 环境变量 > 预设默认地址
  ▼
OpenAICompatEngine                # 唯一的 HTTP 适配实现（urllib，零外部依赖）
  │   POST {base_url}/v1/chat/completions   （OpenAI Completions 兼容契约 — 行业事实标准）
  │   GET  {base_url}/health  →  available()   （绝不谎报在线）
  ▼
EngineResult(text, engine, degraded, meta{preset, base_url, cite, ...})
```

**为什么这不是重写**：所有目标引擎（llama.cpp / vLLM / SGLang / Ollama / LM Studio / TGI / MLX / TensorRT-LLM / NIM）**本就实现** OpenAI 兼容接口。Polaris 只写一份适配器，剩下的是配置。`LlamaCppAdapter` 现在直接继承 `OpenAICompatEngine`——不再有第二套 HTTP 代码。

**探测式发现**（`autodiscover`）：并发探测全部预设的健康端点，报告当前在线者。离线时返回空列表（绝不伪造）。

---

## 5c. 意图编译数据流（自然语言 → ISA 程序）

```
intent: "把这个向量归一化再做 softmax"
  │
  ▼
┌─────────────┐   关键词按字符长度加权（越长越具体）；直接写指令名额外加权
│  前端路由    │   → matches = [("smax", 12), ("norm", 9), ...]
└─────┬───────┘
  ▼
┌─────────────┐   每个家族取「首选指令 + 字典序其余」为候选池
│  IR 构造     │   轮转 family[0], family[1], ... 生成目标长度序列
└─────┬───────┘
  ▼
┌─────────────┐   在探针向量上试跑「已选序列 + 候选」
│  Pass: 干运行 │   跑不通 → 剔除并记录 dropped 原因（LLVM opt 思想）
│  （K4）       │   rank 合法化：1-D → 2-D 提升（LLVM type legalization）
└─────┬───────┘
  ▼
┌─────────────┐   探针上跑两次比对 ⇒ deterministic（K3）
│  Pass: 确定性 │   指纹 = sha256("op1|op2|...")[:16]（K5，只依赖指令序列）
└─────┬───────┘
  ▼
ISAProgram{steps, matches, dropped, deterministic, gates[K1..K5]}
```

**编译期干运行与运行期共用同一个 `_apply()`** —— 编译期通过的，运行期必然同构。这是「编译期多花的钱，换运行期不炸」的工程直觉。

---

## 5d. 共识数据流（一致性 → 可测量的数）

```
prompt
  │
  ├─ Self-Consistency 模式：同一引擎采样 k 次（temperature > 0，seed 序列递增）
  │     └─ majority_vote（票数 → 胜者；平局按字典序打破，保证顺序无关）
  │
  └─ Cross-Engine 模式：N 个不同引擎各出一次
        └─ 同一套投票与相似度计算
  │
  ▼
Consensus{text, agreement, support, total, mean_pairwise, votes, gates[C1..C4]}
```

- `agreement = support / total` —— 最高票占比，∈ [0,1]；
- `mean_pairwise` —— 所有样本两两 Jaccard bigram 相似度的均值（单样本时为 1.0）；
- 相似度用 **Jaccard bigram**（标准集合相似度，不是自造指标）；
- 固定 seed ⇒ 两次共识的指纹逐位相同（可复现）。

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
5. **意图编译器是规则驱动的确定性子集**：做关键词路由，**不做开放域语义理解**。`"归一化"` 能编译；`"帮我写首诗"` 抛 `NoIntentMatch` —— 那类请求应走 `PolarisOS.dispatch()` 交给世界顶级 LLM 引擎。硬编的编译器能理解一切，是自欺。
6. **共识层默认不做模型质量评判**：agreement 衡量的是**输出一致性**，不是正确性。要判正确性需外部 oracle（基准答案/人工），Polaris 不假装自己能判。
7. **基准数字仅本机有效**：`polaris bench` 输出必带 `env` 指纹（Python/numpy/OS/CPU），跨机比较必须连同 env 一起看，否则数字无意义。
