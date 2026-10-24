# 设计文档 · DESIGN

> 北极星 POLARIS · 设计决策、门禁判据、缺陷记录
> 作者：晨星 (Morningstar) · Apache-2.0

---

## 1. 为什么需要 Polaris（问题定义）

晨星舰队已具备：混合架构模型（TIANJI）、FlashAttention 内核（FlagshipAI）、十层全栈 + 1024 ISA（ifai）、JS 侧操作系统（NEXUS）。缺的是 **Python 侧的系统层**：

| 需求 | 舰队现状 | Polaris 的答案 |
|---|---|---|
| 接入真实世界顶级引擎 | 仅 JS 侧 apex-omega 做过 llama.cpp 合流 | `LlamaCppAdapter`（Python，OpenAI 兼容 HTTP） |
| 引擎按约束调度 | NEXUS 有（JS） | `Scheduler`（确定性优先/外部优先/诚实失败） |
| 产出可信证明 | AXIOM 64 不变量（JS） | G1/G2/G3 运行时门禁 + P1–P7 测试门禁 |
| 治理审计 | 无 | 哈希链账本（追加-only，可重放） |
| 系统永远能跑 | — | 三级引擎降级 + `degraded` 诚实标记 |

**一句话**：Polaris 不造更强的引擎，它让已有顶级引擎**可编排、可证明、可审计**。

---

## 2. 关键设计决策

### 2.1 为什么 ifai 用 vendor 而非 pip 依赖

- ifai 未发布 PyPI；`pip install git+…` 要求构建期联网，违背「clone 即可跑」；
- vendor 源码 + 保留 LICENSE/NOTICE 是 pip/requests 等世界顶级项目的标准做法；
- **代价透明化**：vendor 意味着与上游漂移风险——用 P7b 测试（抽样 SHA-256）机检，上游升级时重新 vendor 并跑测试。

### 2.2 为什么 llama.cpp 经 HTTP 而非 FFI/子管道

llama.cpp 官方 server（`tools/server`）提供 OpenAI 兼容 HTTP 面，这是其**公开契约**：

- 复用官方 server 的全部工程（GGUF 量化、批处理、并行槽位、热加载）；
- Polaris 只依赖契约不依赖实现——远程 vLLM/SGLang 同样可经同一适配器形态接入（改 base_url 即可）；
- HTTP 客户端用标准库 `urllib`，零第三方依赖。

### 2.3 为什么 G1 对外部引擎默认跳过重放

G1 的判据是「同 seed 同输入 → 逐位一致」。对内置/mock 引擎，双调成本可忽略且能抓住实现级不确定性；对 llama.cpp，双调 = 双倍推理成本，而其确定性由引擎自身保证（temperature=0 贪心解码）。**诚实的分工**：引擎保证自己的确定性，Polaris 保证编排层的可复现性。需要强制时 `PolarisOS(g1_replay=True)`。

### 2.4 为什么审计用哈希链而非明文日志

明文日志可被无痕篡改（改一行不留痕）。哈希链（Git 对象模型 / RFC 6962 Merkle 日志的同款思想）：

```
entry_hash[i] = sha256( entry_hash[i-1] ‖ canonical_json(entry_i) )
```

性质：改任何历史条目 → 该条之后全部哈希断裂；删条目 → 序号断裂。`verify_chain()` 30 行代码完成全部验证，无外部依赖。

---

## 3. 七条门禁判据（可独立复现）

| 门禁 | 判据 | 违反后果 |
|---|---|---|
| **G1 确定性** | `fingerprint(a) == fingerprint(b)`（重放比对，SHA-256 前 64bit） | `GateViolation`，dispatch 拒绝返回 |
| **G2 有限性** | 非空 / 长度 ≤ max_tokens×8 / 词边界正则无 `\bnan\b\|\binf\b` | 同上 |
| **G3 契约** | `validate_result`：EngineResult 类型 / text 非空 / engine 已注册 / 无 U+FFFD / tokens 长度合理 | 同上 |
| **P1 契约 fuzz** | 10 种畸形输入（空、4096 长、emoji、控制符、SQL 注入字样、NaN 字样…）不抛异常且契约完整 | 测试红 |
| **P3 调度** | 四情形：kind 优先级 / preferred 生效 / 空注册表诚实失败 / max_tokens 约束 | 测试红 |
| **P5 审计** | 篡改历史条目 / 序号断裂 / 重放载荷干净 / 落盘重载 | 测试红 |
| **P7 零重写** | 每模块 600 字符内出处声明 / vendor LICENSE 保留 / 抽样哈希 / `cmd_train` 被复用且无自写训练循环 | 测试红 |

---

## 4. 被测试体系抓出并修复的真实缺陷（开发记录）

诚实记账——以下都是本次开发中**被自己的门禁抓到**的缺陷，不是彩排：

| # | 缺陷 | 抓它的门禁 | 修法 |
|---|---|---|---|
| 1 | `validate_result` 用子串 `"nan" in text` 判 NaN 泄漏，误伤回显型引擎（用户输入含 "NaN" 时 mock 回显即触雷） | P1 fuzz | NaN 检查下沉 G2 + 词边界正则 `\bnan\b\|\binf\b`（不再误伤 "Nanning"） |
| 2 | `polaris/runtime/__init__.py` 未导出三个门禁函数，测试导入即红 | P4 | 补全 `__all__` |
| 3 | `demo.py` / 两个 `__init__.py` 缺出处声明 | P7a | 补「零重写 · 出处」docstring |
| 4 | orchestrator 审计条目 `prompt_fp` 写作 `result.fingerprint() and _fp(prompt)`（逻辑丑陋且依赖副作用） | 代码审查 | 改 `_fp(prompt)` |
| 5 | `polaris isa` 的 properties 统计用了不存在的键（`ok/total` vs 上游 `passed/checked`） | demo 实跑 | 对齐上游 `property_report` 契约 |

**第 1 条是教科书级案例**：防御性检查本身成为缺陷源——子串匹配把「用户原文」当「引擎泄漏」。修正后判据更精确（词边界），且职责分层（契约层管结构、门禁层管语义）。

---

## 5. 性能特征（诚实口径）

Polaris 自身是薄编排层（单次 dispatch 的调度+门禁+审计开销 < 1ms，实测 0.3–0.8ms）；系统性能由被编排件决定：

| 路径 | 实测（本机，CPU） | 口径 |
|---|---|---|
| ISA 全量门禁 1024 条 + 390 性质断言 | ~2.5 s | `polaris isa` |
| 内置引擎现场训练 120 步 | ~21 s | ifai TinyTrainer，3 层 128 维 |
| 内置引擎 dispatch（模型就绪） | ~2.3 s/16 token | 逐 token 前向，NumPy |
| llama.cpp dispatch | 取决于模型与硬件 | 不重写，继承 llama.cpp 性能 |

Polaris 不做任何「加速自研引擎」的宣称——那是被复用件的功劳，诚实归位。

---

## 6. 路线图

- [x] L11/L12 引擎层 + 操作系统层（注册/调度/门禁/审计/编排）
- [x] llama.cpp 发现式接入 + 三级降级
- [x] 75 项测试全绿 + 1024 ISA 全量门禁
- [ ] vLLM/SGLang 远程适配器（同一契约，改 base_url）
- [ ] 多机审计汇聚（账本外接日志系统）
- [ ] 意图编译器（自然语言 → ISA 程序）的确定性子集
