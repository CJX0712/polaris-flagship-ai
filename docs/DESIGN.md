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

### 2.5 为什么 10 个世界顶级引擎只写一份适配器（v1.1.0）

**关键事实**：llama.cpp / vLLM / SGLang / Ollama / LM Studio / TGI / MLX / TensorRT-LLM / NIM **全部实现 OpenAI Completions 兼容接口**。这是行业事实标准，不是 Polaris 发明的协议。

因此：

- 适配层只需一份 `OpenAICompatEngine`（urllib，零第三方依赖）；
- `LlamaCppAdapter` 直接继承它——**不再有第二套 HTTP 代码**；
- 接入新引擎 = 加一条预设（URL/健康端点/环境变量/引用），**零代码改动**；
- 契约不变 ⇒ 共识层可跨引擎交叉验证，门禁层对所有引擎一视同仁。

**替代方案（被否）**：为每个引擎写专用适配器。代价是 N 份重复实现、N 份 bug、无法跨引擎交叉验证。**一个契约统治 N 个引擎**才是工程。

### 2.6 为什么编译器在编译期就要干运行（v1.1.0）

ISA 有 1024 条微指令，形状要求各异（如 `attn.*` 需 2-D 输入）。若把形状错误留到运行期：

- 用户看到的是运行期异常，而非「这条意图编译不通过」；
- 同一个坏程序可能被反复执行、反复失败。

所以编译器做**编译期干运行**（LLVM `opt` pass 思想）：候选指令先在探针向量上试跑，跑不通的**当场剔除并记录原因**（`dropped` 字段），运行期只剩「必然可执行」的程序。

配套的 **rank 合法化**（LLVM type legalization 同构）：1-D 输入遇 2-D 要求时按 `np.atleast_2d` 提升一次。

**编译期干运行与运行期共用同一个 `_apply()`** —— 这是「编译期通过 ⇒ 运行期必然同构」的实现保证，不是口头承诺。

### 2.7 为什么共识层只测「一致性」不测「正确性」（v1.1.0）

`agreement`（最高票占比）衡量的是**输出一致性**，不是正确性。要判正确性需要外部 oracle（标准答案 / 人工标注 / 可验证执行），Polaris 手里没有。

**诚实的边界**：一致性高 ≠ 正确（三个引擎可能一致地答错）。所以 `Consensus` 只声明它能声明的：多个采样/多个引擎之间**是否收敛**。正确性由下游的基准答案或人工裁决负责。**把一致性包装成正确性，是最容易被发现的夸大。**

### 2.8 为什么性能报告用分位数而非平均值（v1.1.0）

平均值会掩盖长尾——这是 Google《Tail at Scale》（Dean & Barroso, CACM 2013）确立的共识：生产系统看的是分位数。

- `polaris bench` 报 **p50/p90/p99**（最近秩次定义，可手算复现，与 numpy `percentile` 的线性插值不同但对整数秩次更直观）；
- 吞吐用 SPEC/MLPerf 通用定义：总操作数 / 墙钟耗时；
- 多轮取中位数（Google Benchmark 的聚合策略）抵抗抖动；
- **必带 `env` 指纹**（Python/numpy/OS/CPU）——缺了它，任何性能数字都不可复现。

**mock 引擎的 dispatch 延迟只代表框架开销**，不含模型推理成本。输出里显式标注 `model_inference_included`，不让人误读。

---

## 3. 十一条门禁判据（可独立复现）

### 3.1 运行时门禁（每次 dispatch 强制过检）

| 门禁 | 判据 | 违反后果 |
|---|---|---|
| **G1 确定性** | `fingerprint(a) == fingerprint(b)`（重放比对，SHA-256 前 64bit） | `GateViolation`，dispatch 拒绝返回 |
| **G2 有限性** | 非空 / 长度 ≤ max_tokens×8 / 词边界正则无 `\bnan\b\|\binf\b` | 同上 |
| **G3 契约** | `validate_result`：EngineResult 类型 / text 非空 / engine 已注册 / 无 U+FFFD / tokens 长度合理 | 同上 |

### 3.2 自证门禁（测试体系）

| 门禁 | 判据 | 违反后果 |
|---|---|---|
| **P1 契约 fuzz** | 10 种畸形输入（空、4096 长、emoji、控制符、SQL 注入字样、NaN 字样…）不抛异常且契约完整 | 测试红 |
| **P3 调度** | 四情形：kind 优先级 / preferred 生效 / 空注册表诚实失败 / max_tokens 约束 | 测试红 |
| **P5 审计** | 篡改历史条目 / 序号断裂 / 重放载荷干净 / 落盘重载 | 测试红 |
| **P7 零重写** | 每模块 600 字符内出处声明 / vendor LICENSE 保留 / 抽样哈希 / `cmd_train` 被复用且无自写训练循环 | 测试红 |
| **P8 引擎联邦** | 预设库键唯一且字段完整 / 未知预设抛 KeyError / 显式 URL 优先于环境变量 / 离线 `available()` 为 False | 测试红 |
| **P9 共识** | agreement ∈ [0,1] / 1 ≤ support ≤ total / 全票 ⇒ agreement == 1.0 / 全不同 ⇒ agreement == 1/n / 平局打破与顺序无关 | 测试红 |
| **P10 编译器** | K1 程序非空（未命中抛 NoIntentMatch）/ K2 指令存在且 family 一致 / K3 探针两次逐位相同 / K4 编译期干运行通过 / K5 指纹只依赖指令序列 | 测试红 |
| **P11 基准** | p50 ≤ p90 ≤ p99 / 吞吐为正且 calls 自洽 / 输出含 env 指纹 / JSON 可序列化 | 测试红 |

**为什么共识门禁要测「全不同 ⇒ agreement == 1/n」**：只测上界（≤1）会放过一个把 agreement 恒返回 1.0 的实现——那是最危险的假阳性。测下界与退化情形，才能证明这个数真的在测量。

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
| 6 | `BuiltinEngineAdapter(model_dir=None)` 会**静默拾取**工作区的 `.polaris_model`，导致 `allow_autotrain=False` 仍加载真实模型（`degraded=False`），语义与调用方意图相反 | P1 fuzz | 仅在 `allow_autotrain=True` 时才自动拾取；否则 `model_dir=None` ⇒ 诚实占位 |
| 7 | `polaris/cli.py` 改写头部后丢失「零重写 · 出处」声明（P7a 机检直接红） | P7a | 把出处声明提到 docstring 首行，确保落在前 600 字符内 |
| 8 | `polaris engines --all` 表格固定列宽 34，sglang 的 health 串（38 字符）溢出并与 env 列黏连 | CLI 实跑 | 列宽放宽至 42 + 超长截断 |

**第 1 条是教科书级案例**：防御性检查本身成为缺陷源——子串匹配把「用户原文」当「引擎泄漏」。修正后判据更精确（词边界），且职责分层（契约层管结构、门禁层管语义）。

**第 6 条最值得记**：这不是「测试红了改测试」，而是**语义 bug 被测试抓到**。`allow_autotrain=False` 意为「我不训练」，实现却偷偷用了一个已存在的模型 —— 用户拿到的产出不是他声明的那条路径。诚实的降级必须连「模型来源」都如实反映。修法是让「禁止训练」真正禁止一切隐式加载。

---

## 5. 性能特征（诚实口径）

Polaris 自身是薄编排层（单次 dispatch 的调度+门禁+审计开销 < 1ms，实测 0.3–0.8ms）；系统性能由被编排件决定：

| 路径 | 实测（本机，CPU） | 口径 |
|---|---|---|
| ISA 全量门禁 1024 条 + 390 性质断言 | ~2.5 s | `polaris isa` |
| 内置引擎现场训练 120 步 | ~21 s | ifai TinyTrainer，3 层 128 维 |
| 内置引擎 dispatch（模型就绪） | ~2.3 s/16 token | 逐 token 前向，NumPy |
| llama.cpp dispatch | 取决于模型与硬件 | 不重写，继承 llama.cpp 性能 |
| ISA 算子吞吐（探针向量） | ~2.2×10⁵ ops/s | `polaris bench`（纯算子，无模型推理） |
| 意图编译（含编译期干运行） | ~1.4 ms/程序 | `polaris bench`，宽度 2 |

Polaris 不做任何「加速自研引擎」的宣称——那是被复用件的功劳，诚实归位。

**基准数字的边界**：`polaris bench` 输出必带 `env` 指纹；ISA 吞吐是**本机**数字（受 numpy 版本、CPU、BLAS 影响），跨机比较必须连 env 一起看。dispatch 段若跑在 mock 引擎上，只反映框架开销（`model_inference_included` 字段如实标注引擎类型）。

---

## 6. 路线图

- [x] L11/L12 引擎层 + 操作系统层（注册/调度/门禁/审计/编排）
- [x] llama.cpp 发现式接入 + 三级降级
- [x] 75 项测试全绿 + 1024 ISA 全量门禁
- [x] **v1.1.0**：引擎联邦（10 个世界顶级引擎预设，一份 OpenAI 兼容契约）
- [x] **v1.1.0**：共识层（Self-Consistency + 多引擎交叉验证，agreement 可测量）
- [x] **v1.1.0**：意图编译器（自然语言 → ISA 程序，编译期干运行 + K1–K5 门禁）
- [x] **v1.1.0**：可复现基准套件（分位数 + 吞吐 + env 指纹）
- [x] **v1.1.0**：159 项测试全绿（P8–P11 新增四条门禁）
- [ ] 多机审计汇聚（账本外接日志系统）
- [ ] 编译器扩展：类型推导 + 常量折叠 + 更多家族路由（当前是规则确定性子集）
- [ ] 共识层扩展：加权投票 + 语义相似度（当前用 Jaccard bigram）
- [ ] 引擎联邦扩展：流式输出（SSE）与批量推理
