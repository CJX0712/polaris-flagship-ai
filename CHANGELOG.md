# 变更日志 · CHANGELOG

本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [1.1.0] - 2026-10-26

### 新增

**引擎联邦**（`polaris/engines/federation.py`）
- `OpenAICompatEngine`：唯一的 OpenAI Completions 兼容 HTTP 适配器（urllib，零依赖）；
- `PRESETS`：10 个世界顶级引擎预设（llama.cpp / vLLM / SGLang / Ollama / LM Studio / TGI / MLX / TensorRT-LLM / NIM / openai-compatible），各带默认地址、健康端点、环境变量、引用 URL、能力声明；
- `build_engine` / `preset_table` / `autodiscover`（探测式发现，离线返回空，绝不伪造在线）；
- `LlamaCppAdapter` 改为继承 `OpenAICompatEngine` —— **不再有第二套 HTTP 代码**。

**共识层**（`polaris/runtime/consensus.py`）
- `Consensus.self_consistency`：Self-Consistency 多样本投票（Wang et al. 2022, arXiv:2203.11171）；
- `Consensus.cross_engine`：多引擎交叉验证（FlagshipAI 交叉验证哲学）；
- `agreement`（最高票占比）/ `mean_pairwise`（Jaccard bigram 两两均值）/ 确定性平局打破（同票取字典序，与顺序无关）；
- C1–C4 可证门禁。

**意图编译器**（`polaris/compiler.py`）
- 自然语言 → 1024 微指令 ISA 程序：前端关键词路由 → IR → **编译期干运行**（LLVM `opt` pass 思想）→ 后端执行；
- `compile_intent` / `route` / `family_candidates` / `extract_numbers` / `ISAProgram`；
- rank 合法化（1-D → 2-D，LLVM type legalization 同构）；
- K1–K5 可证门禁；未命中任何家族抛 `NoIntentMatch`（该走 LLM，不该硬编）。

**基准套件**（`polaris/bench.py`）
- `bench_isa`（算子吞吐）/ `bench_dispatch`（端到端分位数）/ `bench_audit`（哈希链写吞吐）/ `bench_compile`；
- 分位数用**最近秩次**定义（p50/p90/p99），多轮取中位数（Google Benchmark 聚合策略）；
- 输出必带 `env` 指纹（Python/numpy/OS/CPU），缺它数字不可复现。

**CLI**
- 新增 `polaris engines --all` / `polaris engines --discover` / `polaris consensus` / `polaris compile` / `polaris bench`；
- `polaris doctor` 增加引擎联邦与 ISA 统计。

### 修复（开发期被自家门禁抓出的缺陷）

- `BuiltinEngineAdapter(model_dir=None)` 静默拾取工作区 `.polaris_model`，致 `allow_autotrain=False` 仍加载真实模型（语义与调用方意图相反）→ 仅在 `allow_autotrain=True` 时才自动拾取；
- `polaris/cli.py` 头部丢失「零重写 · 出处」声明 → 提到 docstring 首行（P7a 机检红）；
- `polaris engines --all` 表格列宽溢出致 env 列黏连 → 放宽列宽 + 超长截断。

### 验证（全部实测）

- **159/159 测试通过**（156 快速通道 + 3 真实训练集成测试）；
- ISA 门禁 1024/1024 + 数学性质断言 390/390；
- 哈希链审计验证 PASS；
- 门禁体系由 P1–P7 扩至 **P1–P11**（新增 P8 联邦 / P9 共识 / P10 编译器 / P11 基准）。

### 文档

README、ARCHITECTURE（新增联邦/编译/共识三节数据流）、DESIGN（新增 2.5–2.8 决策 + 缺陷记录 + 路线图）、REFERENCES（新增第 1b 节方法学出处）、CHANGELOG 全套更新。

---

## [1.0.0] - 2026-10-25

### 新增（北极星首航）

**L11 引擎层**（`polaris/engines/`）
- `EngineAdapter` 契约 + `EngineResult` 标准化产出（含 `fingerprint()`）；
- `LlamaCppAdapter`：llama.cpp 官方 server 的 OpenAI 兼容 HTTP 适配（标准库 urllib，零依赖），发现式启动（`POLARIS_LLAMA_CPP_URL` / 默认 8080）；
- `BuiltinEngineAdapter`：ifai 内置旗舰，无模型时调用 ifai 官方 `cmd_train` 现场训练（零重写）；
- `MockEngine`：确定性测试替身（如实标记 `degraded=True`）。

**L12 操作系统层**（`polaris/runtime/`）
- `EngineRegistry`：引擎注册表（名字 → 适配器唯一真相源）；
- `Scheduler`：约束调度（确定性过滤 → kind 优先级 world-class-external > builtin > mock → 偏好 → 诚实失败 `NoEngineAvailable`）；
- `gates`：G1 确定性（重放指纹）/ G2 有限性（词边界 NaN/Inf 判据）/ G3 契约，失败即 `GateViolation`；
- `AuditLog`：哈希链审计账本（RFC 6962 同款思想），追加-only、篡改即断、可重放、JSONL 落盘；
- `PolarisOS` 门面：`register_engine / dispatch / run_isa / isa_stats / isa_gates / audit_tail / audit_verify`。

**交付面**
- CLI（`polaris`）：`doctor / engines / dispatch / isa / train-tiny / demo / audit`；
- 端到端演示（`polaris demo`）：与 dispatch 同一代码路径；
- vendor ifai v0.8.0 全栈于 `third_party/ifai/`（Apache-2.0 原样保留）。

**验证（全部实测）**
- 75/75 测试通过（含真实训练集成测试）；
- ISA 门禁 1024/1024 + 数学性质断言 390/390（26 条显式豁免并注明理由，继承上游）；
- 哈希链审计验证 PASS；
- 七条 Polaris 门禁（P1–P7）机检就位。

### 修复（开发期被自家门禁抓出的缺陷）

- `validate_result` NaN 子串检查误伤回显型引擎 → 下沉 G2 + 词边界正则；
- `runtime/__init__.py` 导出不全 → 补全；
- 三模块缺出处声明 → 补齐（P7a）；
- orchestrator 审计 `prompt_fp` 逻辑丑陋 → 修正；
- `polaris isa` properties 统计键名与上游契约不一致 → 对齐。

### 文档

README、ARCHITECTURE、DESIGN、API、REFERENCES、REPRODUCE、NOTICE、CHANGELOG、CITATION 全套。

---

## [0.1.0] - 2026-10-25（内部）

- 舰队缺口分析（JS 侧已有 NEXUS/APEX-Ω，Python 侧操作系统层空白）；
- 架构决策：ifai vendor + llama.cpp HTTP 适配 + 原创编排层。
