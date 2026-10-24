# 变更日志 · CHANGELOG

本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

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
