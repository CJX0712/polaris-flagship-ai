# 北极星 POLARIS · 晨星旗舰 AI 操作系统

> **真实世界顶级引擎 × 1024 微指令 ISA × 可证门禁 × 治理审计**
> 零重写 · 集大成 · 可运行 · 可复现 · 可验证

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-75%20passed-success.svg)](tests/)
[![ISA](https://img.shields.io/badge/ISA-1024%20%E5%BE%AE%E6%8C%87%E4%BB%A4-blueviolet.svg)](docs/ARCHITECTURE.md)
[![Gates](https://img.shields.io/badge/gates-G1%2FG2%2FG3%20%E5%85%A8%E7%BB%BF-success.svg)](docs/DESIGN.md)
[![Zero-Rewrite](https://img.shields.io/badge/zero--rewrite-audited-success.svg)](tests/unit/test_zero_rewrite.py)

**作者：晨星 (Morningstar)** · GitHub [@CJX0712](https://github.com/CJX0712) · Apache-2.0

---

## 这是什么

**Polaris（北极星）** 是一个把「世界顶级开源引擎」编排成可编程 AI 系统的**操作系统**：

- **智力**用世界顶级推理引擎 **llama.cpp**（ggerganov/llama.cpp, MIT）—— 不重写一行推理代码，经 OpenAI 兼容 HTTP 面接入；
- **全栈**用同门旗舰 **ifai**（innovation-flagship-ai v0.8.0, Apache-2.0）—— 十层全栈 + **1024 条可执行微指令 ISA**，以源码形式 vendor 于 `third_party/ifai/`，原样保留 LICENSE/NOTICE；
- **尺子**用**可证门禁** —— G1 确定性 / G2 有限性 / G3 契约，每次 dispatch 强制过检，不过即拦截；
- **黑匣子**用**哈希链审计账本** —— 追加-only，任何篡改使后续哈希全部断裂，可独立重放验证。

Polaris 自己只写「没人做的那一层」：引擎注册表、约束调度器、运行时门禁、治理审计、元编排。这正是零重写的定义——**不重造轮子，只把已被证明的轮子装成一辆能跑的车**。

```python
from polaris import PolarisOS
from polaris.engines import LlamaCppAdapter, BuiltinEngineAdapter

os_ = PolarisOS()
os_.register_engine(LlamaCppAdapter())          # 世界顶级外部引擎（在线时优先）
os_.register_engine(BuiltinEngineAdapter())     # 内置旗舰（离线确定性回退）

r = os_.dispatch("你好", max_tokens=32)          # 调度 → 生成 → 门禁 → 审计
print(r.text)                                    # 经 G1/G2/G3 放行的产出
print(r.meta["dispatch"]["gates"])               # ['G2=PASS', 'G3=PASS', 'G1=PASS']
```

---

## 三级引擎体系（诚实降级，永不假装）

| 引擎 | 类别 | 说明 | 缺席时行为 |
|---|---|---|---|
| `llama.cpp` | world-class-external | 真实大模型推理（GGUF 量化，CPU/GPU） | 调度器自动切下一级，`available()` 如实返回 False |
| `ifai-builtin` | builtin-flagship | ifai 全栈（微型模型，现场训练秒级） | 系统最后保证，永远可用 |
| `mock` | mock | 测试替身 | 仅测试/演示使用 |

调度策略（NEXUS Dispatch 契约的 Python 合流）：**确定性请求优先 + 世界顶级外部引擎优先 + 约束匹配 + 诚实失败**（无引擎满足时抛 `NoEngineAvailable`，绝不静默造假）。

---

## 十二层架构

| 层 | 名称 | 来源 | Polaris 动作 |
|---|---|---|---|
| L0–L10 | 算子内核 / 数值 / 架构 / 推理引擎 / 训练 / 检索 / 智能体 / 服务 / 评测 / 应用 / **1024 微指令 ISA** | **ifai v0.8.0**（vendor） | 原样复用，零修改 |
| L11 | 引擎层 | llama.cpp（外部）+ ifai（内置）+ mock | 薄适配（HTTP/契约翻译） |
| L12 | 操作系统层 | **Polaris 原创** | 注册表 / 调度 / 门禁 / 审计 / 编排 |

```
┌──────────────────────────────────────────────────────────────┐
│ L12  PolarisOS 门面      register_engine / dispatch / run_isa │
├──────────────────────────────────────────────────────────────┤
│ L11  引擎层              llama.cpp ⇄ ifai ⇄ mock（三级降级）   │
├──────────────────────────────────────────────────────────────┤
│ L10  微指令 ISA          1024 条可执行微指令（ifai，15 core    │
│      （ifai vendor）     + 17 sweep，390 条数学性质断言）       │
├──────────────────────────────────────────────────────────────┤
│ L0–L9 ifai 全栈          算子/训练/检索/Agent/OpenAI 服务…     │
└──────────────────────────────────────────────────────────────┘
```

---

## 七条可证门禁（Polaris 自证体系）

| 编号 | 门禁 | 判据 | 机检 |
|---|---|---|---|
| P1 | 引擎契约 fuzz | 畸形输入（空/超长/unicode/注入）不崩、契约完整 | `test_engines.py` |
| P2 | 注册表真相源 | 名字→适配器唯一映射，可用性如实反映 | `test_runtime.py` |
| P3 | 约束调度 | 确定性优先、外部优先、偏好生效、无满足则诚实失败 | `test_runtime.py` |
| P4 | 运行时门禁 | G1 确定性（重放指纹）/ G2 有限性 / G3 契约 | `test_runtime.py` |
| P5 | 哈希链审计 | 追加-only、篡改即断（序号+哈希双断裂）、可重放 | `test_runtime.py` |
| P6 | ISA 复用完整性 | 恰好 1024 条、32 家族、core/sweep 分开记账、全量门禁绿 | `test_isa.py` |
| P7 | 零重写审计 | 每模块出处声明 + vendor 完整性 + 关键符号来自上游 | `test_zero_rewrite.py` |

实测（2026-10-25，本机复现）：**75/75 测试通过 · ISA 门禁 1024/1024 + 性质断言 390/390 · 审计链验证 PASS**。

---

## 快速开始

```bash
git clone https://github.com/CJX0712/polaris-flagship-ai.git
cd polaris-flagship-ai

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                             # 仅 numpy 硬依赖（ifai 已 vendor）

polaris doctor        # 环境自检：numpy / ifai / 三级引擎 / ISA
polaris isa           # 1024 微指令统计 + 全量门禁（冒烟 + 390 性质断言）
polaris demo          # 端到端演示（含现场训练微型模型，CPU 秒级）
polaris dispatch "你好" --max-tokens 32
```

零 GPU、零网络（llama.cpp 缺席时自动走内置引擎，语义如实标记 `degraded`）。

### 接入真实 llama.cpp（世界顶级智力）

```bash
# 1) 安装 llama.cpp（MIT，ggerganov/llama.cpp）并启动其 OpenAI 兼容 server：
#    macOS/Linux: brew install llama.cpp && llama-server -m model.gguf --port 8080
#    Windows:     见 https://github.com/ggerganov/llama.cpp#build
llama-server -m ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080

# 2) Polaris 自动发现（默认 http://127.0.0.1:8080，也可用 POLARIS_LLAMA_CPP_URL 指定）
polaris dispatch "你好"          # 调度理由将显示选中 llama.cpp
```

Polaris 不打包任何模型权重；接什么模型由你决定（这是操作系统的本分）。

---

## 测试与验证

```bash
pytest tests/                       # 75 项全绿（含真实训练集成测试）
POLARIS_SKIP_SLOW=1 pytest tests/   # 快速通道（跳过现场训练）
```

| 测试层 | 覆盖 |
|---|---|
| `tests/unit/test_engines.py` | 契约 fuzz（10 种畸形输入 × 引擎）、确定性、llama.cpp 发现语义 |
| `tests/unit/test_runtime.py` | 注册表 / 调度四情形 / 门禁三判据 / 审计篡改断裂 |
| `tests/unit/test_isa.py` | ISA 统计诚实性、单条直执、1024 全量门禁 |
| `tests/unit/test_orchestrator.py` | dispatch 端到端、GateViolation 拦截、审计联动 |
| `tests/unit/test_zero_rewrite.py` | 出处声明机检、vendor 完整性、无重复造轮子 |
| `tests/integration/` | 内置引擎真实训练 + dispatch + 逐位确定性 |

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 十二层映射、模块依赖、数据流、舰队关系 |
| [docs/DESIGN.md](docs/DESIGN.md) | 设计决策、七条门禁判据、被测试抓出的真实缺陷 |
| [docs/API.md](docs/API.md) | 全部公开接口用法与示例 |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 世界顶级方案出处（论文/项目/标准）逐条清单 |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | 一键复现指南（环境/命令/预期输出） |
| [NOTICE](NOTICE) | 第三方组件与算法出处（法律声明） |
| [CHANGELOG.md](CHANGELOG.md) | 版本历史 |
| [CITATION.md](CITATION.md) | 引用格式 |

---

## 零重写原则（本仓铁律）

1. **世界顶级推理引擎**（llama.cpp）：不重写，只经 HTTP 适配接入；
2. **世界顶级 AI 全栈**（ifai，同门旗舰）：不重写，以源码 vendor 原样复用；
3. **ISA 数学判据**：不发明，全部取自公开论文与 IEEE 754 标准；
4. **Polaris 自身只写「没人做的编排 + 自证」**：引擎注册/调度/门禁/审计/元编排；
5. **机检闭环**：`test_zero_rewrite.py` 对每条铁律自动审计，违规即红。

Polaris 在晨星旗舰舰队（[apex-flagship](https://github.com/CJX0712/apex-flagship)）中的位置：**Python 侧的操作系统总装**——与 NEXUS（JS 侧操作系统）、TIANJI（混合架构模型）、FlagshipAI（FlashAttention 内核）、AXIOM（JS 尺子）互补，共同构成「造 AI，并造尺子证明它没骗你」的完整闭环。

---

## 引用

```bibtex
@software{polaris2026,
  title  = {Polaris: 晨星旗舰 AI 操作系统},
  author = {晨星 (Morningstar)},
  year   = {2026},
  url    = {https://github.com/CJX0712/polaris-flagship-ai},
  note   = {Apache-2.0}
}
```

## 许可证

[Apache License 2.0](LICENSE) © 2026 晨星 (Morningstar) / CJX0712。
vendor 的 ifai 保留其原始 Apache-2.0 授权（见 `third_party/ifai/LICENSE`）。

> **零重写 · 集大成 · 有尺子 · 闭环开源**
