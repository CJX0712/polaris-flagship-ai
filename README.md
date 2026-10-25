# 北极星 POLARIS · 晨星旗舰 AI 操作系统

> **引擎联邦（N 个世界顶级引擎）× 1024 微指令 ISA × 意图编译器 × 共识层 × 可证门禁 × 治理审计**
> 零重写 · 集大成 · 可运行 · 可复现 · 可验证

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-159%20passed-success.svg)](tests/)
[![ISA](https://img.shields.io/badge/ISA-1024%20%E5%BE%AE%E6%8C%87%E4%BB%A4-blueviolet.svg)](docs/ARCHITECTURE.md)
[![Gates](https://img.shields.io/badge/gates-P1..P11%20%E5%85%A8%E7%BB%BF-success.svg)](docs/DESIGN.md)
[![Engines](https://img.shields.io/badge/engines-10%20world--class%20presets-ff69b4.svg)](polaris/engines/federation.py)
[![Zero-Rewrite](https://img.shields.io/badge/zero--rewrite-audited-success.svg)](tests/unit/test_zero_rewrite.py)

**作者：晨星 (Morningstar)** · GitHub [@CJX0712](https://github.com/CJX0712) · Apache-2.0

---

## 这是什么

**Polaris（北极星）** 是一个把「世界顶级开源引擎」编排成可编程 AI 系统的**操作系统**：

- **智力**用世界顶级推理引擎 —— 不重写一行推理代码，经 **OpenAI Completions 兼容契约**接入 **10 种**世界顶级引擎（llama.cpp / vLLM / SGLang / Ollama / LM Studio / TGI / MLX / TensorRT-LLM / NIM / 任意 OpenAI 兼容云端），**起服即接入，零改代码**；
- **全栈**用同门旗舰 **ifai**（innovation-flagship-ai v0.8.0, Apache-2.0）—— 十层全栈 + **1024 条可执行微指令 ISA**，以源码形式 vendor 于 `third_party/ifai/`，原样保留 LICENSE/NOTICE；
- **前端**用**意图编译器** —— 自然语言 → 1024 微指令 ISA 程序（前端路由 → IR → 编译期干运行 → 后端执行，对齐经典编译器结构）；
- **可靠性**用**共识层** —— Self-Consistency 多样本投票 + 多引擎交叉验证，把「一致性」变成**可测量的数**（agreement ∈ [0,1]）而不是形容词；
- **尺子**用**可证门禁** —— G1 确定性 / G2 有限性 / G3 契约 + P1–P11 自证体系，每次 dispatch 强制过检，不过即拦截；
- **黑匣子**用**哈希链审计账本** —— 追加-only，任何篡改使后续哈希全部断裂，可独立重放验证。

Polaris 自己只写「没人做的那一层」：引擎注册表、约束调度器、运行时门禁、治理审计、元编排、意图编译器、共识层。**不重造轮子，只把已被证明的轮子装成一辆能跑的车**。

```python
from polaris import PolarisOS
from polaris.engines import build_engine, BuiltinEngineAdapter

os_ = PolarisOS()
os_.register_engine(build_engine("llama.cpp"))      # 世界顶级外部引擎（在线时优先）
os_.register_engine(build_engine("vllm"))           # 换个引擎只改字符串，契约不变
os_.register_engine(BuiltinEngineAdapter())         # 内置旗舰（离线确定性回退）

r = os_.dispatch("你好", max_tokens=32)              # 调度 → 生成 → 门禁 → 审计
print(r.text)                                        # 经 G1/G2/G3 放行的产出
print(r.meta["dispatch"]["gates"])                   # ['G2=PASS', 'G3=PASS', 'G1=PASS']
```

---

## 引擎联邦（一份契约统治 N 个世界顶级引擎）

所有目标引擎都实现了 **OpenAI Completions 兼容**接口 —— 这是**行业事实标准**，不是 Polaris 发明的协议。因此适配层只需一份：

```python
from polaris.engines import build_engine, autodiscover

e = build_engine("sglang")                  # 预设名 → 适配器（自动读 POLARIS_SGLANG_URL）
e = build_engine("vllm", base_url="http://gpu-node:8000")   # 显式地址优先于环境变量
for live in autodiscover():                 # 扫描全部预设，报告当前在线者
    print(live.name, live.base_url)
```

| 预设 | 默认地址 | 健康端点 | 环境变量 |
|---|---|---|---|
| `llama.cpp` | http://127.0.0.1:8080 | /health, /v1/models | `POLARIS_LLAMA_CPP_URL` |
| `vllm` | http://127.0.0.1:8000 | /health, /v1/models | `POLARIS_VLLM_URL` |
| `sglang` | http://127.0.0.1:30000 | /health_generate, /health, /v1/models | `POLARIS_SGLANG_URL` |
| `ollama` | http://127.0.0.1:11434 | /api/version, /v1/models | `POLARIS_OLLAMA_URL` |
| `lmstudio` | http://127.0.0.1:1234 | /v1/models | `POLARIS_LMSTUDIO_URL` |
| `tgi` | http://127.0.0.1:8080 | /health, /v1/models | `POLARIS_TGI_URL` |
| `mlx` | http://127.0.0.1:8080 | /v1/models, /health | `POLARIS_MLX_URL` |
| `tensorrt-llm` | http://127.0.0.1:8000 | /health, /v1/models | `POLARIS_TRTLLM_URL` |
| `nim` | http://127.0.0.1:8000 | /v1/health/ready, /health | `POLARIS_NIM_URL` |
| `openai-compatible` | https://api.openai.com/v1 | /v1/models, /models | `POLARIS_OPENAI_COMPAT_URL` |

**诚实降级，永不假装**：`available()` 绝不谎报在线；引擎缺席时调度器自动切下一级，并在审计中如实记录 `degraded` 标记。

---

## 三级引擎体系（调度降级链）

| 引擎 | 类别 | 说明 | 缺席时行为 |
|---|---|---|---|
| `llama.cpp` / `vllm` / … | world-class-external | 真实大模型推理 | 调度器自动切下一级，`available()` 如实返回 False |
| `ifai-builtin` | builtin-flagship | ifai 全栈（微型模型，现场训练秒级） | 系统最后保证，永远可用 |
| `mock` | mock | 测试替身 | 仅测试/演示使用 |

调度策略（NEXUS Dispatch 契约的 Python 合流）：**确定性请求优先 + 世界顶级外部引擎优先 + 约束匹配 + 诚实失败**（无引擎满足时抛 `NoEngineAvailable`，绝不静默造假）。

---

## 意图编译器（自然语言 → 1024 微指令程序）

```python
from polaris import compile_intent

p = compile_intent("把这个向量归一化再做 softmax", width=2)
p.steps        # [Step(op='smax.softmax'), Step(op='norm.cosine_norm')]
p.gates        # K1..K5 全 PASS
p.run([1.0, 2.0, 3.0, 4.0])
```

编译器按**经典编译器结构**落地，不发明新理论：前端关键词路由 → IR（线性微指令序列，对应 LLVM IR 形态）→ **编译期干运行**（LLVM `opt` 的 pass 思想：跑不通的候选当场剔除并记录原因）→ 后端执行。

**诚实边界**：本编译器是**规则驱动的确定性子集**——做关键词路由，不做开放域语义理解。`"归一化"` 能编译，`"帮我写首诗"` 抛 `NoIntentMatch`（该交给 LLM 引擎，不该硬编）。

---

## 共识层（一致性是数，不是形容词）

```python
from polaris.runtime.consensus import Consensus

cr = Consensus().self_consistency(engine, "2+2=?", k=5, temperature=0.8)
cr.agreement      # 0.0–1.0，最高票占比
cr.mean_pairwise  # 两两相似度均值（Jaccard bigram）
cr.gates          # C1..C4 门禁

cr2 = Consensus().cross_engine([engine_a, engine_b], "2+2=?")
cr2.agreement     # 多引擎交叉验证（FlagshipAI 的交叉验证哲学）
```

方法学取自 **Self-Consistency**（Wang et al., arXiv:2203.11171）与 **Cross-Engine Verification**，相似度用 **Jaccard bigram**（标准集合相似度）。

---

## 十二层架构

| 层 | 名称 | 来源 | Polaris 动作 |
|---|---|---|---|
| L0–L10 | 算子内核 / 数值 / 架构 / 推理引擎 / 训练 / 检索 / 智能体 / 服务 / 评测 / 应用 / **1024 微指令 ISA** | **ifai v0.8.0**（vendor） | 原样复用，零修改 |
| L11 | 引擎层 | llama.cpp / vLLM / SGLang / Ollama …（10 预设） + ifai（内置） + mock | 薄适配（HTTP/契约翻译） |
| L12 | 操作系统层 | **Polaris 原创** | 注册表 / 调度 / 门禁 / 审计 / 编排 / 共识 |

```
┌──────────────────────────────────────────────────────────────┐
│ L12  PolarisOS 门面   register_engine / dispatch / run_isa    │
│      + 共识 / 编译 / 基准（工具层，polaris.compiler 等）      │
├──────────────────────────────────────────────────────────────┤
│ L11  引擎联邦          llama.cpp ⇄ vLLM ⇄ SGLang ⇄ Ollama   │
│                       ⇄ ifai ⇄ mock（同一 OpenAI 兼容契约） │
├──────────────────────────────────────────────────────────────┤
│ L10  微指令 ISA        1024 条可执行微指令（ifai，15 core    │
│      （ifai vendor）   + 17 sweep，390 条数学性质断言）       │
├──────────────────────────────────────────────────────────────┤
│ L0–L9 ifai 全栈        算子/训练/检索/Agent/OpenAI 服务…     │
└──────────────────────────────────────────────────────────────┘
```

---

## 十一条可证门禁（Polaris 自证体系）

| 编号 | 门禁 | 判据 | 机检 |
|---|---|---|---|
| P1 | 引擎契约 fuzz | 畸形输入（空/超长/unicode/注入）不崩、契约完整 | `test_engines.py` |
| P2 | 注册表真相源 | 名字→适配器唯一映射，可用性如实反映 | `test_runtime.py` |
| P3 | 约束调度 | 确定性优先、外部优先、偏好生效、无满足则诚实失败 | `test_runtime.py` |
| P4 | 运行时门禁 | G1 确定性（重放指纹）/ G2 有限性 / G3 契约 | `test_runtime.py` |
| P5 | 哈希链审计 | 追加-only、篡改即断（序号+哈希双断裂）、可重放 | `test_runtime.py` |
| P6 | ISA 复用完整性 | 恰好 1024 条、32 家族、core/sweep 分开记账、全量门禁绿 | `test_isa.py` |
| P7 | 零重写审计 | 每模块出处声明 + vendor 完整性 + 关键符号来自上游 | `test_zero_rewrite.py` |
| P8 | 引擎联邦 | 预设库结构完整、未知预设 KeyError、离线不谎报在线 | `test_federation.py` |
| P9 | 共识门禁 | agreement∈[0,1]、1≤support≤total、全票=1.0、平局顺序无关 | `test_consensus.py` |
| P10 | 编译器门禁 | K1 非空 / K2 指令存在 / K3 确定性 / K4 干运行 / K5 指纹稳定 | `test_compiler.py` |
| P11 | 基准门禁 | p50≤p90≤p99、吞吐为正、含 env 指纹、JSON 可序列化 | `test_bench.py` |

实测（2026-10-26，本机复现）：**159/159 测试通过 · ISA 门禁 1024/1024 + 性质断言 390/390 · 审计链验证 PASS**。

---

## 快速开始

```bash
git clone https://github.com/CJX0712/polaris-flagship-ai.git
cd polaris-flagship-ai

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                             # 仅 numpy 硬依赖（ifai 已 vendor）

polaris doctor        # 环境自检：numpy / ifai / 引擎联邦 / ISA
polaris engines --all # 10 个世界顶级引擎预设库
polaris engines --discover   # 扫描当前在线的引擎
polaris isa           # 1024 微指令统计 + 全量门禁（冒烟 + 390 性质断言）
polaris compile "归一化后 softmax" --run 1 2 3   # 意图 → ISA 程序（K1..K5）
polaris consensus "你好" --mode cross            # 共识：Self-Consistency / 交叉验证
polaris bench         # 可复现性能基准（分位数 + 吞吐 + 环境指纹）
polaris demo          # 端到端演示（含现场训练微型模型，CPU 秒级）
polaris dispatch "你好" --max-tokens 32
```

零 GPU、零网络（llama.cpp 缺席时自动走内置引擎，语义如实标记 `degraded`）。

### 接入真实世界顶级引擎

```bash
# 1) 任选其一起服（全部 OpenAI 兼容，Polaris 无需改代码）
llama-server -m ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080   # llama.cpp
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000                          # vLLM
ollama serve                                                                 # Ollama

# 2) Polaris 自动发现（也可用 POLARIS_LLAMA_CPP_URL 等环境变量指定）
polaris engines --discover
polaris dispatch "你好"          # 调度理由将显示选中的引擎
```

Polaris 不打包任何模型权重；接什么模型由你决定（这是操作系统的本分）。

---

## 测试与验证

```bash
pytest tests/                       # 159 项全绿（含真实训练集成测试）
POLARIS_SKIP_SLOW=1 pytest tests/   # 快速通道（跳过现场训练）
```

| 测试层 | 覆盖 |
|---|---|
| `tests/unit/test_engines.py` | 契约 fuzz（10 种畸形输入 × 引擎）、确定性、llama.cpp 发现语义 |
| `tests/unit/test_runtime.py` | 注册表 / 调度四情形 / 门禁三判据 / 审计篡改断裂 |
| `tests/unit/test_isa.py` | ISA 统计诚实性、单条直执、1024 全量门禁 |
| `tests/unit/test_orchestrator.py` | dispatch 端到端、GateViolation 拦截、审计联动 |
| `tests/unit/test_zero_rewrite.py` | 出处声明机检、vendor 完整性、无重复造轮子 |
| `tests/unit/test_federation.py` | 预设库结构、未知预设诚实失败、离线不谎报在线 |
| `tests/unit/test_consensus.py` | 投票/平局/Jaccard/agreement 边界、指纹确定性 |
| `tests/unit/test_compiler.py` | K1–K5 编译器门禁、指纹稳定性、程序可执行 |
| `tests/unit/test_bench.py` | 分位数定义、吞吐自洽、env 指纹、JSON 可序列化 |
| `tests/integration/` | 内置引擎真实训练 + dispatch + 逐位确定性 |

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 十二层映射、模块依赖、数据流、舰队关系 |
| [docs/DESIGN.md](docs/DESIGN.md) | 设计决策、十一条门禁判据、被测试抓出的真实缺陷 |
| [docs/API.md](docs/API.md) | 全部公开接口用法与示例 |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 世界顶级方案出处（论文/项目/标准）逐条清单 |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | 一键复现指南（环境/命令/预期输出） |
| [NOTICE](NOTICE) | 第三方组件与算法出处（法律声明） |
| [CHANGELOG.md](CHANGELOG.md) | 版本历史 |
| [CITATION.md](CITATION.md) | 引用格式 |

---

## 零重写原则（本仓铁律）

1. **世界顶级推理引擎**（llama.cpp / vLLM / SGLang / Ollama / …）：不重写，只经 OpenAI 兼容契约适配接入；
2. **世界顶级 AI 全栈**（ifai，同门旗舰）：不重写，以源码 vendor 原样复用；
3. **ISA 数学判据**：不发明，全部取自公开论文与 IEEE 754 标准；
4. **编译器结构**：不发明，经典编译器前端/IR/Pass/后端；
5. **共识方法学**：不发明，Self-Consistency（arXiv:2203.11171）+ Jaccard 相似度；
6. **性能方法学**：不发明，Tail at Scale 的分位数 + SPEC/MLPerf 的吞吐定义；
7. **Polaris 自身只写「没人做的编排 + 自证」**：引擎注册/调度/门禁/审计/元编排/编译器/共识；
8. **机检闭环**：`test_zero_rewrite.py` 对每条铁律自动审计，违规即红。

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
