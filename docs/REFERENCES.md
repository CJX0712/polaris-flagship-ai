# 出处清单 · REFERENCES

> 北极星 POLARIS 复用的世界顶级成果，逐条注明
> 作者：晨星 (Morningstar) · Apache-2.0

Polaris 铁律：**每一行代码、每一个模块都必须站在世界顶级成果之上**。本清单是它的机检化声明——`tests/unit/test_zero_rewrite.py` 会审计每个模块的出处 docstring。

---

## 1. 引擎与运行时（L11/L12 编排契约）

| 组件 | 出处 | 许可证 | Polaris 的复用方式 |
|---|---|---|---|
| llama.cpp server | ggerganov/llama.cpp, `tools/server`（OpenAI 兼容 HTTP 面） | MIT | 经 `LlamaCppAdapter` 接入，不重写推理 |
| OpenAI Completions API | OpenAI 公开 API 规范 | — | 请求/响应字段契约 |
| NEXUS 引擎契约 | CJX0712/nexus-flagship（作者晨星，MIT） | MIT | `EngineAdapter`/Dispatch 的 Python 对应物 |
| 注册表模式 | Python `importlib`/`setuptools entry_points`/`pluggy` 生态 | — | `EngineRegistry` |
| LiteLLM / vLLM 引擎选择 | LiteLLM provider registry；vLLM engine 注册 | — | 约束调度策略参考 |

## 2. 门禁判据（数学标准，不发明新数学）

| 判据 | 出处 |
|---|---|
| 有限性（无 NaN/Inf） | IEEE 754 浮点标准；`math.isfinite` 语义 |
| 确定性（同输入同输出） | 可复现计算的一般定义；Bitwise Reproducibility 实践 |
| 哈希链账本 | Git 对象模型；RFC 6962（Certificate Transparency Merkle 日志） |
| 规范化 JSON 哈希 | JCS（JSON Canonicalization Scheme, RFC 8785）同款思想：排序键 + 无空白 |

## 3. 全栈与 ISA（L0–L10，vendor 自 ifai v0.8.0）

| 组件 | 出处 | 许可证 |
|---|---|---|
| ifai 全栈 | CJX0712/innovation-flagship-ai v0.8.0（作者晨星） | Apache-2.0（vendor 保留） |
| 1024 微指令 ISA | 移植自 CJX0712/tianji-ai（MIT，作者晨星） | MIT |
| FlashAttention-2 在线 softmax | Dao et al. 2023, *FlashAttention-2* | 论文 |
| RoPE | Su et al. 2021, *RoFormer* | 论文 |
| RMSNorm | Zhang & Sennrich 2019 | 论文 |
| SwiGLU | Shazeer 2020, *GLU Variants Improve Transformer* | 论文 |
| GQA | Ainslie et al. 2023 | 论文 |
| MoE 稀疏路由 | Fedus et al. 2022, *Switch Transformer* | 论文 |
| PagedAttention / 连续批处理 | Kwon et al. 2023（vLLM）；Yu et al. 2022（Orca） | 论文 |
| 投机解码 | Leviathan et al. 2023；Chen et al. 2023 | 论文 |
| AdamW | Loshchilov & Hutter 2019 | 论文 |
| LoRA | Hu et al. 2021 | 论文 |
| QAT（STE 伪量化） | Jacob et al. 2018 | 论文 |
| Newton–Schulz 正交化 | Keller Jordan 2024（Muon 优化器） | 公开实现 |
| BM25 | Robertson & Zaragoza 2009 | 论文 |
| RRF 融合 | Cormack et al. 2009 | 论文 |
| QFR（ifai 原创，RRF 严格超集） | ifai L5，INV-43 逐位相等锁定 | Apache-2.0 |
| ReAct / Reflexion / MemGPT | Yao et al. 2022；Shinn et al. 2023；Packer et al. 2023 | 论文 |
| BPE | Sennrich et al. 2016 | 论文 |

## 4. 测试方法论

| 方法 | 出处 |
|---|---|
| xUnit 测试替身（mock） | xUnit Test Patterns, Meszaros 2007 |
| 属性测试 / 契约 fuzz | QuickCheck（Claessen & Hughes 2000）思想 |
| 黄金测试（解析对照） | ifai/AXIOM/NEXUS 的「不变量」方法论（作者晨星） |
| 零重写机检审计 | ifai `scripts/audit_no_reinvention.py` 同款思想 |

---

## 致谢

以上成果的作者们。Polaris 的存在只为证明一件事：**把世界顶级零件正确地装在一起，并且造一把尺子证明装对了**——这本身就是工程价值。
