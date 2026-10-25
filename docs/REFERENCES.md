# 出处清单 · REFERENCES

> 北极星 POLARIS 复用的世界顶级成果，逐条注明
> 作者：晨星 (Morningstar) · Apache-2.0

Polaris 铁律：**每一行代码、每一个模块都必须站在世界顶级成果之上**。本清单是它的机检化声明——`tests/unit/test_zero_rewrite.py` 会审计每个模块的出处 docstring。

---

## 1. 引擎与运行时（L11/L12 编排契约）

| 组件 | 出处 | 许可证 | Polaris 的复用方式 |
|---|---|---|---|
| llama.cpp server | ggerganov/llama.cpp, `tools/server`（OpenAI 兼容 HTTP 面） | MIT | 经 `LlamaCppAdapter` 接入，不重写推理 |
| OpenAI Completions API | OpenAI 公开 API 规范 | — | 请求/响应字段契约；**也是全部目标引擎的共同契约** |
| vLLM | vllm-project/vllm（`/v1/chat/completions` OpenAI 兼容 server） | Apache-2.0 | 预设 `vllm`，零代码接入 |
| SGLang | sgl-project/sglang（OpenAI 兼容 server，`/health_generate`） | Apache-2.0 | 预设 `sglang` |
| Ollama | ollama/ollama（`/v1/chat/completions` + `/api/version`） | MIT | 预设 `ollama` |
| LM Studio | lmstudio-ai（OpenAI 兼容端点） | 商业 | 预设 `lmstudio` |
| HuggingFace TGI | huggingface/text-generation-inference（OpenAI 兼容） | Apache-2.0 | 预设 `tgi` |
| MLX | ml-explore/mlx-lm（OpenAI 兼容 server） | MIT | 预设 `mlx` |
| TensorRT-LLM | NVIDIA/TensorRT-LLM（OpenAI 兼容 trtllm-server） | Apache-2.0 | 预设 `tensorrt-llm` |
| NVIDIA NIM | NVIDIA NIM LLM 容器（`/v1/health/ready`） | 商业 | 预设 `nim` |
| NEXUS 引擎契约 | CJX0712/nexus-flagship（作者晨星，MIT） | MIT | `EngineAdapter`/Dispatch 的 Python 对应物 |
| 注册表模式 | Python `importlib`/`setuptools entry_points`/`pluggy` 生态 | — | `EngineRegistry` |
| LiteLLM / vLLM 引擎选择 | LiteLLM provider registry；vLLM engine 注册 | — | 约束调度策略参考 |
| 跨引擎一致性验证 | FlagshipAI（FlashAttention 与 PyTorch 交叉验证，同门旗舰，作者晨星） | — | 共识层 `cross_engine` 的哲学来源 |

## 1b. 编译器与共识（v1.1.0 方法学）

| 组件 | 出处 | 许可证 | Polaris 的复用方式 |
|---|---|---|---|
| 编译器前端/IR/Pass/后端 | 经典编译器结构（前端 → IR → 优化 Pass → 后端） | — | `polaris/compiler.py` 的骨架 |
| LLVM `opt` pass 思想（编译期剪枝） | LLVM 官方文档 | Apache-2.0 with LLVM Exception | 编译期干运行（K4）：跑不通的候选当场剔除 |
| LLVM type legalization（rank 提升） | LLVM LangRef | 同上 | 1-D → 2-D 合法化 |
| Self-Consistency | Wang et al. 2022, *Self-Consistency Improves Chain of Thought Reasoning*, arXiv:2203.11171 | 论文 | `Consensus.self_consistency`（多样本投票） |
| 交叉验证 | 同上 + FlagshipAI 交叉验证实践 | — | `Consensus.cross_engine`（多引擎交叉验证） |
| Jaccard 相似度 | Jaccard 1901（集合相似度系数） | 论文 | bigram 集合的相似度度量 |
| 分位数（tail latency） | Dean & Barroso 2013, *The Tail at Scale*, CACM | 论文 | p50/p90/p99 报告口径 |
| 吞吐基准定义 | SPEC / MLPerf 通用定义（总操作数 ÷ 墙钟耗时） | — | `bench_isa` |
| 基准多次聚合取中位数 | Google Benchmark 的聚合策略 | Apache-2.0 | `bench.measure` |
| 词法分析（关键词路由） | 经典 `lex` 词法分析器思想 | — | `compiler.route` |
| 环境指纹 | 可复现基准的通行做法（记录 Python/依赖/OS/CPU） | — | `bench.env_info` |

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
| 门禁编号体系（G/C/K/B 前缀） | Polaris 自有（作者晨星），语义见 docs/DESIGN.md 第 3 节 |
| 确定性边界测试（退化情形） | 测「全票 / 全不同」两个极端，防止实现恒返回 1.0 的假阳性 |

---

## 致谢

以上成果的作者们。Polaris 的存在只为证明一件事：**把世界顶级零件正确地装在一起，并且造一把尺子证明装对了**——这本身就是工程价值。
