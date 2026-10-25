# 复现指南 · REPRODUCE

> 北极星 POLARIS · 一键复现（环境 / 命令 / 预期输出）
> 作者：晨星 (Morningstar) · Apache-2.0

本项目的复现承诺：**同一条命令，在任何机器上得到同一结论**。以下输出取自 2026-10-26 Windows + Python 3.13.12 + numpy 2.5.4 实测；其余平台数值路径逐位一致（确定性由 ifai 全局 PCG64 RNG 保证）。

---

## 1. 环境准备（约 2 分钟）

```bash
git clone https://github.com/CJX0712/polaris-flagship-ai.git
cd polaris-flagship-ai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

依赖清单（全部）：`numpy>=1.24`（硬依赖）、`pytest>=7.0`（dev）。**无需 GPU、无需网络、无需 llama.cpp**。

## 2. 环境自检（约 5 秒）

```bash
polaris doctor
```

预期输出（关键行）：

```
  polaris 版本 : 1.1.0
  numpy       : 2.x.x
  ifai(vendor): 0.8.0 (third_party/ifai)
  引擎[OK]    : ifai-builtin (builtin-flagship)
  引擎[--]    : llama.cpp (world-class-external)      # 未启动 llama.cpp 时此为正常
  引擎[OK]    : mock (mock)
  引擎联邦     : 10 个世界顶级引擎预设（OpenAI 兼容契约）
  联邦发现     : 0 个在线 （无外部引擎，走内置降级）
  ISA         : 1024 微指令 / 32 家族
  结论: 环境就绪（无外部引擎时系统经内置引擎降级运行，功能不减、语义如实标记）
```

## 3. ISA 全量门禁（约 3 秒）

```bash
polaris isa
```

预期输出（尾部）：

```
结果: total=1024 ok=1024 failed=0 properties=390/390 skipped=26
```

## 4. 测试全套（约 25 秒）

```bash
pytest tests/
```

预期：`159 passed`（含 3 项真实训练集成测试）。

```bash
POLARIS_SKIP_SLOW=1 pytest tests/     # 快速通道：156 passed, 3 skipped
```

## 5. 端到端演示（约 30 秒，含现场训练）

```bash
polaris demo
```

预期输出（关键行）：

```
第 3 步 · 全量微指令门禁（冒烟 + 数学性质断言）
  总条数 1024 · 通过 1024 · 失败 0
  性质断言: 390/390 (skipped=26)

第 4 步 · 端到端 dispatch（调度 + 生成 + 门禁 + 审计）
  训练完成: 120 步, 用时 ~21s, 末损 ~1.11
  引擎: ifai-builtin (degraded=False)
  门禁: ['G2=PASS', 'G3=PASS', 'G1=PASS']

第 5 步 · 哈希链审计账本（防篡改验证）
  账本校验: PASS - chain verified: 2 entries
```

## 5b. v1.1.0 新增能力的复现

```bash
polaris engines --all        # 10 个预设表格（llama.cpp/vllm/sglang/ollama/lmstudio/tgi/mlx/tensorrt-llm/nim/openai-compatible）
polaris engines --discover   # 离线时输出「未发现任何在线引擎」——这是正确行为，不是失败
polaris compile "归一化后 softmax" --run 1 2 3
```

预期（compile）：

```
  1. smax.softmax            family=smax   arity=1
  2. norm.cosine_norm        family=norm   arity=1
gates  : ['K1=PASS', 'K2=PASS', 'K3=PASS', 'K4=PASS', 'K5=PASS']
```

```bash
polaris consensus "1+1=?" --mode self -k 3 --max-tokens 8
```

预期：打印 `mode / agreement / pairwise / gates / engines / output` 六行，`gates` 为 `['C1=PASS', 'C2=PASS', 'C3=PASS', 'C4=PASS']`。

```bash
polaris bench --isa-ops 16 --dispatch-iters 2 --audit-iters 20
```

预期：输出 JSON，含 `env`（python/numpy/os/machine）、`isa`、`dispatch`、`audit`、`compile` 四段与 `disclaimer`。**数字随机器变化属正常**——判据是结构完整、`env` 在场、`ops_per_sec > 0`。

## 6. 接入真实世界顶级引擎（可选，质量路径）

```bash
# 任选其一（全部 OpenAI 兼容，Polaris 零改代码接入）：
llama-server -m <your-model>.gguf --port 8080 &     # llama.cpp
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000    # vLLM
ollama serve                                          # Ollama
sglang.launch_server --model-path <model>             # SGLang

polaris engines --discover          # 报告在线引擎
polaris doctor                      # 引擎[OK]: <在线引擎名>
polaris dispatch "你好"              # 调度理由显示选中的引擎
```

也可用环境变量指定地址：`POLARIS_LLAMA_CPP_URL`、`POLARIS_VLLM_URL`、`POLARIS_SGLANG_URL`、`POLARIS_OLLAMA_URL` 等。

## 7. 复现性保证

| 层面 | 保证 | 机检 |
|---|---|---|
| ISA 数值 | ifai 全局 PCG64 RNG，固定 seed 跨进程逐位一致 | `polaris isa` 1024/1024 |
| 训练 | ifai TinyTrainer 显式 seed，初始化/数据/采样全确定 | 集成测试 `test_determinism_bitwise` |
| dispatch | G1 门禁强制重放指纹比对 | `test_dispatch_mock_green` |
| 审计 | 哈希链，篡改即断 | `test_tamper_breaks_chain` |
| 零重写 | 出处声明 + vendor 哈希 | `TestP7Provenance` |
| 编译器 | 探针上两次执行逐位相同；指纹只依赖指令序列 | `TestK3Determinism` / `TestK5FingerprintStable` |
| 共识 | 固定 seed ⇒ 两次共识指纹逐位相同；平局打破与顺序无关 | `TestSelfConsistency.test_seed_sequence_is_deterministic_c5` |
| 引擎联邦 | 离线恒不可用；未知预设抛 KeyError | `TestOpenAICompatEngine` |

## 8. 已知平台差异（如实记录）

- 训练/推理耗时随 CPU 性能浮动（数值结果不变）；
- `polaris bench` 的**性能数字**随机器变化（这是基准的定义）；**结构与判据**跨机一致（`env` 必在、吞吐为正、分位数单调）；
- Windows 路径分隔符在审计 JSONL 中不出现在载荷内（载荷只含哈希/枚举/浮点），跨平台账本可互验。
