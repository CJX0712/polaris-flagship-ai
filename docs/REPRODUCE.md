# 复现指南 · REPRODUCE

> 北极星 POLARIS · 一键复现（环境 / 命令 / 预期输出）
> 作者：晨星 (Morningstar) · Apache-2.0

本项目的复现承诺：**同一条命令，在任何机器上得到同一结论**。以下输出取自 2026-10-25 Windows + Python 3.13.12 + numpy 2.5.4 实测；其余平台数值路径逐位一致（确定性由 ifai 全局 PCG64 RNG 保证）。

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
  polaris 版本 : 1.0.0
  numpy       : 2.x.x
  ifai(vendor): 0.8.0 (third_party/ifai)
  引擎[OK]    : ifai-builtin (builtin-flagship)
  引擎[--]    : llama.cpp (world-class-external)      # 未启动 llama.cpp 时此为正常
  引擎[OK]    : mock (mock)
  ISA         : 1024 微指令 / 32 家族
  结论: 环境就绪（无 llama.cpp 时系统经内置引擎降级运行，功能不减、语义如实标记）
```

## 3. ISA 全量门禁（约 3 秒）

```bash
polaris isa
```

预期输出（尾部）：

```
结果: total=1024 ok=1024 failed=0 properties=390/390 skipped=26
```

## 4. 测试全套（约 16 秒）

```bash
pytest tests/
```

预期：`75 passed`。

```bash
POLARIS_SKIP_SLOW=1 pytest tests/     # 快速通道（跳过现场训练集成测试）
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

## 6. 接入真实 llama.cpp（可选，质量路径）

```bash
# 任选其一获取 llama.cpp：
#   macOS:  brew install llama.cpp
#   Linux:  见 https://github.com/ggerganov/llama.cpp#build
#   Windows: Release 页面预编译包
llama-server -m <your-model>.gguf --port 8080 &

polaris doctor                       # 引擎[OK]: llama.cpp
polaris dispatch "你好"               # 调度理由显示选中 llama.cpp
```

## 7. 复现性保证

| 层面 | 保证 | 机检 |
|---|---|---|
| ISA 数值 | ifai 全局 PCG64 RNG，固定 seed 跨进程逐位一致 | `polaris isa` 1024/1024 |
| 训练 | ifai TinyTrainer 显式 seed，初始化/数据/采样全确定 | 集成测试 `test_determinism_bitwise` |
| dispatch | G1 门禁强制重放指纹比对 | `test_dispatch_mock_green` |
| 审计 | 哈希链，篡改即断 | `test_tamper_breaks_chain` |
| 零重写 | 出处声明 + vendor 哈希 | `TestP7Provenance` |

## 8. 已知平台差异（如实记录）

- 训练/推理耗时随 CPU 性能浮动（数值结果不变）；
- Windows 路径分隔符在审计 JSONL 中不出现在载荷内（载荷只含哈希/枚举/浮点），跨平台账本可互验。
