# API 文档 · API

> 北极星 POLARIS · 全部公开接口
> 作者：晨星 (Morningstar) · Apache-2.0

---

## 1. PolarisOS（操作系统门面）

```python
from polaris import PolarisOS

os_ = PolarisOS(audit_path=None, g1_replay=None)
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `audit_path` | `str \| None` | `None` | 审计账本 JSONL 落盘路径 |
| `g1_replay` | `bool \| None` | `None` | `None`=自动（外部引擎跳过、内置/mock 执行） |

### register_engine(adapter) -> EngineAdapter

注册引擎（重名覆盖）。返回适配器本身，便于链式。

```python
from polaris.engines import LlamaCppAdapter, BuiltinEngineAdapter
os_.register_engine(LlamaCppAdapter(base_url="http://127.0.0.1:8080"))
os_.register_engine(BuiltinEngineAdapter(model_dir="./.polaris_model"))
```

### dispatch(prompt, *, max_tokens=64, temperature=0.0, seed=0, require_deterministic=True, preferred=None) -> EngineResult

端到端：调度 → 生成 → 门禁 → 审计。

| 参数 | 说明 |
|---|---|
| `prompt` | 输入提示 |
| `max_tokens` | 最大生成 token 数 |
| `temperature` | 0 = 贪心（确定性）；>0 = 采样 |
| `seed` | 随机种子（G1 的前提） |
| `require_deterministic` | 是否只考虑声明确定性的引擎（默认 True） |
| `preferred` | 偏好引擎名（满足约束时优先） |

**Raises**：`NoEngineAvailable`（无引擎满足约束）、`GateViolation`（门禁拦截）。

```python
r = os_.dispatch("你好", max_tokens=32)
r.text                          # 经门禁放行的文本
r.meta["dispatch"]["gates"]     # [{'gate': 'G2', 'passed': True, ...}, ...]
r.meta["dispatch"]["reason"]    # 调度理由（可审计）
```

### run_isa(op_name, *args) -> Any

执行一条 1024 微指令 ISA 成员。

```python
os_.run_isa("smax.softmax", [0.1, 0.2, 0.7])   # 行和 = 1.0
os_.run_isa("norm.l2norm", [3.0, 4.0])         # 单位向量
```

### isa_stats() -> dict

```python
{"total_ops": 1024, "families": 32,
 "family_kinds": {"core": 15, "sweep": 17},
 "family_names": ["act", "agg", ...]}
```

### isa_gates() -> dict

跑 ifai L10 全量微指令门禁（冒烟 + 数学性质断言）。

```python
{"total": 1024, "ok": 1024, "failed": [],
 "families": {...}, "properties": {"checked": 390, "passed": 390, "failed": [], "skipped": 26}}
```

### engines() -> list[dict] · audit_tail(n=10) · audit_verify()

引擎自描述 / 审计尾部 / 哈希链校验（`(ok, detail)`）。

---

## 2. 引擎适配器

### LlamaCppAdapter（世界顶级外部引擎）

```python
from polaris.engines import LlamaCppAdapter

LlamaCppAdapter(base_url=None, timeout=60.0, model_hint="llama.cpp-server")
```

- `base_url` 缺省读 `POLARIS_LLAMA_CPP_URL`，再退 `http://127.0.0.1:8080`；
- `available()`：`GET {base}/health`（llama-server 官方健康端点）；
- `complete()`：`POST {base}/v1/completions`（OpenAI 兼容契约）。

### BuiltinEngineAdapter（内置旗舰）

```python
from polaris.engines import BuiltinEngineAdapter

BuiltinEngineAdapter(model_dir=None, seed=0, allow_autotrain=True, train_steps=120)
```

- `model_dir` 有模型 → 驱动 ifai SDK 生成；
- 无模型且 `allow_autotrain` → 调用 ifai 官方 `cmd_train` 现场训练（MICRO_CORPUS，120 步约 21s CPU）；
- 恒可用（系统「永远能跑」的最后保证）。

### MockEngine（测试替身）

```python
MockEngine(prefix="mock")   # complete("x") -> "mock:x"，degraded=True
```

---

## 3. runtime 四件套

```python
from polaris.runtime import EngineRegistry, Scheduler, run_gates, AuditLog

reg = EngineRegistry(); reg.register(MockEngine())
sched = Scheduler(reg)
from polaris.runtime import DispatchRequest
decision = sched.decide(DispatchRequest(prompt="hi", require_deterministic=True))

reports = run_gates(result, registry_names=set(reg.names()), max_tokens=64, replay=None)
# [GateReport(gate='G2', passed=True), GateReport(gate='G3', ...), GateReport(gate='G1', ...)]

log = AuditLog(path="audit.jsonl")
log.append({"intent": "dispatch", "engine": "mock"})
ok, detail = log.verify_chain()   # (True, "chain verified: 1 entries")
```

---

## 4. CLI

```bash
polaris doctor                  # 环境自检
polaris engines                 # 引擎清单（JSON）
polaris dispatch "提示" [--max-tokens N] [--temperature T] [--seed S]
polaris isa [--family smax] [--run smax.softmax]
polaris train-tiny [--out DIR] [--steps N] [--seed S]
polaris demo [--quick]
polaris audit [--verify] [--tail N]
```

退出码：`0` 成功 · `2` dispatch 失败（无引擎/门禁拦截）· `3` ISA 门禁失败 · `4` 审计链断裂。

---

## 5. EngineResult（数据契约）

```python
@dataclass
class EngineResult:
    text: str                      # 生成文本
    engine: str                    # 产出引擎名（必须已注册）
    tokens: list[int] | None       # token 序列（可缺失）
    degraded: bool                 # True = 降级路径产出（如实标记）
    meta: dict                     # 引擎元数据 + dispatch 明细
    def fingerprint(self) -> str   # SHA-256[:16]，G1 与审计消费
```
