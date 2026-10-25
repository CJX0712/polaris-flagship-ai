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

### OpenAICompatEngine（引擎联邦的统一适配器 · v1.1.0）

```python
from polaris.engines import OpenAICompatEngine, build_engine, autodiscover, preset_table, PRESETS

e = OpenAICompatEngine("vllm")                       # 预设名 → 适配器
e = build_engine("vllm", base_url="http://gpu:8000") # 显式地址优先于环境变量
e = build_engine("sglang")                           # 环境变量 POLARIS_SGLANG_URL 优先于预设默认

e.base_url      # 生效地址
e.capabilities  # 能力声明（按实例拷贝，实例间不串味）
e.available()   # 探测健康端点；离线恒为 False（绝不谎报在线）
e.complete(prompt, max_tokens=64, temperature=0.0, seed=0)  # POST {base}/v1/chat/completions
e.describe()    # {name, kind, base_url, preset, cite, available, ...}

for live in autodiscover(timeout=1.0):  # 探测全部预设，报告在线者；离线返回 []
    print(live.name, live.base_url)

preset_table()   # 表格化预设（文档/CLI 消费）
PRESETS.keys()   # 10 个预设名
```

**覆盖优先级**：显式 `base_url` > 环境变量（如 `POLARIS_VLLM_URL`）> 预设默认地址。
**未知预设名**抛 `KeyError`（诚实失败，绝不退化到「随便一个引擎」）。

### LlamaCppAdapter（世界顶级外部引擎）

```python
from polaris.engines import LlamaCppAdapter

LlamaCppAdapter(base_url=None, timeout=60.0, model_hint="llama.cpp-server")
```

- 继承 `OpenAICompatEngine`（**唯一一份 HTTP 实现**）；
- `base_url` 缺省读 `POLARIS_LLAMA_CPP_URL`，再退 `http://127.0.0.1:8080`；
- `available()`：`GET {base}/health`（llama-server 官方健康端点）；
- `complete()`：`POST {base}/v1/chat/completions`（OpenAI 兼容契约）。

### BuiltinEngineAdapter（内置旗舰）

```python
from polaris.engines import BuiltinEngineAdapter

BuiltinEngineAdapter(model_dir=None, seed=0, allow_autotrain=True, train_steps=120)
```

- `model_dir` 有模型 → 驱动 ifai SDK 生成；
- `model_dir=None` 且 `allow_autotrain=True` → 复用工作区 `.polaris_model`（若有），否则调用 ifai 官方 `cmd_train` 现场训练（MICRO_CORPUS，120 步约 21s CPU）；
- `model_dir=None` 且 `allow_autotrain=False` → **不隐式加载任何模型**，返回诚实占位（`degraded=True`）；
- 恒可用（系统「永远能跑」的最后保证）。

### MockEngine（测试替身）

```python
MockEngine(prefix="mock")   # complete("x") -> "mock:x"，degraded=True
```

---

## 2b. 意图编译器（v1.1.0）

```python
from polaris import compile_intent
from polaris.compiler import route, family_candidates, extract_numbers, NoIntentMatch, ProgramError

p = compile_intent("把这个向量归一化再做 softmax", width=2, max_families=2,
                   scalar=1.0, require_deterministic=True)

p.steps            # [Step(op='smax.softmax', family='smax', arity=1, note=...), ...]
p.matches          # [(family, score), ...] 路由打分（解释「为什么选这些指令」）
p.dropped          # [{op, family, reason}, ...] 编译期干运行剔除的候选及原因
p.deterministic    # True（探针上两次执行逐位一致）
p.gates            # [{'gate': 'K1', 'passed': True, ...}, ... K1..K5]
p.fingerprint()    # sha256("op1|op2|...")[:16]，只依赖指令序列

p.run([1.0, 2.0, 3.0, 4.0])              # 执行
p.run([1.0, 2.0], trace=True)            # → (结果, 每步快照列表)
p.as_dict()                              # JSON 可序列化

route("计算自注意力")           # [('attn', 6), ...] 家族 → 打分，按 (-score, family) 排序
family_candidates("smax")      # ['smax.softmax', ...] 首选在前，其余字典序
extract_numbers("取前 3 个")    # [3.0]
```

**Raises**：`NoIntentMatch`（未命中任何 ISA 家族 —— 该走 LLM 引擎，不该硬编）、`ProgramError`（干运行全灭 / 确定性不满足）。

---

## 2c. 共识层（v1.1.0）

```python
from polaris.runtime.consensus import (
    Consensus, consensus_gates, majority_vote,
    jaccard, similarity, bigrams, mean_pairwise_similarity,
)

cr = Consensus().self_consistency(engine, "2+2=?", k=5, temperature=0.8, seed=0)
cr = Consensus().cross_engine([engine_a, engine_b], "2+2=?")

cr.method          # 'self-consistency' | 'cross-engine'
cr.text            # 胜出文本
cr.agreement       # 0.0–1.0，最高票占比 = support / total
cr.support         # 最高票数（1 ≤ support ≤ total）
cr.total           # 样本/引擎总数
cr.mean_pairwise   # 两两相似度均值（单样本时 1.0）
cr.votes           # [(文本, 票数), ...]
cr.reports         # 每样本明细（含 seed / latency_ms）
cr.gates           # [{'gate': 'C1', 'passed': True}, ... C1..C4]
cr.fingerprint()   # 固定 seed ⇒ 逐位可复现
cr.as_dict()       # JSON 可序列化

majority_vote(["A", "A", "B"])   # ('A', 2, [('A', 2), ('B', 1)])  平局按字典序 ⇒ 顺序无关
similarity("人工智能", "机器学习")  # Jaccard bigram 相似度 ∈ [0,1]
```

---

## 2d. 基准套件（v1.1.0）

```python
from polaris import bench

r = bench.run_all(os_, isa_ops=128, dispatch_iterations=10, audit_iterations=200)
r["env"]            # {python, numpy, os, machine, processor} —— 环境指纹
r["isa"]            # {ops, rounds, calls, ops_per_sec, ms_per_call, note}
r["dispatch"]       # {p50, p90, p99, mean, min, max, samples_ms, calls, engine, model_inference_included}
r["audit"]          # 同上（哈希链追加吞吐）
r["compile"]        # 同上（含编译期干运行）
r["disclaimer"]     # 数字口径说明（必读）

bench.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 90)   # 最近秩次分位数
bench.measure(lambda: None, repeats=5, iterations=20)
bench.env_info()
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
polaris doctor                  # 环境自检（numpy / ifai / 引擎联邦 / ISA）
polaris engines                 # 已注册引擎清单（JSON）
polaris engines --all           # 10 个世界顶级引擎预设库（表格）
polaris engines --discover      # 探测当前在线的引擎
polaris dispatch "提示" [--max-tokens N] [--temperature T] [--seed S]
polaris consensus "提示" [--mode self|cross] [-k N] [--max-tokens N]
polaris compile "意图" [--width N] [--max-families N] [--run V ...]
polaris bench [--isa-ops N] [--dispatch-iters N] [--audit-iters N]
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
