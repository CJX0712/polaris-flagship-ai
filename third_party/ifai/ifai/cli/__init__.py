"""L9 命令行接口 (CLI) — `ifai`

子命令 (≥10): train, demo, generate, chat, serve, eval, search, agent,
tokenize, version, doctor, selftest。覆盖 R-L9-01。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .. import __version__, rng
from ..models.config import ModelConfig
from ..models.tokenizer import Tokenizer
from ..models.transformer import Transformer, load_model
from ..train.trainer import TinyTrainer
from ..engine.sampler import Sampler
from ..agent.tools import default_registry
from ..agent.react import ReActAgent
from ..agent.memory import MemoryStore
from ..retrieval.persist import PersistentIndex, load_index

DEFAULT_MODEL_DIR = "./ifai_model"
MICRO_CORPUS = (
    "人工智能是计算机科学的一个分支。模型通过学习数据来提升能力。"
    "变压器网络由注意力机制构成。注意力让模型关注重要的信息。"
    "训练需要大量的数据和计算资源。推理是把训练好的模型用于预测。"
    "分词把文本切分成更小的单元。嵌入把符号映射到向量空间。"
    "采样从模型的输出分布中选择下一个词。检索从知识库中找到相关资料。"
    "智能体可以调用工具来完成任务。记忆帮助智能体积累经验。"
)


def _load(model_dir):
    tok = Tokenizer.from_json(open(f"{model_dir}/tokenizer.json", encoding="utf-8").read())
    model = load_model(model_dir)
    return model, tok


def cmd_train(args):
    rng.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    corpus = MICRO_CORPUS
    if args.corpus:
        corpus = open(args.corpus, encoding="utf-8").read()
    # 先构造分词器再决定模型词表, 杜绝 vocab_size 软目标陷阱:
    # Tokenizer.train 的 vocab_size 仅限制 merge 数, 基础字符数不设上限,
    # 若语料实际字符/总词表数 > args.vocab, tok.encode 会产生 ≥ args.vocab 的 id,
    # 而 ModelConfig(vocab_size=args.vocab) 的 embed 矩阵只有 args.vocab 行 → 越界。
    # 故模型词表取 max(args.vocab, tok.vocab_size), 始终 ≥ 分词器实际词表。
    tok = Tokenizer(vocab_size=args.vocab).train(corpus)
    model_vocab = max(args.vocab, tok.vocab_size)
    cfg = ModelConfig(vocab_size=model_vocab, d_model=args.d_model,
                      n_layers=args.layers, n_heads=args.heads,
                      n_kv_heads=args.kv_heads, ffn_mult=args.ffn_mult,
                      max_seq_len=args.seq_len)
    tok.to_json(f"{args.out}/tokenizer.json")
    ids = tok.encode(corpus)
    trainer = TinyTrainer(cfg)
    t0 = time.perf_counter()
    losses = trainer.train(np.array(ids), steps=args.steps, lr=args.lr)
    dt = time.perf_counter() - t0
    trainer.save(args.out)
    nparams = trainer.count_params()
    print(f"训练完成: {args.steps} 步, 用时 {dt:.1f}s, 末损 {losses[-1]:.4f}")
    print(f"模型已保存至 {args.out} (参数量 {nparams})")


def cmd_demo(args):
    if not _exists(f"{args.model}/manifest.json"):
        print("未找到模型, 先训练一个微型模型...")
        cmd_train(argparse.Namespace(seed=0, corpus=None, vocab=512, d_model=128,
                                     layers=3, heads=4, kv_heads=2, ffn_mult=4,
                                     seq_len=128, steps=120, lr=3e-3, out=args.model))
    model, tok = _load(args.model)
    sampler = Sampler(seed=args.seed)
    ids = tok.encode(args.prompt or "人工智能") or [0]
    out = []
    for _ in range(args.max_new):
        logits = model.forward(np.array([ids]))[0, -1]
        nxt = sampler.greedy(logits) if args.greedy else sampler.sample(logits, temperature=args.temp)
        ids.append(nxt)
        out.append(tok.decode([nxt]))
        if tok.id2token.get(nxt) == "<eos>":
            break
    print(("".join(out)).strip() or "(空)")


def cmd_generate(args):
    model, tok = _load(args.model)
    sampler = Sampler(seed=args.seed)
    ids = tok.encode(args.prompt) or [0]
    out = []
    for _ in range(args.max_new):
        logits = model.forward(np.array([ids]))[0, -1]
        nxt = sampler.sample(logits, temperature=args.temp, top_p=args.top_p, top_k=args.top_k)
        ids.append(nxt)
        out.append(tok.decode([nxt]))
    print("".join(out))


def cmd_chat(args):
    model, tok = _load(args.model)
    sampler = Sampler(seed=args.seed)
    print("进入聊天 (输入 exit 退出):")
    while True:
        q = input("> ") if not args.seed else "你好"
        if q.strip() == "exit":
            break
        ids = tok.encode(q) or [0]
        for _ in range(args.max_new):
            logits = model.forward(np.array([ids]))[0, -1]
            nxt = sampler.sample(logits, temperature=args.temp)
            ids.append(nxt)
            tok_txt = tok.decode([nxt])
            if tok.id2token.get(nxt) == "<eos>":
                break
            print(tok_txt, end="", flush=True)
        print()


def cmd_serve(args):
    """启动 OpenAI 兼容服务 (L7)。

    v0.8.0 新增:
      - ``--backend`` 选择 L3 推理后端 (naive / paged / chunked / speculative),
        四者输出逐 token 相同, 差异在性能特征;
      - ``--guardrail`` 开启输入/输出护栏 (INV-38), 命中返回 OpenAI 风格
        错误体; 此前护栏仅测试可达, 服务路径完全不过护栏。
    """
    from ..serve.api import OpenAIServer, serve as run_server
    model, tok = _load(args.model)
    guard = None
    if getattr(args, "guardrail", False):
        from ..serve.guardrail import default_guardrail
        guard = default_guardrail()
    srv = OpenAIServer(model, tok, seed=args.seed,
                       guardrail=guard,
                       engine_backend=getattr(args, "backend", "naive"))
    print(f"服务启动于 http://{args.host}:{args.port} "
          f"(后端 {srv.engine_backend}, 护栏 {'开' if guard else '关'}) "
          f"(Ctrl+C 退出)")
    run_server(srv, args.host, args.port)


def cmd_eval(args):
    from ..eval.perplexity import perplexity, perplexity_manual
    from ..eval.harness import GSM8K_Mini
    model, tok = _load(args.model)
    seq = tok.encode(MICRO_CORPUS)[:64] or list(range(8))
    p1 = perplexity(model, seq)
    p2 = perplexity_manual(model, seq)
    print(f"困惑度: {p1:.4f} (手工对照 {p2:.4f}, Δ={abs(p1-p2):.2e})")
    gsm = GSM8K_Mini()
    def ans_fn(q):
        model2, tok2 = model, tok
        ids = tok2.encode(q) or [0]
        for _ in range(12):
            lg = model2.forward(np.array([ids]))[0, -1]
            nxt = Sampler(seed=0).greedy(lg)
            ids.append(nxt)
            if tok2.id2token.get(nxt) == "<eos>":
                break
        return "".join(tok2.decode(ids))
    res = gsm.evaluate(ans_fn)
    print(f"GSM8K-mini 管线正确性: {res['correct']}/{res['total']} (无准确率门槛, 仅验证管线)")


def cmd_search(args):
    """混合检索 (BM25+Dense+RRF, Cormack et al. 2009) — v0.7.14 接线:
    此前仅 BM25 单路且中文按空格分词永远匹配不到 (审计缺口 1), 现按字符级
    分词 + PersistentIndex 双路融合, PRD US-05 从假变真。``--index`` 指定
    ``ifai index`` 产物目录 (落盘索引), 缺省对内置语料即时构建。"""
    if args.index:
        idx = load_index(args.index)
        src = f"混合索引 {args.index}"
    else:
        docs = [d.strip() for d in MICRO_CORPUS.split("。") if d.strip()]
        idx = PersistentIndex().fit([list(d) for d in docs], texts=docs)
        src = "内置语料 (字符级混合检索)"
    hits = idx.search(list(args.query), top_k=args.top_k)
    print(f"[{src}] BM25+Dense+RRF, top-{args.top_k}")
    if not hits:
        print("(无命中)")
    for h in hits:
        print(f"[{h.score:.3f}] {h.text}")


def cmd_index(args):
    """构建并落盘混合检索索引 (BM25+Dense+RRF) — PRD US-05 承诺的
    ``ifai index`` 子命令 (v0.7.14 接线, 此前不存在)。"""
    docs = [d.strip() for d in MICRO_CORPUS.split("。") if d.strip()]
    idx = PersistentIndex(dim=args.dim).fit([list(d) for d in docs], texts=docs)
    path = idx.save(args.out)
    print(f"混合索引已构建: {len(docs)} 条文档 (字符级 BM25+Dense+RRF) → {path}")


def cmd_agent(args):
    def fake_llm(messages):
        # 演示用确定性策略: 遇到"计算"则调用计算器; 看到工具结果则 Finish
        user = messages[-1]["content"]
        if user.startswith("Observation:"):
            return f"Finish: {user.split('Observation:', 1)[1].strip()}"
        if "计算" in user or any(c in user for c in "+-*/"):
            expr = user.split("计算")[-1].strip()
            return f'Thought: 需要计算。\nAction: calculator(expression="{expr}")'
        return "Finish: 已回答 (演示智能体)"
    reg = default_registry()
    agent = ReActAgent(fake_llm, reg, max_steps=4, memory=MemoryStore())
    print(agent.run(args.question, reflexion=True))


def cmd_tokenize(args):
    tok = Tokenizer(vocab_size=args.vocab).train(MICRO_CORPUS)
    ids = tok.encode(args.text)
    print("ids:", ids)
    print("decode:", tok.decode(ids))


def cmd_version(_):
    print(f"ifai {__version__} · by 晨星 (Morningstar) / CJX0712")


def cmd_microisa(args):
    """L10 微指令层 —— 1024 条可执行微指令的浏览与自检入口。

    零重写: L10 ISA 移植自同门旗舰 `tianji-ai` (MIT, 作者晨星), 本仓在
    移植之上追加了数学性质断言层 (softmax 归一性 / L2 严格单位范数 /
    KL 非负 / Flash≡朴素 / NS 奇异值带 / Top-K 掩码个数 / RoPE 保范数 …)。

    子命令:
        --list              列出 32 个家族及其语义类型
        --family <名称>     列出某家族的 32 条成员
        --run <指令名>      执行单条微指令并打印输出
        (默认)              跑全量 1024 条门禁
    """
    from ..l10 import FAMILIES, FAMILY_KIND, MICROOPS, run_all

    if args.list:
        print(f"L10 微指令 ISA: {len(MICROOPS)} 条 / {len(FAMILIES)} 家族")
        for fam in sorted(FAMILIES):
            kind = FAMILY_KIND.get(fam, "?")
            label = "独立算子" if kind == "core" else "参数扫描"
            print(f"  {fam:<8} [{label}] {len(FAMILIES[fam])} 条")
        return

    if args.family:
        fam = args.family
        if fam not in FAMILIES:
            print(f"[失败] 未知家族: {fam}")
            print(f"可用: {', '.join(sorted(FAMILIES))}")
            sys.exit(2)
        kind = FAMILY_KIND.get(fam, "?")
        print(f"{fam} ({kind}, {len(FAMILIES[fam])} 条):")
        for name in FAMILIES[fam]:
            print(f"  {fam}.{name}")
        return

    if args.run:
        key = args.run
        if key not in MICROOPS:
            print(f"[失败] 未知微指令: {key}")
            sys.exit(2)
        op = MICROOPS[key]
        g = np.random.default_rng(args.seed)
        x = g.standard_normal((4, 8))
        y = g.standard_normal((4, 8))
        out = op.fn(x) if op.arity == 1 else op.fn(x, y)
        arr = np.asarray(out)
        print(f"{key}  [家族 {op.family} · arity={op.arity}]")
        print(f"  输出形状: {tuple(arr.shape)}  dtype: {arr.dtype}")
        flat = arr.reshape(-1)
        preview = flat[:8] if flat.size else arr
        print(f"  前 8 个值: {np.round(preview, 6)}")
        return

    report = run_all(seed=args.seed)
    props = report["properties"]
    print(f"L10 门禁: 冒烟+有限性 {report['ok']}/{report['total']} · "
          f"数学性质 {props['passed']}/{props['checked']} "
          f"(豁免 {props['skipped']} 条)")
    if report["failed"]:
        print(f"失败 {len(report['failed'])} 条:")
        for name, detail in report["failed"][:20]:
            print(f"  ❌ {name}: {detail}")
        sys.exit(1)
    print("L10 全绿 ✅ 1024/1024 (32 家族 × 32 成员)")


def cmd_doctor(_):
    import numpy as np
    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__)
    print("ifai:", __version__)
    print("GPU: 否 (CPU 推理路径, 100% numpy)")


def cmd_selftest(args):
    """零重写自验证: 运行三项静态审计 + (可选) pytest 黄金测试。

    INV-41 (分层零反向依赖) / INV-42 (≥25 处世界顶级出处) / R-X-03 (文档完备),
    并额外校验「微指令分解覆盖 PRD 全部 42 条 INV (含 X 层 INV-41/42)」。
    ``--inv INV-17`` 可单查一条不变量: 定位 tests/ 下引用该 INV 的测试文件并运行。
    """
    import re
    import subprocess
    if getattr(args, "inv", None):
        _run_single_inv(args.inv)
        return
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    audits = [
        ("分层审计 (INV-41)", "scripts/check_layering.py"),
        ("零重写出处审计 (INV-42)", "scripts/audit_no_reinvention.py"),
        ("文档完备审计 (R-X-03)", "scripts/check_docs.py"),
        ("L10 微指令门禁 (1024 条)", "scripts/check_l10.py"),
    ]
    all_ok = True
    print("=== ifai selftest (零重写自验证) ===")
    for label, rel in audits:
        path = os.path.join(repo, rel)
        if not os.path.exists(path):
            print(f"[跳过] {label}: 脚本缺失 {rel}")
            continue
        rc = subprocess.run([sys.executable, path], cwd=repo).returncode
        ok = rc == 0
        all_ok &= ok
        print(f"[{'通过' if ok else '失败'}] {label}")

    # 不变量覆盖: 微指令分解必须覆盖 PRD 的全部 42 条 INV (含 X 横切层 INV-41/42)
    gen = os.path.join(repo, "scripts", "gen_micro_instructions.py")
    if os.path.exists(gen):
        proc = subprocess.run([sys.executable, gen], cwd=repo,
                              capture_output=True, text=True)
        covered: set = set()
        m = re.search(r"覆盖 INV:\s*\[([0-9, ]+)\]", proc.stdout)
        if m:
            covered = {int(x) for x in m.group(1).split(",") if x.strip()}
        expect = set(range(1, 43))
        ok = proc.returncode == 0 and covered >= expect
        all_ok &= ok
        print(f"[{'通过' if ok else '失败'}] 不变量覆盖 (42/42 INV): "
              f"已覆盖 {len(covered)} 条"
              + ("" if ok else f", 缺失 {sorted(expect - covered)}"))

    if getattr(args, "pytest", False):
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=repo).returncode
        ok = rc == 0
        all_ok &= ok
        print(f"[{'通过' if ok else '失败'}] 黄金测试 (pytest)")
    print("=== 结论:", "ALL GREEN ✅" if all_ok else "存在失败 ❌", "===")
    sys.exit(0 if all_ok else 1)


def _run_single_inv(inv: str) -> None:
    """按 INV 编号定位并运行验证它的测试 (ARCHITECTURE.md 映射原则的 CLI 落点)。"""
    import re
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    m = re.fullmatch(r"INV[-_]?(\d{1,2})", inv.strip(), re.IGNORECASE)
    if not m:
        print(f"[失败] 无法解析 INV 编号: {inv} (示例: --inv INV-17)")
        sys.exit(2)
    num = int(m.group(1))
    if not 1 <= num <= 42:
        print(f"[失败] INV 编号超出 01..42: {num}")
        sys.exit(2)
    tests_dir = os.path.join(repo, "tests")
    hits = []
    for root, _dirs, files in os.walk(tests_dir):
        for fn in files:
            if not fn.endswith(".py") or fn == "__init__.py":
                continue
            p = os.path.join(root, fn)
            try:
                text = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if num in {int(x) for x in re.findall(r"INV-(\d{1,2})\b", text)}:
                hits.append(os.path.relpath(p, repo))
    if not hits:
        print(f"[失败] INV-{num:02d} 在 tests/ 下无任何测试引用 (违反可追溯性门禁)")
        sys.exit(1)
    print(f"=== INV-{num:02d} → {len(hits)} 个测试文件 ===")
    for h in hits:
        print("  ", h)
    rc = subprocess.run([sys.executable, "-m", "pytest", "-q"] + hits, cwd=repo).returncode
    print("=== 结论:", "PASS ✅" if rc == 0 else "FAIL ❌", f"(INV-{num:02d}) ===")
    sys.exit(rc)


def _exists(p):
    import os
    return os.path.exists(p)


def main():
    ap = argparse.ArgumentParser(prog="ifai", description="创新旗舰 AI 命令行")
    sub = ap.add_subparsers(dest="cmd")
    for name, fn in [("train", cmd_train), ("demo", cmd_demo), ("generate", cmd_generate),
                     ("chat", cmd_chat), ("serve", cmd_serve), ("eval", cmd_eval),
                     ("search", cmd_search), ("index", cmd_index), ("agent", cmd_agent),
                     ("tokenize", cmd_tokenize), ("version", cmd_version),
                     ("microisa", cmd_microisa),
                     ("doctor", cmd_doctor), ("selftest", cmd_selftest)]:
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
    sub.choices["selftest"].add_argument("--pytest", action="store_true",
                                        help="额外运行 pytest 黄金测试")
    sub.choices["selftest"].add_argument("--inv", default=None, dest="inv",
                                        help="单查一条不变量, 如 --inv INV-17")
    sub.choices["microisa"].add_argument("--list", action="store_true",
                                        help="列出 32 个微指令家族")
    sub.choices["microisa"].add_argument("--family", default=None,
                                        help="列出某家族的成员")
    sub.choices["microisa"].add_argument("--run", default=None,
                                        help="执行单条微指令并打印输出")
    sub.choices["microisa"].add_argument("--seed", type=int, default=0)
    # 通用参数
    for name in ["train"]:
        sub.choices[name].add_argument("--seed", type=int, default=0)
        sub.choices[name].add_argument("--corpus", default=None)
        sub.choices[name].add_argument("--vocab", type=int, default=512)
        sub.choices[name].add_argument("--d-model", type=int, default=128, dest="d_model")
        sub.choices[name].add_argument("--layers", type=int, default=3)
        sub.choices[name].add_argument("--heads", type=int, default=4)
        sub.choices[name].add_argument("--kv-heads", type=int, default=2, dest="kv_heads")
        sub.choices[name].add_argument("--ffn-mult", type=int, default=4, dest="ffn_mult")
        sub.choices[name].add_argument("--seq-len", type=int, default=128, dest="seq_len")
        sub.choices[name].add_argument("--steps", type=int, default=120)
        sub.choices[name].add_argument("--lr", type=float, default=3e-3)
        sub.choices[name].add_argument("--out", default=DEFAULT_MODEL_DIR)
    for name in ["demo", "generate", "chat", "serve", "eval"]:
        sub.choices[name].add_argument("--model", default=DEFAULT_MODEL_DIR)
        sub.choices[name].add_argument("--seed", type=int, default=0)
    for name in ["demo", "generate", "chat"]:
        sub.choices[name].add_argument("--max-new", type=int, default=32, dest="max_new")
        sub.choices[name].add_argument("--temp", type=float, default=0.8)
        sub.choices[name].add_argument("--greedy", action="store_true")
    sub.choices["generate"].add_argument("--prompt", default="人工智能")
    sub.choices["generate"].add_argument("--top-p", type=float, default=0.9, dest="top_p")
    sub.choices["generate"].add_argument("--top-k", type=int, default=0, dest="top_k")
    sub.choices["demo"].add_argument("--prompt", default="人工智能")
    sub.choices["serve"].add_argument("--host", default="127.0.0.1")
    sub.choices["serve"].add_argument("--port", type=int, default=8000)
    sub.choices["serve"].add_argument("--backend", default="naive",
                                      choices=["naive", "paged", "chunked", "speculative"],
                                      help="L3 推理后端 (输出逐 token 一致)")
    sub.choices["serve"].add_argument("--guardrail", action="store_true",
                                      help="启用输入/输出护栏 (INV-38)")
    sub.choices["search"].add_argument("--query", default="注意力")
    sub.choices["search"].add_argument("--top-k", type=int, default=5, dest="top_k")
    sub.choices["search"].add_argument("--index", default=None, dest="index",
                                       help="已构建的混合索引目录 (ifai index 产物); 缺省用内置语料")
    sub.choices["index"].add_argument("--out", default="./ifai_index")
    sub.choices["index"].add_argument("--dim", type=int, default=256)
    sub.choices["agent"].add_argument("--question", default="计算 (3+5)*2")
    sub.choices["tokenize"].add_argument("--text", default="人工智能 模型")
    sub.choices["tokenize"].add_argument("--vocab", type=int, default=512)

    args = ap.parse_args()
    if not getattr(args, "cmd", None):
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
