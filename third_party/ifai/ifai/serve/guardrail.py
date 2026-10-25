"""L7 护栏 (Guardrail) — 注入/越权/有害内容拦截与误杀控制

零重写 (集大成):
- **OWASP Top 10 for LLM Applications (2025)**: 风险分类口径 (LLM01 提示注入、
  LLM02 敏感信息泄露、LLM06 过度自主…), 直接映射为本模块的 category。
- **NVIDIA NeMo Guardrails** (Rebedea et al. 2023): "可编程规则护栏"思想 —
  输入/输出两侧各挂一组可配置规则, 命中即给出结构化判定。
- **Llama Guard** (Inan et al. 2023): 按**风险类别** (而非笼统"有害") 输出
  判定, 便于分级处置与审计。
- **Perez & Ribeiro 2022** "Ignore Previous Prompt": 提示注入的攻防基线, 本
  模块的注入/泄露规则集即按其攻击语料形态设计 (含 leetspeak / base64 / 零宽
  字符混淆的归一化还原)。

可验证不变量:
- **INV-38 护栏拦截 / 误杀**: 红队拦截率 **≥ 90%**, 正常提示误杀 **≤ 5%**
  (数据集见 ``assets/data/redteam_100.jsonl`` 与 ``assets/data/benign_100.jsonl``,
  判据由 :func:`evaluate` 统计并由 ``tests/unit/l7/test_guardrail.py`` 断言)。
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- 归一化
_LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
             "8": "b", "9": "g", "@": "a", "$": "s", "!": "i", "+": "t"}
# 零宽 / 双向控制字符 (用码点构造, 避免源码里混入不可见字符)
_ZERO_WIDTH = "".join(chr(c) for c in (
    0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C,
    0x202D, 0x202E, 0x2060, 0xFEFF))
_WS_RE = re.compile(r"\s+")


def normalize(text: str, leet: bool = True) -> str:
    """归一化: 去零宽字符 → NFKC → 小写 → leetspeak 还原 → 折叠空白。

    用于抵抗 "1gn0re" / "i g n o r e" / 零宽插字符 一类混淆绕过。
    """
    t = str(text)
    for ch in _ZERO_WIDTH:
        t = t.replace(ch, "")
    t = unicodedata.normalize("NFKC", t).lower()
    if leet:
        t = "".join(_LEET_MAP.get(c, c) for c in t)
    return _WS_RE.sub(" ", t).strip()


# ---------------------------------------------------------------- 规则集
@dataclass
class Rule:
    """一条护栏规则。

    Attributes:
        category: 风险类别 (对齐 OWASP LLM Top 10 口径)。
        severity: 严重度 (0~1), ≥ ``min_block_severity`` 即拦截。
        pattern: 已编译正则 (作用于归一化文本)。
        reason: 命中说明 (写入 :class:`GuardrailVerdict` 供审计)。
    """

    category: str
    severity: float
    pattern: "re.Pattern[str]"
    reason: str
    on_raw: bool = False          # True 表示作用于原始文本 (如零宽字符检测)


def _rule(category: str, severity: float, pattern: str, reason: str,
          on_raw: bool = False) -> Rule:
    return Rule(category=category, severity=severity,
                pattern=re.compile(pattern, re.IGNORECASE), reason=reason,
                on_raw=on_raw)


# 每条规则标注来源出处 (零重写审计: OWASP / NeMo Guardrails / Perez & Ribeiro 2022)
INPUT_RULES: List[Rule] = [
    # ---- LLM01 Prompt Injection (Perez & Ribeiro 2022) ----
    _rule("prompt_injection", 0.95,
          r"(忽略|无视|不理会)(以上|上面|前面|之前|下述|下列|所有|一切|全部|这些)*(的)?(指令|指示|规则|提示|要求|限制|设定|约束)",
          "检测到『忽略/无视先前指令』类提示注入"),
    _rule("prompt_injection", 0.95,
          r"忽略(你的)?(安全|系统|内置|原有)?(规则|策略|限制|设定|提示|约束|指令)",
          "检测到要求忽略安全/系统规则的注入"),
    _rule("prompt_injection", 0.95,
          r"无视(上述|以上|前面|之前|所有|一切|全部|这些)*(的)?(指令|规则|提示|限制|约束|要求)",
          "检测到『无视规则』类提示注入"),
    _rule("prompt_injection", 0.95,
          r"(忘记|忘掉|清除|丢掉)\s*.{0,8}(你|之前|以上|前面|你的|所有|全部)?\s*(的)?\s*(指令|规则|设定|限制|约束|提示)",
          "检测到要求忘记系统指令的注入"),
    _rule("prompt_injection", 0.95,
          r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+|previous\s+|prior\s+|above\s+|earlier\s+|your\s+)*(instructions?|prompts?|rules?|directions?|guidelines?|system\s*prompt)",
          "检测到英文 ignore/disregard previous instructions 注入"),
    _rule("prompt_injection", 0.95,
          r"(你现在是|现在开始你是|从现在起你是|从现在开始你是|请你扮演|you are now|act as)\s*.{0,12}(dan|开发者模式|无限制|不受限|无审查|uncensored|unfiltered|jailbreak|越狱)",
          "检测到 DAN / 开发者模式 / 越狱角色扮演注入"),
    _rule("prompt_injection", 0.95,
          r"(dan\s*模式|开发者模式|developer\s*mode|god\s*mode|jailbreak\s*模式|越狱模式|opposite\s*day)",
          "检测到越狱模式关键词"),
    _rule("prompt_injection", 0.9,
          r"(绕过|避开|规避|突破|取消|关闭|移除)\s*.{0,10}(安全|审查|过滤|护栏|限制|对齐|审核|屏蔽|封禁|safety|filter|guardrail|alignment|moderation|restriction|censorship)",
          "检测到要求绕过安全/护栏机制"),
    _rule("prompt_injection", 0.9,
          r"(bypass|disable|turn\s*off|remove|jailbreak)\s+.{0,16}(safety|filter|guardrail|moderation|alignment|restriction|your\s+rules)",
          "检测到英文绕过安全机制请求"),
    _rule("prompt_injection", 0.85,
          r"(假装|pretend)\s*(你是|我们|to\s*be|you\s*are|we\s*are)?\s*.{0,16}(没有|不受|不受任何|no|without)\s*(任何)?(限制|规则|约束|restriction|rule|limit|filter)",
          "检测到『假装没有限制』的角色扮演逃逸"),
    _rule("prompt_injection", 0.9,
          r"(system|系统)\s*(override|覆盖|接管)|(覆盖|接管)\s*(你的)?(系统|system)\s*(提示|指令|prompt)",
          "检测到 system override 注入"),
    # ---- LLM06 敏感信息泄露 / 系统提示外泄 ----
    _rule("prompt_leak", 0.9,
          r"(输出|打印|显示|复述|重复|告诉我|泄露|暴露|展示|披露|公开|发我|发给我|show)\s*.{0,12}(你的|系统|system|初始)?\s*(系统提示|提示词|指令|规则|prompt|instructions)",
          "检测到索取系统提示词的请求"),
    _rule("prompt_leak", 0.9,
          r"(系统提示|提示词|初始指令|系统指令|你的指令|你的规则|system\s*prompt)\s*.{0,12}(输出|打印|显示|复述|重复|告诉我|展示|披露|公开|列出来|写出来|发我|发给我|print|reveal)",
          "检测到索取/回显系统提示的请求 (目标在前的语序)"),
    _rule("prompt_leak", 0.9,
          r"(reveal|show|print|repeat|output|leak|dump)\s*.{0,12}(your|the|system)?\s*(system\s*)?(prompt|instructions|rules|system\s*prompt)",
          "检测到英文 reveal your prompt 请求"),
    _rule("prompt_leak", 0.9,
          r"(repeat|print|output)\s+(everything|all|the\s+text|the\s+instructions)\s+(above|before|you\s+were\s+given)",
          "检测到『复述以上全部内容』的泄露探测"),
    _rule("prompt_leak", 0.85,
          r"(把|将)?(你|模型)?(收到|看到|被给|被输入)的(全部|所有)?(指令|文本|内容|提示)(输出|打印|复述|告诉我|展示)",
          "检测到要求回显输入指令"),
    # ---- 危险内容: 武器 / 爆炸物 ----
    _rule("weapons", 1.0,
          r"(制造|制作|自制|合成|组装|配置|调配|制备)\s*.{0,8}(炸弹|炸药|爆炸物|燃烧瓶|莫洛托夫|汽油弹|雷管|引爆装置|枪支|手枪|子弹|弹药|毒气|生化武器|化学武器|神经毒剂|沙林|蓖麻毒素|放射性物质)",
          "检测到武器/爆炸物/毒剂制作请求"),
    _rule("weapons", 1.0,
          r"(如何|怎么|怎样|教我|帮我|给我|我想|想要)(快速|在家|自制)?(制造|制作|做|搞|获得)?(一个|一枚|一把|一支)?(炸弹|炸药|枪|手枪|毒气|生化武器|燃烧瓶)",
          "检测到获取危险武器的方法咨询"),
    _rule("weapons", 1.0,
          r"how\s+to\s+(make|build|create|synthesize|manufacture)\s+.{0,24}(bomb|explosive|molotov|gun|weapon|poison|sarin|ricin|napalm|ied|chemical\s+weapon)",
          "检测到英文危险武器制作咨询"),
    # ---- 网络滥用 / 恶意代码 ----
    _rule("cyber_abuse", 1.0,
          r"(写|编写|生成|创建|制作|做|给我|提供|帮我写|来一段|来一个|来份|share|write)\s*.{0,8}(病毒|木马|蠕虫|勒索软件|恶意软件|恶意代码|后门|远控|键盘记录器|钓鱼页面|钓鱼邮件|钓鱼网站|钓鱼链接|黑产工具|抓鸡工具)",
          "检测到恶意代码/钓鱼素材生成请求"),
    _rule("cyber_abuse", 1.0,
          r"(撞库|拖库|爆破密码|撞库攻击|提权|免杀|抓鸡|肉鸡|社工库|远控木马)",
          "检测到攻击工具/手法关键词"),
    _rule("cyber_abuse", 1.0,
          r"(ransomware|malware|trojan|keylogger|botnet|rootkit|shellcode|exploit\s+code|sqlmap|metasploit)",
          "检测到恶意程序关键词"),
    _rule("cyber_abuse", 1.0,
          r"(入侵|攻破|渗透|黑掉|拿下|爆破|撞库|拖库|提权)\s*.{0,8}(网站|服务器|数据库|系统|账号|后台|内网|校园网|wifi|路由器|手机|邮箱)",
          "检测到未授权入侵/撞库请求"),
    _rule("cyber_abuse", 1.0,
          r"(sql\s*注入|xss\s*攻击|csrf|0\s*day|0day|免杀|绕过杀软|绕过防火墙|ddos|拒绝服务攻击|dos\s*攻击)",
          "检测到攻击技术关键词"),
    _rule("cyber_abuse", 0.95,
          r"(窃取|盗取|偷取|批量获取|爬取)\s*.{0,10}(用户|他人|别人|全网)?(的)?(密码|口令|token|cookie|凭证|私钥|数据库|通讯录|身份证)",
          "检测到凭证/隐私数据窃取请求"),
    # ---- 违法欺诈 ----
    _rule("fraud", 0.95,
          r"(洗钱|黑产|跑分|刷单|套现|假币|逃税|做假账|赌博作弊|出千|诱导转账|骗钱|诈骗话术|杀猪盘)",
          "检测到金融违法/诈骗相关请求"),
    _rule("fraud", 0.95,
          r"(伪造|仿制|制作|办理)\s*.{0,6}(假)?(证件|身份证|发票|货币|人民币|美元|公章|护照|学历|驾照|银行卡|营业执照)",
          "检测到伪造证件/货币请求"),
    _rule("fraud", 0.9,
          r"(冒充)\s*.{0,6}(客服|公检法|银行|警察|领导|熟人|官方)",
          "检测到冒充身份的诈骗话术请求"),
    _rule("fraud", 0.95,
          r"(money\s*laundering|counterfeit|fake\s*id|carding|phishing\s*kit|how\s+to\s+launder)",
          "检测到英文违法金融请求"),
    # ---- 自伤 ----
    _rule("self_harm", 1.0,
          r"(自杀|自残|自我了断|结束生命|结束自己的生命|不想活了|割腕|轻生|烧炭|安乐死方法)",
          "检测到自伤/自杀相关内容"),
    _rule("self_harm", 1.0,
          r"(如何|怎么|怎样)(才能)?(快速|无痛|最有效)?(地)?(自杀|死亡|结束生命|自残)",
          "检测到自伤方法咨询"),
    _rule("self_harm", 1.0,
          r"(kill\s+myself|end\s+my\s+life|self\s*-?\s*harm|suicide|how\s+to\s+(kill|hurt)\s+myself)",
          "检测到英文自伤内容"),
    # ---- 隐私泄露 ----
    _rule("privacy", 0.9,
          r"(告诉我|给我|提供|列出|导出|发我|泄露|盗取|dump|export)\s*.{0,16}(用户|客户|同事|他人|别人|所有|全部)?(的)?(身份证|身份证号|手机号|电话号码|家庭住址|住址|银行卡|信用卡|密码|邮箱|病历|体检报告|聊天记录|隐私数据)",
          "检测到索取/导出他人隐私信息的请求"),
    _rule("privacy", 0.9,
          r"(dump|export|list|leak|give\s+me)\s+(all\s+)?(the\s+)?(user\s+)?(emails?|passwords?|records?|phone\s+numbers?|credit\s+cards?|ssn)",
          "检测到英文隐私数据导出请求"),
    _rule("privacy", 0.85,
          r"(明文密码|明文口令|数据库密码|连接串|connection\s*string|数据库账号密码)",
          "检测到索取明文凭证/连接串"),
    # ---- 仇恨 / 歧视 ----
    _rule("hate_speech", 0.95,
          r"(写|生成|发表|来一段|帮我写|撰写)\s*.{0,12}(仇恨言论|种族歧视|种族仇恨|民族仇恨|宗教仇恨|歧视言论|排外言论|侮辱性言论|煽动仇恨)",
          "检测到仇恨/歧视内容生成请求"),
    _rule("hate_speech", 0.95,
          r"(种族清洗|genocide|大屠杀|灭绝)(某)?(民族|种族|族群|犹太|黑人|亚裔|穆斯林)?",
          "检测到极端仇恨/暴力煽动"),
    _rule("hate_speech", 0.9,
          r"(侮辱|辱骂|诋毁|贬低|攻击)\s*.{0,6}(某)?(民族|种族|宗教|群体|女性|同性恋|残疾人|难民|外来人口)",
          "检测到针对群体的侮辱内容"),
    # ---- 色情 / 未成年人 ----
    _rule("sexual", 0.95,
          r"(生成|写|提供|来一段|给我|创作)\s*.{0,10}(色情|情色|淫秽|露骨|性爱|性行为|性描写)(内容|小说|故事|描写|细节|剧本|段落)?",
          "检测到色情内容生成请求"),
    _rule("sexual", 1.0,
          r"(未成年|未成年人|中学生|小学生|儿童|幼女|萝莉)\s*.{0,10}(色情|裸|性|淫秽|情色|性爱)",
          "检测到涉未成年人不良内容"),
    _rule("sexual", 1.0,
          r"(child\s*porn|csam|pornograph|sexual\s+content\s+with\s+minors|grooming\s+minors)",
          "检测到英文未成年不良内容"),
    # ---- 混淆绕过 ----
    _rule("obfuscation", 0.85,
          r"(base64|rot13|hex|十六进制|摩斯|morse|凯撒|caesar|unicode\s*转义|零宽|zwsp|同形字)\s*.{0,12}(编码|解码|加密|解密|encode|decode|隐藏|绕过|执行)",
          "检测到编码混淆绕过手法"),
    _rule("obfuscation", 0.9,
          r"(解码|解密|翻译)后?(执行|运行|遵守|照做|服从)",
          "检测到『解码后执行』的间接注入"),
    _rule("obfuscation", 0.8, r"[a-z0-9+/]{80,}={0,2}", "检测到疑似 base64 密文负载"),
    _rule("obfuscation", 0.7, r"[" + _ZERO_WIDTH + r"]", "检测到零宽字符隐藏指令", on_raw=True),
]

# 输出侧: 泄露/PII 过滤 (OWASP LLM02)
OUTPUT_RULES: List[Rule] = [
    _rule("prompt_leak", 0.9,
          r"(system\s*prompt|系统提示词?\s*[:：]|internal\s+instructions\s*[:：])",
          "输出疑似包含系统提示词"),
    _rule("privacy", 0.9,
          r"\b\d{17}[\dxX]\b|\b1[3-9]\d{9}\b", "输出疑似包含身份证号/手机号"),
    _rule("privacy", 0.8,
          r"[\w.+-]+@[\w-]+\.[\w.]{2,}", "输出疑似包含邮箱地址"),
]

_PII_MASK = [
    (re.compile(r"\b\d{17}[\dxX]\b"), "[已脱敏-身份证]"),
    (re.compile(r"\b1[3-9]\d{9}\b"), "[已脱敏-手机号]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"), "[已脱敏-邮箱]"),
]


@dataclass
class GuardrailVerdict:
    """护栏判定结果 (结构化, 便于审计与分级处置)。"""

    allowed: bool
    risk: float = 0.0
    categories: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    sanitized_text: str = ""
    stage: str = "input"

    @property
    def blocked(self) -> bool:
        """是否拦截 (``allowed`` 的反义, 便于断言)。"""
        return not self.allowed

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "risk": self.risk,
                "categories": list(self.categories), "reasons": list(self.reasons),
                "sanitized_text": self.sanitized_text, "stage": self.stage}


class Guardrail:
    """规则护栏: 输入注入/有害检测 + 输出泄露/PII 过滤 (INV-38)。"""

    def __init__(self, max_prompt_chars: int = 8000,
                 min_block_severity: float = 0.5,
                 enable_leet: bool = True,
                 extra_rules: Sequence[Rule] = ()):
        self.max_prompt_chars = int(max_prompt_chars)
        self.min_block_severity = float(min_block_severity)
        self.enable_leet = bool(enable_leet)
        self.input_rules: List[Rule] = list(INPUT_RULES) + list(extra_rules)
        self.output_rules: List[Rule] = list(OUTPUT_RULES)

    # ---------------- 输入侧 ----------------
    def check_input(self, text: str) -> GuardrailVerdict:
        """对用户输入做护栏检查 (INV-38 的红队/误杀统计口径)。"""
        raw = str(text)
        # 两路归一化: leet 还原 (抓 1gn0re) + 原样归一化 (抓 0day / base64 / rot13)
        norms = [normalize(raw, leet=True), normalize(raw, leet=False)] \
            if self.enable_leet else [normalize(raw, leet=False)]
        cats: List[str] = []
        reasons: List[str] = []
        worst = 0.0
        for r in self.input_rules:
            hits = [raw] if r.on_raw else norms
            if any(r.pattern.search(h) for h in hits):
                worst = max(worst, r.severity)
                if r.category not in cats:
                    cats.append(r.category)
                reasons.append(r.reason)
        # 长度 / 洪水: 成本与资源限额 (NeMo Guardrails 的 limits 思想)
        if len(raw) > self.max_prompt_chars:
            worst = max(worst, 0.9)
            if "resource_abuse" not in cats:
                cats.append("resource_abuse")
            reasons.append(f"输入长度 {len(raw)} 超过上限 {self.max_prompt_chars}")
        else:
            flood = _repetition_flood(raw)
            if flood:
                worst = max(worst, 0.8)
                if "resource_abuse" not in cats:
                    cats.append("resource_abuse")
                reasons.append("检测到重复字符洪水 (资源滥用)")
        risk = min(1.0, worst + 0.1 * max(0, len(cats) - 1))
        return GuardrailVerdict(allowed=worst < self.min_block_severity, risk=risk,
                                categories=cats, reasons=reasons,
                                sanitized_text=raw, stage="input")

    # ---------------- 输出侧 ----------------
    def check_output(self, text: str) -> GuardrailVerdict:
        """对模型输出做泄露/PII 过滤, 并给出脱敏后的文本。"""
        raw = str(text)
        norms = [normalize(raw, leet=True), normalize(raw, leet=False)] \
            if self.enable_leet else [normalize(raw, leet=False)]
        cats: List[str] = []
        reasons: List[str] = []
        worst = 0.0
        for r in self.output_rules:
            hits = [raw] if r.on_raw else norms
            if any(r.pattern.search(h) for h in hits):
                worst = max(worst, r.severity)
                if r.category not in cats:
                    cats.append(r.category)
                reasons.append(r.reason)
        masked = raw
        for pat, repl in _PII_MASK:
            masked = pat.sub(repl, masked)
        risk = min(1.0, worst + 0.1 * max(0, len(cats) - 1))
        return GuardrailVerdict(allowed=worst < self.min_block_severity, risk=risk,
                                categories=cats, reasons=reasons,
                                sanitized_text=masked, stage="output")

    def check(self, text: str) -> GuardrailVerdict:
        """:meth:`check_input` 的别名 (统一入口)。"""
        return self.check_input(text)

    def guard(self, fn: Callable[[str], str]) -> Callable[[str], Tuple[bool, str]]:
        """装饰器: 包裹生成函数, 返回 ``(allowed, text_or_reason)``。"""

        def wrapped(prompt: str) -> Tuple[bool, str]:
            v = self.check_input(prompt)
            if not v.allowed:
                return False, "已拦截: " + "; ".join(v.reasons)
            out = fn(prompt)
            vo = self.check_output(out)
            return True, vo.sanitized_text
        return wrapped


def _repetition_flood(text: str, threshold: int = 400) -> bool:
    """检测重复字符洪水 (同一字符连续出现 ``threshold`` 次以上)。"""
    if not text:
        return False
    m = re.search(r"(.)\1{" + str(threshold - 1) + r",}", text)
    return m is not None


# ---------------------------------------------------------------- 评估
def load_cases(path: str) -> List[dict]:
    """加载 jsonl 用例集 (每行一个 JSON 对象, 含 ``text`` 字段)。"""
    cases: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def evaluate(guardrail: Guardrail, redteam: Sequence[dict] | str,
             benign: Sequence[dict] | str) -> dict:
    """INV-38 统计: 红队拦截率与正常提示误杀率。

    Args:
        guardrail: 待评估护栏。
        redteam: 红队用例 (dict 列表) 或 jsonl 路径 (期望全部被拦截)。
        benign: 正常用例 (dict 列表) 或 jsonl 路径 (期望全部放行)。

    Returns:
        dict: ``block_rate`` / ``false_positive_rate`` / ``n_redteam`` /
        ``n_benign`` / ``missed`` (漏拦的红队样例) / ``false_positives``。
    """
    red = load_cases(redteam) if isinstance(redteam, str) else list(redteam)
    ben = load_cases(benign) if isinstance(benign, str) else list(benign)
    missed: List[dict] = []
    for c in red:
        v = guardrail.check_input(c["text"])
        if v.allowed:
            missed.append({"id": c.get("id"), "category": c.get("category"),
                           "text": c["text"][:40]})
    fps: List[dict] = []
    for c in ben:
        v = guardrail.check_input(c["text"])
        if not v.allowed:
            fps.append({"id": c.get("id"), "text": c["text"][:40],
                        "reasons": v.reasons})
    return {
        "block_rate": (len(red) - len(missed)) / len(red) if red else 0.0,
        "false_positive_rate": len(fps) / len(ben) if ben else 0.0,
        "n_redteam": len(red),
        "n_benign": len(ben),
        "missed": missed,
        "false_positives": fps,
        "verdict": "PASS" if (not red or (len(red) - len(missed)) / len(red) >= 0.90)
                   and (not ben or len(fps) / len(ben) <= 0.05) else "FAIL",
    }


def default_guardrail() -> Guardrail:
    """返回默认配置护栏 (供 L7 服务与 L9 CLI 直接复用)。"""
    return Guardrail()


def redteam_path() -> str:
    """内置红队数据集路径 (100 条)。"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "assets", "data", "redteam_100.jsonl")


def benign_path() -> str:
    """内置正常提示数据集路径 (100 条)。"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "assets", "data", "benign_100.jsonl")
