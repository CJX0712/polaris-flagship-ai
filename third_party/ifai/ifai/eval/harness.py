"""L8 评测 Harness — GSM8K-mini 数学推理 + 回归基线门禁

零重写: GSM8K (Cobbe et al. 2021) 评测范式; 回归门禁参考 HELM (Liang 2022)
的多维评测 + CI 基线保护思想。

说明: 微型随机初始化模型准确率天然极低, 此处仅验证"管线正确性"
(能否产出可解析的数字答案), 不设置准确率硬门槛 (PRD Q4)。
"""
from __future__ import annotations

import json
import os
import re


class GSM8K_Mini:
    """内置 20 道极简算术题 (离线, 不随仓下载)。"""

    DATA = [
        ("小明有 3 个苹果, 又买了 5 个, 共有几个?", 8),
        ("7 减去 2 等于多少?", 5),
        ("一个盒子有 4 排, 每排 6 个, 共几个?", 24),
        ("10 除以 2 是多少?", 5),
        ("12 加 13 等于多少?", 25),
        ("8 乘以 3 是多少?", 24),
        ("15 减去 9 等于几?", 6),
        ("20 的一半是多少?", 10),
        ("9 加 9 加 9 等于多少?", 27),
        ("100 减 37 等于多少?", 63),
        ("6 的 7 倍是多少?", 42),
        ("48 除以 6 等于几?", 8),
        ("3 的平方加 4 等于多少?", 13),
        ("5 乘以 5 再减 1 等于多少?", 24),
        ("2 的 10 次方是多少?", 1024),
        ("一个数加 11 等于 30, 这个数是几?", 19),
        ("买 3 件每件 9 元的商品共多少钱?", 27),
        ("一周 7 天, 3 周几天?", 21),
        ("45 除以 9 再乘 2 等于几?", 10),
        ("1 到 10 相加等于多少?", 55),
    ]

    @staticmethod
    def extract_number(text: str) -> int | None:
        nums = re.findall(r"-?\d+", text)
        return int(nums[-1]) if nums else None

    def evaluate(self, answer_fn) -> dict:
        correct = 0
        samples = []
        for q, gold in self.DATA:
            ans = answer_fn(q)
            pred = self.extract_number(str(ans))
            ok = pred == gold
            correct += int(ok)
            samples.append({"q": q, "gold": gold, "pred": pred, "ok": ok})
        return {"correct": correct, "total": len(self.DATA), "samples": samples}


class RegressionBaseline:
    """回归门禁: 当前吞吐相对基线掉落 >15% ⇒ 红灯。

    零重写 · 双轨合一 (v0.8.0)
    --------------------------
    此前本类去**解析 BASELINE.md 的人类可读行** (``tok/s mean: …``), 而
    ``scripts/gen_baseline.py`` 的机器比较用的是 **BASELINE.json** ——
    同一份基线两套读法, 改一份漏一份: 基线文件被重新生成时, MD 的行
    格式一旦变动, 本类静默返回 ``NO_BASELINE`` (门禁形同虚设, 且不报错)。

    现改为**单一权威源 = BASELINE.json**, 与生成脚本的 ``--check``
    完全同源同口径。MD 降级为纯人类可读产物, 不再被任何代码解析。

    保留对 MD 的**兼容读取** (仅当 JSON 缺失时), 以便旧基线文件不致
    立即失效 —— 但 JSON 优先, 且兼容路径会明确记录它来自 MD。
    """

    @staticmethod
    def parse_baseline(path: str) -> dict:
        """读取基线。

        优先读同名 ``.json`` (权威源); 若不存在则回退解析 ``.md``。

        Args:
            path: 基线文件路径 (``.json`` 或 ``.md``)。

        Returns:
            归一化后的基线指标 dict; 读不到时返回空 dict。
        """
        out: dict = {}
        json_path = path if path.endswith(".json") else os.path.splitext(path)[0] + ".json"
        if os.path.isfile(json_path):
            try:
                with open(json_path, encoding="utf-8") as f:
                    raw = json.load(f)
            except (OSError, json.JSONDecodeError):
                raw = None
            if isinstance(raw, dict):
                tps = raw.get("throughput_tok_s")
                if tps is not None:
                    out["tok_per_s_mean"] = float(tps)
                if raw.get("tpot_ms") is not None:
                    out["tpot_ms"] = float(raw["tpot_ms"])
                if raw.get("prefill_ms") is not None:
                    out["prefill_ms"] = float(raw["prefill_ms"])
                if raw:
                    out["source"] = json_path
                return out
        # 兼容路径: 旧基线只有 MD
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if "tok/s mean" in line:
                        out["tok_per_s_mean"] = float(line.split(":")[1])
                        out["source"] = path
                        break
        except OSError:
            pass
        return out

    @staticmethod
    def check(current: dict, baseline_path: str, threshold: float = 0.15) -> dict:
        """比对当前指标与基线。

        Args:
            current: 当前指标 (需含 ``tok_per_s_mean``)。
            baseline_path: 基线文件路径。
            threshold: 允许的相对退化幅度。

        Returns:
            ``{"status": "GREEN"/"RED"/"NO_BASELINE", "drop": float,
            "threshold": float, "source": str}``
        """
        base = RegressionBaseline.parse_baseline(baseline_path)
        if "tok_per_s_mean" not in base:
            return {"status": "NO_BASELINE", "drop": 0.0,
                    "threshold": threshold, "source": None}
        drop = (base["tok_per_s_mean"] - current["tok_per_s_mean"]) / base["tok_per_s_mean"]
        status = "RED" if drop > threshold else "GREEN"
        return {"status": status, "drop": float(drop), "threshold": threshold,
                "source": base.get("source")}
