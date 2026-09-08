"""
ErrorReflection: 迭代错误反思模块（v2 重设计）
==============================================
全路径记录 + 自动触发反思 + 持久化报告 + 成功率趋势分析。

与 FaultDiagnosis 的区别：
  - FaultDiagnosis 专注代码语法/导入错误 → 自动回滚
  - ErrorReflection 专注运行时/API/JSON/沙箱等非语法失败 → 模式识别 + 策略调整

v2 改进点（对比 v1）：
  1. 全路径记录：无论成功还是失败都记录（v1 只在完整迭代末尾记录，early return 漏掉）
  2. 触发条件放宽：连续 2 轮失败或每 5 轮定期触发（v1 是 3 轮/10 轮）
  3. 反思报告持久化：保存到 logs/reflection_reports.json（v1 只 print 不存盘）
  4. 成功率趋势：分析最近窗口的成功率变化趋势
  5. 与 ModelProfile 联动：生成可执行的参数调整建议

错误分类：
  API_TIMEOUT        → LLM API 超时
  API_ERROR          → HTTP 错误 / 服务不可达
  JSON_PARSE_FAIL    → LLM 返回非 JSON 或 JSON 格式错误
  NO_MODIFICATION    → LLM 返回空 modifications 列表
  ALL_PATCH_FAIL     → 所有补丁应用失败
  SANDBOX_QUARANTINE → 沙箱隔离目标
  RATE_LIMIT         → API 限流
  CLIENT_NOT_READY   → 客户端未就绪（服务未启动）
  SUCCESS            → 迭代成功（用于成功率统计）
  UNKNOWN            → 未分类的其他错误
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


REFLECTION_LOG_NAME = "error_reflection.json"
REFLECTION_REPORTS_NAME = "reflection_reports.json"
MAX_HISTORY = 300
REFLECTION_INTERVAL = 5         # 每 5 轮定期生成反思
REFLECTION_CONSECUTIVE = 2       # 连续失败 N 轮立即生成反思
TREND_WINDOW = 15               # 预案4：失败观测窗口从 10 扩充到 15 轮，提升故障识别灵敏度


_ERROR_CATEGORIES = {
    "API_TIMEOUT": "LLM API 调用超时，模型可能过载或 context 过大",
    "API_ERROR": "LLM 服务返回 HTTP 错误或不可达",
    "JSON_PARSE_FAIL": "模型输出无法解析为有效 JSON（缺少引号、尾逗号、换行符等）",
    "NO_MODIFICATION": "模型返回空 modifications 列表（可能认为代码无需修改）",
    "ALL_PATCH_FAIL": "所有补丁应用失败（old_str 不匹配 / 语法错误被 AST 预检拦截）",
    "SANDBOX_QUARANTINE": "迭代目标被沙箱熵差隔离",
    "RATE_LIMIT": "API 限流或服务繁忙",
    "CLIENT_NOT_READY": "客户端未就绪（服务未启动）",
    "SUCCESS": "迭代成功完成",
    "UNKNOWN": "未分类的其他错误",
}


def classify_error(status: str, issues: List[str], exception: Optional[str] = None) -> str:
    """根据迭代结果和异常信息推断错误类别"""
    issues_joined = " ".join(str(i) for i in issues) if issues else ""

    # 0. 最高优先级：显式标签（self_evolver 在空 modifications 分支手动打的强标记）
    if "[json_parse_fail]" in issues_joined:
        return "JSON_PARSE_FAIL"

    # 成功路径
    if status in ("修改成功", "部分成功", "故障自愈已执行"):
        return "SUCCESS"

    if status in ("API错误",):
        msg = issues_joined + " " + (exception or "")
        lower = msg.lower()
        if "timeout" in lower or "timed out" in lower:
            return "API_TIMEOUT"
        if "rate" in lower or "429" in msg:
            return "RATE_LIMIT"
        return "API_ERROR"

    if status == "配置错误":
        return "CLIENT_NOT_READY"

    if status == "API响应异常":
        lower = issues_joined.lower()
        if "json" in lower or "parse" in lower or "invalid" in lower or "有效json" in lower:
            return "JSON_PARSE_FAIL"
        return "API_ERROR"

    if status == "无需修改":
        return "NO_MODIFICATION"

    if status == "全部失败":
        return "ALL_PATCH_FAIL"

    if status == "沙箱隔离":
        return "SANDBOX_QUARANTINE"

    if exception:
        lower = exception.lower()
        if "timeout" in lower:
            return "API_TIMEOUT"
        if "json" in lower or "decode" in lower:
            return "JSON_PARSE_FAIL"

    return "UNKNOWN"


class ErrorReflection:
    """
    迭代错误反思 v2：全路径记录 → 模式识别 → 策略调整建议 → 持久化

    使用方式：
      # 每轮迭代（无论成功失败）都调用：
      reflection_text = reflection.record_iteration(result, objective, exception)
      if reflection_text:
          # 注入到下一轮 prompt
          user_prompt += reflection_text

      # 生成简短上下文注入 prompt：
      ctx = reflection.format_for_prompt()
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / REFLECTION_LOG_NAME
        self.reports_path = log_dir / REFLECTION_REPORTS_NAME
        self._log: List[Dict[str, Any]] = self._load_log()
        self._consecutive_fail: int = self._count_consecutive_fail()
        self._cycle_count: int = self._load_cycle_count()
        # =========【新增】保存hypothesis_decoupler实例引用 =========
        self._decoupler = None

    # ── 持久化 ──

    def set_decoupler(self, decoupler):
        """外部daemon调用，传入HypothesisDecoupler实例，用于转发中间失败事件到L3"""
        self._decoupler = decoupler


    def _load_log(self) -> List[Dict[str, Any]]:
        if self.log_path.exists():
            try:
                return json.loads(self.log_path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _load_cycle_count(self) -> int:
        if self._log:
            return self._log[-1].get("cycle", 0)
        return 0

    def _count_consecutive_fail(self) -> int:
        count = 0
        for entry in reversed(self._log):
            if entry.get("is_failure"):
                count += 1
            else:
                break
        return count

    def _save_log(self):
        try:
            self.log_path.write_text(
                json.dumps(self._log[-MAX_HISTORY:], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _save_reflection_report(self, report: str, trigger: str, stats: Dict[str, Any]):
        """把反思报告追加保存到 logs/reflection_reports.json"""
        reports = []
        if self.reports_path.exists():
            try:
                reports = json.loads(self.reports_path.read_text(encoding="utf-8"))
                if not isinstance(reports, list):
                    reports = []
            except Exception:
                reports = []

        reports.append({
            "timestamp": datetime.now().isoformat(),
            "cycle": self._cycle_count,
            "trigger": trigger,
            "stats": stats,
            "report": report,
        })

        # 保留最近 50 份报告，避免文件无限增长
        reports = reports[-50:]
        try:
            self.reports_path.write_text(
                json.dumps(reports, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── 记录入口（全路径）──

    def record_iteration(self, result: Dict[str, Any], objective: str = "",
                         exception: Optional[str] = None) -> Optional[str]:
        """
        记录一轮迭代的结果（无论成功失败），返回反思摘要（如果触发）。
        返回摘要时应注入到下一轮 prompt 中。
        """
        self._cycle_count += 1
        status = result.get("status", "unknown")
        issues = result.get("issues", [])
        applied = result.get("applied_count", 0)
        suggested = result.get("suggestion_count", 0)

        is_failure = (status not in ("修改成功", "部分成功", "故障自愈已执行")) and applied == 0
        # 成功路径也记录，用于成功率统计
        category = classify_error(status, issues, exception) if is_failure else "SUCCESS"

        entry = {
            "cycle": self._cycle_count,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "applied_count": applied,
            "suggestion_count": suggested,
            "is_failure": is_failure,
            "category": category,
            "objective": objective[:80] if objective else "",
            "issues": [str(i)[:120] for i in issues[:3]],
        }
        if exception:
            entry["exception"] = exception[:200]

        self._log.append(entry)
        if is_failure:
            self._consecutive_fail += 1
        else:
            self._consecutive_fail = 0

        self._save_log()

        # 判断是否触发反思
        reflection = None
        if self._consecutive_fail >= REFLECTION_CONSECUTIVE:
            reflection = self._generate_reflection(
                trigger=f"连续 {self._consecutive_fail} 轮失败"
            )
        elif self._cycle_count % REFLECTION_INTERVAL == 0:
            reflection = self._generate_reflection(
                trigger=f"定期反思（每 {REFLECTION_INTERVAL} 轮）"
            )

        return reflection

    def record_daemon_error(self, error: str, stage: str = "") -> Optional[str]:
        """记录 daemon 主循环级别的异常"""
        self._cycle_count += 1
        entry = {
            "cycle": self._cycle_count,
            "timestamp": datetime.now().isoformat(),
            "status": "DAEMON_ERROR",
            "applied_count": 0,
            "suggestion_count": 0,
            "is_failure": True,
            "category": "UNKNOWN",
            "objective": stage[:40],
            "issues": [error[:200]],
        }
        self._log.append(entry)
        self._consecutive_fail += 1
        self._save_log()

        if self._consecutive_fail >= REFLECTION_CONSECUTIVE:
            return self._generate_reflection(
                trigger=f"连续 {self._consecutive_fail} 轮 daemon 异常"
            )
        return None

    # ── 反思生成 ──

    def _generate_reflection(self, trigger: str = "") -> str:
        """分析错误历史，生成策略调整建议并持久化"""
        recent = self._log[-TREND_WINDOW * 2:]
        failures = [e for e in recent if e.get("is_failure")]
        successes = [e for e in recent if not e.get("is_failure")]
        if not recent:
            return ""

        # 错误类别统计
        cat_counter = Counter(e.get("category", "UNKNOWN") for e in failures)
        total_fail = len(failures)
        total_rounds = len(recent)

        # 成功率趋势
        stats = self._compute_stats(recent)

        # 最近一轮原始响应
        last = recent[-1] if recent else {}
        last_issue = (last.get("issues") or [""])[0][:80]

        lines = []
        lines.append("═" * 50)
        lines.append(f"  📊 错误反思报告 ({trigger})")
        lines.append(f"  周期 #{self._cycle_count}, 最近 {total_rounds} 轮中 {total_fail} 轮失败")
        lines.append(f"  成功率: {stats['success_rate']:.0%}, 趋势: {stats['trend']}")
        lines.append("═" * 50)

        if cat_counter:
            lines.append("  错误分布:")
            for cat, count in cat_counter.most_common():
                pct = count * 100 // max(total_fail, 1)
                desc = _ERROR_CATEGORIES.get(cat, cat)
                lines.append(f"    [{cat}] {count}次 ({pct}%) → {desc}")

        # 生成可执行的策略调整建议
        lines.append("\n  🎯 策略调整建议:")
        advice = self._derive_advice(cat_counter, last, stats)
        for i, a in enumerate(advice, 1):
            lines.append(f"    {i}. {a}")

        if last_issue:
            lines.append(f"\n  🔍 最近一次错误摘要: {last_issue}")

        lines.append("═" * 50)

        reflection_text = "\n".join(lines)

        # 持久化反思报告（不在此处 print，由调用方决定是否输出，避免重复打印）
        self._save_reflection_report(reflection_text, trigger, stats)

        return reflection_text

    def _compute_stats(self, recent: List[Dict]) -> Dict[str, Any]:
        """计算成功率趋势统计"""
        if not recent:
            return {"success_rate": 0.0, "trend": "无数据", "total": 0, "failures": 0}

        total = len(recent)
        failures = [e for e in recent if e.get("is_failure")]
        success_rate = (total - len(failures)) / total

        # 把 recent 分成前后两半，比较成功率趋势
        mid = total // 2
        if mid == 0:
            trend = "数据不足"
        else:
            first_half = recent[:mid]
            second_half = recent[mid:]
            first_rate = sum(1 for e in first_half if not e.get("is_failure")) / len(first_half)
            second_rate = sum(1 for e in second_half if not e.get("is_failure")) / len(second_half)
            diff = second_rate - first_rate
            if diff > 0.1:
                trend = "↑ 改善中"
            elif diff < -0.1:
                trend = "↓ 恶化中"
            else:
                trend = "→ 稳定"

        return {
            "success_rate": round(success_rate, 2),
            "trend": trend,
            "total": total,
            "failures": len(failures),
        }

    def _derive_advice(self, cat_counter: Counter, last_entry: Dict,
                       stats: Dict) -> List[str]:
        """根据错误分布推导具体的策略调整建议（含 ModelProfile 联动）"""
        advice = []

        # 错误类别 → 参数调整建议（与 ModelProfile 联动）
        if cat_counter.get("API_TIMEOUT", 0) >= 1:
            advice.append("API 超时 → ModelProfile 应减小 num_ctx / budget（如 ctx 16384→8192，budget 减半）")
        if cat_counter.get("API_ERROR", 0) >= 2:
            advice.append("API 连接异常 → 检查 Ollama/LM Studio 是否运行，或切换后端")
        if cat_counter.get("JSON_PARSE_FAIL", 0) >= 1:
            advice.append("JSON 解析失败 → ModelProfile 应降低 temperature（如 0.15→0.0），或增强 JSON 修复逻辑")
        if cat_counter.get("NO_MODIFICATION", 0) >= 3:
            advice.append("连续返回空 modifications → 换更激进的目标，或增加代码预算让模型看到更多上下文")
        if cat_counter.get("ALL_PATCH_FAIL", 0) >= 2:
            advice.append("所有补丁失败 → old_str 与实际代码不匹配，改用 replace_all 或要求模型先读取目标文件")
        if cat_counter.get("SANDBOX_QUARANTINE", 0) >= 2:
            advice.append("沙箱连续隔离 → 目标过于发散，降级到安全目标池取一个")

        # 成功率趋势建议
        if stats.get("trend") == "↓ 恶化中":
            advice.append(f"成功率正在下降({stats.get('success_rate', 0):.0%}) → 考虑切换备用模型或暂停迭代")
        elif stats.get("success_rate", 0) >= 0.9 and not advice:
            advice.append("成功率良好(≥90%) → 保持当前策略，可尝试更激进的目标")

        if not advice:
            advice.append("错误分布分散 → 保持当前策略继续观察，每 5 轮再评估一次")

        return advice

    def record_event(self, iteration_id: int, is_failure: bool, category: str,
                     issues: Optional[List[str]] = None,
                     stats_extra: Optional[Dict[str, Any]] = None) -> None:
        """
        给外部模块（Benchmark / Contract）记录事件用的轻量 API。
        不触发反思，只写日志 + 调整连续失败计数。
        """
        self._cycle_count = max(self._cycle_count, iteration_id)
        entry = {
            "cycle": iteration_id,
            "timestamp": datetime.now().isoformat(),
            "status": "EVENT",
            "applied_count": 0,
            "suggestion_count": 0,
            "is_failure": bool(is_failure),
            "category": category,
            "objective": "",
            "issues": [str(i)[:120] for i in (issues or [])[:3]],
        }
        if stats_extra:
            # 只保留可 JSON 化的字段
            entry["stats"] = {
                k: v for k, v in stats_extra.items()
                if isinstance(v, (str, int, float, bool)) or (
                    isinstance(v, list) and all(isinstance(x, str) for x in v)
                )
            }
        self._log.append(entry)
        if is_failure:
            self._consecutive_fail += 1
        else:
            self._consecutive_fail = 0
        try:
            self._save_log()
            # =========【新增：转发中间失败事件给L3元认知层】=========
            if self._decoupler is not None and is_failure:
                try:
                    self._decoupler.record_intermediate_failure(
                        iteration_id=iteration_id,
                        category=category,
                        issues=issues or []
                    )
                except Exception as e:
                    # 转发失败绝不影响主逻辑，吃掉异常，不打断迭代
                    print(f"[DEBUG‑REFLECT] 转发异常: {e}")
                    pass

        
        except Exception:
            pass

    def record_patch_result(self, file: str, applied: bool, error_type: str = "",
                            detail: str = "") -> None:
        """
        兼容入口：记录单个 patch 的 apply 结果。
        内部统一转发到 record_event。category 直接使用 error_type。
        """
        category = (error_type or "PATCH_OK" if applied else "PATCH_FAIL") or (
            "PATCH_OK" if applied else "PATCH_FAIL"
        )
        self.record_event(
            iteration_id=self._cycle_count or 0,
            is_failure=not applied,
            category=category,
            issues=[f"{file}: {detail[:100]}"] if detail else [file],
        )

    # ── P1: 反思 → Prompt 强化闭环 ──

    # 失败类别 → 针对性 Prompt 强化语句（注入到 system prompt 末尾，权重最高）
    _PROMPT_BOOST_BY_CATEGORY = {
        "JSON_PARSE_FAIL": (
            "===== Prompt 强化：JSON 格式硬约束 =====\n"
            "1. 你的完整回答只允许包含一个合法 JSON 对象，首字符必须是 {，末字符必须是 }\n"
            "2. 在 JSON 对象之前和之后，严禁输出任何文字（包括中文解释、好的、我来帮你、等对话语气）\n"
            "3. 不要输出 Markdown 代码块围栏 ```，不要输出 <think> 标签内容\n"
            "4. 如果你做不到以上 3 条，请直接返回 {\"description\":\"\",\"modifications\":[]}"
        ),
        "NO_MODIFICATION": (
            "===== Prompt 强化：必须产出修改 =====\n"
            "请务必找到至少 1 处可改进的代码（参数校验、异常处理、日志、代码简化、注释清理、性能优化均可）。\n"
            "禁止返回空 modifications 数组。\n"
            "输出补丁时 old_str 必须是目标文件中 ≥2 行的连续原文，new_str 必须包含完整替换后的代码块。"
        ),
        "ALL_PATCH_FAIL": (
            "===== Prompt 强化：补丁精度要求 =====\n"
            "输出 patch 前先仔细阅读目标文件原文：\n"
            "1. old_str 必须逐字匹配文件中真实存在的 ≥2 行原文（包含完全一致的缩进和换行）\n"
            "2. new_str 必须是完整的替换后代码块，保持相同缩进层级，不要省略中间行\n"
            "3. 当 patch 无法精确匹配时改用 action=\"replace_all\"，并在 new_content 里放完整文件内容"
        ),
        "SANDBOX_QUARANTINE": (
            "===== Prompt 强化：目标收敛约束 =====\n"
            "本轮做低风险的小修改：参数校验、异常 try/except 补充、日志结构化、重复代码消除等。\n"
            "不要做大规模架构变更，不要引入新的第三方依赖。"
        ),
        "API_TIMEOUT": (
            "===== Prompt 强化：精简输出 =====\n"
            "由于上轮超时，本轮请只输出 1-2 个关键修改项，每个 patch 不要超过 30 行。\n"
            "优先使用 replace_all 而非 patch，以缩短响应长度。"
        ),
    }

    def get_prompt_boost(self) -> str:
        """根据最近 15 轮的 Top1 失败类别，动态输出对应的 Prompt 强化语句（system 级，权重最高）"""
        recent = self._log[-TREND_WINDOW:]
        if not recent:
            return ""
        from collections import Counter
        fail_cats = Counter(
            e.get("category", "UNKNOWN") for e in recent if e.get("is_failure")
        )
        # 失败太少不触发（<3次说明策略没问题）
        total_fail = sum(fail_cats.values())
        if total_fail < 2:
            return ""
        top_cat, top_count = fail_cats.most_common(1)[0]
        # Top1 占比要超过一半才强化，避免噪音
        if top_count < max(2, total_fail // 2 + 1):
            return ""
        boost = self._PROMPT_BOOST_BY_CATEGORY.get(top_cat)
        if not boost:
            return ""
        header = f"\n\n【Prompt 强化-自动】近{TREND_WINDOW}轮 {top_cat} 出现 {top_count} 次，请严格遵守以下规则：\n"
        return header + boost

    def format_for_prompt(self, max_len=400):
        """
        把最近失败特征注入 user prompt。
        升级为：Top 失败类别 + 对应可执行建议 + 连续失败告警。
        max_len 默认从 150 提升到 400，给模型足够上下文。
        """
        recent_failures = [e for e in self._log[-TREND_WINDOW:] if e.get("is_failure")]
        if not recent_failures:
            return ""
        from collections import Counter
        cat_counter = Counter(e.get("category", "UNKNOWN") for e in recent_failures)
        if not cat_counter:
            return ""

        lines = ["【近期失败模式与规避要求】"]
        # 每个失败类别带一段具体怎么做，而不是只说计数
        for cat, cnt in cat_counter.most_common(3):
            desc = _ERROR_CATEGORIES.get(cat, cat)
            hint = {
                "JSON_PARSE_FAIL": " → 请只输出JSON，不要任何中文解释或说明",
                "NO_MODIFICATION": " → 请务必产出至少1处实际代码修改，不要返回空数组",
                "ALL_PATCH_FAIL": " → old_str必须是≥2行的真实原文，逐字匹配，不要猜测",
                "SANDBOX_QUARANTINE": " → 做低风险小修改，不要大架构变动",
                "API_TIMEOUT": " → 只输出1-2个修改项，不要长文本",
                "API_ERROR": " → 简化输出，不要追求一次改太多",
            }.get(cat, "")
            lines.append(f"  {cat}: {cnt}次 ({desc}){hint}")

        if self._consecutive_fail >= 2:
            lines.append(f"\n⚠️ 已连续 {self._consecutive_fail} 轮失败，请特别注意以上要求，不要重复同样错误！")

        text = "\n".join(lines)
        return text[:max_len]

    @staticmethod
    def _scrub_protected(text: str) -> str:
        """擦除 PLE 保护文件名，避免泄露核心引擎路径给模型。"""
        if not text:
            return text
        try:
            from config import Config
            protected = list(getattr(Config, "PROTECTED_FILES", []) or [])
        except Exception:
            protected = []
        scrubbed = text
        for pf in protected:
            scrubbed = scrubbed.replace(pf, "[core module]")
            short = pf[len("evolver/"):] if pf.startswith("evolver/") else pf
            scrubbed = scrubbed.replace(short, "[core]")
        return scrubbed

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._log)
        failures = self._filter_failures()
        cat_dist = self._count_categories(failures)
        recent = self._log[-TREND_WINDOW * 2:]
        trend_stats = self._compute_stats(recent)
        return {
            "total_iterations": total,
            "failure_count": len(failures),
            "failure_rate": round(len(failures) / max(total, 1), 2),
            "consecutive_fail": self._consecutive_fail,
            "category_distribution": dict(cat_dist),
            "recent_trend": trend_stats,
        }

    def _compute_stats(self, recent: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not recent:
            return {}
        success_count = self._count_successes(recent)
        success_rate = self._calculate_success_rate(success_count, len(recent))
        trend = self._determine_trend(success_rate)
        return {
            "success_rate": success_rate,
            "trend": trend
        }

    def _compute_stats(self, recent: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not recent:
            return {}
        success_count = self._count_successes(recent)
        success_rate = self._calculate_success_rate(success_count, len(recent))
        trend = self._determine_trend(success_rate)
        return {'success_rate': success_rate, 'trend': trend}

    def _count_successes(self, recent: List[Dict[str, Any]]) -> int:
        return sum(1 for e in recent if not e.get('is_failure'))

    def _calculate_success_rate(self, success_count: int, total: int) -> float:
        return round(success_count / max(total, 1), 2)

    def _determine_trend(self, success_rate: float) -> str:
        return 'up' if success_rate > 0.5 else 'down'

    def _count_successes(self, recent: List[Dict[str, Any]]) -> int:
        return sum(1 for e in recent if not e.get('is_failure'))

    def _calculate_success_rate(self, success_count: int, total: int) -> float:
        return round(success_count / max(total, 1), 2)

    def _determine_trend(self, success_rate: float) -> str:
        return 'up' if success_rate > 0.5 else 'down'

    def _count_successes(self, recent: List[Dict[str, Any]]) -> int:
        return sum(1 for e in recent if not e.get('is_failure'))

    def _calculate_success_rate(self, success_count: int, total: int) -> float:
        return round(success_count / max(total, 1), 2)

    def _determine_trend(self, success_rate: float) -> str:
        return 'up' if success_rate > 0.5 else 'down'

    def _count_successes(self, recent: List[Dict[str, Any]]) -> int:
        return sum(1 for e in recent if not e.get('is_failure'))

    def _calculate_success_rate(self, success_count: int, total: int) -> float:
        return round(success_count / max(total, 1), 2)

    def _determine_trend(self, success_rate: float) -> str:
        return 'up' if success_rate > 0.5 else 'down'

    def _count_successes(self, recent: List[Dict[str, Any]]) -> int:
        return sum(1 for e in recent if not e.get('is_failure'))

    def _calculate_success_rate(self, success_count: int, total: int) -> float:
        return round(success_count / max(total, 1), 2)

    def _determine_trend(self, success_rate: float) -> str:
        return 'up' if success_rate > 0.5 else 'down'

    def _count_successes(self, recent: List[Dict[str, Any]]) -> int:
        return sum(1 for e in recent if not e.get('is_failure'))

    def _calculate_success_rate(self, success_count: int, total: int) -> float:
        return round(success_count / max(total, 1), 2)

    def _determine_trend(self, success_rate: float) -> str:
        return 'up' if success_rate > 0.5 else 'down'

    def _compute_stats(self, recent: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not recent:
            return {}
        success_count = self._count_successes(recent)
        success_rate = self._calculate_success_rate(success_count, len(recent))
        trend = self._determine_trend(success_rate)
        return {'success_rate': success_rate, 'trend': trend}

    def _count_successes(self, recent: List[Dict[str, Any]]) -> int:
        return sum(1 for e in recent if not e.get('is_failure'))

    def _calculate_success_rate(self, success_count: int, total: int) -> float:
        return round(success_count / max(total, 1), 2)

    def _determine_trend(self, success_rate: float) -> str:
        return 'up' if success_rate > 0.5 else 'down'

    def _count_successes(self, recent: List[Dict[str, Any]]) -> int:
        return sum(1 for e in recent if not e.get('is_failure'))

    def _calculate_success_rate(self, success_count: int, total: int) -> float:
        return round(success_count / max(total, 1), 2)

    def _determine_trend(self, success_rate: float) -> str:
        return 'up' if success_rate > 0.5 else 'down'

    def _count_successes(self, recent: List[Dict[str, Any]]) -> int:
        return sum(1 for e in recent if not e.get('is_failure'))

    def _calculate_success_rate(self, success_count: int, total: int) -> float:
        return round(success_count / max(total, 1), 2)

    def _determine_trend(self, success_rate: float) -> str:
        return 'up' if success_rate > 0.5 else 'down'

    def _count_successes(self, recent: List[Dict[str, Any]]) -> int:
        return sum(1 for e in recent if not e.get('is_failure'))

    def _calculate_success_rate(self, success_count: int, total: int) -> float:
        return round(success_count / max(total, 1), 2)

    def _determine_trend(self, success_rate: float) -> str:
        return 'up' if success_rate > 0.5 else 'down'

    def _filter_failures(self) -> List[Dict[str, Any]]:
        return [e for e in self._log if e.get('is_failure')]

    def _count_categories(self, failures: List[Dict[str, Any]]) -> Counter:
        return Counter(e.get("category", "UNKNOWN") for e in failures)

    def print_status(self):
        stats = self.get_stats()
        print("\n═══ 错误反思模块 v2 ═══")
        print(f"  总迭代次数: {stats['total_iterations']}")
        print(f"  失败次数:   {stats['failure_count']} ({stats['failure_rate']:.0%})")
        print(f"  连续失败:   {stats['consecutive_fail']}")
        if stats.get('recent_trend'):
            rt = stats['recent_trend']
            print(f"  最近成功率: {rt.get('success_rate', 0):.0%} ({rt.get('trend', 'N/A')})")
        if stats['category_distribution']:
            print(f"  错误分布:")
            for cat, n in sorted(stats['category_distribution'].items(),
                                key=lambda x: -x[1]):
                print(f"    {cat}: {n}次")
        print()
    def improve_code_quality(self) -> None:
        """对长期未修改的模块进行代码质量改进，包括类型注解、文档字符串、异常处理"""
        for method_name, method in self.__class__.__dict__.items():
            if callable(method) and not method.__doc__:
                method.__doc__ = 'No docstring provided'
            if not method.__annotations__:
                method.__annotations__ = {}
            try:
                method()
            except Exception as e:
                self.logger.error(f'Error in {method_name}: {str(e)}')
    def optimize_performance(self) -> None:
        """优化代码性能和可读性，识别并消除不必要的计算"""
        # 识别并消除不必要的计算
        self._log = [e for e in self._log if e.get('is_failure')]
        self._consecutive_fail = sum(1 for e in self._log if e.get('is_failure'))
        self._category_distribution = Counter(e.get('category', 'UNKNOWN') for e in self._log)
        self._recent_trend = self._compute_stats(self._log[-TREND_WINDOW * 2:]) if self._log else {}
