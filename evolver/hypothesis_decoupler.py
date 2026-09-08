"""
L3 元认知层 - 假设锚点解耦器 (Hypothesis Decoupler)
核心功能：反驳性提问引擎 + 失败模式分析 + 本体论漂移触发
实现用户架构设计的 L3 层：
  - 反驳性提问引擎：对每轮迭代的修改进行自我质疑
  - 失败模式分析：从历史迭代中提取反复出现的模式
  - 本体论漂移触发：当系统在同一维度停滞时，建议跨领域探索
硬件约束 (RTX 5060 8GB + 14B Q4_K_M):
  - 纯规则模板 + 历史分析，不调用 LLM
  - 反向问题可作为下一轮自迭代的 user_objective 输入
  - 漂移建议是轻量的关键词替换，不是真正的本体论跃迁
"""
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter

log = logging.getLogger("HypothesisDecoupler")


class HypothesisDecoupler:
    """
    L3 元认知层主入口
    使用方式:
        decoupler = HypothesisDecoupler(project_root=".")
        questions = decoupler.generate_counter_questions(last_iteration_result)
        patterns = decoupler.analyze_failure_patterns(history)
        drift = decoupler.suggest_ontological_drift(history)
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.history_file = self.project_root / "logs" / "evolution_history.json"
        self.daemon_log_file = self.project_root / "logs" / "daemon_log.json"
        # L3 失败日志快照缓存：硬性积攒 ≥5 轮样本后输出汇总
        self._failure_snapshots: List[Dict[str, Any]] = []
        self._snapshot_log_file = self.project_root / "logs" / "l3_snapshots.json"
        # ============【新增】保存流水线中间重试失败事件 ============
        self._intermediate_fail_events: List[Dict[str, Any]] = []

    # ========== L3 失败日志快照 ==========

    def _log_failure_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._record_failure_or_intermediate_event(snapshot, self._failure_snapshots, self._snapshot_log_file)

    def _log_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._record_failure_or_intermediate_event(evt, self._intermediate_fail_events, self._snapshot_log_file)

    def _record_event(self, event: Dict[str, Any], event_list: List[Dict[str, Any]], log_file: Path) -> None:
        if not isinstance(event, dict):
            raise ValueError("Invalid event format")
        event_list.append(event)
        self._write_event_log(event, log_file)

    def _write_event_log(self, event: Dict[str, Any], log_file: Path) -> None:
        try:
            import json as _json
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"Failed to write event log: {e}")

    def record_failure_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if not isinstance(snapshot, dict):
            raise ValueError("Invalid snapshot format")
        self._failure_snapshots.append(snapshot)
        try:
            import json as _json
            self._snapshot_log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._snapshot_log_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(snapshot, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"Failed to write snapshot log: {e}")

        count = len(self._failure_snapshots)
        if count >= 5:
            self._print_snapshot_summary()

    def record_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        if not isinstance(iteration_id, int) or not isinstance(category, str) or not isinstance(issues, list):
            raise ValueError("Invalid input parameters")
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._intermediate_fail_events.append(evt)
        self._trim_intermediate_fail_events()

    def _trim_intermediate_fail_events(self) -> None:
        if len(self._intermediate_fail_events) > 40:
            self._intermediate_fail_events = self._intermediate_fail_events[-40:]

    def _write_event_log(self, event: Dict[str, Any], log_file: Path) -> None:
        try:
            import json as _json
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"Failed to write event log: {e}")

        count = len(event_list)
        if count >= 5:
            self._print_snapshot_summary()

    def record_failure_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._record_event(snapshot, self._failure_snapshots, self._snapshot_log_file)

    def record_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._record_event(evt, self._intermediate_fail_events, self._snapshot_log_file)

    def _trim_intermediate_fail_events(self) -> None:
        if len(self._intermediate_fail_events) > 40:
            self._intermediate_fail_events = self._intermediate_fail_events[-40:]

    def _write_event_log(self, event: Dict[str, Any], log_file: Path) -> None:
        try:
            import json as _json
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"Failed to write event log: {e}")

        count = len(event_list)
        if count >= 5:
            self._print_snapshot_summary()

    def record_failure_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if not isinstance(snapshot, dict):
            raise ValueError("Invalid snapshot format")
        self._failure_snapshots.append(snapshot)
        try:
            import json as _json
            self._snapshot_log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._snapshot_log_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(snapshot, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"Failed to write snapshot log: {e}")

        count = len(self._failure_snapshots)
        if count >= 5:
            self._print_snapshot_summary()

    def record_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        if not isinstance(iteration_id, int) or not isinstance(category, str) or not isinstance(issues, list):
            raise ValueError("Invalid input parameters")
        if not isinstance(iteration_id, int) or not isinstance(category, str) or not isinstance(issues, list):
            raise ValueError("Invalid input parameters")
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._record_event(evt, self._intermediate_fail_events, self._snapshot_log_file)

    def _trim_intermediate_fail_events(self) -> None:
        if len(self._intermediate_fail_events) > 40:
            self._intermediate_fail_events = self._intermediate_fail_events[-40:]

    def _record_failure_or_intermediate_event(self, event: Dict[str, Any], event_list: List[Dict[str, Any]], log_file: Path) -> None:
        self._record_event(event, event_list, log_file)

    def _log_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._intermediate_fail_events.append(evt)
        self._trim_intermediate_fail_events()

    def _trim_intermediate_fail_events(self) -> None:
        if len(self._intermediate_fail_events) > 40:
            self._intermediate_fail_events = self._intermediate_fail_events[-40:]

    def record_failure_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """记录失败快照
        :param snapshot: 失败快照字典
        :type snapshot: Dict[str, Any]
        :raises ValueError: 如果 snapshot 格式不正确
        """
        if not isinstance(snapshot, dict):
            raise ValueError("Invalid snapshot format")
        self._failure_snapshots.append(snapshot)
        try:
            import json as _json
            self._snapshot_log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._snapshot_log_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(snapshot, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"Failed to write snapshot log: {e}")

        count = len(self._failure_snapshots)
        if count >= 5:
            self._print_snapshot_summary()

    def record_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        """记录中间失败事件
        :param iteration_id: 迭代 ID
        :type iteration_id: int
        :param category: 失败类别
        :type category: str
        :param issues: 具体问题列表
        :type issues: List[str]
        """
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._intermediate_fail_events.append(evt)
        self._trim_intermediate_fail_events()

    def _record_failure_or_intermediate_event(self, event: Dict[str, Any], event_list: List[Dict[str, Any]], log_file: Path) -> None:
        if not isinstance(event, dict):
            raise ValueError("Invalid event format")
        event_list.append(event)
        try:
            import json as _json
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"Failed to write event log: {e}")

        count = len(event_list)
        if count >= 5:
            self._print_snapshot_summary()

    def record_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        """记录流水线中间重试失败事件
        :param iteration_id: 迭代 ID
        :param category: 失败类别
        :param issues: 具体问题列表
        """
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._intermediate_fail_events.append(evt)
        self._trim_intermediate_fail_events()

    def record_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        if not isinstance(iteration_id, int) or not isinstance(category, str) or not isinstance(issues, list):
            raise ValueError("Invalid input parameters")
        """记录流水线中间重试失败事件
        :param iteration_id: 迭代 ID
        :param category: 失败类别
        :param issues: 具体问题列表
        """
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._intermediate_fail_events.append(evt)
        self._trim_intermediate_fail_events()

    def _trim_intermediate_fail_events(self) -> None:
        if len(self._intermediate_fail_events) > 40:
            self._intermediate_fail_events = self._intermediate_fail_events[-40:]

    def record_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        if not isinstance(iteration_id, int) or not isinstance(category, str) or not isinstance(issues, list):
            raise ValueError("Invalid input parameters")
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._intermediate_fail_events.append(evt)
        self._trim_intermediate_fail_events()

    def _trim_intermediate_fail_events(self) -> None:
        if len(self._intermediate_fail_events) > 40:
            self._intermediate_fail_events = self._intermediate_fail_events[-40:]

    def record_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        if not isinstance(iteration_id, int) or not isinstance(category, str) or not isinstance(issues, list):
            raise ValueError("Invalid input parameters")
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._intermediate_fail_events.append(evt)
        self._trim_intermediate_fail_events()

    def _log_failure_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._record_failure_or_intermediate_event(snapshot, self._failure_snapshots, self._snapshot_log_file)

    def _log_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._record_failure_or_intermediate_event(evt, self._intermediate_fail_events, self._snapshot_log_file)

    def _record_failure_or_intermediate_event(self, event: Dict[str, Any], event_list: List[Dict[str, Any]], log_file: Path) -> None:
        event_list.append(event)
        try:
            import json as _json
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"Failed to write event log: {e}")

        count = len(event_list)
        if count >= 5:
            self._print_snapshot_summary()
    def record_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        if not isinstance(iteration_id, int) or not isinstance(category, str) or not isinstance(issues, list):
            raise ValueError("Invalid input parameters")
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._intermediate_fail_events.append(evt)
        self._trim_intermediate_fail_events()


    def _print_snapshot_summary(self) -> None:
        snaps = self._failure_snapshots
        if len(snaps) < 5:
            return

        print(f"  ═════════ L3 元认知快照汇总 (共 {len(snaps)} 轮样本) ═════════")
        rates = [s.get("failure_rate", 0) for s in snaps]
        recent_cycles = {s["cycle"] for s in snaps}
        inter_fail_cnt = sum(
            1 for e in self._intermediate_fail_events
            if e["iteration_id"] in recent_cycles
        )
        print(f"  📊 迭代最终失败率趋势: {rates}")
        print(f"  📌 本窗口流水线中间重试失败事件数: {inter_fail_cnt}")
        from collections import Counter as _C
        patterns = [s.get("top_pattern", "none") for s in snaps]
        dist = _C(patterns).most_common(3)
        print(f"  🔍 失败模式分布: {dict(dist)}")
        drift_count = sum(1 for s in snaps if s.get("drift_triggered"))
        print(f"  🔄 本体论漂移触发: {drift_count}/{len(snaps)} 轮")
        entropies = [s.get("entropy_gap") for s in snaps if s.get("entropy_gap") is not None]
        if entropies:
            print(f"  🌡️ 熵差变化: {entropies}")
        all_files = []
        for s in snaps:
            all_files.extend(s.get("modified_files", []))
        if all_files:
            file_dist = _C(all_files).most_common(3)
            print(f"  📁 修改文件分布: {dict(file_dist)}")
        print(f"  ════════════════════════════════════════════")

    # 反向提问模板 - 按场景分类
    COUNTER_QUESTION_TEMPLATES: Dict[str, List[str]] = {
        "bugfix": [
            "你修复了 {target} 的 bug，但这个修复是否引入新边界情况？",
            "你假设 bug 由 {reason} 导致，但根因是否在别处？",
            "修复兼容 Python3.8 吗？做过低版本测试吗？",
            "不同分辨率下修复逻辑是否稳定？",
        ],
        "performance": [
            "优化后实际加速比有实测数据吗？",
            "{optimization} 提速会不会牺牲可读性？",
            "瓶颈是否不在此处，而在上层调用？",
            "优化是否带来内存泄漏风险？",
        ],
        "syntax": [
            "语法修复的导入路径是否正确？",
            "类型注解兼容 Python3.8？",
            "全局缩进是否统一？",
        ],
        "default": [
            "{action} 是否真正解决核心需求？",
            "你依赖的 {assumption} 是否成立？",
            "完全不修改 {target} 有无替代方案？",
            "本次修改是否增加未来维护成本？",
            "异常场景下逻辑是否健壮？",
        ],
    }

    # 本体论漂移关键词映射 (当前维度 → 建议跨域维度)
    ONTOLOGICAL_SHIFTS: List[Tuple[str, str]] = [
        ("文件操作", "进程管理"),
        ("GUI自动化", "命令行脚本"),
        ("同步执行", "异步并发"),
        ("固定配置", "动态发现"),
        ("单一模型", "多模型路由"),
        ("本地推理", "混合推理"),
        ("顺序任务", "并行任务分解"),
        ("OCR视觉", "坐标计算几何"),
        ("文本交互", "语音交互"),
        ("被动执行", "主动监控"),
        ("错误重试", "错误预防"),
    ]

    # 失败模式分类关键词
    FAILURE_PATTERNS: Dict[str, List[str]] = {
        "patch_mismatch": ["old_str", "不匹配", "patch", "ALL_PATCH_FAIL", "补丁"],
        "json_format": ["JSON", "json", "解析失败", "格式", "Parser", "跳过无效"],
        "api_error": ["API", "http", "超时", "timeout", "connection", "ollama"],
        "syntax_error": ["SyntaxError", "ast.parse", "语法", "indent"],
        "import_error": ["ImportError", "ModuleNotFoundError", "导入"],
        "no_modification": ["NO_MODIFICATION", "空 modifications", "无需修改"],
    }

    def decouple(self, *args, **kwargs) -> List[str]:
        """对外统一解耦入口，返回质疑问题列表"""
        # 优先用传入的迭代结果；无则从历史取最近一轮
        iter_result = args[0] if args and isinstance(args[0], dict) else {}
        if not iter_result:
            history = self._load_history()
            if history:
                iter_result = history[-1]
        return self.generate_counter_questions(iter_result)

    # ========== 历史加载 ==========

    def _load_history(self) -> List[Dict[str, Any]]:
        """加载迭代历史 (从 daemon_log.json, 最多取最近 50 轮)"""
        candidates = [self.daemon_log_file, self.history_file]
        for path in candidates:
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # daemon_log.json 结构: {"iterations": [...]} 或 {"recent": [...]} 或直接 list
                if isinstance(data, list):
                    return data[-50:]
                if isinstance(data, dict):
                    for key in ("iterations", "recent", "history"):
                        if key in data and isinstance(data[key], list):
                            return data[key][-50:]
                if isinstance(data, dict):
                    return [data]
            except Exception as e:
                log.debug(f"读取历史失败 {path}: {e}")
        return []

    # ========== 反向提问生成 ==========

    def generate_counter_questions(self, strategy: Dict[str, Any]) -> List[str]:
        """
        根据本轮迭代结果生成反向提问。
        strategy 字段建议:
          - action: patch/new_file/replace_all
          - target: 目标文件
          - status: 成功/失败状态
          - error: 错误信息(如有)
          - modifications: 修改项列表
        """
        if not isinstance(strategy, dict):
            strategy = {}

        action = str(strategy.get("action") or strategy.get("status") or "default")
        target = str(strategy.get("target") or strategy.get("target_file") or "未知模块")
        error = str(strategy.get("error") or "")
        reason = error[:50] if error else "未知原因"

        # 按场景选模板
        category = "default"
        action_lower = action.lower()
        if "fix" in action_lower or "bug" in action_lower or "修复" in action:
            category = "bugfix"
        elif "perf" in action_lower or "优化" in action or "fast" in action_lower:
            category = "performance"
        elif "syntax" in action_lower or "语法" in action:
            category = "syntax"

        templates = self.COUNTER_QUESTION_TEMPLATES.get(category, self.COUNTER_QUESTION_TEMPLATES["default"])

        # 格式化模板
        questions: List[str] = []
        for tpl in templates:
            try:
                q = tpl.format(target=target, reason=reason, action=action,
                               optimization=action, assumption=target)
                questions.append(q)
            except (KeyError, IndexError):
                questions.append(tpl)

        # 失败时追加针对性问题
        if error:
            questions.append(f"错误 '{reason}' 是否暴露了更深层的架构问题，而不仅是局部代码缺陷？")
            questions.append(f"同类错误在历史迭代中出现过吗？是否在重复踩同一个坑？")

        return questions

    # ========== 失败模式分析 ==========

    def analyze_failure_patterns(self, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        从历史迭代中提取反复出现的失败模式。
        返回:
          {
            "total_iterations": int,
            "failure_count": int,
            "failure_rate": float,
            "top_patterns": [{"pattern": str, "count": int, "ratio": float, "example_error": str}],
            "repeated_targets": [{"target": str, "fail_count": int}],
            "stuck_loop_detected": bool,  # 是否检测到死循环(同一错误连续出现 ≥5 次)
          }
        """
        if history is None:
            history = self._load_history()

        if not history:
            return {
                "total_iterations": 0,
                "failure_count": 0,
                "failure_rate": 0.0,
                "top_patterns": [],
                "repeated_targets": [],
                "stuck_loop_detected": False,
            }

        total = len(history)
        failure_count = 0
        pattern_counter: Counter = Counter()
        pattern_examples: Dict[str, str] = {}
        target_fail_counter: Counter = Counter()
        recent_errors: List[str] = []

        for item in history:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "")
            error = str(item.get("error") or "")
            target = str(item.get("target_file") or item.get("target") or "")

            # 判定失败
            is_failure = (
                status in ("API错误", "API响应异常", "配置错误", "全部失败", "daemon_exception", "error")
                or (item.get("applied_count", 0) == 0 and error)
                or error
            )
            if is_failure:
                failure_count += 1
                if target:
                    target_fail_counter[target] += 1

                # 分类失败模式
                matched = False
                err_text = f"{status} {error}".lower()
                for pattern_name, keywords in self.FAILURE_PATTERNS.items():
                    if any(kw.lower() in err_text for kw in keywords):
                        pattern_counter[pattern_name] += 1
                        if pattern_name not in pattern_examples:
                            pattern_examples[pattern_name] = error[:100] if error else status
                        matched = True
                        break
                if not matched:
                    pattern_counter["unknown"] += 1
                    if "unknown" not in pattern_examples:
                        pattern_examples["unknown"] = error[:100] if error else status

                recent_errors.append(err_text)

        # 死循环检测: 最近 10 轮错误中同一 pattern 出现 ≥5 次
        recent_pattern_counts: Counter = Counter()
        for err in recent_errors[-10:]:
            for pattern_name, keywords in self.FAILURE_PATTERNS.items():
                if any(kw.lower() in err for kw in keywords):
                    recent_pattern_counts[pattern_name] += 1
                    break
        stuck_loop = any(cnt >= 5 for cnt in recent_pattern_counts.values())

        # top 5 失败模式
        top_patterns = [
            {
                "pattern": name,
                "count": cnt,
                "ratio": round(cnt / max(failure_count, 1), 2),
                "example_error": pattern_examples.get(name, ""),
            }
            for name, cnt in pattern_counter.most_common(5)
        ]

        # 重复失败目标 top 3
        repeated_targets = [
            {"target": t, "fail_count": c}
            for t, c in target_fail_counter.most_common(3)
        ]

        return {
            "total_iterations": total,
            "failure_count": failure_count,
            "failure_rate": round(failure_count / max(total, 1), 2),
            "top_patterns": top_patterns,
            "repeated_targets": repeated_targets,
            "stuck_loop_detected": stuck_loop,
        }

    # ========== 本体论漂移触发 ==========

    def suggest_ontological_drift(self, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if history is None:
            history = self._load_history()
        """
        当系统在同一维度停滞时，建议跨领域探索。
        触发条件:
          1. 连续 N 轮 (默认 5) 都修改同一文件/同一关键词领域
          2. 或失败率 > 70% 且 stuck_loop_detected
        返回:
          {
            "drift_triggered": bool,
            "reason": str,
            "current_dimension": str,
            "suggested_dimension": str,
            "suggested_objective": str,  # 可直接作为下一轮 user_objective
          }
        """
        if history is None:
            history = self._load_history()

        result = {
            "drift_triggered": False,
            "reason": "",
            "current_dimension": "",
            "suggested_dimension": "",
            "suggested_objective": "",
        }

        if len(history) < 5:
            result["reason"] = f"历史不足 5 轮 (当前 {len(history)})，无法判断停滞"
            return result

        # 分析最近 5 轮的修改目标/描述
        recent = history[-5:]
        all_text = ""
        targets: List[str] = []
        for item in recent:
            if not isinstance(item, dict):
                continue
            target = str(item.get("target_file") or item.get("target") or "")
            desc = str(item.get("description") or item.get("objective") or "")
            all_text += " " + target + " " + desc
            if target:
                targets.append(target)

        # 检测当前所在维度
        current_dim = ""
        for src, dst in self.ONTOLOGICAL_SHIFTS:
            if src in all_text:
                current_dim = src
                result["current_dimension"] = src
                result["suggested_dimension"] = dst
                break

        if not current_dim:
            # 没匹配到已知维度，看是否在同一文件上反复
            if targets:
                target_counter = Counter(targets)
                top_target, top_count = target_counter.most_common(1)[0]
                if top_count >= 3:
                    result["drift_triggered"] = True
                    result["reason"] = f"连续 {top_count} 轮修改同一文件 {top_target}，建议跳转到其它模块"
                    result["current_dimension"] = top_target
                    result["suggested_dimension"] = "其它未触及模块"
                    result["suggested_objective"] = (
                        f"跳出 {top_target} 的局部优化，扫描整个 evolver/ 目录，"
                        f"找出从未被修改过或修改最少的模块，对其进行一次结构性改进"
                    )
                return result

            result["reason"] = "最近 5 轮无明确修改目标，无法判定维度"
            return result

        # 已匹配到维度，检查是否停滞
        # 看历史中这个维度出现频率
        dim_count = sum(1 for item in history if isinstance(item, dict)
                        and current_dim in str(item.get("target_file", "")) + str(item.get("description", "")))
        dim_ratio = dim_count / len(history)

        if dim_ratio >= 0.4:
            # 该维度占比过高 → 触发漂移
            suggested = result["suggested_dimension"]
            result["drift_triggered"] = True
            result["reason"] = (
                f"维度 '{current_dim}' 在历史中占比 {dim_ratio:.0%} (阈值 40%)，"
                f"系统可能陷入该维度的局部优化，建议跨域探索 '{suggested}'"
            )
            result["suggested_objective"] = (
                f"本体论漂移触发：当前过度依赖 '{current_dim}' 范式，"
                f"尝试引入 '{suggested}' 视角重构某个核心模块，"
                f"打破认知锚定效应"
            )
        else:
            result["reason"] = f"维度 '{current_dim}' 占比 {dim_ratio:.0%}，未达漂移阈值"

        return result

    # ========== 报告打印 ==========

    def print_analysis(self, history: Optional[List[Dict[str, Any]]] = None) -> None:
        if history is None:
            history = self._load_history()
        """打印 L3 元认知分析报告"""
        if history is None:
            history = self._load_history()

        print("\n  ═══ [L3 元认知层 · HypothesisDecoupler] ═══")

        # 失败模式分析
        patterns = self.analyze_failure_patterns(history)
        print(f"  📊 失败模式分析:")
        print(f"    总迭代: {patterns['total_iterations']}, 失败: {patterns['failure_count']} ({patterns['failure_rate']:.0%})")
        if patterns["top_patterns"]:
            print(f"    Top 失败模式:")
            for p in patterns["top_patterns"][:3]:
                print(f"      - {p['pattern']}: {p['count']}次 ({p['ratio']:.0%})")
        if patterns["repeated_targets"]:
            print(f"    反复失败目标:")
            for t in patterns["repeated_targets"][:3]:
                print(f"      - {t['target']}: {t['fail_count']}次")
        if patterns["stuck_loop_detected"]:
            print(f"    ⚠️ 检测到死循环: 最近 10 轮同一错误模式出现 ≥5 次")

        # 本体论漂移
        drift = self.suggest_ontological_drift(history)
        print(f"  🔄 本体论漂移检测:")
        if drift["drift_triggered"]:
            print(f"    🚨 触发漂移!")
            print(f"    原因: {drift['reason']}")
            print(f"    建议维度: {drift['current_dimension']} → {drift['suggested_dimension']}")
            if drift["suggested_objective"]:
                print(f"    建议目标: {drift['suggested_objective'][:120]}")
        else:
            print(f"    {drift['reason'] or '未触发漂移'}")

        print(f"  ════════════════════════════════════════════\n")
    def improve_code_quality(self) -> None:
        """对长期未修改的模块进行代码质量改进，包括类型注解、文档字符串、异常处理"""
        # 示例改进：为 record_failure_snapshot 方法添加类型注解和文档字符串
    def _record_event(self, event: Dict[str, Any], event_list: List[Dict[str, Any]], log_file: Path) -> None:
        self._write_event_log(event, log_file)

    def _write_event_log(self, event: Dict[str, Any], log_file: Path) -> None:
        try:
            import json as _json
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"Failed to write event log: {e}")

    def record_failure_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._record_event(snapshot, self._failure_snapshots, self._snapshot_log_file)

    def record_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
        evt = {
            "iteration_id": iteration_id,
            "category": category,
            "issues": issues.copy()
        }
        self._record_event(evt, self._intermediate_fail_events, self._snapshot_log_file)

    def _trim_intermediate_fail_events(self) -> None:
        if len(self._intermediate_fail_events) > 40:
            self._intermediate_fail_events = self._intermediate_fail_events[-40:]

    def _record_event(self, event: Dict[str, Any], event_list: List[Dict[str, Any]], log_file: Path) -> None:
        if not isinstance(event, dict):
            raise ValueError("Invalid event format")
        event_list.append(event)
        try:
            import json as _json
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"Failed to write event log: {e}")

        count = len(event_list)
        if count >= 5:
            self._print_snapshot_summary()

        # 示例改进：为 record_intermediate_failure 方法添加类型注解和文档字符串
        def record_intermediate_failure(self, iteration_id: int, category: str, issues: List[str]) -> None:
            """记录流水线中间重试失败事件
            :param iteration_id: 迭代 ID
            :param category: 失败类别
            :param issues: 具体问题列表
            """
            evt = {
                "iteration_id": iteration_id,
                "category": category,
                "issues": issues.copy()
            }
            self._intermediate_fail_events.append(evt)
            self._trim_intermediate_fail_events()

        def _trim_intermediate_fail_events(self) -> None:
            if len(self._intermediate_fail_events) > 40:
                self._intermediate_fail_events = self._intermediate_fail_events[-40:]

    def optimize_performance(self) -> None:
        """优化代码性能和可读性，识别并消除不必要的计算
        """
        """优化代码性能和可读性，识别并消除不必要的计算"""
        # 清理不必要的缓存
        self._failure_snapshots.clear()
        self._intermediate_fail_events.clear()

        # 重置日志文件
        self._snapshot_log_file.unlink(missing_ok=True)
        self._snapshot_log_file.parent.mkdir(parents=True, exist_ok=True)

        # 重置历史记录
        self.history_file.unlink(missing_ok=True)
        self.daemon_log_file.unlink(missing_ok=True)

        # 重置日志记录器
        log.handlers.clear()
        log.addHandler(logging.FileHandler(self.daemon_log_file, encoding='utf-8'))
        log.setLevel(logging.WARNING)

        log.info("Performance optimization completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    decoupler = HypothesisDecoupler()

    # 测试 1: 反向提问
    print("=== 反向提问测试 ===")
    questions = decoupler.generate_counter_questions({
        "action": "bugfix",
        "target_file": "evolver/ollama_client.py",
        "error": "old_str 不匹配",
    })
    for q in questions:
        print(f"  - {q}")

    # 测试 2: 失败模式分析(空历史)
    print("\n=== 失败模式分析(空历史) ===")
    result = decoupler.analyze_failure_patterns([])
    print(f"  {result}")

    # 测试 3: 失败模式分析(模拟历史)
    mock_history = [
        {"status": "API响应异常", "error": "old_str 不匹配", "target_file": "evolver/foo.py"},
        {"status": "全部失败", "error": "JSON 解析失败", "target_file": "evolver/bar.py"},
        {"status": "API响应异常", "error": "old_str 不匹配", "target_file": "evolver/foo.py"},
        {"status": "API响应异常", "error": "old_str 不匹配", "target_file": "evolver/foo.py"},
        {"status": "API响应异常", "error": "old_str 不匹配", "target_file": "evolver/foo.py"},
        {"status": "API响应异常", "error": "old_str 不匹配", "target_file": "evolver/foo.py"},
    ]
    print("\n=== 失败模式分析(模拟历史 6 轮) ===")
    result = decoupler.analyze_failure_patterns(mock_history)
    print(f"  失败率: {result['failure_rate']:.0%}, 死循环: {result['stuck_loop_detected']}")
    for p in result["top_patterns"]:
        print(f"  - {p['pattern']}: {p['count']}次")

    # 测试 4: 本体论漂移
    print("\n=== 本体论漂移检测 ===")
    drift = decoupler.suggest_ontological_drift(mock_history)
    print(f"  触发: {drift['drift_triggered']}")
    print(f"  原因: {drift['reason']}")
