"""
目标优先级排序器 - 根据影响面决定修改顺序

核心功能：
1. 收集各文件的状态数据（Contract 失败、黑名单、修改历史）
2. 计算影响面评分
3. 生成按优先级排序的目标列表
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter

from config import Config


@dataclass
class FilePriority:
    """文件的优先级信息"""
    file_path: str
    score: float              # 综合评分，越高越优先
    contract_failures: int    # Contract 失败次数
    blacklisted: bool         # 是否在黑名单中
    blacklist_remaining: int  # 剩余禁改轮数
    recent_modifications: int # 最近 20 轮修改次数
    error_severity: int       # 0-4, 4=最严重
    last_modified_iter: int   # 最后修改的迭代号
    reason: str               # 优先级原因

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "score": round(self.score, 2),
            "contract_failures": self.contract_failures,
            "blacklisted": self.blacklisted,
            "blacklist_remaining": self.blacklist_remaining,
            "recent_modifications": self.recent_modifications,
            "error_severity": self.error_severity,
            "last_modified_iter": self.last_modified_iter,
            "reason": self.reason,
        }


class PriorityPlanner:
    """目标优先级排序器"""

    # 错误严重性等级
    SEVERITY_SYNTAX = 4      # 语法错误最严重
    SEVERITY_IMPORT = 3      # 导入错误
    SEVERITY_CONTRACT = 2    # Contract 失败
    SEVERITY_TEST = 1        # 测试失败
    SEVERITY_NONE = 0        # 无已知问题

    def __init__(self):
        self._contract_failures: Dict[str, int] = {}
        self._blacklist: Dict[str, Dict] = {}
        self._modification_history: List[Dict] = []
        self._load_data()

    def _load_data(self) -> None:
        """加载各类数据"""
        # 1. 加载黑名单
        bl_path = Config.LOG_DIR / "modification_blacklist.json"
        if bl_path.exists():
            try:
                with open(bl_path, "r", encoding="utf-8") as f:
                    self._blacklist = json.load(f)
            except Exception:
                self._blacklist = {}

        # 2. 加载迭代历史（用于统计修改频率和错误）
        history_path = Config.LOG_DIR / "evolution_history.json"
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._modification_history = data
                    elif isinstance(data, dict) and "history" in data:
                        self._modification_history = data["history"]
            except Exception:
                self._modification_history = []

        # 3. 从历史中提取 Contract 失败
        self._extract_contract_failures()

    # 仅统计最近窗口内的 Contract 失败（修复闭环：避免历史累计让已修复文件永远霸占第一优先级）
    FAILURE_HISTORY_WINDOW = 60

    def _extract_contract_failures(self, window: int = None) -> None:
        """从历史记录中提取每个文件的 Contract 失败次数（只统计最近 window 条迭代）。

        修复闭环：失败计数必须是"最近状态"而非"历史累计"。
        此前 branch_manager.py 曾累计失败 5 次 → 永久排第一 → 目标永不轮换。
        窗口化后，一旦文件被修好、后续无新失败，计数自然衰减归零，优先级回落。
        """
        if window is None:
            window = self.FAILURE_HISTORY_WINDOW
        for item in self._modification_history[-window:]:
            if not isinstance(item, dict):
                continue
            contract = item.get("contract", {})
            if not isinstance(contract, dict):
                continue
            if contract.get("failed", 0) > 0:
                # 尝试从 applied_details 中获取文件列表
                details = item.get("applied_details", [])
                if isinstance(details, list):
                    for d in details:
                        if isinstance(d, dict):
                            file_path = d.get("file", "")
                            if file_path:
                                self._contract_failures[file_path] = (
                                    self._contract_failures.get(file_path, 0) + 1
                                )

    def record_outcome(self, file_path: str, success: bool) -> None:
        """增量维护某个文件的 Contract 失败计数（daemon 每轮迭代后调用）。

        success=True  → 该文件本轮 Contract 通过 → 清空累计失败，优先级回落
        success=False → 该文件本轮 Contract 失败 → 计数 +1，优先级提升
        """
        if not file_path:
            return
        if success:
            if file_path in self._contract_failures:
                del self._contract_failures[file_path]
        else:
            self._contract_failures[file_path] = self._contract_failures.get(file_path, 0) + 1

    def calculate_priority(
        self,
        file_path: str,
        current_iteration: int,
        error_type: Optional[str] = None,
    ) -> FilePriority:
        """计算单个文件的优先级"""
        # 1. Contract 失败次数
        contract_failures = self._contract_failures.get(file_path, 0)

        # 2. 黑名单状态
        bl_entry = self._blacklist.get(file_path, {})
        blacklisted = bl_entry.get("banned_until_iter", 0) > current_iteration
        blacklist_remaining = max(0, bl_entry.get("banned_until_iter", 0) - current_iteration)

        # 3. 最近修改次数（最近 20 轮）
        recent_mods = 0
        last_modified_iter = 0
        for item in self._modification_history[-20:]:
            if not isinstance(item, dict):
                continue
            details = item.get("applied_details", [])
            if isinstance(details, list):
                for d in details:
                    if isinstance(d, dict) and d.get("file") == file_path:
                        recent_mods += 1
                        last_modified_iter = max(last_modified_iter, item.get("iteration", 0))

        # 4. 错误严重性
        error_severity = self._get_error_severity(file_path, error_type)

        # 5. 计算综合评分
        score = self._calculate_score(
            contract_failures=contract_failures,
            blacklisted=blacklisted,
            blacklist_remaining=blacklist_remaining,
            recent_mods=recent_mods,
            error_severity=error_severity,
        )

        # 6. 生成优先级原因
        reason = self._generate_reason(
            contract_failures=contract_failures,
            blacklisted=blacklisted,
            blacklist_remaining=blacklist_remaining,
            recent_mods=recent_mods,
            error_severity=error_severity,
        )

        return FilePriority(
            file_path=file_path,
            score=score,
            contract_failures=contract_failures,
            blacklisted=blacklisted,
            blacklist_remaining=blacklist_remaining,
            recent_modifications=recent_mods,
            error_severity=error_severity,
            last_modified_iter=last_modified_iter,
            reason=reason,
        )

    def _get_error_severity(self, file_path: str, error_type: Optional[str]) -> int:
        """获取错误严重性"""
        if error_type:
            severity_map = {
                "syntax": self.SEVERITY_SYNTAX,
                "import": self.SEVERITY_IMPORT,
                "contract": self.SEVERITY_CONTRACT,
                "test": self.SEVERITY_TEST,
            }
            return severity_map.get(error_type, self.SEVERITY_NONE)

        # 如果没有指定错误类型，根据历史推断
        # 检查是否有该文件的 Contract 失败
        if self._contract_failures.get(file_path, 0) > 3:
            return self.SEVERITY_CONTRACT
        if self._contract_failures.get(file_path, 0) > 0:
            return self.SEVERITY_CONTRACT

        # 检查黑名单
        if file_path in self._blacklist:
            return self.SEVERITY_SYNTAX

        return self.SEVERITY_NONE

    def _calculate_score(
        self,
        contract_failures: int,
        blacklisted: bool,
        blacklist_remaining: int,
        recent_mods: int,
        error_severity: int,
    ) -> float:
        """计算综合评分"""
        score = 0.0

        # Contract 失败：每次 +5 分，上限 30 分
        score += min(contract_failures * 2 , 30)

        # 黑名单：+20 分（说明有严重问题）
        if blacklisted:
            score += 20 + min(blacklist_remaining, 20)

        # 最近修改次数：越少越优先（+5 分/次，但超过 3 次开始扣分）
        if recent_mods <= 1:
            score += 15  # 长期未修改，优先级高
        elif recent_mods <= 3:
            score += 10
        elif recent_mods <= 5:
            score += 5
        else:
            score -= min(recent_mods - 5, 10)  # 过度修改，降低优先级

        # 错误严重性
        score += error_severity * 5

        return max(0, score)

    def _generate_reason(
        self,
        contract_failures: int,
        blacklisted: bool,
        blacklist_remaining: int,
        recent_mods: int,
        error_severity: int,
    ) -> str:
        """生成优先级原因"""
        reasons = []

        if error_severity >= 4:
            reasons.append("存在语法错误")
        elif error_severity >= 3:
            reasons.append("存在导入错误")
        elif contract_failures >= 3:
            reasons.append(f"Contract 失败 {contract_failures} 次")
        elif contract_failures > 0:
            reasons.append(f"Contract 失败 {contract_failures} 次")

        if blacklisted:
            reasons.append(f"黑名单中 (剩余 {blacklist_remaining} 轮)")

        if recent_mods <= 1:
            reasons.append("长期未修改")
        elif recent_mods >= 6:
            reasons.append(f"最近修改 {recent_mods} 次，需冷却")

        return ", ".join(reasons) if reasons else "状态良好"

    def get_prioritized_files(
        self,
        files: List[str],
        current_iteration: int,
        error_map: Optional[Dict[str, str]] = None,
    ) -> List[FilePriority]:
        """
        获取按优先级排序的文件列表
        error_map: {file_path: error_type}
        """
        priorities = []

        for file_path in files:
            error_type = error_map.get(file_path) if error_map else None
            priority = self.calculate_priority(file_path, current_iteration, error_type)
            priorities.append(priority)

        # 按评分降序排列
        priorities.sort(key=lambda p: p.score, reverse=True)
        return priorities

    def get_next_objective(
        self,
        files: List[str],
        current_iteration: int,
        error_map: Optional[Dict[str, str]] = None,
        recent_objectives: Optional[List[str]] = None,
    ) -> Tuple[str, FilePriority]:
        """
        获取下一个最优目标
        返回: (目标描述, FilePriority)
        """
        priorities = self.get_prioritized_files(files, current_iteration, error_map)

        if not priorities:
            # ✅ 修复：返回一个有效的 FilePriority 对象
            empty_priority = FilePriority(
                file_path="",
                score=0,
                contract_failures=0,
                blacklisted=False,
                blacklist_remaining=0,
                recent_modifications=0,
                error_severity=0,
                last_modified_iter=0,
                reason="无可用文件"
            )
            return "无可用目标", empty_priority

        # 如果最近目标中有相同文件，跳过（避免重复）
        if recent_objectives:
            for p in priorities:
                if p.file_path not in recent_objectives:
                    return self._make_objective(p), p

        # 否则取第一个
        return self._make_objective(priorities[0]), priorities[0]

    def _make_objective(self, priority: FilePriority) -> str:
        """根据优先级生成目标描述"""
        file_path = priority.file_path
        score = priority.score
        reason = priority.reason

        if priority.error_severity >= 4:
            return f"修复 {file_path} 的语法错误（优先级 {score:.0f}）"
        elif priority.error_severity >= 3:
            return f"修复 {file_path} 的导入错误（优先级 {score:.0f}）"
        elif priority.contract_failures >= 3:
            return f"优化 {file_path} 的合约验证（失败 {priority.contract_failures} 次，优先级 {score:.0f}）"
        elif priority.blacklisted:
            return f"检查 {file_path} 的黑名单状态（剩余 {priority.blacklist_remaining} 轮，优先级 {score:.0f}）"
        elif priority.recent_modifications <= 1:
            return f"初步改进 {file_path}（长期未修改，优先级 {score:.0f}）"
        else:
            return f"常规优化 {file_path}（优先级 {score:.0f}）"


# 全局单例
_priority_planner: Optional[PriorityPlanner] = None


def get_priority_planner() -> PriorityPlanner:
    global _priority_planner
    if _priority_planner is None:
        _priority_planner = PriorityPlanner()
    return _priority_planner