"""
成功模式库 - 存储被判定为"改善"的修改案例
这些案例会被注入到 AI 的 Prompt 中，指导后续行为
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from config import Config


@dataclass
class SuccessPattern:
    """一个成功的修改案例"""
    pattern_id: str  # 唯一标识（基于文件+目标+改动内容的hash）
    iteration: int
    objective: str
    file_path: str
    old_snippet: str  # 修改前的代码片段（前200字符）
    new_snippet: str  # 修改后的代码片段（前200字符）
    quality_improvement: float  # 质量提升百分比
    verdict: str  # 判定结果
    dimensions_improved: List[str]  # 改善的维度列表
    summary: str  # 人类可读的总结
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    times_used: int = 0  # 被后续迭代参考的次数
    success_rate: float = 1.0  # 该模式后续的复用成功率
    related_files: List[str] = field(default_factory=list)  # 关联文件
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuccessPattern":
        return cls(**data)
    
    def generate_id(self) -> str:
        """生成唯一ID"""
        content = f"{self.file_path}|{self.objective}|{self.old_snippet[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class SuccessPatternLibrary:
    """
    成功模式库 - 存储和管理"改善"案例
    支持：添加、检索、格式化、去重、老化淘汰
    """
    
    MAX_PATTERNS = 50  # 最多保留50个模式
    MIN_IMPROVEMENT = 0.02  # 最低质量提升阈值（2%）
    PATTERN_EXPIRY_DAYS = 30  # 30天未使用则淘汰
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Config.LOG_DIR / "success_patterns.json"
        self.patterns: List[SuccessPattern] = []
        self._load()
    
    def add_pattern(self, pattern: SuccessPattern) -> bool:
        """
        添加一个新的成功模式
        返回: 是否成功添加
        """
        # 检查是否已达到阈值
        if pattern.quality_improvement < self.MIN_IMPROVEMENT:
            return False
        
        # 生成唯一ID
        pattern.pattern_id = pattern.generate_id()
        
        # 检查是否已存在（去重）
        for i, p in enumerate(self.patterns):
            if p.pattern_id == pattern.pattern_id:
                # 更新已有模式（保留最新的）
                self.patterns[i] = pattern
                self._save()
                return True
        
        # 添加新模式
        self.patterns.append(pattern)
        
        # 如果超出最大数量，淘汰最旧的
        if len(self.patterns) > self.MAX_PATTERNS:
            self._evict_oldest()
        
        self._save()
        return True
    # 在 SuccessPatternLibrary 类中添加以下方法

    def export(self, file_path: Path) -> bool:
        """导出成功模式库到文件"""
        try:
            data = {
                "version": 1,
                "exported_at": datetime.now().isoformat(),
                "total_patterns": len(self.patterns),
                "patterns": [p.to_dict() for p in self.patterns],
                "statistics": self.get_statistics(),
            }
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ 已导出 {len(self.patterns)} 条成功经验到 {file_path}")
            return True
        except Exception as e:
            print(f"  ❌ 导出失败: {e}")
            return False

    def import_from_file(self, file_path: Path, merge: bool = True) -> int:
        """
        从文件导入成功模式
        :param merge: True=合并，False=替换
        :return: 成功导入的条数
        """
        if not file_path.exists():
            print(f"  ❌ 文件不存在: {file_path}")
            return 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            imported_patterns = data.get("patterns", [])
            if not imported_patterns:
                print("  ⚠️ 文件中没有成功模式")
                return 0

            imported_count = 0
            existing_ids = {p.pattern_id for p in self.patterns}

            for p_data in imported_patterns:
                pattern = SuccessPattern.from_dict(p_data)
                pattern.pattern_id = pattern.generate_id()  # 重新生成 ID

                if pattern.pattern_id in existing_ids:
                    # 已存在，跳过
                    continue

                self.patterns.append(pattern)
                imported_count += 1

            # 如果超出最大数量，淘汰最旧的
            if len(self.patterns) > self.MAX_PATTERNS:
                self._evict_oldest()

            self._save()
            print(f"  ✅ 成功导入 {imported_count} 条经验（跳过 {len(imported_patterns) - imported_count} 条重复）")
            return imported_count

        except Exception as e:
            print(f"  ❌ 导入失败: {e}")
            return 0
    
    def get_top_patterns(self, limit: int = 5) -> List[SuccessPattern]:
        """获取综合评分最高的模式（基于成功率和使用频率）"""
        sorted_patterns = sorted(
            self.patterns,
            key=lambda p: (
                p.success_rate * 0.6 + 
                (p.times_used / max(1, max(p.times_used for p in self.patterns))) * 0.4
            ),
            reverse=True
        )
        return sorted_patterns[:limit]
    
    def get_recent_patterns(self, limit: int = 5) -> List[SuccessPattern]:
        """获取最近的成功模式"""
        sorted_patterns = sorted(
            self.patterns,
            key=lambda p: p.iteration,
            reverse=True
        )
        return sorted_patterns[:limit]
    
    def get_patterns_by_file(self, file_path: str, limit: int = 3) -> List[SuccessPattern]:
        """获取特定文件的成功模式"""
        matched = [p for p in self.patterns if file_path in p.file_path or file_path in p.related_files]
        # 按迭代号排序（最新的优先）
        matched.sort(key=lambda p: p.iteration, reverse=True)
        return matched[:limit]
    
    def get_patterns_by_dimension(self, dimension: str, limit: int = 3) -> List[SuccessPattern]:
        """获取改善了特定维度的模式"""
        matched = [p for p in self.patterns if dimension in p.dimensions_improved]
        matched.sort(key=lambda p: p.quality_improvement, reverse=True)
        return matched[:limit]
    
    def record_usage(self, pattern_id: str, success: bool) -> None:
        """记录模式被复用后的结果"""
        for p in self.patterns:
            if p.pattern_id == pattern_id:
                p.times_used += 1
                # 更新成功率（指数移动平均）
                if success:
                    p.success_rate = min(1.0, p.success_rate * 1.1)
                else:
                    p.success_rate = max(0.0, p.success_rate * 0.9)
                self._save()
                break
    
    def format_for_prompt(self, limit: int = 3, file_context: Optional[str] = None) -> str:
        """
        格式化成功模式，用于注入 Prompt
        如果提供了 file_context，优先返回相关文件的模式
        """
        if not self.patterns:
            return "（暂无成功案例库，请尝试从基础质量改进入手）"
        
        # 如果提供了文件上下文，优先匹配相关文件
        if file_context:
            file_patterns = self.get_patterns_by_file(file_context, limit=limit)
            if file_patterns:
                patterns = file_patterns
            else:
                patterns = self.get_top_patterns(limit)
        else:
            patterns = self.get_top_patterns(limit)
        
        if not patterns:
            return "（暂无相关成功案例）"
        
        lines = ["=== 📚 成功的修改案例（参考学习） ===\n"]
        lines.append("以下是之前被判定为'改善'的修改案例，你可以从中学习：\n")
        
        for i, p in enumerate(patterns, 1):
            lines.append(f"## 案例 {i}（迭代 #{p.iteration}）")
            lines.append(f"- **目标**：{p.objective[:100]}")
            lines.append(f"- **文件**：{p.file_path}")
            lines.append(f"- **质量提升**：{p.quality_improvement:.1%}")
            lines.append(f"- **改善维度**：{', '.join(p.dimensions_improved) if p.dimensions_improved else '整体质量'}")
            lines.append(f"- **总结**：{p.summary}")
            lines.append(f"- **复用次数**：{p.times_used} 次，成功率 {p.success_rate:.0%}")
            
            # 显示代码片段
            if p.old_snippet and p.new_snippet:
                lines.append(f"- **修改示例**：")
                lines.append(f"  ```python")
                lines.append(f"  # 修改前：")
                lines.append(f"  {p.old_snippet[:150]}...")
                lines.append(f"  # 修改后：")
                lines.append(f"  {p.new_snippet[:150]}...")
                lines.append(f"  ```")
            lines.append("")
        
        lines.append("---")
        lines.append("💡 **建议**：参考以上成功案例的模式，尝试类似的改进方向。")
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.patterns:
            return {"total": 0}
        
        total = len(self.patterns)
        avg_improvement = sum(p.quality_improvement for p in self.patterns) / total
        avg_success_rate = sum(p.success_rate for p in self.patterns) / total
        total_uses = sum(p.times_used for p in self.patterns)
        
        # 统计各维度出现频率
        dimension_counts: Dict[str, int] = {}
        for p in self.patterns:
            for dim in p.dimensions_improved:
                dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        
        return {
            "total": total,
            "avg_improvement": round(avg_improvement, 3),
            "avg_success_rate": round(avg_success_rate, 3),
            "total_uses": total_uses,
            "top_dimensions": sorted(
                dimension_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
        }
    
    def _evict_oldest(self) -> None:
        """淘汰最旧的模式（基于迭代号）"""
        self.patterns.sort(key=lambda p: p.iteration)
        # 移除最旧的25%
        evict_count = max(1, len(self.patterns) // 4)
        self.patterns = self.patterns[evict_count:]
    
    def _load(self) -> None:
        """从磁盘加载"""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.patterns = [SuccessPattern.from_dict(p) for p in data]
        except Exception as e:
            print(f"[SuccessPattern] 加载失败: {e}")
    
    def _save(self) -> None:
        """保存到磁盘"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(
                    [p.to_dict() for p in self.patterns],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        except Exception as e:
            print(f"[SuccessPattern] 保存失败: {e}")


# 全局单例
_global_pattern_library: Optional[SuccessPatternLibrary] = None


def get_pattern_library() -> SuccessPatternLibrary:
    """获取全局成功模式库单例"""
    global _global_pattern_library
    if _global_pattern_library is None:
        _global_pattern_library = SuccessPatternLibrary()
    return _global_pattern_library


def create_pattern_from_report(
    report: BenchReport,
    iteration: int,
    objective: str,
    modified_files: List[str],
    old_snippets: Dict[str, str],
    new_snippets: Dict[str, str]
) -> Optional[SuccessPattern]:
    """
    从基准报告创建成功模式
    只有在判定为"进步"时才创建
    """
    # 只有"进步"才记录
    if report.verdict != "进步" and "进步" not in report.verdict:
        return None
    
    # 计算综合质量提升
    quality_improvement = 0.0
    if hasattr(report, 'quality_delta') and report.quality_delta is not None:
        quality_improvement = report.quality_delta
    else:
        # 如果没有 quality_delta，使用 score_delta 归一化
        quality_improvement = report.score_delta / 100
    
    if quality_improvement < 0.02:
        return None
    
    # 提取改善的维度
    dimensions_improved = list(report.improving_tasks) if report.improving_tasks else []
    if hasattr(report, 'improving_tasks') and "quality_score" in report.improving_tasks:
        dimensions_improved.append("quality_score")
    
    if not dimensions_improved:
        dimensions_improved = ["整体质量"]
    
    # 生成摘要
    summary = f"质量评分提升 {quality_improvement:.1%}，"
    if dimensions_improved:
        summary += f"改善维度: {', '.join(dimensions_improved[:3])}"
    else:
        summary += "整体质量改善"
    
    # 获取代码片段
    old_snippet = ""
    new_snippet = ""
    if modified_files and old_snippets and new_snippets:
        first_file = modified_files[0]
        old_snippet = old_snippets.get(first_file, "")[:200]
        new_snippet = new_snippets.get(first_file, "")[:200]
    
    # 创建模式
    pattern = SuccessPattern(
        pattern_id="",  # 将由 add_pattern 生成
        iteration=iteration,
        objective=objective[:150],
        file_path=", ".join(modified_files[:3]),
        old_snippet=old_snippet,
        new_snippet=new_snippet,
        quality_improvement=quality_improvement,
        verdict=report.verdict,
        dimensions_improved=dimensions_improved,
        summary=summary,
        related_files=modified_files,
    )
    pattern.pattern_id = pattern.generate_id()
    
    return pattern


# 在 self_evolver.py 中集成时使用：
# from success_patterns import get_pattern_library, create_pattern_from_report