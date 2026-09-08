"""
细粒度回滚系统 - 只恢复有问题的代码片段，保留好的修改

核心功能：
1. 对比修改前后的代码差异
2. 根据错误信息定位问题代码行
3. 只回滚问题片段，保留其他修改
"""

import ast
import difflib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field

from config import Config


@dataclass
class RollbackSuggestion:
    """回滚建议"""
    file_path: str
    start_line: int
    end_line: int
    original_code: str       # 修改前的代码
    current_code: str        # 修改后的代码
    issue_type: str          # syntax / import / contract / test
    issue_detail: str
    confidence: float        # 0-1 置信度


class FineGrainedRollback:
    """细粒度回滚引擎"""

    def __init__(self, code_manager):
        self.code_mgr = code_manager

    def analyze_and_suggest(
        self,
        file_path: str,
        error_type: str,
        error_msg: str,
    ) -> List[RollbackSuggestion]:
        """
        分析错误，生成回滚建议
        error_type: syntax / import / contract / test
        """
        suggestions = []

        # 获取当前文件和备份
        current_content = self.code_mgr.read_file(file_path)
        if not current_content:
            return suggestions

        # 找到最近的备份
        backup_path = self._find_latest_backup(file_path)
        if not backup_path:
            return suggestions

        backup_content = self.code_mgr.read_file(str(backup_path))
        if not backup_content:
            return suggestions

        # 根据错误类型定位问题
        if error_type == "syntax":
            suggestions = self._analyze_syntax_error(
                file_path, current_content, backup_content, error_msg
            )
        elif error_type == "import":
            suggestions = self._analyze_import_error(
                file_path, current_content, backup_content, error_msg
            )
        elif error_type in ("contract", "test"):
            suggestions = self._analyze_runtime_error(
                file_path, current_content, backup_content, error_msg
            )

        return suggestions

    def execute_rollback(
        self,
        suggestion: RollbackSuggestion,
    ) -> Tuple[bool, str]:
        """
        执行回滚建议
        返回: (成功, 信息)
        """
        try:
            content = self.code_mgr.read_file(suggestion.file_path)
            if not content:
                return False, "无法读取文件"

            lines = content.splitlines()
            start = suggestion.start_line - 1  # 转 0-indexed
            end = suggestion.end_line  # 切片不包含 end

            # 替换问题片段
            new_lines = lines[:start] + suggestion.original_code.splitlines() + lines[end:]
            new_content = "\n".join(new_lines)

            # 写入文件
            success, info = self.code_mgr.write_file(
                suggestion.file_path,
                new_content,
                create_backup=True,
                tag=f"rollback_{suggestion.issue_type}",
            )

            if success:
                return True, f"已回滚 {suggestion.file_path} 第 {suggestion.start_line}-{suggestion.end_line} 行"
            return False, info

        except Exception as e:
            return False, f"回滚执行异常: {e}"

    def _find_latest_backup(self, file_path: str) -> Optional[Path]:
        """找到该文件最近的备份"""
        backups = self.code_mgr.list_backups()
        safe_name = file_path.replace("/", "__").replace("\\", "__")
        matches = [b for b in backups if safe_name in b.name]
        if not matches:
            return None
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    def _analyze_syntax_error(
        self,
        file_path: str,
        current_content: str,
        backup_content: str,
        error_msg: str,
    ) -> List[RollbackSuggestion]:
        """分析语法错误，定位问题行"""
        suggestions = []

        # 提取错误行号
        line_match = re.search(r'line (\d+)', error_msg)
        if not line_match:
            return suggestions

        error_line = int(line_match.group(1))

        # 获取修改前后的差异
        diff_lines = self._get_diff_lines(current_content, backup_content)

        # 找到包含错误行的修改区域
        affected_region = self._find_affected_region(
            diff_lines, error_line, current_content
        )

        if affected_region:
            original_code = self._extract_lines(backup_content, affected_region)
            current_code = self._extract_lines(current_content, affected_region)

            suggestions.append(RollbackSuggestion(
                file_path=file_path,
                start_line=affected_region[0],
                end_line=affected_region[1],
                original_code=original_code,
                current_code=current_code,
                issue_type="syntax",
                issue_detail=error_msg[:200],
                confidence=0.9,
            ))

        return suggestions

    def _analyze_import_error(
        self,
        file_path: str,
        current_content: str,
        backup_content: str,
        error_msg: str,
    ) -> List[RollbackSuggestion]:
        """分析导入错误"""
        suggestions = []

        # 提取缺失的模块名
        module_match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", error_msg)
        if not module_match:
            return suggestions

        # 查找包含该 import 的修改区域
        import_line = None
        lines = current_content.splitlines()
        for i, line in enumerate(lines, 1):
            if f"import {module_match.group(1)}" in line or f"from {module_match.group(1)}" in line:
                import_line = i
                break

        if import_line:
            # 找到包含该 import 的修改块
            diff_lines = self._get_diff_lines(current_content, backup_content)
            affected = self._find_region_by_line(diff_lines, import_line)

            if affected:
                original = self._extract_lines(backup_content, affected)
                current = self._extract_lines(current_content, affected)

                suggestions.append(RollbackSuggestion(
                    file_path=file_path,
                    start_line=affected[0],
                    end_line=affected[1],
                    original_code=original,
                    current_code=current,
                    issue_type="import",
                    issue_detail=error_msg[:200],
                    confidence=0.85,
                ))

        return suggestions

    def _analyze_runtime_error(
        self,
        file_path: str,
        current_content: str,
        backup_content: str,
        error_msg: str,
    ) -> List[RollbackSuggestion]:
        """分析运行时错误（Contract/Test 失败）"""
        suggestions = []

        # 尝试从错误信息中提取函数/方法名
        func_match = re.search(r"'(\\w+)'", error_msg)
        if not func_match:
            return suggestions

        func_name = func_match.group(1)

        # 找到该函数在文件中的位置
        func_start = None
        func_end = None
        lines = current_content.splitlines()
        in_func = False
        indent = None
        for i, line in enumerate(lines, 1):
            if f"def {func_name}" in line or f"async def {func_name}" in line:
                func_start = i
                in_func = True
                indent = len(line) - len(line.lstrip())
                continue
            if in_func and line.strip() and len(line) - len(line.lstrip()) <= indent:
                func_end = i - 1
                break
        if in_func and func_end is None:
            func_end = len(lines)

        if func_start and func_end:
            original = self._extract_lines(backup_content, (func_start, func_end))
            current = self._extract_lines(current_content, (func_start, func_end))

            # 检查这个函数是否被修改过
            if original != current:
                suggestions.append(RollbackSuggestion(
                    file_path=file_path,
                    start_line=func_start,
                    end_line=func_end,
                    original_code=original,
                    current_code=current,
                    issue_type="runtime",
                    issue_detail=error_msg[:200],
                    confidence=0.75,
                ))

        return suggestions

    def _get_diff_lines(self, current: str, backup: str) -> List[Tuple[int, int]]:
        """获取修改的行范围列表"""
        diff = difflib.unified_diff(
            backup.splitlines(),
            current.splitlines(),
            lineterm="",
        )

        regions = []
        for line in diff:
            if line.startswith("@@"):
                # 解析 @@ -a,b +c,d @@
                match = re.search(r'\+(\d+),?(\d+)?', line)
                if match:
                    start = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    regions.append((start, start + count - 1))

        return regions

    def _find_affected_region(
        self,
        diff_regions: List[Tuple[int, int]],
        error_line: int,
        content: str,
    ) -> Optional[Tuple[int, int]]:
        """找到包含错误行的修改区域"""
        for start, end in diff_regions:
            if start <= error_line <= end:
                return (start, end)

        # 如果错误行不在修改区域，找最近的修改块
        for start, end in diff_regions:
            if error_line < start:
                return (start, end)
            if error_line > end:
                continue

        # 如果没有任何修改区域，回滚整个文件
        lines = content.splitlines()
        return (1, len(lines))

    def _find_region_by_line(
        self,
        diff_regions: List[Tuple[int, int]],
        target_line: int,
    ) -> Optional[Tuple[int, int]]:
        """找到包含目标行的区域"""
        for start, end in diff_regions:
            if start <= target_line <= end:
                return (start, end)

        # 找最近的
        for start, end in diff_regions:
            if target_line < start:
                return (start, end)

        return None

    def _extract_lines(self, content: str, region: Tuple[int, int]) -> str:
        """提取指定行范围的内容"""
        lines = content.splitlines()
        start, end = region
        if start < 1:
            start = 1
        if end > len(lines):
            end = len(lines)
        return "\n".join(lines[start-1:end])


# 全局单例
_rollback_engine: Optional[FineGrainedRollback] = None


def get_rollback_engine(code_manager) -> FineGrainedRollback:
    """获取回滚引擎实例"""
    global _rollback_engine
    if _rollback_engine is None:
        _rollback_engine = FineGrainedRollback(code_manager)
    return _rollback_engine