"""
代码质量评估标准 - 这是系统的"真值锚点"
只有人类手动修改，或在特定条件下（如连续5轮改善）才允许AI提议修改此文件
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import ast
import re


@dataclass
class QualityMetric:
    """单个质量指标的定义"""
    name: str
    description: str
    weight: float  # 0-1 之间的权重
    threshold_good: float  # 达到此值算"良好"
    threshold_excellent: float  # 达到此值算"优秀"
    current_value: Optional[float] = None
    previous_value: Optional[float] = None


@dataclass
class QualityStandards:
    """好代码的量化标准"""
    
    # 圈复杂度阈值（McCabe）
    cyclomatic_complexity: QualityMetric = field(default_factory=lambda: QualityMetric(
        name="cyclomatic_complexity",
        description="圈复杂度，衡量代码逻辑复杂程度。越低越好，通常 < 10 为良好",
        weight=0.25,
        threshold_good=10,
        threshold_excellent=5,
    ))
    
    # 函数最大行数
    function_max_lines: QualityMetric = field(default_factory=lambda: QualityMetric(
        name="function_max_lines",
        description="单个函数的最大行数。越短越易读，通常 < 50 行为良好",
        weight=0.15,
        threshold_good=50,
        threshold_excellent=30,
    ))
    
    # 类型注解覆盖率
    type_annotation_coverage: QualityMetric = field(default_factory=lambda: QualityMetric(
        name="type_annotation_coverage",
        description="函数参数和返回值有类型注解的比例。越高越好，> 80% 为优秀",
        weight=0.15,
        threshold_good=0.60,
        threshold_excellent=0.80,
    ))
    
    # 文档字符串覆盖率
    docstring_coverage: QualityMetric = field(default_factory=lambda: QualityMetric(
        name="docstring_coverage",
        description="有文档字符串的函数/类比例。越高越易维护，> 70% 为优秀",
        weight=0.10,
        threshold_good=0.50,
        threshold_excellent=0.70,
    ))
    
    # 重复代码块数量
    duplicate_blocks: QualityMetric = field(default_factory=lambda: QualityMetric(
        name="duplicate_blocks",
        description="重复代码块的数量。越少越好，0 为理想",
        weight=0.10,
        threshold_good=3,
        threshold_excellent=0,
    ))
    
    # 导入数量（太多可能说明职责不单一）
    import_count: QualityMetric = field(default_factory=lambda: QualityMetric(
        name="import_count",
        description="单个文件的导入数量。太多可能意味着职责不单一",
        weight=0.05,
        threshold_good=15,
        threshold_excellent=8,
    ))
    
    # 异常处理覆盖率
    exception_handling: QualityMetric = field(default_factory=lambda: QualityMetric(
        name="exception_handling",
        description="使用 try/except/with 处理异常的覆盖度。越高越健壮",
        weight=0.10,
        threshold_good=0.60,
        threshold_excellent=0.85,
    ))
    
    # 类/函数数量（模块内聚性）
    module_cohesion: QualityMetric = field(default_factory=lambda: QualityMetric(
        name="module_cohesion",
        description="模块内类和函数的关联度。基于文件名和内容的语义匹配",
        weight=0.10,
        threshold_good=0.60,
        threshold_excellent=0.80,
    ))
    
    def get_dimensions(self) -> List[QualityMetric]:
        """返回所有维度，按权重排序"""
        return [
            self.cyclomatic_complexity,
            self.function_max_lines,
            self.type_annotation_coverage,
            self.docstring_coverage,
            self.duplicate_blocks,
            self.import_count,
            self.exception_handling,
            self.module_cohesion,
        ]
    
    def get_weighted_score(self, values: Dict[str, float]) -> float:
        """计算加权总分"""
        total_weight = 0
        weighted_sum = 0
        
        for metric in self.get_dimensions():
            value = values.get(metric.name, 0)
            # 某些指标越低越好，需要反转
            if metric.name in ["cyclomatic_complexity", "function_max_lines", "duplicate_blocks", "import_count"]:
                # 归一化：值越小得分越高（用阈值做分母）
                max_val = metric.threshold_good
                if max_val > 0:
                    normalized = max(0, 1 - (value / max_val))
                else:
                    normalized = 1 if value == 0 else 0
            else:
                # 其他指标越高越好
                normalized = min(1, value / metric.threshold_excellent if metric.threshold_excellent > 0 else value)
            
            weighted_sum += normalized * metric.weight
            total_weight += metric.weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0


class CodeQualityAnalyzer:
    """分析代码质量并评分"""
    
    def __init__(self, standards: Optional[QualityStandards] = None):
        self.standards = standards or QualityStandards()
    
    def analyze_file(self, file_path: str, content: str) -> Dict[str, float]:
        """分析单个文件的所有质量维度"""
        result = {}
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {k.name: 0 for k in self.standards.get_dimensions()}
        
        # 1. 圈复杂度
        result["cyclomatic_complexity"] = self._calc_cyclomatic_complexity(tree)
        
        # 2. 函数最大行数
        result["function_max_lines"] = self._calc_max_function_lines(tree)
        
        # 3. 类型注解覆盖率
        result["type_annotation_coverage"] = self._calc_type_annotation_coverage(tree)
        
        # 4. 文档字符串覆盖率
        result["docstring_coverage"] = self._calc_docstring_coverage(tree)
        
        # 5. 重复代码块（简化版，实际可用更复杂的算法）
        result["duplicate_blocks"] = self._calc_duplicate_blocks(content)
        
        # 6. 导入数量
        result["import_count"] = self._calc_import_count(tree)
        
        # 7. 异常处理覆盖率
        result["exception_handling"] = self._calc_exception_handling(tree)
        
        # 8. 模块内聚性（基于文件名与内容的语义匹配）
        result["module_cohesion"] = self._calc_module_cohesion(file_path, tree)
        
        return result
    
    def _calc_cyclomatic_complexity(self, tree: ast.AST) -> float:
        """计算平均圈复杂度（简化版）"""
        count_conditions = 0
        count_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                count_functions += 1
                # 在函数内统计条件分支
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.If, ast.While, ast.For, ast.And, ast.Or, ast.ExceptHandler)):
                        count_conditions += 1
        
        if count_functions == 0:
            return 1
        return count_conditions / count_functions + 1  # McCabe: 条件数 + 1
    
    def _calc_max_function_lines(self, tree: ast.AST) -> int:
        """计算所有函数中的最大行数"""
        max_lines = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                end_line = getattr(node, 'end_lineno', node.lineno)
                lines = end_line - node.lineno + 1
                max_lines = max(max_lines, lines)
        
        return max_lines
    
    def _calc_type_annotation_coverage(self, tree: ast.AST) -> float:
        """计算类型注解覆盖率"""
        total_funcs = 0
        annotated = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_funcs += 1
                # 检查返回值注解
                if node.returns:
                    annotated += 1
                    continue
                # 检查参数注解
                for arg in node.args.args:
                    if arg.annotation:
                        annotated += 1
                        break
        
        if total_funcs == 0:
            return 1.0
        
        # 函数级别覆盖率（只要有一个参数或返回值有注解就算覆盖）
        # 更精确的版本可以计算参数级别覆盖率
        return annotated / total_funcs
    
    def _calc_docstring_coverage(self, tree: ast.AST) -> float:
        """计算文档字符串覆盖率"""
        total_defs = 0
        with_docstring = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                total_defs += 1
                if ast.get_docstring(node):
                    with_docstring += 1
        
        if total_defs == 0:
            return 1.0
        
        return with_docstring / total_defs
    
    def _calc_duplicate_blocks(self, content: str) -> int:
        """简化版重复代码检测（按行匹配）"""
        lines = content.splitlines()
        line_patterns = {}
        duplicates = 0
        
        # 只检测 >5 行的重复块（简化）
        for i in range(len(lines) - 5):
            block = "\n".join(lines[i:i+5])
            if len(block.strip()) > 20:  # 忽略空行
                line_patterns[block] = line_patterns.get(block, 0) + 1
        
        duplicates = sum(1 for count in line_patterns.values() if count > 1)
        return duplicates
    
    def _calc_import_count(self, tree: ast.AST) -> int:
        """计算导入语句数量"""
        imports = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports += 1
        return imports
    
    def _calc_exception_handling(self, tree: ast.AST) -> float:
        """计算异常处理覆盖率（try/except/with 覆盖比例）"""
        total_statements = 0
        handled_statements = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # 检查函数体是否有异常处理
                body = getattr(node, 'body', [])
                total_statements += len(body) if body else 1
                
                for stmt in body:
                    if isinstance(stmt, (ast.Try, ast.ExceptHandler, ast.With)):
                        handled_statements += 1
        
        if total_statements == 0:
            return 1.0
        
        return handled_statements / total_statements
    
    def _calc_module_cohesion(self, file_path: str, tree: ast.AST) -> float:
        """模块内聚性（基于文件名和内容的主题匹配）"""
        # 简化版：统计有多少定义的名称与文件名相关
        import re
        from pathlib import Path
        
        file_name = Path(file_path).stem
        # 提取文件名中的关键词
        keywords = set(re.findall(r'[a-z]+', file_name.lower()))
        
        if not keywords:
            return 1.0
        
        # 获取所有定义的名称
        defined_names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                defined_names.add(node.name.lower())
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id.lower())
        
        if not defined_names:
            return 0.5
        
        # 计算名称中与文件名关键词匹配的比例
        matches = sum(1 for name in defined_names if any(kw in name for kw in keywords))
        
        return matches / len(defined_names) if defined_names else 0.5


# 全局单例
_global_standards: Optional[QualityStandards] = None
_global_analyzer: Optional[CodeQualityAnalyzer] = None


def get_quality_standards() -> QualityStandards:
    global _global_standards
    if _global_standards is None:
        _global_standards = QualityStandards()
    return _global_standards


def get_quality_analyzer() -> CodeQualityAnalyzer:
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = CodeQualityAnalyzer(get_quality_standards())
    return _global_analyzer