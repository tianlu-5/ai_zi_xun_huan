"""
测试生成器 - 为修改后的代码自动生成单元测试

核心功能：
1. 根据修改内容分析需要测试的函数/方法
2. 调用 LLM 生成对应的测试用例
3. 将测试用例写入临时文件并执行
4. 返回测试结果
"""

import json
import subprocess
import tempfile
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from config import Config


class TestGenerator:
    """自动生成并执行单元测试"""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self._test_dir = Config.LOG_DIR / "test_cache"
        self._test_dir.mkdir(parents=True, exist_ok=True)

    def generate_tests(
        self,
        file_path: str,
        new_content: str,
        old_content: str,
        objective: str = "",
    ) -> Dict[str, Any]:
        """
        为修改后的代码生成测试
        返回: {
            "generated": bool,
            "tests": str,          # 生成的测试代码
            "test_file": str,      # 测试文件路径
            "result": dict         # 测试执行结果
        }
        """
        # 1. 分析修改，确定需要测试的函数
        changed_functions = self._analyze_changes(old_content, new_content)
        if not changed_functions:
            return {
                "generated": False,
                "reason": "未检测到可测试的函数变更",
                "tests": "",
                "test_file": "",
                "result": {},
            }

        # 2. 提取要测试的函数代码
        functions_to_test = self._extract_functions(new_content, changed_functions)

        # 3. 调用 LLM 生成测试
        test_code = self._call_llm_generate_tests(
            file_path=file_path,
            functions=functions_to_test,
            objective=objective,
        )

        if not test_code:
            return {
                "generated": False,
                "reason": "LLM 测试生成失败",
                "tests": "",
                "test_file": "",
                "result": {},
            }

        # 4. 执行测试
        result = self._run_tests(test_code, file_path)

        return {
            "generated": True,
            "tests": test_code,
            "test_file": result.get("test_file", ""),
            "result": result,
        }

    def _analyze_changes(self, old_content: str, new_content: str) -> List[str]:
        """分析新旧代码差异，找出变更的函数名"""
        def extract_funcs(content: str) -> set:
            names = set()
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.add(node.name)
            except Exception:
                pass
            return names

        old_funcs = extract_funcs(old_content)
        new_funcs = extract_funcs(new_content)

        # 新增或修改的函数
        changed = list(new_funcs - old_funcs)
        # 也可能有函数体变更但名没变，需要更细粒度检测
        # 这里先用简单方法：找名没变但内容可能变的（通过行数变化）
        common = new_funcs & old_funcs
        for name in common:
            # 简单检测：如果函数行数变化超过3行，认为有修改
            old_lines = self._extract_function_lines(old_content, name)
            new_lines = self._extract_function_lines(new_content, name)
            if old_lines and new_lines and abs(len(old_lines) - len(new_lines)) > 3:
                changed.append(name)

        return changed[:5]  # 最多测试5个函数

    def _extract_function_lines(self, content: str, func_name: str) -> List[str]:
        """提取函数的代码行"""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == func_name:
                        start = node.lineno - 1
                        end = getattr(node, 'end_lineno', node.lineno + 10)
                        lines = content.splitlines()
                        return lines[start:min(end, len(lines))]
            return []
        except Exception:
            return []

    def _extract_functions(self, content: str, func_names: List[str]) -> Dict[str, str]:
        """提取函数的完整代码"""
        result = {}
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in func_names:
                        start = node.lineno
                        end = getattr(node, 'end_lineno', node.lineno + 20)
                        lines = content.splitlines()
                        func_code = "\n".join(lines[start-1:min(end, len(lines))])
                        result[node.name] = func_code
        except Exception:
            pass
        return result

    def _call_llm_generate_tests(
        self,
        file_path: str,
        functions: Dict[str, str],
        objective: str,
    ) -> str:
        """调用 LLM 生成测试代码"""
        if not self.llm:
            return ""

        # 构建提示词
        func_text = "\n\n".join([
            f"### {name}\n```python\n{code}\n```"
            for name, code in functions.items()
        ])

        system_prompt = """你是 Python 单元测试专家。为给定的函数生成 pytest 风格的单元测试。

要求：
1. 覆盖正常情况、边界情况和异常情况
2. 使用 pytest 的 assert 语句
3. 测试文件必须包含 import 语句
4. 只输出测试代码，不要输出其他文字
5. 所有测试函数以 test_ 开头
6. 如果函数需要实例化类，提供必要的 mock 或 fixture

请生成完整可运行的测试代码。"""

        user_prompt = f"""
目标文件: {file_path}
本轮目标: {objective or "代码修改后验证"}

需要测试的函数:
{func_text}

请生成这些函数的单元测试代码。
"""

        try:
            response = self.llm.chat(
                user_message=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                force_json=False,  # 测试代码不是 JSON
            )
            return response or ""
        except Exception as e:
            print(f"  ⚠️ [TestGen] LLM 调用失败: {e}")
            return ""

    def _run_tests(self, test_code: str, source_file: str) -> Dict[str, Any]:
        """执行测试代码并返回结果"""
        try:
            # 将测试代码写入临时文件
            test_file = self._test_dir / f"test_{Path(source_file).stem}_{datetime.now().strftime('%H%M%S')}.py"

            # 添加必要的 import
            full_test_code = f"""
import sys
import os
sys.path.insert(0, r"{Config.PROJECT_ROOT}")
sys.path.insert(0, r"{Config.PROJECT_ROOT / 'evolver'}")

import pytest
import json

{test_code}
"""

            test_file.write_text(full_test_code, encoding="utf-8")

            # 使用 pytest 运行
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", "--maxfail=3"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(Config.PROJECT_ROOT),
            )

            passed = "passed" in result.stdout or result.returncode == 0
            return {
                "passed": passed,
                "stdout": result.stdout[:2000] if result.stdout else "",
                "stderr": result.stderr[:2000] if result.stderr else "",
                "returncode": result.returncode,
                "test_file": str(test_file),
            }

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "error": "测试执行超时（>30秒）",
                "test_file": "",
            }
        except Exception as e:
            return {
                "passed": False,
                "error": f"测试执行异常: {e}",
                "test_file": "",
            }


# 全局单例
_test_generator: Optional[TestGenerator] = None


def get_test_generator() -> TestGenerator:
    global _test_generator
    if _test_generator is None:
        _test_generator = TestGenerator()
    return _test_generator