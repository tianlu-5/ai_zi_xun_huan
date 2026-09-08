# -*- coding: utf-8 -*-
"""
能力注册表 + 渐进式合约体系

设计目标：
  1. 模块声明自己能做什么（Capability），合约检查声明而非硬编码方法名
  2. 合约分 4 档（ContractTier），失败时按档差异化处理
  3. 缺方法时自动生成 stub，不让系统卡住

使用方式：
  from capability_registry import CapabilityRegistry, ContractTier

  # 注册能力
  reg = CapabilityRegistry()
  reg.register("branch_manager", [
      Capability(name="list_all_branches", return_type=list, fallback="return []"),
      Capability(name="create_branch", min_params=1, fallback="return {}"),
  ])

  # 验证
  result = reg.verify("branch_manager")

  # 自动 stub
  reg.auto_stub_missing("branch_manager")
"""
from __future__ import annotations

import ast
import os
import sys
import inspect
import traceback
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Stub 标记装饰器 ──

def auto_stub(fn):
    """
    标记该方法为自动生成的 stub（待 AI 补全）。
    合约检查时含此标记的方法计为 'stub_pending' 而非 'passed'。
    """
    fn._is_auto_stub = True
    return fn


def is_auto_stub(method) -> bool:
    """检查方法是否为 auto_stub 生成"""
    return getattr(method, '_is_auto_stub', False) or (
        inspect.isfunction(method) and 'AUTO-GENERATED STUB' in (inspect.getdoc(method) or '')
    )


# ── 渐进式合约分档 ──

class ContractTier(IntEnum):
    """合约严重度分档，数值越小越严重"""
    CRITICAL = 1     # 如：不能删 main.py、不能改 kill-switch → 立即 kill-switch
    STRUCTURAL = 2   # 如：BranchManager 必须有 list/create → 自动 stub 修复，不拉黑
    BEHAVIORAL = 3   # 如：create_branch 必须幂等 → 允许 3 次重试，带 prompt 修正
    PERFORMANCE = 4  # 如：ctx 必须 ≥ 16K → 仅警告，不阻断


# ── 能力声明 ──

@dataclass
class Capability:
    """单个方法/能力声明"""
    name: str                                    # 方法名
    min_params: int = 0                          # self 之外最少参数数
    return_type: type = None                     # 期望返回类型（None=不检查）
    return_schema: Optional[Dict[str, type]] = None  # 返回 dict 时的字段约束
    idempotent: bool = False                     # 是否要求幂等
    fallback: str = "pass"                       # stub 生成时的兜底返回语句
    auto_stub: bool = True                       # 缺失时是否允许自动生成 stub


# ── 能力注册表 ──

class CapabilityRegistry:
    """能力注册表：模块声明能力，合约检查声明而非硬编码"""

    # 默认能力模板：常见模块的基础能力
    _DEFAULTS: Dict[str, List[Capability]] = {
        "branch_manager": [
            Capability(name="list_all_branches", return_type=list, fallback="return []"),
            Capability(name="create_branch", min_params=1, fallback="return {}"),
        ],
    }

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self._registry: Dict[str, List[Capability]] = {}
        # 加载默认模板
        for mod, caps in self._DEFAULTS.items():
            self._registry[mod] = list(caps)

    def register(self, module_name: str, capabilities: List[Capability]) -> None:
        """注册模块的能力声明"""
        existing = self._registry.get(module_name, [])
        existing.extend(capabilities)
        self._registry[module_name] = existing

    def get_capabilities(self, module_name: str) -> List[Capability]:
        return self._registry.get(module_name, [])

    def verify(self, module_name: str) -> Tuple[bool, List[str]]:
        """
        检查模块实际实现是否覆盖注册的能力。
        :return: (all_ok, missing_list)
        """
        caps = self._registry.get(module_name, [])
        if not caps:
            return True, []

        try:
            mod = __import__(module_name)
        except ImportError:
            return False, [f"模块 {module_name} 无法导入"]

        missing = []
        for cap in caps:
            cls_name = self._find_class_with_method(mod, cap.name)
            if cls_name is None:
                # 也检查模块级函数
                if hasattr(mod, cap.name):
                    continue
                missing.append(cap.name)
                continue

            cls = getattr(mod, cls_name)
            method = getattr(cls, cap.name, None)
            if method is None:
                missing.append(cap.name)
                continue

            # 检查参数数量
            if cap.min_params > 0:
                try:
                    sig = inspect.signature(method)
                    non_self = len(sig.parameters)
                    if non_self < cap.min_params:
                        missing.append(f"{cap.name}(参数不足: {non_self}<{cap.min_params})")
                except (ValueError, TypeError):
                    pass

        return len(missing) == 0, missing

    @staticmethod
    def _find_class_with_method(module, method_name: str) -> Optional[str]:
        """在模块中找到包含指定方法的类名"""
        for name in dir(module):
            obj = getattr(module, name, None)
            if inspect.isclass(obj) and hasattr(obj, method_name):
                # 确保方法定义在类自身而非继承
                if method_name in obj.__dict__ or any(
                    method_name in base.__dict__ for base in obj.__mro__
                ):
                    return name
        return None

    def auto_stub_missing(self, module_name: str) -> Tuple[int, List[str]]:
        """
        自动为缺失的方法生成 stub，追加到模块文件末尾。
        :return: (stubbed_count, stubbed_method_names)
        """
        caps = self._registry.get(module_name, [])
        if not caps:
            return 0, []

        # 找模块文件路径
        try:
            mod = __import__(module_name)
            mod_file = inspect.getfile(mod)
        except (ImportError, TypeError):
            mod_file = str(self.project_root / "evolver" / f"{module_name}.py")
        mod_path = Path(mod_file)
        if not mod_path.exists():
            return 0, [f"模块文件不存在: {mod_path}"]

        # 读取现有代码
        try:
            source = mod_path.read_text(encoding="utf-8")
        except Exception as e:
            return 0, [f"读取文件失败: {e}"]

        # 用 AST 解析现有方法
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return 0, [f"语法错误无法解析: {e}"]

        existing_methods = set()
        main_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if main_class is None:
                    main_class = node.name
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        existing_methods.add(item.name)

        if main_class is None:
            return 0, ["未找到类定义"]

        # 找缺失方法
        to_stub = []
        for cap in caps:
            if cap.name not in existing_methods and cap.auto_stub:
                to_stub.append(cap)

        if not to_stub:
            return 0, []

        # 生成 stub 代码（不依赖装饰器，靠 docstring 标记识别）
        stub_lines = []
        for cap in to_stub:
            params = "self"
            if cap.min_params > 0:
                params += ", *args, **kwargs"
            stub_lines.append("")
            stub_lines.append(f"    def {cap.name}({params}):")
            stub_lines.append(f'        """AUTO-GENERATED STUB: TODO AI 进化时优先实现此方法"""')
            stub_lines.append(f"        {cap.fallback}")

        stub_code = "\n".join(stub_lines)

        # 追加到文件末尾（注意缩进：在类内部）
        # 找到最后一个类定义的末尾位置
        class_end_line = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == main_class:
                class_end_line = max(class_end_line, node.end_lineno or 0)

        if class_end_line == 0:
            return 0, ["无法定位类末尾"]

        lines = source.splitlines(keepends=True)
        # 在类末尾后插入（需要确保缩进正确）
        insert_idx = class_end_line
        # 插入 stub
        new_lines = list(lines)
        stub_with_newline = stub_code + "\n"
        new_lines.insert(insert_idx, stub_with_newline)

        new_source = "".join(new_lines)

        # 验证语法
        try:
            ast.parse(new_source)
        except SyntaxError as e:
            return 0, [f"stub 生成后语法错误: {e}"]

        # 写入文件
        try:
            mod_path.write_text(new_source, encoding="utf-8")
        except Exception as e:
            return 0, [f"写入文件失败: {e}"]

        stubbed_names = [cap.name for cap in to_stub]
        print(f"  🔧 [auto-stub] 为 {module_name} 生成 {len(to_stub)} 个 stub: {stubbed_names}")
        return len(to_stub), stubbed_names


# ── 便捷函数 ──

def get_tier_for_contract(module: str, name: str) -> ContractTier:
    """
    根据合约模块和名称推断 tier（用于合约注册时的默认分档）。
    匹配优先级：CRITICAL > BEHAVIORAL（行为关键词）> STRUCTURAL（方法名关键词）
    """
    # CRITICAL：main.py 相关
    if "main" in module:
        return ContractTier.CRITICAL

    # BEHAVIORAL 优先：行为类关键词（幂等/返回/异常/不越界/不炸/形状）
    #   即使合约名含方法名，只要也含行为关键词就归 BEHAVIORAL
    behavioral_keywords = ("幂等", "返回", "异常", "不越界", "不炸", "形状",
                           "idempotent", "return", "boundary", "shape",
                           "空操作", "不触发", "绝不", "不抛")
    if any(kw in name for kw in behavioral_keywords):
        return ContractTier.BEHAVIORAL

    # STRUCTURAL：方法必须存在（list/create/run/detect/decouple/mine/snapshot）
    structural_keywords = ("list_all_branches", "create_branch", "run_full_inspection",
                           "compute_entropy_gap", "check_time_drift", "snapshot",
                           "register_branch", "decouple", "mine_simple", "detect")
    if any(kw in name for kw in structural_keywords):
        return ContractTier.STRUCTURAL

    # 默认 BEHAVIORAL
    return ContractTier.BEHAVIORAL
