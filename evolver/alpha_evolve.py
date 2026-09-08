"""
AlphaEvolve 模式 (DeepMind 2025 风格)

进化闭环：
1. 生成 N 个候选变体（不同温度的 LLM 调用）
2. 用评估函数对每个变体打分
3. 保留最好的 K 个
4. 在 K 个基础上变异/交叉
5. 循环

评估维度（从高到低权重）：
- 致命错误（语法/运行时崩溃）→ 直接淘汰
- 幻觉方法调用（调用不存在的方法）→ 重罚
- 净功能改进（新增函数数 > 0）→ 奖励
- 代码质量（不可达代码/重复方法/先用后定义）→ 扣分
- 新函数有调用者 → 奖励
"""
import ast
import os
import re
import time
import json
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from config import Config


@dataclass
class Variant:
    """一个候选变体"""
    variant_id: int
    source: str = ""              # 补丁后的完整源码
    patch_raw: str = ""           # LLM 原始输出
    score: float = 0.0
    feedback: List[str] = field(default_factory=list)
    eliminated: bool = False      # 是否被淘汰
    generation: int = 0           # 第几代


class Evaluator:
    """评估函数：对补丁后的源码打分"""

    # 评分常量
    SCORE_SYNTAX_OK = 10
    SCORE_AST_OK = 10
    SCORE_RUNTIME_OK = 15
    SCORE_NO_HALLUCINATION = 20
    SCORE_NET_NEW_FUNC = 25
    SCORE_HAS_CALLERS = 15
    SCORE_NO_DEAD_CODE = 5

    PENALTY_SYNTAX = -100
    PENALTY_RUNTIME = -60
    PENALTY_HALLUCINATION = -30
    PENALTY_NO_NEW_FUNC = -10
    PENALTY_DEAD_CODE = -5
    PENALTY_DUPLICATE_METHOD = -15

    # 賻统路由方法豁免清单（这些方法是系统要求 AI 添加的，无调用者不算违规）
    EXEMPT_METHODS = {
        "cleanup", "get_stats", "validate", "to_dict",
        "get_module_info", "reset",
    }

    @classmethod
    def evaluate(
        cls,
        original_source: str,
        patched_source: str,
        target_file: str,
        variant_id: int = 0,
    ) -> Tuple[float, List[str]]:
        """评估一个变体，返回 (分数, 反馈列表)"""
        score = 0.0
        feedback = []

        # ── 1. 语法检查 ──
        try:
            compile(patched_source, target_file, 'exec')
            score += cls.SCORE_SYNTAX_OK
        except SyntaxError as e:
            return cls.PENALTY_SYNTAX, [f"语法错误: 行{e.lineno}: {e.msg}"]

        # ── 2. AST 质量检查 ──
        try:
            tree = ast.parse(patched_source)
        except Exception as e:
            return cls.PENALTY_SYNTAX, [f"AST 解析失败: {e}"]

        ast_issues = cls._check_ast_quality(tree)
        if ast_issues:
            for issue in ast_issues:
                if "重复" in issue:
                    score += cls.PENALTY_DUPLICATE_METHOD
                elif "不可达" in issue:
                    score += cls.PENALTY_DEAD_CODE
                elif "先用后定义" in issue:
                    score += cls.PENALTY_DEAD_CODE
                feedback.append(issue)
        else:
            score += cls.SCORE_AST_OK

        # ── 3. 运行时安全（exec + 实例化）──
        runtime_ok = True
        try:
            ns = {}
            exec(patched_source, ns)
            for name, obj in ns.items():
                if isinstance(obj, type) and obj.__module__ == '<string>':
                    try:
                        obj()
                    except TypeError:
                        pass  # 需要参数，可接受
                    except Exception as e:
                        runtime_ok = False
                        feedback.append(f"实例化 {name}() 失败: {type(e).__name__}: {e}")
                        break
        except Exception as e:
            runtime_ok = False
            feedback.append(f"运行时错误: {type(e).__name__}: {e}")

        if runtime_ok:
            score += cls.SCORE_RUNTIME_OK
        else:
            score += cls.PENALTY_RUNTIME

        # 如果已经致命失败，直接返回
        if score < 0:
            return score, feedback

        # ── 4. 幻觉方法检测 ──
        hallucinated = cls._check_hallucinated_calls(
            original_source, patched_source, target_file
        )
        if hallucinated:
            score += cls.PENALTY_HALLUCINATION
            for h in hallucinated:
                feedback.append(f"幻觉方法: self.{h}() 不存在")
        else:
            score += cls.SCORE_NO_HALLUCINATION

        # ── 5. 净功能改进 ──
        new_funcs = cls._find_new_functions(original_source, patched_source)
        if new_funcs:
            score += cls.SCORE_NET_NEW_FUNC
            feedback.append(f"新增函数: {new_funcs}")
        else:
            # 没有新函数 = 只是移动代码
            score += cls.PENALTY_NO_NEW_FUNC
            feedback.append("无新函数（仅移动/修改现有代码）")

        # ── 6. 新函数调用者检查 ──
        if new_funcs:
            uncalled = cls._check_callers_in_project(new_funcs, target_file)
            # 豁免路由方法
            real_uncalled = uncalled - cls.EXEMPT_METHODS
            exempted = uncalled & cls.EXEMPT_METHODS
            if exempted:
                feedback.append(f"路由豁免（系统要求添加）: {exempted}")
            if real_uncalled:
                score += cls.PENALTY_NO_NEW_FUNC  # 轻微扣分
                feedback.append(f"无调用者: {real_uncalled}")
            else:
                score += cls.SCORE_HAS_CALLERS
        else:
            score += cls.SCORE_HAS_CALLERS  # 没有新函数，不扣分

        return score, feedback

    @staticmethod
    def _check_ast_quality(tree: ast.AST) -> List[str]:
        """AST 质量检查：重复方法、不可达代码、先用后定义"""
        issues = []

        # 重复方法检测（同一个类内同名方法）
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            seen = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name in seen:
                        issues.append(
                            f"重复方法: {node.name}.{item.name} "
                            f"(行{seen[item.name]} 和 行{item.lineno})"
                        )
                    else:
                        seen[item.name] = item.lineno

        # 不可达代码
        for node in ast.walk(tree):
            for battr in ('body', 'orelse', 'finalbody'):
                body = getattr(node, battr, None)
                if not isinstance(body, list) or len(body) < 2:
                    continue
                for i, stmt in enumerate(body[:-1]):
                    if isinstance(stmt, (ast.Return, ast.Break, ast.Continue, ast.Raise)):
                        next_stmt = body[i + 1]
                        issues.append(
                            f"不可达代码: 行{next_stmt.lineno} "
                            f"在 {type(stmt).__name__} (行{stmt.lineno}) 之后"
                        )

        # 先用后定义（仅检查 __init__）
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != '__init__':
                continue
            assigned = {}
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if (isinstance(t, ast.Attribute)
                                and isinstance(t.value, ast.Name)
                                and t.value.id == 'self'):
                            if t.attr not in assigned:
                                assigned[t.attr] = sub.lineno
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == 'self'):
                    attr = sub.attr
                    if attr in assigned and sub.lineno < assigned[attr]:
                        issues.append(
                            f"先用后定义: self.{attr} 行{sub.lineno} "
                            f"先于赋值行{assigned[attr]}"
                        )

        return issues

    @staticmethod
    def _find_new_functions(original: str, patched: str) -> set:
        """对比原始和补丁后源码，找出新增的函数名"""
        def _collect(source):
            names = set()
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.add(node.name)
            except Exception:
                pass
            return names

        orig = _collect(original)
        new = _collect(patched)
        return {n for n in (new - orig) if not n.startswith('_')}

    @staticmethod
    def _check_hallucinated_calls(
        original_source: str, patched_source: str, target_file: str
    ) -> list:
        """检测补丁新增的代码是否调用了不存在的方法

        逻辑：
        1. 找出补丁新增的 self.xxx() 调用
        2. 检查 xxx 是否在原始源码中已定义
        3. 检查 xxx 是否在补丁后源码中已定义
        4. 都没有 → 幻觉方法
        """
        try:
            orig_tree = ast.parse(original_source)
            patched_tree = ast.parse(patched_source)
        except Exception:
            return []

        # 收集原始源码中所有方法定义
        orig_methods = set()
        for node in ast.walk(orig_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                orig_methods.add(node.name)

        # 收集补丁后源码中所有方法定义
        patched_methods = set()
        for node in ast.walk(patched_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                patched_methods.add(node.name)

        # 找出补丁新增的 self.xxx() 调用（必须是方法调用，不是属性访问）
        new_calls = set()
        for node in ast.walk(patched_tree):
            # 只检测 Call 节点中的 self.xxx()，不检测普通属性访问 self.xxx
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'self'):
                method = node.func.attr
                # 只检查原始源码中不存在的方法
                if method not in orig_methods:
                    new_calls.add(method)

        # 检查这些调用是否在补丁后源码中有定义
        hallucinated = []
        for method in new_calls:
            # 排除 dunder 和常见内置
            if method.startswith('_'):
                continue
            # 排除常见 Python 内置/标准库方法
            if method in ('get', 'append', 'extend', 'items', 'keys', 'values',
                          'format', 'split', 'join', 'strip', 'replace',
                          'lower', 'upper', 'startswith', 'endswith',
                          'pop', 'copy', 'update', 'setdefault'):
                continue
            # 排除已在补丁中定义的方法
            if method in patched_methods:
                continue
            # 排除可能是属性而非方法的
            hallucinated.append(method)

        return hallucinated

    @staticmethod
    def _check_callers_in_project(func_names: set, exclude_file: str) -> set:
        """扫描全项目，返回无调用者的函数名集合"""
        project_root = Path(Config.PROJECT_ROOT)
        skip_dirs = {'backups', '__pycache__', '.git', 'logs', 'evolver/branches'}

        patterns = {}
        for name in func_names:
            patterns[name] = re.compile(r'\b' + re.escape(name) + r'\s*\(')

        callers_found = {name: False for name in func_names}

        for py_file in project_root.rglob('*.py'):
            rel = str(py_file.relative_to(project_root)).replace('\\', '/')
            if any(rel.startswith(d) for d in skip_dirs):
                continue
            if rel.startswith('_test') or rel.startswith('_'):
                continue
            if rel == exclude_file:
                continue

            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue

            for name, pattern in patterns.items():
                if callers_found[name]:
                    continue
                for match in pattern.finditer(content):
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    prefix = content[line_start:match.start()].strip()
                    if prefix.startswith('def '):
                        continue
                    callers_found[name] = True
                    break

        return {name for name, found in callers_found.items() if not found}


class PopulationManager:
    """AlphaEvolve 种群管理器

    生成 N 个变体 → 评估 → 保留最优 K 个 → 在最优基础上变异
    """

    def __init__(
        self,
        llm_client,
        code_manager,
        parser,
        checkpoint_dir: Path,
        population_size: int = 3,
        elite_size: int = 1,
        max_generations: int = 2,
    ):
        self.llm = llm_client
        self.cm = code_manager
        self.parser = parser
        self.ckpt_dir = checkpoint_dir
        self.pop_size = population_size
        self.elite_size = elite_size
        self.max_gens = max_generations

        # 不同温度策略：第一代用不同温度生成多样性
        self.temp_schedule = [0.1, 0.3, 0.5]

    def evolve(
        self,
        objective: str,
        target_file: str,
        source_snapshot: Dict[str, str],
        goal: str,
        iteration_id: int,
        system_prompt: str = "",
        user_prompt_template: str = "",
    ) -> Tuple[Optional[Variant], List[Variant]]:
        """执行一轮进化

        返回: (最优变体, 所有变体列表)
        """
        original_source = source_snapshot.get(target_file, "")
        if not original_source:
            return None, []

        all_variants = []
        best_variant = None

        for gen in range(self.max_gens):
            print(f"\n  🧬 AlphaEvolve 第 {gen+1}/{self.max_gens} 代")

            # ── 1. 生成变体 ──
            if gen == 0:
                # 第一代：用不同温度生成
                variants = self._generate_initial(
                    objective, target_file, original_source, goal,
                    system_prompt, user_prompt_template, iteration_id, gen
                )
            else:
                # 后续代：在最优基础上变异
                if best_variant is None:
                    break
                variants = self._generate_mutations(
                    best_variant, objective, target_file, goal,
                    system_prompt, user_prompt_template, iteration_id, gen
                )

            if not variants:
                break

            # ── 2. 评估每个变体 ──
            for v in variants:
                if not v.source:
                    v.eliminated = True
                    v.feedback.append("空源码")
                    continue

                v.score, v.feedback = Evaluator.evaluate(
                    original_source, v.source, target_file, v.variant_id
                )
                print(f"     变体#{v.variant_id} 分数={v.score:.0f} "
                      f"{'❌淘汰' if v.score < 0 else '✅'} "
                      f"{v.feedback[0] if v.feedback else ''}")

            # ── 3. 排序，保留最优 ──
            viable = [v for v in variants if not v.eliminated and v.score >= 0]
            viable.sort(key=lambda x: x.score, reverse=True)

            all_variants.extend(variants)

            if viable:
                best_variant = viable[0]
                print(f"     🏆 最优: 变体#{best_variant.variant_id} "
                      f"分数={best_variant.score:.0f}")
            else:
                print(f"     ⚠️ 本代全部淘汰")
                # 如果第一代全淘汰，尝试用最后一轮的重试
                if gen == 0:
                    continue
                else:
                    break

        # 保存进化记录
        self._save_evolution_log(all_variants, iteration_id, target_file)

        return best_variant, all_variants

    def _generate_initial(
        self, objective, target_file, source, goal,
        system_prompt, user_prompt_template, iteration_id, gen,
    ) -> List[Variant]:
        """第一代：用不同温度生成 N 个变体"""
        variants = []
        for i in range(self.pop_size):
            temp = self.temp_schedule[i] if i < len(self.temp_schedule) else 0.3
            variant_id = gen * 100 + i + 1

            print(f"     🧬 生成变体#{variant_id} (temp={temp})...")

            # 设置温度
            if hasattr(self.llm, 'temperature'):
                self.llm.temperature = temp
            if hasattr(self.llm, 'reset_conversation'):
                self.llm.reset_conversation()

            # 构建 prompt
            if user_prompt_template:
                user_prompt = user_prompt_template.format(
                    goal=goal, source_code=source,
                )
            else:
                user_prompt = f"目标: {goal}\n源码:\n{source}\n生成补丁JSON"

            # 调用 LLM
            try:
                kwargs = dict(
                    user_message=user_prompt,
                    system_prompt=system_prompt,
                    stream=False,
                )
                if hasattr(self.llm, '_backend'):
                    kwargs['force_json'] = True
                raw = self.llm.chat(**kwargs) or ""
            except Exception as e:
                print(f"     ❌ LLM 调用失败: {e}")
                raw = ""

            # 解析补丁并应用
            patched = self._apply_patch_in_memory(raw, source, target_file)

            v = Variant(
                variant_id=variant_id,
                source=patched,
                patch_raw=raw,
                generation=gen,
            )
            variants.append(v)

        return variants

    def _generate_mutations(
        self, best: Variant, objective, target_file, goal,
        system_prompt, user_prompt_template, iteration_id, gen,
    ) -> List[Variant]:
        """后续代：在最优变体基础上变异（用更高温度重新生成）"""
        variants = []
        # 用最优变体作为新基准
        base_source = best.source

        # 生成 2 个变异体
        for i in range(2):
            temp = 0.2 + i * 0.15  # 0.2, 0.35
            variant_id = gen * 100 + i + 1

            print(f"     🧬 变异变体#{variant_id} (temp={temp})...")

            if hasattr(self.llm, 'temperature'):
                self.llm.temperature = temp
            if hasattr(self.llm, 'reset_conversation'):
                self.llm.reset_conversation()

            # 在最优基础上改进
            feedback_str = "\n".join(best.feedback) if best.feedback else "无"
            user_prompt = (
                f"目标: {goal}\n"
                f"当前代码（已改进，分数={best.score:.0f}）:\n{base_source}\n"
                f"上轮反馈: {feedback_str}\n"
                f"请在此基础上进一步改进，生成完整补丁JSON。"
            )

            try:
                kwargs = dict(
                    user_message=user_prompt,
                    system_prompt=system_prompt,
                    stream=False,
                )
                if hasattr(self.llm, '_backend'):
                    kwargs['force_json'] = True
                raw = self.llm.chat(**kwargs) or ""
            except Exception as e:
                print(f"     ❌ LLM 调用失败: {e}")
                raw = ""

            patched = self._apply_patch_in_memory(raw, base_source, target_file)

            v = Variant(
                variant_id=variant_id,
                source=patched,
                patch_raw=raw,
                generation=gen,
            )
            variants.append(v)

        return variants

    def _apply_patch_in_memory(
        self, patch_raw: str, source: str, target_file: str
    ) -> str:
        """在内存中应用补丁，返回补丁后的源码（不写磁盘）"""
        if not patch_raw:
            return ""

        try:
            # 尝试解析 JSON
            if patch_raw.strip().startswith('{'):
                patch_data = json.loads(patch_raw)
            else:
                # 尝试提取 JSON
                match = re.search(r'\{[\s\S]*\}', patch_raw)
                if match:
                    patch_data = json.loads(match.group())
                else:
                    return source  # 无法解析，返回原始
        except json.JSONDecodeError:
            return source

        # 应用修改
        result = source
        modifications = patch_data.get("modifications", [])
        if isinstance(modifications, str):
            try:
                modifications = json.loads(modifications)
            except Exception:
                modifications = []

        for mod in modifications:
            action = mod.get("action", "")
            old_str = mod.get("old_str", "")
            new_str = mod.get("new_str", "")

            if action == "str_replace" and old_str and new_str:
                result = result.replace(old_str, new_str, 1)
            elif action == "replace_all" and old_str and new_str:
                result = result.replace(old_str, new_str)
            elif action == "delete" and old_str:
                result = result.replace(old_str, "", 1)
            elif action == "insert_method":
                method_code = mod.get("method_code", "")
                class_name = mod.get("class_name", "")
                result = self._inject_method(result, class_name, method_code)

        return result if result != source else source

    @staticmethod
    def _inject_method(source: str, class_name: str, method_code: str) -> str:
        """在类末尾插入方法"""
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # 找到类最后一行的行号
                    last_line = node.end_lineno or node.lineno
                    lines = source.splitlines()
                    if last_line <= len(lines):
                        # 计算缩进
                        indent = "        "
                        method_lines = method_code.strip().splitlines()
                        indented = [method_lines[0]] + [indent + l for l in method_lines[1:]]
                        lines.insert(last_line, "\n".join(indented))
                        return "\n".join(lines)
        except Exception:
            pass
        return source

    def _save_evolution_log(
        self, variants: List[Variant], iteration_id: int, target_file: str
    ):
        """保存进化记录"""
        log_path = self.ckpt_dir / f"alpha_evolve_{iteration_id}.json"
        log_data = {
            "iteration": iteration_id,
            "target_file": target_file,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "variants": [
                {
                    "id": v.variant_id,
                    "gen": v.generation,
                    "score": v.score,
                    "eliminated": v.eliminated,
                    "feedback": v.feedback,
                    "source_preview": v.source[:500] if v.source else "",
                }
                for v in variants
            ],
        }
        try:
            log_path.write_text(
                json.dumps(log_data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass
