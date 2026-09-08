"""
Agent Pipeline — 任务池 + 工作者架构（Route B）

将单次大 Prompt 拆分为 5 个微智能体串行流水线：
  Agent-1 规划器   (LLM, temp=0.4) → 分析失败原因，产出本轮目标
  Agent-2 代码读取器 (本地)         → 从磁盘读取最新源码快照
  Agent-3 补丁工程师 (LLM, adaptive‑temp) → 只生成补丁 JSON
  Agent-4 校验卫士   (本地)         → 预检 old_str 是否存在、JSON 格式
  Agent-5 复盘反思器 (LLM, temp=0.3) → 分析成功/失败，生成反思报告

核心优势：
  - 一次只唤醒 1 个 Agent，干完即休眠，释放显存
  - Agent-3 拿到的永远是磁盘最新源码，old_str 锚点永远新鲜
  - Agent-4 预检拦截无效补丁，不执行写入
  - 所有中间产物落地为文件，支持断点恢复
"""

import json
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import Config

# AlphaEvolve 模式开关
USE_ALPHA_EVOLVE = True

# P1‑1 LLM任务上层业务熔断：单大模型任务最大允许执行秒数
LLM_TASK_TIMEOUT_SEC = 800

# ═════════════════════════════════════════════════════════
#  数据结构
# ═════════════════════════════════════════════════════════

@dataclass
class AtomicTask:
    """一个原子任务单元"""
    task_id: str               # 唯一标识
    task_type: str              # plan / read_code / generate_patch / validate_patch / reflect
    description: str           # 人类可读描述
    requires_llm: bool         # 是否需要调用 LLM
    temperature: float = 0.3   # LLM 温度（仅 requires_llm=True 时生效）
    system_prompt: str = ""    # LLM 系统提示词
    user_prompt: str = ""      # LLM 用户提示词
    force_json: bool = False   # 是否强制 JSON 输出
    context: Dict[str, Any] = field(default_factory=dict)  # 输入数据
    output_key: str = ""       # 结果存储键名
    # ========= P1新增 =========
    timeout_sec: Optional[int] = None


class TaskPool:
    """任务池：管理原子任务队列 + 结果存储"""

    def __init__(self):
        self._queue: deque = deque()
        self._results: Dict[str, Any] = {}
        self._task_log: List[Dict] = []

    def add(self, task: AtomicTask) -> None:
        self._queue.append(task)

    def get_next(self) -> Optional[AtomicTask]:
        return self._queue.popleft() if self._queue else None

    def store_result(self, key: str, value: Any) -> None:
        self._results[key] = value

    def get_result(self, key: str, default=None) -> Any:
        return self._results.get(key, default)

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def log_task(self, task: AtomicTask, duration_ms: float, success: bool,
                 info: str = "", status_code: str = "") -> None:
        """记录任务执行日志。
        status_code: success / skip / failed / hallucination / timeout
        """
        if not status_code:
            status_code = "success" if success else "failed"
        self._task_log.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "description": task.description,
            "duration_ms": round(duration_ms, 1),
            "success": success,
            "status_code": status_code,
            "info": info[:200],
        })

    @property
    def results(self) -> Dict[str, Any]:
        return dict(self._results)

    @property
    def task_log(self) -> List[Dict]:
        return list(self._task_log)


# ═════════════════════════════════════════════════════════
#  提示词模板
# ═════════════════════════════════════════════════════════

# Agent-1 规划器：分析失败原因，产出目标
PLANNER_SYSTEM = (
    "你是迭代规划器。你的唯一任务是分析上一轮迭代的失败原因，"
    "产出一个简短、具体的下一步改进目标。"
    "不写任何代码，只描述要做什么、改哪个文件、预期效果。"
    "目标不超过 3 句话。"
)

PLANNER_USER = """\
=== 最近迭代历史（最多3轮） ===
{history}

=== 当前迭代目标 ===
{objective}

=== 失败统计 ===
连续失败: {consecutive_fails} 次
近期失败模式: {failure_pattern}

请分析失败原因，产出本轮的具体改进目标。只输出目标文字，不要代码。"""

# Agent-3 补丁工程师：只生成补丁 JSON
PATCH_ENGINEER_SYSTEM = (
    "你是补丁工程师。你的唯一任务是根据目标和源码生成代码补丁JSON。"
    "不写任何解释文字，只输出JSON对象。"
    "\n\n"
    "=== JSON 输出协议 ==="
    "1. 完整回复只能是一个JSON对象，首字符{{，末字符}}。"
    "2. 结构: {{\"description\": \"一句话概述\", \"modifications\": [...]}}"
    "3. 每个modification支持:"
    "   - action: \"patch\" | \"replace_all\" | \"insert_method\" | \"new_file\""
    "   - target_file: 目标文件相对路径"
    "   - old_str: patch必填，源码中连续>=2行的精确原文(含缩进)"
    "   - new_str: patch必填，替换后的代码"
    "   - class_name + method_code: insert_method必填"
    "   - reason: 可选"
    "4. 禁止空modifications数组，至少1条patch。"
    "5. old_str必须逐字复制源码，不可简写脑补。"
    "6. 禁止markdown代码块包裹，禁止三引号。"
)

PATCH_ENGINEER_USER = """\
=== 本轮目标 ===
{goal}

=== 目标文件源码（磁盘最新版本） ===
{source_code}

请根据目标和源码，生成代码补丁JSON。只输出JSON。"""

# Agent-5 反思器：分析结果
REFLECTOR_SYSTEM = (
    "你是迭代反思器。分析本轮迭代的成功或失败，给出简短的改进建议。"
    "不写代码，只分析原因和建议，不超过5句话。"
)

REFLECTOR_USER = """\
=== 本轮目标 ===
{goal}

=== 执行结果 ===
成功: {success}
详情: {details}

=== 补丁执行情况 ===
{patch_results}

请分析成功或失败的原因，给出下一轮的改进建议。"""


# ═════════════════════════════════════════════════════════
#  AgentWorker：单一工作者
# ═════════════════════════════════════════════════════════

class AgentWorker:
    """单一工作者：从任务池取任务，执行，存结果"""

    def __init__(self, llm_client, code_manager, parser, checkpoint_dir: Path):
        self.llm = llm_client
        self.code_mgr = code_manager
        self.parser = parser
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def process(self, task: AtomicTask, pool: "TaskPool" = None) -> Tuple[Any, bool, str]:
        """
        执行单个原子任务。增加P1‑1 LLM上层业务超时熔断
        返回 (result, success, info)
        """
        import threading
        t0 = time.perf_counter()
        result_box: List[Any] = [None]
        exc_box: List[Optional[Exception]] = [None]

        def _inner_work():
            try:
                if task.requires_llm:
                    res = self._call_llm(task)
                else:
                    res = self._execute_local(task, pool)
                result_box[0] = res
            except Exception as e:
                exc_box[0] = e

        # LLM任务 + 设置了timeout_sec → 开启线程超时隔离
        if task.requires_llm and task.timeout_sec is not None:
            th = threading.Thread(target=_inner_work, daemon=True)
            th.start()
            th.join(timeout=task.timeout_sec)
            elapsed = (time.perf_counter() - t0) * 1000
            if th.is_alive():
                # ⏱️触发上层业务熔断
                info = f"⏱️ [P1‑1 熔断] LLM任务超时 {task.timeout_sec}s task_id={task.task_id} ({elapsed:.0f}ms)"
                print(f"  {info}")
                return None, False, info
            # 线程已经结束，检查异常
            if exc_box[0] is not None:
                elapsed = (time.perf_counter() - t0) * 1000
                err = f"{type(exc_box[0]).__name__}: {exc_box[0]}"
                return None, False, f"{err} ({elapsed:.0f}ms)"
            # 正常完成
            elapsed = (time.perf_counter() - t0) * 1000
            return result_box[0], True, f"OK ({elapsed:.0f}ms)"
        else:
            # 非LLM任务 / 未设置超时：直接同步执行
            try:
                if task.requires_llm:
                    result = self._call_llm(task)
                else:
                    result = self._execute_local(task, pool)
                elapsed = (time.perf_counter() - t0) * 1000
                return result, True, f"OK ({elapsed:.0f}ms)"
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                err = f"{type(e).__name__}: {e}"
                return None, False, f"{err} ({elapsed:.0f}ms)"

    def _call_llm(self, task: AtomicTask) -> str:
        """调用 LLM（每次调用前重置对话历史，只传必要上下文）"""
        # 设置温度
        if hasattr(self.llm, 'temperature'):
            self.llm.temperature = task.temperature
        # 重置对话历史（每次调用独立，不累积上下文）
        if hasattr(self.llm, 'reset_conversation'):
            self.llm.reset_conversation()
        # 调用
        kwargs = dict(
            user_message=task.user_prompt,
            system_prompt=task.system_prompt,
            stream=False,
        )
        # OllamaClient 支持 force_json
        if hasattr(self.llm, '_backend') and task.force_json:
            kwargs['force_json'] = True
        raw = self.llm.chat(**kwargs)
        return raw or ""

    def _execute_local(self, task: AtomicTask, pool: "TaskPool" = None) -> Any:
        """执行不需要 LLM 的本地任务"""
        if task.task_type == "read_code":
            return self._do_read_code(task)
        elif task.task_type == "validate_patch":
            return self._do_validate_patch(task, pool)
        else:
            raise ValueError(f"未知的本地任务类型: {task.task_type}")

    def _do_read_code(self, task: AtomicTask) -> Dict[str, str]:
        """Agent-2: 从磁盘读取最新源码"""
        files = task.context.get("files", [])
        if not files:
            # 用 SELF_EVOLVE_FILES_CORE 默认文件
            files = Config.SELF_EVOLVE_FILES_CORE

        snapshot = {}
        for rel_path in files:
            content = self.code_mgr.read_file(rel_path)
            if content is not None:
                snapshot[rel_path] = content
            else:
                snapshot[rel_path] = f"# [读取失败] {rel_path}"
        return snapshot

    def _do_validate_patch(self, task: AtomicTask, pool: "TaskPool" = None) -> Dict[str, Any]:
        """
        Agent-4: 预检补丁
        检查: JSON 格式、old_str 是否存在于源码、target_file 是否被保护
        ★ 新增：运行时验证 — 在内存中应用补丁后 import + instantiate 测试
        raw_response 优先从 pool 获取最新版本（支持重试场景）
        """
        raw_response = task.context.get("raw_response", "")
        if pool is not None:
            # 从 pool 获取最新的补丁输出（重试后自动拿到最新版本）
            latest = pool.get_result("patch_raw")
            if latest:
                raw_response = latest
        source_snapshot = task.context.get("source_snapshot", {})
        # 也从 pool 获取最新源码快照
        if pool is not None:
            latest_src = pool.get_result("source_snapshot")
            if latest_src:
                source_snapshot = latest_src

        # 用 parser 解析 JSON
        plan = self.parser.parse(
            raw_response,
            iteration_id=task.context.get("iteration_id", 0),
            objective=task.context.get("objective", ""),
        )

        if not plan.modifications:
            return {
                "valid": False,
                "reason": "解析失败或无修改项",
                "plan": plan,
            }

        # 逐条检查 old_str 是否存在于源码
        errors = []
        for i, mod in enumerate(plan.modifications):
            if mod.action == "patch" and mod.old_str:
                source = source_snapshot.get(mod.target_file, "")
                if mod.old_str not in source:
                    # 尝试忽略整体缩进差异
                    import textwrap
                    dedented_old = textwrap.dedent(mod.old_str)
                    dedented_src = textwrap.dedent(source)
                    if dedented_old not in dedented_src:
                        errors.append(
                            f"  mod[{i}] old_str 在 {mod.target_file} 中未找到"
                        )
            # 检查 PLE 保护
            if mod.target_file in Config.PROTECTED_FILES:
                errors.append(f"  mod[{i}] {mod.target_file} 是 PLE 保护文件，禁止修改")

        if errors:
            return {
                "valid": False,
                "reason": "预检失败:\n" + "\n".join(errors),
                "plan": plan,
            }

        # ── 运行时验证：在内存中应用补丁，然后 import + instantiate ──
        runtime_errors = self._runtime_validate(plan, source_snapshot)
        if runtime_errors:
            return {
                "valid": False,
                "reason": "运行时验证失败:\n" + "\n".join(runtime_errors),
                "plan": plan,
            }

        return {
            "valid": True,
            "reason": "预检通过（含运行时验证）",
            "plan": plan,
        }

    def _runtime_validate(self, plan, source_snapshot: Dict[str, str]) -> list:
        """
        ★ 运行时验证：在内存中应用补丁后，尝试 import + instantiate 类
        拦截：属性未定义先使用、重复 __init__、语法错误等运行时 bug
        """
        import types
        import traceback
        errors = []

        for mod in plan.modifications:
            target_file = mod.target_file
            original_source = source_snapshot.get(target_file, "")
            if not original_source:
                continue  # 源码不可读，跳过运行时验证

            # 在内存中应用补丁
            patched_source = original_source
            try:
                if mod.action == "patch" and mod.old_str and mod.new_str:
                    patched_source = original_source.replace(mod.old_str, mod.new_str, 1)
                elif mod.action == "insert_method" and mod.method_code:
                    # 在类的最后一行之前插入新方法
                    patched_source = self._inject_method_into_source(
                        patched_source, mod.class_name, mod.method_code
                    )
                elif mod.action == "replace_all" and mod.old_str and mod.new_str:
                    patched_source = original_source.replace(mod.old_str, mod.new_str)
                elif mod.action == "delete" and mod.old_str:
                    patched_source = original_source.replace(mod.old_str, "", 1)
            except Exception as e:
                errors.append(f"  内存应用补丁失败: {type(e).__name__}: {e}")
                continue

            if patched_source == original_source and mod.action != "insert_method":
                errors.append(f"  补丁未产生任何变更（old_str 未匹配）")
                continue

            # 1. 语法检查（compile）
            try:
                compile(patched_source, f"<{target_file}>", "exec")
            except SyntaxError as e:
                errors.append(f"  语法错误: {e}")
                continue

            # 2. import + instantiate 测试
            try:
                ns = {}
                exec(patched_source, ns)
                # 找到目标类并尝试实例化
                class_name = getattr(mod, "class_name", None)
                if class_name and class_name in ns:
                    cls = ns[class_name]
                    if isinstance(cls, type):
                        # 尝试无参实例化
                        try:
                            instance = cls()
                        except TypeError:
                            # 需要参数的类，尝试常见默认值
                            try:
                                instance = cls(max_branches=5)
                            except TypeError:
                                try:
                                    instance = cls(config={})
                                except TypeError:
                                    instance = cls(None)
                        except Exception as e:
                            errors.append(
                                f"  实例化 {class_name}() 失败: {type(e).__name__}: {e}"
                            )
            except Exception as e:
                tb = traceback.format_exc()
                # 简化错误信息
                short_tb = tb.split('\n')[-3].strip() if '\n' in tb else str(e)
                errors.append(f"  运行时错误: {type(e).__name__}: {e}")

            # 3. 无调用函数检测（仅 insert_method 补丁）
            if mod.action == "insert_method":
                new_funcs = self._find_new_functions(original_source, patched_source)
                if new_funcs:
                    uncalled = self._check_calls_in_project(new_funcs, target_file)
                    if uncalled:
                        names = ', '.join(uncalled)
                        errors.append(
                            f"  新增函数无调用者: {names} — "
                            f"请同时在补丁中添加调用点，或使用 str_replace 修改已有代码"
                        )

        return errors

    @staticmethod
    def _find_new_functions(original: str, patched: str) -> set:
        """用 AST 对比原始和补丁后源码，找出新增的函数名"""
        import ast as _ast
        def _collect_names(source):
            names = set()
            try:
                tree = _ast.parse(source)
                for node in _ast.walk(tree):
                    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        names.add(node.name)
            except Exception:
                pass
            return names

        orig_names = _collect_names(original)
        patched_names = _collect_names(patched)
        # 排除 dunder 方法（__init__, __str__ 等）
        new_names = {
            n for n in (patched_names - orig_names)
            if not n.startswith('_')
        }
        return new_names

    @staticmethod
    def _check_calls_in_project(func_names: set, exclude_file: str) -> list:
        """扫描全项目 .py 文件，检查 func_names 中哪些函数完全没有调用者
        返回无调用者的函数名列表
        """
        import os
        import re
        from pathlib import Path

        project_root = Path(Config.PROJECT_ROOT)
        # 排除目录
        skip_dirs = {'backups', '__pycache__', '.git', 'logs', 'evolver/branches'}
        # 排除文件（被修改的文件本身不算调用者）
        exclude_path = project_root / exclude_file

        # 构建搜索模式：func_name( 或 .func_name(
        patterns = {}
        for name in func_names:
            # 匹配 func_name( 或 .func_name( 或 self.func_name(
            patterns[name] = re.compile(
                r'\b' + re.escape(name) + r'\s*\('
            )

        # 找到的调用者
        callers_found = {name: False for name in func_names}

        for py_file in project_root.rglob('*.py'):
            # 跳过排除目录
            rel = str(py_file.relative_to(project_root)).replace('\\', '/')
            if any(rel.startswith(d) for d in skip_dirs):
                continue
            # 跳过临时测试文件
            if rel.startswith('_test') or rel.startswith('_'):
                continue
            if py_file == exclude_path:
                continue

            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue

            for name, pattern in patterns.items():
                if callers_found[name]:
                    continue
                for match in pattern.finditer(content):
                    # 检查匹配是否在 def 行上（排除函数定义）
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    prefix = content[line_start:match.start()].strip()
                    if prefix.startswith('def '):
                        continue
                    # 非定义行 → 真实调用
                    callers_found[name] = True
                    break

        # 返回无调用者的函数名
        return [name for name, found in callers_found.items() if not found]

    @staticmethod
    def _inject_method_into_source(source: str, class_name: str, method_code: str) -> str:
        """在源码中找到指定类，在类末尾插入新方法"""
        import ast
        import re
        try:
            tree = ast.parse(source)
            lines = source.split('\n')
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # 找到类的最后一行
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.body[-1].end_lineno
                    # 插入方法代码（4 空格缩进）
                    indented_code = '\n'.join('    ' + line if line.strip() else line
                                             for line in method_code.strip().split('\n'))
                    lines.insert(end_line, indented_code)
                    return '\n'.join(lines)
        except Exception:
            pass
        # 兜底：直接追加到文件末尾
        return source + '\n' + method_code + '\n'

    def _save_checkpoint(self, task: AtomicTask, result: Any, iteration_id: int) -> None:
        """将中间产物保存到文件（断点恢复）"""
        filename = f"ckpt_{iteration_id:04d}_{task.task_type}.json"
        path = self.checkpoint_dir / filename
        try:
            # 尝试 JSON 序列化
            if isinstance(result, (dict, list, str, int, float, bool, type(None))):
                content = json.dumps(result, ensure_ascii=False, indent=2) if not isinstance(result, str) else result
            else:
                content = str(result)
            path.write_text(content, encoding="utf-8")
        except Exception:
            path.write_text(str(result), encoding="utf-8")


# ═════════════════════════════════════════════════════════
#  PipelineRunner：流水线编排器
# ═════════════════════════════════════════════════════════

class PipelineRunner:
    """5 步串行流水线编排器"""

    MAX_PATCH_RETRIES = 2  # 补丁生成最大重试次数

    @staticmethod
    def _extract_target_file(objective: str) -> str:
        """从目标文本中提取实际要处理的文件路径
        目标格式: '现在你只能够处理 evolver/xxx.py'
        """
        import re
        # 匹配 evolver/xxx.py 或其他相对路径
        match = re.search(r'(evolver/\w+\.py)', objective)
        if match:
            return match.group(1)
        # 兜底：返回第一个可进化文件
        return Config.SELF_EVOLVE_FILES_CORE[0] if Config.SELF_EVOLVE_FILES_CORE else "evolver/branch_manager.py"

    def __init__(self, evolver):
        self.evolver = evolver
        self.llm = evolver.doubao
        self.code_mgr = evolver.code_mgr
        self.parser = evolver.parser
        self.reflection = evolver.reflection
        self.diagnosis = evolver.diagnosis
        self.checkpoint_dir = Config.LOG_DIR / "pipeline"

    def run(self, objective: str, iteration_id: int) -> Dict[str, Any]:
        """
        执行一轮 5 步流水线迭代。
        返回与 SelfEvolver.run_iteration() 兼容的 result dict。
        """
        print(f"\n  ═══════════════════════════════════════════")
        print(f"  🔄 Agent Pipeline 迭代 #{iteration_id}")
        print(f"  ═══════════════════════════════════════════")

        pool = TaskPool()
        worker = AgentWorker(self.llm, self.code_mgr, self.parser, self.checkpoint_dir)

        # 构建初始任务
        self._init_tasks(pool, objective, iteration_id)

        # 执行所有任务
        while not pool.is_empty():
            task = pool.get_next()
            print(f"\n  📋 [{task.task_id}] {task.description}")

            result, success, info = worker.process(task, pool)
            pool.store_result(task.output_key, result)

            # 推导 status_code
            status_code = "success" if success else "failed"
            if not success and isinstance(result, str) and result == "":
                status_code = "skip"
            elif not success and info and ("超时" in info or "timeout" in info.lower()):
                status_code = "timeout"
            elif not success and info and ("未匹配" in info or "无效修改" in info or "解析失败" in info):
                status_code = "hallucination"

            pool.log_task(task, 0, success, info, status_code=status_code)

            if success:
                print(f"  ✅ [{task.task_id}] {info}")
                worker._save_checkpoint(task, result, iteration_id)
            else:
                print(f"  ❌ [{task.task_id}] {info}")

            # 动态任务生成（Route B 核心：根据结果决定后续任务）
            self._maybe_add_followup(pool, task, result, success, objective, iteration_id)

        # 构建结果
        return self._build_result(pool, objective, iteration_id)

    def _init_tasks(self, pool: TaskPool, objective: str, iteration_id: int) -> None:
        """初始化前 3 个任务（后续任务根据结果动态生成）"""

        # ── Agent-1: 规划器 ──
        history = self._get_recent_history(3)
        failure_pattern = self._get_failure_pattern()
        consecutive_fails = self.reflection._consecutive_fail if hasattr(self.reflection, '_consecutive_fail') else 0

        planner_user = PLANNER_USER.format(
            history=history or "(无历史记录)",
            objective=objective,
            consecutive_fails=consecutive_fails,
            failure_pattern=failure_pattern or "(无)",
        )

        pool.add(AtomicTask(
            task_id="agent1_planner",
            task_type="plan",
            description="Agent-1 规划器: 分析失败原因，产出本轮目标",
            requires_llm=True,
            temperature=0.4,
            system_prompt=PLANNER_SYSTEM,
            user_prompt=planner_user,
            force_json=False,
            output_key="goal",
        ))

        # ── Agent-2: 代码读取器 ──
        target_file = self._extract_target_file(objective)
        pool.add(AtomicTask(
            task_id="agent2_reader",
            task_type="read_code",
            description=f"Agent-2 代码读取器: 读取 {target_file} 最新源码",
            requires_llm=False,
            context={"files": [target_file]},
            output_key="source_snapshot",
        ))

        # ── Agent-3: 补丁工程师 ──（在 _maybe_add_followup 中动态构建，
        #   因为需要 agent1 的 goal 和 agent2 的 source）

    def _maybe_add_followup(
        self, pool: TaskPool, task: AtomicTask,
        result: Any, success: bool,
        objective: str, iteration_id: int,
    ) -> None:
        """根据任务结果动态生成后续任务（Route B 核心：动态任务派发）"""

        def _refresh_snapshot() -> Dict[str, str]:
            """局部辅助：重读磁盘获取最新源码快照，存入pool缓存"""
            target_file = self._extract_target_file(objective)
            snapshot = {}
            content = self.code_mgr.read_file(target_file)
            if content is not None:
                snapshot[target_file] = content
            else:
                snapshot[target_file] = f"# [读取失败] {target_file}"
            pool.store_result("source_snapshot", snapshot)
            return snapshot

        if task.task_type == "plan" and success:
            # 规划完成，代码读取器已在队列中
            pass
        elif task.task_type == "read_code" and success:
            # 代码读取完成 → 启动补丁工程师或 AlphaEvolve
            goal = pool.get_result("goal", objective)
            source_snapshot = result
            source_code = self._format_source_for_prompt(source_snapshot)

            if USE_ALPHA_EVOLVE:
                # ★ AlphaEvolve 模式：生成 N 个变体 → 评估 → 选最优
                target_file = self._extract_target_file(objective)
                from evolver.alpha_evolve import PopulationManager
                pop_mgr = PopulationManager(
                    llm_client=self.llm,
                    code_manager=self.code_mgr,
                    parser=self.parser,
                    checkpoint_dir=self.checkpoint_dir,
                    population_size=3,
                    elite_size=1,
                    max_generations=2,
                )
                best_variant, all_variants = pop_mgr.evolve(
                    objective=objective,
                    target_file=target_file,
                    source_snapshot=source_snapshot,
                    goal=goal,
                    iteration_id=iteration_id,
                    system_prompt=PATCH_ENGINEER_SYSTEM,
                    user_prompt_template=PATCH_ENGINEER_USER,
                )
                if best_variant and best_variant.score >= 0:
                    # 用最优变体的源码直接写入
                    pool.store_result("alpha_best", best_variant)
                    pool.store_result("alpha_all", all_variants)
                    pool.store_result("final_plan", "alpha_evolve")
                else:
                    pool.store_result("alpha_exhausted", True)
                    if self.reflection is not None:
                        try:
                            self.reflection.record_event(
                                iteration_id=iteration_id,
                                is_failure=True,
                                category="ALPHA_EVOLVE_FAIL",
                                issues=["AlphaEvolve: 所有变体被淘汰"],
                            )
                        except Exception:
                            pass
            else:
                # 原有单补丁模式
                patch_user = PATCH_ENGINEER_USER.format(
                    goal=goal,
                    source_code=source_code,
                )
                pool.add(AtomicTask(
                    task_id="agent3_engineer",
                    task_type="generate_patch",
                    description="Agent‑3 补丁工程师: 生成代码补丁 JSON",
                    requires_llm=True,
                    temperature=self.llm.profile.calculate_temperature(),
                    system_prompt=PATCH_ENGINEER_SYSTEM,
                    user_prompt=patch_user,
                    force_json=True,
                    context={
                        "source_snapshot": source_snapshot,
                        "goal": goal,
                        "retry_count": 0,
                    },
                    output_key="patch_raw",
                    timeout_sec=LLM_TASK_TIMEOUT_SEC,  # P1‑1新增
                ))
        elif task.task_type == "generate_patch" and success:
            # 补丁生成完成（含重试）→ 启动校验卫士
            pool.add(AtomicTask(
                task_id="agent4_validator",
                task_type="validate_patch",
                description="Agent‑4 校验卫士: 预检补丁合法性",
                requires_llm=False,
                context={
                    "iteration_id": iteration_id,
                    "objective": objective,
                    "retry_count": task.context.get("retry_count", 0),
                },
                output_key="validated_patch", 
            ))
        elif task.task_type == "generate_patch" and not success:
            # 补丁生成失败 → 重试
            retry_count = task.context.get("retry_count", 0) + 1
            if retry_count <= self.MAX_PATCH_RETRIES:
                print(f"  🔄 补丁生成失败，重试 ({retry_count}/{self.MAX_PATCH_RETRIES})")
                goal = pool.get_result("goal", objective)
                source_snapshot = _refresh_snapshot()
                source_code = self._format_source_for_prompt(source_snapshot)
                feedback = f"\n\n⚠️ 上次生成失败: {result}\n请修正问题，重新生成补丁。"
                patch_user = PATCH_ENGINEER_USER.format(
                    goal=goal, source_code=source_code,
                ) + feedback
                pool.add(AtomicTask(
                    task_id=f"agent3_retry{retry_count}",
                    task_type="generate_patch",
                    description=f"Agent‑3 补丁工程师 (重试{retry_count})",
                    requires_llm=True,
                    temperature=self.llm.profile.calculate_temperature(),
                    system_prompt=PATCH_ENGINEER_SYSTEM,
                    user_prompt=patch_user,
                    force_json=True,
                    context={
                        "source_snapshot": source_snapshot,
                        "goal": goal,
                        "retry_count": retry_count,
                    },
                    output_key="patch_raw",
                    timeout_sec=LLM_TASK_TIMEOUT_SEC
                ))
            else:
                pool.store_result("patch_exhausted", True)
                if self.reflection is not None:
                    try:
                        self.reflection.record_event(
                            iteration_id=iteration_id,
                            is_failure=True,
                            category="PATCH_GEN_EXHAUSTED",
                            issues=["补丁生成重试耗尽：AI 连续生成失败，无法产出有效 JSON"],
                        )
                    except Exception:
                        pass
        elif task.task_type == "validate_patch":
            # 校验完成 → 根据结果决定，【整个函数仅此一处validate_patch分支】
            if result and result.get("valid"):
                # 校验通过 → 标记可以应用补丁
                pool.store_result("final_plan", result.get("plan"))
            elif result and not result.get("valid"):
                # =====新增：上报补丁校验失败中间事件=====
                reason_text = result.get("reason", "补丁校验未知失败")
                info_lines = [line.strip() for line in reason_text.splitlines() if line.strip()]
                if self.reflection is not None:
                    try:
                        self.reflection.record_event(
                            iteration_id=iteration_id,
                            is_failure=True,
                            category="VALIDATE_PATCH_FAIL",
                            issues=info_lines
                        )
                    except Exception:
                        pass
        # =====================================
                # 校验失败 → 重试补丁生成
                retry_count = task.context.get("retry_count", 0) + 1
                if retry_count <= self.MAX_PATCH_RETRIES:
                    reason = result.get("reason", "未知原因")
                    print(f"  🔄 校验失败，重新生成补丁 ({retry_count}/{self.MAX_PATCH_RETRIES})")
                    print(f"     原因: {reason}")
                    goal = pool.get_result("goal", objective)
                    source_snapshot = _refresh_snapshot()
                    source_code = self._format_source_for_prompt(source_snapshot)
                    feedback = (
                        f"\n\n⚠️ 上次补丁校验失败:\n{reason}\n"
                        "请修正以上问题，确保 old_str 逐字匹配源码，重新生成补丁。"
                    )
                    patch_user = PATCH_ENGINEER_USER.format(
                        goal=goal, source_code=source_code,
                    ) + feedback
                    pool.add(AtomicTask(
                        task_id=f"agent3_retry{retry_count}",
                        task_type="generate_patch",
                        description=f"Agent‑3 补丁工程师 (校验失败重试{retry_count})",
                        requires_llm=True,
                        temperature=self.llm.profile.calculate_temperature(),
                        system_prompt=PATCH_ENGINEER_SYSTEM,
                        user_prompt=patch_user,
                        force_json=True,
                        context={
                            "source_snapshot": source_snapshot,
                            "goal": goal,
                            "retry_count": retry_count,
                        },
                        output_key="patch_raw",
                        timeout_sec=LLM_TASK_TIMEOUT_SEC,
                    ))
                else:
                    pool.store_result("validation_exhausted", True)
                    if self.reflection is not None:
                        try:
                            self.reflection.record_event(
                                iteration_id=iteration_id,
                                is_failure=True,
                                category="VALIDATE_EXHAUSTED",
                                issues=["校验重试耗尽：补丁连续被拒，原因可能是 old_str 不匹配或运行时验证失败"],
                            )
                        except Exception:
                            pass


    def _build_result(
        self, pool: TaskPool, objective: str, iteration_id: int,
    ) -> Dict[str, Any]:
        """构建与 run_iteration() 兼容的结果 dict"""

        task_log = pool.task_log
        goal = pool.get_result("goal", objective)
        source_snapshot = pool.get_result("source_snapshot", {})
        validated = pool.get_result("validated_patch", {})
        patch_exhausted = pool.get_result("patch_exhausted", False)
        validation_exhausted = pool.get_result("validation_exhausted", False)
        alpha_best = pool.get_result("alpha_best", None)
        alpha_all = pool.get_result("alpha_all", [])
        alpha_exhausted = pool.get_result("alpha_exhausted", False)

        # 初始化结果（键名与 run_iteration 保持兼容）
        result = {
            "iteration": iteration_id,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "objective": objective,
            "pipeline_goal": goal,
            "status": "待执行",
            "issues": [],
            "applied_count": 0,
            "suggestion_count": 0,
            "failed_count": 0,
            "rollback": False,
            "reflection": "",
            "description": "",
            "applied_details": [],
            "task_log": task_log,
            "mode": "alpha_evolve" if alpha_best else "pipeline",
        }

        # ★ AlphaEvolve 模式：直接用最优变体源码写入
        if alpha_best:
            target_file = self._extract_target_file(objective)
            original_source = source_snapshot.get(target_file, "")
            if alpha_best.source and alpha_best.source != original_source:
                iter_tag = f"alpha_{iteration_id}"
                try:
                    # 先备份
                    self.code_mgr.backup_file(target_file, iter_tag)
                    # 写入新源码
                    self.code_mgr.write_file(target_file, alpha_best.source)
                    result["applied_count"] = 1
                    result["suggestion_count"] = len(alpha_all)
                    result["failed_count"] = 0
                    result["status"] = "修改成功"
                    result["applied_details"] = [{"file": target_file, "success": True, "action": "alpha_write"}]
                    result["description"] = (
                        f"AlphaEvolve: 变体#{alpha_best.variant_id} "
                        f"分数={alpha_best.score:.0f} "
                        f"反馈={alpha_best.feedback[:2]}"
                    )
                    print(f"  🏆 AlphaEvolve 最优变体已写入: "
                          f"变体#{alpha_best.variant_id} 分数={alpha_best.score:.0f}")
                    # 验证
                    modified_files = [target_file]
                    try:
                        validation = self.evolver._validate_files(modified_files)
                        failed_files = [f for f, (ok, _) in validation.items() if not ok]
                        if failed_files:
                            result["issues"].append(f"语法/导入校验失败: {failed_files}")
                            for ff in failed_files:
                                ok, msg = self.diagnosis.diagnose_and_repair(
                                    ff, validation[ff][1], self.code_mgr
                                )
                                if ok:
                                    result["issues"].append(f"自愈: {ff} → {msg}")
                                else:
                                    result["rollback"] = True
                                    result["issues"].append(f"自愈失败: {ff} → {msg}")
                    except Exception as e:
                        result["issues"].append(f"验证异常: {e}")
                except Exception as e:
                    result["status"] = "写入失败"
                    result["issues"].append(f"AlphaEvolve 写入异常: {e}")
            else:
                result["status"] = "无变更"
                result["issues"].append("AlphaEvolve: 最优变体与原始源码无差异")

            return result

        if alpha_exhausted:
            result["status"] = "AlphaEvolve 全部淘汰"
            result["issues"].append("所有变体被淘汰（语法/运行时/幻觉检测）")
            return result

        # 处理补丁执行
        plan = None
        if validated and validated.get("valid"):
            plan = validated.get("plan")
            result["suggestion_count"] = len(plan.modifications) if plan else 0

            # 应用补丁
            iter_tag = f"pipe_{iteration_id}"
            try:
                applied_results = self.evolver._apply_modifications(plan, iter_tag)
                result["applied_details"] = applied_results
                result["impact_chars"] = sum(
                    r.get("delta_chars", 0) for r in applied_results if r.get("success")
                )
                success_count = sum(1 for r in applied_results if r.get("success"))
                fail_count = len(applied_results) - success_count
                result["applied_count"] = success_count
                result["failed_count"] = fail_count
                result["description"] = plan.description or "Pipeline 迭代"

                if success_count > 0 and fail_count == 0:
                    result["status"] = "修改成功"
                elif success_count > 0:
                    result["status"] = "部分成功"
                else:
                    result["status"] = "补丁应用失败"
                    result["issues"].append("所有补丁应用失败")

                # 验证 + 自愈
                modified_files = [r["file"] for r in applied_results if r.get("success")]
                if modified_files:
                    try:
                        validation = self.evolver._validate_files(modified_files)
                        failed_files = [f for f, (ok, _) in validation.items() if not ok]
                        if failed_files:
                            result["issues"].append(f"语法/导入校验失败: {failed_files}")
                            # 自愈
                            for ff in failed_files:
                                ok, msg = self.diagnosis.diagnose_and_repair(
                                    ff, validation[ff][1], self.code_mgr
                                )
                                if ok:
                                    result["issues"].append(f"自愈: {ff} → {msg}")
                                else:
                                    result["rollback"] = True
                                    result["issues"].append(f"自愈失败: {ff} → {msg}")
                    except Exception as e:
                        result["issues"].append(f"验证异常: {e}")

                # Tier-2 观察期注册：对受控开放文件，记录观察期
                if hasattr(self.code_mgr, 'is_tier2_file') and hasattr(self.code_mgr, 'register_tier2_modification'):
                    for r in applied_results:
                        if r.get("success"):
                            fpath = r.get("file", "")
                            if fpath and self.code_mgr.is_tier2_file(fpath):
                                backup = r.get("backup", "")
                                try:
                                    self.code_mgr.register_tier2_modification(
                                        fpath, backup, iteration_id
                                    )
                                except Exception:
                                    pass

            except Exception as e:
                result["status"] = "执行异常"
                result["issues"].append(f"{type(e).__name__}: {e}")
        else:
            # 补丁未通过校验或生成失败
            if patch_exhausted:
                result["status"] = "补丁生成失败"
                result["issues"].append("多次重试后补丁生成仍失败")
            elif validation_exhausted:
                result["status"] = "补丁校验失败"
                result["issues"].append("多次重试后补丁校验仍不通过")
            elif validated:
                result["status"] = "补丁校验失败"
                result["issues"].append(validated.get("reason", "未知原因"))
            else:
                result["status"] = "流水线异常"
                result["issues"].append("流水线未产出有效补丁")

        # Agent-5: 反思器（LLM 调用）
        reflector_user = REFLECTOR_USER.format(
            goal=goal,
            success=result["status"] in ("修改成功", "部分成功"),
            details=result["status"] + ": " + "; ".join(result["issues"]) if result["issues"] else result["status"],
            patch_results="\n".join(
                f"  - {t['task_id']} [{t.get('status_code','?')}]: {'OK' if t['success'] else 'FAIL'}/{t['info']}"
                for t in task_log
            ),
        )

        try:
            if hasattr(self.llm, 'temperature'):
                self.llm.temperature = 0.3
            if hasattr(self.llm, 'reset_conversation'):
                self.llm.reset_conversation()
            kwargs = dict(
                user_message=reflector_user,
                system_prompt=REFLECTOR_SYSTEM,
                stream=False,
            )
            if hasattr(self.llm, '_backend'):
                kwargs['force_json'] = False
            reflection_text = self.llm.chat(**kwargs)
            result["reflection"] = reflection_text or ""
            # 保存反思检查点
            ckpt_path = self.checkpoint_dir / f"ckpt_{iteration_id:04d}_reflect.txt"
            ckpt_path.write_text(reflection_text or "", encoding="utf-8")
        except Exception as e:
            result["reflection"] = f"反思异常: {e}"

        # 记录到 ErrorReflection（保持与原系统兼容）
        try:
            self.evolver._record_reflection(result, objective)
        except Exception:
            pass

        # 打印摘要
        print(f"\n  ═══════════════════════════════════════════")
        print(f"  📊 Pipeline 迭代 #{iteration_id} 完成")
        print(f"     状态: {result['status']}")
        print(f"     建议: {result['suggestion_count']} 条")
        print(f"     成功: {result['applied_count']} 条")
        print(f"     失败: {result['failed_count']} 条")
        if result["reflection"]:
            print(f"     反思: {result['reflection'][:200]}")
        print(f"  ═══════════════════════════════════════════")

        return result

    # ═════════════════════════════════════════════════════════
    #  辅助方法
    # ═════════════════════════════════════════════════════════

    def _get_recent_history(self, n: int = 3) -> str:
        """获取最近 n 轮迭代历史"""
        try:
            logs = self.evolver.iteration_logs[-n:] if self.evolver.iteration_logs else []
            if not logs:
                return ""
            lines = []
            for log in logs:
                it = log.get("iteration", "?")
                status = log.get("status", "?")
                obj = log.get("objective", "")[:80]
                lines.append(f"  迭代#{it}: {status} | 目标: {obj}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _get_failure_pattern(self) -> str:
        """获取近期失败模式"""
        try:
            return self.reflection.format_for_prompt(max_len=200) or ""
        except Exception:
            return ""

    def _format_source_for_prompt(self, snapshot: Dict[str, str]) -> str:
        """将源码快照格式化为提示词文本"""
        if not snapshot:
            return "(无源码)"
        lines = []
        for path, code in snapshot.items():
            lines.append(f"--- {path} ---")
            lines.append(code)
        return "\n".join(lines)
