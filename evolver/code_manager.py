"""
代码管理器 - 负责代码文件的读取、备份、修改、验证
确保修改操作的安全性和可回滚性
"""
import os
import sys
import ast
import shutil
import json
import subprocess
import difflib
import re
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any


# 并发线程池大小（IO bound 不用太多线程）
_IO_WORKERS = 8
_CPU_WORKERS = min(4, (os.cpu_count() or 4))


class CodeManager:
    """代码文件管理和操作引擎"""

    def __init__(self, project_root: Path, backup_dir: Path, auto_backup: bool = True):
        """
        初始化代码管理器
        :param project_root: 项目根目录
        :param backup_dir: 备份目录
        :param auto_backup: 是否修改前自动备份
        """
        self.project_root = project_root
        self.backup_dir = backup_dir
        self.auto_backup = auto_backup
        # daemon 模式保护开关：运行时禁止修改 PROTECTED_FILES
        self.daemon_protection = False
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._modification_history: List[Dict[str, Any]] = []

        # 破坏性修改黑名单：{file_path: {"fail_count": int, "banned_until_iter": int}}
        # 连续 2 次合约校验失败 → 禁改 80 轮，避免反复改坏核心模块
        self._blacklist_file = self.project_root / "logs" / "modification_blacklist.json"
        self._blacklist: Dict[str, Dict[str, int]] = self._load_blacklist()

        # ── 分层 PLE: Tier-2 观察期追踪 ──
        # {file_path: {"backup_path": str, "modified_iter": int, "observation_until": int}}
        self._tier2_observations: Dict[str, Dict[str, Any]] = {}

    # ========== 破坏性修改黑名单 ==========

    BLACKLIST_THRESHOLD = 2       # 连续失败次数阈值
    BLACKLIST_BAN_ROUNDS = 80     # 禁改轮数

    def _load_blacklist(self) -> Dict[str, Dict[str, int]]:
        """从磁盘加载黑名单"""
        try:
            if self._blacklist_file.exists():
                with open(self._blacklist_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _auto_fix_syntax(self, content: str, error_msg: str) -> Optional[str]:
        """
        尝试自动修复常见的 Python 语法错误。
        当前支持：未闭合的三引号字符串
        """
        lines = content.split('\n')
        
        # 1. 检测：unterminated string literal (三引号未闭合)
        if "unterminated string literal" in error_msg:
            # 查找未闭合的三引号
            in_triple = False
            triple_char = None
            fixed_lines = []
            
            for line in lines:
                # 检测三引号
                if '"""' in line and not in_triple:
                    in_triple = True
                    triple_char = '"""'
                    # 检查这一行是否闭合了（同一行内有两个三引号）
                    if line.count('"""') >= 2:
                        in_triple = False
                    fixed_lines.append(line)
                elif "'''" in line and not in_triple:
                    in_triple = True
                    triple_char = "'''"
                    if line.count("'''") >= 2:
                        in_triple = False
                    fixed_lines.append(line)
                elif in_triple:
                    # 检查这一行是否闭合
                    if triple_char in line:
                        in_triple = False
                    fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            
            # 如果文档字符串未闭合，强制补上
            if in_triple and triple_char:
                # 找到最后一个非空行，追加闭合引号
                for i in range(len(fixed_lines) - 1, -1, -1):
                    if fixed_lines[i].strip():
                        fixed_lines[i] = fixed_lines[i].rstrip() + triple_char
                        break
                return '\n'.join(fixed_lines)
        
        # 2. 检测：EOL while scanning string literal（单引号/双引号未闭合）
        if "EOL while scanning string literal" in error_msg:
            # 粗暴但有效：给未闭合的字符串补一个引号
            # 解析 error_msg 提取行号（如果有）
           
            match = re.search(r'line (\d+)', error_msg)
            if match:
                line_num = int(match.group(1)) - 1  # 转为 0-indexed
                if 0 <= line_num < len(lines):
                    # 检测这一行的引号类型
                    line = lines[line_num]
                    # 简单处理：如果行中有未配对的引号，补一个
                    if line.count('"') % 2 == 1:
                        lines[line_num] = line + '"'
                        return '\n'.join(lines)
                    elif line.count("'") % 2 == 1:
                        lines[line_num] = line + "'"
                        return '\n'.join(lines)
        
        return None

    def _save_blacklist(self) -> None:
        """持久化黑名单到磁盘"""
        try:
            self._blacklist_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._blacklist_file, "w", encoding="utf-8") as f:
                json.dump(self._blacklist, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def record_contract_failure(self, files: List[str], current_iteration: int,
                                failure_type: str = "contract_violation") -> None:
        """
        合约校验失败后调用：按失败类型差异化处理。
        :param files: 本次修改的文件列表（相对路径）
        :param current_iteration: 当前迭代编号
        :param failure_type: 失败类型
            - "missing_method": 缺方法/缺能力 → 不拉黑（ban_rounds=0），提示 AI 补全
            - "logic_error": 逻辑错误 → 短封 5 轮，给 AI 学习机会
            - "contract_violation"/"syntax_error": 其他 → 长封 80 轮（人工介入）
        """
        # 按失败类型决定封禁轮数
        if failure_type == "missing_method":
            ban_rounds = 0
        elif failure_type == "logic_error":
            ban_rounds = 5
        else:
            ban_rounds = self.BLACKLIST_BAN_ROUNDS

        for f in files:
            entry = self._blacklist.get(f, {"fail_count": 0, "banned_until_iter": 0,
                                            "failure_type": failure_type})
            # 仅在 ban 期已过（banned_until_iter > 0 且已过期）时重置计数
            banned_until = entry.get("banned_until_iter", 0)
            if banned_until > 0 and banned_until < current_iteration:
                entry["fail_count"] = 0
                entry["banned_until_iter"] = 0
            entry["fail_count"] = entry.get("fail_count", 0) + 1
            entry["failure_type"] = failure_type

            if failure_type == "missing_method":
                # 缺方法：不拉黑，只提示
                print(f"  💡 [能力缺失] {f} 缺少方法实现 (fail={entry['fail_count']})，"
                      f"不拉黑，引导 AI 补全")
                entry["banned_until_iter"] = 0
            elif entry["fail_count"] >= self.BLACKLIST_THRESHOLD:
                entry["banned_until_iter"] = current_iteration + ban_rounds
                if ban_rounds > 0:
                    print(f"  🚫 [黑名单] {f} 连续 {entry['fail_count']} 次失败 "
                          f"({failure_type}) → 禁改 {ban_rounds} 轮 "
                          f"(至 iter {entry['banned_until_iter']})")
            self._blacklist[f] = entry
        self._save_blacklist()

    def record_contract_success(self, files: List[str]) -> None:
        """合约通过后重置该文件的失败计数"""
        changed = False
        for f in files:
            if f in self._blacklist:
                self._blacklist[f]["fail_count"] = 0
                changed = True
        if changed:
            self._save_blacklist()

    def is_file_blacklisted(self, file_path: str, current_iteration: int) -> bool:
        """检查文件当前是否在黑名单禁改期内"""
        entry = self._blacklist.get(file_path)
        if not entry:
            return False
        return entry.get("banned_until_iter", 0) > current_iteration

    # ========== 分层 PLE: Tier-2 观察期 ==========

    def register_tier2_modification(self, file_path: str, backup_path: str,
                                     current_iteration: int) -> None:
        """
        Tier-2 文件被修改后注册观察期。
        观察期内（默认 3 轮）若触发 kill-switch，则自动回滚。
        """
        from config import Config
        obs_rounds = getattr(Config, 'PLE_TIER2_OBSERVATION_ROUNDS', 3)
        self._tier2_observations[file_path] = {
            "backup_path": backup_path,
            "modified_iter": current_iteration,
            "observation_until": current_iteration + obs_rounds,
        }

    def check_tier2_rollback(self, current_iteration: int, kill_switch_triggered: bool = False) -> List[str]:
        """
        kill-switch 触发时调用：检查 Tier-2 观察期文件，回滚未过期的修改。
        :param kill_switch_triggered: 是否真的触发了 kill-switch
        :return: 已回滚的文件列表
        """
        rolled_back = []
        to_remove = []
        
        # 只有真正触发 kill-switch 时才回滚
        if not kill_switch_triggered:
            # 清理所有已过期的观察记录（不检查观察期）
            for file_path, obs in self._tier2_observations.items():
                # 只要注册过，就移除（因为观察期已经过去了）
                to_remove.append(file_path)
            for f in to_remove:
                del self._tier2_observations[f]
            return rolled_back
        
        # kill-switch 触发时，回滚观察期内的文件
        for file_path, obs in self._tier2_observations.items():
            if current_iteration <= obs["observation_until"]:
                # 在观察期内 → 回滚
                backup = obs["backup_path"]
                target = self.project_root / file_path
                try:
                    if Path(backup).exists() and target.exists():
                        shutil.copy2(backup, target)
                        rolled_back.append(file_path)
                        print(f"  🔄 [Tier-2 回滚] {file_path} (观察期内 kill-switch)")
                except Exception as e:
                    print(f"  ⚠️ [Tier-2 回滚失败] {file_path}: {e}")
            # 观察期已过 → 移除追踪
            if current_iteration > obs["observation_until"]:
                to_remove.append(file_path)

        for f in to_remove:
            del self._tier2_observations[f]

        return rolled_back

    def is_tier2_file(self, file_path: str) -> bool:
        """检查文件是否属于 Tier-2 受控开放"""
        from config import Config
        tier2_list = getattr(Config, 'PLE_TIER2_CONTROLLED', [])
        return file_path in tier2_list

    def filter_blacklisted(self, modifications: List[Any], current_iteration: int) -> Tuple[List[Any], List[str]]:
        """
        过滤掉黑名单中的修改项。
        :return: (allowed_mods, blocked_files)
        """
        allowed, blocked = [], []
        for mod in modifications:
            f = getattr(mod, "target_file", "")
            if f and self.is_file_blacklisted(f, current_iteration):
                blocked.append(f)
            else:
                allowed.append(mod)
        return allowed, blocked

    def _is_protected(self, relative_path: str) -> bool:
        """检查文件是否受 PLE 保护"""
        if not self.daemon_protection:
            return False
        try:
            from config import Config
            protected = getattr(Config, 'PROTECTED_FILES', [])
            norm = relative_path.replace("\\", "/")
            return norm in protected
        except Exception:
            return False

    # ========== 文件读取 ==========
    def read_file(self, relative_path: str) -> Optional[str]:
        """
        读取项目中的文件内容
        :param relative_path: 相对于项目根目录的路径
        :return: 文件内容，失败返回None
        """
        filepath = self.project_root / relative_path
        try:
            if not filepath.exists():
                print(f"[CodeManager] 文件不存在: {relative_path}")
                return None
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[CodeManager] 读取文件 {relative_path} 失败: {e}")
            return None

    def read_multiple_files(self, relative_paths: List[str]) -> Dict[str, str]:
        """批量读取多个文件（并发），返回 {相对路径: 内容}"""
        if len(relative_paths) <= 1:
            # 单文件不开线程池，省开销
            result = {}
            for path in relative_paths:
                content = self.read_file(path)
                if content is not None:
                    result[path] = content
            return result

        result: Dict[str, str] = {}
        # IO 并发：文件读取纯 IO bound
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(_IO_WORKERS, len(relative_paths))) as pool:
            future_to_path = {pool.submit(self.read_file, p): p for p in relative_paths}
            for future in concurrent.futures.as_completed(future_to_path):
                p = future_to_path[future]
                try:
                    content = future.result()
                    if content is not None:
                        result[p] = content
                except Exception as e:
                    print(f"[CodeManager] 并发读取 {p} 异常: {e}")
        # 按原始输入顺序返回（后续 format_files_for_prompt/预算分配对顺序敏感其实不影响，但保持习惯）
        ordered: Dict[str, str] = {}
        for p in relative_paths:
            if p in result:
                ordered[p] = result[p]
        return ordered

    def format_files_for_prompt(self, relative_paths: List[str],
                                total_budget_chars: int = 25000) -> str:
        """
        将多个文件格式化为适合发送给LLM的字符串。
        均匀预算分配：所有文件都包含（头部+尾部），小文件完整保留，大文件截断。
        读取阶段并发，截断格式化阶段顺序（需要总长度计算）。
        """
        # 阶段 1：并发读文件
        if len(relative_paths) >= 3:
            contents_map = self.read_multiple_files(relative_paths)
            file_contents = [(p, contents_map[p]) for p in relative_paths if p in contents_map]
        else:
            file_contents = []
            for rel_path in relative_paths:
                content = self.read_file(rel_path)
                if content is not None:
                    file_contents.append((rel_path, content))

        n = len(file_contents)
        if n == 0:
            return ""

        overhead_per_file = 130
        total_overhead = n * overhead_per_file
        available = max(6000, total_budget_chars - total_overhead)
        per_file = available // n

        result_chunks: List[str] = []
        used_chars = 0

        for rel_path, content in file_contents:
            if len(content) <= per_file:
                truncated = content
            else:
                head_size = per_file // 2
                tail_size = per_file - head_size - 80
                if head_size < 500:
                    head_size = min(500, per_file // 3)
                    tail_size = per_file - head_size - 80
                head = content[:head_size]
                tail = content[-tail_size:] if tail_size > 200 else ""
                omitted = len(content) - head_size - tail_size
                truncated = f"{head}\n... (中间省略 {omitted} 字符) ...\n{tail}"

            result_chunks.append(f"===== FILE: {rel_path} =====")
            result_chunks.append(truncated)
            result_chunks.append(f"===== END FILE: {rel_path} =====\n")
            used_chars += len(truncated) + overhead_per_file

        return "\n".join(result_chunks)

    # ========== 备份管理 ==========
    def backup_file(self, relative_path: str, tag: str = "") -> Optional[str]:
        """
        备份单个文件到备份目录
        :param relative_path: 相对路径
        :param tag: 备份标签（如迭代编号）
        :return: 备份文件的路径，失败返回None
        """
        src = self.project_root / relative_path
        if not src.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_rel = relative_path.replace(os.sep, "__").replace("/", "__")
        if tag:
            backup_name = f"{timestamp}_{tag}_{safe_rel}"
        else:
            backup_name = f"{timestamp}_{safe_rel}"

        dst = self.backup_dir / backup_name
        try:
            shutil.copy2(src, dst)
            return str(dst)
        except Exception as e:
            print(f"[CodeManager] 备份 {relative_path} 失败: {e}")
            return None

    def backup_files(self, relative_paths: List[str], tag: str = "") -> Dict[str, str]:
        """批量备份文件，返回 {原路径: 备份路径}"""
        results = {}
        for path in relative_paths:
            backup_path = self.backup_file(path, tag)
            if backup_path:
                results[path] = backup_path
        return results

    def list_backups(self) -> List[Path]:
        """列出所有备份文件"""
        return sorted(self.backup_dir.glob("*"))

    def restore_backup(self, backup_path: str, target_relative_path: Optional[str] = None) -> bool:
        """
        从备份恢复文件
        :param backup_path: 备份文件路径
        :param target_relative_path: 恢复到的目标路径（None则从文件名推断）
        :return: 是否成功
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            print(f"[CodeManager] 备份文件不存在: {backup_path}")
            return False

        if target_relative_path is None:
            # 从文件名还原: timestamp_tag__path__to__file.py
            name = backup_file.name
            # 去掉时间戳前缀
            parts = name.split("_", 2)
            if len(parts) >= 3:
                rest = parts[2]
                # 去掉tag（如果有）
                if "__" in rest:
                    # 查找第一次出现在路径分隔符__之前的_不是tag
                    # 简单处理: 找最右的__还原路径
                    restored = rest.replace("__", os.sep)
                    target_relative_path = restored
                else:
                    # 可能只是文件名
                    target_relative_path = rest

        if not target_relative_path:
            print("[CodeManager] 无法推断恢复目标路径")
            return False

        dst = self.project_root / target_relative_path
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_file, dst)
            print(f"[CodeManager] 已恢复: {backup_file.name} -> {target_relative_path}")
            return True
        except Exception as e:
            print(f"[CodeManager] 恢复失败: {e}")
            return False

    # ========== 文件写入/修改 ==========
    def write_file(self, relative_path: str, content: str,
                   create_backup: bool = True, tag: str = "") -> Tuple[bool, str]:
        """
        写入文件（全量覆盖），.py 文件执行写入前 AST 校验
        :param relative_path: 相对路径
        :param content: 新内容
        :param create_backup: 是否创建备份
        :param tag: 备份标签
        :return: (是否成功, 信息)
        """
        filepath = self.project_root / relative_path

        # PLE 保护：守护进程运行时禁止修改核心引擎文件
        if self._is_protected(relative_path):
            return False, f"[PLE保护] 拒绝修改核心引擎 {relative_path}"

        # 写入前 AST 预校验：.py 文件必须能被 ast.parse
        if relative_path.endswith(".py"):
            ok, err = self._preflight_ast(content, relative_path)
            if not ok:
                # ★ 尝试自动修复
                fixed = self._auto_fix_syntax(content, err)
                if fixed and fixed != content:
                    ok2, err2 = self._preflight_ast(fixed, relative_path)
                    if ok2:
                        print(f"      ✅ [AST自动修复] 修复成功")
                        content = fixed
                    else:
                        print(f"      ❌ [AST自动修复] 修复失败: {err2}")
                        return False, f"[AST拦截] {err2}"
                else:
                    return False, f"[AST拦截] {err}"

        # 备份旧文件
        backup_path = None
        if create_backup and self.auto_backup and filepath.exists():
            backup_path = self.backup_file(relative_path, tag)

        try:
            # 确保目录存在
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            # 记录修改历史
            self._modification_history.append({
                "time": datetime.now().isoformat(),
                "action": "write",
                "file": relative_path,
                "backup": backup_path,
                "tag": tag,
            })

            msg = f"写入成功: {relative_path}"
            if backup_path:
                msg += f" (备份: {Path(backup_path).name})"
            return True, msg
        except Exception as e:
            return False, f"写入失败: {relative_path} - {e}"

    @staticmethod
    def _preflight_ast(content: str, filename: str = "<string>") -> Tuple[bool, str]:
        """写入前 AST 预校验
        检查：SyntaxError + 不可达代码 + __init__ 中 self 属性先用后定义
        """
        try:
            tree = ast.parse(content, filename=filename)
        except SyntaxError as e:
            return False, f"SyntaxError: 行{e.lineno}列{e.offset}: {e.msg}"
        except Exception as e:
            return False, f"解析异常: {type(e).__name__}: {e}"

        errors = []

        # ── 检查1：不可达代码（return/break/continue/raise 后的同级语句）──
        for node in ast.walk(tree):
            for body_attr in ('body', 'orelse', 'finalbody'):
                body = getattr(node, body_attr, None)
                if not isinstance(body, list) or len(body) < 2:
                    continue
                for i, stmt in enumerate(body[:-1]):
                    # 检查是否是终止语句
                    is_terminator = isinstance(stmt, (
                        ast.Return, ast.Break, ast.Continue, ast.Raise,
                    ))
                    if is_terminator:
                        next_stmt = body[i + 1]
                        errors.append(
                            f"不可达代码: 行{next_stmt.lineno} "
                            f"在 {type(stmt).__name__} (行{stmt.lineno}) 之后"
                        )

        # ── 检查2：__init__ 中 self.xxx 先用后定义 ──
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != '__init__':
                continue
            # 收集 self.xxx = ... 的赋值行号
            assigned = {}  # attr_name -> first_assign_line
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if (isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == 'self'):
                            attr = target.attr
                            if attr not in assigned:
                                assigned[attr] = sub.lineno

            # 检查 self.xxx 引用是否在赋值之前
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == 'self':
                    attr = sub.attr
                    if attr in assigned:
                        # 检查是否有引用在赋值之前
                        ref_line = sub.lineno
                        assign_line = assigned[attr]
                        if ref_line < assign_line:
                            errors.append(
                                f"先用后定义: self.{attr} 在行{ref_line} 使用，"
                                f"但在行{assign_line} 才赋值"
                            )

        if errors:
            return False, "AST质量检查失败:\n" + "\n".join(f"  {e}" for e in errors)
        return True, "OK"

    @staticmethod
    def _align_indent(original_block: str, new_str: str) -> str:
        """
        把 new_str 的整体缩进对齐到 original_block 的公共最小缩进。
        场景：模型生成的 new_str 通常是 0 缩进（顶格写），但原代码块是缩进的
              → 把 new_str 的每一行都补上原块的公共最小缩进空格数。
        如果 original_block 本身就是顶格的，原样返回 new_str。
        """
        if not original_block or not new_str:
            return new_str

        orig_lines = original_block.split("\n")
        orig_nonempty = [l for l in orig_lines if l.strip() != ""]
        if not orig_nonempty:
            return new_str

        # 计算原块的公共最小缩进（空格+tab）
        def _indent(s: str) -> int:
            n = 0
            for ch in s:
                if ch in (" ", "\t"):
                    n += 1
                else:
                    break
            return n

        min_indent = min(_indent(l) for l in orig_nonempty)
        if min_indent == 0:
            return new_str

        # 计算 new_str 的公共最小缩进
        new_lines = new_str.split("\n")
        new_nonempty = [l for l in new_lines if l.strip() != ""]
        if not new_nonempty:
            return new_str
        new_min_indent = min(_indent(l) for l in new_nonempty)

        # 如果 new 已经缩进量 >= 目标（通常不会），直接返回
        if new_min_indent >= min_indent:
            return new_str

        # 需要补的缩进量
        add_indent = min_indent - new_min_indent
        indent_prefix = " " * add_indent

        # 对 new_str 的每一行补缩进（空行不补）
        aligned = []
        for l in new_lines:
            if l.strip() == "":
                aligned.append("")
            else:
                aligned.append(indent_prefix + l)
        return "\n".join(aligned)

    def insert_method(self, relative_path: str, class_name: str,
                      method_code: str, create_backup: bool = True,
                      tag: str = "") -> Tuple[bool, str]:
        """★ AST-based 方法插入：无需 old_str 精确匹配

        用 AST 找到目标类的末尾行，自动计算缩进，将新方法插入。
        大幅降低 14B 模型的 patch 失败率（模型只需生成方法代码，不需要复制旧代码）。

        :param relative_path: 目标文件相对路径
        :param class_name: 要插入方法的类名
        :param method_code: 新方法的完整 Python 代码（含 def 行，不含类级缩进）
        :return: (success, info)
        """
        import ast as _ast

        # PLE 保护
        if self._is_protected(relative_path):
            return False, f"[PLE保护] 拒绝修改核心引擎 {relative_path}"

        current_content = self.read_file(relative_path)
        if current_content is None:
            return False, f"文件不存在: {relative_path}"

        # 备份
        backup_path = None
        if create_backup and self.auto_backup:
            backup_path = self.backup_file(relative_path, tag or "insert_method")

        try:
            tree = _ast.parse(current_content)
        except SyntaxError as e:
            return False, f"目标文件语法错误，无法解析: {e}"

        # 找到目标类
        target_class = None
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ClassDef) and node.name == class_name:
                target_class = node
                break

        if target_class is None:
            return False, f"类 {class_name} 未在 {relative_path} 中找到"

        # 检查方法是否已存在
        existing_methods = {
            n.name for n in target_class.body
            if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
        }
        # 从 method_code 中提取方法名
        try:
            method_tree = _ast.parse(method_code)
            method_def = None
            for n in _ast.walk(method_tree):
                if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    method_def = n
                    break
            if method_def is None:
                return False, "method_code 中未找到函数定义"
            if method_def.name in existing_methods:
                return False, f"方法 {method_def.name} 已存在于 {class_name} 中"
        except SyntaxError as e:
            return False, f"method_code 语法错误: {e}"

        # 计算插入位置和缩进
        lines = current_content.split('\n')
        # 类的最后一行（end_lineno 是 1-based）
        insert_line = target_class.end_lineno  # 1-based，在 end_lineno 行之后插入

        # 计算类的缩进级别（从类定义行获取）
        class_def_line = lines[target_class.lineno - 1]  # 0-based
        class_indent = len(class_def_line) - len(class_def_line.lstrip())
        method_indent = ' ' * (class_indent + 4)

        # 缩进 method_code
        indented_lines = []
        for line in method_code.split('\n'):
            if line.strip() == '':
                indented_lines.append('')
            else:
                indented_lines.append(method_indent + line)
        indented_method = '\n'.join(indented_lines)

        # 插入到类的最后一行之前（end_lineno 通常是 class body 的最后一行）
        # 我们在 end_lineno 行（0-based: end_lineno-1）之后插入
        insert_idx = insert_line  # 0-based index
        lines.insert(insert_idx, indented_method)

        new_content = '\n'.join(lines)

        # 语法验证
        try:
            _ast.parse(new_content)
        except SyntaxError as e:
            return False, f"插入后语法验证失败: {e}"

        # 写入文件
        ok, info = self.write_file(relative_path, new_content, create_backup=False, tag=tag)
        if ok:
            return True, f"已插入方法 {method_def.name} 到 {class_name} (行 {insert_line + 1})"
        return False, info

    def apply_patch(self, relative_path: str, old_str: str, new_str: str,
                    create_backup: bool = True, tag: str = "") -> Tuple[bool, str]:
        """
        通过"查找替换"方式应用局部修改（带 AST 降级定位）
        主路径：精确匹配 → 行窗口整体strip → 行级逐行strip(缩进容错) → AST节点定位 → difflib提示
        """
        # PLE 保护：守护进程运行时禁止修改核心引擎文件
        if self._is_protected(relative_path):
            return False, f"[PLE保护] 拒绝修改核心引擎 {relative_path}"

        current_content = self.read_file(relative_path)
        if current_content is None:
            return False, f"文件不存在: {relative_path}"

        new_content = None
        original_block = None

        # ── 路径 1：精确字符串匹配（最高优先级，避免误替换） ──
        if old_str in current_content:
            new_content = current_content.replace(old_str, new_str, 1)
            original_block = old_str
        else:
            lines = current_content.split("\n")
            old_lines = old_str.split("\n")
            matched_idx = None
            match_method = ""

            # ── 路径 2：行窗口整体 strip 匹配 ──
            stripped_old = old_str.strip()
            for i in range(len(lines) - len(old_lines) + 1):
                window = "\n".join(lines[i:i + len(old_lines)])
                if window.strip() == stripped_old:
                    matched_idx = i
                    match_method = "整体strip"
                    break

            # ── 路径 3：行级逐行 strip 比较（缩进容错核心） ──
            #    模型生成 old_str 时经常丢掉每行的缩进空格，但实际代码中有缩进
            #    逐行 strip 后比较，同时允许 1 行的边界错位（首尾多/少空行）
            if matched_idx is None:
                old_lines_stripped = [l.rstrip() for l in old_lines if l.strip() != ""]
                # 允许 old_str 首末多带或少带空行
                if len(old_lines_stripped) == 0:
                    old_lines_stripped = [l.strip() for l in old_lines]
                # 过滤空行后的最小行数
                min_match_lines = len(old_lines_stripped)
                if min_match_lines < 1:
                    min_match_lines = max(1, len(old_lines) - 2)
                for i in range(len(lines) - len(old_lines) + 2):
                    # 取窗口：允许窗口比 old_lines 略宽 1 行（首末空行差异）
                    window_lines = lines[i:i + len(old_lines)]
                    win_stripped = [l.rstrip() for l in window_lines if l.strip() != ""]
                    # 核心：逐行 strip 比较（缩进不敏感）
                    if len(win_stripped) == len(old_lines_stripped):
                        ok_all = True
                        for a, b in zip(win_stripped, old_lines_stripped):
                            if a.strip() != b.strip():
                                ok_all = False
                                break
                        if ok_all:
                            matched_idx = i
                            match_method = "逐行strip"
                            break
                    # 也尝试窗口宽度 +1 / -1 的容忍匹配
                    for expand in (-1, 1):
                        wl = len(old_lines) + expand
                        if wl < 1 or i + wl > len(lines):
                            continue
                        w_lines = lines[i:i + wl]
                        w_strip = [l.rstrip() for l in w_lines if l.strip() != ""]
                        if len(w_strip) == len(old_lines_stripped):
                            ok_all = True
                            for a, b in zip(w_strip, old_lines_stripped):
                                if a.strip() != b.strip():
                                    ok_all = False
                                    break
                            if ok_all:
                                matched_idx = i
                                # 用实际匹配的窗口宽度来构造 new_content
                                old_lines = w_lines
                                match_method = f"逐行strip(expand={expand})"
                                break
                    if match_method:
                        break

            if matched_idx is not None:
                end_idx = matched_idx + len(old_lines)
                original_block = "\n".join(lines[matched_idx:end_idx])
                # ── 关键修复：保持新内容的缩进与原窗口一致 ──
                #    如果原窗口每行都有 4 空格缩进，而 new_str 里是 0 缩进，
                #    那就把 new_str 的每一行都补上 4 空格
                new_str = self._align_indent(original_block, new_str)
                new_content = (
                    "\n".join(lines[:matched_idx])
                    + "\n" + new_str + "\n"
                    + "\n".join(lines[end_idx:])
                )
            elif relative_path.endswith(".py"):
                # ── 路径 4：AST 节点定位 ──
                ast_result = self._ast_locate_and_replace(current_content, old_str, new_str)
                if ast_result is not None:
                    new_content, original_block = ast_result

            if new_content is None:
                # 最终兜底：difflib 相似匹配（仅提示，不自动应用）
                suggestion = self._find_similar_block(current_content, old_str)
                msg = f"在 {relative_path} 中找不到要替换的内容。\n"
                msg += f"要查找的内容（前100字）:\n{old_str[:100]}\n"
                if suggestion:
                    msg += f"最相似的现有内容（前100字）:\n{suggestion[:100]}"
                return False, msg

        # patch 后的 AST 预校验：.py 文件必须能被 ast.parse
        if relative_path.endswith(".py"):
            ok, err = self._preflight_ast(new_content, relative_path)
            if not ok:
                return False, f"[AST拦截] 拒绝 patch {relative_path}: {err}"

        # 执行写入
        success, info = self.write_file(relative_path, new_content, create_backup, tag)
        if success:
            self._modification_history[-1]["action"] = "patch"
            self._modification_history[-1]["old_str_preview"] = (original_block or "")[:80]
            self._modification_history[-1]["new_str_preview"] = new_str[:80]
        return success, info

    def _ast_locate_and_replace(self, content: str, old_str: str, new_str: str) -> Optional[Tuple[str, str]]:
        """
        AST 降级定位：从 old_str 中提取函数/类名，在源文件中定位该节点的源码范围，
        用 new_str 替换整个节点（含签名到结尾）。
        :return: (new_content, original_block) 或 None（定位失败）
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        # 从 old_str 提取候选名：def xxx / class Xxx / async def xxx
        import re as _re
        names = _re.findall(r'(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)', old_str)
        if not names:
            return None

        # 在 AST 中查找顶级节点（函数/类定义）
        lines = content.split("\n")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in names:
                    # 使用 end_lineno / end_col_offset 获取完整节点范围
                    start_line = node.lineno - 1  # 转为 0-indexed
                    end_line = getattr(node, 'end_lineno', node.lineno)  # 某些旧 Python 无 end_lineno
                    if end_line is None:
                        # 手动扫描到同级下一个顶级定义
                        end_line = self._find_node_end(tree, node, lines)
                    end_line_exclusive = end_line  # 切片时不含

                    # 收集原始块（保持缩进一致）
                    original_block = "\n".join(lines[start_line:end_line_exclusive])

                    # 替换
                    new_lines = lines[:start_line] + [new_str.rstrip("\n")] + lines[end_line_exclusive:]
                    new_content = "\n".join(new_lines)

                    print(f"[AST降级] 定位到 {type(node).__name__} '{node.name}' (行{start_line+1}-{end_line})，替换成功")
                    return new_content, original_block

        return None

    @staticmethod
    def _find_node_end(tree: ast.AST, target: ast.AST, lines: List[str]) -> int:
        """当 end_lineno 不可用时，找同级下一个顶级节点或文件末尾"""
        siblings = []
        for child in ast.iter_child_nodes(tree):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                siblings.append(child)
        target_idx = None
        for i, s in enumerate(siblings):
            if s is target:
                target_idx = i
                break
        if target_idx is not None and target_idx + 1 < len(siblings):
            return siblings[target_idx + 1].lineno - 1
        return len(lines)

    def _find_similar_block(self, content: str, target: str) -> Optional[str]:
        """在内容中找到与target最相似的块"""
        target_lines = target.strip().split("\n")
        content_lines = content.split("\n")
        window_size = len(target_lines)
        best_ratio = 0.0
        best_block = None

        for i in range(len(content_lines) - window_size + 1):
            window = "\n".join(content_lines[i:i + window_size])
            ratio = difflib.SequenceMatcher(None, window.strip(), target.strip()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_block = window

        if best_ratio > 0.6:
            return best_block
        return None

    # ========== 代码验证 ==========
    def check_python_syntax(self, relative_path: str) -> Tuple[bool, str]:
        """
        检查Python文件语法是否正确
        :return: (是否正确, 错误信息)
        """
        content = self.read_file(relative_path)
        if content is None:
            return False, "文件不存在"
        try:
            ast.parse(content, filename=relative_path)
            return True, "语法正确"
        except SyntaxError as e:
            return False, f"语法错误 - 行{e.lineno}列{e.offset}: {e.msg}"
        except Exception as e:
            return False, f"解析异常: {type(e).__name__}: {e}"

    def test_python_import(self, relative_path: str) -> Tuple[bool, str]:
        """
        测试Python模块是否可以成功import（在子进程中运行，避免污染当前进程）
        :return: (是否成功, 错误信息)
        """
        filepath = self.project_root / relative_path
        if not filepath.exists():
            return False, "文件不存在"

        # 不直接import main.py等入口文件
        if filepath.name == "main.py":
            return True, "跳过入口文件的import测试"

        try:
            # 用子进程执行语法+导入检查
            code = f"""
import sys
sys.path.insert(0, r'{self.project_root}')
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "{filepath.stem}", r"{filepath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("OK")
except Exception as e:
    print(f"FAIL:{{type(e).__name__}}: {{e}}")
"""
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.project_root),
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if stdout.endswith("OK"):
                return True, "导入测试通过"
            else:
                err = stdout if stdout.startswith("FAIL:") else (stderr or stdout or "未知错误")
                return False, f"导入失败: {err[:500]}"
        except subprocess.TimeoutExpired:
            return False, "导入超时（>30秒）"
        except Exception as e:
            return False, f"测试异常: {type(e).__name__}: {e}"

    def verify_module_import(self, relative_path: str) -> Tuple[bool, str]:
        """
        P0-L1 冒烟 import 验证（轻量版，优先 importlib.reload 原地加载，失败时降级为子进程 test_python_import）。
        保证修改过的模块至少能加载 — 拦截 30-40% 语法正确但顶层 NameError/ImportError 之类"表面成功"。
        :return: (是否成功, 说明)
        """
        filepath = self.project_root / relative_path
        if not filepath.exists():
            return False, f"文件不存在: {relative_path}"
        if filepath.suffix != ".py":
            return True, "非 .py 文件跳过 import 验证"
        if filepath.name == "main.py":
            return True, "入口文件 main.py 不做 import 验证（避免启动 CLI）"

        # ── 优先尝试: importlib.util 原地加载（不 spawn 子进程，省 100ms 量级）──
        try:
            import importlib.util as _ilu
            module_name = filepath.stem
            # 避免与已注册的包名冲突，加 _smoke_ 前缀
            safe_name = f"_smoke_{module_name}_{abs(hash(str(filepath))) % 100000}"
            spec = _ilu.spec_from_file_location(safe_name, str(filepath))
            if spec is None or spec.loader is None:
                # spec 构造失败，降级
                return self.test_python_import(relative_path)
            mod = _ilu.module_from_spec(spec)
            sys.modules[safe_name] = mod
            try:
                spec.loader.exec_module(mod)
                return True, "import 加载成功（L1 冒烟）"
            finally:
                # 无论成功与否，从 sys.modules 清掉（避免污染后续迭代的模块缓存）
                sys.modules.pop(safe_name, None)
        except Exception as e:
            # 失败 → 用子进程再确认一次（避免当前进程残留旧模块状态导致误报）
            sub_ok, sub_msg = self.test_python_import(relative_path)
            if sub_ok:
                return True, f"{sub_msg}（原地加载失败但子进程通过: {type(e).__name__}）"
            return False, f"L1 import 失败: {type(e).__name__}: {e[:400] if hasattr(e,'__getitem__') else str(e)[:400]}"

    # ========== 并发验证 ==========
    def validate_files_parallel(self, relative_paths: List[str]) -> Dict[str, Tuple[bool, str]]:
        """
        并发验证多个文件：语法检查（CPU bound）+ 导入测试（子进程 IO bound）。
        语法和导入针对同一个文件串行执行（有依赖关系），不同文件之间完全并发。
        :return: {相对路径: (语法ok, 语法信息), (导入ok, 导入信息)} 的压缩版：
                 {相对路径: (True/false, 合并信息)}  —— 失败优先
        """
        if not relative_paths:
            return {}

        def _validate_one(rel: str) -> Tuple[str, Tuple[bool, str]]:
            syntax_ok, syntax_msg = self.check_python_syntax(rel)
            if not syntax_ok:
                return (rel, (False, f"语法错误 → {syntax_msg}"))
            if not rel.endswith(".py"):
                return (rel, (True, f"语法正确（非Python，跳过导入测试）"))
            import_ok, import_msg = self.verify_module_import(rel)
            if not import_ok:
                return (rel, (False, f"语法正确 | {import_msg}"))
            return (rel, (True, f"语法正确 | {import_msg}"))

        results: Dict[str, Tuple[bool, str]] = {}
        # verify_module_import 优先走进程内 importlib，失败才启子进程；用 _IO_WORKERS（避免进程风暴）
        workers = min(max(2, _CPU_WORKERS * 2), len(relative_paths))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_validate_one, p) for p in relative_paths]
            for future in concurrent.futures.as_completed(futures):
                try:
                    rel, res = future.result()
                    results[rel] = res
                except Exception as e:
                    # 兜底：标记失败但不阻塞整体
                    results[relative_paths[0]] = (False, f"并发验证异常: {e}")

        # 保持输入顺序返回
        ordered: Dict[str, Tuple[bool, str]] = {}
        for p in relative_paths:
            if p in results:
                ordered[p] = results[p]
        return ordered

    def backup_files_parallel(self, relative_paths: List[str], tag: str = "") -> Dict[str, Optional[str]]:
        """并发备份多个文件（纯 IO bound）"""
        if not relative_paths:
            return {}
        if len(relative_paths) == 1:
            return {relative_paths[0]: self.backup_file(relative_paths[0], tag)}

        results: Dict[str, Optional[str]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(_IO_WORKERS, len(relative_paths))) as pool:
            future_to_path = {pool.submit(self.backup_file, p, tag): p for p in relative_paths}
            for future in concurrent.futures.as_completed(future_to_path):
                p = future_to_path[future]
                try:
                    results[p] = future.result()
                except Exception as e:
                    print(f"[CodeManager] 并发备份 {p} 异常: {e}")
                    results[p] = None
        return results

    def generate_diff(self, relative_path: str, backup_path: str) -> Optional[str]:
        """生成修改前后的diff"""
        new_content = self.read_file(relative_path)
        backup_file = Path(backup_path)
        if new_content is None or not backup_file.exists():
            return None
        with open(backup_file, "r", encoding="utf-8") as f:
            old_content = f.read()

        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{relative_path} (备份)",
            tofile=f"b/{relative_path} (当前)",
            lineterm="",
        )
        return "".join(diff)

    # ========== 历史和辅助 ==========
    def get_modification_history(self, limit: int = 20) -> List[Dict]:
        """获取最近的修改历史"""
        return self._modification_history[-limit:]

    def get_file_info(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """获取文件基本信息"""
        filepath = self.project_root / relative_path
        if not filepath.exists():
            return None
        stat = filepath.stat()
        content = self.read_file(relative_path)
        lines = content.count("\n") + 1 if content else 0
        chars = len(content) if content else 0
        return {
            "path": relative_path,
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "lines": lines,
            "chars": chars,
        }
    def get_file_hash(self, file_path: str) -> str:
        """返回指定文件的 sha256 哈希值"""
        import hashlib
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()
        except Exception as e:
            print(f"[CodeManager] 获取文件哈希失败 {file_path}: {e}")
            return ''
    def cleanup_old_backups(self, max_age_days: int = 30) -> None:
        """清理超过指定天数的旧备份

        :param max_age_days: 备份保留的最大天数，默认为 30 天
        """
        import os
        from datetime import datetime, timedelta

        try:
            backup_dir = self.backup_dir
            if not os.path.exists(backup_dir):
                self.logger.warning('Backup directory does not exist: %s', backup_dir)
                return

            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            for filename in os.listdir(backup_dir):
                file_path = os.path.join(backup_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_modified_time < cutoff_date:
                            os.remove(file_path)
                            self.logger.info('Deleted old backup: %s', file_path)
                    except Exception as e:
                        self.logger.error('Failed to delete backup %s: %s', file_path, str(e))
        except Exception as e:
            self.logger.error('Error during cleanup_old_backups: %s', str(e))
