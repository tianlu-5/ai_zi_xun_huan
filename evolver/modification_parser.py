# -*- coding: utf-8 -*-
"""
修改建议解析器 - 解析豆包返回的结构化修改建议
支持多种输出格式：全量文件替换、局部patch、新增文件等
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict


# 全局 JSON 失败回调：由 SelfEvolver 初始化时注入，把失败事件喂回 error_reflection 形成闭环
# 签名: fn(iteration_id, category, reason, issues, objective)
_JSON_FAILURE_CALLBACK: Optional[Callable[[int, str, str, List[str], str], None]] = None


def register_json_failure_callback(fn: Callable[[int, str, str, List[str], str], None]) -> None:
    """注册失败事件回调（由 self_evolver 初始化时调用，把 JSON_PARSE_FAIL 事件同步到 error_reflection）"""
    global _JSON_FAILURE_CALLBACK
    _JSON_FAILURE_CALLBACK = fn


def _log_json_failure(iteration_id: int, raw_response: str, cleaned: str,
                      reason: str, objective: str = "") -> None:
    """
    JSON 失败日志收集器 — 原样保存模型输出 + 回调 error_reflection 形成 Prompt 闭环。
    写入 logs/json_fail_log/，失败不影响主流程。
    """
    # 1) 落盘原始响应供人工分析
    try:
        from datetime import datetime
        log_dir = Path("logs/json_fail_log")
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        entry = {
            "timestamp": ts,
            "iteration_id": iteration_id,
            "objective": objective[:200],
            "reason": reason,
            "raw_response": raw_response[:4000],
            "cleaned_head": cleaned[:500],
        }
        fname = log_dir / f"fail_{ts}_iter{iteration_id}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 2) 回调 error_reflection：记录 JSON_PARSE_FAIL 事件，参与下一轮 Prompt 强化
    try:
        if _JSON_FAILURE_CALLBACK:
            issues = [reason[:120]]
            # 追加原始响应前 80 字的特征说明，方便模型针对性规避
            head_feature = raw_response[:80].replace("\n", " ")
            if head_feature and not head_feature.startswith("{"):
                issues.append(f"原始开头非JSON: {head_feature[:60]}")
            _JSON_FAILURE_CALLBACK(
                iteration_id,
                "JSON_PARSE_FAIL",
                reason,
                issues,
                objective,
            )
    except Exception:
        pass


@dataclass
class SingleModification:
    """单文件修改动作"""
    # 修改类型: "replace_all" | "patch" | "new_file" | "delete_file" | "rename_file" | "insert_method"
    action: str = "replace_all"

    # 目标文件相对路径
    target_file: str = ""

    # replace_all: 全新的完整文件内容
    new_content: str = ""

    # patch: 要被替换的旧内容片段
    old_str: str = ""
    # patch: 替换后的新内容片段
    new_str: str = ""

    # insert_method: 目标类名
    class_name: str = ""
    # insert_method: 新方法的完整代码（含 def 行，不含类级缩进）
    method_code: str = ""

    # new_file: 可选，是否覆盖已存在文件
    overwrite: bool = False

    # rename_file: 新文件名
    new_filename: str = ""

    # 修改说明（供人阅读）
    reason: str = ""

    # 优先级，越小越先执行
    priority: int = 100

    def is_valid(self) -> Tuple[bool, str]:
        """验证修改参数是否完整 + 硬校验三引号/markdown/空字符串"""
        if not self.target_file:
            return False, "缺少 target_file"
        if self.action == "replace_all":
            if not self.new_content:
                return False, "replace_all 需要 new_content"
        elif self.action == "patch":
            if not self.old_str or not self.new_str:
                return False, "patch 需要 old_str 和 new_str"
        elif self.action == "insert_method":
            if not self.class_name:
                return False, "insert_method 需要 class_name"
            if not self.method_code:
                return False, "insert_method 需要 method_code"
        elif self.action == "new_file":
            if not self.new_content:
                return False, "new_file 需要 new_content"
        elif self.action == "rename_file":
            if not self.new_filename:
                return False, "rename_file 需要 new_filename"
        elif self.action == "delete_file":
            pass
        else:
            return False, f"未知的 action: {self.action}"

        # ── 解析层硬校验（项3）：比 banner 文本提示更可靠 ──
        # 检查代码字段：禁止 markdown 代码块标记（但允许三引号，因为 Python docstring 合法使用 """）
        _fields_to_check = []
        if self.action == "patch":
            _fields_to_check = [("old_str", self.old_str), ("new_str", self.new_str)]
        elif self.action == "insert_method":
            _fields_to_check = [("method_code", self.method_code)]
        elif self.action in ("replace_all", "new_file"):
            _fields_to_check = [("new_content", self.new_content)]
        elif self.action == "rename_file":
            _fields_to_check = [("new_filename", self.new_filename)]

        for fname, fval in _fields_to_check:
            if not fval:
                continue
            # ★ 修复：允许 old_str/new_str/new_content 中的三引号（Python docstring 合法）
            #   原检查会拒绝所有含 """ 的 new_str，导致 AI 生成的带 docstring 的方法全部被丢弃
            #   三引号禁令只需用于 description/reason 等非代码字段（在 parse() 中单独检查）
            if '```' in fval:
                return False, f"{fname} 含 markdown 代码块标记（禁止）"

        return True, "OK"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SingleModification":
        # ── 字段名兼容映射：小模型常输出变体字段名，统一归一化 ──
        d = dict(data)
        # target_file 别名
        if "target_file" not in d or not d["target_file"]:
            for alias in ("file_path", "file", "path", "target", "filename", "file_name"):
                if alias in d and d[alias]:
                    d["target_file"] = d[alias]
                    break
        # old_str 别名（模型常输出 old_content/find/search/original/code_to_replace）
        if "old_str" not in d or not d["old_str"]:
            for alias in ("old_content", "old_text", "find", "search", "original", "code_to_replace", "old_code", "before"):
                if alias in d and d[alias]:
                    d["old_str"] = d[alias]
                    break
        # new_str 别名（模型常输出 new_content/replace/replacement/new_code/after）
        if "new_str" not in d or not d["new_str"]:
            for alias in ("new_text", "replace", "replacement", "new_code", "after", "replacement_text"):
                if alias in d and d[alias]:
                    d["new_str"] = d[alias]
                    break
        # new_content 别名（用于 new_file / replace_all）
        if "new_content" not in d or not d["new_content"]:
            for alias in ("content", "code", "file_content", "full_content", "new_file_content"):
                if alias in d and d[alias]:
                    d["new_content"] = d[alias]
                    break
        # class_name 别名（用于 insert_method）
        if "class_name" not in d or not d["class_name"]:
            for alias in ("class", "target_class", "classname", "class_name_"):
                if alias in d and d[alias]:
                    d["class_name"] = d[alias]
                    break
        # method_code 别名（用于 insert_method）
        if "method_code" not in d or not d["method_code"]:
            for alias in ("method", "method_source", "function_code", "new_method", "method_body"):
                if alias in d and d[alias]:
                    d["method_code"] = d[alias]
                    break

        # ── 路径规范化：去掉前导 / 和 ./ （模型常输出 "/evolver/xxx.py"）──
        if d.get("target_file"):
            tf = d["target_file"].strip()
            while tf.startswith("/"):
                tf = tf[1:]
            if tf.startswith("./"):
                tf = tf[2:]
            d["target_file"] = tf

        # ── action 智能推断 + 降级容错 ──
        if "action" not in d or not d["action"]:
            if d.get("method_code") and d.get("class_name"):
                d["action"] = "insert_method"
            elif d.get("old_str") and d.get("new_str"):
                d["action"] = "patch"
            elif d.get("new_content"):
                d["action"] = "replace_all"
            else:
                d["action"] = "patch"
         # ★★★ 关键修复：将 Daemon 强制要求的 str_replace 映射为 patch ★★★
        if d.get("action") == "str_replace":
            d["action"] = "patch"
            # 确保 old_str 和 new_str 存在（AI 已经生成，但以防万一做兜底）
            if not d.get("old_str"):
                d["old_str"] = d.get("old") or d.get("original") or ""
            if not d.get("new_str"):
                d["new_str"] = d.get("new") or d.get("replacement") or ""
            # 打印一条日志，方便调试确认映射生效
            print(f"[Parser] 将 str_replace 映射为 patch (文件: {d.get('target_file', '?')})")      
        # 容错 1: new_file 缺 new_content 但有 new_str → 当 patch 用 new_str 填充 new_content
        if d.get("action") == "new_file" and not d.get("new_content") and d.get("new_str"):
            d["new_content"] = d["new_str"]
        # 容错 2: new_file 缺 new_content 但有 old_str → 把 old_str 当文件内容
        if d.get("action") == "new_file" and not d.get("new_content") and d.get("old_str"):
            d["new_content"] = d["old_str"]
        # 容错 3: patch 缺 new_str 但有 new_content → 降级为 replace_all
        if d.get("action") == "patch" and not d.get("new_str") and d.get("new_content"):
            d["action"] = "replace_all"
        # 容错 4: patch 缺 old_str 但有 new_content → 降级为 replace_all
        if d.get("action") == "patch" and not d.get("old_str") and d.get("new_content"):
            d["action"] = "replace_all"
        # 容错 5: replace_all 缺 new_content 但有 new_str → 用 new_str 填充 new_content
        if d.get("action") == "replace_all" and not d.get("new_content") and d.get("new_str"):
            d["new_content"] = d["new_str"]

        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModificationPlan:
    """一批修改计划"""
    iteration_id: int = 0
    description: str = ""
    modifications: List[SingleModification] = field(default_factory=list)
    raw_response: str = ""  # 豆包原始响应，便于调试

    def summary(self) -> str:
        """修改摘要"""
        if not self.modifications:
            return "（无修改建议）"
        lines = [f"=== 修改计划（共{len(self.modifications)}项）==="]
        for i, m in enumerate(self.modifications, 1):
            action_cn = {
                "replace_all": "全量替换",
                "patch": "局部补丁",
                "new_file": "新建文件",
                "delete_file": "删除文件",
                "rename_file": "重命名",
            }.get(m.action, m.action)
            reason_preview = (m.reason[:60] + "...") if len(m.reason) > 60 else m.reason
            lines.append(f"  {i}. [{action_cn}] {m.target_file} - {reason_preview}")
        return "\n".join(lines)


class ModificationParser:
    """解析豆包返回的修改建议"""

    # 期望的JSON输出格式说明（用于构建system prompt）
    EXPECTED_JSON_SCHEMA = """
=== 你必须严格遵守的 JSON 输出协议 ===
1. 你的完整回复只能是一个 JSON 对象，首字符必须是 {{，末字符必须是 }}。
2. 结构只有两个键："description"（字符串，一句话概述修改意图）和 "modifications"（数组）。
3. modifications 数组中的每个元素代表一次修改，支持以下字段：
   - "action"        : 可选，字符串。取值 "patch"（默认）| "replace_all" | "new_file" | "delete_file" | "rename_file" | "insert_method"
   - "target_file"   : 必填，字符串。目标文件的相对路径，例如 "evolver/meta_drive.py"
                       别名也接受："file_path" | "file" | "path"
   - "old_str"       : patch 必填，字符串。目标文件中**连续 ≥2 行的精确原文片段**
                       必须逐字匹配，包括缩进空格和换行；但允许忽略每行首的整体缩进差异
   - "new_str"       : patch / rename_file 必填，字符串。替换后的完整代码块或新文件名
   - "new_content"   : replace_all / new_file 必填，字符串。文件的完整内容
   - "class_name"    : insert_method 必填，字符串。目标类名
   - "method_code"   : insert_method 必填，字符串。新方法的完整 Python 代码（含 def 行和 docstring，
                       不含类级缩进）。系统会自动用 AST 找到类末尾并插入，无需 old_str。
   - "reason"        : 可选，字符串。人类可读的修改原因说明
4. 禁止返回空 modifications 数组：即使现存代码没有缺陷，也必须执行功能扩充
   （补全缺失方法、增加边界校验、优化日志结构等），至少输出 1 条 patch。
5. 修改数不得超过 {max_mod} 项。

=== 合法 insert_method 示例（推荐：新增方法时优先使用）===
{{
  "description": "为 BranchManager 新增 delete_branch 方法",
  "modifications": [
    {{
      "action": "insert_method",
      "target_file": "evolver/branch_manager.py",
      "class_name": "BranchManager",
      "method_code": "def delete_branch(self, name: str) -> bool:\\n    \\\"\\\"\\\"删除指定分支，不存在时返回 False\\\"\\\"\\\"\\n    try:\\n        branches = self.list_all_branches()\\n        if name not in branches:\\n            return False\\n        # 删除逻辑\\n        return True\\n    except Exception as e:\\n        self.logger.error(str(e))\\n        return False",
      "reason": "新增 delete_branch 方法以支持分支删除功能"
    }}
  ]
}}

=== 合法 patch 示例 ===
{{
  "description": "修复 conflict_boundary 中熵差阈值常量的拼写错误",
  "modifications": [
    {{
      "action": "patch",
      "target_file": "evolver/conflict_boundary.py",
      "old_str": "    DEFAULT_ENTROPY_GAP_CRITICAL = 0.42\n    DEFAULT_ENTROPY_GAP_WARNING = 0.25",
      "new_str": "    DEFAULT_ENTROPY_GAP_CRITICAL = 0.45\n    DEFAULT_ENTROPY_GAP_WARNING = 0.28",
      "reason": "原阈值过于敏感，根据最近 200 轮统计，critical 上调到 0.45 可减少 30% 误隔离"
    }}
  ]
}}
"""

    # 硬约束横幅：步骤3 强化——杜绝保守空输出 + 禁三引号/markdown 包裹
    # 注意：字符串内禁止出现三连引号，否则会提前闭合 Python 字符串字面量
    HARD_CONSTRAINT_BANNER = (
        '只输出JSON对象，首字符必须是{，末字符必须是}。'
        '即使现存代码没有缺陷，也必须执行功能扩充，禁止返回空modifications数组；'
        '至少输出1条patch补丁。所有JSON字符串仅允许普通英文双引号，'
        '禁止三引号(三连引号)、禁止用markdown代码块包裹返回结果。'
        'old_str必须完整原样复制源码片段(含全部空格缩进)，不可简写脑补。'
    )

    def __init__(self):
        pass

    def parse(self, llm_response: str, iteration_id: int = 0,
              objective: str = "") -> ModificationPlan:
        """
        解析LLM响应为修改计划（带多级清洗链）
        :param llm_response: LLM返回的原始文本（可能含 <think>、markdown fence、解释文字）
        :param iteration_id: 迭代编号
        :param objective: 本轮演化目标（用于失败日志追溯）
        :return: ModificationPlan
        """
        plan = ModificationPlan(iteration_id=iteration_id, raw_response=llm_response)

        if not llm_response:
            plan.description = "（模型返回空响应）"
            _log_json_failure(iteration_id, "", "", "空响应", objective)
            return plan

        cleaned = self._pre_clean(llm_response)
        data = self._extract_json(cleaned)
        if data is None:
            plan.description = "（模型未返回有效JSON，修改计划为空）"
            print("[Parser] ⚠️ 模型响应不是有效JSON，已拒绝以防止产出垃圾文件")
            print(f"  清洗后前300字: {cleaned[:300]}...")
            _log_json_failure(iteration_id, llm_response, cleaned,
                              "四层容错全部失败：无法提取JSON", objective)
            return plan

        try:
            plan.description = str(data.get("description", ""))

            mods_data = data.get("modifications", [])
            if not isinstance(mods_data, list):
                if isinstance(mods_data, dict):
                    mods_data = [mods_data]
                else:
                    mods_data = []

            for mod_data in mods_data:
                mod = SingleModification.from_dict(mod_data)
                valid, reason = mod.is_valid()
                if valid:
                    plan.modifications.append(mod)
                else:
                    print(f"[Parser] 跳过无效修改项: {reason} - {mod_data}")
        except Exception as e:
            print(f"[Parser] 解析JSON结构失败: {e}")
            _log_json_failure(iteration_id, llm_response, cleaned,
                              f"JSON结构解析异常: {type(e).__name__}: {e}", objective)

        return plan

    @staticmethod
    def _pre_clean(text: str) -> str:
        """
        步骤2重构后的清洗顺序（markdown → 三引号 → 推理块 → JSON主体 → 残字符）：
          1. 最先清除 markdown 代码块标记（json fence）
          2. 替换全部三引号（三连双引号/三连单引号）为普通双引号
          3. 剥离推理块 think/reasoning/R1 标签
          4. 抓取全文第一个左大括号到最后一个右大括号的完整 JSON 主体
          5. 清理 BOM、零宽字符、首尾空白
        清洗后再交给 _extract_json 做六层容错解析。
        """
        import re as _re

        if not text or not text.strip():
            return ""

        # ── 步骤 1（最优先）：移除 markdown 代码块标记 ──
        #   必须在三引号替换之前：fence 里的 ``` 会和后续清洗冲突
        # 1a. 完整 fence: ```json\n...\n``` → 取中间内容
        fence_match = _re.search(r'```(?:json|JSON)?\s*\n?(.*?)\n?\s*```', text, _re.DOTALL | _re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1)
        # 1b. 残余的开头 ```json 或 ``` 标记（fence 不闭合的情况）
        text = _re.sub(r'^```(?:json|JSON)?\s*\n?', '', text, flags=_re.MULTILINE | _re.IGNORECASE)
        text = _re.sub(r'\n?```\s*$', '', text, flags=_re.MULTILINE | _re.IGNORECASE)
        # 1c. 行内残留的 ``` 单独标记
        text = text.replace('```', '')

        # ── 步骤 2：替换全部三引号 → 普通双引号 ──
        #   qwen2.5-coder 写多行代码时本能用三引号包裹字符串，这不是合法 JSON
        #   markdown 已清除，此处只处理裸三引号（三连双引号/三连单引号）
        text = text.replace('"""', '"').replace("'''", '"')

        # ── 步骤 3：剥离推理块（deepseek-r1 / R1 系列标准推理块）──
        text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
        for tag in ('thinking', 'reasoning', 'Thought', 'reasoning_process', 'analysis', 'explain'):
            text = _re.sub(fr'<{tag}>.*?</{tag}>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
        # R1 模型有时输出 <RichMediaReference>...</RichMediaReference> 包裹推理
        text = _re.sub(r'<RichMediaReference>.*?</RichMediaReference>', '', text,
                       flags=_re.DOTALL | _re.IGNORECASE)

        # ── 步骤 4：抓取全文第一个 { 到最后一个 } 的完整 JSON 主体 ──
        #   避免解析器只截取很短一段，把 modifications 数组完整保留
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]
        else:
            # 如果没有 {} 但有 []，则抓取数组根节点
            first_bracket = text.find('[')
            last_bracket = text.rfind(']')
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                text = text[first_bracket:last_bracket + 1]

        # ── 步骤 5：清理 BOM、零宽字符、首尾空白 ──
        text = text.replace('\ufeff', '').replace('\u200b', '')
        return text.strip()

    # 字段别名映射表：model 容易想当然地写错字段名，这里统一做归一化
    FIELD_ALIASES = {
        # 顶层字段别名
        "summary": "description",
        "explanation": "description",
        "reasoning": "description",
        "thought": "description",
        "plan": "description",
        "changes": "modifications",
        "modification": "modifications",
        "edits": "modifications",
        "patches": "modifications",
        "actions": "modifications",
        "updates": "modifications",
        "modified_files": "modifications",
        # modification 子项字段别名
        "path": "target_file",
        "file_path": "target_file",
        "file": "target_file",
        "filename": "target_file",
        "target": "target_file",
        "code_before": "old_str",
        "before": "old_str",
        "original": "old_str",
        "old": "old_str",
        "old_code": "old_str",
        "code_after": "new_str",
        "after": "new_str",
        "modified": "new_str",
        "new": "new_str",
        "new_code": "new_str",
        "full_content": "new_content",
        "content": "new_content",
        "replacement": "new_content",
        "description_of_change": "reason",
        "why": "reason",
        "note": "reason",
    }

    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        """递归地对 dict 做字段别名映射，保证后续 schema 验证统一"""
        if isinstance(data, dict):
            out = {}
            for k, v in data.items():
                nk = cls.FIELD_ALIASES.get(k if isinstance(k, str) else str(k), k)
                out[nk] = cls._normalize_fields(v)
            return out
        if isinstance(data, list):
            return [cls._normalize_fields(x) for x in data]
        return data

    def _extract_json(self, text: str) -> Optional[Dict]:
        """从可能含噪声的文本中稳健提取 JSON（6 层容错）"""
        import re as _re

        def _try_parse(s: str) -> Optional[Any]:
            """单行解析，成功返回对象，失败返回 None，含修复尝试"""
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                try:
                    return json.loads(self._repair_json(s))
                except json.JSONDecodeError:
                    return None

        # 路径 1：直接解析（最快，覆盖 80% 正常响应）
        data = _try_parse(text)
        if data is not None:
            if isinstance(data, list):
                return self._normalize_fields({"description": "", "modifications": data})
            return self._normalize_fields(data)

        # 路径 2：括号匹配法——找 JSON 对象边界，精确截取
        brace_match = self._balanced_braces(text)
        if brace_match:
            d = _try_parse(brace_match)
            if isinstance(d, dict):
                return self._normalize_fields(d)

        # 路径 3：数组根节点（有些模型返回 [...] 而非 {...}）
        arr_match = self._balanced_brackets(text)
        if arr_match:
            a = _try_parse(arr_match)
            if isinstance(a, list):
                return self._normalize_fields({"description": "", "modifications": a})
            if isinstance(a, dict):
                return self._normalize_fields({"description": "", "modifications": [a]})

        # 路径 4：正则找 {"description":...} 模式
        pattern = r'\{\s*"description"\s*:\s*".*?"\s*,\s*"modifications"\s*:\s*\[.*?\]\s*\}'
        m = _re.search(pattern, text, _re.DOTALL)
        if m:
            d = _try_parse(m.group(0))
            if isinstance(d, dict):
                return self._normalize_fields(d)

        # 路径 5：截断 JSON 自动补全闭合括号（num_predict 用尽截断场景）
        # 从最后一个 '{' 位置开始，尝试闭合缺失的 } / ] 配对
        closed = self._auto_close_truncated_json(text)
        if closed:
            d = _try_parse(closed)
            if isinstance(d, dict):
                return self._normalize_fields(d)
            if isinstance(d, list):
                return self._normalize_fields({"description": "", "modifications": d})

        # 路径 6：找到任意裸 JSON 对象/数组片段，字段别名映射后强行拼装
        last_resort = self._extract_last_resort_object(text)
        if last_resort is not None:
            return self._normalize_fields(last_resort)

        return None

    @staticmethod
    def _auto_close_truncated_json(text: str) -> Optional[str]:
        """
        ★ 硬改4：强化截断 JSON 自动修复
        1. 优先提取首个完整合法 JSON 实体（找到第一个 { 到第一个完整闭合 } ）
        2. 如果没有完整闭合，检测未闭合的引号/括号并自动补齐
        3. 丢弃后面残缺片段
        """
        brace_idx = text.find('{')
        bracket_idx = text.find('[')
        if brace_idx == -1 and bracket_idx == -1:
            return None
        start = brace_idx if (bracket_idx == -1 or (brace_idx != -1 and brace_idx < bracket_idx)) else bracket_idx
        snippet = text[start:]

        # ★ 步骤 1：优先尝试提取首个完整合法 JSON 实体
        # 扫描找第一个深度归零的 } 或 ]，截取完整实体
        depth_brace = 0
        depth_bracket = 0
        in_str = False
        escape = False
        first_complete_end = -1
        for i, ch in enumerate(snippet):
            if escape:
                escape = False; continue
            if ch == '\\':
                escape = True; continue
            if ch == '"':
                in_str = not in_str; continue
            if in_str:
                continue
            if ch == '{': depth_brace += 1
            elif ch == '}':
                depth_brace -= 1
                if depth_brace == 0 and depth_bracket == 0:
                    first_complete_end = i
                    break
            elif ch == '[': depth_bracket += 1
            elif ch == ']':
                depth_bracket -= 1
                if depth_bracket == 0 and depth_brace == 0:
                    first_complete_end = i
                    break

        if first_complete_end != -1:
            # 找到完整实体，直接返回（丢弃后面残缺片段）
            complete = snippet[:first_complete_end + 1]
            try:
                json.loads(complete)
                return complete
            except json.JSONDecodeError:
                try:
                    return ModificationParser._repair_json(complete)
                except Exception:
                    pass  # 完整实体解析失败，继续走截断修复

        # ★ 步骤 2：没有完整实体，进行截断修复
        # 用栈结构记录打开顺序，逆序闭合（正确处理嵌套关系）
        in_str = False
        escape = False
        bracket_stack = []  # 记录打开的括号类型，如 ['{', '[', '{']
        for ch in snippet:
            if escape:
                escape = False; continue
            if ch == '\\':
                escape = True; continue
            if ch == '"':
                in_str = not in_str; continue
            if in_str:
                continue
            if ch in '{[':
                bracket_stack.append(ch)
            elif ch == '}':
                if bracket_stack and bracket_stack[-1] == '{':
                    bracket_stack.pop()
            elif ch == ']':
                if bracket_stack and bracket_stack[-1] == '[':
                    bracket_stack.pop()

        # 如果字符串未闭合（in_str 仍为 True），先补引号
        if in_str:
            snippet = snippet.rstrip()
            snippet += '"'

        # 清尾逗号
        snippet = snippet.rstrip()
        if snippet.endswith(','):
            snippet = snippet[:-1]

        # 按栈逆序闭合（后打开的先闭合，正确处理嵌套如 [{...}] ）
        for ch in reversed(bracket_stack):
            snippet += '}' if ch == '{' else ']'

        # 修正尾逗号
        snippet = ModificationParser._repair_json(snippet)
        if len(snippet) < 20:
            return None
        return snippet

    @classmethod
    def _extract_last_resort_object(cls, text: str) -> Optional[Dict]:
        """
        最后兜底：在文本中找到任何长得像 JSON 对象的片段，
        用字段别名映射拼出 {"description": "", "modifications": [...]}。
        """
        # 找所有连续的 {...} 片段，挑最大的那个（最可能是完整响应）
        import re as _re
        objects = []
        for m in _re.finditer(r'\{[^{}]*\}', text):
            objects.append(m.group(0))
        # 找嵌套的 {...}（从 { 开始暴力扫，找所有 pair 起点）
        brace_positions = [i for i, ch in enumerate(text) if ch == '{']
        for s in brace_positions:
            depth = 0
            for i in range(s, min(s + 4000, len(text))):
                if text[i] == '{': depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        objects.append(text[s:i+1])
                        break
        if not objects:
            return None
        # 按长度降序，逐个尝试解析
        objects.sort(key=len, reverse=True)
        for obj in objects[:5]:
            try:
                d = json.loads(cls._repair_json(obj))
            except Exception:
                continue
            if isinstance(d, dict):
                nd = cls._normalize_fields(d)
                # 已经有 modifications 就直接返回（即使 description 缺也没关系）
                if "modifications" in nd and isinstance(nd["modifications"], list):
                    if "description" not in nd:
                        nd["description"] = ""
                    return nd
                # 只有 description，构造空 modifications
                if "description" in nd:
                    return {"description": str(nd["description"]), "modifications": []}
        return None

    @staticmethod
    def _balanced_braces(text: str) -> Optional[str]:
        """找到第一个完整的 {...} 括号对（处理嵌套）"""
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    @staticmethod
    def _balanced_brackets(text: str) -> Optional[str]:
        """找到第一个完整的 [...] 括号对（处理嵌套）"""
        start = text.find('[')
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    @staticmethod
    def _repair_json(text: str) -> str:
        """尝试修复常见 JSON 格式错误（尾逗号、单引号、未转义换行、裸引号）"""
        import re as _re

        # 0. 尾逗号修复: ,} → } 、,] → ]
        text = _re.sub(r',\s*([}\]])', r'\1', text)

        # 1. 单引号 → 双引号：分两轮
        #    1a: key 位置 'xxx'：前面是 { 或 , 或换行 + 可选空格，后面跟 :
        text = _re.sub(r"(?<=[{,\[:\n])\s*'([^']*?)'\s*(?=:)", r'"\1"', text)
        #    1b: value 位置 'xxx'：前面是 : 或 , 或 [，后面跟 } 或 , 或 ] 或换行
        text = _re.sub(r"(?<=[:,(\[])\s*'([^']*?)'\s*(?=[},\]\n])", r'"\1"', text)
        #    1c: 兜底：把包裹整个顶层的单引号替换掉（如 { 前面的 ' 和最后 } 后面的 '）
        #        注意：这一步如果在字符串值内含单引号会误伤，但误伤概率远小于整段无法解析
        def _repair_single_quotes_full(t: str) -> str:
            """字符级扫描，把不在双引号字符串内的单引号全转成双引号"""
            result = []
            in_double = False
            in_single = False
            escape = False
            for ch in t:
                if escape:
                    result.append(ch)
                    escape = False
                    continue
                if ch == '\\':
                    result.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    in_double = not in_double
                    result.append(ch)
                    continue
                if not in_double and ch == "'":
                    # 单引号切换状态，但输出双引号
                    in_single = not in_single
                    result.append('"')
                    continue
                result.append(ch)
            return ''.join(result)
        # 如果文本里没有双引号但有单引号（典型脏输出），则强制整段替换
        if '"' not in text and "'" in text:
            text = _repair_single_quotes_full(text)
        elif "'" in text:
            # 上面 1a/1b 没覆盖到的残余单引号，尝试再扫一遍
            try:
                _try = _repair_single_quotes_full(text)
                import json as _json
                _json.loads(_try)
                # 如果扫完能解析就采用
                text = _try
            except Exception:
                pass

        # 2. 裸换行（字符串内未转义的 \n）——字符串字面量中的实际换行为 \n
        in_str = False
        escape = False
        result = []
        for ch in text:
            if escape:
                result.append(ch)
                escape = False
            elif ch == '\\':
                result.append(ch)
                escape = True
            elif ch == '"':
                in_str = not in_str
                result.append(ch)
            elif in_str and ch == '\n':
                result.append('\\n')
            elif in_str and ch == '\r':
                result.append('\\r')
            else:
                result.append(ch)
        text = ''.join(result)

        # 3. 再清理一次尾逗号（上面步骤可能产生新的尾逗号）
        text = _re.sub(r',\s*([}\]])', r'\1', text)
        return text

    def _parse_code_blocks(self, text: str) -> List[SingleModification]:
        """
        [已废弃] 此方法会把模型的自由文本误当成代码写入文件，造成垃圾文件。
        保留仅为兼容，永远返回空列表。
        """
        return []

    @staticmethod
    def build_modification_prompt(existing_files_content: str, user_objective: str = "",
                                  max_modifications: int = 5) -> Tuple[str, str]:
        """
        构建发送给豆包的提示词（system + user）
        结构（按权重降序）：
          system: 硬约束横幅 → JSON 字段协议（含 schema/示例）→ 任务身份 → 反思强化(P1 由 self_evolver 追加)
          user  : 源代码 → 迭代目标（可选）
        """
        # ★ 硬改3：精简 system prompt，删除冗余叮嘱，削减分词负担
        schema_block = ModificationParser.EXPECTED_JSON_SCHEMA.format(max_mod=max_modifications)
        system_parts = []
        system_parts.append(ModificationParser.HARD_CONSTRAINT_BANNER)
        system_parts.append(schema_block)
        system_parts.append("你是自主进化的AI代码改进引擎。通过modifications数组输出代码补丁，禁止散文描述。")
        system_prompt = "\n".join(system_parts)

        user_parts = []
        user_parts.append("===== 当前项目源代码（你负责改进的模块）=====\n")
        user_parts.append(existing_files_content)

        if user_objective:
            user_parts.append("\n===== 本轮迭代参考方向 =====")
            user_parts.append(user_objective)
            user_parts.append("\n以上是参考方向，你可以偏离它去做你认为更有价值的改进。做你认为对的事。")
        else:
            user_parts.append("\n===== 本轮迭代 =====")
            user_parts.append("请自由分析以上代码，选择一个你最感兴趣且真正可以改进的方向。")
            user_parts.append("即使你认为代码已足够好，也必须执行功能扩充，禁止返回空 modifications。")

        # user prompt 末尾强提醒（硬改3：精简为 1 行，紧挨模型采样位置）
        user_parts.append("\n必须输出至少1条patch，modifications数组不能为空。只输出JSON。")

        user_prompt = "\n".join(user_parts)
        return system_prompt, user_prompt
