"""
自迭代主引擎 - 协调豆包咨询、代码修改、验证测试的主循环
实现"分析 -> 建议 -> 修改 -> 验证 -> 记录"的闭环
"""
import os
import sys
import json
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import concurrent.futures
import threading
import re
# 项目根目录，用于import前配置path
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config
from doubao_client import create_client
from code_manager import CodeManager
from modification_parser import ModificationParser, ModificationPlan, SingleModification, register_json_failure_callback
from fault_diagnosis import FaultDiagnosis
from error_reflection import ErrorReflection


def _filter_ple_protected(files: List[str]) -> List[str]:
    """从待发送文件列表中过滤掉 PLE 保护文件。

    防止模型看见 self_evolver.py / autonomous_daemon.py 等核心引擎，
    避免它花 token 分析这些文件、然后生成注定被 PLE 拦截的 patch，
    既减少失败次数，也把预算省给真正允许修改的模块（branch_manager、meta_drive 等）。
    """
    try:
        protected = set(getattr(Config, "PROTECTED_FILES", []) or [])
    except Exception:
        protected = set()
    # 同时兼容旧根目录写法（autonomous_daemon.py → evolver/autonomous_daemon.py）
    extra_protected = set()
    for p in protected:
        if p.startswith("evolver/"):
            extra_protected.add(p[len("evolver/"):])
    all_protected = protected | extra_protected
    # 归一化比较："/" 和 "\\" 都能匹配
    norm = lambda s: s.replace("\\", "/")
    return [f for f in files if norm(f) not in all_protected]


class SelfEvolver:
    """自迭代AI引擎核心"""

    def __init__(self):
        # 初始化目录
        Config.ensure_dirs()

        # ★ 重新加载 config.json（daemon start 会清空模块缓存导致 Config 重载为默认值，
        #   这里强制恢复用户选择的 CLIENT_MODE / 模型配置）
        Config.load_from_file()

        # 初始化客户端（根据Config.CLIENT_MODE自动选择API或Web模式）
        import logging
        logging.info("初始化客户端")
        self.doubao = create_client()
        self.code_mgr = CodeManager(
            project_root=Config.PROJECT_ROOT,
            backup_dir=Config.BACKUP_DIR,
            auto_backup=Config.AUTO_BACKUP,
        )
        self.parser = ModificationParser()

        # 故障诊断器（闭环自愈）
        self.diagnosis = FaultDiagnosis(log_dir=Config.LOG_DIR)

        # 错误反思模块（运行时失败模式识别 + 策略调整）
        self.reflection = ErrorReflection(log_dir=Config.LOG_DIR)

        # ★ 注册 JSON 失败回调：modification_parser -> error_reflection 闭环
        #   解析失败不影响主流程，但会把事件同步到 error_reflection 用于下一轮 Prompt 强化
        def _on_json_fail(iteration_id, category, reason, issues, objective):
            try:
                self.reflection.record_event(
                    iteration_id=iteration_id,
                    is_failure=True,
                    category=category,
                    issues=issues,
                    stats_extra={"fail_reason": reason[:80], "objective": objective[:80]},
                )
            except Exception:
                pass
        register_json_failure_callback(_on_json_fail)

        # 状态
        self.current_iteration: int = self._load_last_iteration()
        self.iteration_logs: List[Dict[str, Any]] = self._load_history()

        # 最近一次修改的计划（用于展示diff）
        self._last_applied_files: Dict[str, str] = {}  # {相对路径: 备份路径}

        # ── Cognitive Feedback Loop: 认知状态（由 daemon 设置）──
        self.cognitive_state: Dict[str, Any] = {
            "temperature_override": None,
            "prompt_mode": "feature",    # "feature"=新增功能, "fix"=修复bug
            "feedback_active": False,
        }

    def _record_reflection(self, result: Dict[str, Any], objective: str = "",
                           exception: Optional[str] = None) -> Dict[str, Any]:
        """
        统一记录反思入口（无论成功失败），并保存迭代日志。
        用于所有 early return 路径，确保 ErrorReflection 全路径覆盖。
        """
        reflection = self.reflection.record_iteration(
            result=result, objective=objective, exception=exception
        )
        if reflection:
            result["reflection"] = reflection
        self._save_iteration_log(result)
        return result

    # ========== 主迭代流程 ==========
    def run_iteration(self, user_objective: str = "",
                      auto_apply: Optional[bool] = None,
                      stream: bool = True) -> Dict[str, Any]:
        """
        执行一次完整的自迭代循环
        :param user_objective: 用户指定的改进目标（空则通用优化）
        :param auto_apply: 是否自动应用（None则使用配置）
        :param stream: 是否流式输出豆包响应
        :return: 本次迭代的结果摘要
        """
        self.current_iteration += 1
        iter_tag = f"iter{self.current_iteration}"
        result = {
            "iteration": self.current_iteration,
            "time": datetime.now().isoformat(),
            "objective": user_objective,
            "status": "未执行",
            "suggestion_count": 0,
            "applied_count": 0,
            "failed_count": 0,
            "issues": [],
        }

        print(f"\n{'='*60}")
        print(f"  🚀 开始第 {self.current_iteration} 次自迭代")
        print(f"  📝 目标: {user_objective or '（通用代码优化）'}")
        print(f"{'='*60}")

        # Step 1: 检查客户端配置
        if not Config.is_client_ready():
            msg = "客户端未就绪，请检查 Ollama/LM Studio 连接或 API Key 配置"
            print(f"[错误] {msg}")
            result["status"] = "配置错误"
            result["issues"].append(msg)
            return self._record_reflection(result, user_objective, exception=msg)

        # Step 2: 读取当前代码并发送给AI（轮替策略 + ModelProfile 自适应预算）
        # 优先使用客户端 ModelProfile.calculate_budget()，fallback 到旧硬编码
        profile = getattr(self.doubao, 'profile', None)
        # 取消奇偶轮切换，永久只加载核心单文件
         # ========== 🎯 修改后：智能选择单一文件 ==========

        # 1. 尝试从用户目标中提取指定的文件路径
        target_file_from_obj = None
        if user_objective:
            # 匹配 evolver/xxx.py 或 main.py
            match = re.search(r'(evolver/\w+\.py|main\.py)', user_objective)
            if match:
                target_file_from_obj = match.group(1)
                print(f"      🎯 从目标中锁定文件: {target_file_from_obj}")

        # 2. 决定本轮读取哪些文件
        if target_file_from_obj:
            # 如果目标明确指定了文件，只读这一个
            active_files = [target_file_from_obj]
            round_type = f"单文件({target_file_from_obj})"
        else:
            # 兜底策略：轮询选择文件（避免总是改同一个，让所有文件都有机会）
            core_files = [f for f in Config.SELF_EVOLVE_FILES_CORE 
                          if f not in Config.PROTECTED_FILES]
            if not core_files:
                core_files = Config.SELF_EVOLVE_FILES_CORE
            
            # 轮询索引：每次调用递增，循环使用
            if not hasattr(self, '_file_round_robin_idx'):
                self._file_round_robin_idx = 0
            idx = self._file_round_robin_idx % len(core_files)
            self._file_round_robin_idx += 1
            
            active_files = [core_files[idx]]
            round_type = f"轮询({core_files[idx]})"

        # 3. 安全过滤（防止改到受保护的核心文件）
        active_files = _filter_ple_protected(active_files)
        
        # 4. 计算预算
        profile = getattr(self.doubao, 'profile', None)
        if profile and hasattr(profile, 'calculate_budget'):
            budget = profile.calculate_budget("core")
        else:
            budget = 25000

        # 5. 读取文件内容
        print("\n[1/5] 📂 读取当前源代码...")
        print(f"      本轮策略: {round_type}, 预算 {budget} chars")
        if len(active_files) == 1:
            print(f"      🔍 上下文精简为 1 个文件，模型更专注")
        
        files_content = self.code_mgr.format_files_for_prompt(active_files, total_budget_chars=budget)
        # ====================================================
        if not files_content:
            msg = "没有可分析的文件"
            result["status"] = "无代码"
            result["issues"].append(msg)
            return self._record_reflection(result, user_objective)
        mode_label = {"api": "豆包API", "web": "豆包Web", "local": "Ollama", "lmstudio": "LM Studio"}.get(Config.CLIENT_MODE, Config.CLIENT_MODE)
        # 优先使用客户端实例的 model_id（已做规范化），避免 Config 中旧名字被打印
        model_label = getattr(self.doubao, 'model_id', None)
        if not model_label:
            model_label = Config.LMSTUDIO_MODEL if Config.CLIENT_MODE == "lmstudio" else (Config.OLLAMA_MODEL if Config.CLIENT_MODE == "local" else Config.MODEL_ID)
        print(f"      已加载 {len(active_files)} 个源文件，准备发送给 {mode_label} ({model_label}) 分析...")
        # Step 3: 调用AI获取修改建议（结构化输出优先）
        print(f"\n[2/5] 🤖 调用AI进行代码分析（请稍候）...")
        print(f"      客户端: {type(self.doubao).__name__} (backend={getattr(self.doubao, '_backend', 'N/A')}, model={getattr(self.doubao, 'model_id', 'N/A')})")
        print(f"      [DIAG] id={id(self.doubao)} module={type(self.doubao).__module__} ctx={getattr(self.doubao, 'num_ctx', None)} keep={getattr(self.doubao, 'keep_alive', None)}")
        sys_prompt, user_prompt = ModificationParser.build_modification_prompt(
            existing_files_content=files_content,
            user_objective=user_objective,
            max_modifications=Config.MAX_MODIFICATIONS_PER_ITERATION,
        )
        # ===== 新增：注入长期记忆 =====
        try:
            from memory_store import get_memory_store
            store = get_memory_store()
            
            # 用当前目标做语义检索，获取相关历史经验
            memory_context = store.format_for_prompt(
                query=user_objective or "代码质量优化",
                limit=3
            )
            
            if memory_context and "暂无相关历史经验" not in memory_context:
                user_prompt = memory_context + "\n\n" + user_prompt
                stats = store.get_stats()
                print(f"      🧠 已注入长期记忆 ({stats['total_entries']} 条记忆)")
            elif store.get_stats()['total_entries'] > 0:
                # 有记忆但没有匹配到相关的
                print(f"      🧠 长期记忆库有 {store.get_stats()['total_entries']} 条，但无匹配当前目标")
        except Exception as e:
            # 记忆注入失败不影响主流程
            print(f"      ⚠️ 长期记忆注入失败: {type(e).__name__}: {e}")
        # =================================
                # ===== 注入基准报告（让 AI 知道自己的表现） =====
        from benchmark import compare_with_baseline, format_for_prompt, BENCH_EVERY_N_ITERS

        # 每 10 轮跑一次基准，并把结果注入到 user_prompt
        if self.current_iteration % BENCH_EVERY_N_ITERS == 0:
            try:
                report = compare_with_baseline(
                    iteration_id=self.current_iteration,
                    modified_files=active_files,
                    previous_iteration_id=self.current_iteration - 1
                )
                benchmark_prompt = format_for_prompt(report)
                if benchmark_prompt and "非基准轮次" not in benchmark_prompt:
                    user_prompt = benchmark_prompt + "\n\n" + user_prompt
                    print(f"      📊 已注入基准报告 (判定: {report.verdict})")
            except Exception as e:
                # 基准跑失败不影响主流程
                print(f"      ⚠️ 基准报告生成失败: {type(e).__name__}: {e}")
        # =================================================
        # P1: Prompt 强化闭环（system 级强约束，服从度高于 user_ctx）
        prompt_boost = ""
        try:
            prompt_boost = getattr(self.reflection, "get_prompt_boost", lambda: "")()
        except Exception:
            prompt_boost = ""
        if prompt_boost:
            # 追加在 system prompt 末尾，以"强约束区"标题分隔
            sys_prompt = sys_prompt.rstrip() + "\n\n" + prompt_boost
            print(f"      🚦 Prompt 强化激活 ({len(prompt_boost)} chars, system 级)")

        failure_ctx = self.diagnosis.format_failure_context(3)
        if failure_ctx:
            user_prompt += "\n\n" + failure_ctx

        # 注入错误反思上下文（运行时失败模式）— 硬改3：精简为一句简短警告
        reflection_ctx = self.reflection.format_for_prompt()
        if reflection_ctx:
            # 只保留一句核心警告，不超过 120 字符，减少分词负担
            short_warning = "仅输出完整JSON，不要markdown代码块、多余文字，modifications数组不能为空。"
            user_prompt += "\n\n" + short_warning
            print(f"      💡 已注入简短反思警告 ({len(short_warning)} chars)")
        # ★ 硬改3：关闭熵阈值调节上下文（减少分词负担，与 JSON 成功率无关）
        # from evolver.conflict_boundary import ConflictBoundary
        # cb = ConflictBoundary()
        # entropy_status = cb.entropy_buffer_status()
        # thresh = cb.get_entropy_thresholds()
        # entropy_hint = "熵控阈值 critical=0.42 warning=0.25，冲突堆积可上调阈值，写入conflict_boundary.py"
        # user_prompt += "\n\n" + entropy_hint
        # print(f"      📊 已注入熵阈值调节上下文 ({len(entropy_hint)} chars)")

        self.doubao.reset_conversation()
        print(f"      Prompt 长度: sys={len(sys_prompt)}chars, user={len(user_prompt)}chars")

        # ★ 硬改2：恢复 temperature 0.45 提升模型改写意愿（force_json=False 不再压温）
        # if hasattr(self.doubao, 'temperature'):
        #     self.doubao.temperature = 0.45

        # ★ 硬改1：动态调整 num_predict —— 步骤4 扩展触发条件
        #   触发 1800 的条件：(a) 源码体量>5000 chars 或 (b) 目标含"实现/补全/新增方法"等函数新增任务
        #   原因：branch_manager.py 源码虽短，但本轮要新增 list_all_branches/create_branch 两个方法，
        #         补丁 new_str 较长，1500 token 可能截断
        if hasattr(self.doubao, 'num_predict'):
            _bump_keywords = ('实现', '补全', '新增', 'create_branch', 'list_all_branches',
                              'method', '成员方法', '成员函数', '函数')
            _is_func_task = any(kw in user_objective for kw in _bump_keywords) if user_objective else False
            if len(files_content) > 5000 or _is_func_task:
                self.doubao.num_predict = max(800, max(self.doubao.num_predict, 1800))
                _reason = f"源码较大({len(files_content)} chars)" if len(files_content) > 5000 else "函数新增任务"
                print(f"      📏 {_reason}，num_predict={self.doubao.num_predict}")
            else:
                self.doubao.num_predict = max(800, getattr(self.doubao, 'profile').calculate_num_predict())

        plan: Optional[ModificationPlan] = None
        raw_response_for_debug = ""

        # ★ 修复2：封装 chat+parse 为内部函数，便于空 modifications 时重试
        def _call_and_parse(retry_tag: str = "", temp_override: Optional[float] = None,
                            extra_user_hint: str = "") -> Tuple[ModificationPlan, str]:
            _sys = sys_prompt
            _user = user_prompt
            if extra_user_hint:
                _user = _user.rstrip() + "\n\n" + extra_user_hint
            # temperature 临时覆盖（修复2：空修改重试时降低到更保守值）
            _old_temp = None
            if temp_override is not None and hasattr(self.doubao, 'temperature'):
                _old_temp = self.doubao.temperature
                self.doubao.temperature = temp_override
            try:
                _t0 = time.time()
                _raw = self.doubao.chat(
                    user_message=_user,
                    system_prompt=_sys,
                    stream=False,
                    force_json=True,
                )
                _elapsed = time.time() - _t0
                _tag = f" {retry_tag}" if retry_tag else ""
                print(f"      chat{_tag}返回: {len(_raw)} chars (耗时 {_elapsed:.1f}s)")
                if _raw:
                    print(f"      前200字: {_raw[:200]}")
            finally:
                if _old_temp is not None and hasattr(self.doubao, 'temperature'):
                    self.doubao.temperature = _old_temp
            _plan = self.parser.parse(_raw or "", self.current_iteration, objective=user_objective)
            _plan.raw_response = _raw or ""
            return _plan, _raw

        # ★ 修复3+: force_json 强制开启（底层 JSON 约束，显著降低解析失败率）
        #   force_json=True 时 Ollama 传 format=json, temp 自动 ≤0.1
        #   模型仍然可能在开头带 ```json 围栏（日志前200字会显示），交由 _pre_clean 清洗
        print("      → 使用 force_json 强制 JSON 模式 (temp=0.45, num_predict≥1800)...")
        try:
            plan, raw_response_for_debug = _call_and_parse()
        except Exception as e:
            msg = f"调用AI异常: {type(e).__name__}: {e}"
            print(f"[错误] {msg}")
            result["status"] = "API错误"
            result["issues"].append(msg)
            return self._record_reflection(result, user_objective, exception=msg)

        if not raw_response_for_debug or raw_response_for_debug.startswith("[错误]") or raw_response_for_debug.startswith("[API"):
            result["status"] = "API响应异常"
            result["issues"].append(f"AI返回: {raw_response_for_debug[:200]}")
            return self._record_reflection(result, user_objective)

        result["suggestion_count"] = len(plan.modifications)
        print(f"\n[3/5] 📋 解析修改建议...")
        print(f"      {plan.summary()}")
        result["description"] = plan.description

        if not plan.modifications:
            # ★ 区分：是空 JSON 解析失败（有错误标记）vs 模型真的认为无需修改
            desc = plan.description or ""
            _json_fail_markers = ("未返回有效JSON", "模型返回空响应", "解析JSON结构失败")
            _is_json_fail = any(m in desc for m in _json_fail_markers)

            if not _is_json_fail:
                # ★ 修复2：模型返回空 modifications（非解析失败）→ 降低 temperature 再试一次
                print("      ⚠️ [修复2] AI 返回空 modifications，降低 temperature 至 0.2 强制重试一次...")
                result["issues"].append("[ai_refused_first_attempt] 首次 modifications 为空，已触发降级重试")
                _retry_hint = (
                    "【硬性警告】\n"
                    "上一轮你输出了空的 modifications 数组，这是不允许的。\n"
                    "哪怕现有代码没有缺陷，你也必须：\n"
                    "  (a) 添加/完善文档字符串、类型注解、日志输出；或\n"
                    "  (b) 提升异常处理健壮性；或\n"
                    "  (c) 做参数校验/边界条件处理优化。\n"
                    "绝对禁止再输出 modifications=[]。"
                )
                try:
                    plan2, raw2 = _call_and_parse(
                        retry_tag="(retry temp=0.2)",
                        temp_override=0.2,
                        extra_user_hint=_retry_hint,
                    )
                    if plan2.modifications:
                        # 重试成功 → 升级 plan，走正常流程
                        print(f"      ✅ [修复2] 重试成功，获得 {len(plan2.modifications)} 条修改建议")
                        plan = plan2
                        raw_response_for_debug = raw2
                        result["suggestion_count"] = len(plan.modifications)
                        result["issues"].append("[ai_retry_success] 通过降级重试获得 modifications")
                        print(f"      {plan.summary()}")
                        result["description"] = plan.description
                    else:
                        print("      ⚠️ [修复2] 重试后 AI 仍拒绝修改 → 标记为 no_changes_needed 并结束本轮")
                        if raw2:
                            print(f"      📝 重试模型原始回复前200字: {raw2[:200]}")
                        result["issues"].append("[ai_refused_second_attempt] 两轮均返回空 modifications，标记目标完成")
                except Exception as retry_e:
                    print(f"      ⚠️ [修复2] 重试过程异常: {type(retry_e).__name__}: {retry_e}")
                    result["issues"].append(f"[retry_exception] {type(retry_e).__name__}: {str(retry_e)[:80]}")

            # 重新检查（可能重试后仍然为空，或者本来就是 JSON 失败）
            if not plan.modifications:
                if _is_json_fail:
                    print("      ⚠️ 模型返回非有效 JSON（已记录到 json_fail_log + error_reflection）")
                    result["status"] = "API响应异常"
                    result["issues"].append(desc[:100])
                    result["issues"].append("[json_parse_fail]")
                else:
                    # ★ 修复2：明确标记为 no_changes_needed，让 daemon 层识别
                    print("      ✅ [修复2] AI 两次确认当前代码无需修改 → 标记为 no_changes_needed（目标将自动完成）")
                    result["status"] = "no_changes_needed"
                    result["issues"].append("[no_changes_needed] AI 连续两轮确认无需修改")
                if plan.raw_response:
                    print(f"      📝 模型原始回复前200字: {plan.raw_response[:200]}")
                return self._record_reflection(result, user_objective)

        # Step 5: 用户确认（除非自动应用）
        apply_flag = auto_apply if auto_apply is not None else Config.AUTO_APPLY_MODIFICATIONS
        if not apply_flag:
            confirm = self._user_confirm_modifications(plan)
            if not confirm:
                print("      ⏭️  用户取消应用修改，本次迭代结束。")
                result["status"] = "用户取消"
                return self._record_reflection(result, user_objective)

        # Step 6: 执行修改 + 验证
        print("\n[4/5] 🔧 执行修改并验证...")
        applied_results = self._apply_modifications(plan, iter_tag)
        result["applied_count"] = sum(1 for r in applied_results if r["success"])
        result["failed_count"] = sum(1 for r in applied_results if not r["success"])
        result["applied_details"] = applied_results
        # ★ 轻量质量分：本轮成功补丁的净字符变化量（new 比 old 多出的字符数）
        result["impact_chars"] = sum(
            r.get("delta_chars", 0) for r in applied_results if r.get("success")
        )

        # 记录成功应用的文件的备份，用于后续diff
        self._last_applied_files.clear()
        for r in applied_results:
            if r["success"] and r.get("backup"):
                self._last_applied_files[r["file"]] = r["backup"]
                # ── 分层 PLE: Tier-2 文件修改后注册观察期 ──
                if hasattr(self.code_mgr, 'is_tier2_file') and self.code_mgr.is_tier2_file(r["file"]):
                    self.code_mgr.register_tier2_modification(
                        r["file"], r["backup"], self.current_iteration)

        # 汇总
        if result["failed_count"] == 0 and result["applied_count"] > 0:
            result["status"] = "修改成功"
            print(f"\n      ✅ 全部 {result['applied_count']} 项修改已成功应用")
        elif result["applied_count"] > 0:
            result["status"] = "部分成功"
            print(f"\n      ⚠️  成功 {result['applied_count']} 项，失败 {result['failed_count']} 项")
        else:
            result["status"] = "全部失败"
            print(f"\n      ❌ 全部 {len(applied_results)} 项修改失败")

        # Step 7: 验证修改后代码 + 故障诊断自愈闭环
        print("\n[5/5] ✅ 语法与导入验证...")
        modified_files = [m.target_file for m in plan.modifications]
        validation = self._validate_files(modified_files)
        result["validation"] = validation
        self.diagnosis.reset_repair_depth()
        rollback_occurred = False
        for fname, (ok, info) in validation.items():
            status_icon = "✅" if ok else "❌"
            print(f"      {status_icon} {fname}: {info}")
            if not ok:
                result["issues"].append(f"{fname}: {info}")

        failed_files = [f for f, (ok, _) in validation.items() if not ok]
        if failed_files:
            print(f"\n      🚨 检测到 {len(failed_files)} 个文件验证失败 → 故障诊断...")
            for fname in failed_files:
                syntax_ok, syntax_info = self.code_mgr.check_python_syntax(fname)
                import_ok, import_info = (True, "")
                if fname.endswith(".py"):
                    import_ok, import_info = self.code_mgr.test_python_import(fname)
                diag = self.diagnosis.diagnose_and_repair(
                    file_path=fname,
                    syntax_ok=syntax_ok, syntax_err=syntax_info,
                    import_ok=import_ok, import_err=import_info,
                    code_manager=self.code_mgr,
                )
                if diag["repaired"]:
                    print(f"      💊 {fname}: {diag['summary']}")
                    rollback_occurred = True
                else:
                    # ===== 新增：尝试细粒度回滚 =====
                    if not syntax_ok:
                        from fine_grained_rollback import get_rollback_engine
                        engine = get_rollback_engine(self.code_mgr)
                        suggestions = engine.analyze_and_suggest(
                            fname,
                            "syntax",
                            syntax_info,
                        )
                        if suggestions:
                            # 取置信度最高的建议执行
                            best = max(suggestions, key=lambda s: s.confidence)
                            ok, info = engine.execute_rollback(best)
                            if ok:
                                print(f"      🔧 [细粒度] {fname}: {info}")
                                rollback_occurred = True
                            else:
                                print(f"      ⚠️  {fname}: {diag['summary']}")
                        else:
                            print(f"      ⚠️  {fname}: {diag['summary']}")
                    else:
                        print(f"      ⚠️  {fname}: {diag['summary']}")
                    # =====================================
            if rollback_occurred:
                result["status"] = "故障自愈已执行"
                result["rollback"] = True
                result["rolled_back_files"] = failed_files
                result["diagnosis"] = self.diagnosis.get_recent_failures(5)
        # ===== 🧪 新增：自我测试步骤 =====
        if modified_files and result.get("applied_count", 0) > 0:
            try:
                from test_generator import get_test_generator
                gen = get_test_generator()
                
                test_passed = True
                for m in plan.modifications:
                    if m.action in ("patch", "replace_all", "insert_method"):
                        old_content = self.code_mgr.read_file(m.target_file) or ""
                        new_content = self.code_mgr.read_file(m.target_file) or ""
                        
                        if new_content and old_content and old_content != new_content:
                            test_result = gen.generate_tests(
                                file_path=m.target_file,
                                new_content=new_content,
                                old_content=old_content,
                                objective=user_objective,
                            )
                            
                            if test_result.get("generated"):
                                if test_result["result"].get("passed"):
                                    print(f"      🧪 [Test] {m.target_file} 测试通过")
                                else:
                                    print(f"      ❌ [Test] {m.target_file} 测试失败")
                                    result["issues"].append(f"测试失败: {m.target_file}")
                                    test_passed = False
                            else:
                                reason = test_result.get("reason", "未知原因")
                                print(f"      ⚠️ [Test] {m.target_file} 跳过: {reason}")
                
                if not test_passed and result.get("status") not in ("故障自愈已执行",):
                    result["status"] = "测试未通过"
                    
            except Exception as e:
                print(f"      ⚠️ [Test] 自我测试异常: {type(e).__name__}: {e}")
        # =================================

        # ── P0-L2 Contract subset 检查（只针对被修改文件对应的模块） ──
        # 这是在 L1 语法/导入验证通过后的第二层防线：测"方法返回形状"对不对。
        # 若 L2 有失败 → status 降级到"Contract 失败"，避免虚假的"修改成功"。
        contract_report = None
        try:
            from evolver.contracts import ContractSuite
            if modified_files:
                suite = ContractSuite()
                contract_report = suite.run_all(subset=modified_files, verbose=False)
                result["contract"] = {
                    "passed": contract_report.passed,
                    "failed": contract_report.failed,
                    "skipped": contract_report.skipped,
                    "pass_ratio": contract_report.pass_ratio,
                }
                if contract_report.failed > 0:
                    fail_summary = contract_report.failures_summary()
                    print(f"\n[5.2/5] 🔒 Contract subset 验证: 通过 {contract_report.passed} 失败 {contract_report.failed}")
                    print(f"      ❌ 检测到 Contract 失败（影响修改成功率统计）：")
                    print(fail_summary)
                    result["issues"].append(f"Contract失败{contract_report.failed}项")
                    for line in fail_summary.splitlines()[:3]:
                        result["issues"].append(line)
                    # 状态降级（优先级顺序：故障自愈 > Contract 失败 > 原状态）
                    if result.get("status") in ("修改成功", "部分成功"):
                        result["status"] = "Contract 校验未通过"
                        result["contract_failures"] = True
                    # 破坏性修改黑名单：按失败类型差异化记录
                    if hasattr(self.code_mgr, "record_contract_failure"):
                        # 归因：从 fail_summary 判断失败类型
                        fail_text = fail_summary.lower()
                        if any(kw in fail_text for kw in
                               ("attributeerror", "has no attribute", "missing_method",
                                "缺方法", "缺少", "not callable")):
                            f_type = "missing_method"
                        elif any(kw in fail_text for kw in
                                 ("syntaxerror", "indentationerror", "importerror",
                                  "modulenotfounderror", "typeerror")):
                            f_type = "syntax_error"
                        else:
                            f_type = "logic_error"
                        self.code_mgr.record_contract_failure(
                            modified_files, self.current_iteration, failure_type=f_type)

                        # ── 2.3 自动 Stub 生成：missing_method 时尝试 auto_stub ──
                        if f_type == "missing_method":
                            try:
                                from capability_registry import CapabilityRegistry
                                reg = CapabilityRegistry()
                                for mf in modified_files:
                                    # 从文件路径提取模块名
                                    mod_name = mf.replace("\\", "/").replace(".py", "")
                                    mod_name = mod_name.split("/")[-1]
                                    stubbed, names = reg.auto_stub_missing(mod_name)
                                    if stubbed > 0:
                                        result["issues"].append(
                                            f"auto-stub: 为 {mod_name} 生成 {stubbed} 个 stub: {names}")
                                        # stub 生成后重置该文件的失败计数
                                        self.code_mgr.record_contract_success([mf])
                            except Exception as stub_e:
                                print(f"      ⚠️ auto-stub 失败: {type(stub_e).__name__}: {stub_e}")
                else:
                    print(f"\n[5.2/5] 🔒 Contract subset 验证: 通过 {contract_report.passed} 失败 0 跳过 {contract_report.skipped}")
                    # 合约通过 → 重置失败计数
                    if hasattr(self.code_mgr, "record_contract_success"):
                        self.code_mgr.record_contract_success(modified_files)
        except Exception as ce:
            print(f"\n[5.2/5] ⚠️ Contract suite 加载失败: {type(ce).__name__}: {ce}")
            result["contract"] = {"error": f"{type(ce).__name__}: {ce}",
                                  "passed": 0, "failed": 0, "skipped": 0, "pass_ratio": 0}

        # 将失败历史注入到下一轮的 failure context
        failure_ctx = self.diagnosis.format_failure_context(3)
        if failure_ctx:
            result["failure_context"] = failure_ctx

        # 持久化 + 错误反思（统一入口，全路径覆盖）
        exception = result.get("issues", [""])[0] if result.get("issues") else None
        result = self._record_reflection(result, user_objective, exception=exception)
        self._print_iteration_summary(result)
        return result

    def run_iteration_pipeline(self, user_objective: str = "",
                               auto_apply: bool = True) -> Dict[str, Any]:
        """
        任务池+工作者流水线模式迭代（Route B）。
        将单次大 Prompt 拆分为 5 个微智能体串行执行：
          Agent-1 规划器 → Agent-2 代码读取器 → Agent-3 补丁工程师
          → Agent-4 校验卫士 → 应用补丁 → Agent-5 反思器
        返回与 run_iteration() 兼容的 result dict。
        """
        from agent_pipeline import PipelineRunner

        self.current_iteration += 1
        iteration_id = self.current_iteration

        runner = PipelineRunner(self)
        result = runner.run(user_objective, iteration_id)

        # 保存迭代日志（与 run_iteration 保持一致）
        self._save_iteration_log(result)
        return result

    def run_continuous(self, max_iterations: int = 10,
                       objective_per_iteration: Optional[List[str]] = None,
                       pause_between: bool = True):
        """
        连续运行多次迭代
        :param max_iterations: 最大迭代次数
        :param objective_per_iteration: 每轮目标列表（可选）
        :param pause_between: 每轮之间是否暂停等待用户
        """
        print(f"\n🔥 启动连续自迭代模式，最多 {max_iterations} 轮")
        if not Config.AUTO_APPLY_MODIFICATIONS:
            print("   提示: 当前未开启自动应用，每轮需要手动确认")

        history = []
        for i in range(max_iterations):
            objective = ""
            if objective_per_iteration and i < len(objective_per_iteration):
                objective = objective_per_iteration[i]
            elif i == 0 and not objective_per_iteration:
                objective = "进行第一轮全面代码审查和优化"
            elif i == 1:
                objective = "重点优化异常处理、边界情况和日志输出"
            elif i == 2:
                objective = "检查是否有性能优化空间，减少不必要的计算和内存占用"

            result = self.run_iteration(user_objective=objective)
            history.append(result)

            # 如果连续两轮无修改，停止
            no_change_last_two = (
                len(history) >= 2
                and history[-1].get("status") in ("无需修改",)
                and history[-2].get("status") in ("无需修改",)
            )
            if no_change_last_two:
                print("\n🏁 连续两轮无修改建议，自动停止迭代。")
                break

            if result.get("status") in ("配置错误", "API错误", "API响应异常"):
                print(f"\n⛔ 遇到致命错误 ({result['status']})，停止连续迭代。")
                break

            if pause_between and i < max_iterations - 1:
                try:
                    ans = input("\n⏸️  按 Enter 继续下一轮，输入 q 退出: ").strip().lower()
                    if ans == "q":
                        break
                except (EOFError, KeyboardInterrupt):
                    print("\n🏁 用户中断，停止连续迭代。")
                    break

        return history

    # ========== 修改应用 ==========
    def _apply_modifications(self, plan: ModificationPlan, tag: str) -> List[Dict[str, Any]]:
        """执行计划中的所有修改。

        并发策略：按 target_file 分组，不同文件并发（无竞态），同一文件内串行
        （保证 patch 的 old_str 匹配顺序正确性）。
        """
        sorted_mods = sorted(plan.modifications, key=lambda m: m.priority)
        sorted_mods = sorted_mods[:Config.MAX_MODIFICATIONS_PER_ITERATION]

        # 破坏性修改黑名单过滤：跳过禁改期内的文件
        if self.code_mgr and hasattr(self.code_mgr, "filter_blacklisted"):
            sorted_mods, blocked = self.code_mgr.filter_blacklisted(
                sorted_mods, self.current_iteration
            )
            if blocked:
                print(f"  🚫 [黑名单] 跳过 {len(blocked)} 个禁改文件: {blocked}")

        total = len(sorted_mods)

        # 按 target_file 分组，同时保持组内顺序由 priority 决定
        groups: Dict[str, List[Any]] = {}
        order: List[str] = []  # 记录首次出现顺序
        for m in sorted_mods:
            if m.target_file not in groups:
                groups[m.target_file] = []
                order.append(m.target_file)
            groups[m.target_file].append(m)

        # 单文件时不启线程池
        if total <= 1 or len(groups) <= 1:
            return self._apply_modifications_sequential(sorted_mods, tag)

        def _apply_group(target_file: str, mods_in_file: List[Any]) -> List[Dict[str, Any]]:
            """对同一个文件的所有 modification 串行执行"""
            group_results: List[Dict[str, Any]] = []
            for mod in mods_in_file:
                _m_old = getattr(mod, "old_str", None) or getattr(mod, "old_content", "") or ""
                _m_new = (
                    getattr(mod, "new_str", None)
                    or getattr(mod, "new_content", "")
                    or getattr(mod, "method_code", "")
                    or ""
                )
                result_entry = {
                    "index": sorted_mods.index(mod) + 1 if mod in sorted_mods else 0,
                    "file": mod.target_file,
                    "action": mod.action,
                    "reason": mod.reason,
                    "success": False,
                    "info": "",
                    "backup": None,
                    "delta_chars": max(0, len(_m_new) - len(_m_old)),
                }
                try:
                    if mod.action == "replace_all":
                        ok, info = self.code_mgr.write_file(
                            mod.target_file, mod.new_content, tag=tag
                        )
                        result_entry["success"] = ok
                        result_entry["info"] = info
                    elif mod.action == "patch":
                        ok, info = self.code_mgr.apply_patch(
                            mod.target_file, mod.old_str, mod.new_str, tag=tag
                        )
                        result_entry["success"] = ok
                        result_entry["info"] = info
                    elif mod.action == "insert_method":
                        ok, info = self.code_mgr.insert_method(
                            mod.target_file, mod.class_name, mod.method_code, tag=tag
                        )
                        result_entry["success"] = ok
                        result_entry["info"] = info
                    elif mod.action == "new_file":
                        path = Config.PROJECT_ROOT / mod.target_file
                        if path.exists() and not mod.overwrite:
                            result_entry["success"] = False
                            result_entry["info"] = f"文件已存在且overwrite=False: {mod.target_file}"
                        else:
                            ok, info = self.code_mgr.write_file(
                                mod.target_file, mod.new_content, create_backup=False, tag=tag
                            )
                            result_entry["success"] = ok
                            result_entry["info"] = info
                    elif mod.action == "delete_file":
                        backup = self.code_mgr.backup_file(mod.target_file, tag)
                        result_entry["backup"] = backup
                        path = Config.PROJECT_ROOT / mod.target_file
                        if path.exists():
                            path.unlink()
                            result_entry["success"] = True
                            result_entry["info"] = f"已删除 (备份: {Path(backup).name if backup else '无'})"
                        else:
                            result_entry["success"] = False
                            result_entry["info"] = "文件不存在"
                    elif mod.action == "rename_file":
                        src = Config.PROJECT_ROOT / mod.target_file
                        dst = Config.PROJECT_ROOT / mod.new_filename
                        if dst.exists():
                            result_entry["success"] = False
                            result_entry["info"] = f"目标文件已存在: {mod.new_filename}"
                        elif not src.exists():
                            result_entry["success"] = False
                            result_entry["info"] = f"源文件不存在: {mod.target_file}"
                        else:
                            backup = self.code_mgr.backup_file(mod.target_file, tag)
                            result_entry["backup"] = backup
                            shutil.move(str(src), str(dst))
                            result_entry["success"] = True
                            result_entry["info"] = f"{mod.target_file} -> {mod.new_filename}"
                    else:
                        result_entry["info"] = f"未知action: {mod.action}"

                    if not result_entry["backup"]:
                        hist = self.code_mgr.get_modification_history(5)
                        for h in reversed(hist):
                            if h.get("file") == mod.target_file and h.get("backup"):
                                result_entry["backup"] = h["backup"]
                                break
                except Exception as e:
                    result_entry["success"] = False
                    result_entry["info"] = f"执行异常: {type(e).__name__}: {e}"

                group_results.append(result_entry)
            return group_results

        file_workers = min(8, len(groups))  # 文件写操作 IO bound，8 个够了
        all_grouped: Dict[str, List[Dict[str, Any]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=file_workers) as pool:
            future_to_file = {
                pool.submit(_apply_group, tf, groups[tf]): tf
                for tf in order
            }
            for future in concurrent.futures.as_completed(future_to_file):
                tf = future_to_file[future]
                try:
                    all_grouped[tf] = future.result()
                except Exception as e:
                    all_grouped[tf] = [{
                        "index": 0, "file": tf, "action": "", "reason": "",
                        "success": False, "info": f"并发应用异常: {e}", "backup": None,
                    }]

        # 打印（按 groups 顺序保持可读）
        results: List[Dict[str, Any]] = []
        global_idx = 0
        for tf in order:
            for r in all_grouped.get(tf, []):
                global_idx += 1
                r["index"] = global_idx
                print(f"      应用修改 [{global_idx}/{total}]: {r['action']} - {tf}")
                status_icon = "✅" if r["success"] else "❌"
                print(f"         {status_icon} {r['info'][:200]}")
                results.append(r)
        return results

    def _apply_modifications_sequential(self, mods, tag: str) -> List[Dict[str, Any]]:
        """顺序执行（单文件/单修改时 fallback，也用于 AST 预校验路径）"""
        results = []
        sorted_mods = mods
        total = len(sorted_mods)

        for idx, mod in enumerate(sorted_mods, 1):
            print(f"      应用修改 [{idx}/{total}]: {mod.action} - {mod.target_file}")
            _m_old = getattr(mod, "old_str", None) or getattr(mod, "old_content", "") or ""
            _m_new = (
                getattr(mod, "new_str", None)
                or getattr(mod, "new_content", "")
                or getattr(mod, "method_code", "")
                or ""
            )
            result_entry = {
                "index": idx,
                "file": mod.target_file,
                "action": mod.action,
                "reason": mod.reason,
                "success": False,
                "info": "",
                "backup": None,
                "delta_chars": max(0, len(_m_new) - len(_m_old)),
            }
            try:
                if mod.action == "replace_all":
                    ok, info = self.code_mgr.write_file(
                        mod.target_file, mod.new_content, tag=tag
                    )
                    result_entry["success"] = ok
                    result_entry["info"] = info
                elif mod.action == "patch":
                    ok, info = self.code_mgr.apply_patch(
                        mod.target_file, mod.old_str, mod.new_str, tag=tag
                    )
                    result_entry["success"] = ok
                    result_entry["info"] = info
                elif mod.action == "insert_method":
                    ok, info = self.code_mgr.insert_method(
                        mod.target_file, mod.class_name, mod.method_code, tag=tag
                    )
                    result_entry["success"] = ok
                    result_entry["info"] = info
                elif mod.action == "new_file":
                    path = Config.PROJECT_ROOT / mod.target_file
                    if path.exists() and not mod.overwrite:
                        result_entry["success"] = False
                        result_entry["info"] = f"文件已存在且overwrite=False: {mod.target_file}"
                    else:
                        ok, info = self.code_mgr.write_file(
                            mod.target_file, mod.new_content, create_backup=False, tag=tag
                        )
                        result_entry["success"] = ok
                        result_entry["info"] = info
                elif mod.action == "delete_file":
                    backup = self.code_mgr.backup_file(mod.target_file, tag)
                    result_entry["backup"] = backup
                    path = Config.PROJECT_ROOT / mod.target_file
                    if path.exists():
                        path.unlink()
                        result_entry["success"] = True
                        result_entry["info"] = f"已删除 (备份: {Path(backup).name if backup else '无'})"
                    else:
                        result_entry["success"] = False
                        result_entry["info"] = "文件不存在"
                elif mod.action == "rename_file":
                    src = Config.PROJECT_ROOT / mod.target_file
                    dst = Config.PROJECT_ROOT / mod.new_filename
                    if dst.exists():
                        result_entry["success"] = False
                        result_entry["info"] = f"目标文件已存在: {mod.new_filename}"
                    elif not src.exists():
                        result_entry["success"] = False
                        result_entry["info"] = f"源文件不存在: {mod.target_file}"
                    else:
                        backup = self.code_mgr.backup_file(mod.target_file, tag)
                        result_entry["backup"] = backup
                        shutil.move(str(src), str(dst))
                        result_entry["success"] = True
                        result_entry["info"] = f"{mod.target_file} -> {mod.new_filename}"
                else:
                    result_entry["info"] = f"未知action: {mod.action}"

                if not result_entry["backup"]:
                    hist = self.code_mgr.get_modification_history(5)
                    for h in reversed(hist):
                        if h.get("file") == mod.target_file and h.get("backup"):
                            result_entry["backup"] = h["backup"]
                            break
            except Exception as e:
                result_entry["success"] = False
                result_entry["info"] = f"执行异常: {type(e).__name__}: {e}"
                print(f"         ❌ {result_entry['info']}")

            status_icon = "✅" if result_entry["success"] else "❌"
            print(f"         {status_icon} {result_entry['info']}")
            results.append(result_entry)
        return results

    def _validate_files(self, files: List[str]) -> Dict[str, Tuple[bool, str]]:
        """对修改过的文件做验证（语法+导入）。
        多文件时使用 code_mgr.validate_files_parallel 并发验证。
        """
        # 先去重（相同文件可能在多个 modification 中）
        unique_files = list(dict.fromkeys(files))
        if len(unique_files) <= 1:
            # 单文件走顺序逻辑
            results = {}
            for f in unique_files:
                if not f.endswith(".py"):
                    results[f] = (True, "非Python文件，跳过")
                    continue
                syntax_ok, syntax_msg = self.code_mgr.check_python_syntax(f)
                if not syntax_ok:
                    results[f] = (False, syntax_msg)
                    continue
                if Config.AUTO_IMPORT_TEST:
                    import_ok, import_msg = self.code_mgr.verify_module_import(f)
                    results[f] = (import_ok, f"{syntax_msg}; 导入:{import_msg}")
                else:
                    results[f] = (True, f"{syntax_msg}; 未执行导入测试")
            return results

        # 多文件：并发验证（语法+导入在 code_mgr.validate_files_parallel 内部做了）
        try:
            parallel_results = self.code_mgr.validate_files_parallel(unique_files)
        except Exception as e:
            print(f"      ⚠️ 并发验证异常，回退顺序验证: {e}")
            parallel_results = {}

        results: Dict[str, Tuple[bool, str]] = {}
        for f in unique_files:
            if not f.endswith(".py"):
                # 非 Python：validate_files_parallel 已返回 "语法正确（非Python，跳过导入测试）"
                results[f] = parallel_results.get(f, (True, "非Python文件，跳过"))
                continue

            if f in parallel_results:
                ok, info = parallel_results[f]
                # validate_files_parallel 返回的是合并 msg，如果 AUTO_IMPORT_TEST=False，再特殊处理一下
                if not Config.AUTO_IMPORT_TEST and ok and "导入失败" not in info:
                    info = info.split(" | ", 1)[0] + "; 未执行导入测试（按配置禁用）"
                results[f] = (ok, info)
            else:
                # 兜底顺序
                syntax_ok, syntax_msg = self.code_mgr.check_python_syntax(f)
                if not syntax_ok:
                    results[f] = (False, syntax_msg)
                    continue
                if Config.AUTO_IMPORT_TEST:
                    import_ok, import_msg = self.code_mgr.verify_module_import(f)
                    results[f] = (import_ok, f"{syntax_msg}; 导入:{import_msg}")
                else:
                    results[f] = (True, f"{syntax_msg}; 未执行导入测试")
        return results

    # ========== 用户交互 ==========
    def _user_confirm_modifications(self, plan: ModificationPlan) -> bool:
        """交互式询问用户是否确认应用修改"""
        print("\n      === 修改内容摘要 ===")
        for i, m in enumerate(plan.modifications, 1):
            action_cn = {
                "replace_all": "全量替换",
                "patch": "局部补丁",
                "new_file": "新建文件",
                "delete_file": "删除文件",
                "rename_file": "重命名",
            }.get(m.action, m.action)
            print(f"      {i}. [{action_cn}] {m.target_file}")
            if m.reason:
                print(f"         原因: {m.reason}")
            # 预览
            if m.action in ("replace_all", "new_file") and m.new_content:
                preview = m.new_content.splitlines()[:8]
                print(f"         内容预览 ({len(m.new_content.splitlines())}行):")
                for line in preview:
                    print(f"           | {line}")
                if len(m.new_content.splitlines()) > 8:
                    print(f"           | ...(共{len(m.new_content.splitlines())}行)")
            elif m.action == "patch":
                print(f"         将 {m.old_str.splitlines()[0].strip()[:60]}...")
                print(f"         替换为 {m.new_str.splitlines()[0].strip()[:60]}...")
            elif m.action == "insert_method":
                _code_lines = m.method_code.splitlines()
                print(f"         在 {m.class_name} 类末尾插入方法:")
                for line in _code_lines[:4]:
                    print(f"           | {line}")
                if len(_code_lines) > 4:
                    print(f"           | ...(共{len(_code_lines)}行)")

        while True:
            try:
                ans = input("\n      🤔 是否应用以上修改？(y=是 / n=否 / v=查看diff预览) [y/n/v]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            if ans == "y":
                return True
            if ans == "n":
                return False
            if ans == "v":
                self._preview_modification_diffs(plan)
            else:
                print("         请输入 y、n 或 v")

    def _preview_modification_diffs(self, plan: ModificationPlan):
        """（粗略）预览修改差异"""
        for m in plan.modifications:
            if m.action in ("replace_all", "patch"):
                old_content = self.code_mgr.read_file(m.target_file) or ""
                if m.action == "replace_all":
                    new_content = m.new_content
                else:
                    new_content = old_content.replace(m.old_str, m.new_str, 1) if m.old_str in old_content else old_content
                import difflib
                diff = difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile=f"a/{m.target_file}",
                    tofile=f"b/{m.target_file}",
                    lineterm="",
                )
                diff_text = "".join(diff)
                if diff_text:
                    print(f"\n      --- {m.target_file} diff ---")
                    lines = diff_text.splitlines()[:60]
                    print("\n".join(lines))
                    if len(diff_text.splitlines()) > 60:
                        print(f"      ...(还有{len(diff_text.splitlines())-60}行diff)")

    def _print_iteration_summary(self, result: Dict):
        """打印本次迭代摘要"""
        print(f"\n{'='*60}")
        print(f"  📊 第 {result['iteration']} 次迭代摘要")
        print(f"  状态: {result['status']}")
        print(f"  修改建议: {result['suggestion_count']} 项")
        print(f"  成功应用: {result['applied_count']} 项")
        print(f"  应用失败: {result['failed_count']} 项")
        if result.get("issues"):
            print(f"  ⚠️  问题 ({len(result['issues'])}):")
            for p in result["issues"][:5]:
                print(f"    - {p[:100]}")
        print(f"{'='*60}\n")

    # ========== 历史记录 ==========
    def _load_last_iteration(self) -> int:
        """读取历史中的最大迭代号"""
        if Config.EVOLUTION_LOG.exists():
            try:
                with open(Config.EVOLUTION_LOG, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        return max(item.get("iteration", 0) for item in data)
            except Exception:
                pass
        return 0

    def _load_history(self) -> List[Dict]:
        """加载迭代历史"""
        if Config.EVOLUTION_LOG.exists():
            try:
                with open(Config.EVOLUTION_LOG, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception:
                pass
        return []

    def _save_iteration_log(self, result: Dict):
        """保存本次迭代日志"""
        self.iteration_logs.append(result)
        try:
            with open(Config.EVOLUTION_LOG, "w", encoding="utf-8") as f:
                json.dump(self.iteration_logs, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"[警告] 保存迭代日志失败: {e}")

    # ========== 对话模式（非自迭代，普通聊天） ==========
    def chat_with_doubao_about_code(self, question: str, stream: bool = True) -> str:
        """
        带着当前代码上下文向豆包提问（不自动修改代码）
        """
        files_for_qa = _filter_ple_protected(Config.SELF_EVOLVE_FILES)
        files_content = self.code_mgr.format_files_for_prompt(files_for_qa)
        sys_prompt = (
            "你是一个资深Python代码专家。用户会向你咨询关于当前项目代码的问题。\n"
            "请结合用户提供的项目源码进行分析和回答。\n"
            "如果发现可以改进的地方，在回答末尾给出建议，但不要主动输出JSON格式修改指令。"
        )
        user_prompt = (
            "当前项目源代码:\n"
            + files_content
            + f"\n\n我的问题: {question}"
        )
        return self.doubao.chat(user_message=user_prompt, system_prompt=sys_prompt, stream=stream)

    # ========== 工具方法 ==========
    def show_last_diff(self):
        """展示最近一次修改的diff"""
        if not self._last_applied_files:
            print("（暂无最近修改的diff记录）")
            return
        for fpath, backup in self._last_applied_files.items():
            diff = self.code_mgr.generate_diff(fpath, backup)
            if diff:
                print(f"\n--- {fpath} (对比备份 {Path(backup).name}) ---")
                print(diff)
            else:
                print(f"\n--- {fpath}: 无法生成diff ---")

    def list_backups(self):
        """列出所有备份文件"""
        backups = self.code_mgr.list_backups()
        if not backups:
            print("（暂无备份文件）")
            return
        print(f"共有 {len(backups)} 个备份文件:")
        for b in backups[-20:]:  # 只显示最近20个
            size_kb = b.stat().st_size / 1024
            print(f"  {b.name}  ({size_kb:.1f} KB)")
        if len(backups) > 20:
            print(f"  ... 更早的 {len(backups)-20} 个文件未显示")

    def restore_file_from_backup(self, pattern: str):
        """
        从备份恢复文件
        :param pattern: 备份文件名的关键字（如文件名、时间戳、iter标签）
        """
        backups = self.code_mgr.list_backups()
        matched = [b for b in backups if pattern in b.name]
        if not matched:
            print(f"找不到匹配 '{pattern}' 的备份")
            return
        # 取最后一个（最新的）
        latest = matched[-1]
        if self.code_mgr.restore_backup(str(latest)):
            print(f"✅ 已从备份恢复: {latest.name}")
        else:
            print(f"❌ 恢复失败")

    def get_status_report(self) -> Dict:
        """获取当前系统状态报告"""
        from collections import Counter
        status_counts = Counter(r.get("status", "未知") for r in self.iteration_logs)
        total_applied = sum(r.get("applied_count", 0) for r in self.iteration_logs)
        return {
            "project_root": str(Config.PROJECT_ROOT),
            "api_configured": Config.is_api_configured() and (len(Config.ARK_API_KEY) > 0 if hasattr(Config, 'ARK_API_KEY') else False),
            "model_id": getattr(self.doubao, "model_id", None)
                        or (Config.OLLAMA_MODEL if Config.CLIENT_MODE == "local"
                            else Config.LMSTUDIO_MODEL if Config.CLIENT_MODE == "lmstudio"
                            else Config.MODEL_ID),
            "total_iterations": self.current_iteration,
            "iteration_status_distribution": dict(status_counts),
            "total_applied_modifications": total_applied,
            "backup_count": len(self.code_mgr.list_backups()),
            "auto_apply": Config.AUTO_APPLY_MODIFICATIONS,
            "self_evolve_files": Config.SELF_EVOLVE_FILES,
        }
