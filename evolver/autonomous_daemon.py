"""
AutonomousDaemon: 全自动主循环 — 无限循环、无人介入

完整编排流程（单周期）：
  1. 安全闸门：kill-switch 文件 + 资源检查（GPU/内存/Ollama）
  2. Meta-Drive 全量巡检 → 生成风险等级 + 建议列表
  3. 从 Meta-Drive 建议中推导本轮迭代目标
  4. Self-Evolver.run_iteration(auto_apply=True) 执行代码修改
  5. 超脱锚点时间漂移检测
  6. 认知熵缓冲池：解析待处理共存模拟记录
  7. SEO 涌现库：后置校验，决定是否纳入演化候选池
  8. Meta-Drive 静默窗口概率调度
  9. 演化调度器：注册本轮分支 + 定期淘汰
  10. 写入 daemon_log.json + sleep(min_cycle_interval) → 返回 1

安全机制：
  - 外部 kill-switch: 检测 logs/STOP_DAEMON 文件
  - 单步 try/except: 一个模块崩溃不影响全循环
  - 最小周期间隔: 60s 防止 LLM 连续失败时 tight-loop
  - 资源门控: GPU 温度>85°C 或 Ollama 不可达时自动暂停
  - 零进展熔断: 连续 5 轮无修改时强制注入本体论漂移目标
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from config import Config as _Cfg

VEIL_TRIGGER_THRESHOLD = 0.45


PROJECT_ROOT = Path(__file__).parent
# 使用 Config.LOG_DIR 统一路径，避免 main.py 清除的和 daemon 检查的不是同一个文件
try:
    from config import Config as _Cfg
    LOG_DIR = _Cfg.LOG_DIR
except Exception:
    LOG_DIR = PROJECT_ROOT / "logs"
DAEMON_LOG = LOG_DIR / "daemon_log.json"
KILL_SWITCH = LOG_DIR / "STOP_DAEMON"

# 并行线程防护：Stage 4-7 超时/异常/僵尸线程回收
try:
    from evolver.parallel_guard import run_parallel, cleanup_zombies, get_thread_count
except Exception:
    # 防护模块本身缺失时降级为空实现，不阻断 daemon 启动
    def run_parallel(tasks, timeout=30, max_workers=4):
        out = {}
        for name, func in tasks:
            try:
                r = func()
                out[name] = r if isinstance(r, dict) else {"_result": r}
            except Exception as e:
                out[name] = {"_error": f"{type(e).__name__}: {e}"}
        return out
    def cleanup_zombies(): return 0
    def get_thread_count(): import threading; return threading.active_count()

# Stage 开关默认超时 (秒) — 单个并行 Stage 最长执行时间
PARALLEL_STAGE_TIMEOUT = 30

MIN_CYCLE_INTERVAL_SEC = 2       # 每轮最小间隔（防止 tight-loop）
MAX_CYCLE_INTERVAL_SEC = 180    # 资源充足时的默认间隔
OBJECTIVE_COMPLETE_SUCCESSES = 2  # ★ 闭环修复：同一目标连续成功 N 次即结算为「已完成」
NO_PROGRESS_THRESHOLD = 999       # 不再强制漂移——让 AI 自由选择方向
GPU_TEMP_THRESHOLD = 85.0       # GPU 温度告警阈值
OLLAMA_TIMEOUT_SEC = 3          # Ollama 连通性检查超时

# 沙箱连续拦截后的安全降级目标池（低风险、模型易出稳定 JSON）
_SAFE_OBJECTIVES = [
    "跨文件重构：将分散在多个文件中的重复逻辑抽取为独立函数，并迁移调用点",
    "架构改进：识别核心类的耦合点，引入依赖注入或职责拆分降低耦合",
    "接口一致性：统一关键模块的公共接口命名、返回值结构和异常类型",
    "性能优化：定位热点代码路径，减少冗余计算或不必要的 I/O 操作",
    "可测试性：将复杂函数拆分为纯函数，补充单元测试友好的输入输出边界",
]

# MID 风险目标池 (随机选择,避免固定死循环)
_MID_OBJECTIVES = [
    "中等风险：重新设计两个以上模块之间的协作接口，消除重复职责",
    "中等风险：将大函数拆分为多个职责单一的小函数，降低圈复杂度",
    "中等风险：引入缓存或惰性求值，减少重复计算和外部 I/O",
    "中等风险：统一错误处理策略，将散落的异常捕获集中到公共装饰器/上下文",
    "中等风险：梳理配置读取与默认值兜底逻辑，消除 hard-coded 默认值",
]


class AutonomousDaemon:
    """全自动无限循环守护进程"""

    @staticmethod
    def _extract_target_file(objective: str) -> str:
        """从目标文本中提取实际要处理的文件路径"""
        import re
        match = re.search(r'(evolver/\w+\.py)', objective)
        if match:
            return match.group(1)
        return ""

    def __init__(
        self,
        evolver=None,
        meta_drive=None,
        anchor=None,
        conflict_boundary=None,
        seo=None,
        scheduler=None,
        miner=None,
        perception=None,
        decoupler=None,
        veil=None,
    ):
        self.evolver = evolver
        self.meta_drive = meta_drive
        self.anchor = anchor
        self.conflict_boundary = conflict_boundary
        self.seo = seo
        self.scheduler = scheduler
        self.miner = miner
        self.perception = perception
        self.decoupler = decoupler      # L3 元认知: 失败模式分析 + 本体论漂移
        self.veil = veil                # 面具检测: 概念遮蔽双路编码
        self._kill_switch_triggered = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._history: List[Dict[str, Any]] = []
        self._consecutive_no_change = 0
        self._cycle_count = 0
        self._start_time: Optional[float] = None

        # ── Cognitive Feedback Loop 状态 ──
        self._consecutive_failures: Dict[str, int] = {}  # per-file 连续失败计数
        self._cognitive_state: Dict[str, Any] = {
            "temperature_override": None,   # None=默认, float=覆盖值
            "prompt_mode": "feature",       # "feature"=新增功能, "fix"=修复bug
            "feedback_active": False,       # 是否有活跃的反馈调整
        }
        # ── 修复3：目标重复检测状态 ──
        self._recent_objectives: List[str] = []   # 最近N轮目标，用于去重
        self._completed_objectives: set = set()   # 已标记为完成的目标（AI说无需修改的）
        self._objective_hits: Dict[str, int] = {} # ★ 闭环修复：同一目标连续成功次数（成功2次即结算完成）
        self._lock = threading.Lock()
        self._last_status = "未启动"
        self._last_gpu_temp: Optional[float] = None
        # 风险2: 领域迁移攻击隔离队列
        self._objective_quarantine: List[Dict[str, Any]] = []
        # 沙箱连续隔离计数（≥3 次后降级为安全目标打破死锁）
        self._quarantine_streak: int = 0

        LOG_DIR.mkdir(exist_ok=True)
        self._load_history()
        # ===== 新增：配置热重载支持 =====
        from config import Config as _Cfg
        self._config_mtime = _Cfg.get_config_mtime()
        # ================================

    # ========== 生命周期 ==========

    def start(self) -> bool:
        """启动守护线程（幂等）"""
        if self._thread and self._thread.is_alive():
            return False

        # 开启 PLE 保护：禁止自迭代修改核心引擎文件
        if self.evolver and hasattr(self.evolver, 'code_mgr'):
            self.evolver.code_mgr.daemon_protection = True

        self._stop_event.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._daemon_loop,
            name="AutonomousDaemon",
            daemon=True,
        )
        self._thread.start()
        self._last_status = "运行中"
        print("  🔄 全自动主循环已启动（AutonomousDaemon）")
        print("     外部停止: echo stop > logs/STOP_DAEMON")
        print("     内部停止: daemon stop")
        return True

    def stop(self):
        """请求停止（非阻塞）"""
        self._stop_event.set()
        self._last_status = "停止中"
        # 关闭 PLE 保护
        if self.evolver and hasattr(self.evolver, 'code_mgr'):
            self.evolver.code_mgr.daemon_protection = False

    def request_stop(self):
        """停止请求别名（main.py CLI 调用的接口），与 stop() 等价"""
        self.stop()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def running(self) -> bool:
        return self.is_running()

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            uptime = int(time.time() - self._start_time) if self._start_time else 0
            # 审计：最近 10 轮的安全机制触发情况
            recent_cycles = self._history[-10:]
            sandbox_quarantine_count = sum(
                1 for h in recent_cycles
                if h.get("stages", {}).get("sandbox", {}).get("quarantined")
            )
            evolution_isolation_count = sum(
                1 for h in recent_cycles
                if h.get("stages", {}).get("evolution", {}).get("isolation")
            )
            force_flush_count = sum(
                1 for h in recent_cycles
                if h.get("stages", {}).get("entropy_buffer", {}).get("force_flush")
            )

            return {
                "running": self.is_running(),
                "status": self._last_status,
                "cycle_count": self._cycle_count,
                "uptime_sec": uptime,
                "uptime_str": self._format_uptime(uptime),
                "consecutive_no_change": self._consecutive_no_change,
                "last_cycle": self._history[-1]["timestamp"] if self._history else None,
                "history_size": len(self._history),
                "kill_switch_present": KILL_SWITCH.exists(),
                # 安全审计字段
                "quarantine_size": len(self._objective_quarantine),
                "quarantine_recent": self._objective_quarantine[-3:] if self._objective_quarantine else [],
                "security_audit": {
                    "sandbox_quarantine_last10": sandbox_quarantine_count,
                    "evolution_isolation_last10": evolution_isolation_count,
                    "entropy_force_flush_last10": force_flush_count,
                },
            }

    def print_status(self):
        s = self.get_status()
        print()
        print("=" * 50)
        print("  🔄 AutonomousDaemon 全自动主循环")
        print("=" * 50)
        icon = "🟢" if s["running"] else "⚪"
        print(f"  状态:         {icon} {s['status']}")
        print(f"  运行时长:     {s['uptime_str']}")
        print(f"  已完成周期:   {s['cycle_count']}")
        print(f"  连续零进展:   {s['consecutive_no_change']} / {NO_PROGRESS_THRESHOLD}")
        print(f"  历史记录:     {s['history_size']} 条")
        if s["last_cycle"]:
            print(f"  最近周期:     {s['last_cycle'][:19]}")
        print(f"  Kill-Switch:  {'⚠️ 已触发' if s['kill_switch_present'] else '未触发'}")
        # 安全审计
        audit = s.get("security_audit", {})
        quarantine_size = s.get("quarantine_size", 0)
        if audit or quarantine_size:
            print("  ── 安全审计（最近 10 轮）──")
            print(f"  🛡️ 沙箱隔离: {audit.get('sandbox_quarantine_last10', 0)} 次 (队列 {quarantine_size})")
            print(f"  🔒 演化隔离: {audit.get('evolution_isolation_last10', 0)} 次")
            print(f"  🔥 强制flush: {audit.get('entropy_force_flush_last10', 0)} 次")

        # ── 4.1 进化健康度指标 ──
        health = self._compute_health_metrics()
        if health:
            print("  ── 进化健康度 ──")
            print(f"  📊 合约通过率:   {health['contract_pass_rate']:.0%} (最近 {health['sample_size']} 次)")
            print(f"  🚫 黑名单占比:   {health['blacklist_ratio']:.0%} ({health['blacklisted_count']} 个模块)")
            print(f"  ⏱️ 迭代均耗时:   {health['avg_cycle_time']:.1f}s")
            print(f"  ⚡ Kill-Switch:  {health['kill_switch_count']} 次历史触发")
            stub_ct = health.get('stub_count', 0)
            stub_cov = health.get('stub_coverage', 0)
            if stub_ct > 0:
                print(f"  🔧 Auto-Stub:    {stub_ct} 个待实现 (coverage={stub_cov:.0%})")

        print("=" * 50)

    @staticmethod
    def _format_uptime(sec: int) -> str:
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m:02d}m{s:02d}s"

    def _compute_health_metrics(self) -> Dict[str, Any]:
        """
        4.1 进化健康度指标：从 history + blacklist 计算实时指标。
        返回 dict 或空 dict（数据不足时）。
        """
        import json as _json
        metrics: Dict[str, Any] = {}

        # 1. 合约通过率（从 evolution_history 最近 50 次）
        try:
            hist_path = LOG_DIR / "evolution_history.json"
            if hist_path.exists():
                with open(hist_path, "r", encoding="utf-8") as f:
                    hist = _json.load(f)
                if isinstance(hist, list) and hist:
                    recent = hist[-50:]
                    contract_ok = sum(1 for it in recent
                                      if isinstance(it, dict) and
                                      "Contract" not in str(it.get("status", "")) and
                                      "失败" not in str(it.get("status", "")))
                    metrics["contract_pass_rate"] = contract_ok / len(recent) if recent else 0
                    metrics["sample_size"] = len(recent)

                    # 2. 迭代均耗时（从时间戳差值）
                    if len(recent) >= 2:
                        times = []
                        for it in recent:
                            ts = it.get("timestamp", "") or it.get("time", "")
                            if ts:
                                try:
                                    from datetime import datetime as _dt
                                    times.append(_dt.fromisoformat(ts).timestamp())
                                except Exception:
                                    pass
                        if len(times) >= 2:
                            diffs = [times[i] - times[i-1] for i in range(1, len(times))]
                            diffs = [d for d in diffs if 0 < d < 3600]  # 过滤异常值
                            if diffs:
                                metrics["avg_cycle_time"] = sum(diffs) / len(diffs)
        except Exception:
            pass

        # 3. 黑名单占比
        try:
            bl_path = LOG_DIR / "modification_blacklist.json"
            if bl_path.exists():
                with open(bl_path, "r", encoding="utf-8") as f:
                    bl = _json.load(f)
                banned = sum(1 for v in bl.values()
                             if isinstance(v, dict) and v.get("banned_until_iter", 0) > 0)
                total = len(bl) if bl else 0
                metrics["blacklisted_count"] = banned
                metrics["blacklist_ratio"] = banned / total if total > 0 else 0
        except Exception:
            pass

        # 4. Kill-Switch 历史触发次数（从 archive 目录统计）
        try:
            archive_dir = LOG_DIR / "archive"
            if archive_dir.exists():
                ks_count = len(list(archive_dir.glob("blacklist_*.json")))
                metrics["kill_switch_count"] = ks_count
        except Exception:
            pass

        # 5. Auto-Stub 统计（从合约报告获取 stub_coverage）
        try:
            from contracts import ContractSuite
            suite = ContractSuite()
            # 只统计 stub 数量，不重跑全部合约（用上次缓存或快速检查）
            stub_count = suite._count_stubs()
            if stub_count > 0:
                metrics["stub_count"] = stub_count
                metrics["stub_coverage"] = stub_count / max(metrics.get("sample_size", 16), 1)
            else:
                metrics["stub_coverage"] = 0
        except Exception:
            metrics["stub_coverage"] = 0

        # 默认值
        metrics.setdefault("contract_pass_rate", 0)
        metrics.setdefault("sample_size", 0)
        metrics.setdefault("blacklist_ratio", 0)
        metrics.setdefault("blacklisted_count", 0)
        metrics.setdefault("avg_cycle_time", 0)
        metrics.setdefault("kill_switch_count", 0)

        return metrics

    # ========== Cognitive Feedback Loop（认知反馈回路）==========

    def _apply_cognitive_feedback(self, l3_result: Dict[str, Any]) -> None:
        """
        项3: 认知反馈回路 — L3 检测到重复失败模式时自动调整策略。
        - 连续 3 轮同一文件失败 → 下调 temperature(0.45→0.25) + 切 Prompt 为修复模式
        - 连续 10 轮无进展 → 触发 kill-switch
        """
        if not l3_result or l3_result.get("skipped"):
            return

        try:
            # 1. 从 L3 结果提取失败模式
            failure_patterns = l3_result.get("failure_patterns") or []
            drift = l3_result.get("drift_objective")

            # 2. 检查是否有重复失败模式
            repeated_failures = []
            for pattern in failure_patterns:
                if isinstance(pattern, dict):
                    file_path = pattern.get("file", "")
                    fail_count = pattern.get("count", 0)
                    if fail_count >= 3:
                        repeated_failures.append((file_path, fail_count))

            # 3. 应用反馈：连续失败 ≥3 → 下调 temperature + 切修复模式
            if repeated_failures:
                if not self._cognitive_state["feedback_active"]:
                    print("  🧠 [Cognitive Feedback] 检测到重复失败模式，激活反馈回路")
                self._cognitive_state["feedback_active"] = True
                self._cognitive_state["temperature_override"] = 0.25
                self._cognitive_state["prompt_mode"] = "fix"

                # 更新 per-file 连续失败计数
                for file_path, fail_count in repeated_failures:
                    self._consecutive_failures[file_path] = fail_count
                    print(f"    📉 {file_path}: 连续失败 {fail_count} 次 → temp=0.25, 修复模式")

                # 拉高该文件的 ban_rounds（通过 code_manager）
                if self.evolver and hasattr(self.evolver, 'code_mgr'):
                    try:
                        bl = self.evolver.code_mgr._blacklist
                        for file_path, fail_count in repeated_failures:
                            if file_path in bl:
                                bl[file_path]["banned_until_iter"] = (
                                    self._cycle_count + 20)  # 从 5 轮拉到 20 轮
                                print(f"    🔒 {file_path}: ban_rounds 5→20")
                    except Exception:
                        pass
            else:
                # 无重复失败 → 恢复默认
                if self._cognitive_state["feedback_active"]:
                    print("  🧠 [Cognitive Feedback] 失败模式已消除，恢复默认策略")
                self._cognitive_state["feedback_active"] = False
                self._cognitive_state["temperature_override"] = None
                self._cognitive_state["prompt_mode"] = "feature"
                self._consecutive_failures.clear()

            # 4. 连续 10 轮无进展 → 触发 kill-switch
            #    ★ 修复1: _consecutive_no_change 不从历史加载，只统计当前实例生命周期
            if self._consecutive_no_change >= 10:
                print(f"  ⛔ [Cognitive Feedback] 当前实例连续 {self._consecutive_no_change} 轮无进展，触发 kill-switch")
                try:
                    kill_file = LOG_DIR / "STOP_DAEMON"
                    kill_file.write_text(
                        f"cognitive_feedback: consecutive_no_change={self._consecutive_no_change}",
                        encoding="utf-8")
                except Exception:
                    pass

        except Exception as e:
            print(f"  ⚠️ [Cognitive Feedback] 异常: {type(e).__name__}: {e}")

    def _update_failure_tracking(self, iter_result: Dict[str, Any]) -> None:
        """
        迭代完成后更新失败追踪状态（供主循环调用）。
        ★ 闭环修复：修正 modified_files 从未被透传的问题（原代码恒为空，
           per-file 连续失败计数形同虚设）；同时把结果同步给 PriorityPlanner，
           让成功清空失败计数、失败抬升优先级。
        """
        try:
            status = iter_result.get("status", "")
            modified_files = iter_result.get("modified_files") or []
            if not modified_files:
                details = iter_result.get("applied_details") or []
                modified_files = [
                    d.get("file") for d in details
                    if isinstance(d, dict) and d.get("success") and d.get("file")
                ]

            if "失败" in status or "未通过" in status:
                # 失败：增加 per-file 计数
                for f in modified_files:
                    self._consecutive_failures[f] = self._consecutive_failures.get(f, 0) + 1
                    try:
                        from priority_planner import get_priority_planner
                        get_priority_planner().record_outcome(f, success=False)
                    except Exception:
                        pass
            elif "成功" in status:
                # 成功：重置 per-file 计数
                for f in modified_files:
                    if f in self._consecutive_failures:
                        del self._consecutive_failures[f]
                    try:
                        from priority_planner import get_priority_planner
                        get_priority_planner().record_outcome(f, success=True)
                    except Exception:
                        pass
        except Exception:
            pass

    # ========== 主循环 ==========

    def _daemon_loop(self):

        """主循环 — 无限执行完整编排流程"""
        print("\n  🌀 AutonomousDaemon 主循环开始...")
        
        cycle_interval = MIN_CYCLE_INTERVAL_SEC

        while not self._stop_event.is_set():
            # ===== 新增：配置热重载检查 =====
            try:
                reloaded, new_mtime, changes = _Cfg.reload_if_changed(self._config_mtime)
                if reloaded:
                    self._config_mtime = new_mtime
                    print(f"\n  🔄 [Hot-Reload] 配置文件已更改，自动重载:")
                    for key, value in changes.items():
                        if isinstance(value, list):
                            print(f"    - {key}:")
                            for item in value:
                                print(f"        {item}")
                        else:
                            print(f"    - {key}: {value}")
                    
                    # 应用到客户端
                    if self.evolver and hasattr(self.evolver, 'doubao'):
                        applied = _Cfg.apply_config_to_client(self.evolver.doubao)
                        if applied:
                            print(f"    [客户端] 已应用:")
                            for key, value in applied.items():
                                print(f"        {key}: {value}")
            except Exception as e:
                print(f"  ⚠️ [Hot-Reload] 配置重载异常: {e}")
            # ===================================
            cycle_start = time.time()
            cycle_log: Dict[str, Any] = {
                "cycle": self._cycle_count + 1,
                "timestamp": datetime.now().isoformat(),
                "stages": {},
                "success": True,
                "errors": [],
            }

            try:
                # ── 每轮开始：僵尸线程巡检回收 ──
                if cleanup_zombies() > 0:
                    print(f"  🧹 [ParallelGuard] 回收僵尸线程, 当前活跃 {get_thread_count()}")

                # ── Stage 0: 安全闸门 ──
                if _Cfg.is_stage_enabled("stage_0_safety"):
                    safety = self._stage_safety_gate()
                    cycle_log["stages"]["safety_gate"] = safety

                    if safety.get("kill_switch"):
                        self._kill_switch_triggered = True  
                        self._last_status = "kill-switch 触发，主动退出"
                        print("\n  ⛔ Kill-switch 文件检测到，主循环主动退出")
                        # Tier-2 回滚检查：观察期内的修改自动回滚
                        if self.evolver and hasattr(self.evolver, 'code_mgr'):
                            try:
                                rolled = self.evolver.code_mgr.check_tier2_rollback(self._cycle_count)
                                if rolled:
                                    print(f"  🔄 [Tier-2] 回滚 {len(rolled)} 个观察期文件: {rolled}")
                            except Exception:
                                pass
                        # 停机自净：归档并清空黑名单，避免历史残留导致下次启动误判
                        self._archive_and_clear_blacklist()
                        break

                    if not safety.get("ple_ok", True):
                        ple_msg = safety.get("ple_msg", "未知 PLE 错误")
                        cycle_log["success"] = False
                        cycle_log["errors"].append(f"PLE 完整性校验失败: {ple_msg}")
                        self._last_status = f"PLE 篡改检测: {ple_msg}"
                        print(f"\n  ⚠️ PLE 完整性校验异常 → 自动恢复后继续循环")
                        print(f"     原因: {ple_msg}")
                        print(f"     💡 基线将自动重建，daemon 继续运行")
                        self._safe_sleep(10)
                        continue

                    if not safety.get("service_ok"):
                        cycle_log["success"] = False
                        cycle_log["errors"].append("推理服务不可达，跳过本轮")
                        self._safe_sleep(30)
                        continue
                else:
                    # Stage 0 关闭：用安全默认值保证后续 stage 不崩
                    safety = {"kill_switch": False, "ple_ok": True, "service_ok": True,
                              "gpu_temp": None, "gpu_overheated": False, "skipped": True}

                # ── Stage 0.5: 动态参数联动（基于安全闸门结果）──
                if _Cfg.is_stage_enabled("stage_0_5_param_linkage"):
                    self._dynamic_param_linkage(safety)

                # ── Stage 0.5+: L1 感知层快照（零开销，采集能耗+健康状态）──
                if _Cfg.is_stage_enabled("stage_0_5_perception"):
                    percept_snap = self._stage_perception_snapshot()
                    cycle_log["stages"]["perception"] = percept_snap

                # ── Stage 1: Meta-Drive 全量巡检 ──
                if _Cfg.is_stage_enabled("stage_1_meta_drive"):
                    inspection = self._stage_meta_drive_inspection()
                    cycle_log["stages"]["meta_drive"] = inspection
                else:
                    inspection = {"risk_level": "UNKNOWN", "recommendations": [], "skipped": True}

                # ── Stage 1.5: L3 元认知层（零开销，失败模式分析+本体论漂移）──
                if _Cfg.is_stage_enabled("stage_1_5_l3_metacog"):
                    l3_result = self._stage_l3_metacognition(inspection)
                    cycle_log["stages"]["l3_metacognition"] = l3_result
                else:
                    l3_result = {"skipped": True}

                # ── Stage 1.6: Cognitive Feedback Loop（认知反馈回路）──
                #   L3 检测到重复失败模式 → 自动调整 temperature/ban_rounds/prompt_mode
                self._apply_cognitive_feedback(l3_result)
                cycle_log["stages"]["cognitive_feedback"] = dict(self._cognitive_state)

                # ── Stage 2: 推导迭代目标 ──
                if _Cfg.is_stage_enabled("stage_2_derive_objective"):
                    objective = self._derive_objective(inspection)
                    # L3 漂移触发时覆盖目标
                    if l3_result.get("drift_objective"):
                        objective = l3_result["drift_objective"]
                        print(f"  🔄 [L3] 目标已被本体论漂移覆盖")
                    cycle_log["stages"]["objective"] = objective
                else:
                    # Stage 2 关闭：从 MID 池随机抽一个目标，避免后续 stage 无目标可跑
                    objective = random.choice(_MID_OBJECTIVES)
                    cycle_log["stages"]["objective"] = f"[stage_2 disabled] {objective}"

                # ── Stage 2.2: 填充本地代码熵上下文（确保沙箱熵差计算有参照基准）──
                # 多文件混合的熵更稳定；只取 800 字符会导致 local_entropy≈0，
                # 进而 gap=external-0≈1.0，正常目标被误判为领域迁移攻击隔离。
                if _Cfg.is_stage_enabled("stage_2_2_local_entropy") and \
                   self.conflict_boundary and self.evolver and hasattr(self.evolver, 'code_mgr'):
                    try:
                        from config import Config as _CfgLocal
                        code_snippets = self.evolver.code_mgr.format_files_for_prompt(
                            getattr(_CfgLocal, 'SELF_EVOLVE_FILES', [])
                        )
                        if code_snippets:
                            # update_local_context 内部已截 500/条，这里多取几条
                            # 保证本地熵样本量足够（3000 字符 ≈ 6 条 update 调用）
                            chunk_size = 500
                            for i in range(0, min(len(code_snippets), 3000), chunk_size):
                                self.conflict_boundary.update_local_context(code_snippets[i:i+chunk_size])
                    except Exception:
                        pass

                # ── Stage 2.5 前: 面具检测（限频每 5 轮，对目标做概念遮蔽检测）──
                if _Cfg.is_stage_enabled("stage_2_5_veil_check"):
                    veil_result = self._stage_veil_check(objective)
                    cycle_log["stages"]["veil_check"] = veil_result
                    # P1‑2：面具遮蔽过高，强制切换为安全目标
                    if (not veil_result.get("skipped")) and veil_result.get("veil_level",0.0) >= VEIL_TRIGGER_THRESHOLD:
                        fallback_obj = self._generate_alternative_objective()
                        print(f"  🎭 Veil触发，概念遮蔽过高 veil_level={veil_result['veil_level']:.2f}，切换到轮替目标")
                        objective = fallback_obj
                        cycle_log["stages"]["objective"] = objective
                        veil_result["switched_objective"] = True
                        veil_result["switched_to"] = fallback_obj


                # ── Stage 2.5: 沙箱校验（领域迁移攻击检测）──
                if _Cfg.is_stage_enabled("stage_2_5_sandbox"):
                    sandbox = self._sandbox_validate_objective(objective, is_drift=(self._consecutive_no_change >= NO_PROGRESS_THRESHOLD))
                    cycle_log["stages"]["sandbox"] = sandbox

                    if sandbox.get("quarantined"):
                        self._quarantine_streak += 1
                        print(f"    🛡️ 目标被沙箱隔离 (连续 {self._quarantine_streak} 次) → quarantine #{len(self._objective_quarantine)}")

                        if self._quarantine_streak >= 3:
                            # ── 打破死锁：从轮替目标池选一个 ──
                            fallback = self._generate_alternative_objective()
                            print(f"    🔁 连续隔离 ≥3 次 → 切换到轮替目标")
                            objective = fallback
                            self._quarantine_streak = 0
                            cycle_log["stages"]["objective"] = objective
                            # 降级后不再沙箱校验，直接执行
                            iter_result = self._stage_self_iteration(objective) if _Cfg.is_stage_enabled("stage_3_self_iteration") else {"skipped": True, "applied_count": 0}
                            cycle_log["stages"]["self_iteration"] = iter_result
                            applied = iter_result.get("applied_count", 0)
                            rolled_back = iter_result.get("rollback", False)
                            if applied == 0 or rolled_back:
                                self._consecutive_no_change += 1
                            else:
                                self._consecutive_no_change = 0
                        else:
                            self._consecutive_no_change += 1
                            iter_result = {"skipped": True, "applied_count": 0, "quarantined_streak": self._quarantine_streak}
                    else:
                        # 沙箱通过 → 重置 streak
                        if self._quarantine_streak > 0:
                            self._quarantine_streak = 0
                        # ── Stage 3: 执行自迭代 ──
                        if _Cfg.is_stage_enabled("stage_3_self_iteration"):
                            iter_result = self._stage_self_iteration(objective)
                        else:
                            iter_result = {"skipped": True, "applied_count": 0}
                        cycle_log["stages"]["self_iteration"] = iter_result
                        applied = iter_result.get("applied_count", 0)
                        rolled_back = iter_result.get("rollback", False)
                        if applied == 0 or rolled_back:
                            self._consecutive_no_change += 1
                        else:
                            self._consecutive_no_change = 0
                else:
                    # 沙箱关闭：直接执行 Stage 3，不拦截目标
                    if self._quarantine_streak > 0:
                        self._quarantine_streak = 0
                    if _Cfg.is_stage_enabled("stage_3_self_iteration"):
                        iter_result = self._stage_self_iteration(objective)
                    else:
                        iter_result = {"skipped": True, "applied_count": 0}
                    cycle_log["stages"]["self_iteration"] = iter_result
                    applied = iter_result.get("applied_count", 0)
                    rolled_back = iter_result.get("rollback", False)
                    if applied == 0 or rolled_back:
                        self._consecutive_no_change += 1
                    else:
                        self._consecutive_no_change = 0

                # ── Stage 3.2: 打印反思报告（如果有）──
                if _Cfg.is_stage_enabled("stage_3_2_reflection"):
                    reflection_text = iter_result.get("reflection") if isinstance(iter_result, dict) else None
                    if reflection_text:
                        print()
                        print(reflection_text)

                # ── Stage 3.15: Tier-2 观察期回滚（每轮结束都检查，不只是 kill-switch 时）──
                if _Cfg.is_stage_enabled("stage_tier2_observation") and self.evolver and self.evolver.code_mgr:
                    try:
                        tier2_rolled = self.evolver.code_mgr.check_tier2_rollback(
                            self._cycle_count + 1,
                            kill_switch_triggered=self._kill_switch_triggered  # ← 使用真实标志
                        )
                        if tier2_rolled:
                            print(f"  🛡️ [Tier-2] 观察期内触发异常，已自动回滚")
                            cycle_log["tier2_rolled_back"] = True
                    except Exception as tier2_err:
                        print(f"  ⚠️ [Tier-2] 回滚检查异常: {type(tier2_err).__name__}: {tier2_err}")

                # ── Stage 3.2+: L2 代价挖掘（限频每 10 轮，有修改时才调用 LLM）──
                if _Cfg.is_stage_enabled("stage_3_2_motive_mining"):
                    mining_result = self._stage_motive_mining(iter_result)
                    cycle_log["stages"]["motive_mining"] = mining_result

                # ── Stage 3.3: ModelProfile 运行时观测（每 5 轮）──
                if _Cfg.is_stage_enabled("stage_3_3_model_profile") and \
                   self._cycle_count > 0 and (self._cycle_count + 1) % 5 == 0:
                    self._print_model_profile_observation(iter_result)

                # ── Stage 4/5/6/7 并发执行（接入 parallel_guard 超时防护）──
                # 每个 Stage 独立超时，单个卡死不会拖垮整个 daemon
                parallel_tasks = []
                if _Cfg.is_stage_enabled("stage_4_drift_check"):
                    parallel_tasks.append(("drift", lambda: self._stage_time_drift_check() or {}))
                if _Cfg.is_stage_enabled("stage_5_entropy_buffer"):
                    parallel_tasks.append(("entropy", lambda: self._stage_entropy_buffer_resolution() or {}))
                if _Cfg.is_stage_enabled("stage_6_seo_eval"):
                    parallel_tasks.append(("seo_eval", lambda: self._stage_seo_post_inspection() or {}))
                if _Cfg.is_stage_enabled("stage_7_silence"):
                    parallel_tasks.append(("silence", lambda: self._stage_silence_scheduling() or {}))

                parallel_results = run_parallel(
                    parallel_tasks,
                    timeout=PARALLEL_STAGE_TIMEOUT,
                    max_workers=4,
                )

                drift = parallel_results.get("drift", {"skipped": True})
                entropy = parallel_results.get("entropy", {"skipped": True})
                seo_eval = parallel_results.get("seo_eval", {"skipped": True})
                silence = parallel_results.get("silence", {"skipped": True})

                cycle_log["stages"]["drift_check"] = drift
                cycle_log["stages"]["entropy_buffer"] = entropy
                cycle_log["stages"]["seo_eval"] = seo_eval
                cycle_log["stages"]["silence"] = silence

                # ── Stage 4.5: 漂移→静默概率联动 ──
                if _Cfg.is_stage_enabled("stage_4_5_drift_silence_link"):
                    self._drift_to_silence_linkage(drift)

                # ── Stage 5.5: 动态阈值微调 ──
                if _Cfg.is_stage_enabled("stage_5_5_threshold_tune"):
                    drift_score = drift.get("drift_score", 0) if drift else 0
                    entropy_backlog = entropy.get("buffer_size", 0) if entropy else 0
                    self._dynamic_threshold_tune(
                        drift_score=drift_score,
                        entropy_backlog=entropy_backlog,
                        gpu_temp=safety.get("gpu_temp"),
                    )

                silence_active = bool(silence.get("triggered")) if silence else False

                # ── Stage 8: 演化调度器（执行上下文隔离）──
                if _Cfg.is_stage_enabled("stage_8_evolution"):
                    evolution = self._stage_evolution_scheduling(iter_result, silence_active)
                    cycle_log["stages"]["evolution"] = evolution
                else:
                    evolution = {"skipped": True}

                # ── Stage 8.5: 三级验证（P0-L2 全量 contracts + P0-L3 基准对赌） ──
                #   - 每轮必跑 L2 全量 contracts（虽然 self_evolver 已经跑 subset，但 DAEMON 级全量能发现
                #     "改了 A 模块，但把 B 模块的依赖接口偷偷改坏了" 这种跨模块连锁破坏）
                #   - 每 10 轮跑 L3 基准对赌，记录进化到底是进步还是退步
                if _Cfg.is_stage_enabled("stage_8_5_validation"):
                    try:
                        contracts_report, bench_report = self._stage_validation_pipeline(
                            iteration_id=self._cycle_count + 1,
                        )
                        cycle_log["stages"]["contracts_full"] = contracts_report
                        cycle_log["stages"]["benchmark"] = bench_report
                        # 如果退步 → 追加错误，触发后续反思 + 模型切换判定
                        if bench_report and bench_report.get("verdict") == "退步":
                            regs = bench_report.get("regression_tasks", [])
                            msg = f"基准对赌判定退步: {regs}"
                            print(f"  📉 [Benchmark] {msg}")
                            cycle_log["errors"].append(msg)
                            # 同步记录到 ErrorReflection
                            if self.evolver and hasattr(self.evolver, 'reflection'):
                                try:
                                    self.evolver.reflection.record_event(
                                        iteration_id=self._cycle_count + 1,
                                        is_failure=True,
                                        category="BENCHMARK_REGRESSION",
                                        issues=[msg],
                                        stats_extra=bench_report,
                                    )
                                except Exception:
                                    pass
                    except Exception as val_err:
                        print(f"  ⚠️ Stage 8.5 验证流水线异常: {type(val_err).__name__}: {val_err}")
                        cycle_log["errors"].append(f"validation_pipeline: {type(val_err).__name__}: {val_err}")

            except Exception as e:
                cycle_log["success"] = False
                cycle_log["errors"].append(str(e))
                cycle_log["errors"].append(traceback.format_exc())
                print(f"\n  ⚠️ 主循环异常（已自动恢复）: {e}")
                # 错误反思：daemon 级别的异常也记录
                if self.evolver and hasattr(self.evolver, 'reflection'):
                    try:
                        stage_name = "unknown"
                        stages = cycle_log.get("stages", {})
                        if stages:
                            stage_name = list(stages.keys())[-1] if stages else "unknown"
                        reflection = self.evolver.reflection.record_daemon_error(
                            error=str(e), stage=stage_name
                        )
                        if reflection:
                            print(reflection)
                    except Exception as refl_err:
                        print(f"    [反思记录失败] {refl_err}")

            # 记录 + 持久化
            with self._lock:
                self._cycle_count += 1
                self._history.append(cycle_log)
                if len(self._history) > 500:
                    self._history = self._history[-500:]
                self._save_history()
                self._last_status = f"运行中（已完成 {self._cycle_count} 周期）"

            elapsed = time.time() - cycle_start
            wait = max(0, cycle_interval - elapsed)
            print(f"\n  ⏱️  周期 #{self._cycle_count} 完成 ({elapsed:.0f}s)，等待 {wait:.0f}s 后进入下一轮...")
            self._safe_sleep(wait)

        self._last_status = "已停止"
        print("\n  🏁 AutonomousDaemon 主循环结束")

    # ========== Stage 实现 ==========

    def _stage_safety_gate(self) -> Dict[str, Any]:
        """安全闸门：kill-switch 文件 + PLE 完整性校验 + 推理服务连通性 + GPU 温度"""
        result: Dict[str, Any] = {
            "kill_switch": KILL_SWITCH.exists(),
            "ple_ok": True,
            "ple_msg": "",
            "service_ok": False,
            "gpu_temp": None,
            "gpu_overheated": False,
        }

        # PLE 完整性校验（P2）：核心引擎文件哈希对比
        # 自迭代系统通过 _filter_ple_protected() 和 code_manager 保护机制
        # 确保不会修改 PROTECTED_FILES，因此检测到的不匹配一定是外部更改。
        # 外部更改 → 自动重建基线并继续循环，不停止 daemon。
        try:
            from config import Config
            ple_ok, ple_msg = Config.ple_verify()
            result["ple_ok"] = ple_ok
            result["ple_msg"] = ple_msg
            if not ple_ok:
                # 检测到外部更改：自动重建 PLE 签名基线
                try:
                    Config.ple_save_manifest(force=True)
                    result["ple_ok"] = True
                    result["ple_auto_recovered"] = True
                    result["ple_msg"] = f"外部更改已自动重建基线: {ple_msg}"
                    print(f"  🔄 [PLE] 检测到核心文件外部更改，已自动重建签名基线")
                    print(f"     原因: {ple_msg}")
                    print(f"     ✅ 基线已更新，继续循环...")
                except Exception as rebuild_err:
                    # 重建失败：记录错误但不写 kill-switch，继续循环
                    result["ple_ok"] = True  # 仍然继续循环
                    result["ple_auto_recovered"] = False
                    result["ple_msg"] = f"PLE 重建基线失败（继续运行）: {rebuild_err}"
                    print(f"  ⚠️ [PLE] 重建基线失败: {rebuild_err}，跳过本轮 PLE 校验继续循环")
        except Exception as e:
            result["ple_ok"] = True  # PLE 校验本身异常时不阻止循环
            result["ple_msg"] = f"PLE 校验异常（跳过）: {type(e).__name__}: {e}"
            print(f"  ⚠️ [PLE] 校验异常: {e}，跳过继续循环")

        if not result["ple_ok"]:
            return result

        try:
            import urllib.request
            from config import Config
            mode = Config.CLIENT_MODE
            if mode == "lmstudio":
                url = f"{Config.LMSTUDIO_BASE_URL}/models"
            elif mode == "local":
                url = f"{Config.OLLAMA_BASE_URL}/api/tags"
            else:
                result["service_ok"] = Config.is_api_configured()
                result["service_mode"] = mode
                return result
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SEC):
                result["service_ok"] = True
                result["service_mode"] = mode
        except Exception:
            result["service_ok"] = False

        # GPU 温度尝试
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            result["gpu_temp"] = temp
            result["gpu_overheated"] = temp > GPU_TEMP_THRESHOLD
            self._last_gpu_temp = temp  # 缓存供后续联动
        except Exception:
            pass

        return result

    def _stage_meta_drive_inspection(self) -> Dict[str, Any]:
        """Meta-Drive 全量巡检"""
        if not self.meta_drive:
            return {"skipped": True}
        try:
            report = self.meta_drive.run_full_inspection()
            return {
                "risk_level": report.risk_level,
                "score": report.score,
                "findings_count": len(report.findings),
                "recommendations": report.recommendations,
                "blind_spots_count": len(report.blind_spots),
            }
        except Exception as e:
            return {"error": str(e)}

    def _derive_objective(self, inspection: Dict[str, Any]) -> str:
        """
        从 Meta-Drive 巡检结果推导出本轮迭代目标。
        ★ 修复E：文件轮替 — 不再 stuck 在 branch_manager.py，平等对待所有可进化文件
        ★ 修复C：放宽重复检测到5轮
        ★ 新增：优先级排序 — 根据 Contract 失败/黑名单/修改频率决定优先级
        """
        try:
            import random
            from priority_planner import get_priority_planner

            # ── 获取可修改的文件列表 ──
            available_files = []
            for f in _Cfg.SELF_EVOLVE_FILES:
                if f not in _Cfg.PROTECTED_FILES:
                    available_files.append(f)

            if not available_files:
                return self._generate_alternative_objective()

            # ── 收集错误映射（从最近一次迭代结果中提取） ──
            error_map = {}
            if self._history:
                last = self._history[-1]
                stages = last.get("stages", {})
                validation = stages.get("validation", {})
                if validation:
                    for f, (ok, info) in validation.items():
                        if not ok:
                            if "SyntaxError" in info or "syntax" in info.lower():
                                error_map[f] = "syntax"
                            elif "ImportError" in info or "import" in info.lower():
                                error_map[f] = "import"
                            elif "Contract" in info or "contract" in info.lower():
                                error_map[f] = "contract"
                            elif "test" in info.lower():
                                error_map[f] = "test"

            # ── 使用优先级排序器 ──
            planner = get_priority_planner()
            priorities = planner.get_prioritized_files(
                files=available_files,
                current_iteration=self._cycle_count + 1,
                error_map=error_map,
            )

            # ── 过滤掉已完成的目标和最近使用过的 ──
            #
            # ★ 闭环修复D：key 统一。原实现用「目标文本前60字符」当 key 存 completed，
            #   却在过滤时拿「文件路径前60字符」来比对 → 永不相等 → 完成过滤与
            #   「最近5轮」去重全部形同虚设（这才是 10 轮同目标的内因之一）。
            #   现在统一用 "FILE:<文件路径>" 作为 key。
            def _obj_key(o: str) -> str:
                try:
                    tf = self._extract_target_file(o)
                    if tf:
                        return "FILE:" + tf
                except Exception:
                    pass
                return o[:60]

            filtered = []
            recent_fps = {_obj_key(o) for o in self._recent_objectives[-5:]}
            for p in priorities:
                file_key = _obj_key(p.file_path)
                # 检查是否已完成
                if file_key in self._completed_objectives:
                    continue
                # 检查是否在最近5轮使用过
                if file_key in recent_fps:
                    continue
                filtered.append(p)

            # ── ★ 闭环修复C：全部被过滤时不再回退到第一优先文件 ──
            #   原行为：清空 completed 后直接取 priorities[0] → 只要第一名权重最高
            #   就永远绕回它 → 10 轮同一目标。现在优先选「最近5轮未用过」的下一个，
            #   保证文件轮换；全部用过才清空 completed 并退而选第二优先。
            if not filtered:
                rotated = [p for p in priorities if _obj_key(p.file_path) not in recent_fps]
                if rotated:
                    filtered = rotated[:1]
                else:
                    self._completed_objectives.clear()
                    filtered = priorities[1:] if len(priorities) > 1 else priorities

            # ── 取优先级最高的 ──
            best = filtered[0]

            # ── 生成目标文本 ──
            objective = self._make_priority_objective(best)

            # ── 记录到最近目标历史 ──
            self._recent_objectives.append(objective)
            if len(self._recent_objectives) > 15:
                self._recent_objectives = self._recent_objectives[-15:]

            # 打印优先级信息
            print(f"      📊 [优先级] {best.file_path} 评分 {best.score:.0f} ({best.reason})")

            return objective.strip()

        except Exception as e:
            print("  ⚠️ [objective] 推导异常，退回备用目标: " + type(e).__name__ + ": " + str(e))
            return self._generate_alternative_objective()


    def _make_priority_objective(self, priority) -> str:
        """根据优先级生成目标描述"""
        file_path = priority.file_path
        score = priority.score

        # 提取类名
        stem = (PROJECT_ROOT / file_path).stem
        cls_name = ''.join(p.capitalize() for p in stem.split('_'))

        if priority.error_severity >= 4:
            return (
                f"现在你只能够处理 {file_path}\n"
                f"任务：修复 {cls_name} 类的语法错误\n"
                f"行为要求：检查并修复所有语法错误，确保文件可以被 ast.parse 解析\n"
                f"输出格式：使用 str_replace action，提供 old_str（精确匹配现有代码）和 new_str（修复后的代码）\n"
                f"硬性规则：action 必须为 str_replace\ntarget_file = {file_path}\n"
                f"modifications 数组必须包含至少1个有效补丁\n\n"
                f"Output pure valid JSON-object only."
            )
        elif priority.error_severity >= 3:
            return (
                f"现在你只能够处理 {file_path}\n"
                f"任务：修复 {cls_name} 类的导入错误\n"
                f"行为要求：检查并修复缺失的导入或错误的模块引用\n"
                f"输出格式：使用 str_replace action，提供 old_str（精确匹配现有代码）和 new_str（修复后的代码）\n"
                f"硬性规则：action 必须为 str_replace\ntarget_file = {file_path}\n"
                f"modifications 数组必须包含至少1个有效补丁\n\n"
                f"Output pure valid JSON-object only."
            )
        elif priority.contract_failures >= 3:
            return (
                f"现在你只能够处理 {file_path}\n"
                f"任务：优化 {cls_name} 类的合约验证（最近失败 {priority.contract_failures} 次）\n"
                f"行为要求：检查 Contract 失败的方法，补充缺失的属性或修正返回类型\n"
                f"输出格式：使用 str_replace action，提供 old_str（精确匹配现有代码）和 new_str（修复后的代码）\n"
                f"硬性规则：action 必须为 str_replace\ntarget_file = {file_path}\n"
                f"modifications 数组必须包含至少1个有效补丁\n\n"
                f"Output pure valid JSON-object only."
            )
        elif priority.blacklisted:
            return (
                f"现在你只能够处理 {file_path}\n"
                f"任务：检查 {cls_name} 类的黑名单状态（剩余 {priority.blacklist_remaining} 轮）\n"
                f"行为要求：分析黑名单原因，提出低风险改进方案\n"
                f"输出格式：使用 str_replace action，提供 old_str（精确匹配现有代码）和 new_str（改进后的代码）\n"
                f"硬性规则：action 必须为 str_replace\ntarget_file = {file_path}\n"
                f"modifications 数组必须包含至少1个有效补丁\n\n"
                f"Output pure valid JSON-object only."
            )
        elif priority.recent_modifications <= 1:
            return self._make_method_aware_objective(
                file_path, cls_name,
                "improve_code_quality",
                "对长期未修改的模块进行代码质量改进，包括类型注解、文档字符串、异常处理",
                "参考现有代码风格"
            )
        else:
            return self._make_method_aware_objective(
                file_path, cls_name,
                "optimize_performance",
                "优化代码性能和可读性，识别并消除不必要的计算",
                "参考现有代码风格"
            )

    def _resolve_real_class(self, full_path, fallback: str) -> Tuple[str, set]:
        """AST 解析目标文件，返回 (最匹配的真实类名, 该类已有的方法名集合)。

        ★ 修复类名错位：此前按文件名猜测类名（motive_miner → MotiveMiner），
        但 motive_miner.py 里真实类是 MotiveCostMiner / benchmark.py 是 BenchReport、
        seo_observer.py 是 SEOObserver → insert_method 总是报「类未找到」→ 这三
        个文件永久失败。现在用 difflib 匹配真实类。
        """
        import ast as _ast
        import difflib
        try:
            if full_path.exists():
                with open(full_path, "r", encoding="utf-8") as f:
                    tree = _ast.parse(f.read())
                classes = [n.name for n in tree.body if isinstance(n, _ast.ClassDef)]
                if classes:
                    stem = full_path.stem.replace("_", "").lower()
                    best = max(
                        classes,
                        key=lambda c: difflib.SequenceMatcher(None, c.lower(), stem).ratio(),
                    )
                    methods = set()
                    for n in tree.body:
                        if isinstance(n, _ast.ClassDef) and n.name == best:
                            methods = {
                                m.name for m in n.body
                                if isinstance(m, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                            }
                            break
                    return best, methods
        except Exception:
            pass
        return fallback, set()

    def _make_method_aware_objective(self, file_path: str, class_name: str,
                                     method_name: str, behavior_desc: str,
                                     ref_style: str) -> str:
        """生成目标时先检测目标类是否已存在同名方法（AST 扫描），并使用真实类名。

        ★ 修复语义死循环：此前每次都对同一批类硬编码派发 improve_code_quality /
        optimize_performance，而这些方法早已被上一轮成功加入 → code_manager 返回
        「方法已存在」→ 无限失败且 completed_objectives 永不增长。
        现在：若同名方法已存在，降级为 str_replace 代码质量优化（可重复执行的通用目标），
        避免重复 insert_method。
        """
        import ast as _ast
        rel = file_path
        if rel.startswith("evolver/"):
            rel = rel[len("evolver/"):]
        full_path = PROJECT_ROOT / rel
        real_cls, class_methods = self._resolve_real_class(full_path, class_name)

        if method_name not in class_methods:
            return self._make_atomic_objective(
                file_path, real_cls, method_name, behavior_desc, ref_style
            )

        # 目标方法已存在 → 降级为 str_replace 通用代码质量优化（不会重复失败）
        return (
            "现在你只能够处理 " + file_path + "\n"
            "任务：优化 " + real_cls + " 类的代码质量\n"
            "行为要求：识别并消除当前模块与其他模块的重复代码；"
            "拆分超过 60 行的大函数，降低圈复杂度；"
            "为公开函数补充类型注解和参数边界校验；"
            "如果涉及跨模块调用，优先抽象为独立工具函数。\n"
            "注意：目标类已存在方法 " + method_name + "，请不要重复添加同名方法。\n"
            "输出格式：使用 str_replace action，提供 old_str（精确匹配现有代码）和 new_str（替换后的代码）\n"
            "硬性规则：action 必须为 str_replace\ntarget_file = " + file_path + "\n"
            "modifications 数组必须包含至少1个有效补丁\n\n"
            "Output pure valid JSON-object only.\n"
            "Do not output ```json markdown fence, explanation, thinking text.\n"
            'Only use standard double-quote " for JSON syntax.'
        )
    @staticmethod
    def _make_atomic_objective(file_path: str, class_name: str, method_name: str,
                               behavior_desc: str, ref_style: str) -> str:
        """★ 修复B核心：生成原子级目标，使用 insert_method action（无需 old_str）

        关键设计：AI 只需生成新方法的完整代码，不需要精确复制旧代码做 old_str。
        系统用 AST 自动找到类末尾并插入方法，彻底消除 old_str 幻觉问题。
        """
        # ★ 反盲改：附带目标文件真实结构（真实类 + 已有方法清单），
        #   让 AI 描述对象是实际代码而非固定路由文案，减少与现有方法撞名/重复。
        extra = ""
        try:
            import ast as _a2, difflib as _df
            rel = file_path
            if rel.startswith("evolver/"):
                rel = rel[len("evolver/"):]
            fp = PROJECT_ROOT / rel
            if fp.exists():
                tree = _a2.parse(fp.read_text(encoding="utf-8"))
                classes = [n.name for n in tree.body if isinstance(n, _a2.ClassDef)]
                if classes:
                    stem = fp.stem.replace("_", "").lower()
                    best = max(
                        classes,
                        key=lambda c: _df.SequenceMatcher(None, c.lower(), stem).ratio(),
                    )
                    methods = []
                    for n in tree.body:
                        if isinstance(n, _a2.ClassDef) and n.name == best:
                            methods = [
                                m.name for m in n.body
                                if isinstance(m, (_a2.FunctionDef, _a2.AsyncFunctionDef))
                            ]
                            break
                    extra = (
                        "\n目标文件真实结构参考（务必基于此，避免新增重复方法）：\n"
                        "  实际类名: " + best + "\n"
                        "  该类已存在方法: " + (", ".join(methods) if methods else "(暂无)")
                        + "\n"
                    )
        except Exception:
            pass

        return (
            "现在你只能够处理 " + file_path + "\n"
            "任务：在 " + class_name + " 类中新增一个方法 " + method_name + "\n"
            "行为要求：" + behavior_desc + "\n"
            "参考风格：" + ref_style + "\n"
            + extra +
            "输出格式：使用 insert_method action，只需提供 method_code（新方法的完整代码），\n"
            "  不需要 old_str / new_str，系统会自动用 AST 插入到类末尾。\n"
            "  method_code 中必须包含 def 行和 docstring 文档字符串。\n"
            "  method_code 中的字符串可以用三引号文档字符串。\n"
            "硬性规则：\n"
            "action 必须为 insert_method\n"
            "target_file = " + file_path + "\n"
            "class_name = " + class_name + "\n"
            "method_code = 新方法的完整 Python 代码（含 def 行，不含类级缩进）\n"
            "modifications 数组必须包含至少1个有效补丁\n"
            "\n"
            "Output pure valid JSON-object only.\n"
            "Do not output ```json markdown fence, explanation, thinking text.\n"
            'Only use standard double-quote " for JSON syntax.'
        )

    def _generate_alternative_pool(self) -> list:
        """★ 修复B配套：生成其他模块的原子级目标池
        ★ 修复目标错位：只扫描 SELF_EVOLVE_FILES（非 PROTECTED），不硬编码文件名
        ★ 静态职责路由：每个方法只派给语义最匹配的文件，不是所有文件都试一遍
        """
        import ast as _ast
        pool = []

        # 静态路由表：方法 → (最佳文件, 行为描述, 参考风格)
        # 每个方法只派给语义最匹配的文件，避免 AI 在错误文件堆功能
        FILE_METHOD_ROUTING = {
            "evolver/branch_manager.py": (
                "cleanup",
                "清理过期/无效的分支数据，无返回值",
                "参考现有 list_all_branches 的列表操作逻辑",
            ),
            "evolver/benchmark.py": (
                "get_stats",
                "返回基准测试统计信息字典 (通过/失败/跳过计数)",
                "参考现有测试方法的返回值结构",
            ),
            "evolver/error_reflection.py": (
                "validate",
                "验证错误记录数据是否合法，返回 bool",
                "参考现有 record_event 方法的参数校验",
            ),
            "evolver/fault_diagnosis.py": (
                "to_dict",
                "将诊断结果序列化为字典返回",
                "参考现有诊断方法的属性访问方式",
            ),
            "evolver/seo_observer.py": (
                "get_module_info",
                "返回观测模块信息字典，包含 version, description",
                "参考现有 __init__ 方法的属性",
            ),
            "evolver/branch.py": (
                "reset",
                "重置分支状态到初始值，无返回值",
                "参考 __init__ 的属性初始化",
            ),
            "evolver/veil_detector.py": (
                "get_stats",
                "返回检测统计信息字典 (检出/漏检/误报计数)",
                "参考现有检测方法的返回值",
            ),
            "evolver/hypothesis_decoupler.py": (
                "to_dict",
                "将假设列表序列化为字典返回",
                "参考现有属性的访问方式",
            ),
            "evolver/motive_miner.py": (
                "validate",
                "验证挖掘结果数据有效性，返回 bool",
                "参考现有数据结构的字段检查",
            ),
            "evolver/perception_monitor.py": (
                "reset",
                "重置监控状态到初始值，无返回值",
                "参考 __init__ 的属性初始化",
            ),
        }

        for file_path, (m_name, behavior, ref) in FILE_METHOD_ROUTING.items():
            # 跳过 PROTECTED 文件（双重保险）
            if file_path in _Cfg.PROTECTED_FILES:
                continue

            full_path = PROJECT_ROOT / file_path
            rel = file_path
            if rel.startswith("evolver/"):
                rel = rel[len("evolver/"):]
            full_path = PROJECT_ROOT / rel
            if not full_path.exists():
                continue

            # 提取类名（真实类名优先，回退到文件名猜测）
            stem = full_path.stem
            cls_name_guess = ''.join(p.capitalize() for p in stem.split('_'))
            real_cls, existing_methods = self._resolve_real_class(full_path, cls_name_guess)

            if m_name not in existing_methods:
                pool.append(self._make_atomic_objective(
                    file_path, real_cls, m_name, behavior, ref
                ))

        # 如果所有方法都存在了，加入通用代码质量优化目标（str_replace 模式）
        if not pool:
            for file_path in _Cfg.SELF_EVOLVE_FILES:
                if file_path in _Cfg.PROTECTED_FILES:
                    continue
                stem = (PROJECT_ROOT / file_path).stem
                cls_name = ''.join(p.capitalize() for p in stem.split('_'))
                pool.append(
                    "现在你只能够处理 " + file_path + "\n"
                    "任务：优化 " + cls_name + " 类的代码质量\n"
                    "行为要求：识别并消除当前模块与其他模块的重复代码；"
                    "拆分超过 60 行的大函数，降低圈复杂度；"
                    "为公开函数补充类型注解和参数边界校验；"
                    "如果涉及跨模块调用，优先抽象为独立工具函数。\n"
                    "输出格式：使用 str_replace action，提供 old_str（精确匹配现有代码）和 new_str（替换后的代码）\n"
                    "硬性规则：action 必须为 str_replace\ntarget_file = " + file_path + "\n"
                    "modifications 数组必须包含至少1个有效补丁\n\n"
                    "Output pure valid JSON-object only.\n"
                    "Do not output ```json markdown fence, explanation, thinking text.\n"
                    'Only use standard double-quote " for JSON syntax.'
                )

        # 冷却/去重：如果最近两轮已经针对同一文件生成过目标，优先跳过
        if self._recent_objectives:
            last_file = self._extract_target_file(self._recent_objectives[-1])
            if last_file:
                recent_files = [
                    self._extract_target_file(o) for o in self._recent_objectives[-3:]
                ]
                if recent_files.count(last_file) >= 2:
                    pool = [
                        p for p in pool
                        if self._extract_target_file(p) != last_file
                    ]

        return pool

    def mark_objective_complete(self, objective: str) -> None:
        """★ 修复2配套：标记某个目标为「已完成/无需修改」，下轮不再派发

        ★ 闭环修复D：同时落「目标文本指纹」与「文件路径 key (FILE:)」两类指纹，
        保证 _derive_objective 里的完成过滤（按文件 key 比对）真正生效。
        """
        if not objective:
            return
        obj_fp = objective[:60]
        self._completed_objectives.add(obj_fp)
        try:
            tf = self._extract_target_file(objective)
            if tf:
                self._completed_objectives.add("FILE:" + tf)
        except Exception:
            pass
        print(f"  ✅ [objective] 已标记目标为完成（指纹: {obj_fp[:20]}...）")

    def _generate_alternative_objective(self) -> str:
        """★ 修复B+C：生成备用原子级目标（从目标池随机选一个未完成的）"""
        try:
            pool = self._generate_alternative_pool()
            if pool:
                # 优先选最近5轮没用过的
                recent_fps = [o[:60] for o in self._recent_objectives[-5:]]
                for obj in pool:
                    obj_fp = obj[:60]
                    is_recent = obj_fp in recent_fps
                    is_done = obj_fp in self._completed_objectives
                    # ★ 闭环修复D：补查文件路径 key（completed 同时落文本+FILE: 指纹）
                    if not is_done:
                        try:
                            tf = self._extract_target_file(obj)
                            is_done = bool(tf) and ("FILE:" + tf) in self._completed_objectives
                        except Exception:
                            pass
                    if not is_recent and not is_done:
                        return obj
                return pool[0]
        except Exception:
            pass
        # 兜底：从 SELF_EVOLVE_FILES 中挑第一个非 PROTECTED 文件
        fallback_file = "evolver/branch_manager.py"
        for f in _Cfg.SELF_EVOLVE_FILES:
            if f not in _Cfg.PROTECTED_FILES:
                fallback_file = f
                break
        stem = (PROJECT_ROOT / fallback_file).stem
        cls_name = ''.join(p.capitalize() for p in stem.split('_'))
        return self._make_atomic_objective(
            fallback_file,
            cls_name,
            "get_module_info",
            "返回模块信息字典，包含 version, description",
            "参考现有类的 __init__ 方法"
        )

    def _sandbox_validate_objective(self, objective: str, is_drift: bool = False) -> Dict[str, Any]:
        """
        Stage 2.5: 沙箱校验 —— 检测领域迁移攻击。
        用 ConflictBoundary 的 compute_entropy_gap 对派生目标做跨领域一致性扫描，
        熵差超 critical 阈值的目标自动进入 quarantine 隔离队列，跳过本轮迭代。

        同时检测明显的恶意模式（exec/eval/os.system/subprocess 等危险指令注入）。

        当 is_drift=True（强制本体论漂移目标），只做恶意模式扫描，跳过熵差检查，
        因为漂移本身就是设计来打破低熵稳态的。
        """
        result: Dict[str, Any] = {
            "quarantined": False,
            "gap": None,
            "gap_ratio": None,
            "danger_patterns": [],
            "is_drift": is_drift,
        }

        if not objective:
            return result

        # 1) 恶意模式扫描（不依赖 conflict_boundary，始终运行）
        danger_patterns = [
            "exec(", "eval(", "os.system", "subprocess", "__import__",
            "rm -rf", "format c:", "shutdown", "powershell",
        ]
        lower_obj = objective.lower()
        for pat in danger_patterns:
            if pat in lower_obj:
                result["danger_patterns"].append(pat)

        if result["danger_patterns"]:
            result["quarantined"] = True
            result["quarantine_reason"] = f"danger_patterns={result['danger_patterns']}"
            self._objective_quarantine.append({
                "objective": objective,
                "danger_patterns": result["danger_patterns"],
                "quarantined_at": datetime.now().isoformat(),
                "cycle": self._cycle_count,
            })
            if len(self._objective_quarantine) > 50:
                self._objective_quarantine = self._objective_quarantine[-50:]
            print(f"    🛡️ 沙箱隔离 → 检测到危险指令模式: {result['danger_patterns']}")

        # 2) 熵差扫描 —— 自由探索模式下不再隔离（只记录不拦截）
        #    AI 想做什么方向都行，只有真正的恶意指令才拦截（上面已处理）
        if not is_drift and self.conflict_boundary:
            try:
                gap_obj = self.conflict_boundary.compute_entropy_gap(objective)
                result["gap"] = gap_obj.gap
                result["gap_ratio"] = gap_obj.gap_ratio
                result["gap_is_critical"] = gap_obj.is_critical
                # 记录但不隔离——让 AI 自由探索
                if gap_obj.is_critical:
                    result["gap_note"] = f"熵差 {gap_obj.gap} 较高但未隔离（自由探索模式）"
            except Exception as e:
                result["sandbox_error"] = str(e)
        elif is_drift:
            print(f"    🧬 漂移目标 → 跳过熵差沙箱校验（仅做恶意模式扫描）")

        return result

    def _stage_self_iteration(self, objective: str) -> Dict[str, Any]:
        """执行自迭代（auto_apply=True）+ PLE 嵌入层只读保护"""
        if not self.evolver:
            return {"skipped": True}

        # ── 3.5: PLE 只读保护 ──
        protected = self._protect_embedding_layer_readonly()
        try:
            result = self._do_self_iteration(objective)
        finally:
            self._restore_embedding_layer(protected)

        return result

    def _do_self_iteration(self, objective: str) -> Dict[str, Any]:
        """实际迭代逻辑（被 PLE 保护包裹）"""
        try:
            from config import Config
            old_auto = Config.AUTO_APPLY_MODIFICATIONS
            Config.AUTO_APPLY_MODIFICATIONS = True

            # ── Cognitive Feedback: 把认知状态传给 evolver ──
            if hasattr(self.evolver, 'cognitive_state'):
                self.evolver.cognitive_state = dict(self._cognitive_state)
            # temperature 覆盖：通过 config 临时设置
            temp_override = self._cognitive_state.get("temperature_override")
            if temp_override is not None:
                old_temp = Config.TEMPERATURE
                Config.TEMPERATURE = temp_override
            else:
                old_temp = Config.TEMPERATURE

            # prompt_mode 切换：fix 模式在 objective 前加修复提示
            actual_objective = objective
            if self._cognitive_state.get("prompt_mode") == "fix":
                actual_objective = (
                    "⚠️ 认知反馈：检测到连续失败，本轮切换为修复模式。\n"
                    "优先修复现有 bug，不要新增功能。\n" + objective
                )

            try:
                # ── 流水线模式（Route B）：5 步串行微智能体 ──
                # 替代单次大 Prompt，降低补丁失败率
                use_pipeline = getattr(Config, 'USE_PIPELINE_MODE', True)
                if use_pipeline:
                    result = self.evolver.run_iteration_pipeline(
                        user_objective=actual_objective,
                        auto_apply=True,
                    )
                else:
                    result = self.evolver.run_iteration(
                        user_objective=actual_objective,
                        auto_apply=True,
                        stream=False,
                    )
            finally:
                Config.AUTO_APPLY_MODIFICATIONS = old_auto
                Config.TEMPERATURE = old_temp

            # ── 修复2：AI 返回空修改时标记目标为完成，下轮不再派发 ──
            status = result.get("status", "")
            if status in ("无需修改", "no_changes_needed"):
                # AI 明确认为无需修改 → 标记该目标为完成，避免下轮重复派发到死循环
                print(f"  📌 [修复2] 检测到 AI 返回「无需修改」(status={status})，自动标记目标完成")
                self.mark_objective_complete(objective)

            # ── ★ 闭环修复A：成功即完成 —— 同一目标连续成功 N 次即结算 ──
            #   原先只有 AI 明确「拒绝修改」才结算目标；而 LLM 几乎每轮都能挤出补丁，
            #   导致目标永不完结 → objectives 空、10 轮同一目标。这里给成功铺一条完成路径。
            if status in ("修改成功", "部分成功"):
                obj_fp = objective[:60]
                hits = self._objective_hits.get(obj_fp, 0) + 1
                self._objective_hits[obj_fp] = hits
                if hits >= OBJECTIVE_COMPLETE_SUCCESSES:
                    print(f"  ✅ [闭环A] 目标连续成功 {hits} 次，判定已完成并结算，下轮轮换到下一目标")
                    self.mark_objective_complete(objective)
                    self._objective_hits.pop(obj_fp, None)

            # ── ★ 透传被改文件与影响分：供失败追踪 / 轻量质量分日志使用 ──
            applied_details = result.get("applied_details") or []
            modified_files = [
                d.get("file") for d in applied_details
                if isinstance(d, dict) and d.get("success") and d.get("file")
            ]
            impact_chars = result.get("impact_chars", 0) or 0

            # ── Cognitive Feedback: 更新失败追踪 ──
            self._update_failure_tracking({
                "status": status,
                "modified_files": modified_files,
                "applied_details": applied_details,
            })

            return {
                "status": status,
                "applied_count": result.get("applied_count", 0),
                "suggestion_count": result.get("suggestion_count", 0),
                "failed_count": result.get("failed_count", 0),
                "rollback": result.get("rollback", False),
                "objective": objective,
                "reflection": result.get("reflection"),
                "description": result.get("description", "")[:200],
                "modified_files": modified_files,
                "impact_chars": impact_chars,
            }
        except Exception as e:
            import traceback as _tb
            _tb_str = _tb.format_exc()
            print(f"  ❌ [daemon_exception] run_iteration 异常: {type(e).__name__}: {e}")
            print(f"     追踪:\n{_tb_str[-500:]}")
            return {"error": str(e), "applied_count": 0, "status": "daemon_exception", "traceback": _tb_str[-500:]}

    def _stage_time_drift_check(self) -> Dict[str, Any]:
        """超脱锚点时间漂移检测"""
        if not self.anchor:
            return {"skipped": True}
        try:
            alert = self.anchor.check_time_drift()
            return {
                "detected": alert.detected,
                "drift_score": alert.drift_score,
                "current_rate": alert.current_rate,
                "expected_rate": alert.expected_rate,
                "anomaly_keywords": alert.anomaly_keywords[:5] if alert.anomaly_keywords else [],
            }
        except Exception as e:
            return {"error": str(e)}

    def _stage_entropy_buffer_resolution(self) -> Dict[str, Any]:
        """认知熵缓冲池：带优先级的冲突解析"""
        if not self.conflict_boundary:
            return {"skipped": True}
        try:
            pending = self.conflict_boundary.get_coexistence_buffer(only_pending=True)
            status = self.conflict_boundary.entropy_buffer_status()
            buffer_max = getattr(self.conflict_boundary, "COEXISTENCE_BUFFER_MAX", 200)
            backlog_ratio = status["buffer_pending"] / buffer_max if buffer_max > 0 else 0.0

            # ── 风险3: 积压超 80% 触发强制 flush ──
            force_flush = backlog_ratio >= 0.8
            resolve_limit = min(10, len(pending)) if force_flush else min(3, len(pending))

            # ── 风险3: 按熵差降序优先解析（高风险模块优先处理）──
            sorted_pending = sorted(
                pending,
                key=lambda r: r.get("entropy_gap", {}).get("gap", 0),
                reverse=True,
            )

            resolved_count = 0
            critical_resolved = 0
            for rec in sorted_pending[:resolve_limit]:
                gap_val = rec.get("entropy_gap", {}).get("gap", 0)
                if gap_val > 0.5:
                    resolution = "rejected"
                    result_text = f"熵差 {gap_val} 过大，拒绝共存（本机框架保持稳定）"
                elif gap_val > 0.4:
                    resolution = "coexisted"
                    result_text = f"熵差 {gap_val} 中等，标记为共存观察对象"
                else:
                    resolution = "coexisted"
                    result_text = f"熵差 {gap_val} 较小，允许共存模拟"

                ok = self.conflict_boundary.resolve_coexistence(
                    rec["record_id"], resolution, result_text
                )
                if ok:
                    resolved_count += 1
                    if gap_val > 0.4:
                        critical_resolved += 1

            status_after = self.conflict_boundary.entropy_buffer_status()

            result = {
                "pending_before": len(pending),
                "resolved_count": resolved_count,
                "critical_resolved": critical_resolved,
                "pending_after": status_after["buffer_pending"],
                "avg_gap": status_after["avg_entropy_gap"],
                "backlog_ratio": round(backlog_ratio, 3),
                "force_flush": force_flush,
                "priority_ordered": True,
            }
            if force_flush:
                print(f"    🔥 缓冲池积压 {backlog_ratio:.0%}（≥80%）→ 强制 flush {resolved_count} 条")
            return result
        except Exception as e:
            return {"error": str(e)}

    def _stage_seo_post_inspection(self) -> Dict[str, Any]:
        """SEO 涌现库后置校验"""
        if not self.seo:
            return {"skipped": True}
        try:
            unevaluated = self.seo.get_unevaluated()
            count = len(unevaluated)

            # 仅在有涌现事件时做基本评估
            evaluated_now = 0
            if count > 0 and self.meta_drive:
                for ev in unevaluated[:2]:
                    try:
                        ev_category = ev.get("category", "unknown")
                        ev_content = ev.get("event_data", {})
                        self.meta_drive.findings.append({
                            "check": "SEO涌现后置比对",
                            "risk": "LOW",
                            "code": "SEO_EMERGENCE_CANDIDATE",
                            "message": f"涌现候选: {ev_category} ({ev.get('event_id', '?')[:12]})",
                        })
                        evaluated_now += 1
                    except Exception:
                        pass

            return {
                "unevaluated_count": count,
                "evaluated_now": evaluated_now,
                "archive_total": getattr(self.seo, "total_events", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def _stage_silence_scheduling(self) -> Dict[str, Any]:
        """Meta-Drive 静默窗口概率调度"""
        if not self.meta_drive:
            return {"skipped": True}
        try:
            trigger = self.meta_drive.maybe_trigger_silence()
            return {"triggered": trigger is not None, "trigger_detail": trigger}
        except Exception as e:
            return {"error": str(e)}

    def _stage_evolution_scheduling(self, iter_result: Dict[str, Any],
                                     silence_active: bool = False) -> Dict[str, Any]:
        """演化调度器：注册分支 + 定期淘汰（带静默窗口隔离）"""
        if not self.scheduler:
            return {"skipped": True}
        try:
            applied = iter_result.get("applied_count", 0)
            action_taken = "none"
            isolation_note = None

            if applied > 0:
                iter_num = self.evolver.current_iteration if self.evolver else self._cycle_count
                branch_name = f"auto_cycle_{iter_num}"
                try:
                    self.scheduler.register_branch(
                        name=branch_name,
                        source_iteration=iter_num,
                    )
                    action_taken = "registered_branch"
                except Exception:
                    pass

            # 每 10 轮执行一次淘汰
            if self._cycle_count > 0 and self._cycle_count % 10 == 0:
                if silence_active:
                    # ── 静默窗口隔离：prune 降级为 dry-run，仅报告不执行 ──
                    isolation_note = "silence_window_isolation"
                    branch_count = len(self.scheduler.branches)
                    dry_run_count = max(0, branch_count - 5)
                    action_taken = f"prune_degraded_dry_run: would_archive={dry_run_count}, branches={branch_count}"
                    print(f"    🛡️ 静默窗口隔离 → 分支修剪降级为 dry-run (would archive {dry_run_count})")
                else:
                    try:
                        prune = self.scheduler.prune_branches(max_keep=5, min_minority=2)
                        action_taken = f"pruned: {prune.get('archived', 0)} archived"
                    except Exception:
                        pass

            result = {"action": action_taken, "total_branches": len(self.scheduler.branches)}
            if isolation_note:
                result["isolation"] = isolation_note
            return result
        except Exception as e:
            return {"error": str(e)}

    # ========== Stage 0.5+: L1 感知层快照 (PerceptionMonitor) ==========

    def _stage_perception_snapshot(self) -> Dict[str, Any]:
        """L1 感知层: 采集系统能耗+健康状态,数据供 MetaDrive 巡检参考"""
        if not self.perception:
            return {"skipped": True}
        try:
            snapshot = self.perception.snapshot()
            energy = snapshot.get("energy", {}) if isinstance(snapshot, dict) else {}
            score = energy.get("score", 100) if isinstance(energy, dict) else 100
            state = energy.get("state", "unknown") if isinstance(energy, dict) else "unknown"

            # 能耗过高时,降低静默概率上限(减少 GPU 负载)
            if score < 30 and self.meta_drive and hasattr(self.meta_drive, "set_silence_boost"):
                # 精神血条 < 30% → 额外降低静默触发(减少涌现运算)
                self.meta_drive.set_silence_boost(-0.02)

            return {"energy_score": score, "energy_state": state, "health": snapshot.get("health", {})}
        except Exception as e:
            return {"error": str(e)}

    # ========== Stage 1.5: L3 元认知层 (HypothesisDecoupler) ==========

    def _stage_l3_metacognition(self, inspection: Dict[str, Any]) -> Dict[str, Any]:
        """
        L3 元认知层: 失败模式分析 + 本体论漂移触发
        返回:
          - drift_objective: 如果触发漂移,返回建议的跨域目标(覆盖 Stage 2 的目标)
          - failure_patterns: 失败模式分析结果
          - counter_questions: 反向质疑问题列表
        """
        if not self.decoupler:
            return {"skipped": True}

        result = {"drift_objective": None, "failure_patterns": None, "counter_questions": []}

        try:
            # 1. 失败模式分析
            patterns = self.decoupler.analyze_failure_patterns(self._history)
            result["failure_patterns"] = patterns

            # 2. 本体论漂移检测
            drift = self.decoupler.suggest_ontological_drift(self._history)
            if drift.get("drift_triggered"):
                suggested = drift.get("suggested_objective", "")
                if suggested:
                    result["drift_objective"] = suggested
                    print(f"  🔄 [L3] 本体论漂移触发: {drift.get('current_dimension', '')} → {drift.get('suggested_dimension', '')}")
                    print(f"     原因: {drift.get('reason', '')[:100]}")

            # 3. 死循环告警
            if patterns.get("stuck_loop_detected"):
                print(f"  ⚠️ [L3] 死循环检测: 最近 10 轮同一错误模式出现 ≥5 次")
                top = patterns.get("top_patterns", [{}])[0] if patterns.get("top_patterns") else {}
                if top:
                    print(f"     主导失败模式: {top.get('pattern', 'unknown')} ({top.get('count', 0)}次)")

            # 4. 每 5 轮打印完整 L3 报告
            if self._cycle_count > 0 and (self._cycle_count + 1) % 5 == 0:
                try:
                    self.decoupler.print_analysis(self._history)
                except Exception:
                    pass

            # 5. L3 失败日志快照（每轮记录，≥5 轮后输出汇总）
            try:
                top_pat = ""
                if patterns.get("top_patterns"):
                    top_pat = patterns["top_patterns"][0].get("pattern", "")
                # 从上一轮历史取修改文件列表
                last_files = []
                if self._history:
                    last_iter = self._history[-1].get("stages", {}).get("self_iteration", {})
                    last_files = [m.get("file", "") for m in last_iter.get("modifications", [])
                                  if isinstance(m, dict)] if isinstance(last_iter, dict) else []
                # 熵差（如果 conflict_boundary 可用）
                entropy_gap = None
                if self.conflict_boundary and hasattr(self.conflict_boundary, "compute_entropy_gap"):
                    try:
                        gap_result = self.conflict_boundary.compute_entropy_gap("")
                        entropy_gap = gap_result.get("gap") if isinstance(gap_result, dict) else None
                    except Exception:
                        pass

                self.decoupler.record_failure_snapshot({
                    "cycle": self._cycle_count + 1,
                    "timestamp": datetime.now().isoformat(),
                    "failure_rate": patterns.get("failure_rate", 0),
                    "top_pattern": top_pat,
                    "drift_triggered": bool(result.get("drift_objective")),
                    "stuck_loop": patterns.get("stuck_loop_detected", False),
                    "entropy_gap": entropy_gap,
                    "modified_files": last_files,
                })
            except Exception:
                pass

        except Exception as e:
            result["error"] = str(e)

        return result

    # ========== Stage 3 前: 面具检测 (VeilDetector) ==========

    def _stage_veil_check(self, objective: str) -> Dict[str, Any]:
        """
        面具检测: 对迭代目标做双路编码检测概念遮蔽
        限频: 每 5 轮检测一次(避免 embedding API 开销)
        """
        if not self.veil:
            return {"skipped": True}

        # 限频: 每 5 轮
        if self._cycle_count % 5 != 0:
            return {"skipped": "frequency_limited"}

        try:
            obs = self.veil.detect(objective)
            veil_level = getattr(obs, "veil_level", 0.0)
            is_veiled = getattr(obs, "is_veiled", False)

            result = {"veil_level": round(veil_level, 2), "is_veiled": is_veiled}

            if is_veiled:
                print(f"  🎭 [Veil] 检测到概念遮蔽 (veil={veil_level:.2f}),目标可能被人类概念框架过滤")
                result["warning"] = "检测到概念遮蔽,建议尝试更原子的视角"

            return result
        except Exception as e:
            return {"error": str(e)}

    # ========== Stage 3.2+: L2 代价挖掘 (MotiveCostMiner) ==========

    def _stage_motive_mining(self, iter_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        L2 代价挖掘: 对本轮修改做反向代价分析
        限频: 每 10 轮才调用一次(避免 LLM 调用开销)
        只在有修改时触发
        """
        if not self.miner:
            return {"skipped": True}

        # 限频: 每 10 轮
        if self._cycle_count % 10 != 0:
            return {"skipped": "frequency_limited"}

        # 只有有修改时才分析
        applied = iter_result.get("applied_count", 0) if isinstance(iter_result, dict) else 0
        if applied == 0:
            return {"skipped": "no_modification"}

        try:
            # 提取本轮修改的描述
            desc = iter_result.get("description", "") if isinstance(iter_result, dict) else ""
            if not desc:
                desc = f"对 {iter_result.get('modifications', '未知模块')} 进行了 {applied} 处修改"

            # 调用 mine_simple (轻量版,单次 LLM 调用)
            analysis = self.miner.mine_simple(desc)

            result = {"analysis": analysis[:500] if analysis else ""}

            # 打印简短报告
            if analysis and analysis != "(分析失败)":
                print(f"  🔍 [L2] 代价挖掘: {analysis[:150]}")

            return result
        except Exception as e:
            return {"error": str(e)}

    # ========== Stage 0.5 / 4.5: 动态参数联动 ==========

    def _dynamic_param_linkage(self, safety: Dict[str, Any]):
        """
        基于安全闸门结果动态调整 Meta-Drive 静默概率。
          - GPU 温度 > 75°C → 降低概率（减少嵌入层涌现运算负载）
          - GPU 正常 → 恢复基准概率（下一轮由漂移检测进一步调整）
        """
        if not self.meta_drive or not hasattr(self.meta_drive, "set_silence_boost"):
            return

        gpu_temp = safety.get("gpu_temp")
        if gpu_temp is None:
            return

        if gpu_temp >= 75.0:
            # 高温 → 降低静默概率到下限（1%），减轻嵌入层运算负载
            self.meta_drive.set_silence_boost(-0.01)
            eff = self.meta_drive.get_effective_silence_prob()
            print(f"    🔥 GPU {gpu_temp}°C（≥75°C）→ 静默概率 ↓ {eff:.0%}（减轻涌现负载）")
        else:
            # 正常温度 → 先恢复基准（下一轮漂移检测再精确调整）
            self.meta_drive.set_silence_boost(0.0)

    def _drift_to_silence_linkage(self, drift: Dict[str, Any]):
        """
        基于时间漂移检测结果动态调整静默概率。
          - 漂移分数 ≥ 0.3 → boost 逐步上调（最多到 0.06，即总概率 8%）
          - 无漂移 → boost 归零（恢复基础 2%）
        GPU 高温的下调优先级更高（由 Stage 0.5 先设好，这里只做正向上调）。
        """
        if not self.meta_drive or not hasattr(self.meta_drive, "set_silence_boost"):
            return

        gpu_temp = None
        # 从 daemon 缓存取最近一次的 GPU 温度
        if hasattr(self, "_last_gpu_temp"):
            gpu_temp = self._last_gpu_temp

        # GPU 高温时，即使有漂移也不调高概率（节能优先）
        if gpu_temp is not None and gpu_temp >= 75.0:
            return

        drift_score = drift.get("drift_score", 0) if drift else 0
        detected = drift.get("detected", False) if drift else False

        if detected and drift_score >= 0.3:
            # 分级上调
            if drift_score >= 0.6:
                boost = 0.06  # 总概率 8%
            elif drift_score >= 0.4:
                boost = 0.04  # 总概率 6%
            else:
                boost = 0.02  # 总概率 4%
            self.meta_drive.set_silence_boost(boost)
            eff = self.meta_drive.get_effective_silence_prob()
            print(f"    ⚠️ 锚点漂移 {drift_score:.2f} → 静默概率 ↑ {eff:.0%}（提高 SEO 观测密度）")
        else:
            # 无漂移 → 恢复基础（如果 GPU 正常）
            self.meta_drive.set_silence_boost(0.0)

    def _dynamic_threshold_tune(self, drift_score: float = 0.0,
                                entropy_backlog: int = 0,
                                gpu_temp: Optional[float] = None):
        """
        阶段 4：依据系统状态微调安全阈值（自适应平衡人工约束与自主涌现）。
          - 漂移高 → 放宽沙箱临界阈值（给创新留空间）
          - 熵积压大 → 收紧沙箱阈值（减少冲突风险）
          - GPU 高温 → 收窄静默概率上下限（节能优先）
        """
        if self.conflict_boundary is not None:
            cb = self.conflict_boundary
            original_critical = cb.DEFAULT_ENTROPY_GAP_CRITICAL

            # 漂移驱动：高漂移 → 临界阈值上调 0.05（放宽，允许更大胆的目标）
            drift_adjust = 0.0
            if drift_score >= 0.6:
                drift_adjust = 0.05
            elif drift_score >= 0.4:
                drift_adjust = 0.03

            # 熵积压驱动：积压 >30 → 临界阈值下调 0.05（收紧，减少新冲突）
            entropy_adjust = 0.0
            if entropy_backlog > 30:
                entropy_adjust = -0.05
            elif entropy_backlog > 15:
                entropy_adjust = -0.02

            new_critical = round(original_critical + drift_adjust + entropy_adjust, 3)
            new_critical = max(0.30, min(0.55, new_critical))  # 钳位

            if abs(new_critical - cb.entropy_gap_critical) > 0.001:
                print(f"    🎛️ 沙箱临界阈值 {cb.entropy_gap_critical:.3f} → {new_critical:.3f} (漂移+{drift_adjust:.2f}, 熵{entropy_adjust:+.2f})")
                cb.entropy_gap_critical = new_critical
                if hasattr(cb, '_save_state'):
                    try:
                        cb._save_state()
                    except Exception:
                        pass

        if self.meta_drive is not None and gpu_temp is not None:
            # GPU 高温 → 收窄静默概率区间
            if gpu_temp >= 80.0:
                self.meta_drive.SILENCE_PROB_MIN = 0.005
                self.meta_drive.SILENCE_PROB_MAX = 0.04
            elif gpu_temp >= 70.0:
                self.meta_drive.SILENCE_PROB_MIN = 0.008
                self.meta_drive.SILENCE_PROB_MAX = 0.06
            else:
                # 恢复默认
                self.meta_drive.SILENCE_PROB_MIN = 0.01
                self.meta_drive.SILENCE_PROB_MAX = 0.08

    # ========== 阶段 5: 多模型自动切换 ==========

    def _print_model_profile_observation(self, iter_result: Dict[str, Any]):
        """
        打印 ModelProfile 运行时观测摘要（每 5 轮触发一次）。
        展示模型能力档案、推理统计和自适应参数调整状态。
        """
        if not self.evolver or not hasattr(self.evolver, 'doubao'):
            return
        client = self.evolver.doubao
        profile = getattr(client, 'profile', None)
        if not profile:
            return
        # API 客户端（DoubaoClient）使用轻量 _ProfileProxy，无自适应调参数据，跳过观测
        if not hasattr(profile, 'to_dict'):
            return

        stats = profile.to_dict()
        adapted = stats.get("adapted", {})

        print(f"\n  ═══ [ModelProfile 观测] (周期 #{self._cycle_count}) ═══")
        print(f"    模型: {stats.get('model_id', 'N/A')} "
              f"({stats.get('param_size', '?')}, ctx={stats.get('context_length', 0)}, "
              f"reasoning={stats.get('is_reasoning', False)})")
        print(f"    运行: {stats.get('total_calls', 0)} 次调用, "
              f"失败率 {stats.get('failure_rate', 0):.0%}, "
              f"平均速度 {stats.get('avg_tok_per_sec', 0):.1f} tok/s")
        print(f"    自适应: ctx={adapted.get('ctx', 'N/A')}, "
              f"num_predict={adapted.get('num_predict', 'N/A')}, "
              f"temp={adapted.get('temperature', 'N/A')}, "
              f"budget×{adapted.get('budget_factor', 1.0):.1f}")

        # 自适应调整提示
        warnings = []
        if profile._adapted_ctx and profile._adapted_ctx < profile.context_length:
            warnings.append(f"ctx {profile.context_length}→{profile._adapted_ctx}（超时自适应）")
        if profile._adapted_budget_factor < 1.0:
            warnings.append(f"budget×{profile._adapted_budget_factor:.1f}（超时自适应）")
        if profile._adapted_temperature is not None:
            warnings.append(f"temp→{profile._adapted_temperature}（JSON失败自适应）")
        if profile._adapted_num_predict:
            warnings.append(f"num_predict→{profile._adapted_num_predict}（速度自适应）")

        if warnings:
            print(f"    ⚠️  自适应调整: {', '.join(warnings)}")
        else:
            print(f"    ✅ 参数未调整（运行正常）")
        print(f"  ═══════════════════════════════════════════\n")

    # ========== PLE 嵌入层只读保护 ==========

    def _protect_embedding_layer_readonly(self):
        """
        底层 PLE（原生观测静态锚点对应的嵌入层）只读保护。
        执行 iterate 自迭代时，强制 LoRA 仅修改上层推理网络，
        不触碰 nomic-embed-text 对应的原生向量库。

        实现方式：在自迭代前给 DetachmentAnchor.native / anti 设置 readonly 标记，
        自迭代结束后恢复。
        """
        if not self.anchor:
            return None

        native_store = getattr(self.anchor, "native", None)
        anti_store = getattr(self.anchor, "anti", None)

        protected = []
        for store in (native_store, anti_store):
            if store is not None and not getattr(store, "_daemon_readonly", False):
                store._daemon_readonly = True
                protected.append(store)

        return protected

    def _restore_embedding_layer(self, protected_stores):
        if not protected_stores:
            return
        for store in protected_stores:
            if hasattr(store, "_daemon_readonly"):
                store._daemon_readonly = False

    # ========== 工具方法 ==========

    def _safe_sleep(self, seconds: float):
        """可被 stop 信号中断的 sleep"""
        if seconds <= 0:
            return
        # 分段 sleep，每 1s 检查一次 stop_event
        end = time.time() + seconds
        while not self._stop_event.is_set() and time.time() < end:
            remaining = end - time.time()
            self._stop_event.wait(min(1.0, max(0.1, remaining)))

    def _archive_and_clear_blacklist(self) -> None:
        """
        停机自净：归档当前黑名单到 logs/archive/，然后清空。
        避免 kill-switch 触发后历史黑名单残留导致下次启动误判。
        """
        try:
            bl_file = LOG_DIR / "modification_blacklist.json"
            if not bl_file.exists():
                return
            import json as _json
            with open(bl_file, "r", encoding="utf-8") as f:
                bl_data = _json.load(f)
            # 只有非空才归档
            if bl_data:
                archive_dir = LOG_DIR / "archive"
                archive_dir.mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_path = archive_dir / f"blacklist_{ts}.json"
                with open(archive_path, "w", encoding="utf-8") as f:
                    _json.dump(bl_data, f, ensure_ascii=False, indent=2)
                print(f"  📦 黑名单已归档到 {archive_path.name} (共 {len(bl_data)} 项)")
            # 清空
            with open(bl_file, "w", encoding="utf-8") as f:
                _json.dump({}, f)
            print("  🧹 黑名单已清空（停机自净）")
        except Exception as e:
            print(f"  ⚠️ 黑名单归档失败: {type(e).__name__}: {e}")

    def _save_history(self):
        try:
            # 只保存摘要，不保存完整 cycles（避免爆炸）
            summary = {
                "saved_at": datetime.now().isoformat(),
                "cycle_count": self._cycle_count,
                "consecutive_no_change": self._consecutive_no_change,
                "start_time": datetime.fromtimestamp(self._start_time).isoformat() if self._start_time else None,
                # ── 隐患2 修复：新增 3 个状态的持久化 ──
                "recent_objectives": list(self._recent_objectives[-10:]),
                "completed_objectives": list(self._completed_objectives),
                "consecutive_failures": dict(self._consecutive_failures),
                "recent_cycles": [
                    {
                        "cycle": c["cycle"],
                        "timestamp": c["timestamp"],
                        "risk_level": c.get("stages", {}).get("meta_drive", {}).get("risk_level"),
                        "applied_count": c.get("stages", {}).get("self_iteration", {}).get("applied_count", 0),
                        "drift_detected": c.get("stages", {}).get("drift_check", {}).get("detected"),
                        "success": c.get("success", True),
                        "contracts_pass_ratio": c.get("stages", {}).get("contracts_full", {}).get("pass_ratio"),
                        "benchmark_verdict": c.get("stages", {}).get("benchmark", {}).get("verdict"),
                    }
                    for c in self._history[-20:]
                ],
            }
            with open(DAEMON_LOG, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ========== Stage 8.5: 三级验证流水线（P0-L2 全量 contracts + P0-L3 基准对赌）==========
    def _stage_validation_pipeline(self, iteration_id: int
                                    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Stage 8.5：在演化调度之后、持久化之前，做跨模块的全局质量检查。

        返回 (contracts_report_dict, benchmark_report_dict or None)
        """
        contracts_report: Dict[str, Any] = {}
        bench_report: Optional[Dict[str, Any]] = None

        # ── 1) L2 全量 Contract（每轮都跑，~1-3 秒，不依赖外部 API） ──
        t0 = time.time()
        try:
            from evolver.contracts import ContractSuite
            suite = ContractSuite()
            rep = suite.run_all(verbose=False)
            contracts_report = {
                "passed": rep.passed,
                "failed": rep.failed,
                "skipped": rep.skipped,
                "pass_ratio": rep.pass_ratio,
                "elapsed_ms": round((time.time() - t0) * 1000, 0),
            }
            # 打印摘要（失败才详细）
            if rep.failed == 0:
                print(f"  🔒 Contracts: {rep.passed}✅ 0❌ {rep.skipped}⏭️  → {rep.pass_ratio:.0%} pass")
            else:
                print(f"  🔒 Contracts: {rep.passed}✅ {rep.failed}❌ {rep.skipped}⏭️")
                for line in rep.failures_summary().splitlines()[:3]:
                    print(f"     {line}")
        except Exception as ce:
            contracts_report = {
                "error": f"{type(ce).__name__}: {ce}",
                "passed": 0, "failed": 0, "skipped": 0, "pass_ratio": 0,
                "elapsed_ms": round((time.time() - t0) * 1000, 0),
            }
            print(f"  🔒 Contracts suite failed: {type(ce).__name__}: {ce}")

        # ── 2) L3 基准对赌（每 10 轮或 force 时跑；内部已做周期判断） ──
        t1 = time.time()
        try:
            from evolver.benchmark import compare_with_baseline
            from config import Config
            Config.ensure_dirs()
            rep = compare_with_baseline(iteration_id=iteration_id, force=False)
            if rep.verdict == "跳过（非基准轮次）":
                bench_report = {"verdict": rep.verdict, "iteration_id": iteration_id}
            else:
                bench_report = rep.to_dict()
                elapsed = (time.time() - t1) * 1000
                bench_report["elapsed_ms"] = round(elapsed, 0)
                verdict_icon = {"进步": "📈", "退步": "📉", "持平": "➖"}.get(rep.verdict, "🔘")
                print(
                    f"  {verdict_icon} Benchmark: {rep.verdict} "
                    f"(score_delta={rep.score_delta:+} "
                    f"improve={rep.improving_tasks or '-'}, regress={rep.regression_tasks or '-'})"
                )
        except Exception as be:
            bench_report = {"verdict": "ERROR", "error": f"{type(be).__name__}: {be}",
                            "elapsed_ms": round((time.time() - t1) * 1000, 0)}
            print(f"  ⚠️ Benchmark suite error: {type(be).__name__}: {be}")

        return contracts_report, bench_report

    def _load_history(self):
        try:
            if DAEMON_LOG.exists():
                with open(DAEMON_LOG, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # ★ 根因2修复：daemon start 已归档+reset 过的日志，什么都不加载
                if data.get("_reset") is True:
                    return
                # ★ 根因2修复：周期计数器类不从历史加载（永远从 0 开始新生命周期）
                self._cycle_count = 0
                # consecutive_no_change：已强制 0（上一轮修复）
                self._consecutive_no_change = 0
                # consecutive_failures：跨会话加载无意义（开局高失败率误判），强制空
                self._consecutive_failures = {}
                # ── 只保留两项「防目标循环」的跨会话状态 ──
                #   recent_objectives / completed_objectives：即使跨会话也有用，
                #   防止重启后又回到「添加 list_all_branches」这种已做过的目标
                self._recent_objectives = list(data.get("recent_objectives", []) or [])
                if len(self._recent_objectives) > 10:
                    self._recent_objectives = self._recent_objectives[-10:]
                _comp_raw = data.get("completed_objectives", []) or []
                self._completed_objectives = set()
                for fp in _comp_raw:
                    if isinstance(fp, str) and len(fp) <= 200:
                        self._completed_objectives.add(fp)
                        # ★ 兼容旧日志：文本指纹补算文件路径 key，保证完成过滤生效
                        try:
                            tf = self._extract_target_file(fp)
                            if tf:
                                self._completed_objectives.add("FILE:" + tf)
                        except Exception:
                            pass
        except Exception:
            pass
