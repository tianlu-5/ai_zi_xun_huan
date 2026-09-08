"""
AI 自迭代系统 - 全自动主循环入口
唯一用途：启动 AutonomousDaemon，无人值守运行
"""
import os, sys
import sys
import json
import threading
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_EVOLVER_DIR = _PROJECT_ROOT / "evolver"
if str(_EVOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_EVOLVER_DIR))

from config import Config

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║     AI 自迭代系统 · 全自动主循环                          ║
║     输入 daemon start 启动无人值守演化                    ║
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
═══ 全自动主循环 ═══
  daemon start               启动全自动主循环（后台线程，无人值守）
  daemon stop                请求停止主循环
  daemon status              查看主循环状态
  daemon kill                创建 logs/STOP_DAEMON 文件（外部 kill-switch）
  daemon clear               清除 kill-switch 文件

═══ 配置管理（读写）═══
  config                     查看当前配置快照（默认）
  config entropy set <crit> <warn>   动态调整熵差阈值（例：0.42 0.25）
  config entropy reset              恢复熵差默认阈值
  config ple status                 查看 PLE 核心引擎签名状态
  config ple reset                  重建 PLE 签名基线（核心引擎合法改动后执行）

═══ 运行时查询（只读）═══
  status                     系统状态 + 迭代统计
  cb                         冲突边界状态 + 熵缓冲池
  seo                        SEO 涌现检测状态
  drift-check                时间漂移检测

═══ 验证工具箱（隐患1保障：绕开PowerShell执行策略，直接python.exe运行）═══
  verify                     一键全量校验：py_compile语法 + 合约回归 + 方法AST检查
  verify syntax              仅语法检查（py_compile核心引擎/可修改模块文件）
  verify contracts           仅合约回归（ContractSuite.run_all 全量）
  verify ast                 仅AST逻辑：BranchManager/核心模块方法真实存在性

═══ 统一回滚（P3 语义：分支=迭代级，备份=文件级）═══
  rollback                   列出最近 10 次迭代级快照，以及每快照包含的文件
  rollback last              回滚到上一次迭代（恢复该 iter 标签下的所有文件备份）
  rollback 455               回滚到 iter=455（恢复所有带 _iter455_ 标签的备份）
  rollback evolver/meta_drive.py   文件级回滚：恢复该文件的最近一次备份
  rollback list <file>             列出某文件的所有备份（最新在前）
═══ 可视化窗口═══
  viz                        生成迭代可视化图表（趋势图、雷达图、成功率等）
═══ 经验管理 ═══
  export <文件名>            导出成功经验到文件（如: export my_exp.json）
  import <文件名>            导入成功经验（如: import my_exp.json）
  export list               列出可导出的经验统计
═══ 配置管理 ═══
  config hot-reload          手动触发配置热重载（自动检测已开启）
  config hot-reload on/off   开启/关闭自动热重载
═══ Web Dashboard ═══
  dashboard start [端口]    启动 Web Dashboard（默认 8080）
  dashboard stop            停止 Web Dashboard
  dashboard status          查看 Dashboard 状态
"""


# ════════════════════════════════════════════════════════════════
# 隐患修复工具函数（模块缓存清除 + daemon_log 归档重置）
# ════════════════════════════════════════════════════════════════

def _clear_evolver_module_cache() -> int:
    """★ 根因1修复（修正版）：强制清除 sys.modules 中所有 evolver 相关缓存条目。

    关键发现：evolver/ 目录下的模块通过 sys.path.insert(0, "evolver") 被导入为
    顶层模块（如 autonomous_daemon），而不是 evolver.autonomous_daemon。
    所以只清 evolver.* 前缀是不够的，必须同时清除裸名模块。

    返回：被清除的模块数
    """
    # evolver/ 目录下所有需要清除的模块名（裸名，不带 evolver. 前缀）
    _EVOLVER_BARE_MODULES = {
        "autonomous_daemon", "self_evolver", "ollama_client", "doubao_client",
        "modification_parser", "fault_diagnosis", "code_manager", "meta_drive",
        "capability_registry", "contracts", "model_profile", "error_reflection",
        "branch_manager", "conflict_boundary", "detachment_anchor", "parallel_guard",
        "motive_miner", "veil_detector", "modification_blacklist", "config",
        "ollama_client", "doubao_client", "seo_evaluator", "silence_scheduler",
        "perception", "hypothesis_decoupler", "cognitive_feedback",
    }
    removed = 0
    for mod_name in list(sys.modules.keys()):
        # 清除三种形式：evolver.* / evolver / 裸名（如 autonomous_daemon）
        should_clear = (
            mod_name == "evolver"
            or mod_name.startswith("evolver.")
            or mod_name in _EVOLVER_BARE_MODULES
        )
        if should_clear:
            try:
                del sys.modules[mod_name]
                removed += 1
            except KeyError:
                pass
    if removed > 0:
        print(f"  🧹 [模块缓存] 已清除 {removed} 个缓存条目（含裸名模块，强制加载磁盘最新代码）")
    else:
        print(f"  ✅ [模块缓存] 无缓存，将直接加载磁盘最新代码")
    return removed


def _archive_and_reset_daemon_log() -> Optional[Path]:
    """★ 根因2修复：每次 daemon start 前，归档并清空 logs/daemon_log.json。

    cycle_count / consecutive_no_change 等计数器只追踪「当前生命周期」，
    如果不清空，_load_history 会读取历史值，导致：
    - 第 N 次启动显示周期 #12 而非 #1
    - 认知反馈 consecutive_failures 为旧值，误判失败率
    - 历史 recent_objectives 干扰新会话的去重逻辑

    返回：归档后的文件路径（如果有内容被归档），否则 None
    """
    from datetime import datetime as _dt
    try:
        Config.ensure_dirs()
    except Exception:
        pass
    log_path = Path(Config.LOG_DIR) / "daemon_log.json"
    archive_dir = Path(Config.LOG_DIR) / "archive"
    if not log_path.exists():
        return None
    # 读旧内容判断是否为空
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            raw = f.read()
        import json as _json
        try:
            data = _json.loads(raw) if raw.strip() else {}
        except Exception:
            data = {"raw": raw}
        if not data:
            # 空 JSON，直接删除不归档
            try:
                log_path.unlink()
            except Exception:
                pass
            return None
    except Exception:
        data = {}
    # 归档
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"daemon_log_{ts}.json"
        # 复制内容（保留完整原始快照，含 cycle/recent/completed 等）
        try:
            import shutil
            shutil.copy2(str(log_path), str(archive_path))
        except Exception:
            with open(archive_path, "w", encoding="utf-8") as f:
                f.write(raw or "")
        # 清空 daemon_log.json（写最小空 JSON 对象，防止解析报错）
        with open(log_path, "w", encoding="utf-8") as f:
            _json.dump({"_reset": True, "reset_at": _dt.now().isoformat()}, f, ensure_ascii=False, indent=2)
        print(f"  📦 [daemon_log] 已归档到 {archive_path.name} 并清空当前日志")
        return archive_path
    except Exception as e:
        print(f"  ⚠️ [daemon_log] 归档失败（但继续启动）: {type(e).__name__}: {e}")
        # 至少尝试清空
        try:
            import json as _j
            with open(log_path, "w", encoding="utf-8") as f:
                _j.dump({"_reset": True}, f)
        except Exception:
            pass
        return None


def _make_daemon(evolver):
    """构造全套认知架构模块并组装 AutonomousDaemon"""
    md = cb = anc = seo = sch = miner = percept = decoupler = veil = None
    try:
        from meta_drive import MetaDrive; md = MetaDrive()
    except Exception as e: print(f"  ⚠️ MetaDrive: {e}")
    try:
        from conflict_boundary import ConflictBoundary; cb = ConflictBoundary()
    except Exception as e: print(f"  ⚠️ ConflictBoundary: {e}")
    try:
        from detachment_anchor import DetachmentAnchor; anc = DetachmentAnchor()
    except Exception as e: print(f"  ⚠️ DetachmentAnchor: {e}")
    try:
        from seo_observer import SEOObserver; seo = SEOObserver()
    except Exception as e: print(f"  ⚠️ SEOObserver: {e}")
    try:
        from evolution_scheduler import EvolutionScheduler; sch = EvolutionScheduler()
    except Exception as e: print(f"  ⚠️ EvolutionScheduler: {e}")
    try:
        from motive_miner import MotiveCostMiner; miner = MotiveCostMiner(ollama_client=getattr(evolver, 'doubao', None))
    except Exception as e: print(f"  ⚠️ MotiveCostMiner: {e}")
    try:
        from perception_monitor import PerceptionMonitor; percept = PerceptionMonitor()
    except Exception as e: print(f"  ⚠️ PerceptionMonitor: {e}")
    try:
        from hypothesis_decoupler import HypothesisDecoupler; decoupler = HypothesisDecoupler()
    except Exception as e: print(f"  ⚠️ HypothesisDecoupler: {e}")
    try:
        from veil_detector import VeilDetector; veil = VeilDetector()
    except Exception as e: print(f"  ⚠️ VeilDetector: {e}")
    # =====================【新增这一段】=====================
    # 将L3元认知实例注入reflection，打通中间失败事件转发链路
    if evolver is not None and decoupler is not None and hasattr(evolver, "reflection"):
        try:
            evolver.reflection.set_decoupler(decoupler)
            print("  ✅ 已绑定：reflection ↔ hypothesis_decoupler（中间失败事件转发已开启）")
        except Exception:
            pass
    from autonomous_daemon import AutonomousDaemon
    return AutonomousDaemon(
        evolver=evolver,
        meta_drive=md,
        anchor=anc,
        conflict_boundary=cb,
        seo=seo,
        scheduler=sch,
        miner=miner,
        perception=percept,
        decoupler=decoupler,
        veil=veil,
    )


class InteractiveCLI:

    def __init__(self):
        Config.ensure_dirs()
        Config.load_from_file()
        self.daemon = None
        self.daemon_thread = None
        self.running = True

    def _get_evolver(self):
        from self_evolver import SelfEvolver
        return SelfEvolver()

    def run(self):
        print(BANNER)
        self._preflight()
        print(HELP_TEXT)
        while self.running:
            try:
                raw = input("🔮 AI > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！"); break
            if not raw: continue
            try:
                self._handle(raw)
            except Exception as e:
                print(f"[错误] {type(e).__name__}: {e}")

    def _preflight(self):
        ok = Config.is_client_ready()
        mode_label, model_name = self._mode_display()
        print(f"🔍 引擎: {mode_label} ({model_name}) | 连接: {'✅' if ok else '⚠️'}")

    @staticmethod
    def _mode_display():
        if Config.is_jiyuan_mode():
            return ("基元律动 TokenRhythm", Config.JIYUAN_MODEL)
        elif Config.CLIENT_MODE == "api":
            return ("火山方舟 API", Config.MODEL_ID)
        elif Config.CLIENT_MODE == "lmstudio":
            return ("LM Studio", Config.LMSTUDIO_MODEL)
        else:
            return ("Ollama", Config.OLLAMA_MODEL)

    def _handle(self, raw: str):
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("q", "quit", "exit"):
            self._stop_daemon_if_running(); self.running = False; return
        if cmd in ("cls", "clear"):
            os.system("cls" if os.name == "nt" else "clear"); return
        if cmd in ("h", "help", "?"):
            print(HELP_TEXT); return

        if cmd in ("daemon", "autonomous"):
            self._daemon(arg)
        elif cmd == "status":
            self._status()
        elif cmd == "config":
            self._config(arg)
        elif cmd in ("rollback", "restore", "revert"):
            self._rollback(arg)
        elif cmd in ("cb", "boundary"):
            from conflict_boundary import ConflictBoundary; ConflictBoundary().print_status()
        elif cmd in ("seo", "seo-status"):
            from seo_observer import SEOObserver; SEOObserver().print_status()
        elif cmd == "drift-check":
            from detachment_anchor import DetachmentAnchor
            a = DetachmentAnchor()
            alert = a.check_time_drift()
            print(json.dumps(vars(alert), ensure_ascii=False, indent=2, default=str))
        elif cmd == "verify":
            self._verify(arg)
        elif cmd == "viz":
            from evolver.iter_viz import IterViz
            IterViz().generate_all()
        elif cmd == "export":
            self._export_experience(arg)
        elif cmd == "import":
            self._import_experience(arg)
        elif cmd == "hot-reload":
            self._config_hot_reload(arg)
        elif cmd == "dashboard":
            self._dashboard(arg)
        else:
            print(f"未知命令: {cmd} (输入 help 查看)")

    def _stop_daemon_if_running(self):
        if self.daemon and self.daemon.running:
            self.daemon.request_stop()

    def _dashboard(self, arg: str):
        """Web Dashboard 控制"""
        sub = arg.strip().lower() or "status"
        
        if sub == "start":
            port = 8080
            parts = arg.split()
            if len(parts) > 1 and parts[1].isdigit():
                port = int(parts[1])
            
            from evolver.web_dashboard import start_dashboard
            if start_dashboard(self.daemon, port):
                print(f"🌐 Dashboard 已启动: http://localhost:{port}")
            else:
                print("❌ Dashboard 启动失败")
        
        elif sub == "stop":
            from evolver.web_dashboard import stop_dashboard
            stop_dashboard()
            print("🛑 Dashboard 已停止")
        
        elif sub == "status":
            from evolver.web_dashboard import get_dashboard_status
            status = get_dashboard_status()
            if status["running"]:
                print(f"🌐 Dashboard 运行中: http://localhost:{status['port']}")
            else:
                print("⚪ Dashboard 未运行")
        
        else:
            print("用法: dashboard start [端口] | stop | status")
    def _config_hot_reload(self, arg: str):
        """配置热重载控制"""
        sub = arg.strip().lower() if arg else "status"
        
        if sub == "status":
            # 显示当前状态
            from config import Config as _Cfg
            mtime = _Cfg.get_config_mtime()
            if mtime > 0:
                from datetime import datetime
                print(f"📄 config.json 最后修改: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🔄 自动热重载: {'启用' if getattr(_Cfg, 'HOT_RELOAD_ENABLED', True) else '禁用'}")
            return
        
        if sub in ("on", "enable"):
            from config import Config as _Cfg
            _Cfg.HOT_RELOAD_ENABLED = True
            print("✅ 自动热重载已开启")
            return
        
        if sub in ("off", "disable"):
            from config import Config as _Cfg
            _Cfg.HOT_RELOAD_ENABLED = False
            print("⏸️ 自动热重载已关闭")
            return
        
        if sub == "now" or sub == "trigger":
            # 手动触发一次重载
            from config import Config as _Cfg
            _Cfg.load_from_file()
            print("✅ 配置已手动重载")
            return
        
        print("用法: config hot-reload [status|on|off|now]")

    def _export_experience(self, arg: str):
        """导出成功经验"""
        from pathlib import Path
        from success_patterns import get_pattern_library
        from memory_store import get_memory_store
        
        arg = arg.strip()
        if arg == "list":
            print(f"\n📊 经验库统计:")
            
            # 1. success_patterns.json（成功模式库）
            lib = get_pattern_library()
            stats = lib.get_statistics()
            print(f"\n  📁 成功模式库 (success_patterns.json):")
            print(f"    总条目: {stats.get('total', 0)}")
            
            if stats.get('total', 0) > 0:
                print(f"    平均质量提升: {stats.get('avg_improvement', 0):.1%}")
                print(f"    平均成功率: {stats.get('avg_success_rate', 0):.0%}")
                print(f"    总使用次数: {stats.get('total_uses', 0)}")
                if stats.get('top_dimensions'):
                    print(f"    热门维度:")
                    for dim, count in stats['top_dimensions']:
                        print(f"      - {dim}: {count} 次")
            else:
                print("    📭 暂无数据")
            
            # 2. memory_store.db（长期记忆库）
            try:
                store = get_memory_store()
                mem_stats = store.get_stats()
                print(f"\n  🧠 长期记忆库 (memory_store.db):")
                print(f"    总条目: {mem_stats.get('total_entries', 0)}")
                
                if mem_stats.get('total_entries', 0) > 0:
                    verdict_dist = mem_stats.get('verdict_distribution', {})
                    if verdict_dist:
                        print(f"    判定分布:")
                        for verdict, count in verdict_dist.items():
                            print(f"      - {verdict}: {count} 次")
                else:
                    print("    📭 暂无数据")
            except Exception as e:
                print(f"    ⚠️ 读取失败: {e}")
            
            return
        
        if not arg:
            print("用法: export <文件名>  或  export list")
            return
        
        file_path = Path(arg)
        if not file_path.suffix:
            file_path = file_path.with_suffix('.json')
        
        library = get_pattern_library()
        library.export(file_path)
    def _import_experience(self, arg: str):
        """导入成功经验"""
        from pathlib import Path
        from success_patterns import get_pattern_library
        
        if not arg:
            print("用法: import <文件名>")
            return
        
        file_path = Path(arg)
        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return
        
        library = get_pattern_library()
        count = library.import_from_file(file_path)
        if count > 0:
            print(f"  当前经验库总条目: {len(library.patterns)}")

    def _daemon(self, arg: str):
        sub = arg.strip() or "status"
        if sub == "start":
            if self.daemon and self.daemon.running:
                print("⚠️ Daemon 已在运行"); return
            # 自动清除旧的 kill-switch 文件（用户手动 start 意味着想要运行）
            ks = Path(Config.LOG_DIR) / "STOP_DAEMON"
            if ks.exists():
                ks.unlink()
                print("🧹 已清除旧的 kill-switch 文件")
            # ★ 根因1+2修复：daemon start 前置动作
            _archive_and_reset_daemon_log()   # 先归档历史日志（含旧cycle/recent等）
            _clear_evolver_module_cache()     # 再清模块缓存（强制加载最新代码）
            print("⏳ 初始化认知架构模块...")
            evolver = self._get_evolver()
            self.daemon = _make_daemon(evolver)
            # 统一走 AutonomousDaemon.start()：正确设置 _thread / daemon_protection，
            # 使 daemon status 能反映真实运行状态（此前裸 threading.Thread 绕过导致 status 永远「未启动」）
            self.daemon.start()
            print("🚀 AutonomousDaemon 已启动（后台线程，无人值守）")
        elif sub == "stop":
            if self.daemon:
                self.daemon.stop(); print("🛑 已请求停止 Daemon")
            else: print("Daemon 未运行")
        elif sub == "status":
            if self.daemon: self.daemon.print_status()
            else: print("Daemon 未运行")
        elif sub == "kill":
            p = Path(Config.LOG_DIR) / "STOP_DAEMON"; p.touch()
            print(f"💀 已创建 kill-switch: {p}")
        elif sub == "clear":
            p = Path(Config.LOG_DIR) / "STOP_DAEMON"
            if p.exists(): p.unlink(); print("🧹 已清除 kill-switch")
            else: print("无 kill-switch")
        else:
            print("用法: daemon [start|stop|status|kill|clear]")

    def _status(self):
        evolver = self._get_evolver()
        r = evolver.get_status_report()
        print("═══ 系统状态 ═══")
        mode_label, model_name = self._mode_display()
        print(f"  引擎: {mode_label} ({model_name})")
        print(f"  当前模型: {r['model_id']}")
        print(f"  迭代总次数: {r['total_iterations']}")
        for s, c in r["iteration_status_distribution"].items():
            print(f"    - {s}: {c}次")
        print(f"  累计成功修改: {r['total_applied_modifications']}")
        print(f"  备份文件数: {r['backup_count']}")

    def _config(self, arg: str):
        """config 子命令分发：show(默认) / entropy / ple"""
        sub_parts = arg.strip().split(maxsplit=2)
        sub = sub_parts[0].lower() if sub_parts else ""
        arg = " ".join(sub_parts[1:]) if len(sub_parts) > 1 else ""

        # 默认无参数 → show
        if sub in ("", "show", "status") and not sub_parts or (sub == "" and len(sub_parts) <= 1):
            # 兼容 "config ple status"
            if len(sub_parts) >= 2 and sub_parts[0].lower() == "ple":
                self._config_ple(" ".join(sub_parts[1:]))
                return
            self._config_show()
            return

        if sub == "ple":
            self._config_ple(arg)
        elif sub == "entropy":
            self._config_entropy(arg)
        else:
            print(f"未知 config 子命令: {sub}")
            print("用法: config [show] | config ple [status|reset] | config entropy [set <c> <w>|reset]")

    def _config_ple(self, arg: str):
        """PLE 完整性子命令：status（默认） / reset"""
        sub = arg.strip().lower() or "status"
        if sub == "reset":
            from config import Config
            Config.ple_save_manifest(force=True)
            print("✅ 已重建 PLE 签名基线（核心文件当前快照被视为合法）")
            print("   ⚠️ 如果核心引擎文件是被篡改的，执行此命令将永久覆盖原基线！")
            return
        if sub in ("status", "check", "verify"):
            from config import Config
            ok, msg = Config.ple_verify()
            if ok:
                print(f"✅ {msg}")
                m = Config.ple_compute_manifest()
                print(f"\n当前 7 个核心文件签名:")
                for rel, h in sorted(m["files"].items()):
                    print(f"  {rel:<38s}  sha256={h[:16]}...")
                print(f"\n顶层签名（防 manifest 替换攻击）: {m['manifest_signature'][:16]}...")
            else:
                print(f"🚨 {msg}")
                print("   → 如为合法改动：执行 config ple reset")
                print("   → 如为可疑篡改：恢复 backups/ 中对应文件后再 reset")
            return
        print(f"未知 ple 子命令: {sub}，用法: config ple [status|reset]")

    def _config_entropy(self, arg: str):
        """entropy 子命令：set / reset"""
        sub_parts = arg.strip().split(maxsplit=2)
        sub = sub_parts[0].lower() if sub_parts else "status"
        if sub == "set" and len(sub_parts) >= 3:
            try:
                c = float(sub_parts[1])
                w = float(sub_parts[2])
                if not (0.0 < w < c < 1.0):
                    raise ValueError("需满足 0 < warning < critical < 1")
            except Exception as e:
                print(f"❌ 参数错误: {e}")
                print("   例: config entropy set 0.42 0.25")
                return
            from conflict_boundary import ConflictBoundary
            cb = ConflictBoundary()
            cb.entropy_gap_critical = c
            cb.entropy_gap_warning = w
            cb.save_state()
            print(f"✅ 熵差阈值已更新: critical={c}, warning={w}")
            return
        if sub == "reset":
            from conflict_boundary import ConflictBoundary
            cb = ConflictBoundary()
            cb.entropy_gap_critical = cb.DEFAULT_ENTROPY_GAP_CRITICAL
            cb.entropy_gap_warning = cb.DEFAULT_ENTROPY_GAP_WARNING
            cb.save_state()
            print(f"✅ 熵差阈值恢复默认: critical={cb.entropy_gap_critical}, warning={cb.entropy_gap_warning}")
            return
        # status
        from conflict_boundary import ConflictBoundary
        cb = ConflictBoundary()
        print(f"critical={cb.entropy_gap_critical} (默认{cb.DEFAULT_ENTROPY_GAP_CRITICAL}), warning={cb.entropy_gap_warning} (默认{cb.DEFAULT_ENTROPY_GAP_WARNING})")

    def _rollback(self, arg: str):
        """统一回滚入口（P3）：自动判断迭代级 vs 文件级

        约定:
          - 无 arg / list → 展示最近 10 次迭代快照
          - 'last' / 'prev' → 恢复最新 iter 标签下全部文件
          - 纯数字 → 恢复该 iter 下全部备份（迭代级快照语义）
          - 含 .py / '/' 或 'list <path>' → 文件级回滚
        """
        import re
        from pathlib import Path as P
        from evolver.code_manager import CodeManager

        backup_dir = Config.BACKUP_DIR
        cm = CodeManager(Config.PROJECT_ROOT, backup_dir, auto_backup=False)

        sub_parts = arg.strip().split(maxsplit=1)
        head = sub_parts[0] if sub_parts else ""
        rest = sub_parts[1] if len(sub_parts) > 1 else ""

        # ── 构建 iter 索引 (备份名: 20260811_011654_iter455_evolver__meta_drive.py) ──
        iter_re = re.compile(r"_iter(\d+)_")
        backups = sorted(backup_dir.glob("*"), reverse=True)  # 最新在前
        iter_index: dict = {}
        file_index: dict = {}
        for b in backups:
            m = iter_re.search(b.name)
            if m:
                n = int(m.group(1))
                iter_index.setdefault(n, []).append(b)
            # file 索引：从备份名还原相对路径部分
            name = b.name
            # 截掉 timestamp(15) + "_" + 可能的 iterXXX
            parts = name.split("_", 2)
            if len(parts) == 3:
                rest_part = parts[2]
                # 去掉 iterXXX 前缀（如果有）
                mm = iter_re.match(rest_part)
                if mm:
                    rest_part = rest_part[mm.end():]
                rel = rest_part.replace("__", "/").replace("\\", "/")
                file_index.setdefault(rel, []).append(b)
                # 同时存个短名（无路径前缀）
                short = rel.split("/")[-1]
                if short != rel:
                    file_index.setdefault(short, []).append(b)
                # evolver/xxx.py 这种也可能写成相对 xxx.py
                if rel.startswith("evolver/"):
                    short2 = rel[len("evolver/"):]
                    file_index.setdefault(short2, file_index.get(rel, []))

        # ── 1) 无参数 → 展示最近 10 次迭代快照 ──
        if not head:
            iters = sorted(iter_index.keys(), reverse=True)[:10]
            if not iters:
                print("📦 尚未找到任何迭代级备份（backups/ 为空或未打 iter 标签）")
                return
            print(f"📦 最近 {len(iters)} 次迭代级快照（分支）:")
            for it in iters:
                files = iter_index[it]
                ts = files[0].name[:15]  # 20260811_011654
                print(f"  iter {it:<5d}  时间 {ts}  包含 {len(files):>2d} 个文件:  "
                      + ", ".join(sorted({self._backup_short_name(f.name) for f in files})[:5])
                      + (" ..." if len(files) > 5 else ""))
            print(f"\n  用法: rollback 455   (恢复 iter=455 所有文件)")
            print(f"        rollback last  (恢复最近一次迭代)")
            return

        # ── 2) "list <file>" 列出某文件所有备份 ──
        if head.lower() == "list" and rest:
            key = rest.replace("\\", "/").strip()
            candidates = file_index.get(key, [])
            if not candidates:
                # 模糊查找
                for k, v in file_index.items():
                    if key in k:
                        candidates.extend(v)
            if not candidates:
                print(f"❌ 未找到任何与 [{key}] 匹配的备份")
                print(f"   提示: 不带前缀直接写文件名，例如 meta_drive.py")
                return
            print(f"📜 [{key}] 的备份历史（最新在前）:")
            for idx, b in enumerate(candidates[:15]):
                print(f"  [{idx}] {b.name}")
            if len(candidates) > 15:
                print(f"  ... 还有 {len(candidates)-15} 个更早版本")
            return

        # ── 3) 纯数字 / last / prev → 迭代级回滚 ──
        target_iter: int = -1
        if head.lower() in ("last", "prev", "latest"):
            if iter_index:
                target_iter = sorted(iter_index.keys(), reverse=True)[0]
            else:
                print("❌ backups/ 没有带 iter 标签的备份，无法使用迭代级回滚")
                return
        elif head.isdigit():
            target_iter = int(head)
            if target_iter not in iter_index:
                print(f"❌ backups/ 未找到 iter={target_iter} 的任何文件备份")
                alt = sorted(iter_index.keys(), reverse=True)[:5]
                if alt:
                    print(f"   可选的最近 iter: {alt}")
                return

        if target_iter >= 0:
            files = iter_index[target_iter]
            print(f"⏳ 迭代级回滚 → iter={target_iter}，将恢复 {len(files)} 个文件:")
            ok_count = 0
            for f in files:
                restored_rel = None
                # 用 CodeManager 推断
                # 先读最新版做备份再覆盖？这里直接恢复不再额外备份（因为恢复的就是旧备份）
                if cm.restore_backup(str(f)):
                    ok_count += 1
                else:
                    print(f"  ❌ 失败: {f.name}")
            print(f"\n✅ 迭代级回滚完成: {ok_count}/{len(files)} 个文件已还原到 iter={target_iter} 状态")
            print(f"   建议立即执行: py_compile 语法检查  或  daemon 验证回归")
            return

        # ── 4) 其他 → 文件级回滚（匹配 head 为相对路径或短名） ──
        key = head.replace("\\", "/")
        candidates = file_index.get(key, [])
        if not candidates:
            # 模糊
            for k, v in file_index.items():
                if key.endswith(k) or k.endswith(key) or key in k:
                    candidates.extend(v)
            candidates = list(dict.fromkeys(candidates))  # 去重保序
        if not candidates:
            print(f"❌ 未找到与 [{key}] 匹配的备份")
            print(f"   支持: 完整路径 evolver/meta_drive.py 或短名 meta_drive.py")
            print(f"   查看可用文件备份: rollback list meta_drive.py")
            return
        latest = candidates[0]  # 最新
        print(f"⏳ 文件级回滚 → 恢复 [{key}] 最近一次备份:")
        print(f"   来源: {latest.name}")
        if cm.restore_backup(str(latest)):
            print(f"✅ 文件级回滚完成: {key}")
        else:
            print(f"❌ 文件级回滚失败")

    @staticmethod
    def _backup_short_name(backup_name: str) -> str:
        """从 iter455_evolver__meta_drive.py 截取成 evolver/meta_drive.py 短名展示"""
        import re
        m = re.search(r"_iter\d+_(.+)$", backup_name)
        rest = m.group(1) if m else backup_name
        # 去掉可能残留的 timestamp 前缀
        if not m and len(rest) > 15 and rest[8] == '_':
            rest = rest.split("_", 2)[-1] if "_" in rest else rest
            m2 = re.search(r"_iter\d+_(.+)$", rest)
            if m2:
                rest = m2.group(1)
        return rest.replace("__", "/")

    def _config_show(self):
        print("═══ 运行时配置 ═══")
        evolver = self._get_evolver() if not self.daemon else self.daemon.evolver
        client = getattr(evolver, 'doubao', None) if evolver else None
        if Config.is_jiyuan_mode():
            print(f"推理引擎:      基元律动 TokenRhythm")
            print(f"Base URL:      {Config.JIYUAN_BASE_URL}")
            print(f"模型:          {Config.JIYUAN_MODEL}")
            print(f"temperature:   {getattr(client, 'temperature', 'N/A')}")
            print(f"max_tokens:    {getattr(client, 'max_tokens', 'N/A')}")
        elif Config.CLIENT_MODE == "api":
            print(f"推理引擎:      火山方舟 API")
            print(f"Base URL:      {Config.ARK_BASE_URL}")
            print(f"模型:          {Config.MODEL_ID}")
            print(f"temperature:   {getattr(client, 'temperature', 'N/A')}")
            print(f"max_tokens:    {getattr(client, 'max_tokens', 'N/A')}")
        elif Config.CLIENT_MODE == "lmstudio":
            print(f"推理引擎:      LM Studio")
            print(f"Base URL:      {Config.LMSTUDIO_BASE_URL}")
            print(f"模型:          {getattr(client, 'model_id', Config.LMSTUDIO_MODEL)}")
        else:
            print(f"推理引擎:      Ollama")
            print(f"Ollama 地址:   {Config.OLLAMA_BASE_URL}")
            print(f"模型:          {getattr(client, 'model_id', Config.OLLAMA_MODEL)}")
            print(f"num_ctx:       {getattr(client, 'num_ctx', Config.OLLAMA_NUM_CTX)}")
            profile = getattr(client, 'profile', None) if client else None
            print(f"num_predict:   {profile.calculate_num_predict() if profile else 'N/A'}")
            print(f"temperature:   {profile.calculate_temperature() if profile else 'N/A'}")
            if profile:
                adapted = profile.to_dict().get('adapted', {})
                print(f"budget_factor: {adapted.get('budget_factor', 1.0):.2f}  (调用 {profile._total_calls} 次, 失败率 {profile._total_failures/max(profile._total_calls,1):.0%})")
        from conflict_boundary import ConflictBoundary
        cb = ConflictBoundary()
        print(f"\n🧠 认知熵: critical={cb.entropy_gap_critical} (默认{cb.DEFAULT_ENTROPY_GAP_CRITICAL}), warning={cb.entropy_gap_warning} (默认{cb.DEFAULT_ENTROPY_GAP_WARNING})")
        from meta_drive import MetaDrive
        md = MetaDrive()
        cur = md.get_effective_silence_prob()
        boost = md._dynamic_silence_boost
        print(f"📡 静默概率: 基础{md.SILENCE_PROBABILITY*100:.0f}% boost{boost:+.2f} 当前{cur*100:.0f}% (钳位{md.SILENCE_PROB_MIN*100:.0f}%~{md.SILENCE_PROB_MAX*100:.0f}%)")

    def _verify(self, arg: str):
        """隐患1保障：一键全量验证工具（直接调python.exe内部API，绕开PS脚本限制）
        用法: verify [syntax|contracts|ast]，无参数=全部三项
        """
        sub = arg.strip().lower() or "all"
        total_ok = True
        t0 = __import__("time").time()

        # ── 验证1: py_compile 语法检查 ──
        if sub in ("all", "syntax"):
            print("\n" + "="*60)
            print("  [1/3] 📝 py_compile 语法检查（核心引擎 + 可修改模块）")
            print("="*60)
            import py_compile
            # 优先从 Config.PROTECTED_FILES（PLE 核心）+ 可修改模块取
            try:
                core = list(Config.PROTECTED_FILES or [])
            except Exception:
                core = []
            extra = [
                "evolver/branch_manager.py",
                "evolver/meta_drive.py",
                "evolver/code_manager.py",
                "evolver/modification_parser.py",
                "evolver/capability_registry.py",
                "evolver/contracts.py",
            ]
            targets = list(dict.fromkeys(core + extra))  # 去重保序
            ok_cnt = 0
            for rel in targets:
                p = _PROJECT_ROOT / rel
                if not p.exists():
                    # 兼容带 evolver/ 前缀和不带前缀两种
                    p2 = _PROJECT_ROOT / "evolver" / rel if not rel.startswith("evolver/") else p
                    p = p2 if p2.exists() else p
                try:
                    py_compile.compile(str(p), doraise=True)
                    ok_cnt += 1
                    print(f"  ✅ {rel:<40s}  syntax OK")
                except Exception as e:
                    total_ok = False
                    print(f"  ❌ {rel:<40s}  SYNTAX FAIL: {type(e).__name__}: {e}")
            print(f"  语法检查结果: {ok_cnt}/{len(targets)} 通过")

        # ── 验证2: 合约回归 ContractSuite ──
        if sub in ("all", "contracts", "contract"):
            print("\n" + "="*60)
            print("  [2/3] 🔒 合约回归（ContractSuite 全量）")
            print("="*60)
            try:
                from evolver.contracts import ContractSuite
                suite = ContractSuite()
                report = suite.run_all(verbose=False)
                ratio = report.pass_ratio
                mark = "✅" if report.failed == 0 else "🚨"
                print(f"  {mark} Contracts: {report.passed}✅ {report.failed}❌ {report.skipped}⏭️  → {ratio:.0%} pass")
                if report.failed > 0:
                    total_ok = False
                    print("  失败详情（前3行）:")
                    for line in report.failures_summary().splitlines()[:3]:
                        print(f"     - {line}")
                    print(f"  完整失败数: {report.failed}，查看详细: python main.py verify contracts")
            except Exception as e:
                total_ok = False
                print(f"  ❌ ContractSuite 执行异常: {type(e).__name__}: {e}")

        # ── 验证3: AST 方法存在性（BranchManager + 核心类） ──
        if sub in ("all", "ast"):
            print("\n" + "="*60)
            print("  [3/3] 🔬 AST 方法存在性检查（目标循环根因验证）")
            print("="*60)
            import ast
            checks = [
                ("evolver/branch_manager.py", "BranchManager", ["list_all_branches", "create_branch"]),
            ]
            for rel, cls_name_candidates, methods in checks:
                p = _PROJECT_ROOT / rel
                if not p.exists():
                    print(f"  ⚠️ {rel} 文件不存在，跳过")
                    continue
                try:
                    src = open(p, "r", encoding="utf-8").read()
                    tree = ast.parse(src, filename=rel)
                    # 收集所有类内方法名
                    class_methods = {}
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_methods[node.name] = {
                                n.name for n in node.body if isinstance(n, ast.FunctionDef)
                            }
                    # 从候选类名中匹配
                    found_cls = None
                    for cn in (cls_name_candidates if isinstance(cls_name_candidates, list) else [cls_name_candidates]):
                        if cn in class_methods:
                            found_cls = cn
                            break
                    if found_cls is None and class_methods:
                        # 类名不匹配候选则取第一个有方法的类兜底
                        found_cls = next(iter(class_methods.keys()))
                    ms = class_methods.get(found_cls or "", set())
                    sub_ok = True
                    for m in methods:
                        if m in ms:
                            print(f"  ✅ {rel} :: {found_cls}.{m}()  已存在")
                        else:
                            total_ok = False
                            sub_ok = False
                            print(f"  ❌ {rel} :: ???.{m}()  MISSING → 会触发目标循环！")
                    if sub_ok:
                        print(f"     → {rel} 所有合约方法已覆盖，不会触发『反复添加方法』循环")
                except SyntaxError as e:
                    total_ok = False
                    print(f"  ❌ {rel} AST 解析失败: {e}")
                except Exception as e:
                    total_ok = False
                    print(f"  ❌ {rel} 检查异常: {type(e).__name__}: {e}")

        elapsed = __import__("time").time() - t0
        print("\n" + "="*60)
        if total_ok:
            print(f"  ✅✅✅ 验证全部通过 ✅✅✅  耗时 {elapsed:.1f}s")
            print("     → 可以安全启动 daemon: python main.py daemon start")
        else:
            print(f"  🚨 存在失败项（耗时 {elapsed:.1f}s）")
            print("     → 修复后再启动 daemon，避免空转/停机循环")
        print("="*60 + "\n")


def main():
    if len(sys.argv) > 1:
        Config.ensure_dirs()
        sub_cmd = sys.argv[1].lower()
        if sub_cmd == "daemon":
            arg = sys.argv[2] if len(sys.argv) > 2 else "status"
            # ★ 根因1+2修复：CLI daemon start 也要先归档 + 清缓存再 import
            # （注意：CLI 模式是每命令独立进程，模块缓存风险本就低于交互式；
            #  但 daemon_log 归档、周期计数器清零是必须的）
            if arg == "start":
                ks = Path(Config.LOG_DIR) / "STOP_DAEMON"
                if ks.exists():
                    ks.unlink()
                    print("🧹 已清除旧的 kill-switch 文件")
                _archive_and_reset_daemon_log()
                _clear_evolver_module_cache()
            from self_evolver import SelfEvolver
            evolver = SelfEvolver()
            d = _make_daemon(evolver)
            if arg == "start":
                d._daemon_loop()
            elif arg == "status":
                d.print_status()
            else:
                print(f"用法: python main.py daemon start")
        elif sub_cmd == "status":
            from self_evolver import SelfEvolver
            import json
            print(json.dumps(SelfEvolver().get_status_report(), ensure_ascii=False, indent=2))
        elif sub_cmd == "config":
            from config import Config as C
            print(f"model={C.OLLAMA_MODEL}, num_ctx={C.OLLAMA_NUM_CTX}")
        elif sub_cmd == "verify":
            # 隐患1：直接走内部 API，绕开 PowerShell 脚本执行策略
            arg = sys.argv[2] if len(sys.argv) > 2 else ""
            cli = InteractiveCLI()
            cli._verify(arg)
        else:
            print(f"未知命令: {sub_cmd}")
            print("支持: daemon start | status | config | verify [syntax|contracts|ast]")
        return

    InteractiveCLI().run()


if __name__ == "__main__":
    main()

