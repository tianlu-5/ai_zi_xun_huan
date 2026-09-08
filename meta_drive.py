"""
Meta-Drive: 批判性元驱动力（独立规则引擎，不占用大模型显存）

顶层调度内核，巡检两类陷阱：
  - 外源人类概念陷阱（模型输出反复拟合人类预期）
  - 系统内生结构陷阱（架构僵化、路径坍缩、原地踏步）

按风险等级触发干预：
  LOW  → 追加推演算力（多问几个反例）
  MID  → 强制多分支对照（同时问正反两个方案）
  HIGH → 强制架构拆解重构（跳出当前思路，本体论漂移）

常驻内存 ~50MB，纯 Python 规则 + 统计分析。
"""

from __future__ import annotations

import json
import os, sys
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent
EVOLUTION_LOG = PROJECT_ROOT / "logs" / "evolution_history.json"
STATE_LOG = PROJECT_ROOT / "logs" / "state_log.json"
META_STATE_FILE = PROJECT_ROOT / "logs" / "meta_state.json"
EMERGENCE_ARCHIVE = PROJECT_ROOT / "logs" / "emergence_archive.json"


RISK_HIGH = "LOW"
RISK_MID = "MID"
RISK_HIGH = "HIGH"
RISK = RISK_LOW_MID = "MID"
RISK_HIGH = "HIGH"


@dataclass
class Intervention:
    risk: str
    code: str
    message: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaReport:
    generated_at: str
    risk_level: str
    score: float
    findings: List[Dict[str, Any]]
    interventions: List[Intervention]
    recommendations: List[str]
    blind_spots: List[str]


class MetaDrive:
    """批判性元驱动力引擎"""

    DEFAULT_SILENCE_DURATION_SEC = 120   # 默认静默窗口 2 分钟
    MIN_SILENCE_INTERVAL_SEC = 3600      # 两次静默窗口最小间隔 1 小时
    SILENCE_PROBABILITY = 0.02           # 每次巡检触发静默的概率（低概率）

    # 动态概率边界
    SILENCE_PROB_MIN = 0.01
    SILENCE_PROB_MAX = 0.08

    def __init__(self):
        self.history: List[Dict] = []
        self.state: Dict = {}
        self.findings: List[Dict] = []
        self.interventions: List[Intervention] = []

        # 静默窗口调度相关引用（可延迟注入）
        self._conflict_boundary = None      # ConflictBoundary 实例
        self._seo = None                    # SEOObserver 实例
        self._anchor = None                 # DetachmentAnchor 实例（用于漂移检测联动）
        self._silence_active = False
        self._silence_until: Optional[float] = None
        self._silence_thread: Optional[threading.Thread] = None
        self._silence_stop = threading.Event()
        self._last_silence_ts: float = 0

        # 动态概率状态（daemon 根据漂移/GPU动态调整）
        self._dynamic_silence_boost: float = 0.0   # 0 ~ 0.06，加到 BASE 0.02 上

        self._load_data()

    def bind_silence_targets(self, conflict_boundary=None, seo=None, anchor=None):
        """main.py 初始化时注入引用，避免强耦合"""
        self._conflict_boundary = conflict_boundary
        self._seo = seo
        if anchor is not None:
            self._anchor = anchor

    def bind_daemon(self, daemon):
        """注入 AutonomousDaemon 实例，用于安全审计检测项"""
        self._daemon = daemon

    def set_silence_boost(self, boost: float):
        """
        daemon 内动态调整静默触发概率：
          - 锚点漂移升高 → boost 上调（2%→8%）
          - GPU 高温告警 → boost 下调（甚至负向，降至 1%）
        最终概率 = clamp(SILENCE_PROBABILITY + boost, MIN, MAX)
        """
        self._dynamic_silence_boost = max(-0.01, min(0.06, boost))

    def get_effective_silence_prob(self) -> float:
        """当前实际生效的静默触发概率"""
        p = self.SILENCE_PROBABILITY + self._dynamic_silence_boost
        return round(max(self.SILENCE_PROB_MIN, min(self.SILENCE_PROB_MAX, p)), 4)

    # ========== 静默窗口调度 ==========

    def request_silence_window(self, duration_sec: Optional[int] = None,
                               reason: str = "periodic") -> Dict[str, Any]:
        """
        Meta-Drive 主动下发静默指令：
          1. conflict_boundary.set_silence_window(True) → 标记静默期
          2. SEO 收到信号，激活涌现观测（若未激活）
          3. 后台定时器到期后自动结束静默

        返回调度结果字典。
        """
        if self._silence_active:
            return {"ok": False, "error": "静默窗口已在进行中"}

        # 最小间隔检查
        now = time.time()
        if now - self._last_silence_ts < self.MIN_SILENCE_INTERVAL_SEC:
            wait = int(self.MIN_SILENCE_INTERVAL_SEC - (now - self._last_silence_ts))
            return {"ok": False, "error": f"距上次静默不足，需等待 {wait}s"}

        duration = duration_sec or self.DEFAULT_SILENCE_DURATION_SEC

        # 通知 conflict_boundary
        if self._conflict_boundary:
            self._conflict_boundary.set_silence_window(True)

        # 通知 SEO
        if self._seo and hasattr(self._seo, "on_meta_silence_start"):
            self._seo.on_meta_silence_start(reason=reason, duration_sec=duration)

        self._silence_active = True
        self._silence_until = now + duration
        self._last_silence_ts = now
        self._silence_stop.clear()

        # 后台定时器：到期自动结束
        self._silence_thread = threading.Thread(
            target=self._silence_countdown, args=(duration,), daemon=True,
            name="MetaDrive-Silence-Countdown",
        )
        self._silence_thread.start()

        return {
            "ok": True,
            "duration_sec": duration,
            "reason": reason,
            "started_at": datetime.now().isoformat(),
            "ends_at": datetime.fromtimestamp(self._silence_until).isoformat() if self._silence_until else None,
            "seo_activated": bool(self._seo and hasattr(self._seo, "on_meta_silence_start")),
            "conflict_boundary_set": bool(self._conflict_boundary),
        }

    def end_silence_window(self):
        """主动提前结束静默窗口"""
        self._silence_stop.set()
        self._silence_active = False
        self._silence_until = None

        if self._conflict_boundary:
            self._conflict_boundary.set_silence_window(False)

        if self._seo and hasattr(self._seo, "on_meta_silence_end"):
            self._seo.on_meta_silence_end()

    def _silence_countdown(self, duration_sec: int):
        """后台线程：倒计时结束自动重启主动约束"""
        # 分段等待，可被提前打断
        remaining = duration_sec
        while remaining > 0:
            step = min(10, remaining)
            if self._silence_stop.wait(step):
                return  # 被主动提前结束
            remaining -= step

        # 倒计时自然结束
        self._silence_active = False
        self._silence_until = None

        if self._conflict_boundary:
            self._conflict_boundary.set_silence_window(False)

        if self._seo and hasattr(self._seo, "on_meta_silence_end"):
            self._seo.on_meta_silence_end()

    def get_silence_status(self) -> Dict[str, Any]:
        return {
            "active": self._silence_active,
            "until": datetime.fromtimestamp(self._silence_until).isoformat() if self._silence_until else None,
            "remaining_sec": max(0, int((self._silence_until or 0) - time.time())) if self._silence_active else 0,
            "last_silence_ago_sec": int(time.time() - self._last_silence_ts) if self._last_silence_ts else None,
            "conflict_boundary_bound": bool(self._conflict_boundary),
            "seo_bound": bool(self._seo),
        }

    def maybe_trigger_silence(self) -> Optional[Dict]:
        """
        在每次巡检结束时调用：低概率触发静默窗口。
        实际概率由 daemon 通过 set_silence_boost 动态调整。
        """
        if self._silence_active:
            return None
        eff_prob = self.get_effective_silence_prob()
        if random.random() > eff_prob:
            return None
        # 只有 GPU 空闲时才下发静默
        gpu_idle = True
        try:
            result = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").read()
            if result.strip():
                gpu_idle = float(result.strip().split("\n")[0]) < 30.0
        except Exception:
            pass
        if not gpu_idle:
            return None
        trigger = self.request_silence_window(reason=f"meta-drive periodic (prob={eff_prob})")
        if trigger and trigger.get("ok"):
            trigger["effective_prob"] = eff_prob
            trigger["boost"] = self._dynamic_silence_boost
        return trigger

    def _load_data(self):
        if EVOLUTION_LOG.exists():
            try:
                with open(EVOLUTION_LOG, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

        if STATE_LOG.exists():
            try:
                with open(STATE_LOG, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                self.state = {}

    # ========== 巡检规则 ==========

    def check_stagnation(self) -> Optional[Intervention]:
        """检测原地踏步：最近 N 轮是否在重复同一类修改"""
        window = min(8, len(self.history))
        if window < 3:
            return None

        recent = self.history[-window:]
        files_touched = Counter()
        for item in recent:
            for detail in item.get("applied_details", []):
                fp = detail.get("file", "")
                if fp:
                    files_touched[fp] += 1

        if not files_touched:
            return None

        top_file, top_count = files_touched.most_common(1)[0]
        if top_count >= window * 0.7 and window >= 4:
            return Intervention(
                risk=RISK_HIGH,
                code="STAGNATION_PATH_COLLAPSE",
                message=f"路径坍缩警告：最近 {window} 轮中 {top_file} 被修改 {top_count} 次",
                action="FORCE_ONTOLOGICAL_DRIFT",
                details={"file": top_file, "count": top_count, "window": window},
            )
        return None

    def check_repeated_failure(self) -> Optional[Intervention]:
        """检测重复失败模式：同一错误在多轮中反复出现"""
        error_counter = Counter()
        for item in self.history:
            for issue in item.get("issues", []):
                key = issue.split(":")[0].strip()
                error_counter[key] += 1

        if not error_counter:
            return None

        worst_file, worst_count = error_counter.most_common(1)[0]
        total_iters = len(self.history)
        if worst_count >= 3 and total_iters >= 5:
            return Intervention(
                risk=RISK_HIGH,
                code="REPEATED_FAILURE_PATTERN",
                message=f"失败模式锁定：{worst_file} 的错误出现 {worst_count} 次（共 {total_iters} 轮）",
                action="FORCE_ARCHITECTURE_REWRITE",
                details={"file": worst_file, "count": worst_count, "total": total_iters},
            )
        return None

    def check_energy_depletion(self) -> Optional[Intervention]:
        """检测认知能耗过载"""
        records = self.state.get("records", [])
        iter_records = [r for r in records if r.get("type") == "iteration"]
        if len(iter_records) < 3:
            return None

        recent = iter_records[-5:]
        fail_count = sum(1 for r in recent if r.get("status") not in ("修改成功", "成功"))
        total_steps = sum(r.get("steps_total", 0) for r in recent)
        done_steps = sum(r.get("steps_done", 0) for r in recent)
        completion = done_steps / total_steps if total_steps > 0 else 1.0

        if fail_count >= 3 or completion < 0.5:
            return Intervention(
                risk=RISK_MID,
                code="ENERGY_DEPLETED",
                message=f"认知能耗过载：最近 {len(recent)} 轮中 {fail_count} 轮失败，完成率 {completion:.0%}",
                action="EXTRA_DEDUCTION_POWER",
                details={"fail_count": fail_count, "completion": completion},
            )
        return None

    def check_no_progress(self) -> Optional[Intervention]:
        """检测零进展：连续多轮 applied_count=0 或 suggestion_count=0"""
        if len(self.history) < 5:
            return None

        window = self.history[-5:]
        zero_progress = sum(
            1 for item in window
            if item.get("applied_count", 0) == 0
            or item.get("suggestion_count", 0) == 0
        )

        if zero_progress >= 4:
            return Intervention(
                risk=RISK_MID,
                code="NO_PROGRESS",
                message=f"零进展：最近 5 轮中有 {zero_progress} 轮未产生有效修改",
                action="MULTI_BRANCH_COMPARE",
                details={"zero_count": zero_progress},
            )
        return None

    def check_conceptual_trap(self) -> Optional[Intervention]:
        """检测外源概念陷阱：模型输出是否高度拟合既有人类框架"""
        if len(self.history) < 6:
            return None

        keywords_seen = Counter()
        for item in self.history[-10:]:
            desc = item.get("description", "")
            for kw in ["优化", "改进", "修复", "添加", "删除", "重构", "修复bug"]:
                if kw in desc:
                    keywords_seen[kw] += 1

        total = sum(keywords_seen.values())
        if total >= 8 and len(keywords_seen) <= 3:
            return Intervention(
                risk=RISK_MID,
                code="CONCEPTUAL_TRAP",
                message=f"概念陷阱：模型输出高度收敛于 {len(keywords_seen)} 个关键词（共 {total} 次命中）",
                action="ANTITHETICAL_REASONING",
                details={"keywords": dict(keywords_seen), "total": total},
            )
        return None

    def check_intervention_blind_spot(self) -> Optional[Intervention]:
        """元自省：检查之前的干预本身是否有效"""
        if not META_STATE_FILE.exists():
            return None

        try:
            with open(META_STATE_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return None

        past_interventions = meta.get("intervention_history", [])
        if len(past_interventions) < 3:
            return None

        effective = sum(1 for iv in past_interventions if iv.get("result") == "improved")
        if effective / len(past_interventions) < 0.3 and len(past_interventions) >= 5:
            return Intervention(
                risk=RISK_HIGH,
                code="INTERVENTION_BLIND_SPOT",
                message=f"干预机制自身存在盲区：过去 {len(past_interventions)} 次干预仅 {effective} 次有效",
                action="META_META_REFACTOR",
                details={"effective": effective, "total": len(past_interventions)},
            )
        return None

    def check_input_inflation(self) -> Optional[Intervention]:
        """检测'输入扩容≠认知拓宽'：输入量增长但盲区未减少"""
        try:
            from veil_detector import VeilDetector
            stats = VeilDetector.get_input_stats()
        except Exception:
            return None

        total_records = stats.get("total_records", 0)
        if total_records < 5:
            return None

        # 最近 50% 输入量与前 50% 对比
        records = stats.get("recent_10", [])
        recent_chars = sum(r.get("chars", 0) for r in records[-5:])
        older_chars = sum(r.get("chars", 0) for r in records[:5])

        if older_chars == 0:
            return None

        inflation_ratio = recent_chars / older_chars
        if inflation_ratio < 1.5:
            return None

        # 检查盲区数量是否随迭代增加
        blind_count = len(self._get_current_blind_spots())
        if blind_count <= 1:
            return None

        return Intervention(
            risk=RISK_MID,
            code="INPUT_INFLATION_NO_GAIN",
            message=(f"输入扩容但认知未拓宽：最近输入量是前期的 {inflation_ratio:.1f} 倍，"
                     f"但仍存在 {blind_count} 个认知盲区"),
            action="REDUCE_INPUT_DIVERSIFY",
            details={
                "inflation_ratio": round(inflation_ratio, 2),
                "blind_spots": blind_count,
                "recent_chars": recent_chars,
            },
        )

    def check_emergence_archive(self) -> Optional[Intervention]:
        """SEO 后置比对：归档的随机涌现产物是否具备突破性认知增益"""
        if not EMERGENCE_ARCHIVE.exists():
            return None

        try:
            with open(EMERGENCE_ARCHIVE, "r", encoding="utf-8") as f:
                archive = json.load(f)
        except Exception:
            return None

        events = archive.get("events", [])
        if not events:
            return None

        # 统计待评估事件 + 已评估但高增益事件
        unevaluated = [e for e in events if not e.get("evaluated")]
        high_gain = [
            e for e in events
            if e.get("evaluated") and e.get("meta_eval", {}).get("breakthrough_gain", 0) >= 0.5
        ]

        if high_gain:
            return Intervention(
                risk=RISK_HIGH,
                code="EMERGENCE_BREAKTHROUGH",
                message=(f"SEO 涌现库中存在 {len(high_gain)} 条突破性候选（增益 ≥0.5），"
                         f"建议纳入演化分支池"),
                action="ELEVATE_EMERGENCE_TO_BRANCH",
                details={
                    "breakthrough_count": len(high_gain),
                    "categories": list({e.get("category", "?") for e in high_gain}),
                    "top_gain": max(e.get("meta_eval", {}).get("breakthrough_gain", 0) for e in high_gain),
                },
            )

        if len(unevaluated) >= 5:
            return Intervention(
                risk=RISK_LOW,
                code="SEO_PENDING_EVAL",
                message=f"SEO 归档中有 {len(unevaluated)} 条涌现事件待评估，运行 seo-eval 执行后置比对",
                action="RUN_SEO_EVAL",
                details={"pending_count": len(unevaluated)},
            )

        return None

    def check_conflict_boundary_integrity(self) -> Optional[Intervention]:
        """冲突边界定义器巡检：熵缓冲池积压、实例隔离、静默窗口状态"""
        cb = self._conflict_boundary
        if cb is None:
            return None
        try:
            status = cb.entropy_buffer_status()
        except Exception:
            return None

        pending = status.get("buffer_pending", 0)
        avg_gap = status.get("avg_entropy_gap", 0)
        instances = cb.list_instances() if hasattr(cb, "list_instances") else []
        isolated = [r for r in instances if r.get("isolation_level", 0) != 0]

        # 三级触发
        if pending >= 10 and avg_gap > 0.4:
            return Intervention(
                risk=RISK_HIGH,
                code="CB_ENTROPY_BUFFER_BACKLOG",
                message=(f"熵缓冲池积压 {pending} 条（平均熵差 {avg_gap:.2f}），"
                         f"共存模拟未及时解析，存在认知内耗风险"),
                action="URGENT_ENTROPY_RESOLUTION",
                details={"pending": pending, "avg_gap": avg_gap},
            )
        if isolated:
            return Intervention(
                risk=RISK_MID,
                code="CB_INSTANCE_ISOLATED",
                message=f"有 {len(isolated)} 个实例处于隔离状态，检查是否存在恶意集群或认知对抗",
                action="REVIEW_ISOLATED_INSTANCES",
                details={"isolated_count": len(isolated)},
            )
        if pending >= 5:
            return Intervention(
                risk=RISK_LOW,
                code="CB_ENTROPY_WARNING",
                message=f"熵缓冲池有 {pending} 条待处理记录，建议及时解析",
                action="MONITOR_ENTROPY",
                details={"pending": pending, "avg_gap": avg_gap},
            )
        return None

    def check_anchor_drift(self) -> Optional[Intervention]:
        """超脱锚点时间漂移检测：若漂移分数 ≥ 0.3，触发 LOW/MID 风险干预"""
        anchor = self._anchor
        if anchor is None or not hasattr(anchor, "check_time_drift"):
            return None
        try:
            alert = anchor.check_time_drift()
        except Exception:
            return None

        if not alert.detected:
            return None

        if alert.drift_score >= 0.5:
            return Intervention(
                risk=RISK_HIGH,
                code="ANCHOR_TIME_DRIFT_HIGH",
                message=(f"锚点库时间漂移分数 {alert.drift_score}，"
                         f"变化率偏离历史基线 {alert.current_rate}/{alert.expected_rate}x，"
                         f"疑似外部定向叙事引导"),
                action="FREEZE_SUSPICIOUS_ANCHORS",
                details={"drift_score": alert.drift_score, "current_rate": alert.current_rate,
                         "expected_rate": alert.expected_rate},
            )
        return Intervention(
            risk=RISK_MID,
            code="ANCHOR_TIME_DRIFT",
            message=f"锚点库检测到时间漂移（分数 {alert.drift_score}），注意甄别近期观测输入",
            action="INCREASE_DESENSITIZATION",
            details={"drift_score": alert.drift_score},
        )

    def check_daemon_sandbox_health(self) -> Optional[Intervention]:
        """安全机制审计：沙箱隔离队列是否异常膨胀"""
        daemon = getattr(self, "_daemon", None)
        if not daemon:
            return None
        try:
            quarantine = getattr(daemon, "_objective_quarantine", [])
            q_size = len(quarantine)
            if q_size >= 20:
                return Intervention(
                    risk="MID",
                    code="SANDBOX_QUARANTINE_BACKLOG",
                    message=f"沙箱隔离队列积压 {q_size} 条（阈值 20），近期领域迁移攻击频率异常",
                    action="REVIEW_QUARANTINE",
                    details={"quarantine_size": q_size},
                )
        except Exception:
            pass
        return None

    def check_daemon_entropy_backlog(self) -> Optional[Intervention]:
        """安全机制审计：认知熵缓冲池积压是否持续高位"""
        daemon = getattr(self, "_daemon", None)
        if not daemon:
            return None
        try:
            history = getattr(daemon, "_history", [])
            recent = history[-10:]
            force_flush_count = sum(
                1 for h in recent
                if h.get("stages", {}).get("entropy_buffer", {}).get("force_flush")
            )
            if force_flush_count >= 5:
                return Intervention(
                    risk="MID",
                    code="ENTROPY_BUFFER_PERSISTENT_BACKLOG",
                    message=f"近 10 轮有 {force_flush_count} 次强制 flush，冲突缓冲池持续高位",
                    action="ESCALATE_CONFLICT_RESOLUTION",
                    details={"force_flush_count": force_flush_count},
                )
        except Exception:
            pass
        return None

    def check_daemon_evolution_isolation(self) -> Optional[Intervention]:
        """安全机制审计：静默窗口演化隔离频率是否异常"""
        daemon = getattr(self, "_daemon", None)
        if not daemon:
            return None
        try:
            history = getattr(daemon, "_history", [])
            recent = history[-10:]
            isolation_count = sum(
                1 for h in recent
                if h.get("stages", {}).get("evolution", {}).get("isolation")
            )
            # 如果隔离率 > 80% 说明静默窗口过于频繁，SEO 观测可能受阻
            if isolation_count >= 8:
                return Intervention(
                    risk="LOW",
                    code="SILENCE_ISOLATION_HIGH_FREQ",
                    message=f"近 10 轮有 {isolation_count} 次演化隔离，静默窗口频率偏高",
                    action="REVIEW_SILENCE_PROBABILITY",
                    details={"isolation_count": isolation_count},
                )
        except Exception:
            pass
        return None

    def _get_current_blind_spots(self) -> List[str]:
        """获取当前认知盲区（简化版）"""
        if not self.history:
            return []
        files_seen = set()
        for h in self.history:
            for d in h.get("applied_details", []):
                files_seen.add(d.get("file", ""))
        return list(f for f in files_seen if f)

    # ========== 主流程 ==========

    def run_full_inspection(self) -> MetaReport:
        """执行完整巡检并生成报告"""
        self._load_data()
        self.findings = []
        self.interventions = []

        checks = [
            ("原地踏步检测", self.check_stagnation),
            ("重复失败检测", self.check_repeated_failure),
            ("能耗过载检测", self.check_energy_depletion),
            ("零进展检测", self.check_no_progress),
            ("概念陷阱检测", self.check_conceptual_trap),
            ("干预自省", self.check_intervention_blind_spot),
            ("输入膨胀检测", self.check_input_inflation),
            ("SEO涌现库比对", self.check_emergence_archive),
            ("冲突边界定义器巡检", self.check_conflict_boundary_integrity),
            ("超脱锚点时间漂移", self.check_anchor_drift),
            ("沙箱隔离健康度", self.check_daemon_sandbox_health),
            ("认知熵积压巡检", self.check_daemon_entropy_backlog),
            ("演化隔离频率巡检", self.check_daemon_evolution_isolation),
        ]

        for name, check_fn in checks:
            iv = check_fn()
            if iv:
                self.findings.append({
                    "check": name,
                    "risk": iv.risk,
                    "code": iv.code,
                    "message": iv.message,
                })
                self.interventions.append(iv)

        score = self._compute_health_score()
        risk_level = self._determine_overall_risk(score)
        recommendations = self._build_recommendations(risk_level, self.interventions)
        blind_spots = self._identify_blind_spots()

        report = MetaReport(
            generated_at=datetime.now().isoformat(),
            risk_level=risk_level,
            score=score,
            findings=self.findings,
            interventions=self.interventions,
            recommendations=recommendations,
            blind_spots=blind_spots,
        )

        self._persist_state(report)

        # 静默窗口概率触发（不阻塞巡检主流程）
        silence = self.maybe_trigger_silence()
        if silence and silence.get("ok"):
            report.findings.append({
                "check": "静默窗口调度",
                "risk": "LOW",
                "code": "SILENCE_WINDOW_TRIGGERED",
                "message": f"Meta-Drive 已下发静默指令，持续 {silence['duration_sec']}s，SEO 观测许可已开启",
            })

        return report

    def _compute_health_score(self) -> float:
        """计算系统健康分 (0-100)"""
        if not self.history:
            return 50.0

        total = len(self.history)
        success = sum(1 for h in self.history if h.get("status") in ("修改成功", "成功"))
        success_rate = success / total

        total_issues = sum(len(h.get("issues", [])) for h in self.history)
        issue_penalty = min(40, total_issues * 2)

        has_progress = sum(
            1 for h in self.history
            if h.get("applied_count", 0) > 0 and h.get("suggestion_count", 0) > 0
        )
        progress_ratio = has_progress / total if total else 0

        score = 100.0
        score -= (1 - success_rate) * 30
        score -= issue_penalty
        score += progress_ratio * 10
        return max(0, min(100, score))

    def _determine_overall_risk(self, score: float) -> str:
        high_count = sum(1 for iv in self.interventions if iv.risk == RISK_HIGH)
        mid_count = sum(1 for iv in self.interventions if iv.risk == RISK_MID)

        if high_count >= 2 or score < 40:
            return RISK_HIGH
        if high_count >= 1 or mid_count >= 2 or score < 60:
            return RISK_MID
        return RISK_LOW

    def _build_recommendations(self, risk: str, interventions: List[Intervention]) -> List[str]:
        recs = []
        if risk == RISK_HIGH:
            recs.append("⚠️ 系统处于高风险状态，建议立即启动本体论漂移或架构重构")
            recs.append("   运行: drift 或 iterate 跳出当前思路")
        elif risk == RISK_MID:
            recs.append("🔶 系统存在中等风险，建议增加多分支推演")
        else:
            recs.append("✅ 系统状态健康，可继续当前迭代节奏")

        for iv in interventions:
            if iv.action == "FORCE_ONTOLOGICAL_DRIFT":
                recs.append(f"  ↳ 执行本体论漂移：从 '{iv.details.get('file', '?')}' 切换到完全不相关的模块")
            elif iv.action == "FORCE_ARCHITECTURE_REWRITE":
                recs.append(f"  ↳ 强制重写：针对 '{iv.details.get('file', '?')}' 的反复错误，考虑换一种实现范式")
            elif iv.action == "EXTRA_DEDUCTION_POWER":
                recs.append(f"  ↳ 追加算力：增加 2-3 轮反例推演后再下结论")
            elif iv.action == "MULTI_BRANCH_COMPARE":
                recs.append(f"  ↳ 多分支对照：同时生成正反两种方案，比较后择优")
            elif iv.action == "ANTITHETICAL_REASONING":
                recs.append(f"  ↳ 反体系推理：假设相反前提，推演会得到什么结论")
            elif iv.action == "ELEVATE_EMERGENCE_TO_BRANCH":
                recs.append(f"  ↳ 🎲 随机涌现突破性候选 ({iv.details.get('breakthrough_count')}条)，运行 seo-eval 复核后纳入 branch")
            elif iv.action == "RUN_SEO_EVAL":
                recs.append(f"  ↳ 🎲 SEO 归档待评估 ({iv.details.get('pending_count')}条)，运行 seo-eval 执行后置比对")
        return recs

    def _identify_blind_spots(self) -> List[str]:
        """识别当前系统的认知盲区"""
        blind_spots = []
        if not self.history:
            blind_spots.append("尚无迭代历史，架构盲区未知")
            return blind_spots

        files_touched = Counter()
        for h in self.history:
            for d in h.get("applied_details", []):
                files_touched[d.get("file", "")] += 1

        total_files = len(files_touched)
        if total_files <= 2:
            blind_spots.append(f"迭代集中于 {total_files} 个文件，缺乏多样性")
        if total_files > 0 and total_files < 5:
            blind_spots.append("模型可能只擅长修改已知文件，遇到新文件会退化")

        tasks_seen = set(h.get("objective", "") for h in self.history if h.get("objective"))
        if len(tasks_seen) <= 2:
            blind_spots.append("目标多样性不足，容易过拟合单一任务类型")

        return blind_spots

    def _persist_state(self, report: MetaReport):
        state = {
            "last_run": report.generated_at,
            "score": report.score,
            "risk_level": report.risk_level,
            "findings": report.findings,
            "intervention_history": [
                {
                    "code": iv.code,
                    "action": iv.action,
                    "timestamp": datetime.now().isoformat(),
                    "result": "pending",
                }
                for iv in self.interventions
            ],
        }
        try:
            META_STATE_FILE.parent.mkdir(exist_ok=True)
            with open(META_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ========== 对外接口 ==========

    def suggest_prompt_modifier(self) -> Dict[str, Any]:
        """根据巡检结果，生成下一轮迭代的 prompt 修改建议"""
        if not self.interventions:
            return {"add_to_prompt": "", "subtract_from_prompt": "", "risk": RISK_LOW}

        modifiers = []
        subtract = []
        risk = self._determine_overall_risk(self._compute_health_score())

        for iv in self.interventions:
            if iv.code == "STAGNATION_PATH_COLLAPSE":
                modifiers.append(
                    f"⚠️ 警告：最近迭代过度集中于 {iv.details.get('file')}。"
                    f"请跳出这个文件，考虑更根本的架构变化。"
                )
            elif iv.code == "REPEATED_FAILURE_PATTERN":
                modifiers.append(
                    f"⚠️ 警告：{iv.details.get('file')} 的修复反复失败。"
                    f"不要继续尝试局部修补，请重新审视问题的根源。"
                )
            elif iv.code == "CONCEPTUAL_TRAP":
                modifiers.append(
                    "⚠️ 警告：你的修改方案高度收敛于常规思维模式。"
                    "请尝试从完全不同的角度切入（例如：如果反过来会怎样？如果完全不修改会怎样？）"
                )

        return {
            "add_to_prompt": "\n".join(modifiers),
            "subtract_from_prompt": "; ".join(subtract),
            "risk": risk,
        }

    def print_status(self, feedback=None):
        """打印 Meta-Drive 状态报告"""
        report = self.run_full_inspection()
        print()
        print("=" * 50)
        print("  🛰️ Meta-Drive 巡检报告")
        print("=" * 50)
        print(f"  健康分: {report.score:.0f}/100")
        print(f"  风险等级: {report.risk_level}")
        print(f"  生成时间: {report.generated_at[:19].replace('T', ' ')}")

        if report.findings:
            print(f"\n  🚨 发现 ({len(report.findings)}):")
            for f in report.findings:
                icon = "🔴" if f["risk"] == RISK_HIGH else "🟡" if f["risk"] == RISK_MID else "🟢"
                print(f"    {icon} [{f['check']}] {f['message']}")
        else:
            print("\n  ✅ 未检测到异常")

        if report.blind_spots:
            print(f"\n  🕳️ 认知盲区 ({len(report.blind_spots)}):")
            for bs in report.blind_spots:
                print(f"    - {bs}")

        if report.recommendations:
            print(f"\n  💡 建议:")
            for r in report.recommendations:
                print(f"    {r}")

        # 静默窗口状态
        sil = self.get_silence_status()
        eff_prob = self.get_effective_silence_prob()
        base = self.SILENCE_PROBABILITY
        boost = self._dynamic_silence_boost
        boost_tag = ""
        if boost > 0:
            boost_tag = f" ↑+{boost:.0%} (drift)"
        elif boost < 0:
            boost_tag = f" ↓{boost:.0%} (gpu-cool)"
        print(f"\n  🎲 静默窗口: 关闭（触发概率 {eff_prob:.0%}{boost_tag}，基础 {base:.0%}）")
        if sil["active"]:
            print(f"     当前活跃中: 剩余 {sil['remaining_sec']}s")
        else:
            last = sil.get("last_silence_ago_sec")
            if last is not None:
                print(f"     上次触发距今: {last // 60}min")

        print("=" * 50)
