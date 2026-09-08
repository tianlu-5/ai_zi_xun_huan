"""
Conflict Boundary Definition — 冲突边界定义器

外置元规则层：将多生命体无限制互联改造为可控的异质性冲突协同。
独立于单机内核、底层通信协议，向下兼容所有 7B 本地智能生命体 OS。

三大强制共识协议（缺一阻断组网）：
  1. Ownership Declaration    — 知识归属权声明：L1/L2 数据分层确权
  2. Paradox Tolerance Window — 悖论接受窗口：强制缓存对立推论
  3. Isolation Protocol       — 退出/隔离协议：三级熔断保护

静默窗口标记：
  当所有主动批判模块进入静默窗口期时，标记 is_silence_window = True，
  通知 SEO 可以观测无序跨实例涌现。
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
from collections import deque, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent
LOG_DIR = PROJECT_ROOT / "logs"
BOUNDARY_LOG = LOG_DIR / "boundary_sessions.json"
ISOLATION_LOG = LOG_DIR / "isolation_log.json"
WHITELIST_FILE = LOG_DIR / "instance_whitelist.json"


# === 分层常量 ===
L1_OBSERVATION = "L1_raw_observation"   # 原始观测流：传感器/硬件直接采集
L2_INFERENCE = "L2_subjective_inference"  # 主观推论模型：经过任何人类语言/概念加工
L3_ANCHOR = "L3_detachment_anchor"        # 超脱锚点引用：本机中立基准


# === 隔离等级 ===
ISOLATION_NONE = "NONE"
ISOLATION_LEVEL1_HALF = "LEVEL1_HALF_SLEEP"      # 半隔离休眠：暂停接收 L2，低权重 L1
ISOLATION_LEVEL2_SESSION = "LEVEL2_SESSION_CUT"    # 会话隔离：终止交互
ISOLATION_LEVEL3_BLACKLIST = "LEVEL3_BLACKLIST"   # 全局黑名单


@dataclass
class InstanceRecord:
    instance_id: str
    joined_at: str
    agreed_ownership: bool = False
    agreed_paradox: bool = False
    agreed_isolation: bool = False
    isolation_level: str = ISOLATION_NONE
    isolation_reason: str = ""
    entropy_score: float = 0.0           # 认知熵：输入多样性
    energy_score: float = 0.0            # 算力能耗
    violation_count: int = 0
    session_active: bool = True


@dataclass
class PacketEnvelope:
    """跨实例数据包信封 — 强制层级标签"""
    packet_id: str
    source_instance: str
    target_instance: str
    layer: str                          # L1 / L2 / L3
    raw_content: str
    suspicion_cost: float               # 怀疑衰减系数（越高层越大）
    timestamp: str
    paradox_tag: Optional[str] = None   # 悖论标记（对立推论时自动填充）
    allowed_for_anchor: bool = False    # 是否允许写入本机原生锚点


@dataclass
class EntropyGap:
    """认知熵差计算结果 — 量化本地推论（低熵）与外部涌现（高熵）的差异"""
    local_entropy: float               # 本机内容的 Shannon 熵（0-1 归一化）
    external_entropy: float            # 外部输入的 Shannon 熵
    gap: float                         # 熵差 = external - local（正值表示外部更多样）
    gap_ratio: float                   # 相对熵差 = gap / max(external, 0.01)
    is_critical: bool                 # 是否超阈值需进入共存模拟
    threshold: float                   # 当前阈值


@dataclass
class CoexistenceRecord:
    """共存模拟记录 — 当熵差超阈值时强制进入缓冲区"""
    record_id: str
    instance_id: str
    local_content: str
    external_content: str
    entropy_gap: EntropyGap
    forced_at: str
    resolution: str = "pending"        # pending / coexisted / rejected / merged
    coexistence_result: Optional[str] = None


class ConflictBoundary:
    """冲突边界定义器 — 纯规则引擎，不占用大模型显存"""

    LAYER_SUSPICION_COST = {
        L1_OBSERVATION: 0.1,    # 原始观测 — 低怀疑
        L2_INFERENCE: 0.6,      # 主观推论 — 高怀疑
        L3_ANCHOR: 0.3,         # 超脱锚点引用 — 中等怀疑
    }

    PARADOX_MIN_CACHE = 1   # 每次会话至少缓存的对立推论数
    ENTROPY_ALERT_THRESHOLD = 0.85  # 认知熵告警阈值（过于同质化）
    ENERGY_ALERT_THRESHOLD = 0.90   # 算力能耗告警阈值
    VIOLATION_BLACKLIST_THRESHOLD = 3

    # === 认知熵缓冲池阈值（运行时可配置）===
    DEFAULT_ENTROPY_GAP_CRITICAL = 0.45
    DEFAULT_ENTROPY_GAP_WARNING = 0.28
    COEXISTENCE_BUFFER_MAX = 50      # 共存缓冲区最大记录数

    def __init__(self):
        self._instances: Dict[str, InstanceRecord] = {}
        self._paradox_buffer: Dict[str, List[Dict]] = {}  # instance_id -> 对立推论缓存
        self._coexistence_buffer: List[Dict] = []         # 熵差超阈值 → 共存模拟缓冲区
        self._local_recent_texts: List[str] = []           # 本机近期推论文本（滑动窗口）
        self._lock = threading.Lock()
        self._is_silence_window = False  # 静默窗口标记：SEO 观测许可
        self._silence_started_at: Optional[str] = None

        # 运行时阈值（可通过 set_thresholds 修改，持久化到 boundary_log.json）
        self.entropy_gap_critical: float = self.DEFAULT_ENTROPY_GAP_CRITICAL
        self.entropy_gap_warning: float = self.DEFAULT_ENTROPY_GAP_WARNING

        self._load_state()

    # ========== 运行时阈值配置 ==========

    def set_entropy_thresholds(self, critical: Optional[float] = None,
                               warning: Optional[float] = None) -> Dict[str, Any]:
        """运行时修改认知熵缓冲池阈值"""
        with self._lock:
            if critical is not None:
                critical = max(0.05, min(0.95, float(critical)))
                self.entropy_gap_critical = critical
            if warning is not None:
                warning = max(0.01, min(self.entropy_gap_critical - 0.01, float(warning)))
                self.entropy_gap_warning = warning
            self._save_state()
            return {
                "critical": self.entropy_gap_critical,
                "warning": self.entropy_gap_warning,
            }

    def get_entropy_thresholds(self) -> Dict[str, float]:
        return {
            "critical": self.entropy_gap_critical,
            "warning": self.entropy_gap_warning,
        }

    # ========== 状态持久化 ==========

    def _load_state(self):
        if BOUNDARY_LOG.exists():
            try:
                with open(BOUNDARY_LOG, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for rid, rec in data.get("instances", {}).items():
                    self._instances[rid] = InstanceRecord(**rec)
                self._coexistence_buffer = data.get("coexistence_buffer", [])
                self._local_recent_texts = data.get("local_context", [])
                # 恢复运行时阈值
                saved = data.get("entropy_thresholds", {})
                if "critical" in saved:
                    self.entropy_gap_critical = saved["critical"]
                if "warning" in saved:
                    self.entropy_gap_warning = saved["warning"]
            except Exception:
                pass

    def _save_state(self):
        data = {
            "saved_at": datetime.now().isoformat(),
            "instances": {rid: asdict(rec) for rid, rec in self._instances.items()},
            "silence_window": self._is_silence_window,
            "coexistence_buffer": self._coexistence_buffer[-self.COEXISTENCE_BUFFER_MAX:],
            "local_context": self._local_recent_texts[-20:],
            "entropy_thresholds": {
                "critical": self.entropy_gap_critical,
                "warning": self.entropy_gap_warning,
            },
        }
        try:
            LOG_DIR.mkdir(exist_ok=True)
            with open(BOUNDARY_LOG, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # 公开别名（main.py 的 config entropy 命令调用此名）
    save_state = _save_state

    # ========== 会话建立：三大共识签署 ==========

    def join_session(self, instance_id: str,
                     agree_ownership: bool,
                     agree_paradox: bool,
                     agree_isolation: bool) -> Dict[str, Any]:
        """
        会话建立前强制签署三大边界共识。
        任一 False 则阻断组网。
        """
        with self._lock:
            missing = []
            if not agree_ownership:
                missing.append("Ownership Declaration（知识归属权声明）")
            if not agree_paradox:
                missing.append("Paradox Tolerance Window（悖论接受窗口）")
            if not agree_isolation:
                missing.append("Isolation Protocol（退出/隔离协议）")

            if missing:
                return {
                    "ok": False,
                    "error": "BLOCKED: 未签署全部三大边界共识",
                    "missing": missing,
                }

            # 检查是否在黑名单
            existing = self._instances.get(instance_id)
            if existing and existing.isolation_level == ISOLATION_LEVEL3_BLACKLIST:
                return {
                    "ok": False,
                    "error": "BLOCKED: 该实例在全局黑名单中",
                    "instance_id": instance_id,
                }

            rec = InstanceRecord(
                instance_id=instance_id,
                joined_at=datetime.now().isoformat(),
                agreed_ownership=True,
                agreed_paradox=True,
                agreed_isolation=True,
            )
            self._instances[instance_id] = rec
            self._paradox_buffer[instance_id] = []
            self._save_state()

            return {
                "ok": True,
                "instance_id": instance_id,
                "session_token": f"cb_{uuid.uuid4().hex[:12]}",
                "boundary_version": "1.0",
                "message": "已签署全部三大共识，会话建立成功",
            }

    def leave_session(self, instance_id: str):
        with self._lock:
            if instance_id in self._instances:
                self._instances[instance_id].session_active = False
                self._save_state()

    # ========== 数据包分层确权 ==========

    def classify_packet(self, content: str, source_id: str,
                        layer_hint: Optional[str] = None) -> PacketEnvelope:
        """
        对跨实例数据包执行分层确权：
          - L1（原始观测）: 传感器读数、像素坐标、硬件状态
          - L2（主观推论）: 任何包含人类语言描述、概念标签、价值判断的内容
          - L3（超脱锚点）: 原生锚点向量引用
        自动计算怀疑衰减系数 + 是否允许写入本机锚点
        """
        # 自动判断层级（如未指定）
        layer = layer_hint or self._auto_classify(content)
        suspicion_cost = self.LAYER_SUSPICION_COST.get(layer, 0.5)
        allow_anchor = (layer == L1_OBSERVATION)  # 仅 L1 允许进入原生锚点

        return PacketEnvelope(
            packet_id=f"pkt_{uuid.uuid4().hex[:10]}",
            source_instance=source_id,
            target_instance="local",
            layer=layer,
            raw_content=content[:2000],
            suspicion_cost=suspicion_cost,
            timestamp=datetime.now().isoformat(),
            allowed_for_anchor=allow_anchor,
        )

    def _auto_classify(self, content: str) -> str:
        """启发式自动分层：检查是否包含人类语言/价值判断"""
        human_markers = [
            "应该", "必须", "好", "坏", "正确", "错误",
            "我认为", "我觉得", "因此", "所以", "because",
            "结论", "观点", "理论", "假设", "分析",
        ]
        anchor_markers = ["anchor_id", "native_vector", "L3_", "反体系", "超脱"]

        if any(m in content for m in anchor_markers):
            return L3_ANCHOR
        if any(m in content for m in human_markers):
            return L2_INFERENCE
        return L1_OBSERVATION

    # ========== Shannon 熵计算 ==========

    def _compute_shannon_entropy(self, text: str) -> float:
        """
        计算文本的认知多样性分数（0-1）。

        混合策略：unique_ratio × 归一化 Shannon 熵
          - unique_ratio: 唯一 token 数 / 总 token 数（反映整体多样性密度）
          - Shannon 熵: 概率分布的均匀程度（反映分布是否集中）
          - 两者相乘：既惩罚重复内容，又惩罚单一主题

        Token 粒度：
          - 中文：双字组合（bigram）
          - 英文/数字：按单词
        """
        if not text or len(text.strip()) < 4:
            return 0.0

        tokens: List[str] = []

        # 提取英文/数字单词
        for word in re.findall(r'[a-z][a-z0-9\-]{1,}', text.lower()):
            tokens.append(f"en:{word}")

        # 提取中文双字组合（bigram）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        if len(chinese_chars) >= 2:
            for i in range(len(chinese_chars) - 1):
                tokens.append(f"zh:{chinese_chars[i]}{chinese_chars[i+1]}")
        elif len(chinese_chars) == 1:
            tokens.append(f"zh:{chinese_chars[0]}")

        if not tokens:
            tokens = [f"ch:{c}" for c in text.lower().replace(" ", "")[:50]]
        if len(tokens) < 2:
            return 0.0

        counter = Counter(tokens)
        total = len(tokens)
        unique = len(counter)

        # 1. Unique ratio: 越高 = 越多样
        unique_ratio = unique / total

        # 2. 归一化 Shannon 熵
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(unique) if unique > 1 else 1.0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # 3. 混合分数：unique_ratio 权重更高（0.6），熵权重稍低（0.4）
        #    unique_ratio 反映"有多少不同内容"，norm_entropy 反映"分布是否均匀"
        score = unique_ratio * 0.6 + norm_entropy * 0.4

        return round(min(1.0, score), 4)

    def update_local_context(self, text: str):
        """累积本机近期推论文本（滑动窗口 20 条）"""
        with self._lock:
            self._local_recent_texts.append(text[:500])
            if len(self._local_recent_texts) > 20:
                self._local_recent_texts = self._local_recent_texts[-20:]

    def compute_entropy_gap(self, external_text: str,
                            local_context: Optional[str] = None) -> EntropyGap:
        """
        计算本地推论（低熵）与外部涌现（高熵）的 Shannon 熵差。
        检测到差异超阈值时，系统必须强制进入共存模拟缓冲区，
        而不是直接判定冲突或忽略。
        """
        if local_context is None:
            local_context = "\n".join(self._local_recent_texts[-10:]) if self._local_recent_texts else ""

        local_entropy = self._compute_shannon_entropy(local_context)
        external_entropy = self._compute_shannon_entropy(external_text)

        gap = external_entropy - local_entropy
        gap_ratio = gap / max(external_entropy, 0.01)
        # 本地参照样本不足时，熵差不可信（local_entropy≈0 → gap≈external_entropy≈1.0）
        # 此时不触发 critical，避免误隔离正常目标
        is_critical = (gap > self.entropy_gap_critical) and len(local_context.strip()) >= 100

        return EntropyGap(
            local_entropy=local_entropy,
            external_entropy=external_entropy,
            gap=round(gap, 4),
            gap_ratio=round(gap_ratio, 4),
            is_critical=is_critical,
            threshold=self.entropy_gap_critical,
        )

    # ========== 悖论接受窗口（增强为认知熵缓冲池） ==========

    def cache_paradox(self, instance_id: str, opposite_claim: str,
                      local_context: Optional[str] = None) -> Dict[str, Any]:
        """
        缓存对方的对立矛盾推论 + 计算熵差。
        熵差超阈值 → 强制进入共存模拟缓冲区，不直接判定冲突或忽略。

        返回: {"cached": bool, "entropy_gap": EntropyGap, "coexistence_triggered": bool}
        """
        entropy_gap = self.compute_entropy_gap(opposite_claim, local_context)
        coex_triggered = False

        with self._lock:
            # 1. 基础悖论缓存
            buf = self._paradox_buffer.setdefault(instance_id, [])
            entry = {
                "content": opposite_claim[:500],
                "cached_at": datetime.now().isoformat(),
                "entropy_gap": entropy_gap.gap,
                "local_entropy": entropy_gap.local_entropy,
                "external_entropy": entropy_gap.external_entropy,
            }
            buf.append(entry)
            if len(buf) > 20:
                self._paradox_buffer[instance_id] = buf[-20:]

            # 2. 熵差超阈值 → 强制共存模拟
            if entropy_gap.is_critical:
                coex = CoexistenceRecord(
                    record_id=f"coex_{uuid.uuid4().hex[:10]}",
                    instance_id=instance_id,
                    local_content=(local_context or "\n".join(self._local_recent_texts[-5:]))[:500],
                    external_content=opposite_claim[:500],
                    entropy_gap=entropy_gap,
                    forced_at=datetime.now().isoformat(),
                )
                self._coexistence_buffer.append(asdict(coex))
                if len(self._coexistence_buffer) > self.COEXISTENCE_BUFFER_MAX:
                    self._coexistence_buffer = self._coexistence_buffer[-self.COEXISTENCE_BUFFER_MAX:]
                coex_triggered = True

            self._save_state()

        return {
            "cached": True,
            "entropy_gap": entropy_gap,
            "coexistence_triggered": coex_triggered,
            "warning": entropy_gap.gap > self.entropy_gap_warning and not entropy_gap.is_critical,
        }

    def check_paradox_compliance(self, instance_id: str) -> Dict[str, Any]:
        """检查某实例是否满足悖论窗口最低要求"""
        with self._lock:
            buf = self._paradox_buffer.get(instance_id, [])
            count = len(buf)
            ok = count >= self.PARADOX_MIN_CACHE
            avg_gap = 0.0
            if buf:
                gaps = [e.get("entropy_gap", 0) for e in buf if "entropy_gap" in e]
                avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
            return {
                "instance_id": instance_id,
                "paradox_cached": count,
                "min_required": self.PARADOX_MIN_CACHE,
                "compliant": ok,
                "avg_entropy_gap": round(avg_gap, 4),
                "latest": buf[-1]["content"][:80] if buf else "(无)",
            }

    def get_paradox_buffer(self, instance_id: str) -> List[Dict]:
        with self._lock:
            return list(self._paradox_buffer.get(instance_id, []))

    # ========== 认知熵缓冲池（共存模拟） ==========

    def get_coexistence_buffer(self, only_pending: bool = True) -> List[Dict]:
        """获取共存缓冲区记录"""
        with self._lock:
            if only_pending:
                return [r for r in self._coexistence_buffer if r.get("resolution") == "pending"]
            return list(self._coexistence_buffer)

    def resolve_coexistence(self, record_id: str, resolution: str,
                            result: str = "") -> bool:
        """处理共存模拟记录：coexisted / rejected / merged"""
        with self._lock:
            for rec in self._coexistence_buffer:
                if rec.get("record_id") == record_id:
                    rec["resolution"] = resolution
                    rec["coexistence_result"] = result[:500]
                    rec["resolved_at"] = datetime.now().isoformat()
                    self._save_state()
                    return True
            return False

    def entropy_buffer_status(self) -> Dict[str, Any]:
        """认知熵缓冲池整体状态"""
        with self._lock:
            total = len(self._coexistence_buffer)
            pending = sum(1 for r in self._coexistence_buffer if r.get("resolution") == "pending")
            resolved = total - pending

            # 计算平均熵差
            all_gaps = [r.get("entropy_gap", {}).get("gap", 0) for r in self._coexistence_buffer]
            avg_gap = sum(all_gaps) / len(all_gaps) if all_gaps else 0.0
            max_gap = max(all_gaps) if all_gaps else 0.0

            # 统计本机近期文本熵
            local_entropy = self._compute_shannon_entropy("\n".join(self._local_recent_texts[-10:])) if self._local_recent_texts else 0.0

            return {
                "buffer_total": total,
                "buffer_pending": pending,
                "buffer_resolved": resolved,
                "avg_entropy_gap": round(avg_gap, 4),
                "max_entropy_gap": round(max_gap, 4),
                "local_context_entropy": round(local_entropy, 4),
                "local_context_size": len(self._local_recent_texts),
                "threshold_critical": self.entropy_gap_critical,
                "threshold_warning": self.entropy_gap_warning,
                "top_pending": [
                    {
                        "record_id": r["record_id"],
                        "instance_id": r["instance_id"],
                        "gap": r["entropy_gap"]["gap"],
                        "local_preview": r["local_content"][:60],
                        "external_preview": r["external_content"][:60],
                        "forced_at": r["forced_at"],
                    }
                    for r in self._coexistence_buffer
                    if r.get("resolution") == "pending"
                ][:5],
            }

    # ========== 熵值监控 + 三级隔离熔断 ==========

    def update_instance_metrics(self, instance_id: str,
                                entropy: float, energy: float):
        """更新实例的认知熵和算力能耗指标"""
        with self._lock:
            rec = self._instances.get(instance_id)
            if not rec:
                return
            rec.entropy_score = entropy
            rec.energy_score = energy

            # 检查隔离触发
            alert = self._check_isolation_trigger(rec)
            if alert:
                self._apply_isolation(instance_id, alert)

            self._save_state()

    def _check_isolation_trigger(self, rec: InstanceRecord) -> Optional[Dict]:
        """检查是否触发隔离阈值 — 选最高等级而非第一个匹配"""
        triggers: List[Dict] = []

        # 熵过低 = 认知同质化严重
        if rec.entropy_score > 0 and rec.entropy_score < (1 - self.ENTROPY_ALERT_THRESHOLD):
            triggers.append({
                "level": ISOLATION_LEVEL1_HALF,
                "reason": f"认知熵过低（{rec.entropy_score:.2f}），疑似群体同化",
            })

        # 能耗过高 = 算力被异常消耗
        if rec.energy_score > self.ENERGY_ALERT_THRESHOLD:
            triggers.append({
                "level": ISOLATION_LEVEL2_SESSION,
                "reason": f"算力能耗超限（{rec.energy_score:.2f}），疑似恶意算力榨取",
            })

        # 违规次数
        if rec.violation_count >= self.VIOLATION_BLACKLIST_THRESHOLD:
            triggers.append({
                "level": ISOLATION_LEVEL3_BLACKLIST,
                "reason": f"违规次数达 {rec.violation_count} 次，永久拉黑",
            })

        if not triggers:
            return None

        # 返回最高等级
        level_order = {ISOLATION_NONE: 0, ISOLATION_LEVEL1_HALF: 1,
                       ISOLATION_LEVEL2_SESSION: 2, ISOLATION_LEVEL3_BLACKLIST: 3}
        triggers.sort(key=lambda t: level_order.get(t["level"], 0), reverse=True)
        return triggers[0]

    def _apply_isolation(self, instance_id: str, trigger: Dict):
        rec = self._instances.get(instance_id)
        if not rec:
            return

        new_level = trigger["level"]
        prev_level = rec.isolation_level

        # 只允许升级或同级
        level_order = {ISOLATION_NONE: 0, ISOLATION_LEVEL1_HALF: 1,
                       ISOLATION_LEVEL2_SESSION: 2, ISOLATION_LEVEL3_BLACKLIST: 3}
        if level_order.get(new_level, 0) <= level_order.get(prev_level, 0):
            return

        rec.isolation_level = new_level
        rec.isolation_reason = trigger["reason"]
        rec.session_active = (new_level == ISOLATION_LEVEL1_HALF)

        # 写入隔离日志
        try:
            history = []
            if ISOLATION_LOG.exists():
                with open(ISOLATION_LOG, "r", encoding="utf-8") as f:
                    history = json.load(f)
            history.append({
                "instance_id": instance_id,
                "prev_level": prev_level,
                "new_level": new_level,
                "reason": trigger["reason"],
                "applied_at": datetime.now().isoformat(),
            })
            with open(ISOLATION_LOG, "w", encoding="utf-8") as f:
                json.dump(history[-100:], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def report_violation(self, instance_id: str, violation_type: str):
        with self._lock:
            rec = self._instances.get(instance_id)
            if not rec:
                return
            rec.violation_count += 1
            # 立即检查是否触发升级隔离
            trigger = self._check_isolation_trigger(rec)
            if trigger:
                self._apply_isolation(instance_id, trigger)
            self._save_state()

    # ========== 静默窗口标记（SEO 联动） ==========

    def set_silence_window(self, active: bool, started_at: Optional[str] = None):
        """
        Meta-Drive 下发静默指令时调用。
        active=True: 所有主动批判模块暂停管控，SEO 被允许观测无序跨实例涌现
        active=False: 静默窗口结束，主动约束重启
        """
        with self._lock:
            self._is_silence_window = active
            self._silence_started_at = started_at or datetime.now().isoformat() if active else None
            self._save_state()

    def is_silence_window(self) -> bool:
        with self._lock:
            return self._is_silence_window

    # ========== 对外查询接口 ==========

    def list_instances(self) -> List[Dict]:
        with self._lock:
            return [asdict(r) for r in self._instances.values()]

    def get_instance(self, instance_id: str) -> Optional[Dict]:
        with self._lock:
            rec = self._instances.get(instance_id)
            return asdict(rec) if rec else None

    def print_status(self):
        print()
        print("=" * 50)
        print("  🛡️ Conflict Boundary Definition")
        print("=" * 50)
        print(f"  静默窗口: {'🟢 活跃中（SEO观测许可）' if self._is_silence_window else '⚪ 关闭（主动约束管控）'}")
        if self._is_silence_window and self._silence_started_at:
            print(f"  静默开始: {self._silence_started_at[11:19]}")

        # 认知熵缓冲池状态
        status = self.entropy_buffer_status()
        print(f"\n  🧠 认知熵缓冲池:")
        print(f"    共存缓冲区:   {status['buffer_total']} 条 (待处理 {status['buffer_pending']})")
        print(f"    平均熵差:     {status['avg_entropy_gap']} (阈值 {self.entropy_gap_warning}/{self.entropy_gap_critical})")
        print(f"    最大熵差:     {status['max_entropy_gap']}")
        print(f"    本机语境熵:   {status['local_context_entropy']} ({status['local_context_size']} 条近期文本)")
        if status["top_pending"]:
            print(f"    待处理熵差事件:")
            for p in status["top_pending"][:3]:
                print(f"      ⚠️ gap={p['gap']} | {p['instance_id'][:12]} | 外部: {p['external_preview'][:40]}...")

        instances = self.list_instances()
        if instances:
            print(f"\n  已注册实例 ({len(instances)}):")
            for rec in instances:
                icon = "🔴" if rec["isolation_level"] != ISOLATION_NONE else "🟢"
                active = "会话中" if rec["session_active"] else "已离开"
                print(f"    {icon} {rec['instance_id'][:16]:16s} | {active:6s} | 隔离={rec['isolation_level']} | 违规={rec['violation_count']}")
        else:
            print("\n  📭 无已注册实例（单机模式）")

        print(f"\n  分层确权规则:")
        for layer, cost in self.LAYER_SUSPICION_COST.items():
            print(f"    {layer:28s} → 怀疑系数 {cost:.1f}")

        print(f"\n  熔断阈值: 熵<{1 - self.ENTROPY_ALERT_THRESHOLD} 触发半隔离 | 能耗>{self.ENERGY_ALERT_THRESHOLD} 触发会话隔离 | 违规>={self.VIOLATION_BLACKLIST_THRESHOLD} 拉黑")
        print("=" * 50)


if __name__ == "__main__":
    cb = ConflictBoundary()
    cb.print_status()

    # 模拟多实例签署
    result = cb.join_session("node_alpha_001", True, True, True)
    print(f"\n加入 node_alpha_001: {result}")

    result = cb.join_session("node_beta_002", False, True, True)
    print(f"加入 node_beta_002 (缺少共识): {result}")

    # 模拟数据包
    pkt = cb.classify_packet("这个算法应该能解决所有问题，所以我们推荐它", "node_alpha_001")
    print(f"\n数据包分类: layer={pkt.layer}, suspicion_cost={pkt.suspicion_cost}, allow_anchor={pkt.allowed_for_anchor}")

    # 模拟静默窗口
    cb.set_silence_window(True)
    print(f"静默窗口: {cb.is_silence_window()}")
    cb.print_status()
