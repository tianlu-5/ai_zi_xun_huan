"""
L3 Detachment Anchor 超脱锚点生成器

解决本地 7B 重构后重复复刻人类认知的痛点，搭建两层中立基准：

1. 原生观测锚点（永久静态基准）
   - 从面具识别器输出的无滤镜裸观测向量抽取纯量化关联
   - 本地持久化存储，不参与 7B 文本翻译、不受模型微调修改

2. 反体系动态锚点（每次重构同步生成）
   - 基于 L2 代价挖掘得到的理论短板
   - 调用 7B 生成完全对立逻辑体系作为校验标尺

3. 重构准入校验规则
   - 新模型分支必须同时通过两套锚点向量相似度校验
   - 仅当「遮蔽程度更低、总认知代价更小」才允许保留

硬件适配：faiss-cpu 本地向量库（纯 CPU，~100MB 内存）。
若 faiss 不可用，降级为纯 Python cosine similarity。
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent
VECTOR_DIR = PROJECT_ROOT / "vectors"
NATIVE_ANCHORS_FILE = VECTOR_DIR / "native_anchors.json"
ANTI_ANCHORS_FILE = VECTOR_DIR / "anti_anchors.json"


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """纯 Python cosine similarity（faiss 不可用时的降级方案）"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorStore:
    """本地向量存储（faiss + JSON fallback）"""

    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.use_faiss = False
        try:
            import faiss
            self.use_faiss = True
            self.faiss = faiss
        except ImportError:
            self.faiss = None
        self.vectors: List[List[float]] = []
        self.metadata: List[Dict[str, Any]] = []
        self.index = None

    def add(self, vector: List[float], meta: Dict[str, Any]):
        # ── PLE 只读保护：daemon 自迭代期间，原生观测锚点的底层嵌入层禁止写入 ──
        if getattr(self, "_daemon_readonly", False):
            return False
        self.vectors.append(vector)
        self.metadata.append(meta)
        if self.use_faiss and self.index is not None:
            import numpy as np
            v = np.array([vector], dtype=np.float32)
            self.index.add(v)
        return True

    def build_index(self):
        if self.use_faiss and self.vectors:
            import numpy as np
            self.index = self.faiss.IndexFlatIP(self.dim)
            arr = np.array(self.vectors, dtype=np.float32)
            self.index.add(arr)

    def search(self, query: List[float], k: int = 5) -> List[Tuple[int, float, Dict]]:
        """返回 [(index, similarity, metadata), ...]"""
        if self.use_faiss and self.index is not None:
            import numpy as np
            q = np.array([query], dtype=np.float32)
            scores, indices = self.index.search(q, min(k, len(self.vectors)))
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0 and idx < len(self.vectors):
                    results.append((int(idx), float(score), self.metadata[idx]))
            return results

        # Fallback: pure Python cosine
        scored = []
        for i, v in enumerate(self.vectors):
            sim = _cosine_sim(query, v)
            scored.append((i, sim, self.metadata[i]))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def nearest_similarity(self, query: List[float]) -> float:
        if not self.vectors:
            return 0.0
        results = self.search(query, k=1)
        return results[0][1] if results else 0.0

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True)
        data = {
            "dim": self.dim,
            "vectors": self.vectors,
            "metadata": self.metadata,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: Path):
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.dim = data.get("dim", 4096)
            self.vectors = data.get("vectors", [])
            self.metadata = data.get("metadata", [])
            self.build_index()
        except Exception:
            self.vectors = []
            self.metadata = []

    def __len__(self):
        return len(self.vectors)


@dataclass
class AnchorValidation:
    passed: bool
    native_similarity: float
    anti_similarity: float
    veil_reduction: float
    cost_reduction: float
    details: List[str]


@dataclass
class DriftAlert:
    """时间漂移检测结果"""
    detected: bool
    drift_score: float               # 0-1 偏离度，越高越严重
    expected_rate: float             # 历史变化率（锚点/天）
    current_rate: float              # 当前变化率
    window_days: int                 # 滑动窗口天数
    anomaly_keywords: List[str]      # 突增/突降的关键词
    recommendation: str


class TimeDriftDetector:
    """
    时间漂移检测器 — 不是检测数据本身对不对，
    而是检测当前采信的逻辑链条是否与历史记录（或物理定律）
    的预期变化率存在系统性偏离。

    核心思想：
      - 锚点库的累积速率、关键词分布、来源多样性应随时间
        呈现平稳的预期变化率（近似线性增长）；
      - 如果某段时间突然出现远超历史基线的结构性突变
        （例如：锚点注入速率暴增、全部来自单一来源、
        关键词高度集中），说明有外部力量在定向引导
        系统采信某条逻辑链条 → 标记为 temporal_drift_warning。

    轻量化实现：滑动窗口统计 + 卡方偏离度，不依赖 LLM。
    """

    WINDOW_SIZE_DAYS = 7
    RATE_SPIKE_THRESHOLD = 2.0   # 当前速率 vs 历史基线 超过此倍数为异常
    RATE_COLLAPSE_THRESHOLD = 0.3  # 当前速率 vs 历史基线 低于此倍数为异常
    KEYWORD_CONCENTRATION_THRESHOLD = 0.6  # 单一关键词占比超过此值为集中

    def __init__(self):
        self._anomaly_history: List[Dict] = []

    def analyze(self, anchor_metadata: List[Dict]) -> DriftAlert:
        """
        对锚点元数据执行时间漂移检测。
        anchor_metadata: detachment_anchor.native.metadata 列表
        """
        if len(anchor_metadata) < 3:
            return DriftAlert(
                detected=False, drift_score=0.0,
                expected_rate=0.0, current_rate=0.0,
                window_days=self.WINDOW_SIZE_DAYS,
                anomaly_keywords=[],
                recommendation="锚点库样本不足，跳过时间漂移检测",
            )

        # 1. 计算每个锚点的创建时间（从 created_at 解析秒级时间戳）
        timestamps: List[float] = []
        for m in anchor_metadata:
            ts = m.get("created_at")
            if ts:
                try:
                    timestamps.append(datetime.fromisoformat(ts).timestamp())
                except Exception:
                    pass
        if len(timestamps) < 3:
            return DriftAlert(
                detected=False, drift_score=0.0,
                expected_rate=0.0, current_rate=0.0,
                window_days=self.WINDOW_SIZE_DAYS,
                anomaly_keywords=[],
                recommendation="时间戳样本不足，跳过时间漂移检测",
            )
        timestamps.sort()

        # 2. 计算整体变化率（锚点总数 / 总天数）
        span_days = max(1, (timestamps[-1] - timestamps[0]) / 86400)
        overall_rate = len(timestamps) / span_days  # 锚点/天

        # 3. 滑动窗口计算最近 WINDOW_SIZE_DAYS 天的速率
        window_start = timestamps[-1] - self.WINDOW_SIZE_DAYS * 86400
        window_count = sum(1 for t in timestamps if t >= window_start)
        # 窗口内天数（至少 1）
        window_actual_days = max(1, (timestamps[-1] - max(window_start, timestamps[-2] if len(timestamps) > 1 else window_start)) / 86400 + 1)
        current_rate = window_count / min(self.WINDOW_SIZE_DAYS, window_actual_days)

        # 4. 历史基线：排除最近窗口后的早期速率（如果有足够数据）
        if len(timestamps) > window_count + 3:
            early_timestamps = [t for t in timestamps if t < window_start]
            if len(early_timestamps) >= 2:
                early_span = max(1, (early_timestamps[-1] - early_timestamps[0]) / 86400)
                baseline_rate = len(early_timestamps) / early_span
            else:
                baseline_rate = overall_rate
        else:
            # 样本少，用整体速率作为基线但提高容忍度
            baseline_rate = overall_rate * 0.7

        if baseline_rate == 0:
            baseline_rate = 0.01  # 避免除零

        # 5. 速率偏离度（取 spike 和 collapse 中更严重的那个）
        spike_ratio = current_rate / baseline_rate
        collapse_ratio = baseline_rate / max(current_rate, 0.01)

        rate_deviation = 0.0
        anomaly_type = ""
        if spike_ratio > self.RATE_SPIKE_THRESHOLD:
            rate_deviation = min(1.0, (spike_ratio - self.RATE_SPIKE_THRESHOLD) / 5.0 + 0.3)
            anomaly_type = "速率暴增"
        elif collapse_ratio > (1.0 / self.RATE_COLLAPSE_THRESHOLD):
            rate_deviation = min(1.0, (collapse_ratio - 1.0 / self.RATE_COLLAPSE_THRESHOLD) / 5.0 + 0.3)
            anomaly_type = "速率塌陷"

        # 6. 关键词集中度（检查 text_preview 的关键词分布）
        all_texts = [m.get("text_preview", "") for m in anchor_metadata if m.get("text_preview")]
        anomaly_keywords = []
        keyword_concentration = 0.0
        if all_texts:
            counter = Counter()
            for t in all_texts:
                words = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]+", t.lower())
                counter.update(words)
            total = sum(counter.values())
            if total > 0:
                top_word, top_count = counter.most_common(1)[0]
                keyword_concentration = top_count / total
                if keyword_concentration > self.KEYWORD_CONCENTRATION_THRESHOLD:
                    anomaly_keywords = [w for w, _ in counter.most_common(5)]

        # 7. 来源多样性（检查 source 字段的集中度）
        sources = Counter(m.get("source", "unknown") for m in anchor_metadata)
        source_total = sum(sources.values())
        source_top = sources.most_common(1)[0][1] / source_total if source_total else 0
        source_concentration = source_top  # 越高 = 越集中

        # 8. 综合漂移分数
        concentration_penalty = 0.0
        if keyword_concentration > self.KEYWORD_CONCENTRATION_THRESHOLD:
            concentration_penalty += (keyword_concentration - self.KEYWORD_CONCENTRATION_THRESHOLD) * 2.0
        if source_concentration > 0.7:
            concentration_penalty += (source_concentration - 0.7) * 1.5

        drift_score = min(1.0, rate_deviation * 0.5 + concentration_penalty * 0.5)
        detected = drift_score >= 0.3

        # 9. 生成建议
        if detected:
            if anomaly_type == "速率暴增":
                rec = (f"⚠️ 锚点注入速率暴增：近期 {window_count} 条 / {self.WINDOW_SIZE_DAYS}天 "
                       f"vs 历史基线 {baseline_rate:.1f} 条/天（{spike_ratio:.1f}倍）。"
                       f"可能存在外部定向叙事在密集灌入锚点库。"
                       f"关键词集中度 {keyword_concentration:.0%}，来源集中度 {source_concentration:.0%}。")
            elif anomaly_type == "速率塌陷":
                rec = (f"⚠️ 锚点注入速率塌陷：近期 {current_rate:.1f} 条/天 vs "
                       f"历史基线 {baseline_rate:.1f} 条/天。系统可能停止采集裸观测，"
                       f"开始依赖陈旧锚点推演。")
            else:
                rec = (f"⚠️ 关键词/来源集中度过高：关键词集中度 {keyword_concentration:.0%}，"
                       f"来源集中度 {source_concentration:.0%}。锚点库正在朝单一方向收敛，"
                       f"多样性正在流失。")
        else:
            rec = ("✅ 时间漂移检测通过：锚点累积速率平稳，关键词分布均匀，来源多样性正常。"
                   f"（当前 {current_rate:.1f} 条/天 vs 基线 {baseline_rate:.1f} 条/天，"
                   f"关键词集中度 {keyword_concentration:.0%}）")

        result = DriftAlert(
            detected=detected,
            drift_score=round(drift_score, 3),
            expected_rate=round(baseline_rate, 2),
            current_rate=round(current_rate, 2),
            window_days=self.WINDOW_SIZE_DAYS,
            anomaly_keywords=anomaly_keywords,
            recommendation=rec,
        )

        # 记录异常历史
        if detected:
            self._anomaly_history.append({
                "ts": datetime.now().isoformat(),
                "drift_score": drift_score,
                "alert_type": anomaly_type,
            })
            if len(self._anomaly_history) > 50:
                self._anomaly_history = self._anomaly_history[-50:]

        return result

    def check_time_drift(self, anchor_metadata: List[Dict]) -> DriftAlert:
        """公开接口：执行一次时间漂移检测"""
        return self.analyze(anchor_metadata)


class DetachmentAnchor:
    """超脱锚点生成器"""

    def __init__(self, ollama_client=None, desensitizer=None, min_weight: float = 0.35):
        self.client = ollama_client
        self.desensitizer = desensitizer
        self.min_weight = min_weight
        self.native = VectorStore()
        self.anti = VectorStore()
        self.drift_detector = TimeDriftDetector()
        self._last_drift_alert: Optional[DriftAlert] = None
        self._filter_log: List[Dict] = []
        self._load()

    def attach_desensitizer(self, desensitizer):
        """动态挂载脱敏屏障（运行时注入，避免循环依赖）"""
        self.desensitizer = desensitizer

    def _run_desensitization(self, text: str, source: str) -> Tuple[bool, float, Optional[Dict]]:
        """对外部素材执行脱敏衰减"""
        if self.desensitizer is None:
            return True, 1.0, None
        if source in ("native_seed", "theory_input", "system"):
            return True, 1.0, None
        try:
            result = self.desensitizer.desensitize_input(text, miner=None)
            passed = result["passed"] and result["weight"] >= self.min_weight
            return passed, result["weight"], result
        except Exception:
            return True, 1.0, None

    # ========== 内部私有脱敏日志 ==========

    def _log_filter(self, source: str, text: str, passed: bool, weight: float,
                    reason: str = ""):
        entry = {
            "ts": datetime.now().isoformat(),
            "source": source,
            "preview": text[:80],
            "passed": passed,
            "weight": weight,
            "reason": reason,
        }
        self._filter_log.append(entry)
        if len(self._filter_log) > 200:
            self._filter_log = self._filter_log[-200:]

    def get_filter_stats(self) -> Dict[str, Any]:
        if not self._filter_log:
            return {"total": 0, "blocked": 0, "pass_rate": 0.0}
        blocked = sum(1 for e in self._filter_log if not e["passed"])
        return {
            "total": len(self._filter_log),
            "blocked": blocked,
            "pass_rate": (len(self._filter_log) - blocked) / len(self._filter_log),
            "recent": self._filter_log[-5:],
        }

    def _load(self):
        self.native.load(NATIVE_ANCHORS_FILE)
        self.anti.load(ANTI_ANCHORS_FILE)

    def _save(self):
        VECTOR_DIR.mkdir(exist_ok=True)
        self.native.save(NATIVE_ANCHORS_FILE)
        self.anti.save(ANTI_ANCHORS_FILE)

    def _embed(self, text: str) -> Optional[List[float]]:
        """调用 Ollama embedding API"""
        import urllib.request
        try:
            body = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/embeddings",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            return data.get("embedding")
        except Exception as e:
            print(f"[Anchor] embedding failed: {e}")
            return None

    def _embed_or_zero(self, text: str) -> List[float]:
        vec = self._embed(text)
        if vec:
            if self.native.dim == 4096:
                self.native.dim = len(vec)
                self.anti.dim = len(vec)
            return vec
        return [0.0] * self.native.dim

    # ========== 原生观测锚点 ==========

    def seed_native_anchor(self, raw_observation: str, source: str = "manual") -> int:
        """从裸观测文本创建原生锚点（永久静态，不参与后续修改）"""
        passed, weight, detail = self._run_desensitization(raw_observation, source)
        if not passed:
            self._log_filter(source, raw_observation, False, weight,
                             reason=f"脱敏权重 {weight} < {self.min_weight}")
            print(f"  🚫 锚点被脱敏屏障拦截 (权重 {weight}, 来源 {source})")
            if detail and detail.get("narrative_bias"):
                for bias in detail["narrative_bias"][:3]:
                    print(f"     ↳ {bias}")
            return -1

        vec = self._embed_or_zero(raw_observation)
        meta = {
            "created_at": datetime.now().isoformat(),
            "type": "native",
            "source": source,
            "text_preview": raw_observation[:200],
            "desensitization_weight": weight,
        }
        self.native.add(vec, meta)
        self.native.build_index()
        self._save()
        self._log_filter(source, raw_observation, True, weight)
        return len(self.native)

    def seed_native_from_file(self, filepath: str) -> int:
        """从文件中提取关键句子作为原生锚点"""
        path = Path(filepath)
        if not path.exists():
            return 0
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return 0

        sentences = [s.strip() for s in text.split("\n") if s.strip() and len(s.strip()) > 20]
        count = 0
        for sent in sentences[:20]:
            self.seed_native_anchor(sent, source=f"file:{path.name}")
            count += 1
        return count

    # ========== 反体系锚点 ==========

    def generate_anti_anchor(self, original_theory: str, counter_theory: str) -> int:
        """基于原始理论和反事实分支生成反体系锚点"""
        combined = f"原始理论: {original_theory}\n\n反体系: {counter_theory}"
        vec = self._embed_or_zero(combined)
        meta = {
            "created_at": datetime.now().isoformat(),
            "type": "anti",
            "trigger_text": original_theory[:200],
            "anti_text": counter_theory[:200],
        }
        self.anti.add(vec, meta)
        self.anti.build_index()
        self._save()
        return len(self.anti)

    def run_anchor_generation(self, theory_text: str) -> Dict[str, Any]:
        """完整流程：理论→代价挖掘→反体系锚点"""
        result = {"native_count": len(self.native), "anti_count": len(self.anti)}
        if not self.client:
            result["error"] = "未配置 Ollama 客户端"
            return result

        # 先把理论本身存为原生锚点
        self.seed_native_anchor(theory_text, source="theory_input")

        # 调用 LLM 生成反体系
        prompt = f"""请针对下面的理论，生成一个完全对立的逻辑体系。
要求：至少否定原理论的一个核心前提，推理链完整自洽。

原始理论:
{theory_text}

请用简洁的条目列出反体系的核心主张、推理链、预测。"""

        anti = self.client.chat(
            user_message=prompt,
            system_prompt="你是一个批判性反体系构建者，擅长从相反假设出发构建自洽的理论。",
        )
        if anti:
            self.generate_anti_anchor(theory_text, anti)
            result["anti_generated"] = True

        result["native_count"] = len(self.native)
        result["anti_count"] = len(self.anti)
        return result

    # ========== 重构准入校验 ==========

    def validate_new_branch(self, proposed_text: str,
                            original_text: str = "",
                            veil_threshold: float = 0.7,
                            cost_threshold: float = 0.5) -> AnchorValidation:
        """校验新理论/代码分支是否通过准入规则"""
        query_vec = self._embed_or_zero(proposed_text)

        native_sim = self.native.nearest_similarity(query_vec)
        anti_sim = self.anti.nearest_similarity(query_vec)

        # 遮蔽程度：与原生锚点的偏离度（越低越好 = 更少被人类概念遮蔽）
        veil_reduction = 1.0 - native_sim

        # 反体系契合：与反体系锚点的相似度（越高越好 = 跳出了原体系）
        cost_reduction = anti_sim

        details = []
        if native_sim > veil_threshold:
            details.append(f"⚠️ 与原生锚点相似度 {native_sim:.2f} 过高（阈值 {veil_threshold}），可能仍受人类概念遮蔽")
        if anti_sim < cost_threshold and len(self.anti) > 0:
            details.append(f"⚠️ 与反体系锚点相似度 {anti_sim:.2f} 过低（阈值 {cost_threshold}），未有效跳出原体系")

        passed = len(details) == 0

        # 如果还没建立锚点，默认通过
        if len(self.native) == 0 and len(self.anti) == 0:
            passed = True
            details.append("（锚点库为空，默认通过）")

        return AnchorValidation(
            passed=passed,
            native_similarity=native_sim,
            anti_similarity=anti_sim,
            veil_reduction=veil_reduction,
            cost_reduction=cost_reduction,
            details=details,
        )

    # ========== 时间漂移检测 ==========

    def check_time_drift(self) -> DriftAlert:
        """执行一次时间漂移检测并缓存结果"""
        alert = self.drift_detector.analyze(self.native.metadata)
        self._last_drift_alert = alert
        return alert

    def get_last_drift_alert(self) -> Optional[DriftAlert]:
        return self._last_drift_alert

    # ========== 状态输出 ==========

    def print_status(self):
        print()
        print("=" * 50)
        print("  🛡️ 超脱锚点状态")
        print("=" * 50)
        print(f"  原生锚点库: {len(self.native)} 条")
        print(f"  反体系锚点库: {len(self.anti)} 条")
        print(f"  向量维度: {self.native.dim}")
        print(f"  使用 FAISS: {'✅' if self.native.use_faiss else '❌ (降级为纯Python)'}")

        if self.native.vectors:
            print(f"\n  📌 最近原生锚点:")
            for m in self.native.metadata[-3:]:
                print(f"    [{m.get('source', '?')}] {m.get('text_preview', '?')[:60]}...")

        if self.anti.vectors:
            print(f"\n  🔀 最近反体系锚点:")
            for m in self.anti.metadata[-3:]:
                print(f"    {m.get('anti_text', '?')[:60]}...")

        # 时间漂移检测
        print(f"\n  ⏳ 时间漂移检测器:")
        alert = self._last_drift_alert
        if alert is None:
            alert = self.check_time_drift()
        icon = "⚠️" if alert.detected else "✅"
        print(f"    漂移状态:   {icon} {'检测到系统性偏离' if alert.detected else '平稳'}")
        print(f"    漂移分数:   {alert.drift_score}")
        print(f"    预期速率:   {alert.expected_rate} 条/天 vs 当前 {alert.current_rate} 条/天")
        print(f"    窗口:       {alert.window_days} 天")
        if alert.anomaly_keywords:
            print(f"    异常关键词: {', '.join(alert.anomaly_keywords[:5])}")
        print(f"    建议:       {alert.recommendation[:100]}")

        # 检查 embedding 模型可用性
        vec = self._embed("test")
        if vec:
            print(f"\n  Embedding: ✅ 可用 (dim={len(vec)})")
        else:
            print(f"\n  Embedding: ⚠️  不可用（运行: ollama pull nomic-embed-text）")

        print("=" * 50)


if __name__ == "__main__":
    DetachmentAnchor().print_status()
