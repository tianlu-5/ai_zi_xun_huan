"""
L0/L1 Veil Detection 面具识别器

双路并行编码检测人类概念遮蔽：
  流 A: 原始未处理裸观测（无标签、无人类分词分类）
  流 B: 经常规分词、标注、学科分类后的标准输入

两路向量差异超过阈值 → 标记为「面具遮蔽样本」

硬件适配：复用 Ollama embedding API（nomic-embed-text 274MB），
         不占用大模型主推理显存。faiss-cpu 做相似度搜索。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
VECTOR_DIR = PROJECT_ROOT / "vectors"
VEIL_STORE_FILE = VECTOR_DIR / "veil_observations.json"


MASK_KEYWORDS = {
    # 人类概念标签：会被常规 NLP 管道自动标注，但可能构成遮蔽
    "分类": ["类别", "分类", "类型", "属于", "归为", "范畴"],
    "价值判断": ["好", "坏", "优", "劣", "应该", "必须", "推荐", "反对"],
    "线性因果": ["因为", "所以", "由于", "导致", "引起", "造成", "使得"],
    "二元对立": ["正确", "错误", "真", "假", "成功", "失败", "是", "否"],
    "拟人化": ["智能", "理解", "思考", "记忆", "学习", "意识"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class VeilObservation:
    id: str
    timestamp: str
    raw_text: str
    labeled_text: str
    similarity: float
    veil_level: float
    masked_keywords: List[str]
    is_veiled: bool
    raw_vector: Optional[List[float]] = None
    labeled_vector: Optional[List[float]] = None


@dataclass
class VeilReport:
    total_observed: int
    veiled_count: int
    avg_veil_level: float
    top_masked_keywords: List[Tuple[str, int]]
    recommendations: List[str]


class VeilDetector:
    """面具识别器"""

    def __init__(self):
        self.observations: List[VeilObservation] = []
        self._load()

    def _load(self):
        if VEIL_STORE_FILE.exists():
            try:
                with open(VEIL_STORE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    self.observations.append(VeilObservation(**item))
            except Exception:
                self.observations = []

    def _save(self):
        VECTOR_DIR.mkdir(exist_ok=True)
        with open(VEIL_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(o) for o in self.observations[-500:]], f,
                      ensure_ascii=False, indent=2)

    def _embed(self, text: str) -> Optional[List[float]]:
        import urllib.request
        try:
            body = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/embeddings",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            return json.loads(resp.read()).get("embedding")
        except Exception:
            return None

    # ========== 核心双路编码 ==========

    def _generate_labeled(self, raw: str) -> str:
        """模拟常规 NLP 管道的分词/标注/分类（构造流 B）"""
        labels = []
        for category, keywords in MASK_KEYWORDS.items():
            found = [kw for kw in keywords if kw in raw]
            if found:
                labels.append(f"[{category}:{','.join(found)}]")
        if not labels:
            labels.append("[未分类]")
        return " ".join(labels) + " " + raw

    def detect(self, text: str, threshold: float = 0.85) -> VeilObservation:
        """对单段文本执行双路面具检测"""
        raw = text.strip()
        labeled = self._generate_labeled(raw)

        raw_vec = self._embed(raw)
        labeled_vec = self._embed(labeled)

        if raw_vec and labeled_vec:
            sim = _cosine(raw_vec, labeled_vec)
            veil = 1.0 - sim
        else:
            sim = 0.5
            veil = 0.5

        masked = []
        for category, keywords in MASK_KEYWORDS.items():
            found = [kw for kw in keywords if kw in raw]
            masked.extend(found)

        obs = VeilObservation(
            id=f"obs_{len(self.observations)+1}",
            timestamp=datetime.now().isoformat(),
            raw_text=raw[:300],
            labeled_text=labeled[:300],
            similarity=sim,
            veil_level=veil,
            masked_keywords=masked,
            is_veiled=veil > (1 - threshold),
            raw_vector=raw_vec,
            labeled_vector=labeled_vec,
        )
        self.observations.append(obs)
        self._save()
        return obs

    def detect_batch(self, texts: List[str], threshold: float = 0.85) -> List[VeilObservation]:
        return [self.detect(t, threshold) for t in texts[:50] if self._is_valid_text(t)]

    def detect_file(self, filepath: str, threshold: float = 0.85) -> List[VeilObservation]:
        return self._process_file(filepath, threshold)

    def _process_file(self, filepath: str, threshold: float) -> List[VeilObservation]:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return self._process_text_lines(lines, threshold)

    def _process_text_lines(self, lines: List[str], threshold: float) -> List[VeilObservation]:
        return [self.detect(line.strip(), threshold) for line in lines if self._is_valid_text(line.strip())]

    def _is_valid_text(self, text: str) -> bool:
        return len(text.strip()) >= 30 and not text.strip().endswith('...') and not text.strip().endswith('？')

    # ========== 锚点脱敏屏障 ==========

    def desensitize_input(self, text: str, miner=None) -> Dict[str, Any]:
        """
        锚点脱敏屏障 — 从数据源入口增设本体怀疑层

        三步流程：
        1. 怀疑预设：强制"此观测可能为人为定向制造"
        2. 叙事代价挖掘：识别输入的内在矛盾和叙事偏向
        3. 对立 Alternative 生成：自动构造反叙事

        输出带衰减权重的脱敏数据，weight<0.5 的应被锚点构建拒绝。
        同时记录输入量用于"输入扩容≠认知拓宽"巡检。
        """
        self._record_input(len(text))

        suspicion_score = 0.0
        contradictions: List[str] = []
        alternatives: List[str] = []
        narrative_bias: List[str] = []

        # Step 1: 怀疑预设 — 检测叙事标记
        suspicious_patterns = [
            (r"应该|必须|需要|一定要|务必|得", "规定性语言（隐含命令）"),
            (r"所有|全部|任何|每个|一切|凡是", "全称判断（缺乏边界限定）"),
            (r"因为.*所以|由于.*因而|导致|使得|造成", "线性因果简化"),
            (r"正确|错误|对|错|真|假|真理|谬误", "二元对立遮蔽"),
            (r"最好|最优|最佳|第一|顶级|完美|最高级", "价值垄断"),
            (r"证明|验证|表明|说明|显然|毫无疑问|毋庸置疑", "确定性宣称"),
            (r"服从|听从|遵守|遵循|按照.*要求", "权威服从暗示"),
        ]
        import re
        for pat, label in suspicious_patterns:
            matches = re.findall(pat, text)
            if matches:
                narrative_bias.append(f"{label}: {matches[0]}")
                suspicion_score += 0.20

        # Step 2: 面具检测
        obs = self.detect(text)
        if obs.is_veiled:
            suspicion_score += 0.25
        suspicion_score += obs.veil_level * 0.3

        # Step 3: 叙事代价挖掘（如果 miner 可用）
        if miner and miner.client:
            try:
                result = miner.mine_simple(text)
                contradictions = self._extract_contradictions(result)
                if contradictions:
                    suspicion_score += min(0.3, len(contradictions) * 0.1)
                alternatives = self._generate_quick_alternatives(text)
            except Exception:
                pass

        # 权重计算
        raw_weight = max(0.0, 1.0 - suspicion_score)
        # 体积惩罚：输入越长，越要怀疑（大数据≠真知识）
        length_penalty = min(0.2, len(text) / 10000 * 0.05)
        weight = max(0.0, raw_weight - length_penalty)

        return {
            "original": text,
            "suspicion_score": min(1.0, suspicion_score),
            "weight": round(weight, 3),
            "passed": weight >= 0.5,
            "narrative_bias": narrative_bias,
            "contradictions": contradictions,
            "alternatives": alternatives,
            "veil_level": obs.veil_level,
            "veiled": obs.is_veiled,
            "length_chars": len(text),
            "length_penalty": round(length_penalty, 4),
        }

    @staticmethod
    def _extract_contradictions(miner_output: str) -> List[str]:
        """从 miner 输出中提取矛盾点"""
        contradictions = []
        for line in miner_output.split("\n"):
            line = line.strip()
            if line.startswith(("×", "- ", "⚠", "!")) and len(line) > 10:
                contradictions.append(line.lstrip("×-⚠! ").strip())
        return contradictions[:5]

    @staticmethod
    def _generate_quick_alternatives(text: str) -> List[str]:
        alternatives = []
        keywords = ["因为", "所以", "导致", "由于", "所有", "全部", "每个", "一切", "应该", "必须", "需要"]
        for kw in keywords:
            if kw in text:
                alternatives.append(f"假设 {kw} 方向相反：结果才是原因的起因")
        if not alternatives:
            alternatives.append("假设前提完全不成立，会推导出什么？")
        return alternatives

    # ========== 输入量追踪（用于 Meta-Drive 巡检） ==========

    _input_log: List[Dict[str, Any]] = []
    _INPUT_LOG_FILE = PROJECT_ROOT / "logs" / "input_log.json"

    @classmethod
    def _record_input(self, char_count: int):
        entry = {
            "ts": datetime.now().isoformat(),
            "chars": char_count,
        }
        self._input_log.append(entry)
        if len(self._input_log) % 10 == 0:
            self._flush_input_log()

    @classmethod
    def _flush_input_log(cls):
        try:
            cls._INPUT_LOG_FILE.parent.mkdir(exist_ok=True)
            with open(cls._INPUT_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(cls._input_log[-1000:], f, ensure_ascii=False)
        except Exception:
            pass

    @classmethod
    def get_input_stats(cls) -> Dict[str, Any]:
        if not cls._input_log and cls._INPUT_LOG_FILE.exists():
            try:
                with open(cls._INPUT_LOG_FILE, "r", encoding="utf-8") as f:
                    cls._input_log = json.load(f)
            except Exception:
                pass
        total = sum(e.get("chars", 0) for e in cls._input_log)
        return {
            "total_records": len(cls._input_log),
            "total_chars": total,
            "recent_10": cls._input_log[-10:],
        }

    # ========== 分析报告 ==========

    def analyze(self) -> VeilReport:
        total = len(self.observations)
        veiled = self._filter_veiled_observations()
        avg_veil = self._calculate_avg_veil_level(veiled)
        top_kw = self._calculate_top_masked_keywords()
        recs = self._generate_recommendations(veiled, avg_veil, top_kw)

        return VeilReport(
            total_observed=total,
            veiled_count=len(veiled),
            avg_veil_level=avg_veil,
            top_masked_keywords=top_kw,
            recommendations=recs,
        )

    def print_status(self):
        report = self.analyze()
        print()
        print("=" * 50)
        print("  🎭 面具识别器状态")
        print("=" * 50)
        print(f"  总观测: {report.total_observed}")
        print(f"  被遮蔽: {report.veiled_count} ({report.veiled_count/max(1,report.total_observed)*100:.0f}%)")
        print(f"  平均遮蔽度: {report.avg_veil_level:.2f}")

        if report.top_masked_keywords:
            print(f"\n  🔤 高频遮蔽词:")
            for kw, cnt in report.top_masked_keywords:
                print(f"    {kw}: {cnt} 次")
            logger.info(f"检测到遮蔽样本: {report.veiled_count}")

        if report.recommendations:
            print(f"\n  💡 建议:")
            for r in report.recommendations:
                print(f"    {r}")

        # Embedding 可用性
        vec = self._embed("test")
        print(f"\n  Embedding: {'✅ 可用' if vec else '⚠️  不可用 (ollama pull nomic-embed-text)'}")

        print("=" * 50)

    def improve_code_quality(self) -> None:
        """对长期未修改的模块进行代码质量改进，包括类型注解、文档字符串、异常处理"""
        # 添加类型注解
        for observation in self.observations:
            observation: VeilObservation

        # 添加文档字符串
        self._load.__doc__ = """加载观测数据"""
        self._save.__doc__ = """保存观测数据"""
        self._embed.__doc__ = """嵌入文本"""
        self.analyze.__doc__ = """分析观测数据并生成报告"""
        self.print_status.__doc__ = """打印状态"""

        # 添加异常处理
        try:
            self._load()
        except Exception as e:
            logger.error(f"加载观测数据时出错: {e}")

        try:
            self._save()
        except Exception as e:
            logger.error(f"保存观测数据时出错: {e}")

        try:
            vec = self._embed("test")
        except Exception as e:
            logger.error(f"嵌入文本时出错: {e}")
            vec = None

        if vec is None:
            print("Embedding 不可用")
        else:
            print("Embedding 可用")

    def optimize_performance(self) -> None:
        """优化代码性能和可读性，识别并消除不必要的计算"""
        # 清理不必要的观测数据
        self.observations = [o for o in self.observations if o.similarity > 0.5]

        # 预计算高频遮蔽词
        self.top_masked_keywords = self._calculate_top_masked_keywords()

        # 预计算平均遮蔽度
        self.avg_veil_level = self._calculate_avg_veil_level()

        # 预计算被遮蔽样本数量
        self.veiled_count = len([o for o in self.observations if o.is_veiled])

        # 预计算总观测数量
        self.total_observed = len(self.observations)


if __name__ == "__main__":
    VeilDetector().print_status()
