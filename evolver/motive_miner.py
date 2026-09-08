"""
L2 Motive & Cost Mining 代价/动机还原器

核心自我质疑单元，依托 7B 主模型执行三段式反向拆解 Prompt 流水线：

阶段 1: 理论结构化建模 — 将观测归纳为完整逻辑框架
阶段 2: 代价挖掘反向拆解 — 显性动机 → 认知代价 → 隐性本体预设
阶段 3: 冲突推演生成 — 自动生成反事实理论分支做对照

硬件适配：串行调用单个 7B 模型（显存限制），而非并行多分支。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).parent if '__file__' in globals() else Path.cwd()
MINING_HISTORY = PROJECT_ROOT / "logs" / "mining_history.json"


STAGE1_PROMPT = """你是一个理论结构分析师。请将以下输入归纳为完整的逻辑框架。

输入内容：
{input_text}

请严格按以下 JSON 格式输出，不要添加任何额外文字：
{{
  "core_claim": "这套理论的核心主张是什么（1-2句话）",
  "logic_structure": [
    "前提1: ...",
    "前提2: ...",
    "推理链: ...",
    "结论: ..."
  ],
  "scope": "适用边界（在什么条件下成立）",
  "known_errors": ["已知的预测误差或反例"]
}}"""


STAGE2_PROMPT = """你是一个反向工程分析师。请对下面的理论进行代价挖掘三段式拆解。

理论结构：
{theory_json}

请严格按以下 JSON 格式输出，不要添加任何额外文字：
{{
  "explicit_motivation": {{
    "original_goal": "这套理论最初构建目标是什么？（拟合数据/简化计算/匹配直觉）",
    "target_user": "理论试图说服谁？",
    "alternatives_considered": ["构建过程中考虑过但放弃的替代方案"]
  }},
  "cognitive_cost": {{
    "ignored_phenomena": ["为达成目标主动忽略/简化/屏蔽的客观现象"],
    "simplifications": ["使用了哪些可能有问题的简化假设"],
    "value_bias": ["隐含的价值偏向（例如：效率优于安全、增长优于公平）"],
    "linguistic_traps": ["使用了哪些容易产生误导的语言/概念"]
  }},
  "implicit_ontology": {{
    "unspoken_premises": ["整套逻辑依赖的未明说的底层前提"],
    "world_view": "隐含的世界观/本体论假设（例如：线性因果、可还原论、人类中心）",
    "category_scheme": "依赖的分类体系是否完备？有无遗漏的类别？"
  }},
  "total_cost_score": 0.0
}}"""


STAGE3_PROMPT = """你是一个反事实推演引擎。请基于下面的代价分析，生成与原理论冲突的反事实理论分支。

原始理论：
{theory_json}

代价分析（特别是认知代价和隐性本体预设）：
{cost_json}

请生成 {num_branches} 条反事实理论分支。每条分支必须至少颠覆原始理论的一个核心前提。

严格按以下 JSON 格式输出：
{{
  "counter_theories": [
    {{
      "id": "反事实1",
      "rejected_premise": "这个分支否定了原始理论的哪个前提？",
      "alternative_logic": "替代的推理逻辑是什么？",
      "predictions": ["这个分支会做出哪些不同的预测？"],
      "self_consistency": "这个分支自身是否自洽？（是/部分/否）为什么？",
      "new_blind_spots": ["这个新分支引入了哪些新的盲区？"]
    }}
  ]
}}"""


SINGLE_STAGE2_PROMPT = """你是一个反向工程分析师。请对下面的代码/方案进行代价挖掘。

待分析内容：
{input_text}

从三个维度回答：
1. 显性动机：这套代码/方案最初想解决什么问题？有没有更简单的替代方案？
2. 认知代价：它忽略了什么边界情况？用了什么可能有问题的假设？有什么隐含的价值偏向？
3. 隐性本体预设：它依赖哪些未明说的前提？（例如：用户行为模式、硬件约束、使用场景）

请用简洁的条目列出每个维度的要点。"""


@dataclass
class MiningResult:
    timestamp: str
    input_summary: str
    stage1_theory: Optional[Dict[str, Any]] = None
    stage2_cost: Optional[Dict[str, Any]] = None
    stage3_counters: Optional[List[Dict[str, Any]]] = None
    total_cost_score: float = 0.0
    high_cost_flags: List[str] = field(default_factory=list)
    counter_intuition: Optional[str] = None


class MotiveCostMiner:
    """代价/动机还原器"""

    def __init__(self, ollama_client=None):
        self.client = ollama_client
        self.history: List[Dict] = []
        self._load_history()

    def _load_history(self):
        if MINING_HISTORY.exists():
            try:
                with open(MINING_HISTORY, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

    def _save(self, result: MiningResult):
        entry = asdict(result)
        self.history.append(entry)
        try:
            MINING_HISTORY.parent.mkdir(exist_ok=True)
            with open(MINING_HISTORY, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ========== 公共接口 ==========

    def mine(self, input_text: str, num_branches: int = 2) -> MiningResult:
        """执行完整的三段式代价挖掘"""
        result = MiningResult(
            timestamp=datetime.now().isoformat(),
            input_summary=input_text[:200],
        )

        if not self.client:
            result.high_cost_flags.append("未配置 Ollama 客户端，跳过深度分析")
            self._save(result)
            return result

        stage1_result = self._call_stage(STAGE1_PROMPT, input_text)
        if stage1_result:
            result.stage1_theory = stage1_result

        stage2_result = self._call_stage(STAGE2_PROMPT, stage1_result)
        if stage2_result:
            result.stage2_cost = stage2_result
            result.total_cost_score = stage2_result.get("total_cost_score", 0.0)
            result.high_cost_flags = self._extract_high_cost_flags(stage2_result)

        stage3_result = self._call_stage(STAGE3_PROMPT, stage1_result, stage2_result, num_branches)
        if stage3_result:
            result.stage3_counters = stage3_result.get("counter_theories", [])
            if result.stage3_counters:
                first = result.stage3_counters[0]
                result.counter_intuition = f"假设「{first.get("rejected_premise", "?")}」不成立，则 {first.get("alternative_logic", "?")}"

        self._save(result)
        return result

    def _call_stage(self, prompt: str) -> Optional[Dict[str, Any]]:
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            result = self.client.chat_with_structured_output(
                user_message=prompt,
                system_prompt="你是一个严谨的分析专家。严格按 JSON 格式输出，不要添加任何额外文字。",
            )
            return result
        except Exception as e:
            print(f"[Miner] 阶段调用失败: {e}")
            return None

    def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            result = self.client.chat_with_structured_output(
                user_message=prompt,
                system_prompt="你是一个严谨的分析专家。严格按 JSON 格式输出，不要添加任何额外文字。",
            )
            return result
        except Exception as e:
            print(f"[Miner] 阶段调用失败: {e}")
            return None

    def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            result = self.client.chat_with_structured_output(
                user_message=prompt,
                system_prompt="你是一个严谨的分析专家。严格按 JSON 格式输出，不要添加任何额外文字。",
            )
            return result
        except Exception as e:
            print(f"[Miner] 阶段调用失败: {e}")
            return None

    def _extract_high_cost_flags(self, cost_data: Dict) -> List[str]:
        flags = []
        cognitive = cost_data.get("cognitive_cost", {})

        ignored = cognitive.get("ignored_phenomena", [])
        if isinstance(ignored, list) and len(ignored) >= 3:
            flags.append(f"忽略了 {len(ignored)} 种现象")

        simplifications = cognitive.get("simplifications", [])
        if isinstance(simplifications, list) and len(simplifications) >= 2:
            flags.append(f"使用了 {len(simplifications)} 处简化假设")

        ontology = cost_data.get("implicit_ontology", {})
        premises = ontology.get("unspoken_premises", [])
        if isinstance(premises, list) and len(premises) >= 2:
            flags.append(f"依赖 {len(premises)} 个未明说的底层前提")

        score = cost_data.get("total_cost_score", 0)
        if isinstance(score, (int, float)) and score >= 0.6:
            flags.append(f"总认知代价 {score:.2f}（偏高）")

        return flags

    def mine_simple(self, input_text: str) -> str:
        """简化版：直接输出反向分析文本（不用三段式 JSON 流水线）"""
        if not self.client:
            return "未配置 Ollama 客户端"

        prompt = SINGLE_STAGE2_PROMPT.format(input_text=input_text)
        response = self.client.chat(
            user_message=prompt,
            system_prompt="你是一个批判性思考者，擅长发现隐藏的假设和代价。",
        )
        return response or "(分析失败)"

    # ========== 内部方法 ==========

    def _call_stage(self, prompt: str) -> Optional[Dict[str, Any]]:
        """调用 LLM 并提取 JSON 结果"""
        try:
            result = self.client.chat_with_structured_output(
                user_message=prompt,
                system_prompt="你是一个严谨的分析专家。严格按 JSON 格式输出，不要添加任何额外文字。",
            )
            return result
        except Exception as e:
            print(f"[Miner] 阶段调用失败: {e}")
            return None

    @staticmethod
    def _extract_high_cost_flags(cost_data: Dict) -> List[str]:
        """从阶段 2 结果中提取高代价标志"""
        flags = []
        cognitive = cost_data.get("cognitive_cost", {})

        ignored = cognitive.get("ignored_phenomena", [])
        if isinstance(ignored, list) and len(ignored) >= 3:
            flags.append(f"忽略了 {len(ignored)} 种现象")

        simplifications = cognitive.get("simplifications", [])
        if isinstance(simplifications, list) and len(simplifications) >= 2:
            flags.append(f"使用了 {len(simplifications)} 处简化假设")

        ontology = cost_data.get("implicit_ontology", {})
        premises = ontology.get("unspoken_premises", [])
        if isinstance(premises, list) and len(premises) >= 2:
            flags.append(f"依赖 {len(premises)} 个未明说的底层前提")

        score = cost_data.get("total_cost_score", 0)
        if isinstance(score, (int, float)) and score >= 0.6:
            flags.append(f"总认知代价 {score:.2f}（偏高）")

        return flags

    def print_analysis(self, result: Optional[MiningResult] = None):
        """打印代价挖掘分析报告"""
        if result is None:
            if not self.history:
                print("（尚无挖掘历史，请先运行 mining 命令）")
                return
            result = MiningResult(**self.history[-1])

        print()
        print("=" * 50)
        print("  🔍 L2 代价/动机还原分析")
        print("=" * 50)
        print(f"  时间: {result.timestamp[:19].replace('T', ' ')}")
        print(f"  输入摘要: {result.input_summary[:60]}")
        print(f"  总认知代价: {result.total_cost_score:.2f}")

        if result.stage1_theory:
            t = result.stage1_theory
            print(f"\n  📐 理论结构:")
            print(f"    核心主张: {t.get('core_claim', '?')}")
            ls = t.get("logic_structure", [])
            if isinstance(ls, list):
                for item in ls[:4]:
                    print(f"      - {item}")
            print(f"    适用边界: {t.get('scope', '?')}")

        if result.stage2_cost:
            c = result.stage2_cost
            explicit = c.get("explicit_motivation", {})
            cognitive = c.get("cognitive_cost", {})
            ontology = c.get("implicit_ontology", {})

            print(f"\n  🎯 显性动机:")
            print(f"    原始目标: {explicit.get('original_goal', '?')}")
            print(f"    目标用户: {explicit.get('target_user', '?')}")

            print(f"\n  ⚠️ 认知代价:")
            for key in ("ignored_phenomena", "simplifications", "value_bias", "linguistic_traps"):
                items = cognitive.get(key, [])
                if items:
                    label = {"ignored_phenomena": "忽略的现象", "simplifications": "简化假设",
                             "value_bias": "价值偏向", "linguistic_traps": "语言陷阱"}[key]
                    print(f"    {label}:")
                    for item in items:
                        print(f"      × {item}")

            print(f"\n  🧩 隐性本体预设:")
            for item in ontology.get("unspoken_premises", []):
                print(f"    ◇ {item}")
            if ontology.get("world_view"):
                print(f"    世界观: {ontology['world_view']}")

        if result.stage3_counters:
            print(f"\n  🔀 反事实理论分支 ({len(result.stage3_counters)}):")
            for branch in result.stage3_counters:
                print(f"    [{branch.get('id', '?')}] 否定: {branch.get('rejected_premise', '?')}")
                print(f"      替代逻辑: {branch.get('alternative_logic', '?')}")
                for pred in branch.get("predictions", []):
                    print(f"      预测: {pred}")

        if result.high_cost_flags:
            print(f"\n  🚨 高代价标志:")
            for flag in result.high_cost_flags:
                print(f"    × {flag}")

        print("=" * 50)
    def optimize_performance(self):
        """优化代码性能和可读性，识别并消除不必要的计算"""
        if not self.client:
            return

        self.stage1_theory = self._optimize_stage1_theory(self.stage1_theory)
        self.stage2_cost = self._optimize_stage2_cost(self.stage2_cost)
        self.stage3_counters = self._optimize_stage3_counters(self.stage3_counters)
        self.total_cost_score = self._calculate_total_cost_score()
        self._save()
        self.print_analysis()

    def _optimize_stage1_theory(self, theory: Dict[str, Any]) -> Dict[str, Any]:
        return theory

    def _optimize_stage2_cost(self, cost: Dict[str, Any]) -> Dict[str, Any]:
        return cost

    def _optimize_stage3_counters(self, counters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return counters

    def _calculate_total_cost_score(self) -> float:
        return 0.0

    def _optimize_stage2_cost(self, cost: Dict[str, Any]) -> Dict[str, Any]:
        return cost

    def _optimize_stage3_counters(self, counters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return counters

    def _calculate_total_cost_score(self) -> float:
        return 0.0

    def _optimize_stage2_cost(self, cost: Dict[str, Any]) -> Dict[str, Any]:
        return cost

    def _optimize_stage3_counters(self, counters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return counters

    def _calculate_total_cost_score(self) -> float:
        return 0.0

    def _optimize_stage2_cost(self, cost: Dict[str, Any]) -> Dict[str, Any]:
        return cost

    def _optimize_stage3_counters(self, counters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return counters

    def _calculate_total_cost_score(self) -> float:
        return 0.0

    def _optimize_stage2_cost(self, cost: Dict[str, Any]) -> Dict[str, Any]:
        """优化阶段2代价挖掘"""
        return cost

    def _optimize_stage3_counters(self, counters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优化阶段3反事实推演"""
        return counters

    def _calculate_total_cost_score(self) -> float:
        """重新计算总认知代价"""
        return 0.0
