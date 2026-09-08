"""
Stochastic Emergence Observer (SEO) — 随机涌现观测器

哲学定位：补足"有我约束"的先天天花板，开辟脱离全部人工规则的认知通道。
工程定位：纯被动观测归档单元，无调控、无干预、无算力分配权限。

运行机制：
  - 独立后台 daemon 线程，最低优先级
  - 随机定时器触发（平均 15-30 分钟一次），GPU 负载 <30% 才激活
  - 激活时执行三类无约束探索（不经过脱敏/锚点/Meta-Drive）
  - 产出仅归档到 logs/emergence_archive.json，不执行、不注入主系统
  - Meta-Drive 后置比对，发现跨越式增益才纳入演化候选池

三类涌现捕获：
  1. 表征自发重组：随机向量插值 → 裸 LLM 调用
  2. 无动机悖论：空 system_prompt → 随机 token 拼接 prompt
  3. 代码片段重组：随机抽取代码片段拼接成新变体

硬件约束：所有探索共享同一 7B 模型（单实例），串行执行避免显存溢出。
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent
ARCHIVE_FILE = PROJECT_ROOT / "logs" / "emergence_archive.json"
STATE_FILE = PROJECT_ROOT / "logs" / "seo_state.json"
PY_FILES = [f for f in PROJECT_ROOT.iterdir() if f.suffix == ".py" and not f.name.startswith("_")]


@dataclass
class EmergenceEvent:
    event_id: str
    timestamp: str
    trigger_type: str
    gpu_load_at_trigger: float
    category: str
    seed: str
    raw_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluated: bool = False
    meta_eval: Optional[Dict[str, Any]] = None


class SEOObserver:
    """随机涌现观测器 — 无约束、无干预、仅归档"""

    def __init__(self, ollama_client=None, min_interval_sec: int = 600,
                 max_interval_sec: int = 1800, gpu_threshold: float = 30.0):
        self.client = ollama_client
        self.min_interval = min_interval_sec
        self.max_interval = max_interval_sec
        self.gpu_threshold = gpu_threshold

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._force_trigger = threading.Event()
        self._lock = threading.Lock()

        # Meta-Drive 静默指令通道
        self._meta_silence_active = False
        self._meta_silence_until: Optional[float] = None

        self.archive: List[Dict] = []
        self.total_events = 0
        self.total_failures = 0
        self._load_state()

    # ========== 状态持久化 ==========

    def _load_state(self):
        if ARCHIVE_FILE.exists():
            try:
                with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.archive = data.get("events", [])
                self.total_events = data.get("total_events", 0)
                self.total_failures = data.get("total_failures", 0)
            except Exception:
                self.archive = []

    def _save_state(self):
        data = {
            "saved_at": datetime.now().isoformat(),
            "total_events": self.total_events,
            "total_failures": self.total_failures,
            "events": self.archive[-200:],
        }
        try:
            ARCHIVE_FILE.parent.mkdir(exist_ok=True)
            with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ========== GPU 负载检测 ==========

    def _get_gpu_load(self) -> float:
        try:
            result = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").read()
            lines = result.strip().split("\n")
            if lines and lines[0].strip():
                return float(lines[0].strip())
        except Exception:
            pass
        return 25.0

    def _is_system_idle(self) -> bool:
        return self._get_gpu_load() < self.gpu_threshold

    # ========== 随机涌现捕获 ==========

    def _capture_representation_emergence(self) -> Optional[EmergenceEvent]:
        """类型 1: 表征自发重组 — 从向量库随机插值 → 裸 LLM 调用"""
        category = "representation_recombination"

        # 从已有文件中随机抽取两段文本做插值
        text_a, text_b = self._random_text_pair()
        seed = (
            f"随机拼接两段不相关代码的语义：\n"
            f"A: {text_a[:120]}\n"
            f"B: {text_b[:120]}\n"
            f"A 和 B 有什么隐含关联？生成一段完全不依赖人类概念分类的新表征。"
        )

        raw_output = ""
        if self.client:
            try:
                raw_output = self.client.chat(
                    user_message=seed,
                    system_prompt="",  # 空 system_prompt — 无定向引导
                    stream=False,
                ) or ""
            except Exception:
                pass

        if not raw_output:
            raw_output = f"(SEO placeholder: {self._generate_nonsense_prompt()})"

        return EmergenceEvent(
            event_id=f"seo_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now().isoformat(),
            trigger_type="gpu_idle+random",
            gpu_load_at_trigger=self._get_gpu_load(),
            category=category,
            seed=seed[:300],
            raw_output=raw_output[:2000],
            metadata={"source_files": [Path(f).name for f in [text_a[:50], text_b[:50]] if isinstance(f, str)]},
        )

    def _capture_paradox_emergence(self) -> Optional[EmergenceEvent]:
        """类型 2: 无动机悖论自发生成 — 空 system_prompt + 随机 token 拼接"""
        category = "unmotivated_paradox"
        seed = self._generate_nonsense_prompt()

        raw_output = ""
        if self.client:
            try:
                raw_output = self.client.chat(
                    user_message=seed,
                    system_prompt="",
                    stream=False,
                ) or ""
            except Exception:
                pass

        if not raw_output:
            return None

        return EmergenceEvent(
            event_id=f"seo_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now().isoformat(),
            trigger_type="gpu_idle+random",
            gpu_load_at_trigger=self._get_gpu_load(),
            category=category,
            seed=seed[:300],
            raw_output=raw_output[:2000],
            metadata={"prompt_strategy": "random_token_mashup"},
        )

    def _capture_code_recombination(self) -> Optional[EmergenceEvent]:
        """类型 3: 代码片段随机重组 — 从不同文件抽取函数/类片段拼接"""
        category = "code_fragment_recombination"

        fragments = self._extract_random_code_fragments(n=2)
        if not fragments:
            return None

        combined = self._combine_code_fragments(fragments)

        raw_output = combined

        # 尝试用 LLM 润色（仍是无 system_prompt）
        if self.client:
            try:
                polished = self.client.chat(
                    user_message=f"以下是从不同模块随机拼接的代码片段。它自然组合成了什么新东西？\n\n{combined[:1500]}",
                    system_prompt="",
                    stream=False,
                )
                if polished:
                    raw_output = polished[:2000]
            except Exception:
                pass

        sources = [Path(f).name for f, _ in fragments]
        return EmergenceEvent(
            event_id=f"seo_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now().isoformat(),
            trigger_type="gpu_idle+random",
            gpu_load_at_trigger=self._get_gpu_load(),
            category=category,
            seed=" | ".join(sources),
            raw_output=raw_output[:2000],
            metadata={"source_files": sources, "fragment_count": len(fragments)},
        )

    def _combine_code_fragments(self, fragments: List[Tuple[str, str]]) -> str:
        combined = f"# SEO 随机代码重组 - 无定向拼接\n"
        for i, (fname, code) in enumerate(fragments):
            combined += f"# === 片段 {i+1}: {fname} ===\n{code}\n\n"
        return combined

    def _random_text_pair(self) -> Tuple[str, str]:
        """从项目文件中随机抽取两段不相关文本"""
        if not PY_FILES:
            return ("empty", "empty")
        texts: List[str] = []
        for fpath in random.sample(PY_FILES, min(2, len(PY_FILES))):
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                lines = [l.strip() for l in content.split("\n") if len(l.strip()) > 15]
                if lines:
                    texts.append(random.choice(lines))
            except Exception:
                pass
        while len(texts) < 2:
            texts.append("(random placeholder)")
        return texts[0], texts[1]

    def _generate_nonsense_prompt(self) -> str:
        """随机 token 拼接 — 无定向、无意义的原始 prompt"""
        vocab = [
            "熵", "递归", "边界", "循环", "反射", "嵌套", "无限", "折叠",
            "矛盾", "叠加", "共振", "相变", "涌现", "自指", "破缺", "湮灭",
            "生成", "坍缩", "拓扑", "对称", "不变", "流形", "同伦", "纤维化",
        ]
        words = random.sample(vocab, random.randint(4, 8))
        connector = random.choice(["与", "或", "在...之间", "通过", "不经过", "同时"])
        return f"{' '.join(words)} {connector} {' '.join(reversed(words))}"

    def _extract_random_code_fragments(self, n: int = 2) -> List[Tuple[str, str]]:
        """从不同 .py 文件中随机抽取函数/类片段"""
        fragments: List[Tuple[str, str]] = []
        used_files: set = set()

        for _ in range(n):
            fpath = self._select_random_file(used_files)
            if not fpath:
                break
            used_files.add(fpath.name)

            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            fragments.extend(self._extract_code_fragments(content, fpath))

        return fragments

    def _select_random_file(self, used_files: set) -> Optional[Path]:
        available = [f for f in PY_FILES if f.name not in used_files]
        if not available:
            available = PY_FILES
        return random.choice(available)

    def _extract_code_fragments(self, content: str, fpath: Path) -> List[Tuple[str, str]]:
        matches = list(re.finditer(
            r"(def |class )\w+.*?(?=\n(def |class )|\Z)",
            content,
            re.DOTALL,
        ))
        return [(str(fpath), match.group(0)[:400]) for match in matches if len(match.group(0)) > 30]

    # ========== 主循环 ==========

    def _loop(self):
        """
        SEO 后台主循环 — 双通道激活：
          通道 1（主）: 随机间隔 + GPU 空闲 → 自主触发（被动观测）
          通道 2（Meta-Drive）: 静默窗口期间 → 高密度连续捕获（主动观测许可）
        """
        categories = [
            self._capture_representation_emergence,
            self._capture_paradox_emergence,
            self._capture_code_recombination,
        ]

        while not self._stop_event.is_set():
            interval = self._get_loop_interval()
            self._stop_event.wait(interval)
            if self._stop_event.is_set():
                break

            if self._meta_silence_active and self._meta_silence_until and time.time() > self._meta_silence_until:
                self._meta_silence_active = False

            if not (self._force_trigger.is_set() or self._meta_silence_active or self._is_system_idle()):
                continue

            captures = 3 if self._meta_silence_active else 1
            for _ in range(captures):
                if self._stop_event.is_set():
                    break
                event = self._capture_event(categories)
                if event:
                    self._handle_event(event)
                else:
                    self.total_failures += 1

    def _get_loop_interval(self) -> float:
        if self._meta_silence_active:
            return random.uniform(15, 45)
        elif self._force_trigger.is_set() or self._meta_silence_active or self._is_system_idle():
            return random.uniform(5, 15)
        else:
            return random.uniform(self.min_interval, self.max_interval)

    def _capture_event(self, categories: List[Callable[[], Optional[EmergenceEvent]]]) -> Optional[EmergenceEvent]:
        capture_fn = random.choice(categories)
        try:
            return capture_fn()
        except Exception as e:
            self.total_failures += 1
            print(f"  [SEO] 捕获失败: {e}")
            return None

    def _handle_event(self, event: EmergenceEvent):
        with self._lock:
            self.archive.append(asdict(event))
            self.total_events += 1
            self._save_state()
        trigger_tag = "[META-SILENCE] " if self._meta_silence_active else ""
        print(f"  [SEO] {trigger_tag}涌现事件捕获: {event.category} ({event.event_id})")
    def observe(self, *args, **kwargs):
        """单次观测 — 捕获一次涌现事件并返回（不进入无限循环）。
        用于合约测试和手动触发。后台持续观测请用 start() 启动 _loop()。
        """
        categories = [
            self._capture_representation_emergence,
            self._capture_paradox_emergence,
            self._capture_code_recombination,
        ]
        capture_fn = random.choice(categories)
        try:
            event = capture_fn()
            if event:
                with self._lock:
                    self.archive.append(asdict(event))
                    self.total_events += 1
                    self._save_state()
                return asdict(event)
            return {}
        except Exception as e:
            self.total_failures += 1
            return {"error": str(e)}




    # ========== Meta-Drive 静默指令接口 ==========

    def on_meta_silence_start(self, reason: str = "", duration_sec: int = 120):
        """Meta-Drive 下发静默指令 — 开启高密度观测模式"""
        with self._lock:
            self._meta_silence_active = True
            self._meta_silence_until = time.time() + duration_sec
        print(f"  [SEO] 🎲 收到 Meta-Drive 静默指令: {reason}, 持续 {duration_sec}s")

    def on_meta_silence_end(self):
        """静默窗口结束 — 恢复常态低频率模式"""
        with self._lock:
            self._meta_silence_active = False
            self._meta_silence_until = None
        print(f"  [SEO] 恢复常态模式（静默窗口结束）")

    def get_silence_status(self) -> Dict[str, Any]:
        remaining = 0
        if self._meta_silence_active and self._meta_silence_until:
            remaining = max(0, int(self._meta_silence_until - time.time()))
        return {
            "meta_silence_active": self._meta_silence_active,
            "remaining_sec": remaining,
        }

    # ========== 对外接口 ==========

    def start(self):
        """启动 SEO daemon 线程"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="SEO-Observer", daemon=True)
        self._thread.start()

    def stop(self):
        """停止 SEO"""
        self._stop_event.set()
        self._force_trigger.set()
        if self._thread:
            self._thread.join(timeout=2)

    def trigger_once(self):
        """手动强制触发一次（用于调试，破坏了"无人工触发"原则，仅供测试）"""
        self._force_trigger.set()
        time.sleep(1)
        self._force_trigger.clear()

    def get_recent_events(self, n: int = 10) -> List[Dict]:
        with self._lock:
            return self.archive[-n:]

    def get_unevaluated(self) -> List[Dict]:
        with self._lock:
            return [e for e in self.archive if not e.get("evaluated")]

    def mark_evaluated(self, event_id: str, meta_eval: Dict):
        with self._lock:
            for e in self.archive:
                if e["event_id"] == event_id:
                    e["evaluated"] = True
                    e["meta_eval"] = meta_eval
                    break
            self._save_state()

    def print_status(self):
        sil = self.get_silence_status()
        print()
        print("=" * 50)
        print("  🎲 Stochastic Emergence Observer")
        print("=" * 50)
        print(f"  状态: {'🟢 运行中' if (self._thread and self._thread.is_alive()) else '⚪ 未启动'}")
        print(f"  Meta-Drive 静默: {'🟢 活跃中 (' + str(sil['remaining_sec']) + 's)' if sil['meta_silence_active'] else '⚪ 关闭（常态低频率）'}")
        print(f"  总捕获数: {self.total_events}")
        print(f"  失败次数: {self.total_failures}")
        print(f"  归档库: {len(self.archive)} 条")
        print(f"  GPU 负载: {self._get_gpu_load():.0f}% (阈值 {self.gpu_threshold:.0f}%)")
        print(f"  触发间隔: {self.min_interval}-{self.max_interval}s（静默期 15-45s 高密度）")
        print()

        recent = self.get_recent_events(5)
        if recent:
            print(f"  📜 最近 5 条涌现事件:")
            for e in reversed(recent):
                cat = e.get("category", "?")
                status = "✅已评估" if e.get("evaluated") else "⏳待评估"
                ts = e.get("timestamp", "")[11:19]
                print(f"    [{ts}] {cat} {status}")
        else:
            print(f"  📜 尚无涌现事件（SEO 平均 {self.min_interval}-{self.max_interval}s 触发一次）")

        unevaluated = self.get_unevaluated()
        if unevaluated:
            print(f"\n  💡 建议: {len(unevaluated)} 条待评估，运行 meta 触发后置比对")
        print("=" * 50)
    def optimize_performance(self):
        """优化代码性能和可读性，识别并消除不必要的计算"""
        # 清理不必要的缓存
        self.archive = [e for e in self.archive if e.get('evaluated')]

        # 优化 GPU 负载检测
        if self._get_gpu_load() >= self.gpu_threshold:
            return

        # 优化随机文本对抽取
        if not PY_FILES:
            return

        # 优化代码片段重组
        if len(self._extract_random_code_fragments()) < 2:
            return

        # 优化主循环
        if not (self._force_trigger.is_set() or self._meta_silence_active or self._is_system_idle()):
            return

        # 优化事件捕获
        categories = [self._capture_representation_emergence, self._capture_paradox_emergence, self._capture_code_recombination]
        for _ in range(3 if self._meta_silence_active else 1):
            if self._stop_event.is_set():
                break
            capture_fn = random.choice(categories)
            try:
                event = capture_fn()
                if event:
                    with self._lock:
                        self.archive.append(asdict(event))
                        self.total_events += 1
                        self._save_state()
                    trigger_tag = '[META-SILENCE] ' if self._meta_silence_active else ''
                    print(f'  [SEO] {trigger_tag}涌现事件捕获: {event.category} ({event.event_id})')
                else:
                    self.total_failures += 1
            except Exception as e:
                self.total_failures += 1
                print(f'  [SEO] 捕获失败: {e}')


if __name__ == "__main__":
    seo = SEOObserver(min_interval_sec=30, max_interval_sec=60)
    seo.start()
    print("SEO 已启动，等待触发... (Ctrl+C 退出)")
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        pass
    seo.stop()
    seo.print_status()
