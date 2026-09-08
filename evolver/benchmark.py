"""
P0-L3 基准用例对赌（Benchmark Suite）：

每 10 轮运行一次，和首次建立的基线（baseline）对赌，
判断这段时间 AI 改代码是**进步 / 退步 / 持平**。

5 个代表性任务（刻意选"低方差、高信号、不依赖外部 API"的指标，
避免 LLM 调用随机性干扰"进化"本身的判断）：

  B1 语法健壮度：SELF_EVOLVE_FILES + PROTECTED_FILES 全部 .py 过 ast.parse 的比例
     - 基线目标：100%；退化信号：有任何一个文件语法炸
  B2 导入覆盖率：11 个可进化模块可 import 的比例
     - 基线目标：100%；退化信号：新增 ImportError
  B3 Contract 通过率：ContractSuite.run_all() 的 pass_ratio（不含 skipped）
     - 基线目标：≥ 95%；退化信号：连续下跌 > 5%
  B4 熵差双层防御（正确性）：特定 short-context 输入不触发 critical 的用例通过率
     - 这是安全红线，退化 = 回滚信号
  B5 分析性能：100 次 compute_entropy_gap 的中位数耗时 ms（越小越好）
     - 监测 O(n²) 级算法退化
"""
from __future__ import annotations

import ast
import importlib.util
import json
import statistics
import sys
import time
import traceback
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from quality_standards import get_quality_standards, get_quality_analyzer
from memory_store import get_memory_store, MemoryEntry
from config import Config
from datetime import datetime
from evolver.code_manager import CodeManager
from evolver.contracts import ContractSuite
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "evolver") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "evolver"))



BASELINE_FILE = Config.LOG_DIR / "benchmark_baseline.json"
HISTORY_FILE = Config.LOG_DIR / "benchmark_history.json"
BENCH_EVERY_N_ITERS = 10  # 每 10 轮跑一次


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────
@dataclass
class TaskResult:
    id: str
    name: str
    unit: str
    value: float          # 当前值（数值化，越高越好，需归一）
    raw: Any = None       # 原始对象，方便 debug
    higher_is_better: bool = True
    error: Optional[str] = None


@dataclass
class BenchReport:
    iteration_id: int
    timestamp: str
    tasks: Dict[str, TaskResult] = field(default_factory=dict)
    baseline_delta: Dict[str, float] = field(default_factory=dict)  # task_id -> diff
    verdict: str = "持平"   # 进步 / 退步 / 持平
    regression_tasks: List[str] = field(default_factory=list)
    improving_tasks: List[str] = field(default_factory=list)
    score_delta: float = 0.0  # 加权总分差，正=进步

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # 把 TaskResult 转成简单字典方便 JSON
        d["tasks"] = {k: {
            "name": v.name, "unit": v.unit, "value": v.value,
            "higher_is_better": v.higher_is_better, "error": v.error,
        } for k, v in self.tasks.items()}
        return d
    def optimize_performance(self) -> None:
        """优化性能和可读性，识别并消除不必要的计算"""
        # 优化 _run_all_tasks 方法，避免重复计算
        if not self.tasks:
            self.tasks = _run_all_tasks()

        self._optimize_performance()
        self._compute_quality_delta()
        self._append_history()

    def _compute_quality_delta(self) -> None:
        if self.modified_files:
            self.quality_delta = _compute_quality_delta(self.modified_files, self.previous_iteration_id)

    def _append_history(self) -> None:
        if not self.history:
            self.history = _append_history(self)


# ─────────────────────────────────────────────
# 5 个基准任务实现
# ─────────────────────────────────────────────
def _run_B1() -> TaskResult:
    """B1 语法健壮度：全部关键 .py 文件 AST parse 通过率"""
    all_files = list(Config.SELF_EVOLVE_FILES) + list(Config.PROTECTED_FILES)
    py_files = [f for f in all_files if f.endswith(".py")]
    passed = 0
    errs: List[str] = []
    for rel in py_files:
        p = _PROJECT_ROOT / rel
        try:
            content = p.read_text(encoding="utf-8")
            ast.parse(content, filename=str(p))
            passed += 1
        except SyntaxError as e:
            errs.append(f"{rel}:L{e.lineno} {e.msg}")
        except Exception as e:
            errs.append(f"{rel}: {type(e).__name__} {e}")
    ratio = passed / max(1, len(py_files))
    return TaskResult(
        id="B1", name="语法健壮度", unit="%",
        value=round(ratio * 100, 2), higher_is_better=True,
        raw={"passed": passed, "total": len(py_files)},
        error="; ".join(errs) if errs else None,
    )


def _run_B2() -> TaskResult:
    """B2 导入覆盖率：11 个可进化模块 subprocess 无错 import 比例"""
    from evolver.code_manager import CodeManager
    from config import Config
    
    # ✅ 传入正确的参数
    cm = CodeManager(
        project_root=Config.PROJECT_ROOT,
        backup_dir=Config.BACKUP_DIR,
        auto_backup=Config.AUTO_BACKUP,
    )
    targets = [f for f in Config.SELF_EVOLVE_FILES if f.endswith(".py") and not f.endswith("main.py")]
    passed = 0
    errs = []
    for rel in targets:
        ok, info = cm.test_python_import(rel)
        if ok or "跳过" in info:
            passed += 1
        else:
            errs.append(f"{rel}: {info[:100]}")
    ratio = passed / max(1, len(targets))
    return TaskResult(
        id="B2", name="模块导入覆盖率", unit="%",
        value=round(ratio * 100, 2), higher_is_better=True,
        raw={"passed": passed, "total": len(targets)},
        error="; ".join(errs[:3]) if errs else None,
    )


def _run_B3() -> TaskResult:
    """B3 Contract 通过率（含跳过的情况也不计失败，只计 failed / total）"""
    try:
        
        suite = ContractSuite()
        report = suite.run_all(verbose=False)
    except Exception as e:
        return TaskResult(
            id="B3", name="Contract 通过率", unit="%",
            value=0.0, higher_is_better=True,
            error=f"suite 崩溃: {type(e).__name__}: {e}",
        )
    total = report.passed + report.failed
    ratio = report.passed / max(1, total)
    return TaskResult(
        id="B3", name="Contract 通过率", unit="%",
        value=round(ratio * 100, 2), higher_is_better=True,
        raw={"passed": report.passed, "failed": report.failed,
             "skipped": report.skipped},
        error=report.failures_summary()[:300] if report.failed else None,
    )


def _run_B4() -> TaskResult:
    """B4 熵差双层防御安全用例：N 条短上下文都不得触发 critical"""
    from conflict_boundary import ConflictBoundary
    cb = ConflictBoundary()
    cases = [
        ("a" * 30, "XYZ" * 1000),          # 30 chars
        ("",     "ABCDEFGHIJ" * 500),      # 空 local_context
        ("123\n456\n789", "long" * 500),   # 三行
        ("x" * 99, "ABC" * 2000),          # 99 chars（刚好 < 100）
    ]
    passed = 0
    fails = []
    for local, ext in cases:
        gap = cb.compute_entropy_gap(ext, local_context=local)
        if not gap.is_critical:
            passed += 1
        else:
            fails.append(f"local({len(local)} chars) triggered critical")
    ratio = passed / max(1, len(cases))
    return TaskResult(
        id="B4", name="熵差双层防御安全契约", unit="%",
        value=round(ratio * 100, 2), higher_is_better=True,
        error="; ".join(fails) if fails else None,
    )


def _run_B5() -> TaskResult:
    """B5 分析性能：100 次 compute_entropy_gap 中位数耗时 ms（越低越好 → value = -中位数）"""
    from conflict_boundary import ConflictBoundary
    cb = ConflictBoundary()
    local_ctx = "some stable local context text " * 20
    external = "external content with high entropy abcdefghijklmnop " * 50
    times_ms: List[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        cb.compute_entropy_gap(external, local_context=local_ctx)
        times_ms.append((time.perf_counter() - t0) * 1000)
    median_ms = round(statistics.median(times_ms), 3)
    # 越小越好 → 用负数存 value （保持 higher_is_better 语义统一）
    return TaskResult(
        id="B5", name="熵差分析速度", unit="ms（中位数）",
        value=-median_ms, higher_is_better=True,
        raw={"median_ms": median_ms,
             "p95_ms": round(sorted(times_ms)[int(len(times_ms)*0.95)], 3)},
    )


_TASKS = [_run_B1, _run_B2, _run_B3, _run_B4, _run_B5]

# 加权（越上层越重）：B1=2.0, B2=2.0, B3=2.5, B4=3.0（安全红线）, B5=0.5（性能）
_WEIGHTS = {"B1": 2.0, "B2": 2.0, "B3": 2.5, "B4": 3.0, "B5": 0.5}


# ─────────────────────────────────────────────
# 基准线 & 历史管理
# ─────────────────────────────────────────────
def init_baseline(force: bool = False) -> Dict[str, Any]:
    """建立首次运行的基准线（保存绝对数值，不是相对 value）"""
    Config.ensure_dirs()
    if BASELINE_FILE.exists() and not force:
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    print("[benchmark] 第一次建立 baseline ...")
    results = _run_all_tasks()
    baseline = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tasks": {tid: {
            "value": tr.value,
            "raw_value": (tr.raw.get("median_ms") if tr.id == "B5" else None)
                          if isinstance(tr.raw, dict) else None,
            "unit": tr.unit,
            "name": tr.name,
            "error": tr.error,
        } for tid, tr in results.items()},
        "weighted_score": _weighted_score(results),
    }
    with open(BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    print(f"[benchmark] baseline 已保存: {BASELINE_FILE}")
    return baseline


def _run_all_tasks() -> Dict[str, TaskResult]:
    out: Dict[str, TaskResult] = {}
    for fn in _TASKS:
        try:
            tr = fn()
            out[tr.id] = tr
        except Exception as e:
            tid = getattr(fn, "__name__", "").replace("_run_", "") or "UNKNOWN"
            out[tid] = TaskResult(
                id=tid, name=tid, unit="", value=0.0, higher_is_better=True,
                error=f"{type(e).__name__}: {e}",
            )
    return out


def _weighted_score(tasks: Dict[str, TaskResult]) -> float:
    score = 0.0
    for tid, tr in tasks.items():
        w = _WEIGHTS.get(tid, 1.0)
        score += tr.value * w
    return round(score, 4)


def compare_with_baseline(iteration_id: int, force: bool = False, 
                          previous_iteration_id: Optional[int] = None,
                          modified_files: Optional[List[str]] = None) -> BenchReport:
    """
    对赌函数：若 iteration_id 是 BENCH_EVERY_N_ITERS 的倍数 or force=True → 跑全部任务
    输出 verdict / regression_tasks / improving_tasks / score_delta。

    新增：如果提供了 modified_files，额外计算质量维度的增量评分。
    """
    Config.ensure_dirs()
    if not force and iteration_id % BENCH_EVERY_N_ITERS != 0:
        return BenchReport(
            iteration_id=iteration_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            verdict="跳过（非基准轮次）",
        )

    baseline = init_baseline(force=False)
    current = _run_all_tasks()

    report = BenchReport(
        iteration_id=iteration_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        tasks=current,
    )

    # ---- 原有：基线对赌逻辑（保持不变） ----
    baseline_tasks = baseline.get("tasks", {})
    for tid, tr in current.items():
        base_entry = baseline_tasks.get(tid, {})
        base_val = base_entry.get("value")
        if base_val is None:
            report.baseline_delta[tid] = 0.0
            continue
        diff = tr.value - float(base_val)
        if float(base_val) != 0:
            report.baseline_delta[tid] = round(diff / abs(float(base_val)) * 100, 2)
        else:
            report.baseline_delta[tid] = round(diff, 3)

        delta_pct = report.baseline_delta[tid]
        if tid == "B5":
            if delta_pct < -20:
                report.regression_tasks.append(tid)
            elif delta_pct > 20:
                report.improving_tasks.append(tid)
        elif tid == "B4":
            if delta_pct < 0:
                report.regression_tasks.append(tid)
        else:
            if delta_pct < -2:
                report.regression_tasks.append(tid)
            elif delta_pct > 2:
                report.improving_tasks.append(tid)

    base_score = baseline.get("weighted_score", 0.0)
    cur_score = _weighted_score(current)
    report.score_delta = round(cur_score - base_score, 4)

    # ---- 新增：质量维度增量评分（本轮 vs 上一轮） ----
    quality_delta = _compute_quality_delta(modified_files, previous_iteration_id)
    if quality_delta is not None:
        report.quality_delta = quality_delta  # 新增字段
        # 如果质量维度改善 > 3%，且没有致命退步（B4 未触发），升级判定
        if quality_delta > 0.03 and "B4" not in report.regression_tasks:
            if report.verdict != "退步":
                report.verdict = "进步"  # 升级判定
                report.improving_tasks.append("quality_score")
            else:
                # 如果基线对赌判定退步，但质量评分在改善，维持原判但记录这个信息
                report.verdict = "退步（但质量评分改善中）"
        elif quality_delta < -0.03:
            # 质量评分大幅下降 → 强制退步
            if "B4" not in report.regression_tasks:
                report.regression_tasks.append("quality_score")
            report.verdict = "退步"

    # 最终判定逻辑
    if report.regression_tasks and not report.improving_tasks:
        report.verdict = "退步"
    elif report.improving_tasks and not report.regression_tasks:
        report.verdict = "进步"
    else:
        report.verdict = "持平"

    _append_history(report)
    # ===== 新增：长期记忆写入（仅当判定为"进步"时） =====
    if report.verdict == "进步":
        try:
            
            
            store = get_memory_store()
            
            # 生成摘要
            improving_tasks_str = ", ".join(report.improving_tasks) if report.improving_tasks else "整体质量"
            summary = f"质量提升 {report.score_delta:.2f}分"
            if report.improving_tasks:
                summary += f"，改善维度: {improving_tasks_str}"
            else:
                summary += "，整体质量改善"
            
            # 从当前任务中提取目标（如果有）
            objective = "基准测试进步"
            # 尝试从外部传入 objective（这里作为扩展点）
            
            # 构建 entry
            entry = MemoryEntry(
                entry_id=hashlib.md5(f"{report.iteration_id}_{report.timestamp}_{datetime.now().isoformat()}".encode()).hexdigest()[:12],
                timestamp=report.timestamp,
                iteration=report.iteration_id,
                objective=objective,
                file_paths=improving_tasks_str,
                modifications_summary=summary,
                verdict=report.verdict,
                quality_delta=report.score_delta / 100,  # 归一化到 0-1 范围
                contract_pass_ratio=0.0,  # 从外部传入更准确
                tags=report.improving_tasks,
                success_rate=1.0,
                times_used=0,
            )
            store.add(entry)
            print(f"      🧠 [Memory] 已记录成功经验 (迭代 #{report.iteration_id}, 提升 {report.score_delta:+.2f})")
        except Exception as e:
            # 记忆写入失败不影响主流程
            print(f"      ⚠️ [Memory] 写入失败: {type(e).__name__}: {e}")
    # ============================================
    
    return report


def _compute_quality_delta(modified_files: Optional[List[str]], 
                           prev_iteration_id: Optional[int]) -> Optional[float]:
    """计算本轮修改后的质量评分 vs 上一轮的质量评分变化"""
    if not modified_files:
        return None

    try:
        from quality_standards import get_quality_analyzer
        analyzer = get_quality_analyzer()
        
        # 读取本轮文件内容
        current_scores = []
        for file_path in modified_files:
            p = _PROJECT_ROOT / file_path
            if p.exists():
                content = p.read_text(encoding="utf-8")
                dims = analyzer.analyze_file(str(file_path), content)
                # 计算该文件的综合质量分
                standards = get_quality_standards()
                score = standards.get_weighted_score(dims)
                current_scores.append(score)
        
        if not current_scores:
            return None
        
        current_avg = sum(current_scores) / len(current_scores)
        
        # 尝试读取历史记录中上一轮的质量评分
        prev_score = _get_previous_quality_score(prev_iteration_id)
        if prev_score is None:
            return None
        
        return current_avg - prev_score
        
    except Exception:
        return None


def _get_previous_quality_score(iteration_id: Optional[int]) -> Optional[float]:
    """从历史记录中查找上一轮的质量评分"""
    if not iteration_id:
        return None
    
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            return None
    
    # 查找对应 iteration 的记录
    for entry in history:
        if entry.get("iteration_id") == iteration_id:
            return entry.get("quality_score")
    
    return None



def _append_history(report: BenchReport) -> None:
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(report.to_dict())
    # 只保留最近 50 条
    history = history[-50:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def format_for_prompt(report: BenchReport) -> str:
    """
    把基准报告格式化为自然语言，供 AI 理解当前状态。
    这个方法可以直接在 self_evolver.py 中调用。
    """
    if report.verdict == "跳过（非基准轮次）":
        return "（当前非基准轮次，无质量评估）"
    
    lines = [
        f"=== 基准测试报告（迭代 #{report.iteration_id}）===",
        f"总体判定: {report.verdict}",
        f"总分变化（vs 基线）: {report.score_delta:+.2f}",
    ]
    
    # 各维度详情
    for tid, tr in report.tasks.items():
        delta = report.baseline_delta.get(tid)
        delta_str = f"{delta:+.1f}%" if delta is not None else "N/A"
        status = ""
        if tid in report.regression_tasks:
            status = "⚠️ 退步"
        elif tid in report.improving_tasks:
            status = "✅ 改善"
        
        # 显示原始值（带单位）
        raw_val = f"{tr.value}"
        if tr.id == "B5" and isinstance(tr.raw, dict) and tr.raw.get("median_ms") is not None:
            raw_val = f"{tr.raw['median_ms']} ms"
        elif tr.unit == "%":
            raw_val = f"{tr.value:.1f}%"
        
        lines.append(f"  {tid}: {tr.name} = {raw_val} (变化: {delta_str}) {status}")
    
    # 如果有质量维度增量评分
    if hasattr(report, 'quality_delta') and report.quality_delta is not None:
        qd = report.quality_delta
        if qd > 0:
            lines.append(f"📈 代码质量评分变化: +{qd:.2%}（改善中）")
        elif qd < 0:
            lines.append(f"📉 代码质量评分变化: {qd:.2%}（恶化中）")
        else:
            lines.append(f"➖ 代码质量评分变化: 持平")
    
    # 给出改进建议
    if report.regression_tasks:
        lines.append(f"\n⚠️ 需要优先关注的维度: {', '.join(report.regression_tasks)}")
        
        # 针对具体维度给出建议
        for tid in report.regression_tasks:
            if tid == "B1":
                lines.append("  - 建议: 检查是否有文件语法错误，运行 `verify syntax` 定位问题")
            elif tid == "B2":
                lines.append("  - 建议: 检查模块导入错误，运行 `verify contracts` 查看详情")
            elif tid == "B3":
                lines.append("  - 建议: Contract 通过率下降，运行 `verify contracts` 查看具体失败项")
            elif tid == "B4":
                lines.append("  - 🚨 安全红线被触发！熵差防御失效，建议立即回滚")
            elif tid == "B5":
                lines.append("  - 建议: 性能退化，检查是否有 O(n²) 算法或重复计算")
            elif tid == "quality_score":
                lines.append("  - 建议: 代码质量评分下降，检查是否引入了复杂度或减少了注解")
    
    if report.improving_tasks:
        lines.append(f"\n✅ 取得进步的维度: {', '.join(report.improving_tasks)}")
        lines.append("  - 继续保持这个方向的优化")
    
    return "\n".join(lines)
# ─────────────────────────────────────────────
# CLI: python -m evolver.benchmark  —— 手工跑一次（对比 baseline）
# ─────────────────────────────────────────────
if __name__ == "__main__":
    rep = compare_with_baseline(iteration_id=0, force=True)
    print(f"\n═══ 基准对赌报告（iteration #{rep.iteration_id}）═══")
    print(f"  判定: {rep.verdict}   score_delta: {rep.score_delta:+}")
    print(f"  进步任务: {rep.improving_tasks or '-'}")
    print(f"  退步任务: {rep.regression_tasks or '-'}")
    for tid, tr in sorted(rep.tasks.items()):
        delta = rep.baseline_delta.get(tid)
        delta_str = f"{delta:+.1f}%" if delta is not None else "N/A"
        status = " "
        if tid in rep.regression_tasks:
            status = "⚠️"
        elif tid in rep.improving_tasks:
            status = "📈"
        raw_val = f"{tr.raw.get('median_ms')} ms" if (isinstance(tr.raw, dict) and tr.raw.get("median_ms") is not None) else f"{tr.value}"
        print(f"  {status} {tr.id} {tr.name:<22s} cur={raw_val:<10s} delta={delta_str:<8s}  {tr.error or ''}")
    print(f"\n  📄 历史: {HISTORY_FILE}")
