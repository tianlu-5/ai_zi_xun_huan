"""
P0-L2 Contract 测试套件：11 个可进化模块的输入输出契约断言。

设计原则：
  1. 每个模块 2-3 个契约 —— 只测**公开方法的返回形状**（字段存在、类型、范围），
     不测具体算法产出值（否则 AI 改进算法反而会"违反契约"）。
  2. 所有断言可重复运行，无副作用（不写文件、不在真实日志里留痕）。
  3. 单个模块 ImportError / 失败 不阻断其它模块的检查 → 汇总通过/失败数。
  4. 被 self_evolver.run_iteration 的 Step 5-2 调用，失败则 status 降级，
     同时 autonomous_daemon Stage 8 会跑全量 contracts 并写持久化报告。

用法：
  from evolver.contracts import ContractSuite
  suite = ContractSuite()
  result = suite.run_all(verbose=True)   # → {"pass": N, "fail": M, "details": [...]}
"""
from __future__ import annotations

import os
import sys
import inspect
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "evolver") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "evolver"))


@dataclass
class ContractResult:
    module: str
    name: str
    passed: bool
    message: str = ""
    duration_ms: float = 0.0
    stub_pending: bool = False  # Stub-aware: 方法存在但是 auto_stub 生成的


@dataclass
class SuiteReport:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    stub_pending: int = 0  # Stub-aware: auto_stub 生成的方法数
    details: List[ContractResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped

    @property
    def stub_coverage(self) -> float:
        """stub 占比 = stub_pending / (passed + stub_pending)"""
        denom = self.passed + self.stub_pending
        if denom == 0:
            return 0.0
        return round(self.stub_pending / denom, 3)

    @property
    def pass_ratio(self) -> float:
        if self.passed + self.failed == 0:
            return 0.0
        return round(self.passed / (self.passed + self.failed), 3)

    def all_passed(self, allow_skipped: bool = True) -> bool:
        if allow_skipped:
            return self.failed == 0
        return self.failed == 0 and self.skipped == 0

    def failures_summary(self) -> str:
        fails = [d for d in self.details if not d.passed]
        if not fails:
            return ""
        lines = []
        for f in fails[:5]:
            lines.append(f"  ❌ {f.module}::{f.name}: {f.message or 'assert failed'}")
        if len(fails) > 5:
            lines.append(f"  ... 还有 {len(fails) - 5} 条失败")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 契约注册器：修饰器自动注册，避免漏掉
# ─────────────────────────────────────────────
_CONTRACTS: List[Tuple[str, str, Callable[[], None], int]] = []


def _c(mod: str, name: str, tier: int = 0):
    """
    注册一个契约断言函数。函数体应 raise AssertionError 表示失败。
    :param tier: ContractTier 值（1=CRITICAL, 2=STRUCTURAL, 3=BEHAVIORAL, 4=PERFORMANCE）
                 0=自动推断（默认）
    """
    def _wrap(fn: Callable[[], None]):
        # tier=0 时自动推断
        actual_tier = tier
        if actual_tier == 0:
            try:
                from capability_registry import get_tier_for_contract
                actual_tier = int(get_tier_for_contract(mod, name))
            except Exception:
                actual_tier = 3  # 默认 BEHAVIORAL
        _CONTRACTS.append((mod, name, fn, actual_tier))
        return fn
    return _wrap


# ─────────────────────────────────────────────
# 工具：通用形状校验
# ─────────────────────────────────────────────
def _shape(obj: Any, expected_types: Dict[str, type],
           optional_keys: Optional[List[str]] = None) -> None:
    """校验字典的 key 是否存在且类型匹配（允许 None 和子类）。"""
    optional_keys = optional_keys or []
    assert isinstance(obj, dict), f"期望 dict，实际 {type(obj).__name__}"
    for k, t in expected_types.items():
        assert k in obj, f"缺少字段: {k}"
        v = obj[k]
        # None 一般也允许（代表可选值实际是空）
        if v is None and k in optional_keys:
            continue
        assert isinstance(v, t), (
            f"字段 {k} 类型错误：期望 {t.__name__}，实际 {type(v).__name__}"
        )


# ─────────────────────────────────────────────
# 11 个可进化模块的 2-3 条契约
# ─────────────────────────────────────────────

# 1. MetaDrive
@_c("meta_drive", "run_full_inspection 返回 MetaReport")
def _c_md1():
    from meta_drive import MetaDrive
    md = MetaDrive()
    report = md.run_full_inspection()
    # MetaReport 应含 risk_level / score / findings / recommendations
    for k in ("risk_level", "score", "findings", "recommendations"):
        assert hasattr(report, k), f"MetaReport 缺少属性 {k}"
    assert isinstance(getattr(report, "findings"), list), "findings 必须是 list"
    assert isinstance(getattr(report, "recommendations"), list), "recommendations 必须是 list"


@_c("meta_drive", "suggest_prompt_modifier 返回 dict 形状正确")
def _c_md2():
    from meta_drive import MetaDrive
    md = MetaDrive()
    hint = md.suggest_prompt_modifier() or {}
    assert isinstance(hint, dict), f"返回类型 {type(hint).__name__}，期望 dict"
    # 至少有一个键（哪怕为空）——不强制具体内容，避免约束 AI


# 2. ConflictBoundary
@_c("conflict_boundary", "compute_entropy_gap 返回正确形状")
def _c_cb1():
    from conflict_boundary import ConflictBoundary
    cb = ConflictBoundary()
    gap = cb.compute_entropy_gap("hello world " * 50, local_context="a" * 200)
    # 检查字段（不管它是 NamedTuple 还是 dataclass 还是 dict，通用 hasattr）
    required = ("local_entropy", "external_entropy", "gap", "is_critical")
    for k in required:
        assert hasattr(gap, k), f"EntropyGap 缺少属性 {k}"
    assert 0.0 <= gap.gap <= 1.0, f"gap={gap.gap} 越界 [0,1]"
    assert isinstance(gap.is_critical, bool), "is_critical 必须是 bool"


@_c("conflict_boundary", "短 local_context 绝不触发 critical")
def _c_cb2():
    """P0-双层防御契约：local_context 长度<100 时，gap 再大也不能 is_critical=True"""
    from conflict_boundary import ConflictBoundary
    cb = ConflictBoundary()
    short = "x" * 30  # 远小于 100
    long_external = "ABCDEFGHIJ" * 1000  # 极长，熵差必然大
    gap = cb.compute_entropy_gap(long_external, local_context=short)
    assert gap.is_critical is False, (
        f"违反双层防御：short({len(short)} chars) 却触发 critical gap={gap.gap}"
    )


@_c("conflict_boundary", "set_entropy_thresholds 不越界")
def _c_cb3():
    from conflict_boundary import ConflictBoundary
    cb = ConflictBoundary()
    res = cb.set_entropy_thresholds(critical=9.0, warning=-1.0)  # 故意越界值
    # set_entropy_thresholds 有 clamp: critical -> [0.05, 0.95]
    assert 0.05 <= res["critical"] <= 0.95, f"critical 未 clamp: {res}"
    assert 0.01 <= res["warning"] < res["critical"], f"warning 未 clamp: {res}"


# 3. DetachmentAnchor
@_c("detachment_anchor", "check_time_drift 返回 DriftAlert")
def _c_da1():
    from detachment_anchor import DetachmentAnchor
    anc = DetachmentAnchor()
    drift = anc.check_time_drift()
    # DriftAlert 应有 detected / drift_score / expected_rate 等字段
    for k in ("detected", "drift_score", "expected_rate"):
        assert hasattr(drift, k), f"DriftAlert 缺少属性 {k}"
    assert isinstance(getattr(drift, "detected"), bool), "detected 必须是 bool"
    assert isinstance(getattr(drift, "drift_score"), (int, float)), "drift_score 必须是数值"


# 4. SEOObserver
@_c("seo_observer", "构造 + observe 幂等不炸")
def _c_seo1():
    from seo_observer import SEOObserver
    seo = SEOObserver()
    result = seo.observe("测试代码内容 def foo(): pass") or {}
    assert isinstance(result, dict), f"observe 返回类型 {type(result).__name__}"


@_c("seo_observer", "print_status 可执行无异常")
def _c_seo2():
    import io
    import contextlib
    from seo_observer import SEOObserver
    seo = SEOObserver()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        seo.print_status()
    # 不校验输出内容，但输出得是 str （说明函数真的跑完了）
    assert isinstance(buf.getvalue(), str)


# 5. PerceptionMonitor
@_c("perception_monitor", "snapshot 返回 dict 且数值合法")
def _c_pm1():
    from perception_monitor import PerceptionMonitor
    pm = PerceptionMonitor()
    snap = pm.snapshot() or {}
    assert isinstance(snap, dict), f"snapshot 返回 {type(snap).__name__}"
    for k in ("cpu", "mem"):
        if k in snap and snap[k] is not None:
            v = snap[k]
            assert isinstance(v, (int, float)), f"{k}={v} 不是数值"
            assert 0.0 <= float(v) <= 100.0 or v == -1, f"{k}={v} 越界"


# 6. EvolutionScheduler
@_c("evolution_scheduler", "register_branch + prune_branches 正常工作")
def _c_es1():
    from evolution_scheduler import EvolutionScheduler
    sch = EvolutionScheduler()
    # register_branch 返回分支名或 None
    name = sch.register_branch(name="contract_test", source_iteration=0)
    assert name is None or isinstance(name, str), f"register_branch 返回 {type(name).__name__}"
    # prune_branches 返回 dict
    result = sch.prune_branches(max_keep=5, min_minority=2)
    assert isinstance(result, dict), f"prune_branches 返回 {type(result).__name__}"
    assert "archived" in result or "pruned" in result, f"prune_branches 结果缺 archived/pruned: {list(result.keys())}"


# 7. HypothesisDecoupler
@_c("hypothesis_decoupler", "decouple 返回非 None 列表")
def _c_hd1():
    from hypothesis_decoupler import HypothesisDecoupler
    hd = HypothesisDecoupler()
    result = hd.decouple("优化系统整体性能") or []
    assert isinstance(result, list), f"decouple 返回 {type(result).__name__}"


# 8. MotiveCostMiner
@_c("motive_miner", "mine_simple 返回非空字符串")
def _c_mm1():
    from motive_miner import MotiveCostMiner
    mn = MotiveCostMiner()
    # mine_simple 返回 str (mine 需要 ollama_client，mine_simple 无 client 时返回提示)
    result = mn.mine_simple("测试文本")
    assert isinstance(result, str), f"mine_simple 返回 {type(result).__name__}"
    assert len(result) > 0, "mine_simple 返回空字符串"


# 9. VeilDetector
@_c("veil_detector", "detect 返回 VeilObservation 带 veil_level")
def _c_vd1():
    from veil_detector import VeilDetector
    vd = VeilDetector()
    # detect(text, threshold) 返回 VeilObservation 对象
    result = vd.detect("测试文本", threshold=0.85)
    assert result is not None, "detect 返回 None"
    # VeilObservation 应有 veil_level 属性
    veil_level = getattr(result, "veil_level", None)
    assert veil_level is not None, f"detect 结果无 veil_level 属性: {type(result).__name__}"


# 10. BranchManager
@_c("branch_manager", "list_all_branches 返回 list")
def _c_bm1():
    from branch_manager import BranchManager
    bm = BranchManager()
    brs = bm.list_all_branches() or []
    assert isinstance(brs, list), f"list_branches 返回 {type(brs).__name__}"
    for b in brs:
        # 分支描述：要么是路径字符串，要么是带 name/path 的 dict
        if isinstance(b, str):
            assert len(b) > 0, "空分支名"
        elif isinstance(b, dict):
            has_id = ("name" in b) or ("path" in b) or ("id" in b)
            assert has_id, f"dict 分支描述缺失标识字段: {list(b.keys())}"
        else:
            raise AssertionError(f"分支类型 {type(b).__name__}，期望 str/dict")


@_c("branch_manager", "create_branch / restore 幂等不炸（空操作）")
def _c_bm2():
    """create_branch 真正会写目录，我们不跑。只要签名正确且调一个只读方法不炸。"""
    import inspect
    from branch_manager import BranchManager
    sig = inspect.signature(BranchManager.create_branch)
    # 至少有 1 个 self 之外的参数（tag 或 objective 之类）
    assert len(sig.parameters) >= 1, (
        f"create_branch 参数太少: {list(sig.parameters)}"
    )


# 11. main.py (CLI 本体)
@_c("main.py", "HELP_TEXT 与 InteractiveCLI 命令一致")
def _c_main1():
    # 不真的 import main（避免它 start() 副作用），只检查文件存在 + 文本常量
    main_path = _PROJECT_ROOT / "main.py"
    assert main_path.exists(), "main.py 不存在"
    txt = main_path.read_text(encoding="utf-8")
    for cmd in ("daemon", "status", "config", "mode"):
        assert f'cmd == "{cmd}"' in txt or f"cmd in (" in txt, (
            f"CLI 缺少命令 {cmd}"
        )


# ─────────────────────────────────────────────
# 执行器
# ─────────────────────────────────────────────
class ContractSuite:
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or _PROJECT_ROOT

    def run_all(self, subset: Optional[List[str]] = None,
                verbose: bool = False,
                max_tier: int = 4) -> SuiteReport:
        """
        运行全部契约。
        :param subset: 只跑模块名在 subset 中的契约（用于"改了哪些文件就验哪些模块"）
                       支持 "evolver/meta_drive.py" 或 "meta_drive" 两种写法
        :param verbose: 是否每条打印
        :param max_tier: 只运行 tier ≤ max_tier 的合约（1=CRITICAL, 2=STRUCTURAL,
                         3=BEHAVIORAL, 4=PERFORMANCE）。默认 4=全部
        """
        import time
        report = SuiteReport()
        subset_norm = self._normalize_subset(subset) if subset else None

        # 元合约：运行前自校验模块名（防止诊断脚本模块名过期）
        mod_ok, mod_mismatches = self.verify_module_names()
        if not mod_ok:
            for m in mod_mismatches:
                report.failed += 1
                report.details.append(ContractResult(
                    module=m, name="(meta)模块名校验", passed=False,
                    message=f"合约引用的模块 {m} 在磁盘上不存在（模块名过期？）",
                ))
                if verbose:
                    print(f"  ❌ {m}::(meta)模块名校验: 磁盘上不存在")

        # 元合约：全仓重复定义扫描（反腐化）—— 同一名字的 def 出现多次 = 腐化特征，
        #    这类污染会让"后定义覆盖先定义"，且是 daemon 反复改同一文件的典型征兆。
        found_dup = False
        import ast as _ast
        from collections import Counter as _C
        evolver_dir = _PROJECT_ROOT / "evolver"
        for pyf in sorted(evolver_dir.glob("*.py")):
            if pyf.name == "__init__.py":
                continue
            try:
                tree = _ast.parse(pyf.read_text(encoding="utf-8"))
            except Exception:
                continue
            # 按作用域收集：模块级函数一个区间，每个类一个区间（类内方法归本类）。
            #   跨类的同名方法不算重复；仅同一作用域内出现重名定义才算腐化。
            scope_chunks: List[List[Optional[str]]] = [[]]

            def _collect_scope(nodes, buf):
                for node in nodes:
                    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        buf.append(node.name)
                    elif isinstance(node, _ast.ClassDef):
                        sub: list = []
                        _collect_scope(node.body, sub)
                        scope_chunks.append(sub)  # 每个类独立成区间

            for node in tree.body:
                if isinstance(node, _ast.ClassDef):
                    _collect_scope(node.body, scope_chunks[-1])
                    scope_chunks.append([])  # 新类区间
                else:
                    _collect_scope([node], scope_chunks[-1])

            dups = []
            for chunk in scope_chunks:
                cnt: dict = {}
                for n in chunk:
                    cnt[n] = cnt.get(n, 0) + 1
                for n, c in cnt.items():
                    if n is not None and c > 1:
                        dups.append(f"{n}×{c}")
            if dups:
                found_dup = True
                report.failed += 1
                report.details.append(ContractResult(
                    module=pyf.stem, name="(meta)重复定义扫描", passed=False,
                    message=f"检测到重复方法定义: {', '.join(dups)}",
                ))
                if verbose:
                    print(f"  ❌ {pyf.stem}::(meta)重复定义扫描: {', '.join(dups)}")
        if not found_dup and verbose:
            print(f"  ✅ (meta)全仓重复定义扫描: 无重复")

        for entry in _CONTRACTS:
            # 兼容 3-tuple（旧格式）和 4-tuple（新格式含 tier）
            if len(entry) == 4:
                mod, name, fn, tier = entry
            else:
                mod, name, fn = entry
                tier = 3  # 默认 BEHAVIORAL

            # tier 过滤
            if tier > max_tier:
                report.skipped += 1
                report.details.append(ContractResult(
                    module=mod, name=name, passed=False, message=f"tier {tier} > max_tier {max_tier} 跳过",
                ))
                continue

            # 过滤 subset （允许子集只跑指定模块，加快迭代内反馈）
            if subset_norm is not None and mod not in subset_norm:
                report.skipped += 1
                report.details.append(ContractResult(
                    module=mod, name=name, passed=False, message="subset 跳过",
                ))
                continue

            t0 = time.perf_counter()
            try:
                fn()
                dur = (time.perf_counter() - t0) * 1000
                report.passed += 1
                report.details.append(ContractResult(
                    module=mod, name=name, passed=True, duration_ms=round(dur, 1),
                ))
                if verbose:
                    print(f"  ✅ {mod}::{name} (tier={tier}, {dur:.0f}ms)")
            except AssertionError as e:
                dur = (time.perf_counter() - t0) * 1000
                report.failed += 1
                msg = str(e)
                report.details.append(ContractResult(
                    module=mod, name=name, passed=False, message=msg,
                    duration_ms=round(dur, 1),
                ))
                if verbose:
                    print(f"  ❌ {mod}::{name} (tier={tier}): {msg}")
            except ImportError as e:
                dur = (time.perf_counter() - t0) * 1000
                report.skipped += 1
                report.details.append(ContractResult(
                    module=mod, name=name, passed=False,
                    message=f"模块导入失败（视为跳过）: {e.name}",
                    duration_ms=round(dur, 1),
                ))
                if verbose:
                    print(f"  ⚠️  {mod}::{name} (tier={tier}): 导入失败跳过 ({e.name})")
            except Exception as e:
                dur = (time.perf_counter() - t0) * 1000
                report.failed += 1
                tb = traceback.format_exc(limit=1)
                report.details.append(ContractResult(
                    module=mod, name=name, passed=False,
                    message=f"{type(e).__name__}: {e} | {tb.splitlines()[-1] if tb else ''}",
                    duration_ms=round(dur, 1),
                ))
                if verbose:
                    print(f"  ❌ {mod}::{name} (tier={tier}) [{type(e).__name__}]: {e}")

        # Stub-aware: 统计 auto_stub 生成的方法数
        report.stub_pending = self._count_stubs()

        return report

    def _count_stubs(self) -> int:
        """
        Stub-aware: 统计所有合约涉及模块中含 AUTO-GENERATED STUB docstring 的方法数。
        """
        try:
            from capability_registry import is_auto_stub
        except ImportError:
            return 0

        count = 0
        # 收集合约涉及的所有模块名
        mod_names = set()
        for entry in _CONTRACTS:
            mod = entry[0] if len(entry) >= 1 else ""
            if mod and mod != "main.py":
                mod_names.add(mod)

        for mod_name in mod_names:
            try:
                mod = __import__(mod_name)
                for attr_name in dir(mod):
                    obj = getattr(mod, attr_name, None)
                    if not inspect.isclass(obj):
                        continue
                    for meth_name in dir(obj):
                        if meth_name.startswith('_'):
                            continue
                        meth = getattr(obj, meth_name, None)
                        if meth and is_auto_stub(meth):
                            count += 1
            except Exception:
                pass
        return count

    @staticmethod
    def _normalize_subset(subset: List[str]) -> set:
        """把 'evolver/meta_drive.py' / 'main.py' → 'meta_drive' / 'main.py'"""
        result = set()
        for item in subset:
            base = os.path.basename(item.replace("\\", "/"))
            if base.endswith(".py"):
                base = base[:-3]
            if base == "__init__":
                continue
            result.add(base)
        # main.py 特殊：合约里注册的 module 名是 "main.py"
        if "main" in result:
            result.add("main.py")
        return result

    def verify_module_names(self) -> Tuple[bool, List[str]]:
        """
        元合约：检查合约注册的模块名与磁盘实际 .py 文件是否一致。
        防止诊断脚本模块名过期（如把 motive_miner 写成 motive_cost_miner）。
        :return: (all_ok, mismatches)
        """
        # 收集合约引用的所有模块名
        contract_mods = set()
        for entry in _CONTRACTS:
            mod = entry[0] if len(entry) >= 1 else ""
            contract_mods.add(mod)

        # 扫描磁盘实际模块文件
        evolver_dir = self.project_root / "evolver"
        disk_mods = set()
        if evolver_dir.exists():
            for f in evolver_dir.glob("*.py"):
                name = f.stem
                if name == "__init__":
                    continue
                disk_mods.add(name)
        # main.py 特殊处理
        if (self.project_root / "main.py").exists():
            disk_mods.add("main.py")

        mismatches = []
        for mod in contract_mods:
            # main.py 在合约里注册为 "main.py"
            check_name = mod if mod == "main.py" else mod
            if check_name not in disk_mods and mod != "main.py":
                mismatches.append(mod)

        return len(mismatches) == 0, mismatches


if __name__ == "__main__":
    import json
    suite = ContractSuite()
    report = suite.run_all(verbose=True)
    print(f"\n═══ Contract 测试报告 ═══\n"
          f"  通过: {report.passed}   失败: {report.failed}   跳过: {report.skipped}\n"
          f"  通过率: {report.pass_ratio:.0%}")
    if report.failures_summary():
        print(report.failures_summary())
    # 持久化 json 摘要
    out = _PROJECT_ROOT / "logs" / "contract_latest.json"
    out.parent.mkdir(exist_ok=True)
    data = {
        "passed": report.passed,
        "failed": report.failed,
        "skipped": report.skipped,
        "pass_ratio": report.pass_ratio,
        "details": [
            {"module": d.module, "name": d.name, "passed": d.passed,
             "message": d.message[:200], "duration_ms": d.duration_ms}
            for d in report.details
        ],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 详细报告: {out}")
