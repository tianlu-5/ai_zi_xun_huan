"""
迭代数据统计脚本 — 快速查看运行瓶颈，无需翻看海量日志。
用法: python iter_stats.py
读取: logs/daemon_log.json + logs/evolution_history.json + logs/json_fail_log/ + logs/l3_snapshots.json
"""
import json
import sys
from pathlib import Path
from collections import Counter

LOG_DIR = Path("logs")


def _load_json(path: Path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"  ⚠️ 加载 {path.name} 失败: {e}")
    return default


def _load_jsonl(path: Path):
    """逐行读取 JSONL 格式（l3_snapshots.json 用追加模式写入）"""
    items = []
    if not path.exists():
        return items
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return items


def main():
    print("=" * 60)
    print("  📊 迭代数据统计报告")
    print("=" * 60)

    # ── 1. daemon_log.json: 守护进程周期历史 ──
    daemon_log = _load_json(LOG_DIR / "daemon_log.json", [])
    if isinstance(daemon_log, dict):
        daemon_log = daemon_log.get("history", [])
    total_cycles = len(daemon_log) if isinstance(daemon_log, list) else 0

    # ── 2. evolution_history.json: 自迭代历史 ──
    evo_history = _load_json(LOG_DIR / "evolution_history.json", [])
    if isinstance(evo_history, dict):
        evo_history = evo_history.get("history", evo_history.get("iterations", []))
    total_iters = len(evo_history) if isinstance(evo_history, list) else 0

    print(f"\n  🔄 总迭代轮数: daemon={total_cycles}  evolver={total_iters}")

    # ── 3. JSON 解析成功率 ──
    json_fail_count = 0
    fail_dir = LOG_DIR / "json_fail_log"
    if fail_dir.exists():
        json_fail_count = sum(1 for _ in fail_dir.glob("fail_*.json"))

    # 从 evolution_history 推断 JSON 成功数
    json_success = 0
    json_empty = 0
    for item in evo_history if isinstance(evo_history, list) else []:
        if not isinstance(item, dict):
            continue
        suggestion_count = item.get("suggestion_count", -1)
        if suggestion_count > 0:
            json_success += 1
        elif suggestion_count == 0:
            json_empty += 1
        # status 含 "API错误"/"API响应异常" 也算 JSON 失败
        status = str(item.get("status", ""))
        if "API" in status:
            json_fail_count += 1

    total_attempts = json_success + json_fail_count + json_empty
    success_rate = (json_success / total_attempts * 100) if total_attempts > 0 else 0
    print(f"\n  📝 JSON 解析统计:")
    print(f"     成功: {json_success}  失败: {json_fail_count}  空计划: {json_empty}")
    print(f"     成功率: {success_rate:.1f}% (目标 ≥60%)")

    # ── 4. 合约通过/失败 ──
    contract_pass = 0
    contract_fail = 0
    contract_fail_files = Counter()
    for item in evo_history if isinstance(evo_history, list) else []:
        if not isinstance(item, dict):
            continue
        c = item.get("contract", {})
        if not isinstance(c, dict):
            continue
        if c.get("failed", 0) > 0:
            contract_fail += 1
            # 记录失败的文件
            for issue in item.get("issues", []):
                if "Contract" in str(issue):
                    contract_fail_files[issue[:80]] += 1
        elif c.get("passed", 0) > 0 and c.get("failed", 0) == 0:
            contract_pass += 1

    print(f"\n  🔒 合约校验统计:")
    print(f"     通过: {contract_pass}  失败: {contract_fail}")
    if contract_fail_files:
        print(f"     常见失败项:")
        for item, cnt in contract_fail_files.most_common(3):
            print(f"       {cnt}x  {item}")

    # ── 5. veil_detector 触发次数 ──
    veil_triggers = 0
    veil_high = 0
    for cycle in daemon_log if isinstance(daemon_log, list) else []:
        if not isinstance(cycle, dict):
            continue
        veil = cycle.get("stages", {}).get("veil_check", {})
        if isinstance(veil, dict):
            level = veil.get("veil_level", 0)
            if level and level > 0:
                veil_triggers += 1
                if level >= 2:
                    veil_high += 1

    print(f"\n  🎭 VeilDetector 统计:")
    print(f"     触发次数: {veil_triggers}/{total_cycles}  高级别(≥2): {veil_high}")

    # ── 6. L3 失败模式识别 ──
    l3_snaps = _load_jsonl(LOG_DIR / "l3_snapshots.json")
    l3_patterns = Counter()
    drift_count = 0
    stuck_count = 0
    for snap in l3_snaps:
        pat = snap.get("top_pattern", "")
        if pat:
            l3_patterns[pat] += 1
        if snap.get("drift_triggered"):
            drift_count += 1
        if snap.get("stuck_loop"):
            stuck_count += 1

    print(f"\n  🧠 L3 元认知统计:")
    print(f"     快照样本: {len(l3_snaps)}  漂移触发: {drift_count}  死循环告警: {stuck_count}")
    if l3_patterns:
        print(f"     失败模式分布:")
        for pat, cnt in l3_patterns.most_common(5):
            print(f"       {cnt}x  {pat}")
    elif len(l3_snaps) < 5:
        print(f"     ⏳ 样本不足 5 轮，失败模式识别尚未激活")

    # ── 7. 黑名单状态 ──
    blacklist = _load_json(LOG_DIR / "modification_blacklist.json", {})
    if blacklist:
        print(f"\n  🚫 破坏性修改黑名单:")
        for f, info in blacklist.items():
            banned = info.get("banned_until_iter", 0)
            fails = info.get("fail_count", 0)
            print(f"     {f}  失败{fails}次  禁改至iter {banned}")

    # ── 8. 修改成功率 ──
    applied_count = 0
    rollback_count = 0
    no_change_count = 0
    for item in evo_history if isinstance(evo_history, list) else []:
        if not isinstance(item, dict):
            continue
        applied = item.get("applied_count", 0)
        if applied > 0:
            applied_count += 1
        if item.get("rollback"):
            rollback_count += 1
        if applied == 0:
            no_change_count += 1

    print(f"\n  🔧 修改应用统计:")
    print(f"     有修改: {applied_count}  回滚: {rollback_count}  无变化: {no_change_count}")
    if total_iters > 0:
        print(f"     修改率: {applied_count / total_iters * 100:.1f}%")

    print("\n" + "=" * 60)
    if success_rate < 60 and total_attempts > 0:
        print("  ⚠️ JSON 成功率低于 60%，建议检查 logs/json_fail_log/ 调优提示词")
    if len(l3_snaps) < 5:
        print("  ⏳ L3 样本不足 5 轮，继续运行积累数据")
    print("=" * 60)


if __name__ == "__main__":
    main()
