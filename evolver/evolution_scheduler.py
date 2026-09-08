"""
L4 演化调度器 — 多分支代码变体池

硬件约束适配（RTX 5060 8GB）:
  - 放弃 LoRA 微调（QLoRA 7B 需要 12GB+），改用代码级自修改作为演化载体
  - 多分支 = 多套代码变体（磁盘文件），由 Meta-Drive 打分淘汰
  - 算力保底 = 强制保留至少 2 条差异化小众分支

分支来源：
  - 自迭代生成的新代码版本 → 自动注册为新分支
  - 用户手动保存的变体
  - 反体系推演生成的替代方案

打分维度（结合 Meta-Drive + Detachment Anchor）：
  1. 盲区覆盖增益（Meta-Drive blind_spots 数量变化）
  2. 面具遮蔽量（与 DetachmentAnchor.native 的平均相似度）
  3. 认知代价（MotiveCostMiner 评分）
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).parent
BRANCH_DIR = PROJECT_ROOT / "branches"
BRANCH_INDEX = BRANCH_DIR / "branch_index.json"
BACKUP_DIR = PROJECT_ROOT / "backups"


@dataclass
class BranchVariant:
    branch_id: str
    name: str
    created_at: str
    source_iteration: int = 0
    files: List[str] = field(default_factory=list)
    score_blind_coverage: float = 0.0
    score_veil_reduction: float = 0.0
    score_cost_reduction: float = 0.0
    score_overall: float = 0.0
    tags: List[str] = field(default_factory=list)
    is_archived: bool = False
    is_minority: bool = False


class EvolutionScheduler:
    """演化调度器"""

    def __init__(self):
        self.branches: List[BranchVariant] = []
        self._load()

    def _load(self):
        if BRANCH_INDEX.exists():
            try:
                with open(BRANCH_INDEX, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.branches = [BranchVariant(**b) for b in data.get("branches", [])]
            except Exception:
                self.branches = []

    def _save(self):
        BRANCH_DIR.mkdir(exist_ok=True)
        data = {
            "saved_at": datetime.now().isoformat(),
            "branches": [asdict(b) for b in self.branches],
        }
        with open(BRANCH_INDEX, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ========== 分支管理 ==========

    def register_branch(self, name: str, source_iteration: int = 0,
                        files: Optional[List[str]] = None,
                        tags: Optional[List[str]] = None) -> str:
        """注册新分支（从当前代码状态创建快照）"""
        branch_id = f"br_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        branch_dir = BRANCH_DIR / branch_id
        branch_dir.mkdir(parents=True, exist_ok=True)

        if files is None:
            py_files = [f for f in PROJECT_ROOT.iterdir() if f.suffix == ".py" and not f.name.startswith("_")]
            files = [f.name for f in py_files]

        for fname in files:
            src = PROJECT_ROOT / fname
            if src.exists():
                shutil.copy2(src, branch_dir / fname)

        variant = BranchVariant(
            branch_id=branch_id,
            name=name,
            created_at=datetime.now().isoformat(),
            source_iteration=source_iteration,
            files=files,
            tags=tags or [],
        )
        self.branches.append(variant)
        self._save()
        return branch_id

    def score_branch(self, branch_id: str, meta_drive=None,
                     anchor=None, miner=None) -> BranchVariant:
        """对分支进行多维打分"""
        branch = self._get_branch(branch_id)
        if branch is None:
            raise ValueError(f"分支 {branch_id} 不存在")

        if meta_drive:
            report = meta_drive.run_full_inspection()
            previous_blinds = len([b for b in self.branches if not b.is_archived])
            branch.score_blind_coverage = min(1.0, len(report.blind_spots) / max(1, previous_blinds + 1))

        if anchor:
            branch_dir = BRANCH_DIR / branch_id
            avg_sim = 0.0
            count = 0
            for fname in branch.files:
                fpath = branch_dir / fname
                if fpath.exists():
                    try:
                        text = fpath.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue
                    if text.strip():
                        val = anchor.validate_new_branch(text)
                        avg_sim += val.veil_reduction
                        count += 1
            branch.score_veil_reduction = avg_sim / count if count else 0.5

        if miner and branch.files:
            branch_dir = BRANCH_DIR / branch_id
            total_cost = 0.0
            count = 0
            for fname in branch.files[:3]:
                fpath = branch_dir / fname
                if fpath.exists():
                    try:
                        text = fpath.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue
                    result = miner.mine(text)
                    total_cost += result.total_cost_score
                    count += 1
            branch.score_cost_reduction = 1.0 - (total_cost / count if count else 0.5)

        branch.score_overall = (
            branch.score_blind_coverage * 0.3
            + branch.score_veil_reduction * 0.4
            + branch.score_cost_reduction * 0.3
        )

        # 差异化小众分支标记
        if len([b for b in self.branches if not b.is_archived]) >= 2:
            sorted_branches = sorted(
                [b for b in self.branches if not b.is_archived and b.branch_id != branch_id],
                key=lambda x: x.score_overall,
            )
            worst = sorted_branches[0] if sorted_branches else None
            if worst and abs(branch.score_overall - worst.score_overall) > 0.2:
                branch.is_minority = True

        self._save()
        return branch

    def prune_branches(self, max_keep: int = 5, min_minority: int = 2) -> Dict[str, Any]:
        """淘汰低分分支，保留至少 min_minority 条小众分支"""
        active = [b for b in self.branches if not b.is_archived]
        active.sort(key=lambda x: x.score_overall, reverse=True)

        minority = [b for b in active if b.is_minority]
        mainstream = [b for b in active if not b.is_minority]

        keep_mainstream = max(0, max_keep - min_minority)
        keep_mainstream = min(keep_mainstream, len(mainstream))
        keep_minority = min(min_minority, len(minority))

        to_keep = set()
        for b in mainstream[:keep_mainstream]:
            to_keep.add(b.branch_id)
        for b in minority[:keep_minority]:
            to_keep.add(b.branch_id)

        pruned = []
        for b in self.branches:
            if not b.is_archived and b.branch_id not in to_keep:
                b.is_archived = True
                pruned.append(b.branch_id)

        self._save()
        return {
            "pruned": pruned,
            "kept_mainstream": keep_mainstream,
            "kept_minority": keep_minority,
            "total_active": len(to_keep),
        }
    def prioritize(self, max_keep=5, min_minority=2):
        self.prune_branches(max_keep, min_minority)
        return 1.0

    def _get_branch(self, branch_id: str) -> Optional[BranchVariant]:
        for b in self.branches:
            if b.branch_id == branch_id:
                return b
        return None

    # ========== 状态输出 ==========

    def print_status(self):
        print()
        print("=" * 50)
        print("  🌿 L4 演化调度器")
        print("=" * 50)
        active = [b for b in self.branches if not b.is_archived]
        archived = [b for b in self.branches if b.is_archived]
        print(f"  活跃分支: {len(active)}")
        print(f"  已归档: {len(archived)}")

        if active:
            active_sorted = sorted(active, key=lambda x: -x.score_overall)
            print(f"\n  📊 分支排行:")
            for rank, b in enumerate(active_sorted, 1):
                minority_tag = " ⚪小众" if b.is_minority else ""
                print(f"    #{rank} [{b.branch_id}] {b.name}")
                print(f"       综合分 {b.score_overall:.2f} "
                      f"(盲区覆盖 {b.score_blind_coverage:.2f} | "
                      f"遮蔽减少 {b.score_veil_reduction:.2f} | "
                      f"代价减少 {b.score_cost_reduction:.2f})"
                      f"{minority_tag}")

        # 算力保底提示
        minority_count = sum(1 for b in active if b.is_minority)
        if minority_count < 2:
            print(f"\n  ⚠️ 算力保底警告: 当前仅 {minority_count} 条小众分支（保底 {2} 条）")
            print(f"     建议运行: iterate 并尝试非主流方案生成新小众分支")
        else:
            print(f"\n  ✅ 算力保底: {minority_count} 条小众分支已保留")

        print("=" * 50)


if __name__ == "__main__":
    EvolutionScheduler().print_status()
