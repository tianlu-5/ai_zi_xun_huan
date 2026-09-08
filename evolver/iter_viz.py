"""
迭代可视化工具 - 生成进化趋势图表

用法:
    python -m evolver.iter_viz           # 生成所有图表
    python -m evolver.iter_viz --trend   # 只生成趋势图
    python -m evolver.iter_viz --radar   # 只生成雷达图

输出目录: logs/charts/
"""

import json
import sys
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from config import Config

# 设置中文字体（Windows）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# =================================
class IterViz:
    """迭代可视化引擎"""

    def __init__(self):
        self.chart_dir = Config.LOG_DIR / "charts"
        self.chart_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        self.benchmark_history = self._load_benchmark()
        self.evolution_history = self._load_evolution()

    def _load_benchmark(self) -> List[Dict]:
        """加载基准历史"""
        path = Config.LOG_DIR / "benchmark_history.json"
        if not path.exists():
            print(f"  ⚠️ benchmark_history.json 不存在: {path}")
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                print(f"  ⚠️ benchmark_history.json 格式不是列表: {type(data)}")
                return []
        except Exception as e:
            print(f"  ⚠️ 加载 benchmark_history.json 失败: {e}")
            return []

    def _load_evolution(self) -> List[Dict]:
        """加载迭代历史"""
        path = Config.LOG_DIR / "evolution_history.json"
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def generate_all(self):
        """生成所有图表"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
        except ImportError:
            print("❌ matplotlib 未安装，请运行: pip install matplotlib")
            return

        print("\n📊 生成迭代可视化图表...")

        if not self.benchmark_history and not self.evolution_history:
            print("   ⚠️ 没有历史数据，跳过")
            return

        # 1. 趋势图
        self._plot_trend(plt)
        print("   ✅ 趋势图已生成")

        # 2. 合约通过率
        self._plot_contracts(plt)
        print("   ✅ 合约通过率图已生成")

        # 3. 成功率
        self._plot_success_rate(plt)
        print("   ✅ 成功率图已生成")

        # 4. 雷达图（如果有最新数据）
        self._plot_radar(plt)
        print("   ✅ 雷达图已生成")

        plt.close('all')
        print(f"\n📁 图表已保存到: {self.chart_dir}\n")

    def _plot_trend(self, plt):
        """质量评分趋势图"""
        if not self.benchmark_history:
            return

        iterations = []
        scores = []

        for idx, item in enumerate(self.benchmark_history):
            score = item.get("score_delta", 0)
            iterations.append(idx + 1)
            scores.append(score)

        if len(iterations) < 2:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(iterations, scores, 'b-', linewidth=2, label='质量变化')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax.fill_between(iterations, 0, scores, where=[s > 0 for s in scores],
                        color='green', alpha=0.2, label='进步区域')
        ax.fill_between(iterations, 0, scores, where=[s < 0 for s in scores],
                        color='red', alpha=0.2, label='退步区域')

        ax.set_title('质量评分趋势', fontsize=14)
        ax.set_xlabel('基准测试次数')
        ax.set_ylabel('评分变化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.chart_dir / 'trend.png', dpi=150)

    def _plot_contracts(self, plt):
        """合约通过率趋势"""
        if not self.benchmark_history:
            return

        iterations = []
        pass_ratios = []

        for item in self.benchmark_history:
            tasks = item.get("tasks", {})
            if "B3" in tasks:
                ratio = tasks["B3"].get("value", 0)
                iterations.append(item.get("iteration_id", 0))
                pass_ratios.append(ratio)

        if len(iterations) < 2:
            return

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(iterations, pass_ratios, 'g-', linewidth=2)
        ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='目标 95%')
        ax.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='警告 80%')

        ax.set_title('🔒 合约通过率趋势', fontsize=14)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('通过率 (%)')
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.chart_dir / 'contracts.png', dpi=150)

    def _plot_success_rate(self, plt):
        """成功率趋势（滑动窗口）"""
        if not self.evolution_history:
            return

        window = 20
        iterations = []
        rates = []

        for i in range(len(self.evolution_history)):
            end = i + 1
            start = max(0, i - window + 1)
            window_data = self.evolution_history[start:end]

            success = sum(1 for item in window_data
                         if item.get("status") in ("修改成功", "部分成功", "故障自愈已执行"))
            total = len(window_data)
            if total > 0:
                iterations.append(i + 1)
                rates.append(success / total * 100)

        if len(iterations) < 2:
            return

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.bar(iterations, rates, width=0.8, color='steelblue', alpha=0.7)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% 及格线')
        ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='70% 良好')

        ax.set_title(f'📊 迭代成功率（{window}轮滑动窗口）', fontsize=14)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('成功率 (%)')
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.chart_dir / 'success_rate.png', dpi=150)

    def _plot_radar(self, plt):
        """雷达图 - 各维度表现"""
        if not self.benchmark_history:
            return

        # 找最近一次非跳过的报告
        latest = None
        for item in reversed(self.benchmark_history):
            if item.get("verdict") != "跳过（非基准轮次）":
                latest = item
                break

        if not latest:
            return

        tasks = latest.get("tasks", {})
        if not tasks:
            return

        # ===== 调试打印 =====
        print(f"  🔍 最新基准数据 (迭代 #{latest.get('iteration_id', 0)}):")
        for key, val in tasks.items():
            print(f"    {key}: {val.get('value', 'N/A')} ({val.get('name', '')})")
        # ====================

        # 只提取 B1-B5，使用中文标签
        dim_names = ["B1", "B2", "B3", "B4", "B5"]
        dim_labels = ["语法健壮度", "导入覆盖率", "Contract", "熵差防御", "分析速度"]
        
        dims = []
        values = []
        for name, label in zip(dim_names, dim_labels):
            if name in tasks:
                dims.append(label)
                values.append(tasks[name].get("value", 0))
            else:
                dims.append(label)
                values.append(0)

        # ===== 打印维度数据 =====
        print(f"  📊 雷达图维度:")
        for d, v in zip(dims, values):
            print(f"    {d}: {v}")
        # ========================

        if len(dims) < 3:
            print(f"  ⚠️ 维度不足 3 个，跳过雷达图")
            return

        import numpy as np
        angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
        values_radar = values + values[:1]
        angles_radar = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.plot(angles_radar, values_radar, 'o-', linewidth=2, color='blue')
        ax.fill(angles_radar, values_radar, alpha=0.25, color='blue')

        ax.set_xticks(angles)
        ax.set_xticklabels(dims, size=10)

        ax.set_ylim(0, 110)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=8)

        ax.set_title(f'📊 各维度表现 (迭代 #{latest.get("iteration_id", 0)})', fontsize=14, pad=20)

        plt.tight_layout()
        plt.savefig(self.chart_dir / 'radar.png', dpi=150)

# ── CLI ──
if __name__ == "__main__":
    viz = IterViz()
    viz.generate_all()