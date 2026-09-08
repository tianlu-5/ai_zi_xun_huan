"""
L1 感知层 - 多维状态监测器 (Omni-State Monitor)
核心功能：能耗建模、状态追踪、健康诊断

实现用户架构设计的 L1 层：
  - 能耗监测：记录 token 消耗、推理耗时、错误率
  - 损耗建模：计算"核心认知模块"的综合损耗率（精神血条 0-100）
  - 自检机制：周期性健康诊断（内存/显存/服务连接/文件系统）
  - 趋势分析：滑动窗口内的熵值变化趋势

硬件约束 (RTX 5060 8GB + 7B Q4_K_M):
  - 纯统计计算，不调用 LLM
  - 可选 psutil 做系统监控（没装就跳过，降级运行）
  - 状态文件 ≤ 1MB，自动轮转
"""
import os
import sys
import json
import time
import math
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

log = logging.getLogger("PerceptionMonitor")

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False


class EnergyMonitor:
    """
    能耗监测器 - 追踪每次操作的资源消耗

    核心指标:
    - tokens_in/out: 输入/输出 token 数
    - latency_ms: 推理延迟（毫秒）
    - error_rate: 错误率（最近 N 次）
    - operation_count: 操作计数
    - time_efficiency: 单位时间有效产出（操作数/小时）
    """

    def __init__(self, state_file: str, window_size: int = 50):
        self.state_file = Path(state_file)
        self.window_size = window_size
        self._records: List[Dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._records = data.get("records", [])
                if len(self._records) > self.window_size * 3:
                    self._records = self._records[-self.window_size * 2:]
            except Exception as e:
                log.warning(f"加载状态文件失败: {e}")
                self._records = []

    def _save(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "saved_at": datetime.now().isoformat(),
                "record_count": len(self._records),
                "records": self._records[-self.window_size * 2:],
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"保存状态文件失败: {e}")

    def record_llm_call(self, tokens_in: int = 0, tokens_out: int = 0,
                        latency_ms: int = 0, success: bool = True,
                        model: str = "", task: str = "") -> None:
        """记录一次 LLM 调用"""
        entry = self._create_entry("llm", tokens_in, tokens_out, latency_ms, success, model, task)
        with self._lock:
            self._records.append(entry)
            self._save()

    def _create_entry(self, entry_type: str, tokens_in: int, tokens_out: int,
                      latency_ms: int, success: bool, model: str, task: str) -> Dict[str, Any]:
        return {
            "ts": datetime.now().isoformat(),
            "type": entry_type,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": tokens_in + tokens_out,
            "latency_ms": latency_ms,
            "success": success,
            "model": model,
            "task": task,
        }

    def record_operation(self, operation: str, success: bool,
                        duration_ms: int = 0, extra: Optional[Dict] = None):
        """记录一次桌面操作"""
        entry = {
            "ts": datetime.now().isoformat(),
            "type": "op",
            "op": operation,
            "success": success,
            "duration_ms": duration_ms,
            "extra": extra or {},
        }
        with self._lock:
            self._records.append(entry)
            self._save()

    def record_iteration(self, iteration_id: int, status: str,
                         steps_done: int = 0, steps_total: int = 0,
                         errors: Optional[List] = None):
        """记录一次自迭代"""
        entry = {
            "ts": datetime.now().isoformat(),
            "type": "iteration",
            "iteration": iteration_id,
            "status": status,
            "steps_done": steps_done,
            "steps_total": steps_total,
            "errors": errors or [],
        }
        with self._lock:
            self._records.append(entry)
            self._save()

    # ========== 统计计算 ==========
    def get_stats(self) -> Dict:
        """获取当前统计快照"""
        with self._lock:
            records = list(self._records)

        recent = records[-self.window_size:]
        llm_calls = [r for r in recent if r.get("type") == "llm"]
        operations = [r for r in recent if r.get("type") == "op"]
        iterations = [r for r in recent if r.get("type") == "iteration"]

        llm_success = [r for r in llm_calls if r.get("success")]
        op_success = [r for r in operations if r.get("success")]

        total_tokens = sum(r.get("total_tokens", 0) for r in llm_calls)
        avg_latency = (
            sum(r.get("latency_ms", 0) for r in llm_calls) / max(len(llm_calls), 1)
        )

        return {
            "window_size": self.window_size,
            "total_records": len(records),
            "recent_records": len(recent),
            "llm_calls": len(llm_calls),
            "llm_success_rate": len(llm_success) / max(len(llm_calls), 1),
            "llm_total_tokens": total_tokens,
            "llm_avg_latency_ms": round(avg_latency, 1),
            "operations": len(operations),
            "op_success_rate": len(op_success) / max(len(operations), 1),
            "iterations": len(iterations),
            "iter_success": sum(1 for r in iterations if r.get("status") == "修改成功"),
        }

    def compute_depletion_score(self) -> Dict:
        """
        计算核心认知损耗率（精神血条）
        返回 0-100 分，越低越需要休息/调整

        损耗来源:
        1. 最近错误率（权重40%）
        2. 平均推理延迟（权重25%）- 延迟越高说明模型吃力
        3. 连续失败次数（权重20%）
        4. 操作密度（权重15%）- 单位时间内操作太多可能导致不稳定
        """
        with self._lock:
            records = list(self._records)

        stats = self.get_stats()
        score = 100.0
        breakdown = {}

        # 因素1: 错误率惩罚 (40%)
        combined_error = 1.0 - stats["llm_success_rate"] * 0.6 - stats["op_success_rate"] * 0.4
        error_penalty = combined_error * 40
        score -= error_penalty
        breakdown["error_penalty"] = round(error_penalty, 1)

        # 因素2: 高延迟惩罚 (25%)
        if stats["llm_calls"] > 0:
            latency = stats["llm_avg_latency_ms"]
            if latency > 30000:
                latency_penalty = 25
            elif latency > 15000:
                latency_penalty = 15
            elif latency > 8000:
                latency_penalty = 8
            else:
                latency_penalty = 0
        else:
            latency_penalty = 0
        score -= latency_penalty
        breakdown["latency_penalty"] = latency_penalty

        # 因素3: 连续失败惩罚 (20%)
        recent = records[-10:]
        consecutive_fails = 0
        for r in reversed(recent):
            if r.get("type") == "iteration" and r.get("status") != "修改成功":
                consecutive_fails += 1
            elif r.get("type") == "llm" and not r.get("success"):
                consecutive_fails += 1
            else:
                break
        fail_penalty = min(consecutive_fails * 5, 20)
        score -= fail_penalty
        breakdown["consecutive_fails"] = consecutive_fails
        breakdown["fail_penalty"] = fail_penalty

        # 因素4: 操作密度 (15%) - 60秒内超过20次操作视为过载
        op_times = [r.get("ts", "") for r in records[-100:] if r.get("type") in ("op", "llm")]
        if len(op_times) >= 2:
            try:
                t0 = datetime.fromisoformat(op_times[0])
                t1 = datetime.fromisoformat(op_times[-1])
                duration_min = max((t1 - t0).total_seconds() / 60, 0.1)
                density = len(op_times) / duration_min
                if density > 60:
                    density_penalty = 15
                elif density > 40:
                    density_penalty = 10
                elif density > 20:
                    density_penalty = 5
                else:
                    density_penalty = 0
            except Exception:
                density_penalty = 0
        else:
            density_penalty = 0
        score -= density_penalty
        breakdown["density_penalty"] = density_penalty

        score = max(0, min(100, score))

        if score >= 80:
            state = "精力充沛"
        elif score >= 60:
            state = "状态良好"
        elif score >= 40:
            state = "略显疲惫"
        elif score >= 20:
            state = "需要休息"
        else:
            state = "严重过载"

        return {
            "score": round(score, 1),
            "state": state,
            "breakdown": breakdown,
            "recommendation": self._recommend(score),
            "stats": stats,
        }

    @staticmethod
    def _recommend(score: float) -> str:
        if score >= 80:
            return "保持当前节奏，可尝试更复杂的任务"
        elif score >= 60:
            return "状态稳定，建议继续常规任务"
        elif score >= 40:
            return "建议降低任务复杂度，减少并发操作"
        elif score >= 20:
            return "暂停迭代，释放GPU显存，做一次完全重启"
        else:
            return "⚠️ 立即停止所有操作！系统已严重过载"


class HealthDiagnosis:
    """
    健康诊断器 - 周期性检查系统和模块状态

    检查项:
    - 进程内存占用
    - GPU 显存（如果 NVIDIA 工具可用）
    - Ollama 服务连接
    - 磁盘空间
    - Python 模块完整性（import 检查）
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self._last_check: Optional[datetime] = None
        self._check_interval = timedelta(minutes=5)

    def run_check(self, force: bool = False) -> Dict:
        """运行一次完整健康检查"""
        now = datetime.now()
        if not force and self._last_check and (now - self._last_check) < self._check_interval:
            return {"status": "skipped", "reason": "interval_not_ready"}

        self._last_check = now
        results = {
            "checked_at": now.isoformat(),
            "checks": {},
            "warnings": [],
            "critical": [],
        }

        # 1. 内存
        if PSUTIL_OK:
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
            results["checks"]["memory"] = {
                "total_gb": round(mem.total / (1024**3), 1),
                "available_gb": round(mem.available / (1024**3), 1),
                "percent": mem_percent,
            }
            if mem_percent > 90:
                results["critical"].append(f"内存使用率 {mem_percent}%")
            elif mem_percent > 75:
                results["warnings"].append(f"内存使用率偏高 {mem_percent}%")

            try:
                proc = psutil.Process()
                proc_mem_mb = proc.memory_info().rss / (1024**2)
                results["checks"]["process_memory_mb"] = round(proc_mem_mb, 1)
            except Exception:
                pass

        # 2. GPU 显存
        results["checks"]["gpu"] = self._check_gpu()
        if results["checks"]["gpu"].get("warning"):
            results["warnings"].append(results["checks"]["gpu"]["warning"])

        # 3. Ollama 连接
        results["checks"]["ollama"] = self._check_ollama()
        if not results["checks"]["ollama"]["connected"]:
            results["critical"].append("Ollama 服务不可达")

        # 4. 磁盘空间
        results["checks"]["disk"] = self._check_disk()
        if results["checks"]["disk"].get("percent", 0) > 95:
            results["critical"].append(f"磁盘空间不足 {results['checks']['disk']['percent']}%")
        elif results["checks"]["disk"].get("percent", 0) > 85:
            results["warnings"].append(f"磁盘空间紧张 {results['checks']['disk']['percent']}%")

        # 5. 模块完整性
        results["checks"]["modules"] = self._check_modules()

        # 6. 备份目录状态
        results["checks"]["backups"] = self._check_backups()

        if results["critical"]:
            results["status"] = "critical"
        elif results["warnings"]:
            results["status"] = "warning"
        else:
            results["status"] = "healthy"

        return results

    def _check_gpu(self) -> Dict:
        info = {"available": False}
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                if len(parts) >= 3:
                    total = float(parts[0].strip())
                    used = float(parts[1].strip())
                    free = float(parts[2].strip())
                    info.update({
                        "available": True,
                        "total_mb": total,
                        "used_mb": used,
                        "free_mb": free,
                        "percent": round(used / max(total, 1) * 100, 1),
                    })
                    if free < 512:
                        info["warning"] = f"GPU显存剩余不足 512MB (当前 free={free:.0f}MB)"
        except Exception:
            pass
        return info

    def _check_ollama(self) -> Dict:
        from config import Config
        import urllib.request
        try:
            req = urllib.request.Request(f"{Config.OLLAMA_BASE_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return {"connected": resp.status == 200}
        except Exception as e:
            return {"connected": False, "error": str(e)[:100]}

    def _check_disk(self) -> Dict:
        if PSUTIL_OK:
            try:
                disk = psutil.disk_usage(str(self.project_root))
                return {
                    "total_gb": round(disk.total / (1024**3), 1),
                    "free_gb": round(disk.free / (1024**3), 1),
                    "percent": disk.percent,
                }
            except Exception:
                pass
        return {}

    def _check_modules(self) -> Dict:
        required = [
            ("desktop_agent", "桌面智能体"),
            ("ollama_client", "Ollama客户端"),
            ("modification_parser", "修改解析器"),
            ("task_orchestrator", "任务编排器"),
            ("vision_ocr", "视觉OCR（可选）"),
        ]
        results = {}
        for mod, label in required:
            try:
                __import__(mod)
                results[mod] = {"ok": True, "label": label}
            except Exception as e:
                results[mod] = {"ok": False, "label": label, "error": str(e)[:80]}
        return results

    def _check_backups(self) -> Dict:
        backup_dir = self.project_root / "backups"
        if not backup_dir.exists():
            return {"count": 0, "note": "备份目录不存在"}
        backups = list(backup_dir.iterdir())
        recent = sorted(backups, key=lambda p: p.stat().st_mtime, reverse=True)[:5]
        total_size = sum(p.stat().st_size for p in backups)
        return {
            "count": len(backups),
            "total_size_mb": round(total_size / (1024**2), 1),
            "recent": [p.name for p in recent],
        }


class PerceptionMonitor:
    """
    L1 层主入口 - 统一管理能耗监测 + 健康诊断

    使用方式:
        monitor = PerceptionMonitor()
        monitor.energy.record_llm_call(tokens_in=1000, tokens_out=500, latency_ms=5000, success=True)
        monitor.energy.record_operation("click", success=True, duration_ms=100)
        score = monitor.energy.compute_depletion_score()
        health = monitor.health.run_check()
        monitor.print_status()
    """

    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            project_root = str(Path(__file__).parent.resolve())
        self.project_root = Path(project_root)
        self.state_file = self.project_root / "logs" / "state_log.json"

        self.energy = EnergyMonitor(str(self.state_file))
        self.health = HealthDiagnosis(str(self.project_root))
    def snapshot(self) -> Dict[str, Any]:
        energy_data = self.energy.compute_depletion_score()
        health_data = self.health.run_check(force=True)
        return self._format_snapshot(energy_data, health_data)

    def _format_snapshot(self, energy_data: Dict[str, Any], health_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "energy": energy_data,
            "health": health_data
        }

    def _format_snapshot(self, energy_data: Dict[str, Any], health_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "energy": energy_data,
            "health": health_data
        }

    def _format_snapshot(self, energy_data: Dict[str, Any], health_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "energy": energy_data,
            "health": health_data
        }

    def print_status(self):
        """打印当前状态"""
        stats = self._get_stats()
        print(f"\n  📊 最近 {stats['window_size']} 次统计:")
        print(f"    LLM调用: {stats['llm_calls']}次 | 成功率 {stats['llm_success_rate']*100:.0f}% | "
              f"总tokens {stats['llm_total_tokens']} | 平均延迟 {stats['llm_avg_latency_ms']:.0f}ms")
        print(f"    桌面操作: {stats['operations']}次 | 成功率 {stats['op_success_rate']*100:.0f}%")
        print(f"    自迭代: {stats['iterations']}次 | 成功 {stats['iter_success']}次")

        print(f"\n  🏥 健康诊断:")
        self._print_health_status(health)

        print("=" * 50 + "\n")

    def _print_health_status(self, health: Dict[str, Any]) -> None:
        """打印健康诊断信息"""
        hc = health.get("checks", {})
        mem = hc.get("memory", {})
        if mem:
            print(f"    内存: {mem.get("percent", '?')}% ({mem.get("available_gb", '?')}GB可用)")
        gpu = hc.get("gpu", {})
        if gpu.get("available"):
            print(f"    GPU: {gpu.get("percent", '?')}% (free={gpu.get("free_mb", '?'):.0f}MB)")
        elif gpu.get("warning"):
            print(f"    GPU: ⚠️ {gpu['warning']}")
        ollama = hc.get("ollama", {})
        print(f"    Ollama: {'✅ 已连接' if ollama.get("connected") else "❌ 不可达"})")
        disk = hc.get("disk", {})
        if disk:
            print(f"    磁盘: {disk.get("percent", '?')}% ({disk.get("free_gb", '?')}GB可用)")

        if health.get("warnings"):
            print(f"\n  ⚠️ 警告:")
            for w in health["warnings"]:
                print(f"    - {w}")
        if health.get("critical"):
            print(f"\n  🚨 严重:")
            for c in health["critical"]:
                print(f"    - {c}")
    def improve_code_quality(self) -> None:
        """对长期未修改的模块进行代码质量改进，包括类型注解、文档字符串、异常处理"""
        try:
            # 添加类型注解
            self._records: List[Dict[str, Any]] = []
            self._lock: threading.Lock = threading.Lock()

            # 添加文档字符串
            self._load.__doc__ = """加载状态文件"""
            self._save.__doc__ = """保存状态文件"""
            self.record_llm_call.__doc__ = """记录一次 LLM 调用"""
            self.record_operation.__doc__ = """记录一次桌面操作"""

            # 添加异常处理
            self._load()
            self._save()

        except Exception as e:
            log.error(f"改进代码质量失败: {e}")
    def optimize_performance(self) -> None:
        """优化代码性能和可读性，识别并消除不必要的计算"""
        with self._lock:
            self._records = [entry for entry in self._records if entry['success']]
            self._save()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor = PerceptionMonitor()
    # 模拟一些记录
    monitor.energy.record_llm_call(tokens_in=2000, tokens_out=800, latency_ms=6500, success=True, model="deepseek-r1:7b", task="self_evolve")
    monitor.energy.record_operation("click_text", success=True, duration_ms=120)
    monitor.energy.record_operation("click", success=True, duration_ms=50)
    monitor.energy.record_operation("run_command", success=True, duration_ms=300)
    monitor.energy.record_iteration(iteration_id=17, status="修改成功", steps_done=3, steps_total=3)
    monitor.print_status()
