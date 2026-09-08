"""
模型路由器 - 智能 Fallback 机制

核心功能：
1. 启动时扫描所有可用的推理后端（API / Ollama / LM Studio）
2. 根据策略自动排序（成本优先 / 速度优先 / 平衡 / 自适应）
3. 运行时自动切换（失败 → 降级 → 切换）
4. 记录每个后端的成功率，动态调整优先级

使用方式：
    router = ModelRouter()
    client = router.get_client()  # 获取当前最佳客户端
    # 使用 client.chat(...)
    router.mark_success(client.backend_id)  # 成功后记录
    router.mark_failure(client.backend_id)  # 失败后记录（可能触发降级）
"""

import json
import urllib.request
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from config import Config


@dataclass
class BackendInfo:
    """后端信息"""
    id: str                    # 唯一标识
    name: str                  # 显示名称
    model: str                 # 模型名称
    type: str                  # "api" / "local"
    cost_per_1k: float         # 每 1000 token 成本（美元）
    speed: str                 # "fast" / "medium" / "slow"
    vram_gb: float = 0         # 需要的显存（GB）
    available: bool = True     # 是否可用
    success_rate: float = 1.0  # 历史成功率
    avg_speed_tok_s: float = 0 # 平均速度
    failures: int = 0          # 连续失败次数
    total_calls: int = 0       # 总调用次数
    total_success: int = 0     # 成功次数


class ModelRouter:
    """模型路由器 - 智能 Fallback 引擎"""

    # 默认排序策略
    STRATEGY_COST_FIRST = "cost_first"
    STRATEGY_SPEED_FIRST = "speed_first"
    STRATEGY_BALANCED = "balanced"
    STRATEGY_AUTO = "auto"

    # 连续失败阈值（触发降级）
    FAILURE_THRESHOLD = 3
    # 恢复探测间隔（轮次）
    RECOVERY_CHECK_INTERVAL = 10

    def __init__(self, strategy: str = STRATEGY_BALANCED):
        self.strategy = strategy
        self._backends: List[BackendInfo] = []
        self._ranked: List[BackendInfo] = []
        self._current_index: int = 0
        self._current_client = None
        self._history_file = Config.LOG_DIR / "router_history.json"
        self._lock = threading.Lock()
        self._cycle_count = 0

        # 加载历史数据
        self._load_history()

        # 扫描所有可用后端
        self._scan_backends()

        # 排序
        self._rank()

        # 初始化第一个客户端
        self._init_first_client()

    # ========== 扫描 ==========

    def _scan_backends(self) -> None:
        """扫描所有可用的推理后端"""
        backends = []

        # 1. 基元律动 API
        if Config.JIYUAN_API_KEY and len(Config.JIYUAN_API_KEY) > 10:
            backends.append(BackendInfo(
                id="jiyuan",
                name="基元律动 API",
                model=Config.JIYUAN_MODEL,
                type="api",
                cost_per_1k=0.002,
                speed="fast",
            ))
            print(f"  ✅ [Router] 发现: 基元律动 API ({Config.JIYUAN_MODEL})")

        # 2. 火山方舟 API
        if Config.ARK_API_KEY and len(Config.ARK_API_KEY) > 10:
            backends.append(BackendInfo(
                id="ark",
                name="火山方舟 API",
                model=Config.MODEL_ID,
                type="api",
                cost_per_1k=0.003,
                speed="fast",
            ))
            print(f"  ✅ [Router] 发现: 火山方舟 API ({Config.MODEL_ID})")

        # 3. Ollama 本地模型
        ollama_models = self._scan_ollama()
        for m in ollama_models:
            size_gb = m.get("size_gb", 0)
            # 根据模型大小判断速度
            speed = "slow" if size_gb > 10 else "medium"
            backends.append(BackendInfo(
                id=f"ollama_{m['name']}",
                name=f"Ollama - {m['name']}",
                model=m['name'],
                type="local",
                cost_per_1k=0,
                speed=speed,
                vram_gb=size_gb,
            ))
            print(f"  ✅ [Router] 发现: Ollama - {m['name']} ({size_gb:.1f}GB)")

        # 4. LM Studio 本地模型
        lmstudio_models = self._scan_lmstudio()
        for m in lmstudio_models:
            size_gb = m.get("size_gb", 0)
            speed = "slow" if size_gb > 10 else "medium"
            backends.append(BackendInfo(
                id=f"lmstudio_{m['id']}",
                name=f"LM Studio - {m['id']}",
                model=m['id'],
                type="local",
                cost_per_1k=0,
                speed=speed,
                vram_gb=size_gb,
            ))
            print(f"  ✅ [Router] 发现: LM Studio - {m['id']} ({size_gb:.1f}GB)")

        # 合并历史数据
        self._merge_history(backends)

        self._backends = backends

        if not backends:
            print("  ⚠️ [Router] 未发现任何可用的推理后端！")

    def _scan_ollama(self) -> List[Dict[str, Any]]:
        """扫描 Ollama 已安装的模型"""
        results = []
        try:
            req = urllib.request.Request(f"{Config.OLLAMA_BASE_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                for m in data.get("models", []):
                    name = m.get("name", "")
                    # 只保留代码模型
                    if any(kw in name.lower() for kw in ["coder", "deepseek", "llama"]):
                        size_bytes = m.get("size", 0)
                        results.append({
                            "name": name,
                            "size_gb": round(size_bytes / (1024**3), 2),
                            "digest": m.get("digest", "")[:12],
                        })
        except Exception:
            pass
        return results

    def _scan_lmstudio(self) -> List[Dict[str, Any]]:
        """扫描 LM Studio 已下载的模型"""
        results = []
        try:
            # 从 Config.LMSTUDIO_MODELS_DIR 扫描 GGUF 文件
            models_dir = Path(Config.LMSTUDIO_MODELS_DIR)
            if models_dir.exists():
                for gguf in models_dir.rglob("*.gguf"):
                    if gguf.name.endswith(".part"):
                        continue
                    size_gb = round(gguf.stat().st_size / (1024**3), 2)
                    # 从文件名提取模型名
                    model_id = gguf.stem.replace(".gguf", "")
                    # 只保留代码模型
                    if any(kw in model_id.lower() for kw in ["coder", "deepseek", "llama", "qwen"]):
                        results.append({
                            "id": model_id,
                            "size_gb": size_gb,
                            "path": str(gguf),
                        })
            else:
                # 尝试通过 LM Studio API 获取
                req = urllib.request.Request(f"{Config.LMSTUDIO_BASE_URL}/models")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read())
                    for m in data.get("data", []):
                        model_id = m.get("id", "")
                        results.append({
                            "id": model_id,
                            "size_gb": 0,
                            "path": "",
                        })
        except Exception:
            pass
        return results

    def _merge_history(self, backends: List[BackendInfo]) -> None:
        """合并历史数据"""
        if not self._history:
            return

        for backend in backends:
            hist = self._history.get(backend.id)
            if hist:
                backend.success_rate = hist.get("success_rate", 1.0)
                backend.avg_speed_tok_s = hist.get("avg_speed", 0)
                backend.total_calls = hist.get("total_calls", 0)
                backend.total_success = hist.get("total_success", 0)

    # ========== 排序 ==========

    def _rank(self) -> None:
        """根据策略排序"""
        if self.strategy == self.STRATEGY_COST_FIRST:
            self._ranked = sorted(self._backends, key=lambda b: b.cost_per_1k)
        elif self.strategy == self.STRATEGY_SPEED_FIRST:
            speed_rank = {"fast": 0, "medium": 1, "slow": 2}
            self._ranked = sorted(self._backends, key=lambda b: speed_rank.get(b.speed, 2))
        elif self.strategy == self.STRATEGY_AUTO:
            self._ranked = sorted(self._backends, key=lambda b: (
                b.success_rate * 0.5 + self._speed_score(b) * 0.3 - b.cost_per_1k * 10 * 0.2
            ), reverse=True)
        else:  # BALANCED
            # 平衡：免费优先，速度快的优先
            self._ranked = sorted(self._backends, key=lambda b: (
                b.cost_per_1k,  # 免费优先
                -self._speed_score(b),  # 速度快优先
            ))

        # 过滤不可用的
        self._ranked = [b for b in self._ranked if b.available]

    def _speed_score(self, backend: BackendInfo) -> float:
        """速度评分（0-1）"""
        speed_map = {"fast": 1.0, "medium": 0.6, "slow": 0.3}
        return speed_map.get(backend.speed, 0.5)

    def _init_first_client(self) -> None:
        """初始化第一个客户端"""
        if not self._ranked:
            self._current_client = None
            return

        try:
            self._current_client = self._create_client(self._ranked[0])
            self._current_index = 0
            print(f"  🚀 [Router] 使用默认后端: {self._ranked[0].name}")
        except Exception as e:
            print(f"  ⚠️ [Router] 初始化 {self._ranked[0].name} 失败: {e}")
            self._current_client = None
            # 尝试下一个
            for i in range(1, len(self._ranked)):
                try:
                    self._current_client = self._create_client(self._ranked[i])
                    self._current_index = i
                    print(f"  🔄 [Router] 切换到备用: {self._ranked[i].name}")
                    break
                except Exception:
                    continue

    # ========== 客户端创建 ==========

    def _create_client(self, backend: BackendInfo):
        """根据后端信息创建客户端"""
        if backend.type == "api":
            from doubao_client import DoubaoClient
            if backend.id == "jiyuan":
                return DoubaoClient(
                    api_key=Config.JIYUAN_API_KEY,
                    base_url=Config.JIYUAN_BASE_URL,
                    model_id=backend.model,
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS,
                    backend_id=backend.id,
                )
            else:
                return DoubaoClient(
                    api_key=Config.ARK_API_KEY,
                    base_url=Config.ARK_BASE_URL,
                    model_id=backend.model,
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS,
                    backend_id=backend.id,
                )
        else:
            from ollama_client import OllamaClient
            if "lmstudio" in backend.id:
                return OllamaClient(
                    base_url=Config.LMSTUDIO_BASE_URL,
                    model_id=backend.model,
                    temperature=Config.TEMPERATURE,
                    num_ctx=Config.LMSTUDIO_NUM_CTX,
                    backend_id=backend.id,
                )
            else:
                return OllamaClient(
                    base_url=Config.OLLAMA_BASE_URL,
                    model_id=backend.model,
                    temperature=Config.TEMPERATURE,
                    num_ctx=Config.OLLAMA_NUM_CTX,
                    backend_id=backend.id,
                )

    # ========== 核心接口 ==========

    def get_client(self):
        """获取当前最佳客户端"""
        if self._current_client is None:
            self._try_recover()
        return self._current_client

    def mark_success(self, backend_id: str, tokens: int = 0, elapsed: float = 0) -> None:
        """标记某次调用成功"""
        with self._lock:
            for b in self._backends:
                if b.id == backend_id:
                    b.total_calls += 1
                    b.total_success += 1
                    b.failures = 0
                    if elapsed > 0 and tokens > 0:
                        speed = tokens / elapsed
                        b.avg_speed_tok_s = (b.avg_speed_tok_s * 0.7 + speed * 0.3)
                    b.success_rate = b.total_success / max(b.total_calls, 1)
                    break
            self._save_history()

    def mark_failure(self, backend_id: str, error: str = "") -> bool:
        """
        标记某次调用失败
        返回: True 表示触发了降级切换
        """
        with self._lock:
            for b in self._backends:
                if b.id == backend_id:
                    b.total_calls += 1
                    b.failures += 1
                    b.success_rate = b.total_success / max(b.total_calls, 1)

                    # 如果连续失败超过阈值，触发降级
                    if b.failures >= self.FAILURE_THRESHOLD:
                        self._demote_backend(backend_id)
                        return True
                    break
            self._save_history()
        return False

    def _demote_backend(self, backend_id: str) -> None:
        """降级某个后端"""
        with self._lock:
            for i, b in enumerate(self._ranked):
                if b.id == backend_id:
                    # 移到末尾
                    self._ranked.append(self._ranked.pop(i))
                    print(f"  🔄 [Router] {b.name} 降级到末尾（连续失败 {self.FAILURE_THRESHOLD} 次）")
                    break

        # 尝试切换到下一个可用后端
        self._switch_to_next()

    def _switch_to_next(self) -> bool:
        """切换到下一个可用后端"""
        with self._lock:
            for i in range(len(self._ranked)):
                # 从当前索引的下一个开始
                idx = (self._current_index + i + 1) % len(self._ranked)
                backend = self._ranked[idx]
                try:
                    client = self._create_client(backend)
                    if client.check_connection():
                        self._current_client = client
                        self._current_index = idx
                        print(f"  🔄 [Router] 切换到: {backend.name}")
                        return True
                except Exception:
                    continue

        print(f"  ❌ [Router] 所有后端都不可用！")
        self._current_client = None
        return False

    def _try_recover(self) -> bool:
        """尝试恢复（从降级状态恢复）"""
        with self._lock:
            # 重新排序（恢复可能已恢复的后端）
            self._rank()

            for i, backend in enumerate(self._ranked):
                try:
                    client = self._create_client(backend)
                    if client.check_connection():
                        self._current_client = client
                        self._current_index = i
                        print(f"  🔄 [Router] 恢复: {backend.name}")
                        return True
                except Exception:
                    continue

        self._current_client = None
        return False

    def get_status(self) -> Dict[str, Any]:
        """获取路由器状态"""
        with self._lock:
            current = self._ranked[self._current_index] if self._ranked else None
            return {
                "strategy": self.strategy,
                "total_backends": len(self._backends),
                "active_backends": len([b for b in self._backends if b.available]),
                "current_backend": current.name if current else None,
                "current_index": self._current_index,
                "backends": [
                    {
                        "id": b.id,
                        "name": b.name,
                        "type": b.type,
                        "cost": b.cost_per_1k,
                        "speed": b.speed,
                        "success_rate": round(b.success_rate, 2),
                        "failures": b.failures,
                    }
                    for b in self._ranked
                ],
            }

    def print_status(self) -> None:
        """打印路由器状态"""
        status = self.get_status()
        print("\n" + "=" * 60)
        print("  🔀 模型路由器状态")
        print("=" * 60)
        print(f"  策略: {status['strategy']}")
        print(f"  当前后端: {status['current_backend']} (索引 {status['current_index']})")
        print(f"  活跃后端: {status['active_backends']}/{status['total_backends']}")
        print("\n  📋 后端列表（按优先级排序）:")
        for i, b in enumerate(status["backends"]):
            indicator = "👉" if i == status["current_index"] else "  "
            status_icon = "✅" if b["failures"] < 3 else "⚠️"
            print(f"    {indicator} {i+1}. {status_icon} {b['name']}")
            print(f"         类型: {b['type']} | 速度: {b['speed']} | 成本: ${b['cost']}/1K")
            print(f"         成功率: {b['success_rate']:.0%} | 连续失败: {b['failures']}")
        print("=" * 60 + "\n")

    # ========== 持久化 ==========

    def _load_history(self) -> None:
        """加载历史数据"""
        self._history = {}
        if self._history_file.exists():
            try:
                with open(self._history_file, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except Exception:
                self._history = {}

    def _save_history(self) -> None:
        """保存历史数据"""
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for b in self._backends:
                data[b.id] = {
                    "success_rate": b.success_rate,
                    "avg_speed": b.avg_speed_tok_s,
                    "total_calls": b.total_calls,
                    "total_success": b.total_success,
                }
            with open(self._history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def cycle_tick(self) -> None:
        """每轮迭代调用，用于恢复探测"""
        self._cycle_count += 1
        if self._cycle_count % self.RECOVERY_CHECK_INTERVAL == 0:
            # 尝试恢复被降级的后端
            self._try_recover()


# 全局单例
_router: Optional[ModelRouter] = None


def get_router(strategy: str = ModelRouter.STRATEGY_BALANCED) -> ModelRouter:
    """获取全局路由器单例"""
    global _router
    if _router is None:
        _router = ModelRouter(strategy)
    return _router


# ========== 便捷函数 ==========

def create_fallback_client() -> Any:
    """创建带有 Fallback 能力的客户端"""
    router = get_router()
    return router.get_client()