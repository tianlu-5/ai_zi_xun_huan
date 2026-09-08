"""
Ollama本地推理客户端 - 对接本地Ollama服务运行DeepSeek等开源模型
与DoubaoClient/DoubaoWebClient实现相同接口，可互相替换

架构: HTTP REST API 调用 Ollama (http://localhost:11434)
      Ollama 加载 GGUF 模型，推理全在本地 GPU/CPU 进行

硬件约束 (RTX 5060 8GB):
  - 仅支持 7B Q4_K_M 量化模型 (约 4.7GB)
  - 最大上下文 8192 token
  - 禁止 14B 及以上模型
  - 优先 GPU 推理，显存不够时自动回退 CPU
"""
import os
import sys
import json
import re
import time
import urllib.request
import urllib.error
from typing import List, Dict, Optional, Any
from pathlib import Path


def _extract_quant(filename: str) -> str:
    """从 GGUF 文件名提取量化类型: Q4_K_M, Q5_K_S, Q8_0 等"""
    m = re.search(r'(Q\d(?:_\w+)?|FP\d+|F\d+)', filename, re.IGNORECASE)
    return m.group(1) if m else "unknown"


# LM Studio 风格模型名 → Ollama 标准模型名 映射
# 唯一模型: qwen2.5-coder:14b — 不再需要多模型映射
_MODEL_NAME_MAP = {}


def normalize_model_name(model_id: str, backend: str = "ollama") -> str:
    """模型名规范化（已精简为单一模型，保留接口兼容）"""
    if not model_id:
        return model_id
    return model_id


# ── ModelProfile: 模型能力档案（自适应参数调优） ──

class ModelProfile:
    """
    根据模型名称/元数据推断能力边界，自动计算预算、max_tokens、temperature 等参数。
    支持两种来源：
      1) 从名称推断（离线，快速但粗略）
      2) 从 Ollama /api/show 或 LM Studio /v1/models/{id} 拉取真实 metadata（精确）
    """

    # 模型家族 → 默认 context_length (tokens)
    # 步骤4：ctx 从 32768 下调至 16384，减轻 32k 上下文带来的显存、分词压力
    _KNOWN_CONTEXTS = {
        "qwen2.5": 16384,
        "qwen2.5-coder": 16384,
    }

    # 参数量 → 推荐 context（tokens，考虑 GPU 显存）
    _PARAM_TO_CONTEXT = {
        "1.3b": 8192,
        "3b": 8192,
        "4b": 8192,
        "6.7b": 16384,
        "7b": 16384,
        "8b": 16384,
        "9b": 16384,
        "14b": 16384,
        "32b": 8192,
    }

    # 推理（R1）模型标识
    _REASONING_TAGS = ("r1", "reasoner", "thinking", "distill-r1")

    def __init__(self, base_url: str = "", model_id: str = "",
             temperature: float = -1.0, num_ctx: int = 0,
             num_gpu: int = -1, keep_alive: str = "5m",
             num_predict: int = 0, top_p: float = 0.95,
             backend_id: Optional[str] = None):  # ← 添加 backend_id 参数
        """
        初始化 Ollama 客户端
        :param backend_id: 后端标识（用于路由器追踪）
        """
        # 静态属性（从模型名/服务端推断）
        self.model_id: str = ""
        self.base_url: str = ""
        self.backend: str = "ollama"
        self.context_length: int = 16384          # tokens
        self.param_size: str = "7b"               # e.g. "7b", "14b"
        self.is_reasoning: bool = False            # 是否带 think 块
        self.speed_tok_per_sec: float = 5.0        # 预估速度（推理后更新）
        self._profile_ready: bool = False

        # 运行时观测数据（v2 新增）
        self._observations: List[Dict] = []        # 最近的推理观测记录
        self._consecutive_timeouts: int = 0
        self._consecutive_json_fails: int = 0
        self._consecutive_success: int = 0
        self._avg_tok_per_sec: float = 0.0
        self._total_calls: int = 0
        self._total_failures: int = 0

        # 自适应参数（根据观测动态调整，None=未调整用默认）
        self._adapted_ctx: Optional[int] = None
        self._adapted_num_predict: Optional[int] = None
        self._adapted_temperature: Optional[float] = None
        self._adapted_budget_factor: float = 1.0   # 预算缩放因子 (1.0=不缩放)

        # 持久化状态路径（由 OllamaClient 注入）
        self._state_path: Optional[Path] = None
        self._state_loaded: bool = False
        if "14b" in self.model_id.lower():
            self.num_predict = 3000
            print(f"[OllamaClient] 14B 模型强制设定 num_predict=3000")

    def infer_from_name(self, model_id: str) -> "ModelProfile":
        """从模型名称推断能力（离线快速路径）"""
        self.model_id = model_id.lower()

        # 参数量
        size_match = re.search(r'(\d+(?:\.\d+)?)\s*[Bb]', model_id)
        if size_match:
            raw = float(size_match.group(1))
            self.param_size = f"{raw:g}b"
        else:
            self.param_size = "7b"

        # 推理模型
        mid_lower = model_id.lower()
        self.is_reasoning = any(tag in mid_lower for tag in self._REASONING_TAGS)

        # context 窗口
        ctx = 16384
        for family, default_ctx in self._KNOWN_CONTEXTS.items():
            if family in mid_lower:
                ctx = default_ctx
                break

        # 参数量覆盖（必须用 == 精确匹配，不能用 in 子串匹配：
        #   否则 "4b" 会先匹配到 "14b"，把 14B 模型误判成 4B 的 8K ctx）
        for ps_key, ps_ctx in self._PARAM_TO_CONTEXT.items():
            if ps_key == self.param_size:
                ctx = ps_ctx
                break

        # RTX 5060 8GB 安全上限（步骤4：14B 上限从 8192 上调到 16384）
        #   原 8K 过于保守，导致 num_predict 预算被压到 1500；16k 可兼顾显存与生成质量
        try:
            from config import Config
            model_size_gb = 0
            if self.param_size in ("14b", "32b"):
                ctx = min(ctx, 16384)  # 14B+ 模型上限 16K context
        except Exception:
            pass

        self.context_length = ctx
        self._profile_ready = True
        return self

    def fetch_metadata(self, base_url: str, backend: str = "ollama") -> "ModelProfile":
        """从服务端拉取真实模型元数据（Ollama /api/show 或 LM Studio /v1/models/{id}）"""
        import urllib.request
        self.base_url = base_url
        self.backend = backend
        if not self.model_id:
            self.infer_from_name("unknown")
            return self

        try:
            if backend == "lmstudio":
                url = f"{base_url}/models/{self.model_id}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read())
                # LM Studio 元数据
                ctx = data.get("context_length") or data.get("max_context_length")
                if ctx and isinstance(ctx, (int, float)):
                    self.context_length = int(ctx)
                    self._profile_ready = True
            else:
                url = f"{base_url}/api/show"
                payload = json.dumps({"name": self.model_id}).encode()
                req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read())
                # Ollama 元数据
                info = data.get("model_info", {})
                keys = [k for k in info.keys() if "context_length" in k]
                if keys:
                    self.context_length = int(info[keys[0]])
                    self._profile_ready = True
        except Exception as e:
            print(f"[ModelProfile] 拉取元数据失败 ({backend}): {e}")

        if not self._profile_ready:
            self.infer_from_name(self.model_id)

        # ★ 步骤4：硬性上限 16384，防止 Ollama 报告 32768 造成显存/分词压力
        #   qwen2.5-coder 原生支持 32k，但 RTX 5060 8GB 跑 32k 显存吃紧、分词慢
        if self.context_length > 16384:
            print(f"[ModelProfile] ctx {self.context_length} → 16384 (步骤4 硬性上限)")
            self.context_length = 16384
        return self

    # ── 运行时观测（v2 新增）──

    def observe_inference(self, elapsed: float, eval_count: int = 0,
                          success: bool = True, json_parsed: bool = True):
        """
        记录一次推理观测，触发自适应参数调整。
        应在每次 LLM 调用返回后调用。

        :param elapsed: 推理耗时（秒）
        :param eval_count: 生成 token 数（0=未知）
        :param success: 推理是否成功（False=超时/异常）
        :param json_parsed: 结构化输出是否解析成功
        """
        tok_per_sec = round(eval_count / elapsed, 1) if elapsed > 0 and eval_count > 0 else 0
        obs = {
            "elapsed": round(elapsed, 2),
            "eval_count": eval_count,
            "tok_per_sec": tok_per_sec,
            "success": success,
            "json_parsed": json_parsed,
        }
        self._observations.append(obs)
        self._observations = self._observations[-50:]  # 保留最近50次

        self._total_calls += 1
        if not success:
            self._total_failures += 1

        # 更新连续计数
        if not success:
            self._consecutive_timeouts += 1
            self._consecutive_success = 0
        else:
            self._consecutive_timeouts = 0
            self._consecutive_success += 1

        if success and not json_parsed:
            self._consecutive_json_fails += 1
        elif json_parsed:
            self._consecutive_json_fails = 0

        # 更新平均速度
        valid_speeds = [o["tok_per_sec"] for o in self._observations if o["tok_per_sec"] > 0]
        if valid_speeds:
            self._avg_tok_per_sec = round(sum(valid_speeds) / len(valid_speeds), 1)
            self.speed_tok_per_sec = self._avg_tok_per_sec

        # 触发自适应调整
        self._adapt_params()
        self.save_state()

    def _adapt_params(self):
        """根据观测历史动态调整参数（三层信号：超时/JSON失败/速度慢/连续成功恢复）"""
        # 信号 1：连续超时 ≥2 → 减小 ctx 到 80%，budget 因子 ×0.7
        if self._consecutive_timeouts >= 2:
            base_ctx = self._adapted_ctx or self.context_length
            new_ctx = int(base_ctx * 0.8)
            if new_ctx != self._adapted_ctx:
                self._adapted_ctx = new_ctx
                self._adapted_budget_factor = max(0.5, round(self._adapted_budget_factor * 0.7, 2))
                self._log_adaptation(
                    f"连续{self._consecutive_timeouts}次超时 → ctx {base_ctx}→{new_ctx}, "
                    f"budget×{self._adapted_budget_factor:.1f}"
                )

        # 信号 2：JSON 解析连续失败 ≥2 → 降低 temperature（★ 问题4修复：不再降到 0.0）
        #   低温会让模型死板、更容易虚构补丁文本；保底锁定 0.45
        if self._consecutive_json_fails >= 2:
            floor_temp = 0.45
            if self._adapted_temperature is None or self._adapted_temperature > floor_temp:
                old_temp = self._adapted_temperature if self._adapted_temperature is not None else "default"
                self._adapted_temperature = floor_temp
                self._log_adaptation(
                    f"连续{self._consecutive_json_fails}次JSON失败 → temp {old_temp}锁定到{floor_temp}（不降到0，避免模型死板虚构补丁）"
                )

        # 信号 3：速度太慢 → 减小 num_predict（★ 硬改1：已停用，速度不应缩减输出配额，
        #   否则 JSON-patch 被截断导致解析失败，形成恶性循环）
        # if self._avg_tok_per_sec > 0 and self._avg_tok_per_sec < 3.0:
        #     base_pred = self._adapted_num_predict or self._calculate_num_predict_base()
        #     new_pred = int(base_pred * 0.7)
        #     if new_pred != self._adapted_num_predict:
        #         self._adapted_num_predict = new_pred
        #         self._log_adaptation(
        #             f"平均速度{self._avg_tok_per_sec:.1f}tok/s过低 → "
        #             f"num_predict {base_pred}→{new_pred}"
        #         )

        # 信号 4：连续成功 ≥5 → 尝试恢复参数
        if self._consecutive_success >= 5:
            recovered = False
            if self._adapted_ctx and self._adapted_ctx < self.context_length:
                old_ctx = self._adapted_ctx
                self._adapted_ctx = min(self.context_length, int(self._adapted_ctx * 1.1))
                if self._adapted_ctx != old_ctx:
                    recovered = True
                    self._log_adaptation(
                        f"连续{self._consecutive_success}次成功 → 恢复 ctx {old_ctx}→{self._adapted_ctx}"
                    )
            if self._adapted_budget_factor < 1.0:
                old_factor = self._adapted_budget_factor
                self._adapted_budget_factor = min(1.0, round(self._adapted_budget_factor * 1.1, 2))
                if self._adapted_budget_factor != old_factor:
                    recovered = True
                    self._log_adaptation(
                        f"连续{self._consecutive_success}次成功 → "
                        f"恢复 budget {old_factor:.1f}→{self._adapted_budget_factor:.1f}"
                    )
            # 成功后重置连续计数，避免每轮都触发恢复
            if recovered:
                self._consecutive_success = 0

    def _log_adaptation(self, msg: str):
        """打印参数调整日志"""
        print(f"  [ModelProfile 自适应] {msg}")

    # ── 持久化（v2 新增）──

    def set_state_path(self, path: Path):
        """设置持久化状态文件路径（由 OllamaClient 注入）"""
        self._state_path = path
        if not self._state_loaded:
            self.load_state()

    def save_state(self):
        """持久化观测和调整到 logs/model_profile.json"""
        if not self._state_path:
            return
        try:
            state = {
                "model_id": self.model_id,
                "param_size": self.param_size,
                "context_length": self.context_length,
                "is_reasoning": self.is_reasoning,
                "runtime": {
                    "total_calls": self._total_calls,
                    "total_failures": self._total_failures,
                    "consecutive_timeouts": self._consecutive_timeouts,
                    "consecutive_json_fails": self._consecutive_json_fails,
                    "consecutive_success": self._consecutive_success,
                    "avg_tok_per_sec": round(self._avg_tok_per_sec, 1),
                },
                "adapted_params": {
                    "ctx": self._adapted_ctx,
                    "num_predict": self._adapted_num_predict,
                    "temperature": self._adapted_temperature,
                    "budget_factor": round(self._adapted_budget_factor, 2),
                },
                "recent_observations": self._observations[-20:],
            }
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def load_state(self):
        """从文件加载历史观测和调整"""
        if not self._state_path or not self._state_path.exists():
            self._state_loaded = True
            return
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            # 仅当 model_id 匹配时才加载（避免切换模型后用旧参数）
            if state.get("model_id") and state["model_id"].lower() != self.model_id.lower():
                self._state_loaded = True
                return

            runtime = state.get("runtime", {})
            self._total_calls = runtime.get("total_calls", 0)
            self._total_failures = runtime.get("total_failures", 0)
            self._consecutive_timeouts = runtime.get("consecutive_timeouts", 0)
            self._consecutive_json_fails = runtime.get("consecutive_json_fails", 0)
            self._consecutive_success = runtime.get("consecutive_success", 0)
            self._avg_tok_per_sec = runtime.get("avg_tok_per_sec", 0.0)
            self.speed_tok_per_sec = self._avg_tok_per_sec

            adapted = state.get("adapted_params", {})
            self._adapted_ctx = adapted.get("ctx")
            self._adapted_num_predict = adapted.get("num_predict")
            self._adapted_temperature = adapted.get("temperature")
            self._adapted_budget_factor = adapted.get("budget_factor", 1.0)
            self._state_loaded = True
        except Exception:
            self._state_loaded = True

    # ── 自适应参数计算（考虑运行时调整）──

    def _calculate_num_predict_base(self) -> int:
        """基础 num_predict（不考虑运行时调整）"""
        if self.is_reasoning:
            return 2500  # R1 系列 think + JSON
        # 非 reasoning 模型：根据 param_size 调整
        # 4B 模型上下文有限，但仍需足够空间输出完整 JSON（多文件修改时可能很长）
        if self.param_size in ("4b", "3b", "1b"):
            return 1800  # 4B 小模型：给 1800 token 输出空间
        # ★ 关键改动：14B 模型显式分配 3000 token，避免 JSON 截断
        if self.param_size in ("14b", "13b", "16b", "20b"):
            return 3000
        return 2200  # 7B/8B 普通模型：1500 token 足够输出多文件 JSON

    def calculate_budget(self, round_type: str = "full") -> int:
        """
        计算 prompt 中可用于源代码的字符预算（chars）。
        考虑运行时自适应：超时多→budget 因子缩小。
        round_type: "full" 全覆盖 or "core" 核心集
        """
        # tokens → chars 转换（中文≈1.5 char/token，英文≈4 char/token，保守取 2.5）
        token_to_char = 2.5
        # 使用运行时调整后的 ctx（如果有）
        ctx = self._adapted_ctx or self.context_length
        total_chars = int(ctx * token_to_char)
        # 预留给 system prompt + 模型回复（保守 40%）
        reserved = int(total_chars * 0.4)
        budget = total_chars - reserved

        if round_type == "core":
            budget = int(budget * 0.85)  # 核心集稍紧凑

        # 显存保护
        if self.param_size in ("14b", "32b"):
            budget = min(budget, 8000)   # 14B+ 强限流
        else:
            budget = min(budget, 20000)  # 7B/8B 上限 20K

        # 运行时预算因子（超时多→缩小，成功多→恢复）
        budget = int(budget * self._adapted_budget_factor)

        return max(3000, budget)

    def calculate_num_predict(self) -> int:
        """计算模型最大生成 token 数（考虑运行时调整）"""
        # ★ 硬改1：硬性下限 800，防止自适应缩减导致 JSON-patch 截断
        if self._adapted_num_predict:
            return max(800, self._adapted_num_predict)
        return max(800, self._calculate_num_predict_base())

    def calculate_temperature(self) -> float:
        """计算推荐采样温度（考虑运行时调整）"""
        if self._adapted_temperature is not None:
            return self._adapted_temperature
        if self.is_reasoning:
            return 0.0  # R 需要确定性 JSON 输出
        return 0.3


    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "param_size": self.param_size,
            "context_length": self.context_length,
            "is_reasoning": self.is_reasoning,
            "speed_tok_per_sec": round(self.speed_tok_per_sec, 1),
            "avg_tok_per_sec": round(self._avg_tok_per_sec, 1),
            "total_calls": self._total_calls,
            "failure_rate": round(self._total_failures / max(self._total_calls, 1), 2),
            "adapted": {
                "ctx": self._adapted_ctx or self.context_length,
                "num_predict": self.calculate_num_predict(),
                "temperature": self.calculate_temperature(),
                "budget_factor": round(self._adapted_budget_factor, 2),
            },
        }


class OllamaClient:
    """
    Ollama本地推理客户端
    实现与 DoubaoClient 完全相同的接口
    """

    DEFAULT_BASE_URL = "http://127.0.0.1:11434"
    SUPPORTED_MODELS = {
        "qwen2.5-coder:14b": "qwen2.5-coder:14b",
    }

    MAX_CONTEXT = 16384      # RTX 5060 8GB 安全上限（KV cache ~2.3GB）
    MAX_MODEL_SIZE_GB = 10   # qwen2.5-coder:14b ~8.4GB
    MAX_PREDICT = 2500       # 推理模型: think ~1500 + JSON ~1000

    def __init__(self, base_url: str = "", model_id: str = "",
                 temperature: float = -1.0, num_ctx: int = 0,
                 num_gpu: int = -1, keep_alive: str = "0",
                 num_predict: int = 0, top_p: float = 0.95):
        from config import Config
        self.base_url = base_url or Config.OLLAMA_BASE_URL or self.DEFAULT_BASE_URL
        raw_model_id = model_id or Config.OLLAMA_MODEL or "qwen2.5-coder:14b"

        # 检测后端类型
        self._backend: str = "ollama"
        if "1234" in self.base_url or "/v1" in self.base_url:
            self._backend = "lmstudio"

        # 全链路模型名规范化（修正 LM Studio 风格名 + 拼写错误）
        normalized = normalize_model_name(raw_model_id, self._backend)
        if normalized != raw_model_id:
            print(f"[OllamaClient] 模型名规范化: {raw_model_id} → {normalized}")
            raw_model_id = normalized
            # 同步更新 Config（持久化正确模型名，避免下次再走映射）
            if self._backend == "ollama":
                try:
                    if Config.OLLAMA_MODEL and normalize_model_name(Config.OLLAMA_MODEL, "ollama") != Config.OLLAMA_MODEL:
                        Config.OLLAMA_MODEL = normalized
                        Config.auto_save()
                except Exception:
                    pass
            else:
                try:
                    if Config.LMSTUDIO_MODEL and normalize_model_name(Config.LMSTUDIO_MODEL, "lmstudio") != Config.LMSTUDIO_MODEL:
                        Config.LMSTUDIO_MODEL = normalized
                        Config.auto_save()
                except Exception:
                    pass
        self.model_id = raw_model_id

        # [DIAG] 定位 14B 客户端创建来源
        if "14b" in str(self.model_id).lower():
            import traceback as _tb
            print(f"[DIAG] OllamaClient created with 14B: {self.model_id}")
            _tb.print_stack(limit=8)

        # 构建 ModelProfile：先从名称推断，再尝试拉取服务端 metadata
        self.profile = ModelProfile()
        self.profile.infer_from_name(self.model_id)
        self.profile.fetch_metadata(self.base_url, self._backend)
        # v2: 注入持久化路径，加载历史运行时观测和自适应参数
        self.profile.set_state_path(Config.LOG_DIR / "model_profile.json")

        # 参数自适应：只有调用方显式传入非零/非 -1 时才覆盖
        self.temperature = temperature if temperature >= 0 else self.profile.calculate_temperature()
        self.top_p = top_p
        self.num_ctx = num_ctx if num_ctx > 0 else self.profile.context_length
        # ★ 显存安全：8GB VRAM 下 profile 推断的 16384 会让 7B KV 缓存叠加后 OOM（Ollama 500）
        #   调用方显式传 num_ctx 时优先；否则回落到 Config.OLLAMA_NUM_CTX（8192），仅当该值异常才用 profile
        try:
            from config import Config
            cfg_ctx = int(getattr(Config, "OLLAMA_NUM_CTX", 0) or 0)
            if num_ctx <= 0 and cfg_ctx > 0:
                self.num_ctx = cfg_ctx
        except Exception:
            pass
        self.num_ctx = min(self.num_ctx, self.MAX_CONTEXT)
        self.num_predict = num_predict if num_predict > 0 else self.profile.calculate_num_predict()
        self.num_gpu = num_gpu
        self.keep_alive = keep_alive

        self.api_key = ""
        self.max_tokens = 0
        self.conversation_history: List[Dict[str, str]] = []

        self._connected = False
        self._model_loaded = False
        # 防止同一模型重复触发 pull（进程级去重）
        if not hasattr(OllamaClient, "_pull_triggered"):
            OllamaClient._pull_triggered = set()

        print(f"[OllamaClient] 初始化: backend={self._backend}, model={self.model_id}")
        print(f"  ModelProfile: {self.profile.param_size}, ctx={self.profile.context_length}, "
              f"reasoning={self.profile.is_reasoning}, "
              f"budget={self.profile.calculate_budget('full')}~{self.profile.calculate_budget('core')}chars, "
              f"num_predict={self.num_predict}, temp={self.temperature}")

        # 默认系统提示词
        self.default_system = (
            "你是一个本地AI助手，运行在离线环境中。"
            "回答要简洁准确，优先给出实用代码或具体步骤。"
            "如果需要输出JSON，请直接输出纯JSON，不要markdown代码块。"
            "使用中文回答用户问题。"
        )

    # ========== 连接检查 ==========
    def check_connection(self) -> bool:
        """检查本地推理服务是否可达（Ollama / LM Studio 通用）"""
        if self._backend == "lmstudio":
            try:
                req = urllib.request.Request(f"{self.base_url}/models")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    self._connected = (resp.status == 200)
                    return self._connected
            except Exception as e:
                print(f"[LM Studio] 连接失败: {e}")
                self._connected = False
                return False
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self._connected = (resp.status == 200)
                return self._connected
        except Exception as e:
            print(f"[Ollama] 连接失败: {e}")
            self._connected = False
            return False

    def list_models(self) -> List[Dict]:
        """列出本地已安装的模型"""
        if self._backend == "lmstudio":
            try:
                req = urllib.request.Request(f"{self.base_url}/models")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    return data.get("data", [])
            except Exception as e:
                print(f"[LM Studio] 获取模型列表失败: {e}")
                return []
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return data.get("models", [])
        except Exception as e:
            print(f"[Ollama] 获取模型列表失败: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """拉取模型 (长操作，建议单独执行)
        返回 True 仅当模型确实被下载并出现在 list_models() 中
        """
        if self._backend == "lmstudio":
            print("[LM Studio] 拉取模型请在 LM Studio 应用内操作，或通过 OpenAI 兼容 API 创建")
            return False
        print(f"[Ollama] 正在拉取模型: {model_name} ...")
        try:
            data = json.dumps({"name": model_name, "stream": True}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/pull",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            got_progress = False
            error_msg = ""
            with urllib.request.urlopen(req, timeout=600) as resp:
                for line in resp:
                    try:
                        chunk = json.loads(line.decode().strip())
                        status = chunk.get("status", "")
                        if "total" in chunk and "completed" in chunk:
                            got_progress = True
                            total = chunk["total"]
                            done = chunk["completed"]
                            pct = done * 100 / max(total, 1)
                            print(f"  进度: {pct:.1f}%  ({done}/{total})", end="\r")
                        elif status == "success":
                            got_progress = True
                            print(f"  状态: {status}")
                        else:
                            print(f"  状态: {status}")
                        # Ollama 在 stream 中也可能返回错误
                        if "error" in chunk:
                            error_msg = chunk["error"]
                    except json.JSONDecodeError:
                        continue
            print()

            # ── 关键：拉取后验证模型确实出现在 list_models 中 ──
            # Ollama /api/pull 即使模型不存在也会返回空 stream 假装成功
            if error_msg:
                print(f"[Ollama] ❌ 拉取失败: {error_msg}")
                return False

            # 刷新已安装列表并查找该模型
            installed = self.list_models()
            short_name = model_name.split(":")[0].split("/")[-1].lower()
            found = False
            for m in installed:
                mid = (m.get("name") or m.get("id") or "").lower()
                installed_short = mid.split(":")[0].split("/")[-1]
                if mid == model_name.lower() or installed_short == short_name or mid.startswith(short_name):
                    found = True
                    break

            if found and got_progress:
                print(f"[Ollama] ✅ 模型拉取完成并已安装: {model_name}")
                return True
            elif found and not got_progress:
                # 没看到进度条但模型在 → 可能之前已安装
                print(f"[Ollama] ℹ️ 模型 {model_name} 已存在（无需重新拉取）")
                return True
            else:
                # 关键：stream 结束但模型不在列表中 → Ollama 库中不存在该模型
                print(f"[Ollama] ❌ 拉取流程结束，但模型 {model_name} 未出现在已安装列表中")
                print(f"   可能原因: Ollama 官方库中不存在此模型名")
                print(f"   建议: 1) 检查 https://ollama.com/library 是否有该模型")
                print(f"         2) 改用 LM Studio 加载本地 GGUF 文件")
                return False
        except urllib.error.HTTPError as e:
            print(f"[Ollama] ❌ 拉取失败: HTTP {e.code} - {e.reason}")
            if e.code == 404:
                print(f"   模型 {model_name} 在 Ollama 库中不存在")
            return False
        except Exception as e:
            print(f"[Ollama] 拉取异常: {type(e).__name__}: {e}")
            return False

    @staticmethod
    def scan_lmstudio_models(models_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        直接扫描 LM Studio 本地模型目录（不依赖 LM Studio 运行）
        找到所有完整下载的 GGUF 文件（排除 .part 未完成下载）
        :param models_dir: 目录路径，None 则用 Config.LMSTUDIO_MODELS_DIR
        :return: [{"id": "完整模型ID", "path": "绝对路径", "size_gb": float, "param_size": str}]
        """
        if models_dir is None:
            try:
                from config import Config
                models_dir = Config.LMSTUDIO_MODELS_DIR
            except Exception:
                models_dir = str(Path.home() / ".lmstudio" / "models")

        root = Path(models_dir)
        if not root.exists():
            return []

        results = []
        # 遍历: 作者/模型名/model.gguf
        for gguf in root.rglob("*.gguf"):
            if gguf.name.endswith(".part"):
                continue  # 跳过未完成下载
            size_bytes = gguf.stat().st_size
            size_gb = round(size_bytes / (1024 ** 3), 2)

            # 从路径推断模型 ID: author/model-family:variant
            # 路径: .lmstudio/models/Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF/Qwen3.5-9B-DeepSeek-V4-Flash-Q4_K_M.gguf
            parts = gguf.relative_to(root).parts
            if len(parts) >= 3:
                author = parts[0]
                family_dir = parts[1]  # e.g. "Qwen3.5-9B-DeepSeek-V4-Flash-GGUF"
                filename = gguf.stem  # e.g. "Qwen3.5-9B-DeepSeek-V4-Flash-Q4_K_M"
                # 把 GGUF 文件名映射为 LM Studio API 风格的 ID
                # 去掉作者名里的组织前缀，保留 family + quant
                model_id = f"{author}/{filename}"
            else:
                model_id = gguf.stem

            # 从文件名粗略推断参数量级
            param_size = "unknown"
            size_match = re.search(r'(\d+(?:\.\d+)?)\s*([Bb])', gguf.stem)
            if size_match:
                num = float(size_match.group(1))
                unit = size_match.group(2).upper()
                param_size = f"{num}{unit}"

            results.append({
                "id": model_id,
                "path": str(gguf),
                "size_gb": size_gb,
                "param_size": param_size,
                "quant": _extract_quant(gguf.stem),
            })

        results.sort(key=lambda m: m["size_gb"], reverse=True)
        return results

    def ensure_model(self, model_name: str = "") -> bool:
        """确保模型可用，不可用则提示拉取"""
        raw_name = model_name or self.model_id
        # 先做模型名规范化（防止 LM Studio 风格 + 拼写错误进入匹配逻辑）
        name = normalize_model_name(raw_name, self._backend)
        if name != raw_name:
            self.model_id = name

        models = self.list_models()
        # 规范化匹配：小写化 + 去掉后缀 (:quant) + 去掉作者前缀
        # 扩展：映射后的 Ollama 标准名也加入候选（gemma4:e4b 等）
        candidates = {name.lower()}
        # 映射表中的目标名也加入匹配候选（避免第一次没初始化就直接 match 失败）
        for mapped in _MODEL_NAME_MAP.values():
            candidates.add(mapped.lower())
        # gemma4:e4b 等 tag 格式的短名也加入候选
        short_name = name.split(":")[0].split("/")[-1].lower()
        candidates.add(short_name)

        for m in models:
            model_id = (m.get("name") or m.get("id") or "").lower()
            registered_short = model_id.split(":")[0].split("/")[-1]
            # ★ 修复 ensure_model 换芯 Bug：同家族不同权重（7b / 14b）必须精确匹配 tag，
            #   否则配置的 7b 会被列表里排前面的 14b 悄悄替换 -> 8GB VRAM OOM (Ollama 500)
            name_variant = name.split(":")[1].split("/")[-1] if ":" in name else ""
            candidate_variant = model_id.split(":")[1].split("/")[-1] if ":" in model_id else ""
            variant_ok = (
                not name_variant
                or not candidate_variant
                or candidate_variant == name_variant
                or name_variant in candidate_variant
                or candidate_variant in name_variant
            )
            matched = (
                variant_ok
                and (
                    registered_short == short_name
                    or model_id.startswith(short_name)
                    or model_id in candidates
                    or any(c and model_id.startswith(c) for c in candidates if c)
                )
            )
            if matched:
                size_gb = m.get("size", 0) / (1024 ** 3)
                if 0 < size_gb > self.MAX_MODEL_SIZE_GB:
                    tag = "LM Studio" if self._backend == "lmstudio" else "Ollama"
                    print(f"[{tag}] ⚠️ 模型 {name} 大小 {size_gb:.1f}GB 超过显存限制 {self.MAX_MODEL_SIZE_GB}GB")
                    return False
                self._model_loaded = True
                # 用实际注册的 ID 替换请求模型名
                actual_id = m.get("name") or m.get("id") or name
                if self._backend == "ollama":
                    # Ollama 模式优先使用规范化后的 tag 格式名（gemma4:e4b）
                    self.model_id = normalize_model_name(actual_id, "ollama")
                else:
                    self.model_id = actual_id
                return True
        tag = "LM Studio" if self._backend == "lmstudio" else "Ollama"
        print(f"[{tag}] 模型 {name} 未在已加载列表中")
        if self._backend == "lmstudio":
            # LM Studio 启用了 justInTimeModelLoading，允许即时加载未预加载的模型
            print(f"   ℹ️ LM Studio JIT 加载已启用，将在请求时自动加载模型")
            self._model_loaded = True
            return True
        else:
            # Ollama 模式：如果模型名是已知可拉取的（在映射表中或含已知 tag），自动触发 pull 一次
            # 识别标准模型名（映射表目标值，或常见 Ollama 格式 like abc:xyz）
            is_standard_name = (
                name in set(_MODEL_NAME_MAP.values())
                or (":" in name and not name.startswith("http"))
                or name in ["qwen2.5-coder:14b", "qwen2.5-coder:7b"]
            )
            if is_standard_name and name not in getattr(OllamaClient, "_pull_triggered", set()):
                OllamaClient._pull_triggered.add(name)
                print(f"   📥 检测到标准模型名，自动触发 pull（只触发一次）...")
                try:
                    if self.pull_model(name):
                        self._model_loaded = True
                        # 刷新 list_models 确认
                        refreshed = self.list_models()
                        for m in refreshed:
                            mid = (m.get("name") or m.get("id") or "").lower()
                            if mid == name.lower() or mid.split(":")[0].split("/")[-1] == name.split(":")[0].split("/")[-1]:
                                self.model_id = m.get("name") or m.get("id") or name
                                break
                        print(f"   ✅ 自动 pull 完成: {self.model_id}")
                        return True
                    else:
                        print(f"   ⚠️ 自动 pull 失败，请手动执行: ollama pull {name}")
                except Exception as e:
                    print(f"   ⚠️ 自动 pull 异常: {e}")
            print(f"   请执行: ollama pull {name}")
            return False

    # ========== 核心对话 ==========
    def chat(self, user_message: str, system_prompt: Optional[str] = None,
         stream: bool = False, extra_params: Optional[Dict] = None,
         force_json: bool = False) -> str:
        """
        发送消息并获取回复
        与 DoubaoClient.chat() 接口一致
        :param force_json: True 时通过 API 级 format/json_object 强制模型输出 JSON
        """
        import time
        
        if not self._connected and not self.check_connection():
            tag = "LM Studio" if self._backend == "lmstudio" else "Ollama"
            return f"[错误] {tag} 服务未连接。请确认服务已启动。"

        if not self._model_loaded and not self.ensure_model():
            return f"[错误] 模型 {self.model_id} 不可用"

        # 构建 messages
        messages: List[Dict] = []
        sys_msg = system_prompt if system_prompt else self.default_system
        messages.append({"role": "system", "content": sys_msg})
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})

        start_time = time.time()
        result = ""
        tokens_used = 0

        try:
            if self._backend == "lmstudio":
                result = self._chat_lmstudio(messages, stream, force_json=force_json)
            else:
                # Ollama native
                payload = {
                    "model": self.model_id,
                    "messages": messages,
                    "stream": stream,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict,
                    },
                    "keep_alive": self.keep_alive,
                }
                if force_json:
                    payload["format"] = "json"
                    payload["options"]["temperature"] = min(0.1, max(0.0, self.temperature - 0.25))
                if self.num_gpu != -1:
                    payload["options"]["num_gpu"] = self.num_gpu
                if extra_params:
                    payload.update(extra_params)
                if stream:
                    result = self._chat_stream(payload)
                else:
                    result = self._chat_sync(payload)
            
            # ===== 上报成功到路由器 =====
            if result and not result.startswith("[错误]") and not result.startswith("[API"):
                tokens_used = len(result) // 3
                try:
                    from model_router import get_router
                    elapsed = time.time() - start_time
                    router = get_router()
                    router.mark_success(self._backend_id, tokens_used, elapsed)
                except Exception:
                    pass
            # ============================
            
            return result
            
        except Exception as e:
            # ===== 上报失败到路由器 =====
            try:
                from model_router import get_router
                router = get_router()
                router.mark_failure(self._backend_id, str(e))
            except Exception:
                pass
            # ============================
            return f"[错误] 推理异常: {type(e).__name__}: {e}"

    def _chat_lmstudio(self, messages: List[Dict], stream: bool, force_json: bool = False) -> str:
        """LM Studio OpenAI 兼容 API 调用"""
        # LM Studio JIT 加载的模型默认 ctx 可能远小于 metadata 声称值（如 2048 而非 8192），
        # 当 num_predict + input 超出实际 ctx 时会返回 "Context size has been exceeded"。
        # 此处用 max_tokens 兜底重试：出错时降到一半再试一次。
        return self._chat_lmstudio_inner(messages, stream, self.num_predict, _ctx_retry=0, force_json=force_json)

    def _chat_lmstudio_inner(self, messages: List[Dict], stream: bool,
                              max_tokens: int, _ctx_retry: int = 0, force_json: bool = False) -> str:
        """LM Studio OpenAI 兼容 API 调用（内部，支持 context 超限重试一次）"""
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        # ── API 级强制 JSON：OpenAI 兼容格式 response_format ──
        if force_json:
            payload["response_format"] = {"type": "json_object"}
            payload["temperature"] = min(0.1, max(0.0, self.temperature - 0.25))
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        full_content = ""
        eval_count = 0

        try:
            if stream:
                with urllib.request.urlopen(req, timeout=1200) as resp:
                    for line in resp:
                        line = line.decode().strip()
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw)
                            piece = chunk["choices"][0].get("delta", {}).get("content", "")
                            if piece:
                                full_content += piece
                                print(piece, end="", flush=True)
                        except Exception:
                            continue
                elapsed = time.time() - t0
                if elapsed > 0:
                    print(f"\n[LM Studio] 耗时 {elapsed:.1f}s")
            else:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    result = json.loads(resp.read())
                full_content = result["choices"][0]["message"]["content"]
                elapsed = time.time() - t0
                usage = result.get("usage", {})
                eval_count = usage.get("completion_tokens", 0)
                if eval_count > 0 and elapsed > 0:
                    tok_per_sec = eval_count / elapsed
                    print(f"[LM Studio] 生成 {eval_count} tokens, 耗时 {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)")
        except urllib.error.HTTPError as e:
            elapsed = time.time() - t0
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="ignore")[:300]
            except Exception:
                pass

            # Context 超限兜底：减半 max_tokens 重试一次
            if _ctx_retry == 0 and "Context size" in err_body:
                new_max = max(256, max_tokens // 2)
                if new_max < max_tokens:
                    print(f"[LM Studio] Context 超限 → max_tokens {max_tokens}→{new_max} 重试")
                    # 同步降低 profile.num_predict，后续迭代也用更小值
                    self.num_predict = new_max
                    return self._chat_lmstudio_inner(messages, stream, new_max, _ctx_retry=1)

            print(f"[LM Studio] HTTP {e.code} 调用失败 (耗时 {elapsed:.1f}s): {err_body[:200]}")
            # 记录失败观测（触发自适应）
            self.profile.observe_inference(elapsed, 0, success=False)
            return f"[错误] API 调用失败: HTTP {e.code} - {e.reason}"
        except urllib.error.URLError as e:
            elapsed = time.time() - t0
            print(f"[LM Studio] URL 错误: {e.reason}")
            self.profile.observe_inference(elapsed, 0, success=False)
            return f"[错误] API 调用失败: {e.reason}"
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[LM Studio] 异常: {type(e).__name__}: {e}")
            self.profile.observe_inference(elapsed, 0, success=False)
            return f"[错误] API 调用失败: {type(e).__name__}: {e}"

        self.conversation_history.append({"role": "user", "content": messages[-1]["content"]})
        self.conversation_history.append({"role": "assistant", "content": full_content})
        # v2: 记录推理观测（触发自适应参数调整）
        self.profile.observe_inference(elapsed, eval_count, success=True)
        return full_content

    def _chat_sync(self, payload: Dict) -> str:
        """Ollama 原生非流式调用"""
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1200) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")[:400]
            except Exception:
                pass
            print(f"[DIAG] Ollama sync HTTP {e.code} body: {body}")
            self.profile.observe_inference(time.time() - t0, 0, success=False)
            return f"[错误] 推理异常: HTTPError: HTTP Error {e.code}: {e.reason} body={body[:200]}"
        elapsed = time.time() - t0

        content = result.get("message", {}).get("content", "")
        total_duration = result.get("total_duration", 0)
        load_duration = result.get("load_duration", 0)
        eval_count = result.get("eval_count", 0)

        if eval_count > 0 and total_duration > 0:
            tok_per_sec = eval_count / (total_duration / 1e9) if total_duration > 0 else 0
            print(f"[Ollama] 生成 {eval_count} tokens, "
                  f"耗时 {elapsed:.1f}s ({tok_per_sec:.1f} tok/s), "
                  f"模型加载 {load_duration/1e9:.1f}s")

        # ★ 修复A：检测空响应（eval_count=0 或 content 为空）
        #   根因：14B 模型在 8GB VRAM + num_ctx=16384 时可能 GPU OOM，
        #   Ollama 返回 HTTP 200 但 eval_count=0（模型未生成任何 token）
        #   原代码返回 "" 不加前缀 → run_iteration 第297行 not "" = True → 早返回
        #   但 status 设为 "API响应异常" 而非明确的空响应错误
        if not content or eval_count == 0:
            _err_detail = (
                f"[错误] Ollama 返回空内容 (eval_count={eval_count}, "
                f"load_duration={load_duration/1e9:.1f}s, "
                f"elapsed={elapsed:.1f}s)"
            )
            if load_duration > 0 and eval_count == 0:
                _err_detail += " → 疑似 GPU OOM 或模型加载失败"
            elif elapsed < 3.0 and eval_count == 0:
                _err_detail += " → 响应过快，疑似模型未真正加载"
            print(f"  ⚠️ {_err_detail}")
            self.profile.observe_inference(elapsed, 0, success=False)
            return _err_detail

        self.conversation_history.append({"role": "user", "content": payload["messages"][-1]["content"]})
        self.conversation_history.append({"role": "assistant", "content": content})
        # v2: 记录推理观测（触发自适应参数调整）
        self.profile.observe_inference(elapsed, eval_count, success=True)
        return content

    def _chat_stream(self, payload: Dict) -> str:
        """Ollama 原生流式调用"""
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        full_content = ""
        start_time = time.time()
        eval_count = 0

        try:
            with urllib.request.urlopen(req, timeout=1200) as resp:
                for line in resp:
                    try:
                        chunk = json.loads(line.decode().strip())
                        msg = chunk.get("message", {})
                        piece = msg.get("content", "")
                        if piece:
                            full_content += piece
                            print(piece, end="", flush=True)
                        if chunk.get("done"):
                            eval_count = chunk.get("eval_count", 0)
                            break
                    except json.JSONDecodeError:
                        continue
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")[:400]
            except Exception:
                pass
            print(f"[DIAG] Ollama stream HTTP {e.code} body: {body}")
            self.profile.observe_inference(time.time() - start_time, 0, success=False)
            return f"[错误] 推理异常: HTTPError: HTTP Error {e.code}: {e.reason} body={body[:200]}"

        elapsed = time.time() - start_time
        if eval_count > 0 and elapsed > 0:
            tok_per_sec = eval_count / elapsed
            print(f"\n[Ollama] 生成 {eval_count} tokens, "
                  f"{elapsed:.1f}s ({tok_per_sec:.1f} tok/s)")

        self.conversation_history.append({"role": "user", "content": payload["messages"][-1]["content"]})
        self.conversation_history.append({"role": "assistant", "content": full_content})
        return full_content

    # ========== 兼容接口 ==========
    def reset_conversation(self):
        """清空对话历史"""
        self.conversation_history.clear()

    def add_system_prompt(self, prompt: str):
        """添加系统提示词（会在下次 chat 时生效）"""
        self._system_prompt = prompt

    def chat_with_structured_output(self, user_message: str, system_prompt: str,
                                    stream: bool = False) -> Optional[Dict]:
        """对话并提取JSON结构化输出
        - 优先使用 API 级 force_json（Ollama format=json / LM Studio response_format），
          这是最可靠的强制 JSON 方式，模型采样阶段就被限制在合法 token 空间
        - 其次用 _extract_json + _repair_json 做软件层兜底
        """
        # v3: 不再在 user_message 尾部重复加文字约束（system_prompt 里已经有强约束了）
        #     重复加反而稀释权重，API 级 format/json 才是硬约束
        response = self.chat(
            user_message=user_message,
            system_prompt=system_prompt,
            stream=stream,
            force_json=True,  # ← 关键：API 级强制 JSON 输出
        )
        if not response or response.startswith("[错误]"):
            print(f"[Ollama] 结构化调用失败: {response[:100] if response else '空响应'}")
            return None
        data = self._extract_json(response)
        if data is None:
            # v2: JSON 解析失败，更新 profile 的 JSON 失败计数（触发温度自适应）
            self.profile._consecutive_json_fails += 1
            self.profile._adapt_params()
            self.profile.save_state()
            print(f"[Ollama] 无法提取JSON（剥壳后长度={len(response)}），原始响应前300字: {response[:300]}")
            # v3: 即使 force_json 也可能被截断或边界情况，再调用一次 _repair_json
            repaired = self._repair_json(response)
            try:
                import json as _json
                data = _json.loads(repaired)
                print(f"[Ollama] 二次修复JSON成功")
            except Exception:
                return None
        # v2: JSON 解析成功，重置连续失败计数
        self.profile._consecutive_json_fails = 0
        return self._coerce_schema(data)

    @staticmethod
    def _strip_think(text: str) -> str:
        """剥离 DeepSeek-R1 等推理模型的 <think>...</think> 标签及裸露推理块"""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _repair_json(text: str) -> str:
        """修复常见 JSON 格式问题：尾逗号、字符串内未转义换行、裸引号/反斜杠

        Gemma 4 E4B 会把 Python 代码直接塞进 old_str/new_str 而不做 JSON 转义：
          "old_str":"if __name__ == "__main__":"    ← 这里的 "__main__" 是裸引号

        核心难点：`"` 后跟 `:` 有两种含义
          - JSON 键结束：  "description":"..."   → `":` 是字符串结束
          - Python 代码值："if __name__ == "__main__":"  → `":` 是代码内容，不是结束

        解决方案：追踪 after_colon 标记区分键/值位置
          - 值位置（after_colon=True）：只有 ,}] 才是字符串结束（防止 Python 冒号误判）
          - 键位置（after_colon=False）：,:}] 都是合法的字符串结束（键不含裸引号）
        """
        # 1. 尾逗号
        text = re.sub(r',\s*([}\]])', r'\1', text)

        # 2. 字符串内裸引号转义（键/值位置感知状态机）
        result = []
        i = 0
        in_string = False
        esc = False
        after_colon = False  # True=下一个字符串是值, False=下一个字符串是键
        n = len(text)
        while i < n:
            c = text[i]
            if in_string:
                if esc:
                    result.append(c)
                    esc = False
                elif c == '\\':
                    result.append(c)
                    esc = True
                elif c == '"':
                    # 找下一个非空白字符
                    j = i + 1
                    while j < n and text[j] in ' \t\n\r':
                        j += 1
                    next_c = text[j] if j < n else ''

                    if after_colon:
                        # 值位置：只有 ,}] 才是字符串结束
                        # Python 代码中的 ":" 不会误判为结束
                        if next_c in ',}]':
                            result.append(c)
                            in_string = False
                            after_colon = False
                        else:
                            result.append('\\')
                            result.append('"')
                    else:
                        # 键位置：,:}] 都是合法结束
                        if next_c in ',:}]':
                            result.append(c)
                            in_string = False
                            after_colon = False
                        else:
                            result.append('\\')
                            result.append('"')
                elif c == '\n':
                    result.append('\\n')
                elif c == '\r':
                    result.append('\\r')
                elif c == '\t':
                    result.append('\\t')
                else:
                    result.append(c)
            else:
                if c == '"':
                    in_string = True
                    result.append(c)
                elif c == ':':
                    after_colon = True
                    result.append(c)
                elif c == ',':
                    after_colon = False  # 对象里下一个是键
                    result.append(c)
                elif c in '{}[]':
                    after_colon = False  # 新容器，下一个是键/元素
                    result.append(c)
                else:
                    result.append(c)
            i += 1

        # 3. 括号平衡兜底
        def _count_bal(s: str, lch: str, rch: str) -> int:
            d = 0; in_s = False; es = False
            for cc in s:
                if es: es = False
                elif cc == '\\': es = True
                elif cc == '"': in_s = not in_s
                elif not in_s:
                    if cc == lch: d += 1
                    elif cc == rch: d -= 1
            return d
        s = ''.join(result)
        brace_diff = _count_bal(s, '{', '}')
        if brace_diff > 0:
            s += '}' * brace_diff
        bracket_diff = _count_bal(s, '[', ']')
        if bracket_diff > 0:
            s += ']' * bracket_diff
        return s

    @staticmethod
    def _coerce_schema(data: Dict) -> Dict:
        """把模型输出的 JSON 兼容到内部 schema（对象→数组等）"""
        if not isinstance(data, dict):
            return {}
        mods = data.get("modifications", [])
        if isinstance(mods, dict):
            # R1 有时把 modifications 输出成对象
            data["modifications"] = [mods] if mods else []
        elif not isinstance(mods, list):
            data["modifications"] = []
        return data

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """从可能含推理噪声的文本中稳健提取 JSON"""
        text = OllamaClient._strip_think(text)

        # ========== 路径 1：直接解析（纯 JSON 理想情况）==========
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # ========== 路径 0.5：先剥 markdown ```json 壳（如果存在），避免后续括号/代码块搜索被混淆 ==========
        stripped_json_md = text
        # 1) ```json ... ``` / ```JSON ... ```
        md_m = re.search(r'```(?:json|JSON)\s*(.*?)\s*```', text, re.DOTALL)
        if md_m:
            stripped_json_md = md_m.group(1).strip()
        else:
            # 2) 通用 ``` ... ```：第一行若为 json/python/js 则去掉
            md_m2 = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if md_m2:
                block = md_m2.group(1).strip()
                if "\n" in block:
                    first, rest = block.split("\n", 1)
                    if first.strip().lower() in ("json", "python", "js", "javascript", "yaml"):
                        block = rest.strip()
                stripped_json_md = block
        # 剥壳之后再直接解析一次
        if stripped_json_md != text:
            try:
                return json.loads(stripped_json_md)
            except json.JSONDecodeError:
                pass

        # ========== 路径 2：括号截取（优先用剥壳后文本）==========
        working = stripped_json_md if stripped_json_md != text else text
        # 关键：先对 working 整体做 _repair_json，把字符串内的裸引号转义掉。
        # 否则下面的括号计数器用简单的 " 切换追踪字符串状态，遇到
        #   "old_str":"if __name__ == "__main__":"
        # 会误判字符串结束位置，导致切片错误/截断，_repair_json 再也救不回来。
        working = OllamaClient._repair_json(working)
        # 修复后可能直接就能解析
        try:
            return json.loads(working)
        except json.JSONDecodeError:
            pass
        # 使用"括号计数"定位最外层 { ... } 而不是简单 find/rfind
        brace_l = working.find("{")
        brace_r = -1
        if brace_l != -1:
            depth = 0
            i = brace_l
            in_str = False
            esc = False
            while i < len(working):
                c = working[i]
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    in_str = not in_str
                elif not in_str:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            brace_r = i
                            break
                i += 1

        array_l = working.find("[")
        array_r = -1
        if array_l != -1:
            depth = 0
            i = array_l
            in_str = False
            esc = False
            while i < len(working):
                c = working[i]
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    in_str = not in_str
                elif not in_str:
                    if c == '[':
                        depth += 1
                    elif c == ']':
                        depth -= 1
                        if depth == 0:
                            array_r = i
                            break
                i += 1

        candidates = []
        if brace_l != -1 and brace_r != -1 and brace_r > brace_l:
            sliced = working[brace_l:brace_r + 1]
            candidates.append(("brace", sliced))
        if array_l != -1 and array_r != -1 and array_r > array_l:
            sliced = working[array_l:array_r + 1]
            candidates.append(("array", sliced))

        for tag, raw in candidates:
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e1:
                repaired = OllamaClient._repair_json(raw)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError as e2:
                    # 最后手段：去掉结尾非 } 字符（AI 输出尾行可能多带 , ... } 之外的话）
                    end = repaired.rfind("}")
                    if end != -1 and end + 1 < len(repaired):
                        trimmed = repaired[:end + 1]
                        try:
                            return json.loads(trimmed)
                        except json.JSONDecodeError:
                            pass
                    # 调试：打印解析失败位置附近文本
                    pos = getattr(e2, "pos", getattr(e1, "pos", 0))
                    start = max(0, pos - 30)
                    end_p = min(len(raw), pos + 30)
                    ptr_str = " " * (pos - start) + "^"
                    print(f"  [JSON调试] {tag} 解析失败 e1=[{e1.msg}@pos{e1.pos}] e2=[{e2.msg}@pos{e2.pos}]")
                    print(f"  片段前200字: {raw[:200]}...")
                    print(f"  附近: {raw[start:end_p]}")
                    print(f"         {ptr_str}")
                    continue

        print(f"[Ollama] 无法从回复中提取JSON (剥壳后长度={len(stripped_json_md)}, 候选={len(candidates)})")
        print(f"  原始回复前300字: {text[:300]}...")
        if stripped_json_md != text:
            print(f"  剥壳后片段前300字: {stripped_json_md[:300]}...")

        # ========== 路径 4：截断 JSON 补全（模型 num_predict 用尽时输出被切断）==========
        # 检测：剥壳后文本以 { 开头但没有匹配的 } 结尾
        if stripped_json_md.startswith("{") and not stripped_json_md.rstrip().endswith("}"):
            # 找到最后一个完整 modification 对象的结尾（即最后一个 "}]" 之前的位置）
            # 多文件修改 JSON 结构：{"description":"...","modifications":[{...},{...},{...（截断）
            last_complete = stripped_json_md.rfind("}")
            if last_complete > 0:
                # 截到上一个完整 } 位置
                truncated = stripped_json_md[:last_complete + 1]
                # 补齐数组和对象结尾
                # 计算还需要补多少 ]
                open_arrays = truncated.count("[") - truncated.count("]")
                open_objects = truncated.count("{") - truncated.count("}")
                # 简单粗暴：直接补 ]} 试试（最常见截断情况）
                for trial_suffix in ["]}", "]}}", "]", "}"]:
                    candidate = truncated + trial_suffix
                    try:
                        result_dict = json.loads(candidate)
                        print(f"  ✅ [截断补全] 补 '{trial_suffix}' 后解析成功 (原始 {len(stripped_json_md)} chars → {len(candidate)} chars)")
                        # 但这种情况下，可能有 modification 项被丢弃了，记录警告
                        if isinstance(result_dict, dict):
                            mods = result_dict.get("modifications", [])
                            print(f"  ⚠️ 注意：模型输出可能被 num_predict 截断，仅恢复 {len(mods) if isinstance(mods, list) else 0} 个修改项（可能不完整）")
                        return result_dict
                    except json.JSONDecodeError:
                        continue
            print(f"  ⚠️ 检测到 JSON 可能被截断（num_predict={getattr(self, 'num_predict', '?')} 太小？），但补全尝试失败")
            print(f"     建议：增大 num_predict 或减少 SELF_EVOLVE_FILES 文件数")
        return None

    def get_conversation_token_count(self) -> int:
        """估算当前对话历史的token数"""
        total_chars = sum(len(m["content"]) for m in self.conversation_history)
        return total_chars // 3  # 粗略估算

    def trim_conversation(self, max_tokens: int = 6000):
        """裁剪对话历史"""
        while self.get_conversation_token_count() > max_tokens and len(self.conversation_history) > 4:
            self.conversation_history.pop(0)
            self.conversation_history.pop(0)  # 成对删除 user+assistant

    def get_system_info(self) -> Dict:
        """获取Ollama系统信息（GPU、模型等）"""
        info = {"connected": self._connected, "base_url": self.base_url}
        try:
            req = urllib.request.Request(f"{self.base_url}/api/version")
            with urllib.request.urlopen(req, timeout=3) as resp:
                info["version"] = json.loads(resp.read()).get("version", "unknown")
        except Exception:
            pass
        try:
            models = self.list_models()
            info["local_models"] = [m.get("name") for m in models]
            for m in models:
                if self.model_id in m.get("name", ""):
                    info["model_size_gb"] = round(m.get("size", 0) / (1024 ** 3), 2)
                    info["model_digest"] = m.get("digest", "")[:12]
                    break
        except Exception:
            pass
        return info

    def _init_client(self):
        """兼容接口 - 初始化客户端"""
        self.check_connection()
