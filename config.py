"""
配置管理模块 - 集中管理API密钥、模型配置、路径设置等
"""
import os

# Distributed mode configuration
DISTRIBUTED_MODE = os.getenv("DISTRIBUTED_MODE", "distributed")
# 启用分布式模式，支持协同重构
import multiprocessing as mp
from queue import Queue

# 分布式队列配置
DISTRIBUTED_QUEUE = Queue()
# 添加分布式模式支持，启用分布式协同重构
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any



def _load_dotenv(path: Optional[str] = None) -> None:
    """轻量 .env 加载器（无第三方依赖）：仅在环境变量未设置时写入 os.environ"""
    try:
        p = Path(path) if path else Path(__file__).parent / ".env"
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


_load_dotenv()


class Config:
    current_mode = 'normal'
    # State machine implementation: 'normal', 'migrating', 'hibernating'
    @classmethod
    def set_state(cls, new_state):
        if new_state in ['normal', 'migrating', 'hibernating']:
            cls.current_mode = new_state
            print(f"State changed to {new_state}")
        else:
            raise ValueError("Invalid state. Must be one of 'normal', 'migrating', 'hibernating'.")
    # Add state switching logic here. Define states and transitions for multi-stable system. States: 'normal', 'migrating', 'hibernating'. Transitions based on external triggers.
    # ========== 豆包API配置 ==========
    # 从环境变量读取API Key，推荐配置方式：
    # Windows PowerShell: $env:ARK_API_KEY = "你的API_Key"
    # 或在下方直接填入（不推荐，避免泄露密钥）
    ARK_API_KEY = os.getenv("ARK_API_KEY", "")

    # 方舟平台API Base URL
    ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

    # 使用的模型ID - 推荐使用代码能力强的模型
    # doubao-seed-2-1-pro-260628 为最新主力模型，代码能力强
    # 可在火山方舟控制台获取更多模型ID
    MODEL_ID = os.getenv("ARK_MODEL_ID", "doubao-seed-2-1-pro-260628")

    # ========== 路径配置 ==========
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.resolve()

    # 代码备份目录（每次修改前自动备份）
    BACKUP_DIR = PROJECT_ROOT / "backups"

    # 日志目录
    LOG_DIR = PROJECT_ROOT / "logs"

    # 迭代历史记录
    EVOLUTION_LOG = LOG_DIR / "evolution_history.json"

    # ========== 客户端模式配置 ==========
    # "api"      = 使用火山方舟API（付费，稳定，需ARK_API_KEY）
    # "local"    = 使用本地Ollama+开源模型（完全离线，需GPU）
    # "lmstudio" = 使用 LM Studio 本地服务（兼容 OpenAI API，端口 1234）
    # "jiyuan"   = 使用基元律动 TokenRhythm API（OpenAI 兼容，需 JIYUAN_API_KEY）
    CLIENT_MODE = os.getenv("DOUBAO_MODE", "local")

    # Ollama本地推理配置 (RTX 5060 8GB 显存约束)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")
    OLLAMA_NUM_CTX = 8192        # 安全上限：8GB 显存，KV cache ~2.3GB
    OLLAMA_NUM_GPU = -1           # GPU层数 (-1=全部上GPU, 0=纯CPU)
    OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "0")      # 模型在GPU保活时间（8GB 显存下用 0 每次卸载，避免叠加 OOM）
    OLLAMA_MAX_MODEL_GB = 10      # 允许加载的最大模型大小 (qwen2.5-coder:14b ~8.4GB)

    # LM Studio 本地推理配置（OpenAI 兼容 API，默认端口 1234）
    LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
    LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "qwen2.5-coder:14b")
    LMSTUDIO_NUM_CTX = 8192
    # LM Studio 本地模型存储目录（扫描已下载模型用）
    LMSTUDIO_MODELS_DIR = os.getenv("LMSTUDIO_MODELS_DIR", str(Path.home() / ".lmstudio" / "models"))

    # ========== 基元律动 TokenRhythm API 配置 ==========
    # OpenAI 兼容接口 (https://tokenrhythm.studio/v1)，Key 前缀 sk-tr_
    JIYUAN_BASE_URL = os.getenv("JIYUAN_BASE_URL", "https://tokenrhythm.studio/v1")
    JIYUAN_MODEL = os.getenv("JIYUAN_MODEL", "deepseek-v4-flash")
    JIYUAN_API_KEY = os.getenv("JIYUAN_API_KEY", "")

    # ========== 自我迭代配置 ==========
    # 需要纳入自迭代管理的文件（相对 PROJECT_ROOT 的路径）
    # Tier-3 完全开放 + Tier-2 受控开放，AI 可自由进化这些文件
    SELF_EVOLVE_FILES_CORE = [
        "evolver/branch_manager.py",
        "evolver/error_reflection.py",
        "evolver/veil_detector.py",
        "evolver/hypothesis_decoupler.py",
        "evolver/motive_miner.py",
        "evolver/perception_monitor.py",
        "evolver/benchmark.py",
        "evolver/fault_diagnosis.py",
        "evolver/seo_observer.py",
        "evolver/branch.py",
    ]

    # ── Agent Pipeline 模式（Route B）──
    # True = 5 步串行微智能体流水线（规划器→读取器→补丁工程师→校验卫士→反思器）
    # False = 单次大 Prompt 模式（原 run_iteration）
    USE_PIPELINE_MODE = False
    SELF_EVOLVE_FILES_AUX = []  # 已合并到 CORE
    SELF_EVOLVE_FILES = SELF_EVOLVE_FILES_CORE

    # PLE 嵌入层保护：这些文件在 daemon 模式下禁止自迭代修改
    # （防止 AI 改坏自己的核心引擎导致崩溃）
    PROTECTED_FILES = [
        "evolver/self_evolver.py",
        "evolver/autonomous_daemon.py",
        "evolver/ollama_client.py",
        "evolver/doubao_client.py",
        "evolver/modification_parser.py",
        "evolver/code_manager.py",
        "config.py",
    ]

    # ── 分层 PLE（Graduated PLE）──
    # Tier-1（绝对禁止）：PROTECTED_FILES + kill_switch 相关
    # Tier-2（受控开放）：允许 AI 修改，但修改后进入 3 轮观察期，
    #                    观察期内触发 kill-switch 则自动回滚
    # Tier-3（完全开放）：branch_manager.py
    PLE_TIER2_CONTROLLED = [
        "evolver/branch_manager.py",   # ←新增这一行
        "evolver/error_reflection.py",
        "evolver/veil_detector.py",
        "evolver/hypothesis_decoupler.py",
        "evolver/motive_miner.py",
        "evolver/perception_monitor.py",
        "evolver/benchmark.py",
        "evolver/fault_diagnosis.py",
        "evolver/seo_observer.py",
        "evolver/capability_registry.py",
        "evolver/contracts.py",
        "evolver/branch.py",
    ]
    # Tier-2 观察期：修改后 N 轮内 kill-switch 则回滚
    PLE_TIER2_OBSERVATION_ROUNDS = 3

    # 每次迭代最大修改文件数，防止过度修改
    MAX_MODIFICATIONS_PER_ITERATION = 5

    # 是否自动应用修改（False时需人工确认）
    AUTO_APPLY_MODIFICATIONS = False

    # 是否在修改前自动备份
    AUTO_BACKUP = True

    # ========== 模型调用参数 ==========
    # 生成温度，越低越确定，越高越创新（代码修改建议设低些）
    TEMPERATURE = 0.1  # 预案1：上调至 0.45 提升模型改动意愿，减少空修改

    # 最大生成token数
    MAX_TOKENS = 8192

    # ========== 验证配置 ==========
    # 修改后是否自动运行语法检查
    AUTO_SYNTAX_CHECK = True

    # 修改后是否运行简单导入测试
    AUTO_IMPORT_TEST = True

    # ========== PLE 完整性校验（P2） ==========
    # 7 个核心引擎文件哈希清单。首次运行时写入 config.json，
    # 之后每轮 daemon 启动对比，不匹配 → kill-switch。
    PLE_SALT_PREFIX = "PLE-SIG-v1:"
    # 在 Config 类中添加以下方法

    @classmethod
    def get_config_mtime(cls) -> float:
        """获取 config.json 的最后修改时间"""
        path = cls._config_json_path()
        if path.exists():
            return path.stat().st_mtime
        return 0.0

    @classmethod
    def reload_if_changed(cls, last_mtime: float) -> Tuple[bool, float, Dict[str, Any]]:
        """
        如果配置文件已修改，重新加载
        返回: (是否重新加载, 新的 mtime, 变更摘要)
        """
        current_mtime = cls.get_config_mtime()
        if current_mtime == last_mtime:
            return False, current_mtime, {}
        
        # 保存旧值（用于生成变更摘要）
        old_mode = cls.CLIENT_MODE
        old_temp = cls.TEMPERATURE
        old_auto_apply = cls.AUTO_APPLY_MODIFICATIONS
        old_stage_switch = getattr(cls, "STAGE_SWITCH", {})
        
        # 重新加载
        cls.load_from_file()
        
        # 生成变更摘要
        changes = {}
        if old_mode != cls.CLIENT_MODE:
            changes["CLIENT_MODE"] = f"{old_mode} → {cls.CLIENT_MODE}"
        if old_temp != cls.TEMPERATURE:
            changes["TEMPERATURE"] = f"{old_temp} → {cls.TEMPERATURE}"
        if old_auto_apply != cls.AUTO_APPLY_MODIFICATIONS:
            changes["AUTO_APPLY_MODIFICATIONS"] = f"{old_auto_apply} → {cls.AUTO_APPLY_MODIFICATIONS}"
        
        new_stage_switch = getattr(cls, "STAGE_SWITCH", {})
        changed_stages = []
        for key in set(old_stage_switch.keys()) | set(new_stage_switch.keys()):
            if old_stage_switch.get(key) != new_stage_switch.get(key):
                changed_stages.append(f"{key}: {old_stage_switch.get(key)} → {new_stage_switch.get(key)}")
        if changed_stages:
            changes["stage_switch"] = changed_stages
        
        return True, current_mtime, changes

    @classmethod
    def apply_config_to_client(cls, client) -> Dict[str, Any]:
        """
        将配置应用到客户端实例
        返回: 应用变更的摘要
        """
        applied = {}
        if client is None:
            return applied
        
        # 更新 temperature
        if hasattr(client, 'temperature'):
            old = client.temperature
            client.temperature = cls.TEMPERATURE
            if old != cls.TEMPERATURE:
                applied["temperature"] = f"{old} → {cls.TEMPERATURE}"
        
        # 更新 max_tokens / num_predict
        if hasattr(client, 'max_tokens'):
            old = client.max_tokens
            client.max_tokens = cls.MAX_TOKENS
            if old != cls.MAX_TOKENS:
                applied["max_tokens"] = f"{old} → {cls.MAX_TOKENS}"
        
        # OllamaClient 特有：num_ctx
        if hasattr(client, 'num_ctx'):
            old = client.num_ctx
            client.num_ctx = cls.OLLAMA_NUM_CTX
            if old != cls.OLLAMA_NUM_CTX:
                applied["num_ctx"] = f"{old} → {cls.OLLAMA_NUM_CTX}"
        
        # OllamaClient 特有：num_predict
        if hasattr(client, 'num_predict') and hasattr(client, 'profile'):
            old = client.num_predict
            client.num_predict = client.profile.calculate_num_predict()
            if old != client.num_predict:
                applied["num_predict"] = f"{old} → {client.num_predict}"
        
        return applied

    

    @classmethod
    def ple_protected_files_resolved(cls) -> list:
        """返回真正在磁盘上的 PROTECTED_FILES 绝对路径列表"""
        out = []
        for rel in cls.PROTECTED_FILES:
            p = cls.PROJECT_ROOT / rel
            if p.exists():
                out.append((rel, p))
        return out

    @classmethod
    def _ple_hash(cls, text_bytes: bytes) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update((cls.PLE_SALT_PREFIX).encode("utf-8"))
        h.update(text_bytes)
        return h.hexdigest()

    @classmethod
    def ple_compute_manifest(cls) -> dict:
        """计算当前磁盘上 7 个核心文件的 sha256 签名清单"""
        manifest = {
            "version": 1,
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "files": {},
        }
        for rel, path in cls.ple_protected_files_resolved():
            manifest["files"][rel] = cls._ple_hash(path.read_bytes())
        # 顶层签名：对所有子文件 hash 排序后再签名一次，防止攻击者删掉某条 record
        concat = "|".join(f"{k}={manifest['files'][k]}" for k in sorted(manifest["files"].keys()))
        manifest["manifest_signature"] = cls._ple_hash(concat.encode("utf-8"))
        return manifest

    @classmethod
    def ple_save_manifest(cls, force: bool = False) -> Optional[dict]:
        """把 manifest 写入 config.json（只在不存在 或 force=True 时写）"""
        path = cls._config_json_path()
        current = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    current = json.load(f)
            except Exception:
                current = {}
        if "ple_manifest" in current and not force:
            return current["ple_manifest"]
        manifest = cls.ple_compute_manifest()
        current["ple_manifest"] = manifest
        cls.ensure_dirs()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
        return manifest

    @classmethod
    def ple_verify(cls) -> Tuple[bool, str]:
        """
        校验 PLE 核心文件完整性。
        返回 (ok, message)。
        ok=False 时意味着核心引擎可能被篡改，应立即触发 kill-switch。
        """
        path = cls._config_json_path()
        if not path.exists():
            # 首次运行没有 manifest → 自动建立基线，视为校验通过（因为现在就是"此刻快照"）
            cls.ple_save_manifest(force=False)
            return True, "首次运行，已建立 PLE 签名基线"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        baseline = data.get("ple_manifest")
        if not baseline or not isinstance(baseline, dict) or "files" not in baseline:
            cls.ple_save_manifest(force=True)
            return True, "ple_manifest 缺失，已重新建立基线（建议重启确认一致）"

        # 1) 顶层签名防篡改：对 baseline["files"] 再重新算一次 signature
        baseline_files = baseline["files"]
        concat = "|".join(f"{k}={baseline_files[k]}" for k in sorted(baseline_files.keys()))
        expected_top_sig = cls._ple_hash(concat.encode("utf-8"))
        if baseline.get("manifest_signature") != expected_top_sig:
            return False, f"[PLE] manifest 顶层签名被篡改：期望 {expected_top_sig[:12]}..."

        # 2) 逐个文件核对哈希
        mismatches = []
        for rel, path_obj in cls.ple_protected_files_resolved():
            actual = cls._ple_hash(path_obj.read_bytes())
            expected = baseline_files.get(rel)
            if expected is None:
                # 基线里没有这个文件 → 当作新增，记录但不拦截
                continue
            if actual != expected:
                mismatches.append(rel)
        # 3) 基线里有、但磁盘上不存在的也算异常
        missing = [r for r in baseline_files.keys()
                   if not (cls.PROJECT_ROOT / r).exists()]
        if missing:
            mismatches.append(f"[缺失] {', '.join(missing)}")

        if mismatches:
            return False, ("[PLE] 核心引擎文件哈希不匹配/缺失: "
                           + "; ".join(mismatches))
        return True, "PLE 完整性校验通过"

    @classmethod
    def ensure_dirs(cls):
        """确保所需目录存在"""
        cls.BACKUP_DIR.mkdir(exist_ok=True)
        cls.LOG_DIR.mkdir(exist_ok=True)

    @classmethod
    def _config_json_path(cls):
        return cls.PROJECT_ROOT / "config.json"

    @classmethod
    def load_from_file(cls):
        """从config.json加载用户上次的配置（如果存在）"""
        path = cls._config_json_path()
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 只加载用户会修改的运行时配置，不加载密钥（从环境变量取更安全）
            if "CLIENT_MODE" in data and data["CLIENT_MODE"] in ("api", "local", "lmstudio", "jiyuan"):
                cls.CLIENT_MODE = data["CLIENT_MODE"]
            if "AUTO_APPLY_MODIFICATIONS" in data:
                cls.AUTO_APPLY_MODIFICATIONS = bool(data["AUTO_APPLY_MODIFICATIONS"])
            if "OLLAMA_MODEL" in data:
                cls.OLLAMA_MODEL = data["OLLAMA_MODEL"]
            if "LMSTUDIO_MODEL" in data:
                cls.LMSTUDIO_MODEL = data["LMSTUDIO_MODEL"]
            # 加载 Stage 开关 (默认全部开启)
            cls.STAGE_SWITCH = data.get("stage_switch", {})
            print(f"[Config] 已加载上次配置 (mode={cls.CLIENT_MODE})")
        except Exception as e:
            print(f"[Config] 加载config.json失败 (忽略): {e}")

    @classmethod
    def is_stage_enabled(cls, stage_name: str) -> bool:
        """检查某个 Stage 是否启用。默认未配置 = 启用。"""
        if not hasattr(cls, "STAGE_SWITCH") or not cls.STAGE_SWITCH:
            return True
        return cls.STAGE_SWITCH.get(stage_name, True)

    @classmethod
    def save_to_file(cls, filepath=None):
        """将当前配置保存到JSON文件"""
        if filepath is None:
            filepath = cls._config_json_path()
        data = {
            "CLIENT_MODE": cls.CLIENT_MODE,
            "ARK_BASE_URL": cls.ARK_BASE_URL,
            "MODEL_ID": cls.MODEL_ID,
            "OLLAMA_BASE_URL": cls.OLLAMA_BASE_URL,
            "OLLAMA_MODEL": cls.OLLAMA_MODEL,
            "LMSTUDIO_MODEL": cls.LMSTUDIO_MODEL,
            "OLLAMA_NUM_CTX": cls.OLLAMA_NUM_CTX,
            "PROJECT_ROOT": str(cls.PROJECT_ROOT),
            "BACKUP_DIR": str(cls.BACKUP_DIR),
            "SELF_EVOLVE_FILES": cls.SELF_EVOLVE_FILES,
            "MAX_MODIFICATIONS_PER_ITERATION": cls.MAX_MODIFICATIONS_PER_ITERATION,
            "AUTO_APPLY_MODIFICATIONS": cls.AUTO_APPLY_MODIFICATIONS,
            "AUTO_BACKUP": cls.AUTO_BACKUP,
            "TEMPERATURE": cls.TEMPERATURE,
            "MAX_TOKENS": cls.MAX_TOKENS,
            "AUTO_SYNTAX_CHECK": cls.AUTO_SYNTAX_CHECK,
            "AUTO_IMPORT_TEST": cls.AUTO_IMPORT_TEST,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def auto_save(cls):
        """保存当前配置（不打印日志，避免刷屏）"""
        path = cls._config_json_path()
        data = {
            "CLIENT_MODE": cls.CLIENT_MODE,
            "OLLAMA_MODEL": cls.OLLAMA_MODEL,
            "LMSTUDIO_MODEL": cls.LMSTUDIO_MODEL,
            "AUTO_APPLY_MODIFICATIONS": cls.AUTO_APPLY_MODIFICATIONS,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @classmethod
    def is_api_configured(cls):
        """检查API Key是否已配置（火山方舟 ARK 或基元律动 JIYUAN）"""
        if cls.CLIENT_MODE == "jiyuan":
            return bool(cls.JIYUAN_API_KEY and len(cls.JIYUAN_API_KEY) > 10)
        return bool(cls.ARK_API_KEY and len(cls.ARK_API_KEY) > 10)

    @classmethod
    def is_local_mode(cls):
        """是否使用本地Ollama模式"""
        return cls.CLIENT_MODE == "local"

    @classmethod
    def is_lmstudio_mode(cls):
        """是否使用 LM Studio 模式"""
        return cls.CLIENT_MODE == "lmstudio"

    @classmethod
    def is_jiyuan_mode(cls):
        """是否使用基元律动 TokenRhythm API 模式"""
        return cls.CLIENT_MODE == "jiyuan"

    @classmethod
    def is_local_any(cls):
        """是否使用任何本地推理引擎（Ollama 或 LM Studio）"""
        return cls.CLIENT_MODE in ("local", "lmstudio")

    @classmethod
    def is_client_ready(cls):
        """检查当前模式的客户端是否已配置"""
        if cls.is_local_mode():
            try:
                import urllib.request
                req = urllib.request.Request(f"{cls.OLLAMA_BASE_URL}/api/tags")
                urllib.request.urlopen(req, timeout=2)
                return True
            except Exception:
                return False
        elif cls.is_lmstudio_mode():
            try:
                import urllib.request
                req = urllib.request.Request(f"{cls.LMSTUDIO_BASE_URL}/models")
                urllib.request.urlopen(req, timeout=2)
                return True
            except Exception:
                return False
        elif cls.is_jiyuan_mode():
            return bool(cls.JIYUAN_API_KEY and len(cls.JIYUAN_API_KEY) > 10)
        else:
            return cls.is_api_configured()
