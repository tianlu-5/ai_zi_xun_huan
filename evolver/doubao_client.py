"""
豆包API通信模块 - 封装火山方舟平台的API调用
支持多轮对话、流式输出、结构化响应
"""
import os
import sys
import json
from typing import List, Dict, Optional, Any
from pathlib import Path
import time

def create_client(mode: str = 'distributed'):
    """
    客户端工厂函数 - 根据Config.CLIENT_MODE创建对应的客户端
    "local"    → OllamaClient（本地开源模型，完全离线，需GPU）
    "lmstudio" → OllamaClient（LM Studio OpenAI 兼容 API，本地，端口 1234）
    "api"      → DoubaoClient（火山方舟API，付费但稳定）
    "jiyuan"   → DoubaoClient（基元律动 TokenRhythm API，OpenAI 兼容）
    """
    from config import Config
    if Config.CLIENT_MODE == "local":
        from ollama_client import OllamaClient
        return OllamaClient(
            base_url=Config.OLLAMA_BASE_URL,
            model_id=Config.OLLAMA_MODEL,
            num_ctx=Config.OLLAMA_NUM_CTX,
            num_gpu=Config.OLLAMA_NUM_GPU,
            keep_alive=Config.OLLAMA_KEEP_ALIVE,
        )
    elif Config.CLIENT_MODE == "lmstudio":
        from ollama_client import OllamaClient
        return OllamaClient(
            base_url=Config.LMSTUDIO_BASE_URL,
            model_id=Config.LMSTUDIO_MODEL,
            num_ctx=Config.LMSTUDIO_NUM_CTX,
        )
    elif Config.CLIENT_MODE == "jiyuan":
        # 基元律动 TokenRhythm API（OpenAI 兼容，推理模型）
        return DoubaoClient(
            api_key=Config.JIYUAN_API_KEY,
            base_url=Config.JIYUAN_BASE_URL,
            model_id=Config.JIYUAN_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
        )
    else:
        return DoubaoClient(
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
        )


class DoubaoClient:
    """豆包大模型API客户端"""

    class _ProfileProxy:
        """轻量 profile 适配器，为 agent_pipeline 提供 calculate_temperature() 接口"""
        __slots__ = ("_temperature",)

        def __init__(self, temperature: float):
            self._temperature = temperature

        def calculate_temperature(self) -> float:
            return self._temperature

    def __init__(self, api_key: str = "", base_url: str = "", model_id: str = "",
                 temperature: float = 0.2, max_tokens: int = 8192,
                 backend_id: Optional[str] = None):
        """
        初始化豆包客户端
        :param api_key: 火山方舟API Key
        :param base_url: API Base URL
        :param model_id: 使用的模型ID
        :param temperature: 生成温度
        :param max_tokens: 最大生成token数
        :param backend_id: 后端标识（用于路由器追踪）
        """
        
        # 延迟导入，避免未安装SDK时报错
        try:
            from volcenginesdkarkruntime import Ark
            self._Ark = Ark
            self._sdk_available = True
        except ImportError:
            self._sdk_available = False
            self._Ark = None

        # 兼容OpenAI SDK方式
        try:
            from openai import OpenAI
            self._OpenAI = OpenAI
            self._openai_sdk_available = True
        except ImportError:
            self._openai_sdk_available = False
            self._OpenAI = None

        # 兼容requests原生方式（作为最后fallback）
        try:
            import requests
            self._requests = requests
            self._requests_available = True
        except ImportError:
            self._requests_available = False
            self._requests = None

        from config import Config
        self.api_key = api_key or Config.ARK_API_KEY
        self.base_url = base_url or Config.ARK_BASE_URL
        self.model_id = model_id or Config.MODEL_ID
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 轻量 profile 适配（兼容 agent_pipeline 中 llm.profile.calculate_temperature() 调用）
        self.profile = self._ProfileProxy(self.temperature)

        # 多轮对话历史
        self.conversation_history: List[Dict[str, str]] = []

        # 初始化客户端
        self._client = None
        self._init_client()
        # 保存 backend_id（用于路由器追踪）
        self._backend_id = backend_id or "unknown"

    def _init_client(self):
        """初始化API客户端，按优先级尝试多种方式"""
        if not self.api_key:
            print("[DoubaoClient] 警告: 未配置API Key，请设置环境变量 ARK_API_KEY")
            return

        # 方式1: 火山方舟官方SDK
        if self._sdk_available:
            try:
                self._client = self._Ark(
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
                print("[DoubaoClient] 使用火山方舟官方SDK初始化成功")
                return
            except Exception as e:
                print(f"[DoubaoClient] 官方SDK初始化失败: {e}，尝试OpenAI兼容模式")

        # 方式2: OpenAI兼容SDK
        if self._openai_sdk_available:
            try:
                self._client = self._OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=120,  # 120s 总超时，防止 API 挂死
                )
                print("[DoubaoClient] 使用OpenAI兼容SDK初始化成功")
                return
            except Exception as e:
                print(f"[DoubaoClient] OpenAI SDK初始化失败: {e}，尝试原生requests方式")

        # 方式3: 原生requests（不需要额外SDK）
        if self._requests_available:
            print("[DoubaoClient] 将使用原生requests方式调用API")
            return

        print("[DoubaoClient] 错误: 没有可用的API调用方式，请安装依赖:")
        print("  pip install \"volcengine-python-sdk[ark]\"")
        print("  或 pip install openai")
        print("  或 pip install requests")

    def reset_conversation(self):
        """清空对话历史，开始新会话"""
        self.conversation_history.clear()

    def add_system_prompt(self, prompt: str):
        """添加系统提示词（放在对话最前面）"""
        # 移除已有的system prompt
        self.conversation_history = [
            msg for msg in self.conversation_history if msg["role"] != "system"
        ]
        self.conversation_history.insert(0, {"role": "system", "content": prompt})

    def chat(self, user_message: str, stream: bool = False,
         system_prompt: Optional[str] = None,
         extra_params: Optional[Dict[str, Any]] = None,
         force_json: bool = False) -> str:
        """
        与豆包进行对话
        :param user_message: 用户消息
        :param stream: 是否流式输出
        :param system_prompt: 临时系统提示词（本次对话生效）
        :param extra_params: 额外API参数
        :param force_json: 是否强制JSON输出
        :return: 模型返回的内容
        """
        import time
        
        if not self.api_key:
            return "[错误] 未配置ARK_API_KEY，请先在环境变量或config.py中设置"

        # 构建本次消息
        if system_prompt:
            temp_history = [msg for msg in self.conversation_history if msg["role"] != "system"]
            temp_history.insert(0, {"role": "system", "content": system_prompt})
            messages = temp_history + [{"role": "user", "content": user_message}]
        else:
            messages = self.conversation_history + [{"role": "user", "content": user_message}]

        params = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if extra_params:
            params.update(extra_params)
        if force_json:
            params["response_format"] = {"type": "json_object"}

        result = ""
        tokens_used = 0
        start_time = time.time()
        
        try:
            # 方式1 & 2: 使用SDK客户端
            if self._client is not None:
                if stream:
                    result = self._chat_stream(params)
                else:
                    result = self._chat_non_stream(params)
            # 方式3: 原生requests
            elif self._requests_available:
                result = self._chat_requests(params, stream)
            else:
                return "[错误] 没有可用的API调用方式，请安装SDK依赖"
            
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
            
        except Exception as e:
            # ===== 上报失败到路由器 =====
            try:
                from model_router import get_router
                router = get_router()
                router.mark_failure(self._backend_id, str(e))
            except Exception:
                pass
            # ============================
            result = f"[API调用异常] {type(e).__name__}: {e}"
            print(result)

        # 保存对话历史
        if not result.startswith("[错误]") and not result.startswith("[API调用异常]"):
            if system_prompt:
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": result})
            else:
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": result})

        return result

        

    def _chat_non_stream(self, params: Dict) -> str:
        """SDK方式的非流式调用"""
        params.pop("stream", None)
        completion = self._client.chat.completions.create(**params)
        msg = completion.choices[0].message
        content = msg.content or ""
        # 兼容推理模型：content 为空时尝试返回 reasoning_content（便于诊断）
        if not content:
            content = getattr(msg, "reasoning_content", "") or ""
        return content

    def _chat_stream(self, params: Dict) -> str:
        """SDK方式的流式调用"""
        params["stream"] = True
        stream = self._client.chat.completions.create(**params)
        collected = []
        print("[豆包输出] ", end="", flush=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                collected.append(delta)
                print(delta, end="", flush=True)
        print()
        return "".join(collected)

    def _chat_requests(self, params: Dict, stream: bool) -> str:
        """使用原生requests调用API（不依赖SDK）"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if stream:
            params["stream"] = True

        url = f"{self.base_url}/chat/completions"
        resp = self._requests.post(url, headers=headers, json=params, stream=stream, timeout=120)

        if resp.status_code != 200:
            return f"[API错误] HTTP {resp.status_code}: {resp.text[:500]}"

        if stream:
            collected = []
            print("[豆包输出] ", end="", flush=True)
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if data.get("choices"):
                            delta = data["choices"][0].get("delta", {}).get("content", "")
                            if delta:
                                collected.append(delta)
                                print(delta, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            print()
            return "".join(collected)
        else:
            data = resp.json()
            if data.get("choices"):
                return data["choices"][0]["message"].get("content", "")
            return f"[API错误] 响应格式异常: {json.dumps(data, ensure_ascii=False)[:500]}"

    def chat_with_structured_output(self, user_message: str, system_prompt: str,
                                    stream: bool = False) -> Optional[Dict]:
        """
        要求豆包返回JSON格式的结构化输出
        :param user_message: 用户消息（通常包含当前代码或任务）
        :param system_prompt: 系统提示词，描述JSON格式要求
        :param stream: 是否流式
        :return: 解析后的JSON字典，失败返回None
        """
        full_response = self.chat(
            user_message=user_message,
            system_prompt=system_prompt,
            stream=stream,
        )

        if not full_response or full_response.startswith("["):
            print(f"[DoubaoClient] 结构化调用失败: {full_response}")
            return None

        # 提取JSON部分（处理模型可能输出的额外文字）
        return self._extract_json(full_response)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """从模型输出中提取JSON对象"""
        text = text.strip()

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 查找 ```json ... ``` 块
        start = text.find("```json")
        if start != -1:
            end = text.find("```", start + 7)
            if end != -1:
                json_str = text[start + 7:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"[DoubaoClient] JSON块解析失败: {e}")

        # 查找 ``` ... ``` 块
        start = text.find("```")
        if start != -1:
            end = text.find("```", start + 3)
            if end != -1:
                json_str = text[start + 3:end].strip()
                # 去掉可能的语言标识
                if "\n" in json_str:
                    first_line, rest = json_str.split("\n", 1)
                    if first_line.strip().lower() in ("json", "python", "js"):
                        json_str = rest.strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # 查找第一个 { 到最后一个 }
        first_lbrace = text.find("{")
        last_rbrace = text.rfind("}")
        if first_lbrace != -1 and last_rbrace != -1 and last_rbrace > first_lbrace:
            json_str = text[first_lbrace:last_rbrace + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"[DoubaoClient] 提取JSON解析失败: {e}")

        print("[DoubaoClient] 无法从模型输出中提取JSON")
        print(f"原始输出: {text[:300]}...")
        return None

    def get_conversation_token_count(self) -> int:
        """估算对话历史的token数（简单估算：中文字符=1token，英文单词=1token）"""
        total = 0
        for msg in self.conversation_history:
            content = msg["content"]
            # 中文字符 + 英文单词数
            cn_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
            en_words = sum(1 for w in content.split() if any(c.isalpha() for c in w))
            total += cn_chars + en_words
        return total

    def trim_conversation(self, max_tokens: int = 16000):
        """裁剪对话历史，避免超过上下文窗口"""
        while self.get_conversation_token_count() > max_tokens and len(self.conversation_history) > 4:
            # 保留最前面的system和最近的几轮
            # 移除最早的user+assistant对（跳过第一个system）
            if len(self.conversation_history) >= 3 and self.conversation_history[0]["role"] == "system":
                removed = self.conversation_history.pop(1)  # 移除最早的user
                if len(self.conversation_history) >= 2:
                    self.conversation_history.pop(1)  # 移除对应的assistant
                print(f"[DoubaoClient] 裁剪对话历史，移除消息: {str(removed)[:50]}...")
            else:
                self.conversation_history.pop(0)
                if self.conversation_history:
                    self.conversation_history.pop(0)
