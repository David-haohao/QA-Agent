# ============================================================
# LLM 大模型客户端 — Qwen3.6-27b (兼容OpenAI接口)
# 支持思考模式(enable_thinking)和流式输出两种调用方式
# ============================================================

from openai import OpenAI


class LLMClient:
    """封装Qwen3.6-27b LLM的OpenAI兼容接口（阿里云DashScope），提供统一的大模型调用方法"""

    def __init__(self, config: dict):
        """
        初始化LLM客户端
        config包含: api_key, base_url, model, max_tokens, temperature, timeout
        """
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        self.model = config["model"]
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.0)
        self.timeout = config.get("timeout", 14)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=float(self.timeout),
        )

    def chat(self, messages: list, stream: bool = False):
        """
        通用对话接口
        messages: [{"role": "system/user/assistant", "content": "..."}]
        stream: 是否使用流式输出
        返回: 非流式时返回完整回复文本，流式时返回生成器
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": {"enable_thinking": True},  # Qwen思考模式
        }

        if stream:
            response = self.client.chat.completions.create(**kwargs)
            return self._stream_response(response)
        else:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

    def chat_with_reasoning(self, messages: list, reasoning_effort: str = "high"):
        """
        带深度推理的对话接口(非流式，开启思考模式)
        Qwen3.6-27b 通过 enable_thinking 控制推理深度
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            extra_body={"enable_thinking": True},
        )
        return response.choices[0].message.content

    def chat_simple(self, messages: list):
        """
        简单对话接口(不开启思考模式，用于轻量判断)
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content

    def _stream_response(self, response):
        """处理流式响应，逐块返回文本内容（跳过思考块，只返回实际回答）"""
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

    def get_chat_model(self):
        """
        返回一个可用于LangChain集成的ChatOpenAI实例
        支持Qwen3.6-27b enable_thinking 思考模式和 tool calling
        """
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=float(self.timeout),
            extra_body={"enable_thinking": True},  # Qwen思考模式
        )
