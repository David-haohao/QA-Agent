# ============================================================
# 嵌入模型客户端 — BGE-M3
# 将文本转换为向量，用于语义搜索
# API接口：兼容OpenAI的 /v1/embeddings 端点
# ============================================================

import json
import time
import requests
from typing import List


class EmbeddingClient:
    """封装BGE-M3嵌入模型的HTTP API调用"""

    def __init__(self, config: dict):
        """
        初始化嵌入模型客户端
        config包含: url, model, api_key, batch_size, dimension
        """
        self.url = config["url"]
        self.model = config["model"]
        self.api_key = config.get("api_key", "")
        self.batch_size = config.get("batch_size", 256)
        self.dimension = config.get("dimension", 1024)
        self.request_retries = config.get("request_retries", 3)
        self.retry_delay_seconds = config.get("retry_delay_seconds", 1)
        self.request_timeout_seconds = config.get("request_timeout_seconds", 600)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        对文本列表进行批量向量化
        texts: 文本字符串列表
        返回: 对应长度的向量列表，每个向量为float列表
        """
        result = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            # API接口格式: {"model": "bge-m3", "input": [...]}, 文本截断至512字符
            headers = {
                "Content-Type": "application/json",
                "Authorization": self.api_key,
            }
            data = {
                "model": self.model,
                "input": [x[:512] for x in batch],
            }
            last_error = None
            for attempt in range(1, self.request_retries + 1):
                try:
                    res = requests.post(
                        json=data,
                        url=self.url,
                        headers=headers,
                        timeout=self.request_timeout_seconds,
                    )
                    res.raise_for_status()
                    res_json = res.json()
                    embeddings = [item["embedding"] for item in res_json["data"]]
                    if len(embeddings) != len(batch):
                        raise ValueError(
                            f"Embedding response count mismatch: expected {len(batch)}, got {len(embeddings)}"
                        )
                    break
                except (requests.RequestException, OSError, ValueError, KeyError) as exc:
                    last_error = exc
                    if attempt == self.request_retries:
                        raise RuntimeError(
                            f"Embedding request failed after {attempt} attempts: {exc}"
                        ) from exc
                    time.sleep(self.retry_delay_seconds * attempt)
            result.extend(embeddings)
        return result

    def embed_single(self, text: str) -> List[float]:
        """对单条文本进行向量化"""
        result = self.embed([text])
        return result[0] if result else []
