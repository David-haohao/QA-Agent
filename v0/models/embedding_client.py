# ============================================================
# 嵌入模型客户端 — BGE-M3
# 将文本转换为向量，用于语义搜索
# API接口：兼容OpenAI的 /v1/embeddings 端点
# ============================================================

import json
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
            res = requests.post(
                json=data,
                url=self.url,
                headers=headers,
                timeout=120,  # CPU推理bge-m3单条约10s，批量+冷启动需留足余量
            )
            # 解析API返回: {"data": [{"embedding": [...], "index": 0}, ...]}
            res_json = res.json()
            embeddings = [item["embedding"] for item in res_json["data"]]
            result.extend(embeddings)
        return result

    def embed_single(self, text: str) -> List[float]:
        """对单条文本进行向量化"""
        result = self.embed([text])
        return result[0] if result else []
