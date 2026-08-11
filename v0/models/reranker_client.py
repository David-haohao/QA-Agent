# ============================================================
# 重排序模型客户端 — BGE-Reranker-v2-m3
# 对检索结果进行精排，提高相关文档的排序质量
# API接口：兼容OpenAI的 /v1/rerank 端点
# ============================================================

import json
import requests
from typing import List


class RerankerClient:
    """封装BGE Reranker模型的HTTP API调用"""

    def __init__(self, config: dict):
        """
        初始化重排序客户端
        config包含: url, model, api_key, batch_size
        """
        self.url = config["url"]
        self.model = config["model"]
        self.api_key = config.get("api_key", "")
        self.batch_size = config.get("batch_size", 16)

    def rerank(self, query: str, texts: List[str]) -> List[float]:
        """
        对文档列表进行相关性重排序
        query: 查询文本
        texts: 待排序文档文本列表
        返回: 对应长度的相关性分数列表（统一转为纯float列表）
        """
        if not texts:
            return []

        # API接口格式: {"model": "bge-reranker-v2-m3", "query": "...", "documents": [...]}
        # 文档截断: max(512 - len(query), 1)
        max_doc_len = max(512 - len(query), 1)
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key,
        }
        data = {
            "model": self.model,
            "query": query,
            "documents": [x[:max_doc_len] for x in texts],
        }
        r = requests.post(
            json=data,
            url=self.url,
            headers=headers,
            timeout=120,  # CPU推理cross-encoder重排需留足余量
        )
        # 解析API返回: {"results": [{"index": ..., "relevance_score": ...}, ...]}
        result_json = r.json()
        results = result_json.get("results", [])

        # 转换为按原始索引排序的分数列表
        scores = [0.0] * len(texts)
        for item in results:
            idx = item.get("index", 0)
            score = item.get("relevance_score", 0.0)
            if 0 <= idx < len(scores):
                scores[idx] = float(score)

        return scores

    def rerank_with_threshold(self, query: str, texts: List[str], threshold: float = 0.3) -> List[tuple]:
        """
        带分数阈值的重排序，返回带索引的 (原始索引, score) 元组列表
        只返回分数 >= threshold 的结果
        """
        scores = self.rerank(query, texts)
        result = []
        for idx, score in enumerate(scores):
            if score >= threshold:
                result.append((idx, score))
        return sorted(result, key=lambda x: x[1], reverse=True)
