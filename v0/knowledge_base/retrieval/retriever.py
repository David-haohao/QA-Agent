# ============================================================
# 混合检索器 — 融合Dense(ChromaDB)和Sparse(BM25)两种检索
# 使用RRF(Reciprocal Rank Fusion)融合，再经Reranker精排
# ============================================================

from typing import List, Dict
from knowledge_base.indexing.vector_index import VectorIndexBuilder
from knowledge_base.indexing.bm25_index import BM25Index


class HybridRetriever:
    """
    混合检索器：Dense向量 + Sparse BM25 → RRF融合 → Reranker精排
    在线检索的核心入口
    """

    def __init__(self, vector_index: VectorIndexBuilder, bm25_index: BM25Index, reranker_client):
        """
        vector_index: ChromaDB向量索引
        bm25_index: BM25关键词索引
        reranker_client: RerankerClient实例
        """
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.reranker_client = reranker_client

    def search(self, query: str, dense_top_n: int = 20, sparse_top_n: int = 20,
               final_top_k: int = 5, rrf_k: int = 60) -> List[Dict]:
        """
        执行混合检索
        query: 用户查询
        dense_top_n: 向量检索返回数量
        sparse_top_n: BM25检索返回数量
        final_top_k: 最终返回chunk数量
        rrf_k: RRF融合参数
        返回: 精排后的Top-K chunks
        """
        import asyncio

        # 并行执行Dense和Sparse检索
        dense_results = self.vector_index.search(query, top_n=dense_top_n)
        sparse_results = self.bm25_index.search(query, top_n=sparse_top_n)

        # RRF融合
        fused = self._rrf_fusion(dense_results, sparse_results, k=rrf_k)

        # 如果融合结果为空，直接返回
        if not fused:
            return []

        # Reranker精排
        texts = [item["content"] for item in fused]
        reranked = self.reranker_client.rerank_with_threshold(
            query=query,
            texts=texts,
            threshold=0.0,  # 先不过滤，由SourceBinding过滤
        )

        # 按rerank分数重新排序
        final_results = []
        for idx, score in reranked[:final_top_k]:
            item = fused[idx].copy()
            item["rerank_score"] = round(score, 4)
            final_results.append(item)

        return final_results

    def _rrf_fusion(self, dense: List[Dict], sparse: List[Dict], k: int = 60) -> List[Dict]:
        """
        RRF(Reciprocal Rank Fusion)融合算法
        将两路检索结果合并为一个排序列表
        """
        chunk_map = {}

        # Dense结果
        for rank, item in enumerate(dense):
            chunk_id = item["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = {**item, "rrf_score": rrf_score}
            else:
                chunk_map[chunk_id]["rrf_score"] += rrf_score

        # Sparse结果
        for rank, item in enumerate(sparse):
            chunk_id = item["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = {**item, "rrf_score": rrf_score}
            else:
                chunk_map[chunk_id]["rrf_score"] += rrf_score

        # 按RRF分数排序
        sorted_items = sorted(chunk_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_items
