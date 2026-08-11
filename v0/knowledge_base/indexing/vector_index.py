# ============================================================
# 向量索引构建器 — 基于ChromaDB的语义索引
# ============================================================

import os
import json
import chromadb
from typing import List, Dict
from chromadb.config import Settings


class VectorIndexBuilder:
    """
    基于ChromaDB的向量索引构建和查询
    使用线上Embedding API生成向量
    """

    def __init__(self, kb_data_dir: str, collection_name: str, embedding_client, dimension: int = 1024):
        """
        kb_data_dir: 知识库数据存储目录
        collection_name: ChromaDB集合名称
        embedding_client: EmbeddingClient实例
        dimension: 向量维度
        """
        self.kb_data_dir = kb_data_dir
        self.collection_name = collection_name
        self.embedding_client = embedding_client
        self.dimension = dimension

        # 初始化ChromaDB PersistentClient
        chroma_path = os.path.join(kb_data_dir, "chroma")
        os.makedirs(chroma_path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self, chunks: List[Dict]) -> int:
        """
        为所有文本块构建向量索引
        chunks: 文本块列表
        返回: 已索引的chunk数量
        """
        if not chunks:
            return 0

        # 提取文本和ID
        chunk_ids = [c["chunk_id"] for c in chunks]
        texts = [c["content"] for c in chunks]

        # 批量生成向量
        embeddings = self.embedding_client.embed(texts)

        # 构建元数据(用于过滤)
        metadatas = []
        for c in chunks:
            metadatas.append({
                "doc_id": c["doc_id"],
                "file_name": c["file_name"],
                "file_path": c.get("file_path", ""),
                "chunk_index": c["chunk_index"],
            })

        # 分批写入ChromaDB
        chroma_batch_size = 4000
        total_added = 0
        for i in range(0, len(chunks), chroma_batch_size):
            end = i + chroma_batch_size
            self.collection.add(
                ids=chunk_ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
            total_added += (end - i)
            print(f"  已写入: {total_added}/{len(chunks)}")

        return len(chunks)

    def search(self, query_text: str, top_n: int = 20) -> List[Dict]:
        """
        基于语义相似度搜索
        query_text: 查询文本
        top_n: 返回数量
        返回: [{"chunk_id": ..., "content": ..., "score": ..., "metadata": {...}}, ...]
        """
        if self.collection.count() == 0:
            return []

        # 生成查询向量
        query_embedding = self.embedding_client.embed_single(query_text)

        # 向量检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_n, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # 格式化结果
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                # 余弦距离转相似度分数(1 - distance用于cosine距离)
                score = 1.0 - distance

                formatted.append({
                    "chunk_id": chunk_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "score": round(score, 4),
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return formatted

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        增量追加文本块到已有向量索引（不重建）
        chunks: 新文本块列表
        返回: 追加的 chunk 数量
        """
        if not chunks:
            return 0

        # 去重：跳过已存在的 chunk_id
        existing_ids = set()
        if self.collection.count() > 0:
            try:
                existing = self.collection.get(include=[])
                existing_ids = set(existing.get("ids", []))
            except Exception:
                pass

        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        if not new_chunks:
            return 0

        chunk_ids = [c["chunk_id"] for c in new_chunks]
        texts = [c.get("content", "") for c in new_chunks]

        try:
            embeddings = self.embedding_client.embed(texts)
        except Exception as e:
            print(f"[vector_index] Embedding API 调用失败: {e}")
            print(f"[vector_index] 跳过向量索引追加，仅更新文本索引")
            return 0

        metadatas = []
        for c in new_chunks:
            metadatas.append({
                "doc_id": c.get("doc_id", ""),
                "file_name": c.get("file_name", ""),
                "file_path": c.get("file_path", ""),
                "chunk_index": c.get("chunk_index", 0),
            })

        batch_size = 4000
        total = 0
        for i in range(0, len(new_chunks), batch_size):
            end = i + batch_size
            self.collection.add(
                ids=chunk_ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
            total += (end - i)

        return total

    def delete_document(self, doc_id: str):
        """删除某个文档的所有向量索引"""
        existing = self.collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

    def get_document_count(self) -> int:
        """获取已索引文档数量"""
        return self.collection.count()

    def list_documents(self) -> List[str]:
        """列出所有已索引的文档文件名"""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        files = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                if meta and "file_name" in meta:
                    files.add(meta["file_name"])
        return sorted(list(files))
