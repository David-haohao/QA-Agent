# ============================================================
# BM25 稀疏索引构建器 — 基于jieba中文分词的全文检索
# ============================================================

import os
import json
import pickle
import jieba
from typing import List, Dict
from collections import Counter


class BM25Index:
    """
    基于jieba分词的BM25全文检索索引
    用于关键词精确匹配，与向量检索互补
    """

    def __init__(self, kb_data_dir: str):
        """
        kb_data_dir: 知识库数据存储目录
        """
        self.kb_data_dir = kb_data_dir
        self.index_file = os.path.join(kb_data_dir, "bm25_index.pkl")
        self.k1 = 1.5  # BM25参数：词频饱和度
        self.b = 0.75  # BM25参数：文档长度归一化

        # 索引数据结构
        self.documents: List[str] = []  # 文档文本列表
        self.chunk_ids: List[str] = []  # chunk_id列表
        self.metadatas: List[Dict] = []  # 元数据列表
        self.doc_lengths: List[int] = []  # 文档长度列表
        self.avg_doc_length: float = 0.0  # 平均文档长度
        self.term_doc_freq: Dict[str, int] = {}  # 词→出现文档数
        self.term_freqs: List[Dict[str, int]] = []  # 每个文档中词的频率

        self._loaded = False

    def build_index(self, chunks: List[Dict]) -> int:
        """
        构建BM25索引
        chunks: 文本块列表
        返回: 已索引数量
        """
        if not chunks:
            return 0

        self.documents = [c["content"] for c in chunks]
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.metadatas = [c.get("metadata", {}) for c in chunks]
        self.doc_lengths = []

        # 对每个文档分词并统计词频
        self.term_freqs = []
        all_terms = set()
        doc_term_sets = []

        for doc_text in self.documents:
            # jieba分词
            words = list(jieba.cut(doc_text))
            # 过滤单个字和停用词级别
            words = [w.strip() for w in words if len(w.strip()) > 1]
            self.doc_lengths.append(len(words))

            # 词频统计
            tf = Counter(words)
            self.term_freqs.append(dict(tf))
            doc_term_sets.append(set(tf.keys()))
            all_terms.update(tf.keys())

        # 计算文档频率(Document Frequency)
        for term in all_terms:
            df = sum(1 for term_set in doc_term_sets if term in term_set)
            self.term_doc_freq[term] = df

        # 平均文档长度
        n_docs = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths) / n_docs if n_docs > 0 else 0

        # 持久化到文件
        self._save()
        self._loaded = True
        return n_docs

    def search(self, query: str, top_n: int = 20) -> List[Dict]:
        """
        BM25关键词搜索
        query: 查询文本
        top_n: 返回数量
        返回: 与VectorIndex相同格式的结果列表
        """
        if not self._loaded and os.path.exists(self.index_file):
            self._load()

        if not self.documents:
            return []

        # 对查询分词
        query_terms = list(jieba.cut(query))
        query_terms = [w.strip() for w in query_terms if len(w.strip()) > 1]

        n_docs = len(self.documents)
        scores = []

        for doc_idx in range(n_docs):
            doc_terms = self.term_freqs[doc_idx]
            score = self._score_doc(query_terms, doc_terms, doc_idx)
            if score > 0:
                scores.append((doc_idx, score))

        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = scores[:top_n]

        results = []
        for doc_idx, score in top_indices:
            results.append({
                "chunk_id": self.chunk_ids[doc_idx],
                "content": self.documents[doc_idx],
                "score": round(score, 4),
                "metadata": self.metadatas[doc_idx],
            })

        return results

    def _score_doc(self, query_terms: List[str], doc_terms: Dict[str, int], doc_idx: int) -> float:
        """计算单个文档的BM25分数"""
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        n_docs = len(self.documents)

        for term in query_terms:
            if term not in self.term_doc_freq:
                continue
            df = self.term_doc_freq[term]
            idf = max(0, ((n_docs - df + 0.5) / (df + 0.5)))
            idf = idf + 1.0

            tf = doc_terms.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator

        return score

    def _save(self):
        """持久化索引到文件"""
        data = {
            "documents": self.documents,
            "chunk_ids": self.chunk_ids,
            "metadatas": self.metadatas,
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "term_doc_freq": self.term_doc_freq,
            "term_freqs": self.term_freqs,
        }
        with open(self.index_file, "wb") as f:
            pickle.dump(data, f)

    def _load(self):
        """从文件加载索引"""
        with open(self.index_file, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.chunk_ids = data["chunk_ids"]
        self.metadatas = data["metadatas"]
        self.doc_lengths = data["doc_lengths"]
        self.avg_doc_length = data["avg_doc_length"]
        self.term_doc_freq = data["term_doc_freq"]
        self.term_freqs = data["term_freqs"]
        self._loaded = True

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        增量追加文本块到已有 BM25 索引
        chunks: 新文本块列表
        返回: 追加的 chunk 数量
        """
        if not chunks:
            return 0

        # 加载已有索引
        if not self._loaded and os.path.exists(self.index_file):
            self._load()
        if not self._loaded:
            self._loaded = True  # 首次构建

        # 跳过已存在的 chunk_id
        existing_ids = set(self.chunk_ids)
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        if not new_chunks:
            return 0

        # 处理新块
        new_docs = [c["content"] for c in new_chunks]
        new_ids = [c["chunk_id"] for c in new_chunks]
        new_metas = [c.get("metadata", {}) for c in new_chunks]

        for doc_text in new_docs:
            words = list(jieba.cut(doc_text))
            words = [w.strip() for w in words if len(w.strip()) > 1]
            self.doc_lengths.append(len(words))
            tf = Counter(words)
            self.term_freqs.append(dict(tf))
            for term in tf:
                self.term_doc_freq[term] = self.term_doc_freq.get(term, 0) + 1

        self.documents.extend(new_docs)
        self.chunk_ids.extend(new_ids)
        self.metadatas.extend(new_metas)

        # 更新平均文档长度
        n_docs = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths) / n_docs if n_docs > 0 else 0

        self._save()
        return len(new_chunks)

    def replace_chunks_for_files(self, file_names: List[str], new_chunks: List[Dict]) -> int:
        """Replace all sparse-index chunks belonging to changed source files."""
        if not self._loaded and os.path.exists(self.index_file):
            self._load()

        file_name_set = set(file_names)
        retained_chunks = []
        for chunk_id, content, metadata in zip(self.chunk_ids, self.documents, self.metadatas):
            source_file = metadata.get("source_file", metadata.get("file_name", ""))
            if source_file not in file_name_set:
                retained_chunks.append(
                    {"chunk_id": chunk_id, "content": content, "metadata": metadata}
                )

        self.term_doc_freq = {}
        self.term_freqs = []
        self.doc_lengths = []
        self.documents = []
        self.chunk_ids = []
        self.metadatas = []
        self.avg_doc_length = 0.0
        self._loaded = False
        return self.build_index(retained_chunks + new_chunks)

    def load_if_exists(self) -> bool:
        """加载已有索引，返回是否成功"""
        if os.path.exists(self.index_file):
            self._load()
            return True
        return False
