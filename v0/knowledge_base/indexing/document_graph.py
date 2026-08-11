# ============================================================
# 文档关系图谱 — 基于NetworkX的文档关联图
# 用于关联问题生成时的知识图谱策略
# ============================================================

import os
import json
import pickle
import networkx as nx
from typing import List, Dict, Set


class DocumentGraph:
    """
    构建文档之间的关联关系图
    基于文档共同关键词、引用关系构建边
    """

    def __init__(self, kb_data_dir: str):
        """
        kb_data_dir: 知识库数据存储目录
        """
        self.kb_data_dir = kb_data_dir
        self.graph_file = os.path.join(kb_data_dir, "document_graph.pkl")
        self.graph = nx.Graph()

    def build_graph(self, chunks: List[Dict]):
        """
        从chunks构建文档关系图
        节点: doc_id
        边: 两个文档共享关键词越多则关联越强
        """
        # 按doc_id分组
        doc_chunks: Dict[str, List[Dict]] = {}
        for c in chunks:
            doc_id = c.get("doc_id", "")
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(c)

        # 为每个文档提取关键词集合
        doc_keywords: Dict[str, Set[str]] = {}
        doc_names: Dict[str, str] = {}

        import jieba
        for doc_id, doc_chunk_list in doc_chunks.items():
            doc_names[doc_id] = doc_chunk_list[0].get("file_name", doc_id)
            full_text = " ".join([c["content"] for c in doc_chunk_list])
            # 用jieba提取关键词(TF-IDF简单实现)
            words = list(jieba.cut(full_text))
            words = [w.strip() for w in words if len(w.strip()) >= 2]
            from collections import Counter
            word_freq = Counter(words)
            # 取高频词作为关键词(过滤停用词级别)
            top_words = {w for w, _ in word_freq.most_common(50) if len(w) >= 2}
            doc_keywords[doc_id] = top_words

        # 添加节点
        for doc_id, name in doc_names.items():
            self.graph.add_node(doc_id, name=name)

        # 添加边：两个文档共享关键词越多，权重越高
        doc_ids = list(doc_keywords.keys())
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                id_i, id_j = doc_ids[i], doc_ids[j]
                common = doc_keywords[id_i] & doc_keywords[id_j]
                if common:
                    # Jaccard相似度作为权重
                    union = doc_keywords[id_i] | doc_keywords[id_j]
                    weight = len(common) / len(union) if union else 0
                    if weight > 0.05:  # 只保留有意义的关联
                        self.graph.add_edge(id_i, id_j, weight=round(weight, 4), common_keywords=list(common)[:10])

        # 持久化
        self._save()

    def get_related_documents(self, doc_id: str, top_n: int = 5) -> List[Dict]:
        """
        获取与指定文档最相关的其他文档
        返回: [{"doc_id": ..., "name": ..., "weight": ..., "common_keywords": [...]}, ...]
        """
        if not self.graph.has_node(doc_id):
            self._load_if_exists()
        if not self.graph.has_node(doc_id):
            return []

        neighbors = []
        for neighbor in self.graph.neighbors(doc_id):
            edge_data = self.graph[doc_id][neighbor]
            neighbors.append({
                "doc_id": neighbor,
                "name": self.graph.nodes[neighbor].get("name", neighbor),
                "weight": edge_data.get("weight", 0),
                "common_keywords": edge_data.get("common_keywords", []),
            })

        neighbors.sort(key=lambda x: x["weight"], reverse=True)
        return neighbors[:top_n]

    def _save(self):
        """持久化图到文件"""
        with open(self.graph_file, "wb") as f:
            pickle.dump(self.graph, f)

    def _load_if_exists(self):
        """加载已有图谱"""
        if os.path.exists(self.graph_file):
            with open(self.graph_file, "rb") as f:
                self.graph = pickle.load(f)
