# ============================================================
# 反馈记录模块 — 将用户反馈和问答数据保存到本地JSON文件
# 记录：点赞/点踩、纠错反馈、问答对、来源文档名等关键信息
# ============================================================

import os
import json
import time
from typing import Optional, Dict, List
from datetime import datetime


class EvolutionHook:
    """
    反馈数据记录器
    将用户点赞/点踩、纠错反馈、问答数据保存到本地JSON文件中
    """

    def __init__(self, evolution_data_dir: str, feedback_db_name: str = "feedback.db",
                 sampling_rate: float = 0.1):
        """
        evolution_data_dir: 数据存储目录（保留参数兼容性，实际使用JSON文件）
        feedback_db_name: 保留参数兼容性（不再使用SQLite）
        sampling_rate: 保留参数兼容性
        """
        self.data_dir = evolution_data_dir
        self.sampling_rate = sampling_rate
        os.makedirs(evolution_data_dir, exist_ok=True)

        # JSON文件路径
        self.feedback_file = os.path.join(evolution_data_dir, "feedback.json")
        self.qa_records_file = os.path.join(evolution_data_dir, "qa_records.json")

        # 初始化JSON文件
        self._init_json_files()

    def _init_json_files(self):
        """初始化JSON数据文件"""
        if not os.path.exists(self.feedback_file):
            self._save_json(self.feedback_file, {"feedback_list": [], "total": 0})

        if not os.path.exists(self.qa_records_file):
            self._save_json(self.qa_records_file, {"qa_list": [], "total": 0})

    def _load_json(self, file_path: str) -> dict:
        """加载JSON文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_json(self, file_path: str, data: dict):
        """保存JSON文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def record_qa(self, session_id: str, query_id: str, query: str, answer: str,
                  source_docs: list, elapsed_ms: int, cache_hit: bool = False,
                  retrieval_count: int = 0):
        """
        记录一次完整的问答（保存到qa_records.json）
        包含：问题、答案、来源文档名、响应时间、缓存命中状态
        """
        try:
            data = self._load_json(self.qa_records_file)

            record = {
                "query_id": query_id,
                "session_id": session_id,
                "query": query,
                "answer": answer,
                "source_docs": [s.get("path", s.get("name", "")) for s in source_docs],
                "source_docs_display": [s.get("name", "") for s in source_docs],
                "elapsed_ms": elapsed_ms,
                "cache_hit": cache_hit,
                "retrieval_count": retrieval_count,
                "created_at": datetime.now().isoformat(),
            }

            data["qa_list"].append(record)
            data["total"] = len(data["qa_list"])

            self._save_json(self.qa_records_file, data)
        except Exception as e:
            print(f"记录问答数据失败: {e}")

    def record_feedback(self, session_id: str = "", query_id: str = "",
                        rating: int = 0, comment: str = "",
                        correction_type: str = "", correction_text: str = "",
                        query: str = "", answer: str = "", source_docs: list = None):
        """
        记录用户反馈（保存到feedback.json）
        包含：点赞/点踩(rating: 1=赞, -1=踩)、纠错信息、关联的问答信息
        """
        try:
            data = self._load_json(self.feedback_file)

            record = {
                "feedback_id": f"fb_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}",
                "session_id": session_id,
                "query_id": query_id,
                "query": query,
                "answer": answer[:500] if answer else "",  # 截取前500字
                "source_docs": source_docs or [],
                "rating": rating,  # 1=点赞, -1=点踩, 0=仅评论
                "comment": comment,
                "correction_type": correction_type,  # 纠错类型
                "correction_text": correction_text,  # 纠错内容
                "created_at": datetime.now().isoformat(),
            }

            data["feedback_list"].append(record)
            data["total"] = len(data["feedback_list"])

            self._save_json(self.feedback_file, data)
            print(f"反馈已记录: rating={rating}, query_id={query_id}")
        except Exception as e:
            print(f"记录反馈失败: {e}")

    def record_knowledge_gap(self, query: str, domain: str = "", reason: str = ""):
        """记录知识缺口（检索无结果时），保存到feedback.json中"""
        try:
            data = self._load_json(self.feedback_file)

            record = {
                "feedback_id": f"gap_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}",
                "type": "knowledge_gap",
                "query": query,
                "domain": domain,
                "reason": reason,
                "created_at": datetime.now().isoformat(),
            }

            data["feedback_list"].append(record)
            data["total"] = len(data["feedback_list"])

            self._save_json(self.feedback_file, data)
        except Exception as e:
            print(f"记录知识缺口失败: {e}")

    def record_cache_stats(self, session_id: str, query_id: str, l1_hit: bool, l2_hit: bool):
        """记录缓存命中情况（简化实现）"""
        pass  # 缓存统计由ConsistencyCache自行维护

    def get_top_rated_questions(self, limit: int = 5) -> list:
        """获取点赞最多的问题列表，用于前端"试试"建议"""
        try:
            data = self._load_json(self.feedback_file)
            feedback_list = data.get("feedback_list", [])

            # 筛选rating=1（点赞）的反馈
            liked_queries = {}
            for fb in feedback_list:
                if fb.get("rating") == 1 and fb.get("query"):
                    q = fb["query"].strip()
                    if q and len(q) <= 80:
                        liked_queries[q] = liked_queries.get(q, 0) + 1

            # 按点赞数排序
            sorted_queries = sorted(liked_queries.items(), key=lambda x: x[1], reverse=True)
            return [q for q, _ in sorted_queries[:limit]]
        except Exception as e:
            print(f"查询高分问题失败: {e}")
            return []

    def get_popular_questions(self, limit: int = 5) -> list:
        """获取热门问题（按反馈总数排序）"""
        try:
            data = self._load_json(self.feedback_file)
            feedback_list = data.get("feedback_list", [])

            # 统计每个问题的反馈数
            query_counts = {}
            for fb in feedback_list:
                q = fb.get("query", "").strip()
                if q and len(q) <= 80:
                    query_counts[q] = query_counts.get(q, 0) + 1

            sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
            return [q for q, _ in sorted_queries[:limit]]
        except Exception as e:
            print(f"查询热门问题失败: {e}")
            return []

    def generate_query_id(self) -> str:
        """生成唯一查询ID"""
        return f"q_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"

    def get_feedback_stats(self) -> dict:
        """获取反馈统计信息"""
        try:
            data = self._load_json(self.feedback_file)
            feedback_list = data.get("feedback_list", [])

            likes = sum(1 for fb in feedback_list if fb.get("rating") == 1)
            dislikes = sum(1 for fb in feedback_list if fb.get("rating") == -1)
            corrections = sum(1 for fb in feedback_list if fb.get("correction_text"))

            return {
                "total_feedback": len(feedback_list),
                "likes": likes,
                "dislikes": dislikes,
                "corrections": corrections,
                "satisfaction_rate": round(likes / max(likes + dislikes, 1), 4),
            }
        except Exception:
            return {"total_feedback": 0, "likes": 0, "dislikes": 0, "corrections": 0}
