# ============================================================
# 关联问题生成器 — 后处理Step3
# 策略A+B并行优先: 知识图谱关联 + 历史追问匹配
# 结果不足时用C兜底: LLM推理
# ============================================================

import time
from typing import List, Optional


class FollowUpGenerator:
    """
    关联问题生成器
    策略A: 知识图谱关联(document_graph)
    策略B: 历史追问模式(feedback.db)
    策略C: LLM推理(兜底)
    """

    def __init__(self, llm_client, document_graph=None, feedback_db_path: str = None):
        self.llm_client = llm_client
        self.document_graph = document_graph
        self.feedback_db_path = feedback_db_path  # 保留参数兼容性（实际使用JSON文件）

    def generate(self, query: str, answer: str, source_docs: List[dict],
                 timeout_ms: int = 1000) -> List[str]:
        """
        生成关联问题推荐
        返回: 关联问题文本列表，最多5个
        """
        start = time.time()
        results = []
        seen = set()

        # 策略A+B并行执行
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(self._strategy_a_graph, source_docs)
            future_b = executor.submit(self._strategy_b_history, query)

            # 收集A结果
            try:
                for q in future_a.result(timeout=min(timeout_ms / 1000, 0.3)):
                    if q not in seen:
                        results.append(q)
                        seen.add(q)
            except Exception:
                pass

            # 收集B结果
            try:
                for q in future_b.result(timeout=min(timeout_ms / 1000, 0.2)):
                    if q not in seen:
                        results.append(q)
                        seen.add(q)
            except Exception:
                pass

        # 不足3个时触发策略C
        if len(results) < 3:
            remaining = (timeout_ms / 1000) - (time.time() - start)
            if remaining > 0.5:
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_c = executor.submit(self._strategy_c_llm, query, answer, results)
                        for q in future_c.result(timeout=remaining):
                            if q not in seen:
                                results.append(q)
                                seen.add(q)
                except Exception:
                    pass

        # 兜底：提取关键词生成3个关联问题
        if len(results) < 3:
            for q in self._generate_fallback_questions(query, answer):
                if q not in seen and len(results) < 3:
                    results.append(q)
                    seen.add(q)
        # 过滤与原始查询相同或高度相似的问题
        results = self._dedup_against_query(results, query)
        return results[:5]

    def _generate_fallback_questions(self, query: str, answer: str) -> list:
        """从回答中提取关键实体生成关联追问"""
        import re, jieba
        questions = []
        doc_names = re.findall(r'[《「]([^》」]{3,30})[》」]', answer)
        key_terms = re.findall(r'\*\*(.{2,20})\*\*', answer)
        for term in (key_terms[:3] + doc_names[:2]):
            term = term.strip()
            if term and len(term) >= 2:
                questions.append(f"{term}的具体要求和标准是什么？")
                questions.append(f"{term}适用于哪些机构或业务？")
        keywords = [w for w in jieba.cut(query) if 2 <= len(w) <= 6]
        if keywords:
            kw = keywords[0]
            questions.append(f"{kw}的最新修订内容是什么？")
            questions.append(f"违反{kw}相关规定的处罚有哪些？")
        seen = set()
        result = []
        for q in questions:
            if q not in seen and len(result) < 5:
                seen.add(q); result.append(q)
        return result

    def _dedup_against_query(self, questions: list, query: str) -> list:
        """过滤与原始查询相同或高度相似的问题"""
        import re
        q_clean = query.strip().rstrip("？?")
        query_words = set(re.findall(r'[一-鿿]{2,}', q_clean))
        filtered = []
        for q in questions:
            qc = q.strip().rstrip("？?")
            if qc == q_clean:  # 完全相同
                continue
            qw = set(re.findall(r'[一-鿿]{2,}', qc))
            if query_words and qw:
                overlap = len(query_words & qw)
                if overlap / max(len(query_words), 1) > 0.7 and abs(len(qc) - len(q_clean)) < 10:
                    continue  # 高度相似
            filtered.append(q)
        return filtered

    def _strategy_a_graph(self, source_docs: List[dict]) -> List[str]:
        """策略A: 基于知识图谱的关联文档"""
        if not self.document_graph:
            return []
        questions = []
        for doc in source_docs[:3]:
            doc_name = doc.get("name", "")
            # 从关联文档生成关联问题
            # 简化实现: 从文档名推导关联问题
            related_docs = self.document_graph.get_related_documents(doc_name)
            for rd in related_docs[:2]:
                rd_name = rd.get("name", "")
                questions.append(f"{rd_name}中有哪些相关的重要规定？")
        return questions[:3]

    def _strategy_b_history(self, query: str) -> List[str]:
        """策略B: 历史追问模式（从JSON反馈文件中查找相似问题）"""
        import os, json

        # 尝试从JSON反馈文件读取
        json_path = None
        if self.feedback_db_path and self.feedback_db_path.endswith(".db"):
            # 旧路径指向SQLite，转换为JSON路径
            json_path = os.path.join(os.path.dirname(self.feedback_db_path), "feedback.json")
        elif self.feedback_db_path:
            json_path = self.feedback_db_path

        if not json_path or not os.path.exists(json_path):
            # fallback: 尝试qa_records.json
            if self.feedback_db_path:
                records_path = os.path.join(os.path.dirname(self.feedback_db_path), "qa_records.json")
                if os.path.exists(records_path):
                    json_path = records_path
                else:
                    return []
            else:
                return []

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 从qa_records中找相似问题；如果是feedback.json则从feedback_list提取
            records = data.get("qa_list", data.get("feedback_list", []))

            # 简单关键词匹配找相似问题
            query_keywords = set(query[:15].split())
            similar = []
            for r in records:
                rq = r.get("query", "")
                if rq and rq != query:
                    # 检查是否有共同关键词
                    rq_words = set(rq[:15].split())
                    if query_keywords & rq_words:
                        similar.append(rq)
                    elif len(query) >= 4 and query[:4] in rq:
                        similar.append(rq)

            return similar[:3]
        except Exception:
            return []

    def _strategy_c_llm(self, query: str, answer: str, existing: List[str]) -> List[str]:
        """策略C: LLM推理兜底"""
        existing_str = "\n".join(existing) if existing else "无"
        prompt = f"""基于以下问答，生成3个用户可能还想问的关联问题。
要求: 每个问题必须能用中文PDF知识库回答，问题要具体。

用户问题: {query}
系统回答: {answer[:500]}
已生成的关联问题: {existing_str}

请只输出JSON数组: ["问题1", "问题2", "问题3"]"""

        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat_simple(messages)

        try:
            import json
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            result = json.loads(response)
            if isinstance(result, list):
                return [q for q in result if q and q not in existing]
        except Exception:
            pass
        return []
