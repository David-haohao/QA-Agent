# ============================================================
# 响应格式化器 — 后处理Step4
# 将答案、来源、关联问题组装为统一的最终输出格式
# ============================================================

from typing import List, Dict


class ResponseFormatter:
    """
    最终响应格式化器
    将完整的问答结果组装为统一输出结构
    """

    def format(self, answer: str, sources: List[Dict], followup_questions: List[str],
               query_id: str = "", elapsed_ms: int = 0) -> Dict:
        """
        组装最终输出
        返回: 包含所有字段的统一响应字典
        """
        return {
            "query_id": query_id,
            "answer": answer,
            "sources": [
                {
                    "name": s["name"],
                    "path": s["path"],
                    "url": f"/kb/documents/{s.get('path', '').split('/')[-1]}" if s.get("path") else "",
                    "chunk_count": len(s.get("chunks", [])),
                }
                for s in sources
            ],
            "followup_questions": followup_questions,
            "elapsed_ms": elapsed_ms,
            "has_sources": len(sources) > 0,
            "source_count": len(sources),
        }
