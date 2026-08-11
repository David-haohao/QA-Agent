# ============================================================
# 事件收集器 — 前端埋点数据收集
# 收集用户行为数据用于进化优化
# ============================================================

import time
import json
import sqlite3
import os
from typing import Dict


class EventCollector:
    """
    前端埋点事件收集器
    收集: 关联问题展示/点击、拒绝后行为、反馈行为等
    """

    def __init__(self, analytics_db_path: str):
        self.db_path = analytics_db_path
        os.makedirs(os.path.dirname(analytics_db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化埋点事件表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                event_type TEXT,
                source_query_id TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def record_event(self, session_id: str, event_type: str,
                     source_query_id: str = "", metadata: Dict = None) -> str:
        """
        记录一个埋点事件
        event_type: answer_displayed / followup_click / feedback_positive /
                   feedback_negative / rejection_shown / re_query_after_rejection /
                   source_link_click / guidance_click
        返回: 事件ID
        """
        event_id = f"evt_{int(time.time() * 1000)}_{session_id[:8]}"
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO analytics_events (session_id, event_type, source_query_id, metadata) VALUES (?, ?, ?, ?)",
                (session_id, event_type, source_query_id, json.dumps(metadata or {}, ensure_ascii=False)),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"记录埋点事件失败: {e}")
        return event_id

    def get_recent_events(self, session_id: str, limit: int = 50) -> list:
        """获取某个会话的最近事件"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT event_type, source_query_id, metadata, created_at FROM analytics_events WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            )
            rows = cursor.fetchall()
            conn.close()
            return [
                {"event_type": r[0], "source_query_id": r[1], "metadata": r[2], "created_at": r[3]}
                for r in rows
            ]
        except Exception:
            return []
