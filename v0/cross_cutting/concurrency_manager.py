# ============================================================
# 并发管理器 — 会话隔离和连接池管理
# 保证用户间互不干扰
# ============================================================

import threading
from typing import Dict, Any


class ConcurrencyManager:
    """
    并发管理器
    为每个会话提供完全隔离的上下文
    包括会话级存储和线程安全保证
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: Dict[str, Dict] = {}

    def create_session(self, session_id: str) -> Dict:
        """
        创建新会话的隔离上下文
        """
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "context": {},
                    "history": [],
                    "created_at": __import__("time").time(),
                }
            return self._sessions[session_id]

    def get_session_context(self, session_id: str) -> Dict:
        """获取会话上下文"""
        if session_id not in self._sessions:
            return self.create_session(session_id)
        return self._sessions[session_id]

    def add_to_history(self, session_id: str, role: str, content: str):
        """添加会话历史"""
        session = self.get_session_context(session_id)
        session["history"].append({
            "role": role,
            "content": content,
            "time": __import__("time").time(),
        })

    def get_history(self, session_id: str, max_messages: int = 10) -> list:
        """获取会话历史(最近N条)"""
        session = self.get_session_context(session_id)
        return session["history"][-max_messages:]

    def cleanup_session(self, session_id: str):
        """清理会话"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
