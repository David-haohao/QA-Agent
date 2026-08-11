# ============================================================
# SSE 连接管理器 — 服务端管理SSE流式推送
# 管理会话级SSE连接，支持多用户同时流式接收
# ============================================================

import asyncio
import json
import time
from typing import Dict, Optional


class SSEManager:
    """
    SSE(Server-Sent Events)连接管理器
    每个会话对应一个异步队列，服务器向队列推送事件
    """

    def __init__(self):
        self._connections: Dict[str, asyncio.Queue] = {}

    def create_session(self, session_id: str) -> asyncio.Queue:
        """为会话创建SSE队列"""
        if session_id not in self._connections:
            self._connections[session_id] = asyncio.Queue(maxsize=100)
        return self._connections[session_id]

    def remove_session(self, session_id: str):
        """移除会话的SSE连接"""
        if session_id in self._connections:
            del self._connections[session_id]

    async def send(self, session_id: str, event: str, data: dict):
        """发送SSE事件"""
        if session_id in self._connections:
            await self._connections[session_id].put((event, data))

    async def send_status(self, session_id: str, phase: str, message: str):
        """发送状态更新事件"""
        await self.send(session_id, "status", {"phase": phase, "message": message})

    async def send_token(self, session_id: str, text: str):
        """发送答案文本片段(流式)"""
        await self.send(session_id, "token", {"text": text})

    async def send_sources(self, session_id: str, sources: list):
        """发送来源列表"""
        await self.send(session_id, "sources", {"sources": sources})

    async def send_followup(self, session_id: str, questions: list):
        """发送关联问题"""
        await self.send(session_id, "followup", {"questions": questions})

    async def send_guidance(self, session_id: str, guidance: dict):
        """发送需求引导"""
        await self.send(session_id, "guidance", guidance)

    async def send_rejected(self, session_id: str, rejection: dict):
        """发送领域外拒绝"""
        await self.send(session_id, "rejected", rejection)

    async def send_no_result(self, session_id: str, data: dict):
        """发送检索无结果"""
        await self.send(session_id, "no_result", data)

    async def send_timeout(self, session_id: str, message: str):
        """发送超时提示"""
        await self.send(session_id, "timeout", {"message": message})

    async def send_done(self, session_id: str, query_id: str, elapsed_ms: int):
        """发送完成信号"""
        await self.send(session_id, "done", {"query_id": query_id, "elapsed_ms": elapsed_ms})

    async def send_error(self, session_id: str, error_message: str):
        """发送错误信息"""
        await self.send(session_id, "error", {"message": error_message})

    def event_generator(self, session_id: str):
        """
        生成SSE事件流(async generator)
        配合FastAPI的StreamingResponse使用
        """
        import asyncio

        async def generate():
            queue = self.create_session(session_id)
            try:
                while True:
                    event, data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                    if event == "done":
                        break
            except asyncio.TimeoutError:
                yield f"event: timeout\ndata: {json.dumps({'message': '请求超时'}, ensure_ascii=False)}\n\n"
            finally:
                self.remove_session(session_id)

        return generate()
