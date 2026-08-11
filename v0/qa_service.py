# ============================================================
# QA Service — 核心问答服务编排器
# 串联整个问答流程：预处理 → 缓存检查 → 检索 → Agent生成 → 后处理
# 所有模块通过此服务协调工作，对外暴露统一的流式问答接口
# ============================================================

import os
import json
import time
import uuid
from typing import AsyncIterator, Tuple

from config_loader import load_config, Config
from models.llm_client import LLMClient
from models.embedding_client import EmbeddingClient
from models.reranker_client import RerankerClient

from pre_processing.query_normalizer import QueryNormalizer
from pre_processing.completeness_checker import CompletenessChecker
from pre_processing.domain_guard import DomainGuard
from pre_processing.rejection_templates import RejectionTemplates

from knowledge_base.pipeline import KnowledgeBasePipeline
from knowledge_base.retrieval.retriever import HybridRetriever
from knowledge_base.retrieval.source_binding import SourceBinding
from knowledge_base.indexing.vector_index import VectorIndexBuilder
from knowledge_base.indexing.bm25_index import BM25Index
from knowledge_base.indexing.document_graph import DocumentGraph

from qa_agent.agent import create_qa_agent
from qa_agent.tools import init_kb
from qa_agent.consistency_cache import ConsistencyCache
from qa_agent.intent_extractor import IntentExtractor
from qa_agent.evolution_hook import EvolutionHook

from post_processing.source_attacher import SourceAttacher
from post_processing.answer_verifier import AnswerVerifier
from post_processing.followup_generator import FollowUpGenerator
from post_processing.formatter import ResponseFormatter

from cross_cutting.response_time_controller import ResponseTimeController
from cross_cutting.concurrency_manager import ConcurrencyManager
from cross_cutting.async_executor import AsyncExecutor

from analytics.event_collector import EventCollector
from frontend.sse_manager import SSEManager


class QAService:
    """
    核心问答服务
    封装完整的问答处理链路（预处理→检索→生成→后处理）
    支持SSE流式输出和多用户会话隔离
    """

    def __init__(self, config_path: str = None):
        """
        初始化所有组件
        config_path: 配置文件路径，默认读取config.yaml
        """
        # 加载配置
        self.config = load_config(config_path)

        # 初始化模型客户端
        self.llm_client = LLMClient(self.config.llm_config)
        self.embedding_client = EmbeddingClient(self.config.embeddings_config)
        self.reranker_client = RerankerClient(self.config.reranker_config)

        # 初始化预处理组件
        self.normalizer = QueryNormalizer()
        self.completeness_checker = CompletenessChecker(
            threshold=self.config.pre_processing_config.get("completeness_threshold", 0.6)
        )
        self.domain_guard = DomainGuard(
            domain_keywords=self.config.domain_keywords,
            domains=self.config.domains,
            embedding_client=self.embedding_client,
            llm_client=self.llm_client,
            similarity_threshold=self.config.kb_config.get("embedding_similarity_threshold", 0.3),
            similarity_release=self.config.kb_config.get("embedding_similarity_release", 0.6),
            domain_confidence_threshold=self.config.pre_processing_config.get("domain_confidence_threshold", 0.5),
        )
        self.rejection_templates = RejectionTemplates(self.config.domains)

        # 初始化进化钩子
        self.evolution_hook = EvolutionHook(
            evolution_data_dir=self.config.evolution_config.get("evolution_data_dir", "./evolution_data"),
            feedback_db_name=self.config.evolution_config.get("feedback_db", "feedback.db"),
            sampling_rate=self.config.evolution_config.get("sampling_rate", 0.1),
        )

        # 初始化知识库流水线
        self.kb_pipeline = KnowledgeBasePipeline(
            config=self.config.kb_config,
            embedding_client=self.embedding_client,
            reranker_client=self.reranker_client,
        )

        # 初始化缓存
        self.intent_extractor = IntentExtractor(self.llm_client)
        self.cache = ConsistencyCache(
            config=self.config.cache_config,
            intent_extractor=self.intent_extractor,
        )

        # 初始化后处理组件
        self.source_attacher = SourceAttacher()
        self.answer_verifier = AnswerVerifier(self.llm_client)
        self.formatter = ResponseFormatter()

        # 初始化文档图谱(用于关联问题生成)
        doc_graph = DocumentGraph(self.config.kb_config.get("kb_data_dir", "./kb_data"))

        # 初始化关联问题生成器
        self.followup_generator = FollowUpGenerator(
            llm_client=self.llm_client,
            document_graph=doc_graph,
            feedback_db_path=os.path.join(
                self.config.evolution_config.get("evolution_data_dir", "./evolution_data"),
                "qa_records.json",
            ),
        )

        # 初始化横切组件
        self.time_controller = ResponseTimeController(self.config.timeout_config)
        self.concurrency_mgr = ConcurrencyManager()
        self.async_executor = AsyncExecutor()

        # 初始化SSE管理器
        self.sse_manager = SSEManager()

        # 初始化埋点收集器
        self.analytics = EventCollector(
            analytics_db_path=os.path.join(
                self.config.evolution_config.get("evolution_data_dir", "./evolution_data"),
                "analytics.db",
            )
        )

        # 初始化QA Agent — 注册知识库到工具模块
        init_kb(self.kb_pipeline)

        # 每个会话独立一个MemorySaver checkpointer用于多轮记忆
        self._session_checkpointers: dict = {}
        self._session_agents: dict = {}

        # 短时记忆：存储最近N对问答（OrderedDict实现LRU）
        max_pairs = self.config.cache_config.get("short_memory_max_pairs", 50)
        from collections import OrderedDict
        self._short_memory: OrderedDict = OrderedDict()
        self._short_memory_max = max_pairs

    async def process_query_stream(self, query: str, session_id: str):
        """
        处理查询的完整流式流程（异步生成器）
        返回SSE事件: yield (event_name, data_dict)
        """
        start_time = time.time()
        self.time_controller.start()

        # 创建会话上下文
        self.concurrency_mgr.create_session(session_id)
        sse = self.sse_manager

        # ========== 预处理层: 分析问题并构建Agent指令 ==========
        yield ("status", {
            "phase": "preprocessing",
            "message": "正在理解您的问题...",
            "elapsed_ms": int(self.time_controller.elapsed_ms()),
        })

        # Step 1: 查询标准化
        normalized = self.normalizer.normalize(query)

        # Step 2+3: 检查完整性 + 领域边界，构建Agent引导指令
        is_followup = any(w in normalized for w in
            ["它", "他", "她", "这个", "那个", "这些", "那些", "上面", "前面", "刚才", "之前", "刚刚", "上次"])
        is_complete, completeness_score, _ = self.completeness_checker.check(normalized)
        print('--------is_complete, completeness_score: {} {}'.format(is_complete, completeness_score))
        domain_list_str = "、".join(self.config.domains)

        agent_prefix = ""
        if not is_complete and not is_followup:
            agent_prefix = (
                f"[预处理分析] 用户问题「{normalized}」的完备性评分为{completeness_score:.0%}，表述较简短。"
                f"请先用 search_knowledge_base 搜索知识库中是否有相关文档（用用户问题中的关键词搜索），"
                f"如果搜索到相关内容则基于内容回答；如果搜索结果不足以判断用户意图，"
                f"则自然地引导用户补充具体信息（如：想了解定义/条款/范围/流程？涉及哪个业务场景？）。"
            )
        elif is_complete and not is_followup:
            is_in_domain, reason, detail = self.domain_guard.judge(normalized)
            if not is_in_domain:
                agent_prefix = (
                    f"[预处理分析] 用户问题「{normalized}」的主题可能是「{detail.get('topic', normalized)}」，"
                    f"初步判断超出了知识库覆盖领域（{domain_list_str}）。"
                    f"请先用一段话分析用户问题的主题，然后礼貌告知知识库暂不覆盖此领域，"
                    f"并根据具体问题给出自然的转方向建议，不要使用固定模板，让用户感觉被理解。"
                )

        if agent_prefix:
            augmented_query = f"{agent_prefix}\n\n[用户问题] {normalized}"
        else:
            augmented_query = normalized

        yield ("status", {
            "phase": "preprocessing",
            "message": f"问题分析完成 ({(self.time_controller.elapsed_ms()/1000):.1f}s)",
            "elapsed_ms": int(self.time_controller.elapsed_ms()),
        })

        # ========== 短时记忆查询（最近N对问答） ==========
        max_pairs = self.config.cache_config.get("short_memory_max_pairs", 50)
        short_mem = self._get_short_memory(normalized)
        if short_mem:
            cached_answer = short_mem["answer"]
            cached_sources = short_mem["sources"]
            cached_followups = short_mem.get("followup_questions", [])

            query_id = self.evolution_hook.generate_query_id()
            elapsed = int(self.time_controller.elapsed_ms())

            self.concurrency_mgr.add_to_history(session_id, "user", query)
            self.concurrency_mgr.add_to_history(session_id, "assistant", cached_answer)

            yield ("message_start", {"message_id": query_id, "elapsed_ms": elapsed})
            yield ("content_block_delta", {"block_index": 0, "text": cached_answer})
            yield ("content_block_stop", {"block_index": 0})
            # 仅当问题完备且答案来自知识库时才显示来源和关联问题
            if is_complete and cached_sources:
                yield ("sources", {"sources": cached_sources})
            if is_complete and cached_followups:
                yield ("followup", {"questions": cached_followups})
            yield ("message_stop", {"message_id": query_id, "elapsed_ms": elapsed})
            yield ("done", {"query_id": query_id, "elapsed_ms": elapsed, "cache_hit": True})

            self.evolution_hook.record_qa(
                session_id=session_id, query_id=query_id, query=normalized,
                answer=cached_answer, source_docs=cached_sources,
                elapsed_ms=elapsed, cache_hit=True,
            )
            return

        # ========== 缓存查询 ==========
        cache_result = self.cache.lookup(normalized)
        if cache_result:
            cached_answer = cache_result.get("answer", "")
            cached_sources = cache_result.get("sources", [])
            cached_followups = cache_result.get("followup_questions", [])

            query_id = self.evolution_hook.generate_query_id()
            elapsed = int(self.time_controller.elapsed_ms())

            # 保存历史
            self.concurrency_mgr.add_to_history(session_id, "user", query)
            self.concurrency_mgr.add_to_history(session_id, "assistant", cached_answer)

            # 返回缓存答案（使用content_block_delta事件，前端才能渲染答案文本）
            yield ("message_start", {"message_id": query_id, "elapsed_ms": elapsed})
            yield ("content_block_delta", {"block_index": 0, "text": cached_answer})
            yield ("content_block_stop", {"block_index": 0})
            # 仅当问题完备且答案来自知识库时才显示来源和关联问题
            if is_complete and cached_sources:
                yield ("sources", {"sources": cached_sources})
            if is_complete and cached_followups:
                yield ("followup", {"questions": cached_followups})
            yield ("message_stop", {"message_id": query_id, "elapsed_ms": elapsed})
            yield ("done", {"query_id": query_id, "elapsed_ms": elapsed, "cache_hit": True})

            # 记录
            self.evolution_hook.record_qa(
                session_id=session_id, query_id=query_id, query=normalized,
                answer=cached_answer, source_docs=cached_sources,
                elapsed_ms=elapsed, cache_hit=True,
            )
            return

        # ========== Agent 驱动的检索+生成 ==========
        query_id = self.evolution_hook.generate_query_id()
        yield ("message_start", {
            "message_id": query_id,
            "elapsed_ms": int(self.time_controller.elapsed_ms()),
        })

        # 获取或创建此会话的agent（带MemorySaver checkpointer实现多轮记忆）
        agent = self._get_session_agent(session_id)
        thread_config = self._get_thread_config(session_id)

        # 设置工具端的会话追踪（tools.py 自动记录 search_knowledge_base 检索到的文档）
        from qa_agent.tools import set_tracking_session, get_and_clear_retrieved_docs
        set_tracking_session(session_id)

        agent_tool_called = False
        full_answer = ""
        block_idx = 0
        _thinking_sent = False
        _in_answer_phase = False
        _think_buffer = ""
        _tool_ids_seen = set()
        try:
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": augmented_query}]},
                config=thread_config,
                stream_mode=["messages", "values"],
            ):
                mode, data = chunk

                if mode == "values":
                    # 仅用于检测 tool_result 状态，不干预流式
                    pass

                elif mode == "messages":
                    message = data[0] if isinstance(data, tuple) else data
                    msg_type = getattr(message, "type", "")

                    if msg_type in ("ai", "AIMessageChunk"):
                        # === reasoning_content 思考提取 ===
                        reasoning = None
                        if hasattr(message, "additional_kwargs"):
                            reasoning = message.additional_kwargs.get("reasoning_content", "")
                        if reasoning and not _thinking_sent:
                            text = self._translate_thinking(str(reasoning))
                            text = _strip_english_lines(text)
                            if text.strip():
                                yield ("thinking", {
                                    "block_index": block_idx,
                                    "text": text,
                                    "elapsed_ms": int(self.time_controller.elapsed_ms()),
                                })
                                _thinking_sent = True
                                _in_answer_phase = True

                        # === 工具调用：先于内容发送 ===
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tc in message.tool_calls:
                                tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                                if tc_id and tc_id not in _tool_ids_seen:
                                    _tool_ids_seen.add(tc_id)
                                    agent_tool_called = True
                                    tc_name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                                    tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                                    yield ("tool_use", {
                                        "block_index": block_idx,
                                        "tool_name": tc_name,
                                        "tool_call_id": tc_id,
                                        "tool_input": json.dumps(tc_args, ensure_ascii=False) if isinstance(tc_args, dict) else str(tc_args),
                                        "status": "running",
                                        "elapsed_ms": int(self.time_controller.elapsed_ms()),
                                    })
                            _thinking_sent = False
                            _in_answer_phase = False
                            _think_buffer = ""

                        # === Content token：立即逐 token 流式输出 ===
                        content = getattr(message, "content", "")
                        if isinstance(content, str) and content:
                            full_answer += content

                            if not _in_answer_phase:
                                # 思考阶段：积累直到检测到 </think>
                                _think_buffer += content
                                if '</think>' in _think_buffer.lower():
                                    idx = _think_buffer.lower().find('</think>')
                                    # 发送思考内容（完整块）
                                    think_text = _strip_english_lines(_filter_thinking_text(_think_buffer[:idx].strip()))
                                    if think_text:
                                        yield ("thinking", {
                                            "block_index": block_idx,
                                            "text": think_text,
                                            "elapsed_ms": int(self.time_controller.elapsed_ms()),
                                        })
                                    _thinking_sent = True
                                    _in_answer_phase = True
                                    # </think> 之后的文本立即流式输出
                                    after = _think_buffer[idx + len('</think>'):]
                                    _think_buffer = ""
                                    cleaned = self._safe_answer_text(after)
                                    if cleaned:
                                        yield ("content_block_delta", {
                                            "block_index": block_idx,
                                            "text": cleaned,
                                        })
                            else:
                                # 答案阶段：每个 token 立即流式输出
                                cleaned = self._safe_answer_text(content)
                                if cleaned:
                                    yield ("content_block_delta", {
                                        "block_index": block_idx,
                                        "text": cleaned,
                                    })

                    # === Tool Result ===
                    if msg_type == "tool" and hasattr(message, "tool_call_id"):
                        yield ("tool_result", {
                            "block_index": block_idx,
                            "tool_name": getattr(message, "name", "unknown"),
                            "tool_call_id": getattr(message, "tool_call_id", ""),
                            "result": str(getattr(message, "content", "")),
                            "status": "completed",
                            "elapsed_ms": int(self.time_controller.elapsed_ms()),
                        })

                block_idx += 1

        except Exception as e:
            full_answer = f"[生成失败] {str(e)}"
            yield ("content_block_delta", {"block_index": block_idx, "text": full_answer})

        yield ("content_block_stop", {"block_index": block_idx})
        # 用 _safe_answer_text 清理 full_answer 中的一切 think 标签残留
        answer = self._safe_answer_text(full_answer).strip()
        print('========answer: {}'.format(answer[:200] if len(answer) > 200 else answer))

        # ========== 后处理 ==========
        yield ("status", {
            "phase": "post_processing",
            "message": f"正在整理回答... ({(self.time_controller.elapsed_ms()/1000):.1f}s)",
            "elapsed_ms": int(self.time_controller.elapsed_ms()),
        })

        # ========== 来源附加：从答案文本提取引用（不使用工具追踪数据） ==========
        # 清理工具追踪数据（不使用——来源块只应包含答案文本实际引用的文档）
        get_and_clear_retrieved_docs(session_id)

        # 1. 先从完整答案中提取 Agent 引用的文档名（在剥离来源块之前）
        cited_docs = self._extract_cited_sources_from_answer(answer)

        # 2. 从答案文本中剥离 **来源：** 块（前端来源附加块替代）
        answer = self._strip_source_section(answer)

        # 3. 基于答案文本中实际引用的文档构建来源列表（不使用工具追踪数据）
        sources = self._build_consistent_sources(
            cited_docs=cited_docs,
            tracked_docs=[],  # 不使用工具追踪：来源块只展示文本内容引用的文档
        )

        # 来源显示：只要Agent调用了搜索工具且有匹配的来源文档，就必须展示「参考来源」模块
        # 不依赖完整性评分——即使问题简短，只要从知识库找到了答案就要展示来源
        if agent_tool_called and sources:
            yield ("sources", {"sources": sources})

        # 关联问题生成：仅在问题完整且Agent搜索了知识库时生成
        can_show_followups = is_complete and agent_tool_called
        followups = []
        if can_show_followups and sources and not self.time_controller.should_skip_followup():
            followups = self.followup_generator.generate(
                query=normalized,
                answer=answer,
                source_docs=sources,
                timeout_ms=1000,
            )
            if followups:
                yield ("followup", {"questions": followups})

        elapsed_ms = int(self.time_controller.elapsed_ms())

        # ========== 写入短时记忆 + 缓存 ==========
        self._store_short_memory(normalized, answer, sources, followups)
        cache_data = {
            "answer": answer,
            "sources": sources,
            "followup_questions": followups,
        }
        self.cache.store(normalized, cache_data)

        # ========== 保存会话历史 ==========
        self.concurrency_mgr.add_to_history(session_id, "user", query)
        self.concurrency_mgr.add_to_history(session_id, "assistant", answer)

        # ========== 进化记录 ==========
        self.evolution_hook.record_qa(
            session_id=session_id, query_id=query_id, query=normalized,
            answer=answer, source_docs=sources,
            elapsed_ms=elapsed_ms, cache_hit=False,
            retrieval_count=0,
        )

        # ========== 发送完成 ==========
        yield ("message_stop", {
            "message_id": query_id,
            "elapsed_ms": elapsed_ms,
        })
        yield ("done", {
            "query_id": query_id,
            "elapsed_ms": elapsed_ms,
            "cache_hit": False,
        })

    # ===== 会话Agent管理 =====

    def _get_session_agent(self, session_id: str):
        """获取或创建某个会话的agent实例（带独立的checkpointer用于多轮记忆）"""
        if session_id not in self._session_agents:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            self._session_checkpointers[session_id] = checkpointer
            # 每个会话独立创建agent，确保历史隔离
            agent = create_qa_agent(
                llm_client=self.llm_client,
                kb_pipeline=self.kb_pipeline,
                checkpointer=checkpointer,
            )
            self._session_agents[session_id] = agent
        return self._session_agents[session_id]

    def _get_thread_config(self, session_id: str) -> dict:
        """获取LangGraph的thread配置（thread_id=session_id，实现会话记忆）"""
        return {"configurable": {"thread_id": f"qa_{session_id}"}}

    # ===== 辅助方法 =====

    def record_feedback(self, session_id: str, query_id: str, rating: int, comment: str = "",
                        correction_type: str = "", correction_text: str = "",
                        query: str = "", answer: str = "", source_docs: list = None):
        """记录用户反馈（点赞/点踩/纠错）——保存到本地JSON文件"""
        self.evolution_hook.record_feedback(
            session_id=session_id, query_id=query_id,
            rating=rating, comment=comment,
            correction_type=correction_type, correction_text=correction_text,
            query=query, answer=answer, source_docs=source_docs or [],
        )

    def record_analytics_event(self, session_id: str, event_type: str,
                                source_query_id: str = "", metadata: dict = None):
        """记录前端埋点事件"""
        self.analytics.record_event(
            session_id=session_id,
            event_type=event_type,
            source_query_id=source_query_id,
            metadata=metadata,
        )

    def create_session(self, session_id: str, user_name: str = ""):
        """创建会话"""
        self.concurrency_mgr.create_session(session_id)

    def get_kb_overview(self) -> dict:
        """获取知识库概览"""
        try:
            vi = VectorIndexBuilder(
                kb_data_dir=self.config.kb_config.get("kb_data_dir", "./kb_data"),
                collection_name=self.config.kb_config.get("chroma_collection", "qa_knowledge"),
                embedding_client=self.embedding_client,
                dimension=self.config.embeddings_config.get("dimension", 1024),
            )
            doc_list = vi.list_documents()
            return {
                "domains": self.config.domains,
                "doc_count": len(doc_list),
                "documents": doc_list[:20],
                "last_updated": "见metadata.json",
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_short_memory(self, normalized_query: str) -> dict | None:
        """从短时记忆中查找已存储的问答对"""
        key = normalized_query.strip()
        if key in self._short_memory:
            self._short_memory.move_to_end(key)  # LRU: 移到末尾
            return self._short_memory[key]
        return None

    def _store_short_memory(self, normalized_query: str, answer: str, sources: list, followups: list):
        """存储问答对到短时记忆（LRU淘汰）"""
        key = normalized_query.strip()
        if key in self._short_memory:
            self._short_memory.move_to_end(key)
        self._short_memory[key] = {
            "answer": answer,
            "sources": sources,
            "followup_questions": followups,
        }
        # LRU淘汰: 超过上限时删除最旧的
        while len(self._short_memory) > self._short_memory_max:
            self._short_memory.popitem(last=False)

    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        stats = self.cache.get_stats()
        stats["short_memory_size"] = len(self._short_memory)
        stats["short_memory_max"] = self._short_memory_max
        return stats

    def get_source_detail(self, filename: str) -> dict:
        """根据文档名获取详情，供前端点击来源链接时展示"""
        import os, json
        kb_dir = self.config.kb_config.get("kb_data_dir", "./kb_data")
        meta_path = os.path.join(kb_dir, "metadata.json")

        # 先尝试映射到实际文档名
        resolved = self._resolve_source_name(filename)

        # 检查KB元数据
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            doc_names = meta.get("doc_names", [])
            if resolved in doc_names or filename in doc_names:
                return {
                    "name": resolved if resolved in doc_names else filename,
                    "matched_name": resolved if resolved != filename else None,
                    "found": True,
                    "kb_doc_count": meta.get("total_docs", 0),
                    "kb_chunk_count": meta.get("total_chunks", 0),
                }
        elif self._verify_doc_exists(filename):
            return {
                "name": resolved,
                "matched_name": resolved if resolved != filename else None,
                "found": True,
            }

        return {"name": filename, "found": False}

    def get_document_full_text(self, filename: str) -> tuple:
        """获取文档全文文本，供/kb/view/{filename}页面展示
        返回 (content, matched_name)。支持模糊匹配：LLM简称可匹配实际文件名"""
        import os, re
        kb_dir = self.config.kb_config.get("kb_data_dir", "./kb_data")
        docs_dir = self.config.kb_config.get("documents_dir", "./documents")

        # 先尝试通过_match_doc_in_kb解析到实际文件名
        resolved_candidates = self._match_doc_in_kb(filename)
        resolved_name = resolved_candidates[0] if resolved_candidates else None

        # 提取纯文档名（去路径、去扩展名、去前后缀编号）
        clean_name = os.path.splitext(filename)[0]
        clean_name = re.sub(r'^\d+[_\-\s]*', '', clean_name)
        clean_name = clean_name.strip()

        # 路径1: 先尝试精确匹配resolved_name
        if resolved_name:
            resolved_path = os.path.join(docs_dir, resolved_name)
            if os.path.exists(resolved_path):
                try:
                    from knowledge_base.extractors import DocumentExtractor
                    ext = DocumentExtractor(docs_dir)
                    fl = resolved_name.lower()
                    if fl.endswith(".pdf"):
                        result = ext._extract_pdf(resolved_path, resolved_name)
                    elif fl.endswith(".docx"):
                        result = ext._extract_docx(resolved_path, resolved_name)
                    elif fl.endswith(".doc"):
                        result = ext._extract_doc(resolved_path, resolved_name)
                    elif fl.endswith((".xlsx", ".xls")):
                        result = ext._extract_excel(resolved_path, resolved_name)
                    else:
                        result = ext._extract_text_file(resolved_path, resolved_name)
                    if result and result.get("content"):
                        return result["content"], resolved_name
                except Exception:
                    pass
            # 也尝试从docs_text读取
            docs_text_dir = os.path.join(kb_dir, "docs_text")
            if os.path.isdir(docs_text_dir):
                text_path = os.path.join(docs_text_dir, resolved_name + ".txt")
                if os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8") as f:
                        return f.read(), os.path.splitext(resolved_name)[0]

        # 路径2: 从docs_text/模糊搜索
        docs_text_dir = os.path.join(kb_dir, "docs_text")
        if os.path.isdir(docs_text_dir):
            for fname in sorted(os.listdir(docs_text_dir)):
                fname_no_ext = os.path.splitext(fname)[0]
                # 精确匹配或子串匹配
                if (fname_no_ext == filename or
                    filename in fname_no_ext or fname_no_ext in filename or
                    clean_name in fname_no_ext or
                    (len(clean_name) > 4 and clean_name[:6] in fname_no_ext) or
                    (resolved_name and fname_no_ext == resolved_name)):
                    with open(os.path.join(docs_text_dir, fname), "r", encoding="utf-8") as f:
                        return f.read(), fname_no_ext

        # 路径3: 从documents/模糊搜索
        if os.path.isdir(docs_dir):
            for fname in sorted(os.listdir(docs_dir)):
                if (fname == filename or filename in fname or fname in filename or
                    clean_name in fname or
                    (resolved_name and fname == resolved_name)):
                    doc_path = os.path.join(docs_dir, fname)
                    try:
                        from knowledge_base.extractors import DocumentExtractor
                        ext = DocumentExtractor(docs_dir)
                        fl = fname.lower()
                        if fl.endswith(".pdf"):
                            result = ext._extract_pdf(doc_path, fname)
                        elif fl.endswith(".docx"):
                            result = ext._extract_docx(doc_path, fname)
                        elif fl.endswith(".doc"):
                            result = ext._extract_doc(doc_path, fname)
                        elif fl.endswith((".xlsx", ".xls")):
                            result = ext._extract_excel(doc_path, fname)
                        else:
                            result = ext._extract_text_file(doc_path, fname)
                        if result and result.get("content"):
                            return result["content"], fname
                    except Exception:
                        continue

        # 路径4: 直接当文本文件读
        doc_path = os.path.join(docs_dir, filename)
        if os.path.exists(doc_path):
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    return f.read(), filename
            except Exception:
                pass

        return None, None

    @staticmethod
    def _safe_answer_text(text: str) -> str:
        """确保答案文本不含任何  思考标签残留"""
        import re
        # 移除完整的 ...  标签对（及其内容）
        text = re.sub(r'<\s*think\s*>.*?</think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # 移除孤立的 </think> 或 <think/> 标签
        text = re.sub(r'</?think\s*/?>', '', text, flags=re.IGNORECASE)
        return text

    def _translate_thinking(self, text: str) -> str:
        """将LLM思考内容转为纯中文——英文≥3词时用LLM翻译，否则规则过滤"""
        import re
        if not text or not text.strip():
            return ""

        # 统计英文单词数
        eng_words = len(re.findall(r'\b[A-Za-z]{2,}\b', text))

        # 英文少 → 规则过滤
        if eng_words < 3:
            result = _filter_thinking_text(text)
            return _strip_english_lines(result)

        # 英文较多 → LLM 翻译为中文
        try:
            prompt = (
                '请将以下内容翻译为纯中文。要求：\n'
                '1. 只输出翻译后的中文，不要任何解释\n'
                '2. 专业术语保留英文缩写但必须用中文解释含义\n'
                '3. 中文表达自然流畅\n\n'
                f'{text}'
            )
            msgs = [{"role": "user", "content": prompt}]
            translated = self.llm_client.chat_simple(msgs)
            # 翻译后再做一次规则清理确保无英文残留
            return _strip_english_lines(_filter_thinking_text(translated))
        except Exception:
            return _strip_english_lines(_filter_thinking_text(text))

    def _clean_content_delta(self, raw_delta: str, full_answer: str, prev_answer: str):
        """过滤  response 思考标签，分离思考内容和答案内容。

        当模型（如Qwen）在content文本中输出  response 标签时：
        - 提取  response 之间的内容作为思考过程
        - 返回去除思考标签后的答案内容

        返回: (cleaned_delta, think_text)
        """
        import re

        think_text = ""
        cleaned_delta = raw_delta

        # 检测这次的 full_answer 是否跨过了  结束标签
        think_end_match = re.search(r'</think\s*>', full_answer, re.IGNORECASE)
        think_start_match = re.search(r'<\s*think\s*>', full_answer, re.IGNORECASE)

        if think_end_match:
            think_end_pos = think_end_match.end()
            # 思考部分：从 <s> 或开头到 </think>
            if think_start_match:
                think_content = full_answer[think_start_match.end():think_end_match.start()]
            else:
                think_content = full_answer[:think_end_match.start()]

            # 清理思考文本
            think_text = _strip_english_lines(_filter_thinking_text(think_content.strip()))

            if prev_answer:
                prev_end_match = re.search(r'</think>\s*', prev_answer)
            else:
                prev_end_match = None

            if not prev_end_match:
                # 这是首次跨过 </think> 标签——需要：
                # 1. 发送思考内容
                # 2. 只返回 </think> 之后的文本作为答案delta
                after_think = full_answer[think_end_pos:]
                # delta = (after_think相对于prev_answer的增量)
                # 由于prev_answer在</think>之前，prev_answer的答案部分为空
                # 所以clean_delta = after_think (全量)
                cleaned_delta = after_think
            else:
                # 之前已经跨过 </think>，这次delta应该只包含答案部分
                prev_end_pos = prev_end_match.end()
                prev_after_think = prev_answer[prev_end_pos:]
                after_think = full_answer[think_end_pos:]
                if len(after_think) > len(prev_after_think):
                    cleaned_delta = after_think[len(prev_after_think):]
                else:
                    cleaned_delta = ""

        return cleaned_delta, think_text

    def _build_doc_name_index(self) -> dict:
        """构建文档名索引，用于快速查找和模糊匹配
        返回 {
            "exact_map": {normalized_name: original_filename},
            "all_names": [original_filenames],
            "all_bases": [(base_name, original_filename)],
        }
        优先从metadata.json读取（已索引的文档），fallback到documents目录"""
        import os, json, re

        kb_dir = self.config.kb_config.get("kb_data_dir", "./kb_data")
        docs_dir = self.config.kb_config.get("documents_dir", "./documents")

        all_names = set()
        # 路径1: 从metadata.json获取已索引文档名（权威来源）
        meta_path = os.path.join(kb_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                for name in meta.get("doc_names", []):
                    all_names.add(name)
            except Exception:
                pass

        # 路径2: 补充documents目录中的文件
        if os.path.isdir(docs_dir):
            for fname in os.listdir(docs_dir):
                all_names.add(fname)

        # 构建精确匹配映射: 规范化名 → 原始文件名
        def _normalize(name: str) -> str:
            """规范化文档名用于匹配"""
            n = os.path.splitext(name)[0]  # 去扩展名
            n = re.sub(r'^\d+[_\-\s]*', '', n)  # 去前导编号
            n = re.sub(r'[_\-\s]+', '', n)      # 去分隔符
            return n.lower().strip()

        all_bases = []
        exact_map = {}
        for name in sorted(all_names):
            base = os.path.splitext(name)[0]
            all_bases.append((base, name))
            norm = _normalize(name)
            exact_map[norm] = name
            # 同时注册base名（去扩展名）
            if base != name:
                exact_map.setdefault(_normalize(base), base if '.' in name else name)

        return {
            "exact_map": exact_map,
            "all_names": sorted(all_names),
            "all_bases": all_bases,
        }

    def _match_doc_in_kb(self, short_name: str) -> list:
        """在知识库中模糊搜索匹配的文档名
        多策略匹配：精确匹配 → 规范化匹配 → 关键词匹配 → 序列相似度匹配
        返回匹配到的原始文件名的列表"""
        import os, re

        if not short_name or len(short_name) < 2:
            return []

        docs_dir = self.config.kb_config.get("documents_dir", "./documents")
        kb_dir = self.config.kb_config.get("kb_data_dir", "./kb_data")

        # 构建索引（带缓存）
        if not hasattr(self, '_doc_index_cache'):
            self._doc_index_cache = self._build_doc_name_index()
        idx = self._doc_index_cache

        # 清理输入
        clean = os.path.splitext(short_name)[0]
        clean = re.sub(r'^\d+[_\-\s]*', '', clean)
        clean = clean.strip()

        matches = []

        def _normalize(name: str) -> str:
            n = os.path.splitext(name)[0]
            n = re.sub(r'^\d+[_\-\s]*', '', n)
            n = re.sub(r'[_\-\s]+', '', n)
            return n.lower().strip()

        norm_input = _normalize(short_name)

        # 策略1: 精确规范化匹配
        if norm_input in idx["exact_map"]:
            matched = idx["exact_map"][norm_input]
            if matched not in matches:
                matches.append(matched)

        # 策略2: 子串匹配（short_name在文件名中，或文件名在short_name中）
        for base, orig in idx["all_bases"]:
            if orig in matches:
                continue
            norm_base = _normalize(base)
            # short_name是文件名的子串（如"资本管理办法"在"43_商业银行资本管理办法_附件14-19"中）
            if len(norm_input) >= 4 and norm_input in norm_base:
                matches.append(orig)
            # 文件名是short_name的子串
            elif len(norm_base) >= 6 and norm_base in norm_input:
                matches.append(orig)
            # clean名匹配
            elif len(clean) >= 4:
                if clean in base or base in clean:
                    matches.append(orig)

        # 策略3: 关键词匹配（短名的每个关键词都在文件名中出现）
        if not matches and len(clean) >= 3:
            keywords = re.findall(r'[一-鿿]{2,}|[A-Za-z]{3,}|\d{3,}', clean)
            if keywords:
                for base, orig in idx["all_bases"]:
                    if orig in matches:
                        continue
                    if all(kw in base for kw in keywords):
                        matches.append(orig)

        # 策略4: difflib序列相似度匹配（最后手段）
        if not matches and len(clean) >= 4:
            from difflib import SequenceMatcher
            best_score = 0
            best_match = None
            for base, orig in idx["all_bases"]:
                if orig in matches:
                    continue
                # 用规范化名称比较
                score = SequenceMatcher(None, norm_input, _normalize(base)).ratio()
                if score > best_score:
                    best_score = score
                    best_match = orig
            if best_match and best_score >= 0.6:
                matches.append(best_match)

        # 如果索引匹配不够，回退扫描docs目录（兼容旧逻辑但有索引加速）
        if not matches:
            for d in [os.path.join(kb_dir, "docs_text"), docs_dir]:
                if not os.path.isdir(d):
                    continue
                for fname in sorted(os.listdir(d)):
                    if d.endswith("docs_text"):
                        fname = os.path.splitext(fname)[0]
                    if fname in matches:
                        continue
                    base = os.path.splitext(fname)[0]
                    if clean in base or base in clean or norm_input in _normalize(fname):
                        matches.append(fname)
                        if len(matches) >= 3:
                            break
                if matches:
                    break

        # 后处理：按扩展名优先级排序，并交叉引用 .doc ↔ .pdf/.docx
        matches = self._prioritize_doc_matches(matches)

        return matches[:10]  # 最多返回10个匹配

    def _prioritize_doc_matches(self, matches: list) -> list:
        """对匹配结果按扩展名优先级排序：pdf > docx > doc > xlsx > xls > 其他
        同时，如果匹配到 .doc 文件，检查是否存在同名的 .pdf/.docx 版本并优先返回"""
        import os

        # 扩展名优先级分数（越低越优先）
        ext_priority = {'.pdf': 0, '.docx': 1, '.doc': 2, '.xlsx': 3, '.xls': 4}

        # 收集所有匹配的同base文件（.doc → 找对应的.pdf/.docx）
        expanded = list(matches)
        docs_dir = self.config.kb_config.get("documents_dir", "./documents")

        for m in matches:
            base = os.path.splitext(m)[0]
            ext = os.path.splitext(m)[1].lower()
            if ext == '.doc':
                # 检查是否有同名的 .pdf 或 .docx 版本
                for alt_ext in ['.pdf', '.docx']:
                    alt_name = base + alt_ext
                    if alt_name not in expanded:
                        if os.path.exists(os.path.join(docs_dir, alt_name)):
                            expanded.append(alt_name)

        # 按扩展名优先级排序
        def _sort_key(name: str) -> int:
            ext = os.path.splitext(name)[1].lower()
            return ext_priority.get(ext, 10)

        expanded.sort(key=_sort_key)
        return expanded

    def _resolve_source_name(self, short_name: str) -> str:
        """将LLM提取的文档简称映射为知识库中的实际文件名
        优先返回 .pdf/.docx 版本而非 .doc（因为 KB 可能只索引了 .pdf 版本）
        如果能匹配到，返回实际文件名；否则返回原始名称"""
        matched = self._match_doc_in_kb(short_name)
        if matched:
            # 如果用户引用的是 .doc 文件名，优先匹配的可能是 .doc
            # 但如果存在已索引的 .pdf，应该返回 .pdf
            return matched[0]
        return short_name

    def _verify_doc_exists(self, name: str) -> bool:
        """验证文档名是否在知识库或文档目录中实际存在"""
        import os
        # 快速检查：如果已经有扩展名且存在于documents目录
        docs_dir = self.config.kb_config.get("documents_dir", "./documents")
        if os.path.isdir(docs_dir) and os.path.exists(os.path.join(docs_dir, name)):
            return True

        # 检查元数据索引
        if not hasattr(self, '_doc_index_cache'):
            self._doc_index_cache = self._build_doc_name_index()
        idx = self._doc_index_cache

        # 精确存在于all_names
        if name in idx["all_names"]:
            return True

        # 精确规范匹配
        import re
        def _normalize(n: str) -> str:
            n = os.path.splitext(n)[0]
            n = re.sub(r'^\d+[_\-\s]*', '', n)
            n = re.sub(r'[_\-\s]+', '', n)
            return n.lower().strip()

        norm_name = _normalize(name)
        if norm_name in idx["exact_map"]:
            return True

        # 在docs_text中存在
        kb_dir = self.config.kb_config.get("kb_data_dir", "./kb_data")
        docs_text_dir = os.path.join(kb_dir, "docs_text")
        if os.path.isdir(docs_text_dir):
            for fname in os.listdir(docs_text_dir):
                base = os.path.splitext(fname)[0]
                if base == name or _normalize(base) == norm_name:
                    return True

        # 模糊匹配：至少存在一个候选
        return len(self._match_doc_in_kb(name)) > 0

    def _filter_source_delta(self, delta: str, full_answer_before: str) -> str:
        """过滤流式文本中的来源块。
        检测 full_answer_before + delta 是否进入了 **来源：** 区域，
        如果在来源区域内则返回空字符串(停止流式输出)，
        如果 delta 跨越了来源块的边界则只返回来源块之前的部分。"""
        import re

        combined = full_answer_before + delta

        # 查找 **来源：** 或 来源： 的位置
        source_match = re.search(r'(?:\*\*)?来源[：:](?:\*\*)?', combined)
        if not source_match:
            return delta  # 还没到来源块，正常返回

        source_start = source_match.start()

        # 如果来源块在 full_answer_before 中就已经开始了
        if source_start < len(full_answer_before):
            return ""  # 已经在来源区域，停止输出

        # 来源块在 delta 中开始——只返回来源块之前的部分
        delta_pos = source_start - len(full_answer_before)
        clean_delta = delta[:delta_pos]

        # 也去掉前面的 --- 分隔线
        clean_delta = re.sub(r'\n*---+\s*$', '', clean_delta)

        return clean_delta

    def _strip_source_section(self, answer: str) -> str:
        """从答案文本中剥离末尾的 **来源：** 块（前端来源附加块将替代此功能）
        匹配模式：从最后一个 **来源：** 行开始到文本末尾的内容，
        同时清理前面遗留的分隔线（---）"""
        import re

        if not answer:
            return answer

        # 匹配从 **来源：**（或 来源：）开始到文本末尾的所有内容
        result = re.sub(
            r'\n*\*?\*?来源[：:]\*?\*?\s*\n?[\s\S]*$',
            '',
            answer,
        )

        # 清理末尾遗留的空白分隔线
        result = re.sub(r'\n*---+\s*$', '', result)
        result = re.sub(r'\n{3,}$', '\n\n', result)

        return result.strip()

    def _extract_cited_sources_from_answer(self, answer: str) -> list:
        """从答案文本中提取 Agent 实际引用的文档名
        多策略提取：
        1. 查找 **来源：** 块（如有），从中提取《文档名》
        2. 扫描正文中的 根据/参考/详见/依据《文档名》
        3. 兜底：扫描答案中所有《文档名》格式的文本（排除chunk_id等非文档名）
        返回: 文档名列表（Agent原文中使用的名称）"""
        import re
        cited = []
        seen = set()

        # 策略1: 查找 **来源：** 块
        source_match = re.search(
            r'(?:\*\*)?来源[：:](?:\*\*)?\s*\n?([\s\S]*)$',
            answer,
        )
        if source_match:
            source_text = source_match.group(1)
            for m in re.finditer(r'《([^》]{2,80})》', source_text):
                fn = m.group(1).strip()
                if fn and fn not in seen:
                    seen.add(fn)
                    cited.append(fn)

        # 策略2: 扫描正文中的明确引用模式
        citation_patterns = [
            r'(?:根据|来自|参考|参见|详见|依据|按照|依照|依据于)\s*《([^》]{2,80})》',
            r'(?:如|例如|比如)\s*《([^》]{2,80})》',
            r'(?:在|从|查阅|查看)\s*《([^》]{2,80})》中',
        ]
        if not cited:
            for pattern in citation_patterns:
                for m in re.finditer(pattern, answer):
                    fn = m.group(1).strip()
                    if fn and fn not in seen and len(fn) >= 3 and not fn.startswith('chunk'):
                        seen.add(fn)
                        cited.append(fn)

        # 策略3: 兜底——提取答案中所有《XXX》格式的文本
        # 只要XXX能在知识库中验证到匹配的文档，就认为它是有效的文档引用
        if not cited:
            all_book_refs = re.findall(r'《([^》]{2,80})》', answer)
            for fn in all_book_refs:
                fn = fn.strip()
                if fn and fn not in seen and len(fn) >= 3 and not fn.startswith('chunk'):
                    # 用知识库验证这个名称是否对应真实文档
                    if self._verify_doc_exists(fn):
                        seen.add(fn)
                        cited.append(fn)

        return cited

    def _build_consistent_sources(self, cited_docs: list, tracked_docs: list = None) -> list:
        """构建一致的来源列表：
        1. 仅使用答案文本中实际引用的文档名（cited_docs），不使用工具追踪数据
        2. 每个文档名解析到文档原始完整文件名（用于超链接目标）
        3. 只保留能在KB中验证到的文档
        4. 无引用时返回空列表（不展示来源块）
        返回: [{"name": display_name, "path": original_filename, "url": "/kb/view/original_filename", "chunk_count": 1}, ...]
             其中 name=答案中显示的名称, path=文档原始完整文件名（用于/kb/view/超链接）"""
        import os

        seen_names = set()
        sources = []

        # 仅使用答案文本中引用的文档（来源块只展示文本内容参考的文档）
        primary_docs = cited_docs if cited_docs else []

        for raw_name in primary_docs:
            if not raw_name or len(raw_name.strip()) < 2:
                continue
            raw = raw_name.strip()

            # 解析到文档原始完整文件名
            resolved = self._resolve_source_name(raw)

            # 验证KB中存在
            if not self._verify_doc_exists(resolved):
                continue

            # 去重（以解析后文件名为准）
            if resolved in seen_names:
                continue
            seen_names.add(resolved)

            # 超链接指向原始文档文件（/kb/view/ 现在直接提供原始Word/PDF/Excel文档）
            doc_view_url = f"/kb/view/{resolved}"

            sources.append({
                "name": raw,              # 前端显示用（与答案文本一致）
                "path": resolved,         # 文档原始完整文件名（用于超链接）
                "url": doc_view_url,      # 前端超链接URL → 返回原始文档文件
                "chunk_count": 1,
            })

        return sources

    def _normalize_source_names_in_answer(self, answer: str, sources: list) -> str:
        """将答案文本中的来源引用名称统一为实际KB文件名
        支持单行和多行两种格式，确保流式文本中的文档名与超链接一致"""
        import re

        if not sources or not answer:
            return answer

        # 构建 原始名→解析名 的映射
        name_map = {}
        for s in sources:
            raw = s["name"]
            resolved = s["path"]
            if resolved and resolved != raw:
                name_map[raw] = resolved

        if not name_map:
            return answer

        def replace_ref(match):
            doc_name = match.group(1).strip()
            if doc_name in name_map:
                return f'《{name_map[doc_name]}》'
            return match.group(0)

        # 替换所有《文档名》引用（单行和多行都适用）
        for raw_name, resolved_name in name_map.items():
            # 只替换来源块中的引用
            answer = answer.replace(f'《{raw_name}》', f'《{resolved_name}》')

        return answer

    # 保留旧方法别名以兼容缓存和短时记忆中的旧数据
    def _extract_sources_from_answer(self, answer: str) -> list:
        """[兼容] 从答案中提取来源——新代码请使用 _extract_cited_sources_from_answer + _build_consistent_sources"""
        cited = self._extract_cited_sources_from_answer(answer)
        return self._build_consistent_sources(cited_docs=cited, tracked_docs=[])

    def get_suggestion_questions(self) -> list:
        """获取建议问题列表：优先高分问答对，不足时用热门问题补充"""
        # 优先取高分问答对
        questions = self.evolution_hook.get_top_rated_questions(limit=5)

        # 不足则补充热门问题
        if len(questions) < 5:
            popular = self.evolution_hook.get_popular_questions(limit=5)
            for q in popular:
                if q not in questions:
                    questions.append(q)
                if len(questions) >= 5:
                    break

        # 如果没有足够的问题，用领域默认问题兜底
        if len(questions) < 3:
            defaults = [
                "合格的资本工具有哪些？",
                "反洗钱客户尽职调查的要求是什么？",
                # "资本管理办法与巴塞尔协议III的对应关系是什么？",
                "违反资本管理办法的监管处罚措施有哪些？",
                "人民币图样使用管理办法",
                "中国人民银行行政处罚程序规定",
            ]
            for q in defaults:
                if q not in questions:
                    questions.append(q)
                if len(questions) >= 5:
                    break

        return questions[:5]


def _build_qa_system_prompt() -> str:
    """构建用于LLM生成的系统提示词"""
    return """你是一个专业的金融法规与银行业务知识问答助手。

## 回答规范
- 基于提供的知识库内容全面、完整地回答用户问题
- 对于列举类问题（如"有哪些"），完整列出所有相关项目，不要省略
- 对于定义类问题，引用原文定义并展开说明各要点
- 每段引用知识库内容时必须标注来源chunk_id，格式: [来源:chunk_abc123]
- 数字和日期必须与源文档精确一致
- 不要从多篇文档中综合出新的事实陈述

## 回答结构
1. 先用简洁的语句直接回答核心问题
2. 然后逐条展开详细内容（对于列举类，每条都要列出）
3. 确保每个事实点都有来源标注
4. 答案要完整，不要因为篇幅而省略重要内容"""


def _filter_thinking_text(text: str) -> str:
    """将LLM思考过程中的英文描述转为中文，确保前端展示全中文"""
    import re
    result = text

    # 第一轮: 固定短语替换（按长度降序避免短串先替换导致长串无法匹配）
    replacements = [
        ("The first result is very relevant", "第一个结果非常相关"),
        ("The search results show", "搜索结果显示"),
        ("This is a question about", "这是一个关于"),
        ("Based on the results", "根据搜索结果"),
        ("The user is asking about", "用户在询问"),
        ("This appears to be about", "这似乎是关于"),
        ("The query is about", "这个查询是关于"),
        ("I can see that", "我可以看到"),
        ("The answer should", "答案应该"),
        ("According to", "根据"),
        ("Looking at the", "看一下"),
        ("Let me search", "让我搜索"),
        ("Let me check", "让我检查"),
        ("Let me look", "让我看看"),
        ("I need to", "我需要"),
        ("I should", "我应该"),
        ("I will now", "我现在要"),
        ("I will", "我将"),
        ("I think", "我认为"),
        ("I found", "我找到了"),
        ("Now I", "现在我"),
        ("First,", "首先，"),
        ("Then,", "然后，"),
        ("Next,", "接下来，"),
        ("Finally,", "最后，"),
        ("Therefore,", "因此，"),
        ("However,", "但是，"),
        ("Also,", "另外，"),
        ("In summary,", "总结一下，"),
        ("question about", "关于...的问题"),
        ("financial regulations", "金融法规"),
        ("capital instruments", "资本工具"),
        ("banking", "银行业"),
        ("qualified", "合格"),
        ("categories", "分类"),
        ("specific", "具体"),
        ("details", "详细信息"),
        ("qualification standards", "合格标准"),
        ("capital instruments", "资本工具"),
        ("for more details", "获取更多细节"),
        ("for the", "寻找"),
        ("about the", "关于"),
        ("from the", "从"),
        ("in the", "在"),
        ("of the", "的"),
        ("to the", "到"),
        ("search for", "搜索"),
        ("the ", ""),
        (" a ", ""),
        ("about ", "关于"),
        ("for ", ""),
        ("in ", "在"),
        ("relevant", "相关"),
        ("information", "信息"),
        ("The result", "结果"),
        ("search_knowledge_base", "搜索知识库"),
        ("multi_search_knowledge_base", "多角度搜索"),
        ("list_knowledge_base_sources", "列出文档列表"),
        ("knowledge base", "知识库"),
        ("search query", "搜索查询"),
        ("tool call", "工具调用"),
        ("chunk", "文本块"),
        ("document", "文档"),
        ("results", "结果"),
        ("result", "结果"),
    ]
    for eng, chn in replacements:
        result = result.replace(eng, chn)

    # 第二轮: 移除英文括号注释 "(What are the...)"
    result = re.sub(r'\s*\([A-Z][^)]*\)', '', result)

    # 第三轮: 检测并移除纯英文句子（剩余有连续3个以上英文单词的行）
    lines = result.split('\n')
    filtered_lines = []
    for line in lines:
        # 统计英文单词数
        eng_words = re.findall(r'\b[A-Za-z]{2,}\b', line)
        if len(eng_words) >= 4 and not any('一' <= c <= '鿿' for c in line):
            # 纯英文行，跳过
            continue
        # 单行中有3个以上英文单词且中文很少 → 标记为思考中
        chn_chars = sum(1 for c in line if '一' <= c <= '鿿')
        if len(eng_words) >= 3 and chn_chars < 4:
            continue
        filtered_lines.append(line)
    result = '\n'.join(filtered_lines)

    return result.strip()


def _strip_english_lines(text: str) -> str:
    """彻底清除文本中的英文残留——移除英文行、英文括号、英文短语"""
    import re
    if not text or not text.strip():
        return ""

    lines = text.split('\n')
    filtered = []
    for line in lines:
        eng_words = re.findall(r'\b[A-Za-z]{2,}\b', line)
        chn_chars = sum(1 for c in line if '一' <= c <= '鿿' or '㐀' <= c <= '䶿')
        if len(eng_words) >= 2 and chn_chars == 0:
            continue
        if len(eng_words) >= 3 and chn_chars < 4:
            continue
        filtered.append(line)
    result = '\n'.join(filtered)

    result = re.sub(r'\s*\([A-Za-z][^)]*\)', '', result)
    result = re.sub(r'\b[A-Za-z]{1,2}\b', '', result)
    result = re.sub(r' {2,}', ' ', result)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()
