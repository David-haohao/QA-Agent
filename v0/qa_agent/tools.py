# ============================================================
# QA Agent 工具集 — 让LLM通过工具主动搜索知识库
# 支持: 单查询搜索、多角度搜索、列出已索引文档
# ============================================================

# 全局知识库流水线实例
_kb_pipeline = None

# 按会话追踪检索到的文档（供来源附加使用）
# key: session_id, value: set of file_names
_session_retrieved_docs = {}


def init_kb(kb_pipeline):
    """初始化知识库流水线（在创建Agent时调用一次）"""
    global _kb_pipeline
    _kb_pipeline = kb_pipeline


def _get_kb():
    """获取已初始化的知识库流水线"""
    if _kb_pipeline is None:
        raise RuntimeError("知识库尚未初始化，请先调用 init_kb()")
    return _kb_pipeline


def set_tracking_session(session_id: str):
    """设置当前会话ID，用于追踪检索到的文档（在每次查询前调用）"""
    if session_id not in _session_retrieved_docs:
        _session_retrieved_docs[session_id] = set()


def get_and_clear_retrieved_docs(session_id: str) -> list:
    """获取并清除某个会话检索到的文档列表"""
    docs = _session_retrieved_docs.pop(session_id, set())
    return sorted(list(docs))


def _track_retrieved_docs(results: list):
    """从检索结果中提取并追踪文档文件名"""
    for r in results:
        fname = r.get("metadata", {}).get("source_file", r.get("file_name", ""))
        if not fname:
            fname = r.get("file_name", "")
        if fname:
            # 找到当前活跃的会话并追踪
            for sid in list(_session_retrieved_docs.keys()):
                _session_retrieved_docs[sid].add(fname)


def search_knowledge_base(
    query: str,
    top_k: int = 8,
) -> str:
    """在本地中文知识库中搜索相关信息。

    使用混合搜索（语义搜索 + 关键词搜索 + 重排序）找到最相关的文本块。
    对简单问题直接使用此工具。

    参数:
        query: 中文搜索查询，要具体明确。多词查询效果优于单词
        top_k: 返回结果数，默认8，最大20

    返回:
        格式化的搜索结果，包含来源文件、相关分数和文本内容
    """
    if top_k > 20:
        top_k = 20

    pipeline = _get_kb()

    try:
        # 检查知识库是否有内容
        from knowledge_base.indexing.vector_index import VectorIndexBuilder
        vi = VectorIndexBuilder(
            kb_data_dir=pipeline.kb_data_dir,
            collection_name=pipeline.collection_name,
            embedding_client=pipeline.embedding_client,
            dimension=pipeline.dimension,
        )
        if vi.collection.count() == 0:
            # return "知识库尚未构建索引。请先用 python run.py build-kb 构建知识库。"
            return "知识库尚未构建索引。"

        results = pipeline.search(query)
        if not results:
            return "未找到相关文档。建议尝试换用不同的关键词重新搜索。"

        # 追踪检索到的文档（供来源附加使用）
        _track_retrieved_docs(results)

        # 格式化结果
        return _format_search_results(results[:top_k])
    except Exception as e:
        return f"搜索出错: {str(e)}。请尝试使用不同的查询词。"


def multi_search_knowledge_base(
    queries: list[str],
    top_k: int = 5,
) -> str:
    """用多个查询从多角度搜索知识库（多跳检索）。

    用于复杂问题，每个查询独立搜索，结果合并去重。

    参数:
        queries: 查询列表，覆盖问题的不同方面
        top_k: 每个查询返回的结果数，默认5，最大10

    返回:
        合并去重后的搜索结果
    """
    if top_k > 10:
        top_k = 10

    pipeline = _get_kb()

    try:
        all_results = {}
        for query in queries[:5]:  # 最多5个子查询
            results = pipeline.search(query)
            for r in results:
                chunk_id = r.get("chunk_id", "")
                if chunk_id and chunk_id not in all_results:
                    all_results[chunk_id] = r

        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get("rerank_score", x.get("score", 0)),
            reverse=True,
        )

        # 追踪检索到的文档（供来源附加使用）
        _track_retrieved_docs(sorted_results)

        return _format_search_results(sorted_results[:top_k * len(queries)])
    except Exception as e:
        return f"多查询搜索出错: {str(e)}"


def list_knowledge_base_sources() -> str:
    """列出知识库中所有已索引的文档信息。

    用于了解知识库覆盖范围和可用文档，在搜索前确定搜索策略。

    返回:
        文档列表，包含文件名、文档ID
    """
    pipeline = _get_kb()

    try:
        from knowledge_base.indexing.vector_index import VectorIndexBuilder
        vi = VectorIndexBuilder(
            kb_data_dir=pipeline.kb_data_dir,
            collection_name=pipeline.collection_name,
            embedding_client=pipeline.embedding_client,
            dimension=pipeline.dimension,
        )
        doc_list = vi.list_documents()

        if not doc_list:
            return "知识库中暂无已索引的文档。"

        lines = [f"知识库共有 {len(doc_list)} 个已索引文档:\n"]
        for doc_name in doc_list:
            lines.append(f"  - {doc_name}")

        return "\n".join(lines)
    except Exception as e:
        return f"获取文档列表出错: {str(e)}"


def _format_search_results(results: list) -> str:
    """格式化搜索结果为易读的文本，使用文档原始完整文件名"""
    if not results:
        return "未找到相关文档。"

    formatted = []
    for i, r in enumerate(results):
        chunk_id = r.get("chunk_id", f"chunk_{i}")
        content = r.get("content", "")
        score = r.get("rerank_score", r.get("score", 0))
        # 从metadata中获取文档原始完整文件名（source_file优先，file_name兜底）
        meta = r.get("metadata", {})
        file_name = meta.get("source_file", meta.get("file_name", r.get("file_name", "未知文档")))
        chunk_index = meta.get("chunk_index", r.get("chunk_index", 0))

        formatted.append(
            f"--- 结果 {i+1} ---\n"
            f"来源: {file_name} (chunk_id: {chunk_id}, 位置: 第{chunk_index}块)\n"
            f"相关性: {score:.4f}\n"
            f"内容:\n{content}\n"
        )

    return "\n".join(formatted)
