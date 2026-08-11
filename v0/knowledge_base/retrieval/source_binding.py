# ============================================================
# 来源绑定模块 — 过滤无关chunk + 绑定源文档元数据
# 根据架构设计：分数过滤 → LLM相关性判定 → chunk→源文档映射
# ============================================================

import time
from typing import List, Dict


class SourceBinding:
    """
    来源绑定器
    ① 分数过滤：低于阈值的chunk直接丢弃
    ② LLM轻量相关性判定：批量判断chunk是否与问题相关
    ③ chunk_id → 源文档映射：记录文件名、路径、位置
    """

    def __init__(self, llm_client, min_score: float = 0.3):
        """
        llm_client: LLMClient实例(用于相关性判定)
        min_score: 重排序分数最低阈值
        """
        self.llm_client = llm_client
        self.min_score = min_score

    def bind_sources(self, query: str, chunks: List[Dict], timeout_ms: int = 1500) -> List[Dict]:
        """
        对检索结果进行来源绑定
        返回: 通过过滤的chunks + 源文档绑定信息
        """
        if not chunks:
            return []

        # ① 分数过滤
        filtered = [c for c in chunks if c.get("rerank_score", c.get("score", 0)) >= self.min_score]

        if not filtered:
            return []

        # ② LLM轻量相关性判定(批量)
        start_time = time.time()
        relevant = self._llm_relevance_check(query, filtered, timeout_ms)

        # ③ 绑定源文档元数据(已经是每个chunk自带的信息)
        # chunk的metadata中已有source_file, source_path等信息
        for chunk in relevant:
            meta = chunk.get("metadata", {})
            chunk["source_doc"] = {
                "file_name": meta.get("source_file", chunk.get("file_name", "")),
                "file_path": meta.get("source_path", chunk.get("file_path", "")),
                "chunk_index": meta.get("chunk_index", 0),
            }

        return relevant

    def _llm_relevance_check(self, query: str, chunks: List[Dict], timeout_ms: int) -> List[Dict]:
        """
        使用轻量LLM批量判定chunk是否与查询相关
        超时或失败时降级：仅用分数过滤的结果
        """
        try:
            # 构造批量判定prompt
            chunk_texts = []
            for i, c in enumerate(chunks):
                content = c["content"][:200]  # 截取前200字符减少token消耗
                chunk_texts.append(f"[{i}] {content}")

            prompt = f"""判断以下每个文本片段是否与问题相关。只输出一个JSON数组，每个元素为0(无关)或1(相关)。

问题: {query}

文本片段:
{chr(10).join(chunk_texts)}

请只输出JSON数组，例如: [1, 0, 1, 0]"""

            # LLM判定(使用简单模式，不开深度推理)
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_simple(messages)

            # 解析结果
            import json
            try:
                relevance = json.loads(response.strip())
                if isinstance(relevance, list):
                    return [chunks[i] for i, r in enumerate(relevance) if r == 1 and i < len(chunks)]
            except (json.JSONDecodeError, ValueError):
                # JSON解析失败，尝试提取数字
                import re
                nums = re.findall(r"[01]", response)
                if nums:
                    return [chunks[i] for i, n in enumerate(nums) if n == "1" and i < len(chunks)]
        except Exception as e:
            # 任何异常都降级为分数过滤结果
            print(f"LLM相关性判定异常，降级处理: {e}")

        # 降级：返回分数过滤的所有结果
        return chunks

    def get_source_list(self, used_chunks: List[Dict]) -> List[Dict]:
        """
        从实际使用的chunks生成去重的来源列表
        同一文档的多个chunk合并为一条来源记录
        """
        sources = {}
        for chunk in used_chunks:
            src = chunk.get("source_doc", {})
            file_name = src.get("file_name", chunk.get("file_name", ""))
            file_path = src.get("file_path", chunk.get("file_path", ""))

            if file_name not in sources:
                sources[file_name] = {
                    "name": file_name,
                    "path": file_path,
                    "chunks": [],
                }
            sources[file_name]["chunks"].append({
                "chunk_id": chunk["chunk_id"],
                "chunk_index": src.get("chunk_index", 0),
            })

        return list(sources.values())
