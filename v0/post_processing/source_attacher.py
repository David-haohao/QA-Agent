# ============================================================
# 来源附加器 — 后处理Step1
# 解析答案中的chunk_id引用，生成格式化的来源列表
# ============================================================

import re
from typing import List, Dict


class SourceAttacher:
    """
    来源附加器
    解析LLM答案中的 [来源:chunk_id] 标注
    去重同一文档的多个chunk，生成超链接格式
    """

    def attach_sources(self, answer: str, chunk_map: Dict[str, Dict]) -> Dict:
        """
        从答案中解析chunk引用并附加来源信息

        chunk_map: {chunk_id: {content, metadata, source_doc, ...}}
        返回: {"answer_cleaned": "清理后的答案(去来源标记)",
               "sources": [{"name": "...", "path": "...", "chunks": [...]}]}
        """
        # ① 解析答案中的chunk_id引用
        pattern = r"\[来源:\s*(chunk_?\S*?)\]"
        used_chunk_ids = set()
        for m in re.finditer(pattern, answer):
            chunk_id = m.group(1)
            used_chunk_ids.add(chunk_id)

        # ② 清理答案中的标记文本
        cleaned_answer = re.sub(pattern, "", answer).strip()

        if not used_chunk_ids:
            return {"answer_cleaned": cleaned_answer, "sources": []}

        # ③ 通过chunk_map找到源文档信息，去重
        sources = {}
        for cid in used_chunk_ids:
            if cid not in chunk_map:
                continue
            chunk_info = chunk_map[cid]
            src = chunk_info.get("source_doc", {})
            file_name = src.get("file_name", chunk_info.get("file_name", "未知文档"))
            file_path = src.get("file_path", chunk_info.get("file_path", ""))

            if file_name not in sources:
                sources[file_name] = {
                    "name": file_name,
                    "path": file_path,
                    "chunks": [],
                }
            sources[file_name]["chunks"].append({
                "chunk_id": cid,
                "chunk_index": src.get("chunk_index", 0),
            })

        source_list = list(sources.values())
        return {
            "answer_cleaned": cleaned_answer,
            "sources": source_list,
        }
