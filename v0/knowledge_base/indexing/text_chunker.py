# ============================================================
# 文本切片器 — 自适应切分文档内容
# ============================================================

import re
from typing import List, Dict


class DocumentChunker:
    """
    自适应文本切片器
    将长文档切分为适合检索的语义片段，保留上下文重叠
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        chunk_size: 每个切片的字符数
        chunk_overlap: 相邻切片重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        批量切分文档，返回所有chunk
        每个chunk包含: chunk_id, doc_id, file_name, content, chunk_index, metadata
        """
        all_chunks = []
        for doc in docs:
            doc_chunks = self._chunk_single(doc)
            all_chunks.extend(doc_chunks)
        return all_chunks

    def _chunk_single(self, doc: Dict) -> List[Dict]:
        """
        对单个文档进行自适应切分
        策略:
        1. 先按自然段落切分
        2. 短段落合并
        3. 长段落按句子切分
        4. 保留重叠上下文
        """
        content = doc.get("content", "")
        file_name = doc.get("file_name", "")
        doc_id = doc.get("doc_id", "")

        chunks = []
        paragraphs = self._get_paragraphs(content)

        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n"
            else:
                # 当前chunk已满，保存并开始新chunk
                if current_chunk.strip():
                    chunks.append({
                        "chunk_id": f"{doc_id}_c{chunk_index}",
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "file_path": doc.get("file_path", ""),
                        "content": current_chunk.strip(),
                        "chunk_index": chunk_index,
                        "metadata": {
                            "source_file": file_name,
                            "source_path": doc.get("file_path", ""),
                            "chunk_index": chunk_index,
                        },
                    })
                    chunk_index += 1

                # 处理超长段落：按句子切分
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_long_paragraph(para, doc_id, file_name, doc, chunk_index)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    current_chunk = ""
                else:
                    # 重叠保留上一段的末尾
                    overlap_text = current_chunk[-self.chunk_overlap:] if current_chunk else ""
                    current_chunk = overlap_text + para + "\n"

        # 保存最后一个chunk
        if current_chunk.strip():
            chunks.append({
                "chunk_id": f"{doc_id}_c{chunk_index}",
                "doc_id": doc_id,
                "file_name": file_name,
                "file_path": doc.get("file_path", ""),
                "content": current_chunk.strip(),
                "chunk_index": chunk_index,
                "metadata": {
                    "source_file": file_name,
                    "source_path": doc.get("file_path", ""),
                    "chunk_index": chunk_index,
                },
            })

        return chunks

    def _get_paragraphs(self, text: str) -> List[str]:
        """按换行符和空行分割段落"""
        # 将文本按双换行或单换行切分
        paragraphs = re.split(r"\n\s*\n", text)
        result = []
        for p in paragraphs:
            p = p.strip()
            if p:
                # 再按单换行细分
                lines = p.split("\n")
                if len(lines) > 1 and len(p) > self.chunk_size * 0.5:
                    result.extend([l.strip() for l in lines if l.strip()])
                else:
                    result.append(p)
        return result

    def _split_long_paragraph(self, para: str, doc_id: str, file_name: str, doc: Dict, start_idx: int) -> List[Dict]:
        """将超长段落按句子切分为多个chunk"""
        sentences = re.split(r"([。！？；\n])", para)
        chunks = []
        current = ""
        idx = start_idx

        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            sep = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sent = sent + sep

            if len(current) + len(full_sent) <= self.chunk_size:
                current += full_sent
            else:
                if current.strip():
                    chunks.append({
                        "chunk_id": f"{doc_id}_c{idx}",
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "file_path": doc.get("file_path", ""),
                        "content": current.strip(),
                        "chunk_index": idx,
                        "metadata": {
                            "source_file": file_name,
                            "source_path": doc.get("file_path", ""),
                            "chunk_index": idx,
                        },
                    })
                    idx += 1
                # 重叠保留
                overlap = current[-self.chunk_overlap:] if current else ""
                current = overlap + full_sent

        if current.strip():
            chunks.append({
                "chunk_id": f"{doc_id}_c{idx}",
                "doc_id": doc_id,
                "file_name": file_name,
                "file_path": doc.get("file_path", ""),
                "content": current.strip(),
                "chunk_index": idx,
                "metadata": {
                    "source_file": file_name,
                    "source_path": doc.get("file_path", ""),
                    "chunk_index": idx,
                },
            })

        return chunks
