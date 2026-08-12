#!/usr/bin/env python3
"""Full rebuild entry point for the application's embedded Qdrant knowledge base.

Run only after the local embedding service is healthy and the Web service has
stopped, because embedded Qdrant permits one process to own its storage path.
"""

from qa_service import QAService


def main() -> None:
    print("开始重建本地 Qdrant 知识库...")
    print("提示：会按当前 documents 目录全量提取并重建，不会复用旧向量。")
    service = QAService(config_path="config.yaml")
    result = service.kb_pipeline.build()
    print(
        "重建完成："
        f"文档 {result.get('doc_count', 0)} 份，"
        f"文本块 {result.get('chunk_count', 0)} 条"
    )
    vector_index = service.kb_pipeline.get_vector_index()
    vector_index.close()


if __name__ == "__main__":
    main()
