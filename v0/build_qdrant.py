#!/usr/bin/env python3
"""Compatibility entry point for rebuilding the application's Qdrant index."""

from qa_service import QAService


def main() -> None:
    print("开始重建本地 Qdrant 知识库...")
    service = QAService(config_path="config.yaml")
    result = service.kb_pipeline.build()
    print(
        "重建完成："
        f"文档 {result.get('doc_count', 0)} 份，"
        f"文本块 {result.get('chunk_count', 0)} 条"
    )


if __name__ == "__main__":
    main()
