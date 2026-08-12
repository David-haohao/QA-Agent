import tempfile
import unittest
import gc
import os
from pathlib import Path
from unittest.mock import Mock, patch

from knowledge_base.extractors import DocumentExtractor
from knowledge_base.indexing.bm25_index import BM25Index
from knowledge_base.indexing.qdrant_index import QdrantVectorIndex
from knowledge_base.pipeline import KnowledgeBasePipeline
from models.embedding_client import EmbeddingClient
from qa_agent import tools as qa_tools


class FakeEmbeddingClient:
    dimension = 3

    def embed(self, texts):
        return [self.embed_single(text) for text in texts]

    def embed_single(self, text):
        vectors = {
            "capital adequacy": [1.0, 0.0, 0.0],
            "liquidity management": [0.0, 1.0, 0.0],
            "capital query": [1.0, 0.0, 0.0],
        }
        return vectors[text]


class QdrantVectorIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index = QdrantVectorIndex(
            kb_data_dir=self.temp_dir.name,
            collection_name="qa_knowledge",
            embedding_client=FakeEmbeddingClient(),
            dimension=3,
        )
        self.chunks = [
            {
                "chunk_id": "capital-1",
                "doc_id": "capital-doc",
                "file_name": "capital.pdf",
                "file_path": "documents/capital.pdf",
                "chunk_index": 0,
                "content": "capital adequacy",
            },
            {
                "chunk_id": "liquidity-1",
                "doc_id": "liquidity-doc",
                "file_name": "liquidity.pdf",
                "file_path": "documents/liquidity.pdf",
                "chunk_index": 0,
                "content": "liquidity management",
            },
        ]

    def tearDown(self):
        self.index.close()
        del self.index
        gc.collect()
        self.temp_dir.cleanup()

    def test_rebuild_search_and_delete_by_file_name(self):
        self.assertEqual(2, self.index.rebuild(self.chunks))
        self.assertEqual(2, self.index.get_document_count())

        results = self.index.search("capital query", top_n=2)

        self.assertEqual("capital-1", results[0]["chunk_id"])
        self.assertEqual("capital adequacy", results[0]["content"])
        self.assertEqual("capital.pdf", results[0]["metadata"]["file_name"])
        self.assertGreater(results[0]["score"], results[1]["score"])

        self.index.delete_by_file_name("capital.pdf")

        self.assertEqual(1, self.index.get_document_count())
        self.assertEqual(["liquidity.pdf"], self.index.list_documents())

    def test_reset_removes_leftover_collection_storage_before_rebuild(self):
        self.index.rebuild(self.chunks)
        collection_dir = os.path.join(
            self.temp_dir.name, "qdrant", "collection", "qa_knowledge"
        )
        marker_path = os.path.join(collection_dir, "leftover.marker")
        with open(marker_path, "w", encoding="utf-8") as marker:
            marker.write("stale")

        self.index._reset_local_collection()

        self.assertFalse(self.index.client.collection_exists("qa_knowledge"))
        self.assertFalse(os.path.exists(marker_path))


class KnowledgeBasePipelineConfigTests(unittest.TestCase):
    def test_uses_storage_neutral_vector_collection_setting(self):
        pipeline = KnowledgeBasePipeline(
            config={"vector_collection": "qdrant_collection"},
            embedding_client=FakeEmbeddingClient(),
            reranker_client=object(),
        )

        self.assertEqual("qdrant_collection", pipeline.collection_name)


class KnowledgeBaseToolCompatibilityTests(unittest.TestCase):
    def test_search_tool_uses_vector_store_count_interface(self):
        vector_index = Mock()
        vector_index.get_document_count.return_value = 1
        pipeline = Mock()
        pipeline.get_vector_index.return_value = vector_index
        pipeline.search.return_value = [
            {
                "chunk_id": "capital-1",
                "content": "capital policy content",
                "score": 0.9,
                "metadata": {"file_name": "capital.pdf"},
            }
        ]
        previous_pipeline = qa_tools._kb_pipeline
        try:
            qa_tools.init_kb(pipeline)
            result = qa_tools.search_knowledge_base("capital policy")
        finally:
            qa_tools._kb_pipeline = previous_pipeline

        self.assertIn("capital.pdf", result)
        vector_index.get_document_count.assert_called_once()


class DocumentExtractionReportTests(unittest.TestCase):
    def test_reports_empty_supported_files_without_silently_dropping_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir)
            (docs_dir / "valid.txt").write_text("valid knowledge", encoding="utf-8")
            (docs_dir / "empty.txt").write_text("", encoding="utf-8")
            (docs_dir / "ignored.bin").write_bytes(b"ignored")

            docs, report = DocumentExtractor(str(docs_dir)).extract_all_with_report()

        self.assertEqual(1, len(docs))
        self.assertEqual(2, report["scanned_files"])
        self.assertEqual(1, report["success_files"])
        self.assertEqual(
            [{"file_name": "empty.txt", "reason": "empty_or_unreadable"}],
            report["skipped_files"],
        )


class EmbeddingClientRetryTests(unittest.TestCase):
    @patch("models.embedding_client.requests.post")
    def test_retries_transient_embedding_request_failure(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": [{"embedding": [1.0, 2.0]}]}
        post.side_effect = [ConnectionError("temporary failure"), response]
        client = EmbeddingClient(
            {
                "url": "http://127.0.0.1:4000/v1/embeddings",
                "model": "bge-m3",
                "batch_size": 1,
                "request_retries": 2,
                "retry_delay_seconds": 0,
                "request_timeout_seconds": 600,
            }
        )

        self.assertEqual([[1.0, 2.0]], client.embed(["retry me"]))
        self.assertEqual(2, post.call_count)
        self.assertEqual(600, post.call_args.kwargs["timeout"])


class BM25ReplacementTests(unittest.TestCase):
    def test_replaces_old_chunks_for_changed_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            index = BM25Index(temp_dir)
            index.build_index(
                [
                    {
                        "chunk_id": "old-capital",
                        "content": "old capital policy",
                        "metadata": {"source_file": "capital.pdf"},
                    },
                    {
                        "chunk_id": "liquidity",
                        "content": "liquidity management",
                        "metadata": {"source_file": "liquidity.pdf"},
                    },
                ]
            )

            index.replace_chunks_for_files(
                ["capital.pdf"],
                [
                    {
                        "chunk_id": "new-capital",
                        "content": "new capital policy",
                        "metadata": {"source_file": "capital.pdf"},
                    }
                ],
            )

            self.assertEqual(["liquidity", "new-capital"], sorted(index.chunk_ids))
