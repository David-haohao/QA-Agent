"""Qdrant-backed dense vector index for the local knowledge base."""

import os
import shutil
import uuid
from typing import Dict, List

from qdrant_client import QdrantClient, models


class QdrantVectorIndex:
    """Persist and retrieve dense vectors with a local embedded Qdrant store."""

    def __init__(
        self,
        kb_data_dir: str,
        collection_name: str,
        embedding_client,
        dimension: int = 1024,
    ):
        self.kb_data_dir = kb_data_dir
        self.collection_name = collection_name
        self.embedding_client = embedding_client
        self.dimension = dimension
        self.qdrant_path = os.path.join(kb_data_dir, "qdrant")
        os.makedirs(self.qdrant_path, exist_ok=True)
        self.client = QdrantClient(path=self.qdrant_path)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def rebuild(self, chunks: List[Dict]) -> int:
        """Replace the collection and embed every supplied chunk."""
        self._reset_local_collection()
        self._ensure_collection()
        return self.add_chunks(chunks)

    def _reset_local_collection(self) -> None:
        """Close and remove the embedded collection files before a full rebuild."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.close()
        collection_path = os.path.join(
            self.qdrant_path, "collection", self.collection_name
        )
        shutil.rmtree(collection_path, ignore_errors=True)
        self.client = QdrantClient(path=self.qdrant_path)

    def build_index(self, chunks: List[Dict]) -> int:
        """Build a new collection from chunks; retained for pipeline compatibility."""
        return self.rebuild(chunks)

    def add_chunks(self, chunks: List[Dict]) -> int:
        if not chunks:
            return 0

        total = 0
        upsert_batch_size = 128
        for start in range(0, len(chunks), upsert_batch_size):
            batch = chunks[start:start + upsert_batch_size]
            vectors = self.embedding_client.embed([chunk.get("content", "") for chunk in batch])
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(batch)}, got {len(vectors)}"
                )
            points = [self._to_point(chunk, vector) for chunk, vector in zip(batch, vectors)]
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            total += len(points)
            print(f"  [Qdrant] 已写入向量: {total}/{len(chunks)}")
        return total

    def _to_point(self, chunk: Dict, vector: List[float]) -> models.PointStruct:
        chunk_id = chunk["chunk_id"]
        return models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
            vector=vector,
            payload={
                "chunk_id": chunk_id,
                "doc_id": chunk.get("doc_id", ""),
                "file_name": chunk.get("file_name", ""),
                "file_path": chunk.get("file_path", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "content": chunk.get("content", ""),
            },
        )

    def search(self, query_text: str, top_n: int = 20) -> List[Dict]:
        count = self.get_document_count()
        if count == 0:
            return []

        query_vector = self.embedding_client.embed_single(query_text)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=min(top_n, count),
            with_payload=True,
        )

        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "chunk_id": payload.get("chunk_id", str(point.id)),
                    "content": payload.get("content", ""),
                    "score": round(float(point.score), 4),
                    "metadata": {
                        "doc_id": payload.get("doc_id", ""),
                        "file_name": payload.get("file_name", ""),
                        "file_path": payload.get("file_path", ""),
                        "chunk_index": payload.get("chunk_index", 0),
                    },
                }
            )
        return results

    def delete_by_file_name(self, file_name: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_name",
                            match=models.MatchValue(value=file_name),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def get_document_count(self) -> int:
        return self.client.count(collection_name=self.collection_name, exact=True).count

    def list_documents(self) -> List[str]:
        file_names = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=["file_name"],
                with_vectors=False,
            )
            for point in points:
                file_name = (point.payload or {}).get("file_name")
                if file_name:
                    file_names.add(file_name)
            if offset is None:
                break
        return sorted(file_names)

    def close(self) -> None:
        self.client.close()
