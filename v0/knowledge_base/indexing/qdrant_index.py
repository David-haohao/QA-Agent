"""Qdrant-backed dense vector index for the local knowledge base."""

import os
import shutil
import threading
import uuid
import weakref
from typing import Dict, List

from qdrant_client import QdrantClient, models


class QdrantVectorIndex:
    """Persist and retrieve dense vectors with a local embedded Qdrant store."""

    _clients = {}
    _instances = {}
    _clients_lock = threading.Lock()

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
        self.qdrant_path = os.path.abspath(os.path.join(kb_data_dir, "qdrant"))
        os.makedirs(self.qdrant_path, exist_ok=True)
        self._client_key = os.path.normcase(self.qdrant_path)
        self._closed = False
        self.client = self._acquire_client()
        self._ensure_collection()

    def _acquire_client(self):
        """Return the one embedded Qdrant client permitted for this directory."""
        with self._clients_lock:
            entry = self._clients.get(self._client_key)
            if entry is None:
                entry = {"client": QdrantClient(path=self.qdrant_path), "references": 0}
                self._clients[self._client_key] = entry
                self._instances[self._client_key] = weakref.WeakSet()
            entry["references"] += 1
            self._instances[self._client_key].add(self)
            return entry["client"]

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
        # Rebuilding is only invoked with the Web service stopped. Keep the
        # shared registry coherent if this pipeline recreates local storage.
        with self._clients_lock:
            entry = self._clients.get(self._client_key)
            if entry is not None:
                entry["client"].close()
            collection_path = os.path.join(
                self.qdrant_path, "collection", self.collection_name
            )
            shutil.rmtree(collection_path, ignore_errors=True)
            replacement = QdrantClient(path=self.qdrant_path)
            self._clients[self._client_key] = {
                "client": replacement,
                "references": entry["references"] if entry is not None else 1,
            }
            for index in self._instances.get(self._client_key, ()):
                index.client = replacement

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
        if self._closed:
            return
        self._closed = True
        with self._clients_lock:
            entry = self._clients.get(self._client_key)
            if entry is None:
                return
            entry["references"] -= 1
            if entry["references"] <= 0:
                entry["client"].close()
                self._clients.pop(self._client_key, None)
                self._instances.pop(self._client_key, None)
