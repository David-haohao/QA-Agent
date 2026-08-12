"""Rebuild the embedded Qdrant files from vectors already validated by chunk_doc_index."""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

from qdrant_client import QdrantClient, models


def _load_current_chunk_ids(kb_data_dir: str) -> set[str]:
    with open(os.path.join(kb_data_dir, "chunk_doc_index.json"), encoding="utf-8") as file:
        return set(json.load(file)["chunks"])


def _copy_current_points(source_path: str, destination_path: str, collection_name: str, chunk_ids: set[str]) -> int:
    source = QdrantClient(path=source_path)
    destination = QdrantClient(path=destination_path)
    copied = 0
    try:
        source_info = source.get_collection(collection_name)
        vector_size = source_info.config.params.vectors.size
        destination.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        offset = None
        batch = []
        while True:
            points, offset = source.scroll(
                collection_name=collection_name,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                if (point.payload or {}).get("chunk_id") not in chunk_ids:
                    continue
                batch.append(models.PointStruct(id=point.id, vector=point.vector, payload=point.payload))
                if len(batch) == 128:
                    destination.upsert(collection_name=collection_name, points=batch, wait=True)
                    copied += len(batch)
                    batch = []
            if offset is None:
                break
        if batch:
            destination.upsert(collection_name=collection_name, points=batch, wait=True)
            copied += len(batch)
        actual = destination.count(collection_name=collection_name, exact=True).count
        if copied != len(chunk_ids) or actual != len(chunk_ids):
            raise RuntimeError(
                f"Repair count mismatch: copied={copied}, qdrant={actual}, expected={len(chunk_ids)}"
            )
        return copied
    finally:
        source.close()
        destination.close()


def repair(kb_data_dir: str, keep_backup: bool = True) -> dict:
    kb_data_dir = os.path.abspath(kb_data_dir)
    qdrant_path = os.path.join(kb_data_dir, "qdrant")
    with open(os.path.join(kb_data_dir, "metadata.json"), encoding="utf-8") as file:
        metadata = json.load(file)
    collection_name = metadata["vector_collection"]
    chunk_ids = _load_current_chunk_ids(kb_data_dir)
    staging_path = f"{qdrant_path}.repair-staging"
    backup_path = f"{qdrant_path}.backup-{datetime.now():%Y%m%d-%H%M%S}"

    if os.path.exists(staging_path):
        shutil.rmtree(staging_path)
    copied = _copy_current_points(qdrant_path, staging_path, collection_name, chunk_ids)

    os.replace(qdrant_path, backup_path)
    try:
        os.replace(staging_path, qdrant_path)
    except Exception:
        os.replace(backup_path, qdrant_path)
        raise
    if not keep_backup:
        shutil.rmtree(backup_path)
        backup_path = ""
    return {"copied": copied, "expected": len(chunk_ids), "backup_path": backup_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-data-dir", default="./kb_data")
    parser.add_argument("--discard-backup", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(repair(args.kb_data_dir, not args.discard_backup), ensure_ascii=False))
    except Exception as error:
        print(f"Qdrant repair failed: {error}", file=sys.stderr)
        raise
