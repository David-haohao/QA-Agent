#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# build_qdrant.py —— 独立构建脚本（不修改项目现有代码）
#
# 作用：复用项目既有的「文档提取 + 切片」逻辑，把当前 documents/
#       目录下的全部内容，经 BGE-M3 向量化后写入 Qdrant。
#
# 向量库形态（满足：无网络可用 + 免费 + 本地部署）：
#   - 默认使用 Qdrant 本地持久化模式（QdrantClient(path=...)），
#     数据落在 kb_data/qdrant/，无需任何外部服务进程，完全离线。
#   - 如需改为独立服务端（client-server），设置环境变量
#     QDRANT_URL=http://localhost:6333 后运行本脚本即可。
#
# Embedding 后端：
#   - 默认【本地直接向量化】：加载项目内的 local_models/bge-m3，
#     与 localhost:4000 服务同模型同权重、完全离线。
#   - 如需复用 HTTP 服务，设 USE_HTTP_EMBED=1 运行。
#
# 本次增强（相比第一版）：
#   1) 提取阶段用多进程并发，单文件超 120s 直接跳过（避免坏/超大
#      PDF 卡死整个任务）。
#   2) 实时进度日志（每 50 个文件、每批写入均打印计数与耗时）。
#   3) 集合已存在则沿用，不重建。
#
# 复用的模块（只读，不改动）：
#   config_loader.load_config
#   models.embedding_client.EmbeddingClient   (仅 USE_HTTP_EMBED=1 时)
#   knowledge_base.extractors.DocumentExtractor
#   knowledge_base.indexing.text_chunker.DocumentChunker
#   qdrant_client.QdrantClient
#
# 用法：
#   python build_qdrant.py
#   USE_HTTP_EMBED=1 python build_qdrant.py
#   QDRANT_URL=http://localhost:6333 python build_qdrant.py
# ============================================================
import os
import sys
import uuid
import time
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 确保模型推理完全离线（不联网拉权重）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Windows 控制台输出统一 UTF-8，便于日志查看
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


EXTRACT_WORKERS = 4          # 并发提取进程数
PER_FILE_TIMEOUT = 120       # 单文件提取超时（秒），超时即跳过


def extract_one(file_path: str):
    """在独立子进程里提取单个文件内容（可被 pickle，超时由父进程控制）"""
    try:
        from knowledge_base.extractors import DocumentExtractor
    except Exception:
        return None
    ex = DocumentExtractor(os.path.dirname(file_path))
    name = os.path.basename(file_path)
    low = file_path.lower()
    try:
        if low.endswith(".pdf"):
            return ex._extract_pdf(file_path, name)
        if low.endswith(".docx"):
            return ex._extract_docx(file_path, name)
        if low.endswith((".xlsx", ".xls")):
            return ex._extract_excel(file_path, name)
        if low.endswith(".doc"):
            return ex._extract_doc(file_path, name)
        if low.endswith((".txt", ".md", ".csv")):
            return ex._extract_text_file(file_path, name)
    except Exception:
        return None
    return None


def list_files(docs_dir: str):
    exts = (".pdf", ".docx", ".xls", ".xlsx", ".doc", ".txt", ".md", ".csv")
    out = []
    for root, _, files in os.walk(docs_dir):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(root, f))
    return out


def robust_extract(files, timeout=PER_FILE_TIMEOUT, workers=EXTRACT_WORKERS):
    """多进程提取，单文件超时即跳过（terminate 整个 pool 并重建，避免卡死）"""
    results = []
    skipped = []
    skip_paths = set()
    done_paths = set()
    remaining = list(files)
    while remaining:
        pool = multiprocessing.Pool(processes=workers)
        batch = [
            (f, pool.apply_async(extract_one, (f,)))
            for f in remaining if f not in done_paths and f not in skip_paths
        ]
        try:
            for f, a in batch:
                try:
                    r = a.get(timeout=timeout)
                    done_paths.add(f)
                    if r:
                        results.append(r)
                    else:
                        skipped.append((f, "empty/none"))
                except multiprocessing.TimeoutError:
                    skip_paths.add(f)
                    skipped.append((f, "TIMEOUT"))
                except Exception as e:
                    done_paths.add(f)
                    skipped.append((f, "ERR:" + str(e)[:60]))
        finally:
            pool.terminate()
            pool.join()
        done = len(done_paths) + len(skip_paths)
        print(f"  [提取] {done}/{len(files)}  已用 {int(time.time() - t0)}s  跳过 {len(skipped)}",
              flush=True)
        remaining = [f for f in files if f not in done_paths and f not in skip_paths]
        if not remaining:
            break
    return results, skipped


def main():
    global t0
    multiprocessing.freeze_support()
    t0 = time.time()

    from config_loader import load_config
    from knowledge_base.extractors import DocumentExtractor
    from knowledge_base.indexing.text_chunker import DocumentChunker
    from qdrant_client import QdrantClient, models

    cfg = load_config("config.yaml")
    emb_cfg = cfg.embeddings_config
    kb = cfg.kb_config

    docs_dir = kb.get("documents_dir", "./documents")
    kb_data_dir = kb.get("kb_data_dir", "./kb_data")
    collection = kb.get("chroma_collection", "qa_knowledge")
    dim = emb_cfg.get("dimension", 1024)
    chunk_size = kb.get("chunk_size", 500)
    chunk_overlap = kb.get("chunk_overlap", 50)

    # ---- Qdrant 客户端：本地模式（离线）或服务端模式 ----
    url = os.environ.get("QDRANT_URL")
    if url:
        print(f"[Qdrant] 连接服务端: {url}", flush=True)
        client = QdrantClient(url=url)
    else:
        qdrant_path = os.path.join(kb_data_dir, "qdrant")
        os.makedirs(qdrant_path, exist_ok=True)
        print(f"[Qdrant] 本地持久化模式: {qdrant_path}", flush=True)
        client = QdrantClient(path=qdrant_path)

    try:
        client.get_collection(collection)
        print(f"[Qdrant] 集合已存在，沿用: {collection}", flush=True)
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        print(f"[Qdrant] 集合已就绪: {collection} (dim={dim}, cosine)", flush=True)

    # ---- Embedding 后端选择 ----
    use_http = os.environ.get("USE_HTTP_EMBED", "0") == "1"
    if use_http:
        from models.embedding_client import EmbeddingClient
        emb = EmbeddingClient(emb_cfg)

        def embed_texts(texts):
            return emb.embed(texts)
        print(f"[Embedding] 使用 HTTP 服务: {emb_cfg.get('url')}", flush=True)
    else:
        from FlagEmbedding import BGEM3FlagModel
        device = os.environ.get("EMBED_DEVICE", "cpu")
        local_model = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "local_models", "bge-m3"
        )
        print(f"[Embedding] 加载本地模型: {local_model} device={device}", flush=True)
        _m = BGEM3FlagModel(local_model, use_fp16=False, device=device)
        print("[Embedding] 本地模型加载完成", flush=True)

        def embed_texts(texts):
            truncated = [t[:512] for t in texts]  # 与客户端一致：截断至 512 字符
            out = _m.encode(truncated, return_dense=True,
                            return_sparse=False, return_colbert_vecs=False)
            return [v.tolist() for v in out["dense_vecs"]]

    # ---- Phase 1: 并发提取（单文件超时跳过） + 切片 ----
    files = list_files(docs_dir)
    print(f"[扫描] 待处理文件数: {len(files)}", flush=True)
    docs, skipped = robust_extract(files, timeout=PER_FILE_TIMEOUT, workers=EXTRACT_WORKERS)
    print(f"[提取] 成功文档数: {len(docs)}  跳过: {len(skipped)}  ({int(time.time() - t0)}s)",
          flush=True)
    if skipped:
        print(f"[跳过文件样例] {skipped[:10]}", flush=True)
    if not docs:
        print("未找到可处理的文档!", flush=True)
        return

    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(docs)
    print(f"[切片] 文本块数: {len(chunks)}  ({int(time.time() - t0)}s)", flush=True)

    # chunk_id -> 稳定 UUID
    def pid(cid: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, cid))

    # ---- 断点续传：收集已存在的 chunk_id（幂等跳过，避免重复 embedding） ----
    def load_existing_chunk_ids(client, collection):
        existing = set()
        offset = None
        while True:
            pts, offset = client.scroll(
                collection_name=collection, limit=500,
                offset=offset, with_payload=True, with_vectors=False,
            )
            for p in pts:
                cid = (p.payload or {}).get("chunk_id")
                if cid:
                    existing.add(cid)
            if offset is None:
                break
        return existing

    existing = load_existing_chunk_ids(client, collection)
    print(f"[续传] 集合中已有向量数: {len(existing)}", flush=True)
    chunks_todo = [c for c in chunks if c["chunk_id"] not in existing]
    print(f"[续传] 待写入(新增)块数: {len(chunks_todo)}", flush=True)

    # ---- Phase 2: 向量化并写入 Qdrant ----
    batch = 200
    total = 0
    for i in range(0, len(chunks_todo), batch):
        grp = chunks_todo[i:i + batch]
        texts = [c["content"] for c in grp]
        vectors = embed_texts(texts)
        points = []
        for c, vec in zip(grp, vectors):
            points.append(models.PointStruct(
                id=pid(c["chunk_id"]),
                vector=vec,
                payload={
                    "chunk_id": c["chunk_id"],
                    "doc_id": c.get("doc_id", ""),
                    "file_name": c.get("file_name", ""),
                    "file_path": c.get("file_path", ""),
                    "chunk_index": c.get("chunk_index", 0),
                    "text": c["content"],
                },
            ))
        client.upsert(collection_name=collection, points=points)
        total += len(grp)
        print(f"  [写入] {total}/{len(chunks)}  ({int(time.time() - t0)}s)", flush=True)

    cnt = client.count(collection).count
    print("=" * 60)
    print(f"完成！集合={collection}, Qdrant 向量数={cnt}, 跳过文件={len(skipped)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
