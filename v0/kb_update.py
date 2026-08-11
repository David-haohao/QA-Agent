# -*- coding: utf-8 -*-
"""
知识库增量更新模块
- 从 OA 系统下载新文档
- 检测新增/变更的文档
- 只对新文档进行增量处理，合并到已有知识库
"""
import os
import json
import datetime
import hashlib
import requests
from urllib.parse import quote


def file_download(fileId: str, fileName: str, dirName: str,
                  oa_url: str, oa_client: dict, download_dir: str) -> str | None:
    """
    从 OA 系统下载单个文件到本地
    参数与 oa_deal 中的 file_download 一致
    """
    try:
        # 1. 获取 token
        token_url = f"{oa_url}/seeyon/rest/token"
        token_resp = requests.post(token_url, json={
            "userName": oa_client["userName"],
            "password": oa_client["password"],
            "loginName": oa_client.get("loginName", ""),
        })
        token_info = token_resp.json()
        p_token = token_info.get("id", "")

        # 2. 下载文件
        encoded_name = quote(fileName)
        url = f"{oa_url}/seeyon/rest/attachment/file/{fileId}?fileName={encoded_name}&token={p_token}"

        response = requests.get(url, stream=True)
        response.raise_for_status()

        cur_dt = datetime.datetime.now().strftime("%Y%m%d")
        save_path = os.path.join(download_dir, cur_dt, dirName)
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, fileName)

        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(save_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        if total_size > 0 and downloaded != total_size:
            raise IOError(f"下载不完整: 期望{total_size}字节, 实际{downloaded}字节")

        print(f"[kb_update] 文件下载成功: {save_file}")
        return save_file

    except Exception as e:
        print(f"[kb_update] 文件下载失败 {fileName}: {e}")
        return None


def file_download_paths(request_data: dict, oa_url: str, oa_client: dict,
                        download_dir: str) -> dict:
    """
    批量下载文档 — 从 OA 获取文档地址后统一下载
    参数与 oa_deal 中的 file_download_paths 一致
    返回: {fileId: local_file_path, ...}
    """
    dataIds = request_data["dataIds"]
    dataType = request_data["dataType"]

    # 获取 token
    token_url = f"{oa_url}/seeyon/rest/token"
    token_resp = requests.post(token_url, json={
        "userName": oa_client["userName"],
        "password": oa_client["password"],
        "loginName": oa_client.get("loginName", ""),
    })
    token_info = token_resp.json()
    p_token = token_info.get("id", "")

    url = f"{oa_url}/seeyon/rest/kk/wzbank/getData?&token={p_token}"

    downloaded_files = {}

    for dataid_ori in dataIds:
        dataid = dataid_ori.get("id", dataid_ori) if isinstance(dataid_ori, dict) else dataid_ori
        payload = json.dumps({"id": dataid, "dataType": dataType})
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, headers=headers, data=payload)
            res = response.json()

            if not res or dataType in ("2", "3"):
                continue  # 新闻/公告不做解析

            if dataType == "1":  # 文档中心
                data = res.get("data", [])
                if not data:
                    continue

                # 收集文件信息
                files = {}
                for row in data:
                    file_id = str(row["fileId"])
                    if "废止" in row.get("fileName", ""):
                        continue
                    if file_id not in files:
                        files[file_id] = {
                            "dataid": str(dataid),
                            "fileId": str(row["fileId"]),
                            "fileName": row["fileName"],
                            "folderName": row.get("folderName", ""),
                        }

                # 统一下载
                for fileId, file_info in files.items():
                    if fileId in downloaded_files:
                        continue
                    save_path = file_download(
                        file_info["fileId"],
                        file_info["fileName"],
                        file_info["folderName"],
                        oa_url, oa_client, download_dir,
                    )
                    if save_path:
                        downloaded_files[fileId] = save_path

        except Exception as e:
            print(f"[kb_update] 处理 dataid={dataid} 失败: {e}")
            continue

    return downloaded_files


def merge_documents_to_kb(downloaded_files: dict, documents_dir: str) -> list:
    """
    将下载的新文件合并到知识库 documents 目录
    - 新文件：直接复制
    - 已存在文件：若内容不同则替换
    返回: 新增或变更的文件名列表
    """
    import shutil

    new_or_changed = []

    for fileId, src_path in downloaded_files.items():
        if not src_path or not os.path.exists(src_path):
            continue

        file_name = os.path.basename(src_path)
        dest_path = os.path.join(documents_dir, file_name)

        # 检查是否是新增或变更
        is_new = not os.path.exists(dest_path)

        if not is_new:
            # 已存在：比较文件内容（用 MD5 快速检查）
            src_hash = _file_md5(src_path)
            dst_hash = _file_md5(dest_path)
            if src_hash == dst_hash:
                print(f"[kb_update] 文件未变化，跳过: {file_name}")
                continue
            is_changed = True
        else:
            is_changed = True

        if is_changed:
            shutil.copy2(src_path, dest_path)
            new_or_changed.append(file_name)
            action = "新增" if is_new else "更新"
            print(f"[kb_update] {action}文档: {file_name}")

    return new_or_changed


def _file_md5(filepath: str) -> str:
    """计算文件 MD5"""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def incremental_build_kb(kb_pipeline, new_files: list) -> dict:
    """
    对新增/变更的文件进行增量知识库构建
    只处理新文件，结果合并到已有索引中
    """
    if not new_files:
        return {"status": "no_changes", "message": "没有需要更新的文档"}

    from knowledge_base.extractors import DocumentExtractor
    from knowledge_base.indexing.text_chunker import DocumentChunker
    from knowledge_base.indexing.vector_index import VectorIndexBuilder
    from knowledge_base.indexing.bm25_index import BM25Index
    from knowledge_base.indexing.document_graph import DocumentGraph

    kb_data_dir = kb_pipeline.kb_data_dir
    documents_dir = kb_pipeline.documents_dir

    print(f"[kb_update] 开始增量构建，新/变更文件数: {len(new_files)}")
    print(f"[kb_update] 文件列表: {new_files}")

    # Phase 1: 逐个提取新文件（不从 extract_all 遍历所有文档）
    extractor = DocumentExtractor(documents_dir)
    new_docs = []
    for fname in new_files:
        file_path = os.path.join(documents_dir, fname)
        if not os.path.exists(file_path):
            print(f"[kb_update] 文件不存在，跳过: {file_path}")
            continue
        fname_lower = fname.lower()
        try:
            if fname_lower.endswith(".pdf"):
                extracted = extractor._extract_pdf(file_path, fname)
            elif fname_lower.endswith(".docx"):
                extracted = extractor._extract_docx(file_path, fname)
            elif fname_lower.endswith(".doc"):
                extracted = extractor._extract_doc(file_path, fname)
            elif fname_lower.endswith((".xlsx", ".xls")):
                extracted = extractor._extract_excel(file_path, fname)
            elif fname_lower.endswith((".txt", ".md", ".csv")):
                extracted = extractor._extract_text_file(file_path, fname)
            else:
                try:
                    extracted = extractor._extract_text_file(file_path, fname)
                except Exception:
                    extracted = None
            if extracted and extracted.get("content"):
                new_docs.append(extracted)
            else:
                print(f"[kb_update] 无法提取内容: {fname}")
        except Exception as e:
            print(f"[kb_update] 提取文档失败 {fname}: {e}")

    print(f"[kb_update] 提取了 {len(new_docs)} 个新文档（共 {len(new_files)} 个文件）")

    if not new_docs:
        return {"status": "no_docs", "message": "未提取到有效文档内容"}

    # 保存新文档全文
    docs_text_dir = os.path.join(kb_data_dir, "docs_text")
    os.makedirs(docs_text_dir, exist_ok=True)
    for doc in new_docs:
        fname = doc.get("file_name", "unknown")
        text_path = os.path.join(docs_text_dir, fname + ".txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(doc.get("content", ""))
    print(f"[kb_update] 新文档全文已保存至 {docs_text_dir}")

    # 文本切片
    chunker = DocumentChunker(
        chunk_size=kb_pipeline.chunk_size,
        chunk_overlap=kb_pipeline.chunk_overlap,
    )
    new_chunks = chunker.chunk_documents(new_docs)
    print(f"[kb_update] 生成 {len(new_chunks)} 个新文本块")

    # 更新 chunk→文档 索引（合并到已有）
    index_path = os.path.join(kb_data_dir, "chunk_doc_index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            old_index = json.load(f)
        old_chunks = old_index.get("chunks", {})
        old_files = old_index.get("files", {})
    else:
        old_chunks = {}
        old_files = {}

    for c in new_chunks:
        cid = c["chunk_id"]
        fname = c.get("file_name", "")
        old_chunks[cid] = {
            "file_name": fname,
            "doc_id": c.get("doc_id", ""),
            "chunk_index": c.get("chunk_index", 0),
        }
        if fname not in old_files:
            old_files[fname] = []
        if cid not in old_files[fname]:
            old_files[fname].append(cid)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "chunks": old_chunks,
            "files": old_files,
            "total_chunks": len(old_chunks),
            "total_files": len(old_files),
        }, f, ensure_ascii=False)
    print(f"[kb_update] chunk→文档索引已更新 ({len(old_chunks)} 条)")

    # Phase 2: 向量索引 — 增量追加新块
    vector_index = VectorIndexBuilder(
        kb_data_dir=kb_data_dir,
        collection_name=kb_pipeline.collection_name,
        embedding_client=kb_pipeline.embedding_client,
        dimension=kb_pipeline.dimension,
    )
    vector_count = vector_index.add_chunks(new_chunks)
    print(f"[kb_update] 向量索引增量追加: {vector_count} 条")

    # BM25 索引 — 需要全量重建（基于 chunk_doc_index 的映射无法还原完整内容）
    # 仅对 BM25 做文本追加（使用原有接口）
    try:
        bm25_index = BM25Index(kb_data_dir)
        bm25_count = bm25_index.add_chunks(new_chunks)
        print(f"[kb_update] BM25索引增量追加: {bm25_count} 条")
    except (AttributeError, Exception):
        print("[kb_update] BM25不支持增量追加，跳过")

    # 文档图谱 — 更新
    doc_graph = DocumentGraph(kb_data_dir)
    doc_graph.build_graph(new_chunks)
    print("[kb_update] 文档图谱已更新（新块追加）")

    # 更新元数据
    metadata_path = os.path.join(kb_data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {"doc_names": [], "total_docs": 0, "total_chunks": 0}

    for doc in new_docs:
        fname = doc.get("file_name", "")
        if fname not in meta.get("doc_names", []):
            meta.setdefault("doc_names", []).append(fname)
    meta["total_docs"] = len(meta.get("doc_names", []))
    meta["total_chunks"] = len(old_chunks)
    meta["last_update"] = datetime.datetime.now().isoformat()

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[kb_update] 增量构建完成! 文档总数: {meta['total_docs']}, 文本块总数: {meta['total_chunks']}")

    return {
        "status": "success",
        "new_files": len(new_docs),
        "new_chunks": len(new_chunks),
        "total_docs": meta["total_docs"],
        "total_chunks": meta["total_chunks"],
    }
