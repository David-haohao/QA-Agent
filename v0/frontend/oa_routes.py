# -*- coding: utf-8 -*-
"""
OA 知识库上传路由 — /oa/getinfo
接收 OA 系统推送的文档更新通知，自动下载并增量更新知识库
"""
import os
import json
import datetime
import threading
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal, List


class DataIdItem(BaseModel):
    id: str


class FileItem(BaseModel):
    """OA getinfo 接口入参"""
    requestId: str
    operType: Literal["1", "2", "3"]  # 1:新增 2:删除 3:变更
    dataType: Literal["1", "2", "3", "4"]  # 1:文档中心 2:新闻 3:公告 4:文件
    dataIds: list  # [{"id": "xxx"}, ...]


router = APIRouter()


def create_oa_routes(qa_service):
    """创建 OA 知识库上传路由"""

    @router.post("/oa/getinfo")
    def oa_getinfo(item: FileItem):
        """
        接收 OA 文档更新通知
        自动下载新文档并增量更新知识库
        """
        response = {"code": "1", "message": "Success"}
        try:
            requestId = item.requestId
            operType = item.operType
            dataType = item.dataType
            dataIds = item.dataIds

            request_data = {
                "requestId": requestId,
                "operType": operType,
                "dataType": dataType,
                "dataIds": dataIds,
                "requestTime": datetime.datetime.now().isoformat(),
                "status": "0",
            }

            if operType == "2":
                # 删除操作 — 当前仅记录，不删除索引
                print(f"[oa_getinfo] 收到删除请求 requestId={requestId}")
                return response

            # 异步执行下载和 KB 更新（避免阻塞接口响应）
            thread = threading.Thread(
                target=_process_oa_update,
                args=(request_data, qa_service),
                daemon=True,
            )
            thread.start()

            return response

        except Exception as e:
            response["code"] = "0"
            response["message"] = str(e)
            return response

    return router


def _process_oa_update(request_data: dict, qa_service) -> None:
    """后台处理 OA 文档更新"""
    import sys
    import os
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root_dir)

    from kb_update import file_download_paths, merge_documents_to_kb, incremental_build_kb

    try:
        config = qa_service.config
        oa_config = config.kb_config.get("oa", {})
        oa_url = oa_config.get("oa_url", "http://localhost:8888")
        oa_client = oa_config.get("oa_client", {
            "userName": "test",
            "password": "",
            "loginName": "10455",
        })
        download_dir = oa_config.get("oa_file_dir", os.path.join(root_dir, "oa_downloads"))
        documents_dir = config.kb_config.get("documents_dir", os.path.join(root_dir, "documents"))

        os.makedirs(download_dir, exist_ok=True)

        print(f"[kb_update] 开始处理 OA 更新: requestId={request_data['requestId']}")
        print(f"[kb_update] oa_url={oa_url}, documents_dir={documents_dir}")

        # Step 1: 从 OA 下载文件
        downloaded = file_download_paths(request_data, oa_url, oa_client, download_dir)
        print(f"[kb_update] 下载完成: {len(downloaded)} 个文件")

        if not downloaded:
            print("[kb_update] 没有文件需要处理")
            return

        # Step 2: 合并到知识库 documents 目录
        new_files = merge_documents_to_kb(downloaded, documents_dir)

        # Step 3: 增量构建知识库
        kb_pipeline = qa_service.kb_pipeline
        result = incremental_build_kb(kb_pipeline, new_files)
        print(f"[kb_update] 增量构建结果: {result}")

    except Exception as e:
        import traceback
        print(f"[kb_update] 处理失败: {e}")
        traceback.print_exc()
