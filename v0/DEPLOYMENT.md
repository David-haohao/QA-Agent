# 堡垒机部署与新文档重建

本项目使用本地嵌入式 Qdrant，数据目录为 `kb_data/qdrant`；不需要 Docker 或独立 Qdrant 服务。

## 导入新文档后重建

1. 使用新资料替换 `documents` 目录内容。旧 Demo 文档、`kb_data/qdrant`、`kb_data/docs_text`、`kb_data/bm25_*`、`kb_data/document_graph.*` 和 `kb_data/ocr_cache` 不应随新文档复用。
2. 安装依赖：使用现有环境执行 `uv pip install --python .venv/Scripts/python.exe -r requirements.txt`。
3. 确认 `config.yaml` 的 `knowledge_base.vector_store` 为 `qdrant`，嵌入服务地址为堡垒机可访问的 `localhost:4000/v1/embeddings`。
4. 先启动本地模型服务：`./.venv/Scripts/python.exe local_embedding_server.py`。
5. 停止 Web 服务，再执行全量重建：`./.venv/Scripts/python.exe build_qdrant.py`。
6. 重建完成后启动 Web：`./.venv/Scripts/python.exe run.py web`。

Windows PowerShell 下将 `/` 改为 `\\` 即可。构建完成后，`kb_data/metadata.json` 的扫描数、成功数、跳过清单和文本块数是验收依据。

## 文件提取能力

- 具备文本层的 PDF：`pdfplumber` / `PyPDF2`。
- 扫描 PDF：本地 `PyMuPDF + RapidOCR`，不上传文件；OCR 缓存由源文件 SHA-256 命名。文件内容变化后会自动生成新缓存，不会使用 Demo 缓存。
- `.doc`：Windows 环境通过本机 Microsoft Word COM 提取（依赖已在 `requirements.txt` 声明 `pywin32`）；堡垒机若非 Windows 或未安装 Word，旧 `.doc` 需要先转换为 `.docx` 或安装兼容转换工具。
- `.docx`、`.xlsx`、`.xls`、`.txt`、`.md`、`.csv`：由内置 Python 提取器处理。

## 运行限制

本地嵌入式 Qdrant 同一时刻只允许一个进程访问 `kb_data/qdrant`。因此构建时必须停止 Web 服务；启动 Web 后不要再单独运行构建或检索脚本。
