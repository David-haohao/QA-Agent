# Windows x64 离线安装包

本目录用于将项目部署到**无外网的 Windows x64 堡垒机**。安装包不需要 `uv`，也不会在安装 Python 依赖时访问 PyPI。

## 交付时必须复制的内容

将整个 `v0` 目录复制到堡垒机，尤其不能漏掉下列两项：

- `download/`：官方 CPython 安装器、135 个锁定版本的 wheels、安装与校验脚本；
- `local_models/`：本地 BGE-M3 嵌入模型和 BGE Reranker v2-M3 重排模型，约 4.41 GB。

`documents/`、`kb_data/` 与 `config.yaml` 是环境数据或配置：按堡垒机实际内容准备，不能把旧演示数据当作生产知识库。

## 安装步骤

在堡垒机 PowerShell 中进入 `v0` 后执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\download\verify_offline.ps1 -SkipImportCheck
.\download\install_offline.ps1
```

脚本默认会将 Python 3.12.10 安装到当前用户目录下的 `QA-Agent-Python31210`，然后创建 `v0\.venv`。如果项目已有 `.venv`，脚本会停止而不会删除或覆盖它；确需重建时请由运维人员确认后手工删除该目录再执行安装。

安装完成后，可再次验证：

```powershell
.\download\verify_offline.ps1
.\.venv\Scripts\python.exe -m pip check
```

## 启动与重建知识库

先按堡垒机的 LLM 地址、密钥和模型名更新 `config.yaml`。本地 BGE 服务使用项目内模型，不需要访问 Hugging Face：

```powershell
.\.venv\Scripts\python.exe .\local_embedding_server.py
```

保持该窗口运行。另开一个 PowerShell 窗口，新的 `documents/` 文件就位后再运行知识库重建，再启动 Web 服务：

```powershell
.\.venv\Scripts\python.exe .\build_qdrant.py
.\.venv\Scripts\python.exe .\run.py web
```

`build_qdrant.py` 会调用本地 `http://127.0.0.1:4000/v1/embeddings`；因此 BGE 服务未启动或端口被占用时，知识库不能重新嵌入。LLM 连通性仍需由堡垒机的 `config.yaml` 单独配置。

## 校验范围

- `SHA256SUMS.txt` 校验 Python 安装器、全部 wheels 与交付脚本；
- 安装脚本使用 `pip install --no-index --find-links`，不会请求包仓库；
- 安装后运行 `pip check`，校验依赖关系；
- 校验脚本在 `.venv` 存在时会导入 FastAPI、Qdrant、FlagEmbedding、PyMuPDF 与 RapidOCR。

本包面向 Intel/AMD 的 64 位 Windows。若堡垒机实际为 ARM64 Windows，需要重新准备对应 Python 与 wheelhouse。
