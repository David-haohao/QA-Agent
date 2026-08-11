# ============================================================
# 可自我进化的问答智能体系统 — Docker 镜像
# 构建时需要将 context 设为父目录（可进化的智能体架构图-v2/），
# 以便同时访问 v0/ 和 deepagents/ 两个目录。
#
# 构建命令:
#   cd 可进化的智能体架构图-v2
#   docker build -f v0/Dockerfile -t qa-agent:latest .
# ============================================================

FROM docker.m.daocloud.io/library/python:3.12-slim

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 安装运行时系统依赖（仅 curl 用于健康检查，其余全用预编译 wheel）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ----------------- Python 依赖 -----------------
COPY v0/requirements.txt .

RUN pip install --no-cache-dir \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    --prefer-binary \
    -r requirements.txt

# ----------------- 安装 deepagents（本地依赖）-----------------
COPY deepagents/libs/deepagents /tmp/deepagents
RUN pip install --no-cache-dir \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    --prefer-binary \
    /tmp/deepagents \
    && rm -rf /tmp/deepagents

# ----------------- 复制应用代码 -----------------
COPY v0/ .

# 运行时目录
RUN mkdir -p /app/kb_data /app/evolution_data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

ENTRYPOINT ["python", "run.py", "web"]
