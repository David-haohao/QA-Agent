#!/bin/bash
# ============================================================
# Docker 网络代理 — 将宿主机可达的内部服务端口转发给容器
#
# 问题背景:
#   Docker 容器使用独立网桥，无法直接到达宿主机 VPN 隧道后
#   的 172.18.1.22。本脚本在宿主机上监听端口并转发流量。
#
# 使用方式:
#   chmod +x docker-proxy.sh
#   ./docker-proxy.sh start    启动转发
#   ./docker-proxy.sh stop     停止转发
#   ./docker-proxy.sh status   查看状态
# ============================================================

TARGET_HOST="172.18.1.22"
TARGET_PORT="8019"
LISTEN_PORT="18019"
PID_FILE="/tmp/docker-proxy-embedding.pid"
LOG_FILE="/tmp/docker-proxy-embedding.log"

start() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "[已运行] 端口转发已在运行中: 0.0.0.0:$LISTEN_PORT -> $TARGET_HOST:$TARGET_PORT"
        return
    fi

    echo "[启动] 端口转发: 0.0.0.0:$LISTEN_PORT -> $TARGET_HOST:$TARGET_PORT"
    nohup socat TCP-LISTEN:$LISTEN_PORT,bind=0.0.0.0,fork,reuseaddr \
        TCP:$TARGET_HOST:$TARGET_PORT \
        > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 1

    if kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "[成功] 转发已启动 (PID: $(cat $PID_FILE))"
    else
        echo "[失败] 转发启动失败，请检查是否安装 socat (brew install socat)"
        rm -f "$PID_FILE"
        exit 1
    fi
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill "$PID" 2>/dev/null; then
            echo "[停止] 转发已停止 (PID: $PID)"
        fi
        rm -f "$PID_FILE"
    else
        echo "[已停止] 未找到运行中的转发进程"
    fi
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "[运行中] 0.0.0.0:$LISTEN_PORT -> $TARGET_HOST:$TARGET_PORT (PID: $(cat $PID_FILE))"
    else
        echo "[未运行] 转发未启动"
    fi
}

case "${1:-start}" in
    start)  start ;;
    stop)   stop ;;
    status) status ;;
    *)      echo "用法: $0 {start|stop|status}" ;;
esac
