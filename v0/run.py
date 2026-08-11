#!/usr/bin/env python3
# ============================================================
# 可自我进化的问答智能体系统 — 主启动文件
#
# 用法:
#   python run.py web       启动Web问答服务 (http://localhost:8000)
#   python run.py build-kb  构建知识库(解析PDF+建立索引)
#   python run.py search "查询文本"  命令行检索测试
#   python run.py stats     查看知识库和缓存统计
#
# 三个独立部分可各自单独部署:
#   1. 前端交互 (frontend/) — 可独立运行Web UI服务
#   2. QA Agent (qa_agent/) — 可嵌入其他Python应用调用
#   3. 知识库 (knowledge_base/) — 可独立部署为检索服务
# ============================================================

import sys
import os
import argparse

# 确保当前目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_web(args):
    """启动Web问答服务"""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from frontend.routes import create_frontend_routes
    from frontend.oa_routes import create_oa_routes
    from qa_service import QAService

    print("=" * 60)
    print("  启动智能问答Web服务...")
    print("=" * 60)

    # 初始化QA服务
    config_path = args.config or "config.yaml"
    print(f"[1/3] 加载配置: {config_path}")
    qa_service = QAService(config_path=config_path)

    # 创建FastAPI应用
    print("[2/3] 初始化API路由...")
    app = FastAPI(
        title="智能问答系统",
        description="基于本地知识库的金融法规与银行业务智能问答系统",
        version="1.0.0",
    )

    # 注册前端路由
    router = create_frontend_routes(qa_service)
    app.include_router(router)

    # 注册 OA 知识库上传路由
    oa_router = create_oa_routes(qa_service)
    app.include_router(oa_router)

    # 挂载静态文件目录(CSS/JS) — 使用绝对路径
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # 启动服务
    host = qa_service.config.server_config.get("host", "0.0.0.0")
    port = qa_service.config.server_config.get("port", 8000)
    print(f"[3/3] 启动服务: http://{host}:{port}")
    print("=" * 60)
    print(f"  请在浏览器中打开: http://localhost:{port}")
    print("  按 Ctrl+C 停止服务")
    print("=" * 60)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=qa_service.config.server_config.get("log_level", "info"),
    )


def cmd_build_kb(args):
    """构建知识库"""
    from qa_service import QAService

    print("=" * 60)
    print("  开始构建知识库...")
    print("=" * 60)

    config_path = args.config or "config.yaml"
    qa_service = QAService(config_path=config_path)

    result = qa_service.kb_pipeline.build()

    print("\n构建结果:")
    print(f"  文档数量: {result.get('doc_count', 0)}")
    print(f"  文本块数量: {result.get('chunk_count', 0)}")
    print("知识库构建完成!")


def cmd_search(args):
    """命令行检索测试"""
    from qa_service import QAService

    config_path = args.config or "config.yaml"
    qa_service = QAService(config_path=config_path)

    query = args.query
    print(f"查询: {query}")
    print("-" * 40)

    # 执行检索
    results = qa_service.kb_pipeline.search(query)
    if not results:
        print("未找到相关结果")
    else:
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r.get('chunk_id', '')}")
            meta = r.get('metadata', {})
            file_name = meta.get('source_file', meta.get('file_name', r.get('file_name', '未知文档')))
            print(f"  文档: {file_name}")
            print(f"  分数: {r.get('rerank_score', r.get('score', 0)):.4f}")
            content = r.get('content', '')[:300]
            print(f"  内容: {content}...")


def cmd_stats(args):
    """查看知识库和缓存统计"""
    from qa_service import QAService

    config_path = args.config or "config.yaml"
    qa_service = QAService(config_path=config_path)

    print("=" * 40)
    print("  系统状态")
    print("=" * 40)

    # 缓存统计
    cache_stats = qa_service.get_cache_stats()
    print("\n[缓存统计]")
    print(f"  L1缓存大小: {cache_stats.get('l1_size', 0)}")
    print(f"  L2缓存大小: {cache_stats.get('l2_size', 0)}")
    print(f"  L1命中: {cache_stats.get('l1_hits', 0)}")
    print(f"  L2命中: {cache_stats.get('l2_hits', 0)}")
    print(f"  未命中: {cache_stats.get('misses', 0)}")
    print(f"  总查询: {cache_stats.get('total', 0)}")
    hit_rate = cache_stats.get('hit_rate', 0)
    print(f"  命中率: {hit_rate:.2%}")

    # 知识库概览
    print("\n[知识库概览]")
    kb_info = qa_service.get_kb_overview()
    print(f"  覆盖领域: {', '.join(kb_info.get('domains', []))}")
    print(f"  已索引文档: {kb_info.get('doc_count', 0)}")
    docs = kb_info.get('documents', [])[:5]
    if docs:
        print("  文档列表(前5个):")
        for d in docs:
            print(f"    - {d}")


def main():
    parser = argparse.ArgumentParser(
        description="可自我进化的问答智能体系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run.py web                                  # 启动Web问答服务
  python run.py web --config config.yaml             # 指定配置文件
  python run.py build-kb                             # 构建知识库索引
  python run.py search "资本管理办法的适用范围"         # 命令行检索
  python run.py stats                                # 查看系统状态
        """,
    )
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="配置文件路径")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # web命令
    web_parser = subparsers.add_parser("web", help="启动Web问答服务 (http://localhost:8000)")

    # build-kb命令
    build_parser = subparsers.add_parser("build-kb", help="构建知识库 (解析PDF并建立索引)")

    # search命令
    search_parser = subparsers.add_parser("search", help="命令行检索知识库")
    search_parser.add_argument("query", type=str, help="查询文本")

    # stats命令
    stats_parser = subparsers.add_parser("stats", help="查看知识库和缓存统计信息")

    args = parser.parse_args()

    if args.command == "web":
        cmd_web(args)
    elif args.command == "build-kb":
        cmd_build_kb(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()
        print(f"\n错误: 未知命令 '{args.command}'")


if __name__ == "__main__":
    main()
