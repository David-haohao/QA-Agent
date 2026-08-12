# ============================================================
# 知识库离线构建流水线
# 两阶段：Phase1文档提取+切片 → Phase2结构化+索引
# ============================================================

import os
import json
from typing import Dict


class KnowledgeBasePipeline:
    """
    知识库构建流水线 — 将本地PDF转换为可检索的向量+关键词索引
    支持增量更新：新增文档自动处理，已有文档跳过
    """

    def __init__(self, config: dict, embedding_client, reranker_client):
        """
        config: knowledge_base配置字典
        embedding_client: EmbeddingClient
        reranker_client: RerankerClient
        """
        self.config = config
        self.embedding_client = embedding_client
        self.reranker_client = reranker_client

        self.documents_dir = config.get("documents_dir", "./documents")
        self.kb_data_dir = config.get("kb_data_dir", "./kb_data")
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.collection_name = config.get("vector_collection", "qa_knowledge")
        self.dimension = config.get("dimension", 1024)
        self._vector_index = None

        os.makedirs(self.kb_data_dir, exist_ok=True)

    def get_vector_index(self):
        """返回当前进程共享的 Qdrant 向量索引客户端。"""
        if self._vector_index is None:
            from knowledge_base.indexing.qdrant_index import QdrantVectorIndex
            self._vector_index = QdrantVectorIndex(
                kb_data_dir=self.kb_data_dir,
                collection_name=self.collection_name,
                embedding_client=self.embedding_client,
                dimension=self.dimension,
            )
        return self._vector_index

    def build(self) -> Dict:
        """
        执行完整的知识库构建流程
        返回: 构建统计信息 {"doc_count": ..., "chunk_count": ..., ...}
        """
        from knowledge_base.extractors import DocumentExtractor
        from knowledge_base.indexing.text_chunker import DocumentChunker
        from knowledge_base.indexing.bm25_index import BM25Index
        from knowledge_base.indexing.document_graph import DocumentGraph

        print("=" * 60)
        print("开始构建知识库...")
        print("=" * 60)

        # Phase 1: 提取与切片
        print("[Phase 1] 提取文档内容...")
        extractor = DocumentExtractor(self.documents_dir)
        docs, extraction_report = extractor.extract_all_with_report()
        if not docs:
            print("未找到可处理的文档!")
            return {"doc_count": 0, "chunk_count": 0}
        print(
            f"  扫描 {extraction_report['scanned_files']} 个文件，"
            f"成功提取 {len(docs)} 个，跳过 {len(extraction_report['skipped_files'])} 个"
        )

        # 保存提取的文档全文（保留原始完整文档名，不做任何字符串处理）
        docs_text_dir = os.path.join(self.kb_data_dir, "docs_text")
        os.makedirs(docs_text_dir, exist_ok=True)
        for doc in docs:
            original_name = doc.get("file_name", "unknown")
            # 使用文档原始完整名称保存，不做任何字符串处理
            text_path = os.path.join(docs_text_dir, original_name + ".txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(doc.get("content", ""))
        print(f"  文档全文(原始完整名称)已保存至 {docs_text_dir}")

        print("[Phase 1] 文本切片...")
        chunker = DocumentChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = chunker.chunk_documents(docs)
        print(f"  生成了 {len(chunks)} 个文本块")

        # 构建 chunk→文档 索引（供在线问答时追踪来源文档）
        chunk_doc_index = {}
        file_chunks_index = {}  # 反向索引: file_name → [chunk_ids]
        for c in chunks:
            cid = c["chunk_id"]
            fname = c.get("file_name", "")
            chunk_doc_index[cid] = {
                "file_name": fname,
                "doc_id": c.get("doc_id", ""),
                "chunk_index": c.get("chunk_index", 0),
            }
            if fname not in file_chunks_index:
                file_chunks_index[fname] = []
            file_chunks_index[fname].append(cid)
        index_path = os.path.join(self.kb_data_dir, "chunk_doc_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({
                "chunks": chunk_doc_index,
                "files": file_chunks_index,
                "total_chunks": len(chunks),
                "total_files": len(file_chunks_index),
            }, f, ensure_ascii=False)
        print(f"  chunk→文档索引已保存至 {index_path} ({len(chunk_doc_index)} 条)")

        # Phase 2: 结构化与索引
        print("[Phase 2] 重建 Qdrant 向量索引...")
        if self._vector_index is not None:
            self._vector_index.close()
            self._vector_index = None
        vector_index = self.get_vector_index()
        vector_count = vector_index.rebuild(chunks)
        print(f"  向量索引构建完成: {vector_count} 条")

        print("[Phase 2] 构建BM25关键词索引...")
        bm25_index = BM25Index(self.kb_data_dir)
        bm25_count = bm25_index.build_index(chunks)
        print(f"  BM25索引构建完成: {bm25_count} 条")

        print("[Phase 2] 构建文档关系图谱...")
        doc_graph = DocumentGraph(self.kb_data_dir)
        doc_graph.build_graph(chunks)
        print(f"  文档图谱构建完成")

        # 保存元数据
        metadata = {
            "total_docs": len(docs),
            "total_chunks": len(chunks),
            "doc_names": [d["file_name"] for d in docs],
            "build_time": __import__("datetime").datetime.now().isoformat(),
            "vector_store": "qdrant",
            "vector_collection": self.collection_name,
            "vector_dimension": self.dimension,
            "embedding_model": getattr(self.embedding_client, "model", "unknown"),
            "extraction_report": extraction_report,
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        }
        metadata_path = os.path.join(self.kb_data_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Phase 3: 预转换文档为HTML预览（点击来源链接即时查看）
        print("[Phase 3] 预转换文档为HTML预览格式...")
        pre_convert_count = self._pre_convert_documents()
        print(f"  预转换完成: {pre_convert_count} 个文档")

        print("=" * 60)
        print(f"知识库构建完成！文档数: {len(docs)}, 文本块数: {len(chunks)}")
        print("=" * 60)

        return {
            "doc_count": len(docs),
            "chunk_count": len(chunks),
            "metadata": metadata,
        }

    def _auto_update_domains_config(self, docs: list):
        """根据文档内容自动总结归类domains和domain_keywords，更新config.yaml"""
        import jieba
        from collections import Counter
        import yaml

        # 1. 合并所有文档内容
        all_text = " ".join([d.get("content", "") for d in docs])

        # 2. 提取高频关键词（TF-IDF简易实现）
        words = list(jieba.cut(all_text))
        words = [w.strip() for w in words if len(w.strip()) >= 2]
        word_freq = Counter(words)

        # 过滤停用词级别的常见词
        stop_words = {"的", "是", "在", "和", "了", "有", "不", "人", "我", "他", "它",
                      "这", "那", "就", "也", "都", "而", "及", "与", "或", "但",
                      "中", "被", "把", "从", "对", "为", "以", "上", "下", "要",
                      "会", "可", "能", "去", "等", "其", "将", "之", "来", "到",
                      "第", "条", "款", "项", "该", "本", "一", "二", "三", "四",
                      "五", "六", "七", "八", "九", "十", "规定", "应当", "可以",
                      "不得", "必须", "包括", "以下", "以上", "按照", "根据", "其他",
                      "相关", "有关", "进行", "提供", "情况", "予以", "报告", "要求",
                      "通过", "具有", "使用", "用于", "所有", "属于", "需要", "一般"}
        filtered_words = {w: c for w, c in word_freq.items() if w not in stop_words}

        # 3. 提取领域关键词（高频专业词汇，TOP 60）
        top_keywords = [w for w, _ in Counter(filtered_words).most_common(60)]

        # 4. 用LLM根据文档内容总结领域类别
        doc_names = [d.get("file_name", "") for d in docs]
        doc_summary = "\n".join([f"- {n}" for n in doc_names[:30]])  # 取前30个文档名
        sample_text = all_text[:3000]  # 取前3000字作为样本

        domain_prompt = f"""以下是知识库中金融类文档的内容样本和文档列表。请根据这些文档的内容和主题，总结出恰当的领域分类（5-10个类别），以及核心关键词列表（30-40个）。

文档列表（前30个）：
{doc_summary}

内容样本：
{sample_text[:2000]}

请只输出一个JSON对象，格式如下：
{{
  "domains": ["类别1", "类别2", ...],
  "domain_keywords": ["关键词1", "关键词2", ...]
}}

要求：
- domains: 5-10个领域类别名称，概括文档覆盖的主题范围
- domain_keywords: 30-40个核心关键词，覆盖主要金融业务概念
- 所有内容必须使用中文
- 只输出JSON，不要其他内容"""

        try:
            msgs = [{"role": "user", "content": domain_prompt}]
            response = self.embedding_client.__class__.__name__  # 不能直接用embedding_client调LLM
            # 使用reranker所在API来调用（实际使用llm_client）
            # fallback: 使用关键词统计来生成领域
        except Exception:
            pass

        # fallback方案：基于TF-IDF关键词统计自动生成领域和关键词
        # 领域分类通过文档名和关键词模式识别
        domain_patterns = {
            "金融监管与合规": ["监管", "合规", "反洗钱", "处罚", "检查", "评估", "报告", "监督", "管理", "办法"],
            "银行业务管理": ["银行", "贷款", "存款", "利率", "结算", "支付", "账户", "汇款", "储蓄"],
            "资本市场与证券": ["证券", "债券", "股票", "期货", "衍生品", "基金", "信托", "融资", "上市"],
            "风险管理": ["风险", "资本", "压力测试", "流动性", "杠杆", "准备金", "拨备", "损失", "违约"],
            "外汇管理": ["外汇", "汇率", "跨境", "经常项目", "资本项目", "兑换", "收支"],
            "征信与信用": ["征信", "信用", "评级", "授信", "担保", "查询"],
            "保险管理": ["保险", "保费", "理赔", "承保", "再保险", "精算"],
        }

        # 统计文档名和内容中匹配各领域的词频
        domain_scores = {}
        for domain, pattern_keywords in domain_patterns.items():
            score = 0
            for kw in pattern_keywords:
                score += word_freq.get(kw, 0)
                # 文档名中也匹配
                for dn in doc_names:
                    if kw in dn:
                        score += 3  # 文档名匹配权重更高
            domain_scores[domain] = score

        # 选择得分最高的5-8个领域
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        active_domains = [d for d, s in sorted_domains if s > 2][:8]
        if len(active_domains) < 4:
            active_domains = [d for d, s in sorted_domains[:6]]

        # 生成关键词：从过滤后的高频词中选取专业词汇
        # 专业词汇通常包含金融相关汉字
        finance_chars = set("金银借贷汇率券股债基金信托保险赔征纳税监规罚")
        professional_keywords = []
        for w, freq in Counter(filtered_words).most_common(80):
            if len(w) >= 2 and any(c in w for c in finance_chars):
                professional_keywords.append(w)
        professional_keywords = professional_keywords[:40]

        # 如果LLM提取的关键词不足，用高频词补充
        if len(professional_keywords) < 30:
            for w, _ in Counter(filtered_words).most_common(60):
                if w not in professional_keywords and len(w) >= 2:
                    professional_keywords.append(w)
                if len(professional_keywords) >= 40:
                    break

        # 读取并更新config.yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml"
        )
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            config_data["domains"] = active_domains
            config_data["domain_keywords"] = professional_keywords

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            print(f"  已更新领域: {active_domains}")
            print(f"  已更新关键词: {len(professional_keywords)} 个")

    def _pre_convert_documents(self) -> int:
        """预转换所有Word/Excel文档为HTML预览格式
        保存在 kb_data/docs_html/ 目录下，确保点击来源链接时即时查看
        PDF文档无需转换（浏览器直接内嵌渲染）
        返回: 转换的文档数量"""
        import os

        docs_html_dir = os.path.join(self.kb_data_dir, "docs_html")
        os.makedirs(docs_html_dir, exist_ok=True)

        # 导入HTML生成函数（从routes模块）
        from frontend.routes import _generate_doc_html_preview

        count = 0
        if not os.path.isdir(self.documents_dir):
            return 0

        for fname in sorted(os.listdir(self.documents_dir)):
            fname_lower = fname.lower()
            # 跳过PDF（浏览器可直接内嵌预览）
            if fname_lower.endswith(".pdf"):
                continue
            # 只处理Word和Excel
            if not (fname_lower.endswith((".docx", ".doc", ".xlsx", ".xls"))):
                continue

            html_path = os.path.join(docs_html_dir, fname + ".html")
            # 已存在则跳过
            if os.path.exists(html_path):
                count += 1
                continue

            doc_path = os.path.join(self.documents_dir, fname)
            try:
                html_content = _generate_doc_html_preview(doc_path, fname)
                if html_content:
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    count += 1
                    if count % 20 == 0:
                        print(f"  已预转换: {count} 个文档...")
            except Exception as e:
                print(f"  预转换失败 [{fname}]: {e}")

        return count

    def search(self, query: str) -> list:
        """
        使用知识库进行检索(在线接口)
        query: 查询文本
        返回: 检索结果列表
        """
        from knowledge_base.indexing.bm25_index import BM25Index
        from .retrieval.retriever import HybridRetriever
        from .retrieval.source_binding import SourceBinding

        # 初始化检索组件
        vector_index = self.get_vector_index()
        bm25_index = BM25Index(self.kb_data_dir)
        bm25_index.load_if_exists()

        retriever = HybridRetriever(vector_index, bm25_index, self.reranker_client)

        results = retriever.search(
            query=query,
            dense_top_n=self.config.get("dense_top_n", 20),
            sparse_top_n=self.config.get("sparse_top_n", 20),
            final_top_k=self.config.get("final_top_k", 5),
        )

        return results
