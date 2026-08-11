# ============================================================
# 模型API 统一封装模块
# 将三个线上模型API封装为统一接口，方便后续更换模型
# ============================================================

from .llm_client import LLMClient          # LLM大模型客户端(Qwen3.6-27b, 阿里云DashScope)
from .embedding_client import EmbeddingClient  # 嵌入模型客户端(BGE-Large-ZH)
from .reranker_client import RerankerClient    # 重排序模型客户端(BGE-Reranker)
