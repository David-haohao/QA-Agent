# ============================================================
# 领域边界守门人 — 预处理Step3
# 判断查询是否属于知识库覆盖的领域
# 快速路径(规则+embedding) + 深度路径(LLM判定)
# ============================================================

from typing import Tuple


class DomainGuard:
    """
    领域边界判定器
    判定查询是否属于知识库覆盖的领域
    快速路径: 关键词白名单 + Embedding主题相似度
    深度路径: LLM轻量判定
    """

    def __init__(self, domain_keywords: list, domains: list,
                 embedding_client, llm_client,
                 similarity_threshold: float = 0.3,
                 similarity_release: float = 0.6,
                 domain_confidence_threshold: float = 0.5):
        """
        domain_keywords: 关键词白名单
        domains: 覆盖的领域列表
        embedding_client: EmbeddingClient
        llm_client: LLMClient
        """
        self.domain_keywords = domain_keywords
        self.domains = domains
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.similarity_threshold = similarity_threshold   # 0.3以下拒绝
        self.similarity_release = similarity_release       # 0.6以上放行
        self.domain_confidence_threshold = domain_confidence_threshold

        # 计算领域中心向量(懒加载)
        self._domain_centers = None

    def judge(self, query: str) -> Tuple[bool, str, dict]:
        """
        判定查询是否属于覆盖领域
        返回: (is_in_domain, reason, detail)
        """
        # 快速路径1: 关键词白名单
        if self._keyword_match(query):
            return True, "关键词匹配", {"method": "keyword"}

        # 快速路径2: Embedding主题相似度
        is_in, detail = self._embedding_judge(query)
        if is_in is True:
            return True, "主题相似度匹配", detail
        elif is_in is False:
            return False, "主题相似度过低", detail

        # 深度路径: LLM判定(0.3~0.6不确定区间)
        return self._llm_judge(query)

    def _keyword_match(self, query: str) -> bool:
        """关键词白名单匹配"""
        for keyword in self.domain_keywords:
            if keyword in query:
                return True
        return False

    def _embedding_judge(self, query: str) -> Tuple[bool, dict]:
        """
        使用Embedding计算查询与领域描述的主题相似度
        返回: (is_in_domain_or_None, detail)
        None表示不确定需要深度判定
        """
        if self._domain_centers is None:
            # 懒加载: 计算所有领域描述的向量
            domain_texts = self.domains
            centers = self.embedding_client.embed(domain_texts)
            self._domain_centers = list(zip(self.domains, centers))

        query_vec = self.embedding_client.embed_single(query)
        if not query_vec:
            return None, {"error": "embedding失败"}

        # 计算余弦相似度
        import numpy as np
        max_sim = 0.0
        best_domain = ""
        for domain_name, center_vec in self._domain_centers:
            sim = self._cosine_similarity(query_vec, center_vec)
            if sim > max_sim:
                max_sim = sim
                best_domain = domain_name

        detail = {"max_similarity": round(max_sim, 4), "best_domain": best_domain, "method": "embedding"}

        if max_sim > self.similarity_release:
            return True, detail
        elif max_sim < self.similarity_threshold:
            return False, detail
        else:
            return None, detail  # 不确定

    def _llm_judge(self, query: str) -> Tuple[bool, str, dict]:
        """
        LLM轻量判定(深度路径)
        用于规则引擎不能判定时的区间
        """
        domain_list = "、".join(self.domains)
        prompt = f"""判断以下问题是否属于知识库覆盖的领域范围。

覆盖领域: {domain_list}

用户问题: {query}

请只输出一个JSON:
{{"is_in_domain": true或false, "confidence": 0到1之间的数字, "topic": "话题名称"}}

判断标准:
- 如果问题涉及上述任一领域的法规、业务、合规等内容，则is_in_domain为true
- 如果问题与上述领域完全无关(如物理、化学、体育等)，则is_in_domain为false
- confidence表示你对这个判断的确信度"""

        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat_simple(messages)

        # 解析JSON
        import json
        try:
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            result = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            # 解析失败，倾向放行(宁可误放,不误拒)
            return True, "LLM判定解析失败, 倾向放行", {"method": "llm_fallback"}

        is_in = result.get("is_in_domain", True)
        confidence = result.get("confidence", 0.5)

        # confidence > 0.8 且判断false才拒绝
        # 0.5~0.8的拒绝 → 倾向放行
        if is_in:
            return True, "LLM判定领域内", {"method": "llm", "confidence": confidence}
        elif confidence > 0.8:
            return False, "LLM判定领域外(高置信度)", {"method": "llm", "confidence": confidence, "topic": result.get("topic", "")}
        else:
            return True, "LLM判定倾向放行(低置信度)", {"method": "llm_tend_release", "confidence": confidence}

    def _cosine_similarity(self, a: list, b: list) -> float:
        """计算两个向量的余弦相似度"""
        import numpy as np
        a_arr = np.array(a)
        b_arr = np.array(b)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
