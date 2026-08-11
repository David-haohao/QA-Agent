# ============================================================
# 拒绝话术模板 — 领域外拒绝时的高情商话术
# ============================================================

from typing import List


class RejectionTemplates:
    """
    拒绝话术生成器
    四段式结构: 共情确认 → 边界说明 → 建设性引导 → 关联建议
    """

    def __init__(self, domains: List[str]):
        self.domains = domains

    def build_rejection(self, query: str, topic: str = "") -> dict:
        """
        生成领域外拒绝的完整响应
        返回: 包含所有拒绝话术的字典
        """
        topic = topic or query

        empathy = f"感谢您的提问！我理解您想了解关于「{topic}」的内容。"

        domain_list = " · ".join(self.domains)
        boundary = f"我目前的知识库主要覆盖 {domain_list} 等 {len(self.domains)} 个领域。您提到的「{topic}」暂时超出了我的知识边界。"

        guidance = "如果您有上述领域内的问题，我随时乐意为您解答。"

        # 建议用户将问题与覆盖领域关联
        suggestions = [
            "您可以尝试将问题与金融监管、银行业务合规等领域关联后重新提问",
            "如果您的问题是上述领域的某个子话题，请直接使用领域内的专业术语提问",
        ]

        return {
            "type": "rejection",
            "empathy": empathy,
            "boundary": boundary,
            "guidance": guidance,
            "suggestions": suggestions,
            "domains": self.domains,
        }

    def build_no_result(self, query: str) -> dict:
        """
        生成检索无结果的响应(区别于领域外拒绝)
        检索无结果表示问题属于领域内但知识库缺少相应文档
        """
        return {
            "type": "no_result",
            "empathy": "我理解了您的问题，这确实属于我覆盖的领域。",
            "feedback": "但在当前知识库中暂时没有找到直接相关的资料。",
            "suggestions": [
                "换用更具体的术语重新描述问题",
                "如果您知道相关的法规文号或文件名，可直接告诉我",
            ],
            "note": "此查询已记录，将纳入知识库补充评估",
        }
