# ============================================================
# 需求完整性检查器 — 预处理Step2
# 检查用户查询是否包含足够的实体和意图，不完整时生成引导问题
# ============================================================

from typing import Tuple, Optional, List


class CompletenessChecker:
    """
    需求完整性检查器
    评估用户查询的完备性，对真正不完整的查询生成引导问题
    """

    def __init__(self, threshold: float = 0.4):
        """
        threshold: 完整性评分阈值,低于此值认为不完整
        （降低阈值避免拦截"合格工具有哪些"这类明确问题）
        """
        self.threshold = threshold

    def check(self, query: str) -> Tuple[bool, float, Optional[List[str]]]:
        """
        检查查询完整性
        返回: (is_complete, score, guidance_questions)
        """
        score = self._evaluate_completeness(query)
        print('---------客户问题完整性得分: {}'.format(score))

        if score >= self.threshold:
            return True, score, None
        else:
            guidance = self._generate_guidance(query)
            return False, score, guidance

    def _evaluate_completeness(self, query: str) -> float:
        """
        基于明确的信号评估完备性
        核心原则：只要查询有"明确的主题 + 明确的问法"就放行
        """
        signals = 0.0  # 明确信号计数

        # 信号1: 包含明确的疑问词或需求词(列举、定义、范围、条件、流程等)
        INTENT_KEYWORDS = [
            "有哪些", "哪些", "列出", "列举", "包括",  # 枚举类
            "是什么", "什么是", "定义", "含义", "指什么", "指",  # 定义类
            "要求", "条件", "标准", "规定", "条款", "法规",  # 要求类
            "适用范围", "适用于", "适用对象", "应用范围", "限于",  # 范围类
            "如何", "怎么", "怎样", "流程", "步骤", "方法",  # 流程类
            "区别", "不同", "对比", "关系", "影响",  # 对比类
            "修订", "修改", "更新", "最新", "变化",  # 更新类
            "何时", "时间", "期限", "有效期", "生效",  # 时间类
            "依据", "根据", "来源", "出处",  # 来源类
            "多少", "数量", "比例", "计算",  # 计算类
        ]
        found_intent = any(k in query for k in INTENT_KEYWORDS)
        if found_intent:
            signals += 2.0  # 明确的意图是强烈信号

        # 信号2: 包含具体实体(法规名/机构名/专业术语)
        ENTITY_KEYWORDS = [
            "银行", "保险", "证券", "金融", "资本", "基金", "信托",
            "贷款", "存款", "理财", "债券", "股票", "期货",
            "合规", "风险", "监管", "反洗钱", "巴塞尔",
            "管理", "办法", "条例", "法规", "规定", "通知",
            "尽职调查", "信用", "流动性", "评级", "授信",
            "准备金", "杠杆", "衍生品", "保函", "信用证",
            "资本工具", "一级资本", "二级资本", "核心一级",
        ]
        found_entity = any(k in query for k in ENTITY_KEYWORDS)
        if found_entity:
            signals += 2.0  # 具体的实体是强烈信号

        # 信号3: 查询长度(太短通常信息不足)
        qlen = len(query)
        if qlen >= 10:
            signals += 0.5
        if qlen >= 15:
            signals += 0.5
        if qlen >= 25:
            signals += 0.5

        # 信号4: 包含书名号或引号(明确引用)
        import re
        if re.search(r"[《「『]", query):
            signals += 1.0

        # 信号4.5: 对话引用词（代词指代上文——多轮对话的追问信号）
        REFERENCE_KEYWORDS = ["它", "他", "她", "这个", "那个", "这些", "那些",
                               "上面", "前面", "刚才", "之前", "刚刚", "上次",
                               "第一个", "第二个", "前一个", "上一个"]
        found_reference = any(k in query for k in REFERENCE_KEYWORDS)
        if found_reference:
            signals += 1.5  # 指代词说明是追问，增强信号

        # 信号5: 以问号结尾
        if query.endswith("？") or query.endswith("?") or query.endswith("吗"):
            signals += 1.0

        # 归一化到0~1（最高7分）
        max_score = 7.0
        score = min(signals / max_score, 1.0)

        # 兜底：如果同时有意图和实体(不论分数)，直接放行
        if found_intent and found_entity:
            score = max(score, 0.65)

        return score

    def _generate_guidance(self, query: str) -> List[str]:
        """生成引导问题，只在真正信息不足时才生成"""
        import re

        # 尝试提取主题词
        topic = query
        m = re.search(r"[《「『]([^》」』]+)[》」』]", query)
        if m:
            topic = m.group(1)

        # 短查询→引导具体化
        if len(query) < 8 and not any(k in query for k in ["有哪些", "是什么", "如何", "怎么", "要求", "规定", "条件"]):
            return [
                f"您想了解「{topic}」的哪方面内容？",
                f"📖 它的定义与核心概念",
                f"📋 具体的条款规定",
                f"🎯 适用范围与对象",
                f"📅 最新修订内容与历史版本",
            ]

        # 有主题但无明确问法→引导
        return [
            f"关于「{topic}」，请问您想了解：",
            f"📖 它的定义和含义？",
            f"📋 具体有哪些要求和规定？",
            f"🎯 适用于哪些机构或业务？",
            f"📅 最新修订内容是什么？",
        ]
