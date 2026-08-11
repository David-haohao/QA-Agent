# ============================================================
# 意图提取器 — 从查询中提取意图和核心实体
# 用于L2缓存的组合键生成，以高精确率为目标
# ============================================================

from typing import Optional, Dict


class IntentExtractor:
    """
    查询意图提取器
    使用轻量LLM分析查询，提取intent(意图类型)和entity(核心实体)
    安全策略: 低置信度和多意图查询不参与L2缓存匹配
    """

    # 预定义意图类型
    INTENT_TYPES = [
        "定义查询",  # "XX是什么"
        "条款查询",  # "第X条规定了什么"
        "范围查询",  # "XX适用于哪些"
        "依据查询",  # "XX的制定依据是什么"
        "流程查询",  # "XX的流程/步骤"
        "对比查询",  # "A和B的区别"
        "计算查询",  # "如何计算XX"
        "时间查询",  # "XX是什么时候/有效期"
        "要求查询",  # "XX有什么要求/条件"
        "影响查询",  # "XX对YY有什么影响"
        "复杂查询",  # 多意图混合或无法明确归类
    ]

    def __init__(self, llm_client):
        """
        llm_client: LLM客户端(用于轻量意图分析)
        """
        self.llm_client = llm_client

    def extract(self, query: str) -> Optional[Dict]:
        """
        提取查询的意图和核心实体
        返回: {"intent": "定义查询", "entity": "商业银行资本管理办法",
                "confidence": 0.92, "is_multi_intent": false}
        提取失败时返回None
        """
        try:
            prompt = f"""分析以下用户问题，只输出一个JSON对象。

问题: {query}

请提取:
1. intent: 问题意图类型，从下列选择:
   {" ".join(self.INTENT_TYPES)}
2. entity: 问题核心实体(提取最关键的主题名词，去修饰词)
3. confidence: 你对意图分类的确信度(0-1)
4. is_multi_intent: 是否有多个不同意图(true/false)

只输出JSON，不要其他内容。格式示例:
{{"intent": "定义查询", "entity": "资本管理办法", "confidence": 0.95, "is_multi_intent": false}}"""

            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_simple(messages)

            # 解析JSON
            import json
            response = response.strip()
            # 提取JSON部分(可能被引号包裹)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            result = json.loads(response)

            return {
                "intent": result.get("intent", "复杂查询"),
                "entity": result.get("entity", ""),
                "confidence": result.get("confidence", 0.0),
                "is_multi_intent": result.get("is_multi_intent", False),
            }
        except Exception as e:
            print(f"意图提取失败: {e}")
            return None
