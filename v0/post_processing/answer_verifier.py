# ============================================================
# 答案校验器 — 后处理Step2
# 对生成的答案进行质量校验，确保事实锚定到源文档
# ============================================================

import re
from typing import Dict, List


class AnswerVerifier:
    """
    答案质量校验器
    检查答案中的事实陈述是否可追溯到源文档
    不通过的标记警告但不会阻断返回(首次优先保证响应时间)
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def verify(self, answer: str, source_chunks: List[Dict]) -> Dict:
        """
        校验答案质量
        返回: {"passed": bool, "warnings": [...], "score": float}
        """
        warnings = []

        # ① 基本检查：答案是否为空
        if not answer or len(answer.strip()) < 10:
            return {"passed": False, "warnings": ["答案内容过短"], "score": 0.0}

        # ② 来源引用检查：是否有chunk标注
        source_refs = re.findall(r"\[来源:", answer)
        if not source_refs and source_chunks:
            warnings.append("答案未标注任何来源引用")

        # ③ 数字锚定检查：答案中的数字是否能在源文档中找到
        numbers_in_answer = re.findall(r"\d+\.?\d*%?", answer)
        source_text = " ".join([c.get("content", "") for c in source_chunks])
        unverified_numbers = []
        for num in numbers_in_answer:
            if num not in source_text and len(num) >= 2:
                unverified_numbers.append(num)
        if unverified_numbers and len(unverified_numbers) > len(numbers_in_answer) * 0.5:
            warnings.append(f"部分数字未在源文档中找到: {unverified_numbers}")

        score = max(0.0, 1.0 - len(warnings) * 0.25)
        return {
            "passed": len(warnings) == 0,
            "warnings": warnings,
            "score": round(score, 4),
        }
