# ============================================================
# 查询标准化器 — 预处理Step1
# 将口语化查询转换为标准化查询
# ============================================================

import re


class QueryNormalizer:
    """
    查询标准化器
    对用户输入进行标准化处理：
    - 同义词映射
    - 口语→书面语转换
    - 时间表述标准化
    - 法规编号标准化
    """

    # 同义词映射表
    SYNONYMS = {
        "啥": "什么",
        "咋": "怎么",
        "咋样": "如何",
        "能不能": "是否可以",
        "是不是": "是否是",
        "算不算": "是否属于",
        "咋办": "如何处理",
        "得": "必须",
        "搞": "处理",
        "弄": "办理",
    }

    # 法规编号模式
    REGULATION_PATTERNS = [
        (r"(\d{4})年第?(\d+)号", r"[\1]第\2号"),  # 标准化法规文号
        (r"银监发\[(\d{4})\](\d+)号", r"银监发[\1]\2号"),
        (r"银保监发\[(\d{4})\](\d+)号", r"银保监发[\1]\2号"),
    ]

    def normalize(self, query: str) -> str:
        """
        标准化查询文本
        返回标准化后的查询字符串
        """
        text = query.strip()

        # ① 同义词替换
        for colloquial, formal in self.SYNONYMS.items():
            text = text.replace(colloquial, formal)

        # ② 时间表述标准化
        text = self._normalize_dates(text)

        # ③ 法规编号标准化
        for pattern, replacement in self.REGULATION_PATTERNS:
            text = re.sub(pattern, replacement, text)

        # ④ 去多余空白
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _normalize_dates(self, text: str) -> str:
        """标准化时间表述"""
        # "今年"→当前年份
        import datetime
        current_year = str(datetime.datetime.now().year)
        text = text.replace("今年", f"{current_year}年")
        text = text.replace("去年", f"{int(current_year)-1}年")
        return text
