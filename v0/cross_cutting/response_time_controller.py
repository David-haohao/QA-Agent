# ============================================================
# 全链路超时控制器
# 总预算从config.yaml读取（默认100,000ms），分阶段管控
# 降级阈值按总超时比例动态计算，不硬编码
# ============================================================

import time
from typing import Optional, Dict


class ResponseTimeController:
    """
    全链路超时控制
    监控每个阶段的耗时，触发分级降级策略
    降级阈值按总超时比例动态计算
    """

    def __init__(self, timeout_config: dict):
        """
        timeout_config: 各阶段超时配置(ms)，从config.yaml加载
        """
        self.total_timeout = timeout_config.get("total", 100000)
        self.pre_processing_timeout = timeout_config.get("pre_processing", 500)
        self.cache_timeout = timeout_config.get("cache_query", 300)
        self.retrieval_timeout = timeout_config.get("retrieval", 5000)
        self.llm_timeout = timeout_config.get("llm_generation", 90000)
        self.post_processing_timeout = timeout_config.get("post_processing", 3000)
        self.buffer = timeout_config.get("buffer", 1200)

        # 状态
        self.start_time = 0.0
        self.degradation_level = 0  # 0=正常, 1~5=各级降级

    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.degradation_level = 0

    def elapsed_ms(self) -> float:
        """已用时间(ms)"""
        return (time.time() - self.start_time) * 1000

    def check_and_degrade(self) -> Dict:
        """
        检查累计耗时并返回降级策略
        降级阈值按总超时的百分比动态计算
        返回: {"degrade": bool, "level": int, "action": str}
        """
        elapsed = self.elapsed_ms()
        T = self.total_timeout  # 总超时(ms)

        # 降级梯度: 按总超时百分比触发，适配任意超时配置
        if elapsed > T * 0.975:  # 97.5% → 全局超时
            return {"degrade": True, "level": 5, "action": "global_timeout",
                    "message": "回答时间较长，建议您简化问题后重试"}
        elif elapsed > T * 0.925:  # 92.5% → 跳过关联问题
            return {"degrade": True, "level": 4, "action": "skip_followup",
                    "message": "跳过关联问题生成"}
        elif elapsed > T * 0.875:  # 87.5% → 减少LLM输出
            return {"degrade": True, "level": 3, "action": "reduce_llm_tokens",
                    "message": "减少LLM输出token数"}
        elif elapsed > T * 0.75:   # 75% → LLM快速模式
            return {"degrade": True, "level": 2, "action": "llm_fast_mode",
                    "message": "LLM切换为快速模式"}
        elif elapsed > T * 0.15:   # 15% → 跳过来源LLM判定
            return {"degrade": True, "level": 1, "action": "skip_source_llm_check",
                    "message": "跳过来源相关性LLM判定"}
        elif elapsed > T * 0.10:   # 10% → 减少检索数量
            return {"degrade": True, "level": 0, "action": "reduce_retrieval",
                    "message": "减少检索返回数量"}
        return {"degrade": False, "level": -1, "action": "normal"}

    def should_skip_followup(self) -> bool:
        """是否应跳过关联问题生成"""
        return self.degradation_level >= 4

    def should_use_fast_llm(self) -> bool:
        """是否应使用快速LLM模式"""
        return self.degradation_level >= 2

    def get_reduced_top_n(self, original: int) -> int:
        """根据降级级别调整检索返回数量"""
        if self.degradation_level >= 1:
            return max(3, original // 2)
        return original
