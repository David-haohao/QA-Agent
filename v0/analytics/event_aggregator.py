# ============================================================
# 事件聚合器 — T+1定时聚合埋点数据
# 为进化优化器提供数据支撑
# ============================================================

import sqlite3
import os
from typing import Dict
from datetime import datetime, timedelta


class EventAggregator:
    """
    埋点事件定时聚合器
    按日聚合各类型事件，生成报表供优化器消费
    """

    def __init__(self, analytics_db_path: str):
        self.db_path = analytics_db_path

    def aggregate_daily(self, date: str = None) -> Dict:
        """
        聚合当日(或指定日期)的埋点数据
        date: YYYY-MM-DD格式，默认昨日
        返回: 聚合统计字典
        """
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 各类事件计数
            cursor.execute(
                "SELECT event_type, COUNT(*) FROM analytics_events WHERE DATE(created_at) = ? GROUP BY event_type",
                (date,),
            )
            event_counts = dict(cursor.fetchall())

            # 关联问题点击率
            total_shown = event_counts.get("followup_shown", 0)
            total_clicked = event_counts.get("followup_click", 0)
            followup_ctr = round(total_clicked / total_shown, 4) if total_shown > 0 else 0

            # 反馈满意度
            positive = event_counts.get("feedback_positive", 0)
            negative = event_counts.get("feedback_negative", 0)
            total_feedback = positive + negative
            satisfaction = round(positive / total_feedback, 4) if total_feedback > 0 else 0

            conn.close()

            return {
                "date": date,
                "total_events": sum(event_counts.values()),
                "event_counts": event_counts,
                "followup_ctr": followup_ctr,
                "feedback_satisfaction": satisfaction,
                "feedback_total": total_feedback,
            }
        except Exception as e:
            return {"date": date, "error": str(e)}
