# ============================================================
# 异步并行调度器
# 管理预处理串行、检索并行、后处理并行的执行调度
# ============================================================

import concurrent.futures
from typing import List, Callable, Any


class AsyncExecutor:
    """
    异步并行任务调度器
    支持串行(早停)和并行两种执行模式
    """

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers

    def run_sequential(self, tasks: List[Callable], early_stop_on: Callable = None) -> List[Any]:
        """
        串行执行任务列表，支持早停
        early_stop_on: 接收上一步结果，返回True时停止
        """
        results = []
        for task in tasks:
            result = task()
            results.append(result)
            if early_stop_on and early_stop_on(result):
                break
        return results

    def run_parallel(self, tasks: List[Callable]) -> List[Any]:
        """
        并行执行任务列表
        返回与任务顺序对应的结果列表
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(task) for task in tasks]
            results = []
            for future in futures:
                try:
                    results.append(future.result(timeout=30))
                except Exception as e:
                    results.append(e)
            return results
