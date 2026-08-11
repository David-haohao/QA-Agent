# ============================================================
# 一致性缓存 — 答案一致性管理器
# L1: 精确匹配缓存 (本地进程内存, LRU)
# L2: 意图感知缓存 (intent+entity组合键, Redis或本地)
# ============================================================

import time
import hashlib
from collections import OrderedDict
from typing import Optional, Dict, Any


class ConsistencyCache:
    """
    两级缓存架构，保证同一问题的回答基本一致

    L1 (进程内存LRU): 精确匹配 normalized_query hash
        - 容量: 2000条, TTL: 24h
    L2 (intent+entity): 意图感知匹配
        - 先提取查询的意图和核心实体，然后组合键匹配
        - TTL: 72h
    """

    def __init__(self, config: dict, intent_extractor=None, redis_client=None):
        """
        config: cache配置字典
        intent_extractor: 意图提取器实例
        redis_client: Redis客户端(可选，不启用则L2也是本地)
        """
        self.config = config
        self.l1_max_size = config.get("l1_max_size", 2000)
        self.l1_ttl = config.get("l1_ttl", 86400)
        self.l2_ttl = config.get("l2_ttl", 259200)
        self.redis_enabled = config.get("redis_enabled", False)
        self.intent_extractor = intent_extractor
        self.redis = redis_client

        # L1: 有序字典实现LRU
        self.l1_cache: OrderedDict[str, Dict] = OrderedDict()

        # L2: 本地字典(Redis不使用时的备选)
        self.l2_cache: Dict[str, Dict] = {}

        # 缓存统计
        self.stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0, "total": 0}

    def lookup(self, normalized_query: str) -> Optional[Dict]:
        """
        查询缓存
        返回: 命中时返回缓存条目，未命中返回None
        """
        self.stats["total"] += 1

        # L1: 精确匹配
        l1_key = self._l1_key(normalized_query)
        result = self._l1_lookup(l1_key)
        if result:
            self.stats["l1_hits"] += 1
            return result

        # L2: 意图感知匹配
        result = self._l2_lookup(normalized_query)
        if result:
            self.stats["l2_hits"] += 1
            # 回填L1
            self._l1_store(l1_key, result)
            return result

        self.stats["misses"] += 1
        return None

    def store(self, normalized_query: str, answer_data: Dict):
        """
        存储答案到两级缓存
        """
        # L1存储
        l1_key = self._l1_key(normalized_query)
        self._l1_store(l1_key, answer_data)

        # L2存储(需要先提取意图+实体)
        if self.intent_extractor:
            intent_result = self.intent_extractor.extract(normalized_query)
            if intent_result and intent_result.get("confidence", 0) >= 0.85:
                intent = intent_result.get("intent", "unknown")
                entity = intent_result.get("entity", "")
                l2_key = self._l2_key(intent, entity)
                self._l2_store(l2_key, answer_data)

    def invalidate_document(self, doc_id: str):
        """知识库文档更新时，使相关缓存失效"""
        # 扫描L1中与该doc相关的条目
        to_remove = []
        for key, entry in self.l1_cache.items():
            sources = entry.get("source_docs", [])
            if any(s.get("doc_id") == doc_id for s in sources):
                to_remove.append(key)
        for key in to_remove:
            del self.l1_cache[key]

        # L2同样处理
        to_remove = []
        for key, entry in self.l2_cache.items():
            sources = entry.get("source_docs", [])
            if any(s.get("doc_id") == doc_id for s in sources):
                to_remove.append(key)
        for key in to_remove:
            del self.l2_cache[key]

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        hit_rate = 0
        if self.stats["total"] > 0:
            hits = self.stats["l1_hits"] + self.stats["l2_hits"]
            hit_rate = hits / self.stats["total"]
        return {
            **self.stats,
            "hit_rate": round(hit_rate, 4),
            "l1_size": len(self.l1_cache),
            "l2_size": len(self.l2_cache),
        }

    def _l1_key(self, normalized_query: str) -> str:
        """L1缓存键"""
        return hashlib.md5(normalized_query.encode()).hexdigest()

    def _l1_lookup(self, key: str) -> Optional[Dict]:
        """L1缓存查询"""
        if key not in self.l1_cache:
            return None
        entry = self.l1_cache[key]
        # 检查TTL
        if time.time() - entry.get("created_at", 0) > self.l1_ttl:
            del self.l1_cache[key]
            return None
        # 移到末尾(LRU）
        self.l1_cache.move_to_end(key)
        entry["access_count"] = entry.get("access_count", 0) + 1
        return entry["data"]

    def _l1_store(self, key: str, data: Dict):
        """L1缓存存储"""
        # LRU淘汰
        while len(self.l1_cache) >= self.l1_max_size:
            self.l1_cache.popitem(last=False)
        self.l1_cache[key] = {
            "data": data,
            "created_at": time.time(),
            "access_count": 0,
        }

    def _l2_key(self, intent: str, entity: str) -> str:
        """L2缓存键 = hash(intent + entity)"""
        combined = f"{intent}||{entity}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _l2_lookup(self, query: str) -> Optional[Dict]:
        """L2意图感知缓存查询"""
        if not self.intent_extractor:
            return None

        # 提取意图+实体
        intent_result = self.intent_extractor.extract(query)
        if not intent_result:
            return None

        confidence = intent_result.get("confidence", 0)
        # 安全策略：意图置信度<0.85不参与匹配
        if confidence < 0.85:
            return None
        # 多意图查询不参与匹配
        if intent_result.get("is_multi_intent", False):
            return None

        intent = intent_result.get("intent", "")
        entity = intent_result.get("entity", "")
        if not intent or not entity:
            return None

        key = self._l2_key(intent, entity)

        # 优先Redis
        if self.redis_enabled and self.redis:
            data = self.redis.get(f"l2:{key}")
            if data:
                import json
                return json.loads(data)

        # 本地L2
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            if time.time() - entry.get("created_at", 0) < self.l2_ttl:
                return entry["data"]
            del self.l2_cache[key]
        return None

    def _l2_store(self, key: str, data: Dict):
        """L2缓存存储"""
        entry = {"data": data, "created_at": time.time()}
        self.l2_cache[key] = entry

        if self.redis_enabled and self.redis:
            import json
            self.redis.setex(f"l2:{key}", self.l2_ttl, json.dumps(data, ensure_ascii=False))
