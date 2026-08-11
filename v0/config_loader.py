# ============================================================
# 配置加载器 — 加载config.yaml并统一管理所有配置项
# 整个工程通过此模块获取所有配置，后续更换模型API只需修改配置文件
# ============================================================

import os
import yaml


_API_KEY_ENV_VARS = {
    "llm": "QA_LLM_API_KEY",
    "embeddings": "QA_EMBEDDINGS_API_KEY",
    "reranker": "QA_RERANKER_API_KEY",
}


class Config:
    """全局配置管理器，从config.yaml加载所有配置并提供属性访问"""

    _instance = None

    def __init__(self, config_path: str = None):
        """
        加载配置文件
        优先使用传入路径，其次使用环境变量，最后使用默认路径
        """
        if config_path is None:
            config_path = os.environ.get("QA_CONFIG_PATH", "config.yaml")
        # 将相对路径转为绝对路径(基于工程根目录)
        if not os.path.isabs(config_path):
            root = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(root, config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
        self._apply_environment_overrides()

    def _apply_environment_overrides(self):
        """Allow deployment secrets to override values stored in config.yaml."""
        for section, environment_variable in _API_KEY_ENV_VARS.items():
            api_key = os.environ.get(environment_variable)
            if api_key:
                self._data.setdefault(section, {})["api_key"] = api_key

    def __getitem__(self, key: str):
        return self._data[key]

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    # ========== LLM 相关属性 ==========
    @property
    def llm_config(self) -> dict:
        return self._data["llm"]

    @property
    def embeddings_config(self) -> dict:
        return self._data["embeddings"]

    @property
    def reranker_config(self) -> dict:
        return self._data["reranker"]

    @property
    def kb_config(self) -> dict:
        return self._data["knowledge_base"]

    @property
    def cache_config(self) -> dict:
        return self._data["cache"]

    @property
    def pre_processing_config(self) -> dict:
        return self._data["pre_processing"]

    @property
    def timeout_config(self) -> dict:
        return self._data["timeout"]

    @property
    def server_config(self) -> dict:
        return self._data["server"]

    @property
    def evolution_config(self) -> dict:
        return self._data["evolution"]

    @property
    def domains(self) -> list:
        return self._data["domains"]

    @property
    def domain_keywords(self) -> list:
        return self._data["domain_keywords"]


def load_config(config_path: str = None) -> Config:
    """获取全局配置单例"""
    return Config(config_path)
