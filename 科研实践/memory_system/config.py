"""配置管理模块"""
import os
import yaml
from pathlib import Path


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default=None):
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default

    @property
    def memory_root(self) -> Path:
        return Path(self._config["system"]["memory_root"])

    @property
    def chroma_path(self) -> Path:
        return Path(self._config["system"]["chroma_path"])

    @property
    def llm_provider(self) -> str:
        return self._config["llm"]["provider"]

    @property
    def llm_model(self) -> str:
        return self._config["llm"]["model"]

    @property
    def embedding_model(self) -> str:
        return self._config["llm"]["ollama"]["embedding_model"]

    @property
    def chunk_size(self) -> int:
        return self._config["storage"]["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self._config["storage"]["chunk_overlap"]


config = Config()
