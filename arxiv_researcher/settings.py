import os

import cohere
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    OPENAI_API_KEY: str
    COHERE_API_KEY: str

    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "arxiv-researcher"

    # for Application\
    openai_fast_model: str = "gpt-4o-mini"
    openai_smart_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    cohere_rerank_model: str = "rerank-multilingual-v3.0"
    temperature: float = 0.0
    max_search_results: int = 100
    max_workers: int = 5
    debug: bool = True

    def __init__(self, **values):
        super().__init__(**values)
        self._set_env_variables()

    def _set_env_variables(self):
        for key in self.__annotations__.keys():
            if key.isupper():
                os.environ[key] = getattr(self, key)

    @property
    def llm(self) -> ChatOpenAI:
        return ChatOpenAI(model=self.openai_smart_model)

    @property
    def cohere_client(self) -> cohere.Client:
        return cohere.Client(api_key=self.COHERE_API_KEY)


settings = Settings()
