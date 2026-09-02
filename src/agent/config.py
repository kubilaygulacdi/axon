from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider selection
    llm_provider: Literal["openai", "lmstudio"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_explore_model: str | None = Field(default=None, alias="LLM_EXPLORE_MODEL")
    llm_temperature: float = Field(
        default=0.0, alias="LLM_TEMPERATURE", ge=0.0, le=2.0
    )

    # OpenAI
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # LM Studio
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1", alias="LMSTUDIO_BASE_URL"
    )
    lmstudio_api_key: str = Field(
        default="lm-studio", alias="LMSTUDIO_API_KEY"
    )


def get_config() -> Config:
    return Config()