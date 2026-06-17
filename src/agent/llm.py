from langchain_openai import ChatOpenAI

from agent.config import get_config


def get_llm() -> ChatOpenAI:
    cfg = get_config()
    common = dict(
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
    )
    if cfg.llm_provider == "openai":
        return ChatOpenAI(api_key=cfg.openai_api_key, **common)
    
    return ChatOpenAI(
        base_url=cfg.lmstudio_base_url,
        api_key=cfg.lmstudio_api_key,
        **common,
    )
