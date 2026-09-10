"""One chat model factory. Any OpenAI-compatible endpoint works: the local llama-server by
default, or a hosted API by changing TABLETALK_BASE_URL / TABLETALK_MODEL / TABLETALK_API_KEY."""

from langchain_openai import ChatOpenAI

from tabletalk.config import settings


def make_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        temperature=temperature,
        max_tokens=1024,  # a runaway generation on a 7B is the usual cause of a hung run
        timeout=180,
        max_retries=1,
    )
